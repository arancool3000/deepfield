import Foundation
import simd

/// Drift correction from the scan's own photos and depth maps (what survey software does with a "loop closure"):
/// 1. every photo's LiDAR depth is matched (ICP) to the next photo's and to photos of the same place taken later,
/// 2. all those matches are settled together (pose relaxation) into one correction per photo,
/// 3. every point moves with the corrections of the photos that actually saw it (their depth agrees), so two drifted
///    copies of a surface - each seen by different photos - slide back onto each other, then merge.
enum Drift {
    struct KF { let kf: Keyframe; let local: [SIMD3<Float>]; let normals: [SIMD3<Float>]; let depth: [Float]; let dw: Int; let dh: Int }

    static func avg(_ ts: [simd_float4x4]) -> simd_float4x4 {
        guard let first = ts.first else { return matrix_identity_float4x4 }
        var t = SIMD3<Float>.zero, q = SIMD4<Float>.zero
        let q0 = simd_quatf(first).vector
        for m in ts {
            t += SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            var v = simd_quatf(m).vector; if simd_dot(v, q0) < 0 { v = -v }; q += v
        }
        var out = m44(simd_float3x3(simd_quatf(vector: simd_normalize(q))))
        out.columns.3 = SIMD4(t / Float(ts.count), 1)
        return out
    }

    /// every photo's own LiDAR points (camera space, thinned) plus its depth map
    static func load(_ item: ScanItem, maxViews: Int = 80) -> [KF] {
        guard let cap = Library.captureURL(item) else { return [] }
        let all = ModelBuilder.pick(Keyframe.all(in: cap), maxViews)
        var kfs: [KF] = []
        for k in all where k.dw > 0 {
            let base = k.jpg.deletingPathExtension()
            guard let d = try? Data(contentsOf: base.appendingPathExtension("depth")), d.count >= k.dw * k.dh * 4 else { continue }
            let conf = try? Data(contentsOf: base.appendingPathExtension("conf"))
            var dep = [Float](repeating: 0, count: k.dw * k.dh)
            d.withUnsafeBytes { b in for i in 0..<dep.count { dep[i] = b.loadUnaligned(fromByteOffset: i * 4, as: Float.self) } }
            if let c = conf, c.count == dep.count { for i in 0..<dep.count where c[i] == 0 { dep[i] = 0 } }
            let fx = k.K[0][0] * Float(k.dw) / k.w, fy = k.K[1][1] * Float(k.dh) / k.h, cx = k.K[2][0] * Float(k.dw) / k.w, cy = k.K[2][1] * Float(k.dh) / k.h
            // 5.0: every 2nd depth pixel (was every 4th) with its surface direction, from the pixels beside it
            func at(_ x: Int, _ y: Int) -> SIMD3<Float>? {
                guard x >= 0, y >= 0, x < k.dw, y < k.dh else { return nil }
                let z = dep[y * k.dw + x]; guard z > 0.15 && z < 4 else { return nil }
                return SIMD3((Float(x) - cx) * z / fx, -(Float(y) - cy) * z / fy, -z)
            }
            var pts: [SIMD3<Float>] = [], nrm: [SIMD3<Float>] = []
            for y in stride(from: 2, to: k.dh - 2, by: 2) { for x in stride(from: 2, to: k.dw - 2, by: 2) {
                guard let p = at(x, y), let r = at(x + 2, y), let l = at(x - 2, y), let u = at(x, y - 2), let dn = at(x, y + 2) else { continue }
                // a depth jump (an edge between two things) has no single surface direction: skipped
                if abs(r.z - l.z) > 0.04 * -p.z || abs(u.z - dn.z) > 0.04 * -p.z { continue }
                var nv = simd_cross(r - l, u - dn); let len = simd_length(nv); guard len > 1e-8 else { continue }
                nv /= len; if simd_dot(nv, p) > 0 { nv = -nv }                 // facing the camera
                pts.append(p); nrm.append(nv)
            } }
            if pts.count > 300 { kfs.append(KF(kf: k, local: pts, normals: nrm, depth: dep, dw: k.dw, dh: k.dh)) }
        }
        return kfs
    }

    /// 5.0 point-to-plane ICP with robust weights: moves `src` (world points) onto the surfaces of `dst` (world points
    /// with normals). Returns the correction, the share of points that found a surface, and the median residual.
    static func icpPlane(_ src: [SIMD3<Float>], _ dst: [SIMD3<Float>], _ dn: [SIMD3<Float>], _ g: GridIndex, maxDist: Float, iterations: Int) -> (simd_float4x4, Float, Float) {
        var T = matrix_identity_float4x4, fit: Float = 0, med: Float = 1
        for it in 0..<iterations {
            var A = [Double](repeating: 0, count: 36), bv = [Double](repeating: 0, count: 6)
            var res: [Float] = []; res.reserveCapacity(src.count)
            let lim = it < 3 ? maxDist : max(0.004, min(maxDist, med * 3))
            var used = 0
            for q0 in src {
                let q = CloudOps.apply(T, q0)
                guard let (j, _) = g.nearest(q, in: dst, within: lim) else { continue }
                let n = dn[Int(j)], r = simd_dot(q - dst[Int(j)], n)
                guard abs(r) < lim else { continue }
                res.append(abs(r))
                let w = Double(1 / max(1, abs(r) / 0.004))                     // Huber: far matches count less
                let c = simd_cross(q, n)
                let row: [Double] = [Double(c.x), Double(c.y), Double(c.z), Double(n.x), Double(n.y), Double(n.z)]
                for a in 0..<6 { bv[a] -= w * row[a] * Double(r); for b2 in 0..<6 { A[a * 6 + b2] += w * row[a] * row[b2] } }
                used += 1
            }
            fit = Float(used) / Float(max(1, src.count))
            guard used > 60 else { return (T, fit, 1) }
            res.sort(); med = res[res.count / 2]
            // solve A x = b (6x6, Gaussian elimination with a little damping)
            // directions the surfaces cannot pin down (sliding along a flat desk) stay put: damping against no movement
            var tr = 0.0; for a in 0..<6 { tr += A[a * 6 + a] }
            for a in 0..<6 { A[a * 6 + a] += 0.02 * tr / 6 + 1e-9 }
            var M = A, v = bv
            for c in 0..<6 {
                var piv = c; for r in c..<6 where abs(M[r * 6 + c]) > abs(M[piv * 6 + c]) { piv = r }
                if piv != c { for k in 0..<6 { M.swapAt(c * 6 + k, piv * 6 + k) }; v.swapAt(c, piv) }
                let d = M[c * 6 + c]; guard abs(d) > 1e-12 else { return (T, fit, med) }
                for r in 0..<6 where r != c {
                    let f = M[r * 6 + c] / d; if f == 0 { continue }
                    for k in 0..<6 { M[r * 6 + k] -= f * M[c * 6 + k] }; v[r] -= f * v[c]
                }
            }
            let x = (0..<6).map { Float(v[$0] / M[$0 * 6 + $0]) }
            let w3 = SIMD3<Float>(x[0], x[1], x[2]), ang = simd_length(w3)
            var D = matrix_identity_float4x4
            if ang > 1e-9 { D = m44(simd_float3x3(simd_quatf(angle: ang, axis: w3 / ang))) }
            D.columns.3 = SIMD4(x[3], x[4], x[5], 1)
            T = D * T
            if ang < 1e-5 && simd_length(SIMD3(x[3], x[4], x[5])) < 1e-5 { break }
        }
        return (T, fit, med)
    }

    /// the pose graph: photo-to-photo matches (neighbours and revisits), settled into one correction X per photo
    /// (photo 0 stays put). Returns the corrections, how many matches held, and how many were revisits.
    static func solve(_ kfs: [KF]) -> ([simd_float4x4], Int, Int) {
        let n = kfs.count
        guard n >= 4 else { return ([simd_float4x4](repeating: matrix_identity_float4x4, count: n), 0, 0) }
        let worlds = (0..<n).map { i in kfs[i].local.map { CloudOps.apply(kfs[i].kf.T, $0) } }
        let R = (0..<n).map { i in simd_float3x3(kfs[i].kf.T.columns.0.xyz3, kfs[i].kf.T.columns.1.xyz3, kfs[i].kf.T.columns.2.xyz3) }
        let wn = (0..<n).map { i in kfs[i].normals.map { R[i] * $0 } }
        let pos = kfs.map { SIMD3<Float>($0.kf.T.columns.3.x, $0.kf.T.columns.3.y, $0.kf.T.columns.3.z) }
        let fwd = kfs.map { -SIMD3<Float>($0.kf.T.columns.2.x, $0.kf.T.columns.2.y, $0.kf.T.columns.2.z) }
        var pairs: [(Int, Int)] = []
        for i in 1..<n {
            pairs.append((i, i - 1)); if i >= 2 { pairs.append((i, i - 2)) }
            var loops = 0
            for j in stride(from: i - 4, through: 0, by: -1) where loops < 5 {
                if simd_length(pos[i] - pos[j]) < 0.9 && simd_dot(fwd[i], fwd[j]) > 0.5 { pairs.append((i, j)); loops += 1 }
            }
        }
        // each match: a correction C (moves i's points onto j's surfaces) and how much to trust it
        var results = [(simd_float4x4, Float)?](repeating: nil, count: pairs.count)
        results.withUnsafeMutableBufferPointer { rb in
            let rp = rb.baseAddress!
            DispatchQueue.concurrentPerform(iterations: pairs.count) { pi in
                let (i, j) = pairs[pi]
                let src = stride(from: 0, to: worlds[i].count, by: max(1, worlds[i].count / 2500)).map { worlds[i][$0] }
                let g = GridIndex(worlds[j], cell: 0.03)
                let (T, fit, med) = icpPlane(src, worlds[j], wn[j], g, maxDist: 0.03, iterations: 20)
                let moved = simd_length(SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z))
                if fit > 0.3 && med < 0.006 && moved < 0.08 { rp[pi] = (T, fit / max(0.001, med)) }
            }
        }
        var edges: [(Int, Int, simd_float4x4, Float)] = []
        for (pi, r) in results.enumerated() { if let (T, w) = r { edges.append((pairs[pi].0, pairs[pi].1, T, w)) } }
        let loopsFound = edges.filter { $0.0 - $0.1 > 3 }.count
        // weighted relaxation: X_i = weighted mean of X_j * C_ij over its matches; photo 0 stays put
        var X = [simd_float4x4](repeating: matrix_identity_float4x4, count: n)
        var nb = [[(Int, simd_float4x4, Float)]](repeating: [], count: n)
        for (i, j, C, w) in edges { nb[i].append((j, C, w)); nb[j].append((i, C.inverse, w)) }
        // ARKit's own poses are usually close: each photo is also pulled towards staying where it is
        let wsorted = edges.map { $0.3 }.sorted(), prior = wsorted.isEmpty ? 1 : wsorted[wsorted.count / 2] * 0.5
        for _ in 0..<200 {
            for i in 1..<n where !nb[i].isEmpty { X[i] = wavg(nb[i].map { (X[$0.0] * $0.1, $0.2) } + [(matrix_identity_float4x4, prior)]) }
        }
        // 5.0: never make it worse - the correction is kept only if revisits agree better with it than without it
        let before = gap(kfs, worlds, wn, nil), after = gap(kfs, worlds, wn, X)
        Log.add(String(format: "drift: revisit surface gap %.2f -> %.2f mm (%d matches, %d revisits)", before * 1000, after * 1000, edges.count, loopsFound))
        lastGap = (before, after)
        if after >= before * 0.97 { return ([simd_float4x4](repeating: matrix_identity_float4x4, count: n), edges.count, loopsFound) }
        return (X, edges.count, loopsFound)
    }
    nonisolated(unsafe) static var lastGap: (Float, Float) = (0, 0)
    /// median surface-to-surface distance between photos of the same place taken a while apart (with corrections X)
    static func gap(_ kfs: [KF], _ worlds: [[SIMD3<Float>]], _ wn: [[SIMD3<Float>]], _ X: [simd_float4x4]?) -> Float {
        let n = kfs.count
        let pos = kfs.map { SIMD3<Float>($0.kf.T.columns.3.x, $0.kf.T.columns.3.y, $0.kf.T.columns.3.z) }
        var pairs: [(Int, Int)] = []
        for i in 0..<n { for j in 0..<i where i - j > 8 && simd_length(pos[i] - pos[j]) < 0.5 { pairs.append((i, j)) } }
        if pairs.count > 80 { let st = pairs.count / 80; pairs = stride(from: 0, to: pairs.count, by: st).map { pairs[$0] } }
        var all = [[Float]](repeating: [], count: pairs.count)
        all.withUnsafeMutableBufferPointer { ab in
            let ap = ab.baseAddress!
            DispatchQueue.concurrentPerform(iterations: pairs.count) { k in
                let (i, j) = pairs[k]
                func mv(_ x: Int, _ q: SIMD3<Float>) -> SIMD3<Float> { X.map { CloudOps.apply($0[x], q) } ?? q }
                func mn(_ x: Int, _ v: SIMD3<Float>) -> SIMD3<Float> { guard let X = X else { return v }; let r = X[x] * SIMD4(v, 0); return SIMD3(r.x, r.y, r.z) }
                let b = worlds[j].map { mv(j, $0) }, bn = wn[j].map { mn(j, $0) }
                let g = GridIndex(b, cell: 0.03)
                var d: [Float] = []
                for q0 in stride(from: 0, to: worlds[i].count, by: max(1, worlds[i].count / 1500)).map({ worlds[i][$0] }) {
                    let q = mv(i, q0)
                    if let (m, _) = g.nearest(q, in: b, within: 0.03) { d.append(abs(simd_dot(q - b[Int(m)], bn[Int(m)]))) }
                }
                if d.count > 100 { ap[k] = d }
            }
        }
        var ds = all.flatMap { $0 }; guard !ds.isEmpty else { return 0 }
        ds.sort(); return ds[ds.count / 2]
    }
    static func wavg(_ ts: [(simd_float4x4, Float)]) -> simd_float4x4 {
        guard let first = ts.first else { return matrix_identity_float4x4 }
        var t = SIMD3<Float>.zero, q = SIMD4<Float>.zero, W: Float = 0
        let q0 = simd_quatf(first.0).vector
        for (m, w) in ts {
            t += SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z) * w
            var v = simd_quatf(m).vector; if simd_dot(v, q0) < 0 { v = -v }; q += v * w; W += w
        }
        var out = m44(simd_float3x3(simd_quatf(vector: simd_normalize(q))))
        out.columns.3 = SIMD4(t / max(1e-9, W), 1)
        return out
    }

    static func fix(_ cl: Cloud, item: ScanItem, progress: @escaping (Float, String) -> Void) -> (Cloud, String) {
        guard Library.captureURL(item) != nil else { return (cl, "This scan has no photos, so drift cannot be measured.") }
        progress(0.02, "Reading each photo's depth...")
        let kfs = load(item)
        let n = kfs.count
        guard n >= 4 else { return (cl, "Too few usable depth maps to measure drift.") }
        progress(0.2, "Matching each photo with its neighbours and revisits...")
        let (X, edgeCount, loopsFound) = solve(kfs)
        if X.allSatisfy({ $0 == matrix_identity_float4x4 }) {
            return (cl, String(format: "No drift to fix: photos of the same places already agree to %.1f mm, and no correction made that better. Your scan was left as it is.", lastGap.0 * 1000))
        }
        progress(0.45, "Settled \(edgeCount) matches between photos...")
        // each point moves with the photos whose depth agrees with it
        progress(0.6, "Moving every point with the photos that saw it...")
        let views = kfs.map { DepthView(kf: $0.kf, depth: $0.depth, dw: $0.dw, dh: $0.dh) }
        var out = cl
        var movedSum: Float = 0, movedN = 0
        out.p.withUnsafeMutableBufferPointer { ob in
            let op = ob.baseAddress!
            parallelFor(cl.count) { i in
                let p = cl.p[i]
                var ts: [simd_float4x4] = []
                for (k, v) in views.enumerated() {
                    guard let (u, vv, d) = v.kf.project(p) else { continue }
                    let md = v.measured(u, vv)
                    if md > 0 && abs(d - md) < 0.03 { ts.append(X[k]) }
                    if ts.count >= 12 { break }
                }
                if !ts.isEmpty { op[i] = CloudOps.apply(avg(ts), p) }
            }
        }
        for i in stride(from: 0, to: cl.count, by: max(1, cl.count / 20000)) { movedSum += simd_length(out.p[i] - cl.p[i]); movedN += 1 }
        progress(0.95, "Merging the surfaces that came together...")
        let merged = CloudOps.dedupe(out.sane(), grid: Float(Prefs.mergeGridMM / 1000))
        let avgMove = movedN > 0 ? movedSum / Float(movedN) : 0
        return (merged, String(format: "Revisits now agree to %.1f mm (was %.1f). ", lastGap.1 * 1000, lastGap.0 * 1000) + "Drift fixed: \(edgeCount) matches between \(n) photos (\(loopsFound) revisits), points moved \(Prefs.length(avgMove)) on average, \(EditorModel.change(cl.count, merged.count))")
    }
}

extension SIMD4 where Scalar == Float { var xyz3: SIMD3<Float> { SIMD3(x, y, z) } }
