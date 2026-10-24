import Foundation
import simd

/// one photo's LiDAR depth with where the camera was
struct DepthFrame { let T: simd_float4x4; let fx: Float; let fy: Float; let cx: Float; let cy: Float; let depth: [Float]; let dw: Int; let dh: Int
    /// 5.7 (47): the LiDAR confidence per depth pixel (0 low, 1 medium, 2 high), nil when the capture has none
    var conf: [UInt8]? = nil }

/// 5.0: the surface rebuilt from the photos' own depth maps (a truncated signed distance field, the method LiDAR scan
/// apps use): every depth pixel says "empty space up to here, surface here", so inside and outside can never flip
/// the way they do when a surface is guessed from noisy points. Many views are averaged, so noise cancels out.
enum DepthMesher {
    @inline(__always) static func key(_ x: Int32, _ y: Int32, _ z: Int32) -> Int64 { Int64(x) & 0x1FFFFF | ((Int64(y) & 0x1FFFFF) << 21) | ((Int64(z) & 0x1FFFFF) << 42) }

    static func build(_ frames: [DepthFrame], h: Float, minWeight: Float = 1.5, progress: (Float, String) -> Void) -> MeshData? {
        let tau = max(3 * h, 0.015)
        // a work budget: at most ~45M ray samples whatever the room size and frame count (every Nth pixel when needed)
        let steps = Float(2 * tau / (h * 0.75)) + 1
        let perFrame = Float(frames.first.map { $0.dw * $0.dh } ?? 49152) * steps
        let px = max(1, Int((Float(frames.count) * perFrame / 45_000_000).squareRoot().rounded(.up)))
        var index: [Int64: Int32] = [:]; index.reserveCapacity(4_000_000)
        // 5.7 (47): wt (the old weight) only decides which voxels are kept, so the same voxels survive as before; wq is the
        // quality weight the distance is averaged with (confidence, how square-on the view is, LiDAR noise growing with d^2)
        var sd: [Float] = [], wt: [Float] = [], wq: [Float] = []
        sd.reserveCapacity(4_000_000); wt.reserveCapacity(4_000_000); wq.reserveCapacity(4_000_000)
        for (fi, f) in frames.enumerated() {
            let R = simd_float3x3(SIMD3(f.T.columns.0.x, f.T.columns.0.y, f.T.columns.0.z), SIMD3(f.T.columns.1.x, f.T.columns.1.y, f.T.columns.1.z), SIMD3(f.T.columns.2.x, f.T.columns.2.y, f.T.columns.2.z))
            let o = SIMD3<Float>(f.T.columns.3.x, f.T.columns.3.y, f.T.columns.3.z)
            for y in stride(from: 1, to: f.dh - 1, by: px) { for x in stride(from: 1, to: f.dw - 1, by: px) {
                let d = f.depth[y * f.dw + x]
                guard d > 0.15 && d < 5 else { continue }
                // flying pixels on depth edges (half foreground, half background) are skipped
                let l = f.depth[y * f.dw + x - 1], r = f.depth[y * f.dw + x + 1], u = f.depth[(y - 1) * f.dw + x], dn = f.depth[(y + 1) * f.dw + x]
                let lim = 0.03 * d
                if abs(l - d) > lim || abs(r - d) > lim || abs(u - d) > lim || abs(dn - d) > lim { continue }
                let rw = R * SIMD3((Float(x) - f.cx) / f.fx, -(Float(y) - f.cy) / f.fy, -1)
                let w: Float = 1 / max(0.4, d)                                // nearer readings count more
                // 5.7 (47): grazing views are weighted low, never skipped (camera-space normal from the four neighbours)
                let confW: Float = f.conf.map { $0.count == f.depth.count && $0[y * f.dw + x] < 2 ? 0.45 : 1 } ?? 1
                func cp(_ xx: Int, _ yy: Int, _ dd: Float) -> SIMD3<Float> { SIMD3((Float(xx) - f.cx) / f.fx * dd, -(Float(yy) - f.cy) / f.fy * dd, -dd) }
                let nc = simd_cross(cp(x + 1, y, r) - cp(x - 1, y, l), cp(x, y - 1, u) - cp(x, y + 1, dn))
                let nl = simd_length(nc)
                let cosA: Float = nl < 1e-9 ? 1 : abs(simd_dot(nc / nl, simd_normalize(SIMD3((Float(x) - f.cx) / f.fx, -(Float(y) - f.cy) / f.fy, -1))))
                let q = confW * max(0.1, cosA) / max(0.16, d * d)
                var z = d - tau
                while z <= d + tau {
                    let p = o + rw * z
                    let k = key(ik(p.x / h), ik(p.y / h), ik(p.z / h))
                    let s = min(1, (d - z) / tau)
                    if let i = index[k] { sd[Int(i)] += s * q; wt[Int(i)] += w; wq[Int(i)] += q }
                    else { index[k] = Int32(sd.count); sd.append(s * q); wt.append(w); wq.append(q) }
                    z += h * 0.75
                }
            } }
            if fi % 10 == 0 { progress(0.6 * Float(fi + 1) / Float(frames.count), "Fusing depth \(fi + 1)/\(frames.count)...") }
        }
        // surface nets over the voxel centres
        progress(0.65, "Making triangles...")
        func val(_ k: Int64) -> Float? { guard let i = index[k], wt[Int(i)] >= minWeight else { return nil }; return sd[Int(i)] / wq[Int(i)] }
        func ijk(_ k: Int64) -> SIMD3<Int32> {
            func s(_ v: Int64) -> Int32 { let x = Int32(v & 0x1FFFFF); return x >= 0x100000 ? x - 0x200000 : x }
            return SIMD3(s(k), s(k >> 21), s(k >> 42))
        }
        let corners: [SIMD3<Int32>] = [SIMD3(0,0,0), SIMD3(1,0,0), SIMD3(0,1,0), SIMD3(1,1,0), SIMD3(0,0,1), SIMD3(1,0,1), SIMD3(0,1,1), SIMD3(1,1,1)]
        let edges: [(Int, Int)] = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
        var cellVert: [Int64: UInt32] = [:]
        var mesh = MeshData()
        func vertex(_ c: SIMD3<Int32>) -> UInt32? {
            let ck = key(c.x, c.y, c.z)
            if let v = cellVert[ck] { return v }
            var vals = [Float](repeating: 0, count: 8)
            for (i, cc) in corners.enumerated() { let q = c &+ cc; guard let v = val(key(q.x, q.y, q.z)) else { return nil }; vals[i] = v }
            var sum = SIMD3<Float>.zero, cnt: Float = 0
            for (a, b) in edges {
                let va = vals[a], vb = vals[b]
                guard (va < 0) != (vb < 0) else { continue }
                let t = va / (va - vb)
                let pa = SIMD3<Float>(Float(corners[a].x), Float(corners[a].y), Float(corners[a].z)), pb = SIMD3<Float>(Float(corners[b].x), Float(corners[b].y), Float(corners[b].z))
                sum += pa + (pb - pa) * t; cnt += 1
            }
            guard cnt > 0 else { return nil }
            let p = (SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)) + sum / cnt + 0.5) * h      // voxel centres sit at (i + 0.5) h
            let idx = UInt32(mesh.v.count)
            mesh.v.append(p); mesh.c.append(SIMD3(0.7, 0.7, 0.7)); mesh.n.append(.zero)
            cellVert[ck] = idx
            return idx
        }
        let axes: [SIMD3<Int32>] = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
        for (k, i) in index where wt[Int(i)] >= minWeight {
            let s0 = sd[Int(i)] / wq[Int(i)]
            let c = ijk(k)
            for (ai, ax) in axes.enumerated() {
                let n = c &+ ax
                guard let s1 = val(key(n.x, n.y, n.z)), (s0 < 0) != (s1 < 0) else { continue }
                // a crossing at the truncation edge is not a surface (it is where one view's band ends)
                guard abs(s0) < 0.99 && abs(s1) < 0.99 else { continue }
                let u = axes[(ai + 1) % 3], v = axes[(ai + 2) % 3]
                guard let a = vertex(c), let b = vertex(c &- u), let cc = vertex(c &- u &- v), let d = vertex(c &- v) else { continue }
                if s0 < 0 { mesh.f += [a, b, cc, a, cc, d] } else { mesh.f += [a, cc, b, a, d, cc] }
            }
        }
        // 5.7 (47): numbers for a before/after comparison (how far the averaged distances sit from zero near the surface)
        var nearSum: Double = 0, nearN = 0, keptN = 0
        for i in 0..<sd.count where wt[i] >= minWeight { keptN += 1; let v = sd[i] / wq[i]; if abs(v) < 0.5 { nearSum += Double(abs(v)); nearN += 1 } }
        Log.add("DepthMesher: \(index.count) voxels, \(keptN) kept, mean |sd| near surface \(nearN > 0 ? String(format: "%.4f", nearSum / Double(nearN)) : "-") over \(nearN), conf frames \(frames.filter { $0.conf != nil }.count)/\(frames.count)")
        guard !mesh.f.isEmpty else { return nil }
        mesh = clean(mesh, minTris: 1200)
        guard !mesh.f.isEmpty else { return nil }
        var t = 0
        while t < mesh.f.count {
            let a = Int(mesh.f[t]), b = Int(mesh.f[t + 1]), c = Int(mesh.f[t + 2])
            let fn = simd_cross(mesh.v[b] - mesh.v[a], mesh.v[c] - mesh.v[a])
            mesh.n[a] += fn; mesh.n[b] += fn; mesh.n[c] += fn
            t += 3
        }
        for i in 0..<mesh.n.count { let l = simd_length(mesh.n[i]); mesh.n[i] = l > 0 ? mesh.n[i] / l : SIMD3(0, 1, 0) }
        progress(1, "Done")
        return mesh
    }

    /// removes pieces of surface that touch nothing else and are small (specks, fragments around wires), then smooths
    /// the stair-stepping of the voxel grid with Taubin smoothing (shrinks nothing)
    static func clean(_ m: MeshData, minTris: Int) -> MeshData {
        let nv = m.v.count, nt = m.f.count / 3
        var parent = Array(0..<nv)
        func find(_ a: Int) -> Int { var x = a; while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x] }; return x }
        for t in 0..<nt { let a = find(Int(m.f[3 * t])), b = find(Int(m.f[3 * t + 1])), c = find(Int(m.f[3 * t + 2])); parent[b] = a; parent[find(c)] = find(a) }
        var cnt = [Int](repeating: 0, count: nv)
        for t in 0..<nt { cnt[find(Int(m.f[3 * t]))] += 1 }
        var out = MeshData(); var remap = [Int32](repeating: -1, count: nv)
        for t in 0..<nt {
            guard cnt[find(Int(m.f[3 * t]))] >= minTris else { continue }
            for k in 0..<3 {
                let vi = Int(m.f[3 * t + k])
                if remap[vi] < 0 { remap[vi] = Int32(out.v.count); out.v.append(m.v[vi]); out.c.append(m.c[vi]); out.n.append(m.n[vi]) }
                out.f.append(UInt32(remap[vi]))
            }
        }
        // Taubin smoothing over the edge neighbours
        var nb = [[Int32]](repeating: [], count: out.v.count)
        var t = 0
        while t < out.f.count { for k in 0..<3 { let a = out.f[t + k], b = out.f[t + (k + 1) % 3]; nb[Int(a)].append(Int32(b)); nb[Int(b)].append(Int32(a)) }; t += 3 }
        var P = out.v
        for it in 0..<4 {
            let lam: Float = it % 2 == 0 ? 0.5 : -0.53
            var Q = P
            for i in 0..<P.count where !nb[i].isEmpty {
                var avg = SIMD3<Float>.zero; for j in nb[i] { avg += P[Int(j)] }
                avg /= Float(nb[i].count)
                Q[i] = P[i] + (avg - P[i]) * lam
            }
            P = Q
        }
        out.v = P
        return out
    }
}
