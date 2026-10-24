import Foundation
import simd
import SceneKit

// ---------- conversions that never crash (Swift traps on NaN / infinity / out-of-range when making integers) ----------
@inline(__always) func ik(_ v: Float) -> Int32 {
    guard v.isFinite else { return -1_000_000_000 }
    return v >= 1e9 ? 1_000_000_000 : (v <= -1e9 ? -1_000_000_000 : Int32(v.rounded(.down)))
}
@inline(__always) func ik64(_ v: Float) -> Int64 { Int64(ik(v)) }
@inline(__always) func u8(_ v: Float) -> UInt8 { v.isFinite ? UInt8(max(0, min(255, v.rounded()))) : 128 }
@inline(__always) func finite(_ p: SIMD3<Float>) -> Bool { p.x.isFinite && p.y.isFinite && p.z.isFinite && abs(p.x) < 1e4 && abs(p.y) < 1e4 && abs(p.z) < 1e4 }

/// A finished point cloud (metres, ARKit world, y up) with 8-bit colours. Everything the editor does works on this.
struct Cloud {
    var p: [SIMD3<Float>] = []
    var c: [SIMD3<UInt8>] = []
    var count: Int { p.count }

    struct Box { var lo: SIMD3<Float>; var hi: SIMD3<Float>; var size: SIMD3<Float> { hi - lo }; var centre: SIMD3<Float> { (lo + hi) / 2 } }
    var bounds: Box {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for q in p { lo = simd_min(lo, q); hi = simd_max(hi, q) }
        if p.isEmpty { lo = .zero; hi = .zero }
        return Box(lo: lo, hi: hi)
    }
    /// drops any damaged point (NaN, infinite, absurdly far) - every edit ends with this
    func sane() -> Cloud {
        guard p.contains(where: { !finite($0) }) else { return self }
        return keeping(p.map { finite($0) })
    }
    func keeping(_ keep: [Bool]) -> Cloud {
        var o = Cloud(); o.p.reserveCapacity(p.count); o.c.reserveCapacity(p.count)
        for i in 0..<p.count where keep[i] { o.p.append(p[i]); o.c.append(c[i]) }
        return o
    }

    // ---- files ----
    /// raw: count (UInt32), then x y z Float32 and r g b UInt8 per point
    func data() -> Data {
        var d = Data(capacity: 4 + p.count * 15)
        var n = UInt32(p.count); withUnsafeBytes(of: &n) { d.append(contentsOf: $0) }
        for q in p { var v = q; withUnsafeBytes(of: &v) { d.append(contentsOf: $0.prefix(12)) } }
        for k in c { d.append(contentsOf: [k.x, k.y, k.z]) }
        return d
    }
    /// 5.0: the same file as data(), written in 1M-point pieces (a 12M-point scan no longer needs a 190 MB copy)
    func write(to url: URL) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let h = try FileHandle(forWritingTo: url)
        defer { try? h.close() }
        var n = UInt32(p.count); try h.write(contentsOf: withUnsafeBytes(of: &n) { Data($0) })
        let chunk = 1_000_000
        var i = 0
        while i < p.count {
            let e = min(p.count, i + chunk)
            var d = Data(capacity: (e - i) * 12)
            for j in i..<e { var v = p[j]; withUnsafeBytes(of: &v) { d.append(contentsOf: $0.prefix(12)) } }
            try h.write(contentsOf: d); i = e
        }
        i = 0
        while i < c.count {
            let e = min(c.count, i + chunk)
            var d = Data(capacity: (e - i) * 3)
            for j in i..<e { d.append(contentsOf: [c[j].x, c[j].y, c[j].z]) }
            try h.write(contentsOf: d); i = e
        }
    }
    init() {}
    init(p: [SIMD3<Float>], c: [SIMD3<UInt8>]) { self.p = p; self.c = c }
    init?(data d: Data) {
        guard d.count >= 4 else { return nil }
        let n = Int(d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) })
        guard d.count >= 4 + n * 15 else { return nil }
        p.reserveCapacity(n); c.reserveCapacity(n)
        d.withUnsafeBytes { b in
            for i in 0..<n {
                let o = 4 + i * 12
                p.append(SIMD3(b.loadUnaligned(fromByteOffset: o, as: Float.self), b.loadUnaligned(fromByteOffset: o + 4, as: Float.self), b.loadUnaligned(fromByteOffset: o + 8, as: Float.self)))
            }
            let co = 4 + n * 12
            for i in 0..<n { c.append(SIMD3(b[co + 3 * i], b[co + 3 * i + 1], b[co + 3 * i + 2])) }
        }
    }
    func ply() -> Data {
        var d = Data("ply\nformat binary_little_endian 1.0\ncomment Deepfield 5.5, metres, y up\nelement vertex \(p.count)\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".utf8)
        d.reserveCapacity(d.count + p.count * 15)
        for i in 0..<p.count {
            var v = p[i]; withUnsafeBytes(of: &v) { d.append(contentsOf: $0.prefix(12)) }
            d.append(contentsOf: [c[i].x, c[i].y, c[i].z])
        }
        return d
    }

    // ---- display ----
    /// up to maxPoints points, centred, as a SceneKit point cloud; colour mode "photo" or "height"
    func geometry(maxPoints: Int, mode: String, size: CGFloat) -> (SCNGeometry, SIMD3<Float>)? {
        guard !p.isEmpty else { return nil }
        let b = bounds, step = max(1, p.count / maxPoints)
        var verts: [SCNVector3] = [], cols: [Float] = []
        verts.reserveCapacity(p.count / step + 1); cols.reserveCapacity(3 * (p.count / step + 1))
        let h = max(0.001, b.size.y)
        var i = 0
        while i < p.count {
            let q = p[i] - b.centre
            verts.append(SCNVector3(q.x, q.y, q.z))
            if mode == "height" {
                let t = (p[i].y - b.lo.y) / h
                cols += [min(1, 2 * t), 1 - abs(2 * t - 1) * 0.6, max(0, 1 - 2 * t)]
            } else { cols += [Float(c[i].x) / 255, Float(c[i].y) / 255, Float(c[i].z) / 255] }
            i += step
        }
        return (pointGeometry(verts, cols, size: size), b.centre)
    }
}

/// uniform grid over point indices, for neighbour searches
struct GridIndex {
    let h: Float
    var cells: [SIMD3<Int32>: [Int32]] = [:]
    init(_ pts: [SIMD3<Float>], cell: Float) {
        h = cell
        cells.reserveCapacity(pts.count / 4)
        for (i, q) in pts.enumerated() { cells[key(q), default: []].append(Int32(i)) }
    }
    @inline(__always) func key(_ q: SIMD3<Float>) -> SIMD3<Int32> {
        SIMD3<Int32>(ik((q.x / h)), ik((q.y / h)), ik((q.z / h)))
    }
    /// calls body for every point in the 27 cells around q
    @inline(__always) func around(_ q: SIMD3<Float>, _ body: (Int32) -> Void) {
        let k = key(q)
        for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 {
            if let l = cells[k &+ SIMD3(dx, dy, dz)] { for j in l { body(j) } }
        } } }
    }
    func nearest(_ q: SIMD3<Float>, in pts: [SIMD3<Float>], within r: Float) -> (Int32, Float)? {
        var best: Int32 = -1, bd = r * r
        around(q) { j in let d = simd_length_squared(pts[Int(j)] - q); if d < bd { bd = d; best = j } }
        return best >= 0 ? (best, bd.squareRoot()) : nil
    }
}

/// runs body(i) for i in 0..<n on all cores
func parallelFor(_ n: Int, _ body: (Int) -> Void) {
    let chunks = max(1, min(64, n / 5000))
    let per = (n + chunks - 1) / chunks
    DispatchQueue.concurrentPerform(iterations: chunks) { ci in
        let a = ci * per, b = min(n, a + per)
        if a < b { for i in a..<b { body(i) } }
    }
}

enum CloudOps {
    /// average on a grid: removes doubles (two points in one cell become one).
    /// 5.0: sorted cell keys (12 bytes a point) instead of a dictionary of sums (~60 bytes a point): a 12M-point scan
    /// needed ~700 MB here and could be killed by iOS.
    static func dedupe(_ cl: Cloud, grid g: Float) -> Cloud {
        let n = cl.count
        guard n > 1 else { return cl }
        var keys = [Int64](repeating: 0, count: n)
        keys.withUnsafeMutableBufferPointer { kb in
            let kp = kb.baseAddress!
            parallelFor(n) { i in
                let q = cl.p[i]
                kp[i] = ik64(q.x / g) & 0x1FFFFF | ((ik64(q.y / g) & 0x1FFFFF) << 21) | ((ik64(q.z / g) & 0x1FFFFF) << 42)
            }
        }
        var order = (0..<Int32(n)).map { $0 }
        keys.withUnsafeBufferPointer { kb in order.sort { kb[Int($0)] < kb[Int($1)] } }
        var o = Cloud(); o.p.reserveCapacity(n); o.c.reserveCapacity(n)
        var j = 0
        while j < n {
            let k = keys[Int(order[j])]
            var sp = SIMD3<Float>.zero, sc = SIMD3<Float>.zero, m: Float = 0
            while j < n && keys[Int(order[j])] == k {
                let i = Int(order[j]); sp += cl.p[i]; sc += SIMD3<Float>(Float(cl.c[i].x), Float(cl.c[i].y), Float(cl.c[i].z)); m += 1; j += 1
            }
            o.p.append(sp / m); let c = sc / m; o.c.append(SIMD3(u8(c.x), u8(c.y), u8(c.z)))
        }
        return o
    }

    /// floaters: a point with fewer than minN others within r is dropped
    static func removeFloaters(_ cl: Cloud, radius r: Float, minN: Int) -> Cloud { cl.keeping(floaterKeep(cl, radius: r, minN: minN)) }
    /// 4.5: the radius is never below 4x this scan's own point spacing (a sparse scan is not all "floaters")
    static func floaterKeep(_ cl: Cloud, radius rIn: Float, minN: Int) -> [Bool] {
        let r = max(rIn, spacing(cl) * 4)
        let g = GridIndex(cl.p, cell: r), r2 = r * r
        var keep = [Bool](repeating: false, count: cl.count)
        keep.withUnsafeMutableBufferPointer { kb in
            let kp = kb.baseAddress!
            parallelFor(cl.count) { i in
                let q = cl.p[i]; var n = 0
                g.around(q) { j in if simd_length_squared(cl.p[Int(j)] - q) <= r2 { n += 1 } }
                kp[i] = n - 1 >= minN
            }
        }
        return keep
    }

    /// 4.5: edge-keeping smoothing - each point moves only ACROSS its surface (onto the local plane), and less where
    /// the surface bends (corners, edges, thin parts), so grain goes and shape stays
    static func smooth(_ cl: Cloud, radius r: Float, amount: Float) -> Cloud {
        let g = GridIndex(cl.p, cell: r), r2 = r * r
        var out = cl.p
        out.withUnsafeMutableBufferPointer { ob in
            let op = ob.baseAddress!
            parallelFor(cl.count) { i in
                let q = cl.p[i]; var nb: [SIMD3<Float>] = []; nb.reserveCapacity(32)
                g.around(q) { j in let v = cl.p[Int(j)]; if simd_length_squared(v - q) <= r2 { nb.append(v) } }
                guard nb.count > 5, let (n, c, flat) = planeNormal(nb) else { return }
                let k = amount * max(0, 1 - flat / 0.05)             // flat: full; curved: little; a corner: none
                op[i] = q - n * simd_dot(q - c, n) * k
            }
        }
        var o = cl; o.p = out; return o
    }
    /// typical spacing between neighbouring points (for tools that adapt to the scan)
    static func spacing(_ cl: Cloud) -> Float {
        guard cl.count > 100 else { return 0.003 }
        // 5.0: exact, without indexing the whole scan: 400 random points, and only the points in the 2 cm cells around
        // them are gathered (one pass), then each one's nearest neighbour is found among those
        let h: Float = 0.02
        @inline(__always) func key(_ q: SIMD3<Float>) -> Int64 { ik64(q.x / h) & 0x1FFFFF | ((ik64(q.y / h) & 0x1FFFFF) << 21) | ((ik64(q.z / h) & 0x1FFFFF) << 42) }
        let seeds = (0..<400).map { _ in Int.random(in: 0..<cl.count) }
        var want = Set<Int64>()
        for i in seeds { let k = key(cl.p[i]); for dx: Int64 in -1...1 { for dy: Int64 in -1...1 { for dz: Int64 in -1...1 { want.insert(k &+ dx &+ (dy << 21) &+ (dz << 42)) } } } }
        var local: [SIMD3<Float>] = []
        for q in cl.p where want.contains(key(q)) { local.append(q) }
        let g = GridIndex(local, cell: h)
        var ds: [Float] = []
        for i in seeds {
            let q = cl.p[i]; var best: Float = h
            g.around(q) { j in let d = simd_length(local[Int(j)] - q); if d > 0 && d < best { best = d } }
            ds.append(best)
        }
        ds.sort(); return ds[ds.count / 2]
    }

    /// finds the biggest near-horizontal plane in the lowest part of the cloud and removes it (RANSAC)
    static func removeFloor(_ cl: Cloud, tolerance t: Float = 0.012) -> (Cloud, Bool) {
        guard cl.count > 100 else { return (cl, false) }
        let b = cl.bounds, cutY = b.lo.y + b.size.y * 0.35
        var low: [SIMD3<Float>] = []
        let step = max(1, cl.count / 60000)
        for i in stride(from: 0, to: cl.count, by: step) where cl.p[i].y <= cutY { low.append(cl.p[i]) }
        guard low.count > 50 else { return (cl, false) }
        var rng = SystemRandomNumberGenerator()
        var bestN = 0, bestNormal = SIMD3<Float>(0, 1, 0), bestD: Float = 0
        for _ in 0..<400 {
            let a = low[Int.random(in: 0..<low.count, using: &rng)], b2 = low[Int.random(in: 0..<low.count, using: &rng)], c2 = low[Int.random(in: 0..<low.count, using: &rng)]
            var nrm = simd_cross(b2 - a, c2 - a); let len = simd_length(nrm)
            guard len > 1e-6 else { continue }
            nrm /= len; if nrm.y < 0 { nrm = -nrm }
            guard nrm.y > 0.85 else { continue }                  // within about 30 degrees of level
            let d = -simd_dot(nrm, a)
            var n = 0; for q in low where abs(simd_dot(nrm, q) + d) < t { n += 1 }
            if n > bestN { bestN = n; bestNormal = nrm; bestD = d }
        }
        guard bestN > low.count / 10 else { return (cl, false) }
        // drop the plane and anything just under it
        let keep = cl.p.map { simd_dot(bestNormal, $0) + bestD > t }
        return (cl.keeping(keep), true)
    }

    /// keeps (or removes) what is inside a box given as 0..1 fractions of the bounds
    static func crop(_ cl: Cloud, lo f0: SIMD3<Float>, hi f1: SIMD3<Float>, keepInside: Bool) -> Cloud {
        let b = cl.bounds, lo = b.lo + b.size * f0, hi = b.lo + b.size * f1
        let keep = cl.p.map { q -> Bool in
            let inside = q.x >= lo.x && q.x <= hi.x && q.y >= lo.y && q.y <= hi.y && q.z >= lo.z && q.z <= hi.z
            return inside == keepInside
        }
        return cl.keeping(keep)
    }

    /// 5.0: keeps (or removes) what is inside a box given in scan coordinates, turned `yaw` radians about up
    static func cropBox(_ cl: Cloud, centre: SIMD3<Float>, half: SIMD3<Float>, yaw: Float, keepInside: Bool) -> Cloud {
        let c = cos(yaw), s = sin(yaw)
        let keep = cl.p.map { p -> Bool in
            let d = p - centre
            let q = SIMD3<Float>(c * d.x - s * d.z, d.y, s * d.x + c * d.z)
            let inside = abs(q.x) <= half.x && abs(q.y) <= half.y && abs(q.z) <= half.z
            return inside == keepInside
        }
        return cl.keeping(keep)
    }

    static func colour(_ cl: Cloud, brightness: Float, contrast: Float, saturation: Float) -> Cloud {
        var o = cl
        for i in 0..<o.count {
            var v = SIMD3<Float>(Float(o.c[i].x), Float(o.c[i].y), Float(o.c[i].z)) / 255
            let grey = simd_dot(v, SIMD3(0.299, 0.587, 0.114))
            v = SIMD3(repeating: grey) + (v - SIMD3(repeating: grey)) * saturation
            v = (v - 0.5) * contrast + 0.5 + brightness
            v = simd_clamp(v, .zero, SIMD3(repeating: 1)) * 255
            o.c[i] = SIMD3(u8(v.x), u8(v.y), u8(v.z))
        }
        return o
    }

    /// 5.0: one point per grid cell, quickly (a Set of cell keys; for working copies, not for the saved scan)
    static func thin(_ cl: Cloud, grid g: Float) -> Cloud {
        var seen = Set<Int64>(); seen.reserveCapacity(cl.count / 4)
        var keep = [Bool](repeating: false, count: cl.count)
        for i in 0..<cl.count {
            let q = cl.p[i]
            let k = ik64(q.x / g) & 0x1FFFFF | ((ik64(q.y / g) & 0x1FFFFF) << 21) | ((ik64(q.z / g) & 0x1FFFFF) << 42)
            keep[i] = seen.insert(k).inserted
        }
        return cl.keeping(keep)
    }

    static func downsample(_ cl: Cloud, to n: Int) -> Cloud {
        guard cl.count > n, n > 0 else { return cl }
        let step = Double(cl.count) / Double(n)
        var o = Cloud(); o.p.reserveCapacity(n); o.c.reserveCapacity(n)
        var x = 0.0
        while Int(x) < cl.count { o.p.append(cl.p[Int(x)]); o.c.append(cl.c[Int(x)]); x += step }
        return o
    }

    /// the one-tap clean-up: floaters out, doubles merged, light smoothing
    /// the safe clean-up when Gemini cannot be reached: only truly isolated specks go, doubles in one cell merge
    static func cleanUp(_ cl: Cloud, grid: Float, smoothIt: Bool) -> Cloud {
        var o = removeFloaters(cl, radius: max(0.012, grid * 5), minN: 2)
        o = dedupe(o, grid: grid)
        if smoothIt { o = smooth(o, radius: max(0.004, grid * 2), amount: 0.4) }
        return o
    }

    // ---- aligning one scan to another ----
    static func apply(_ T: simd_float4x4, _ q: SIMD3<Float>) -> SIMD3<Float> {
        let r = T * SIMD4(q, 1); return SIMD3(r.x, r.y, r.z)
    }

    /// best rigid transform taking a onto b (Horn's quaternion method; power iteration for the top eigenvector)
    static func rigid(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> simd_float4x4 {
        let n = Float(a.count)
        var ca = SIMD3<Float>.zero, cb = SIMD3<Float>.zero
        for i in 0..<a.count { ca += a[i]; cb += b[i] }
        ca /= n; cb /= n
        var S = simd_double3x3()
        for i in 0..<a.count {
            let x = SIMD3<Double>(a[i] - ca), y = SIMD3<Double>(b[i] - cb)
            S.columns.0 += x * y.x; S.columns.1 += x * y.y; S.columns.2 += x * y.z    // S[r][c] = sum x_r y_c
        }
        func s(_ r: Int, _ c: Int) -> Double { S[c][r] }
        let Sxx = s(0,0), Sxy = s(0,1), Sxz = s(0,2), Syx = s(1,0), Syy = s(1,1), Syz = s(1,2), Szx = s(2,0), Szy = s(2,1), Szz = s(2,2)
        let N = simd_double4x4(rows: [
            SIMD4(Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx),
            SIMD4(Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz),
            SIMD4(Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy),
            SIMD4(Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz)])
        var shift = 0.0; for c in 0..<4 { for r in 0..<4 { shift += abs(N[c][r]) } }
        let M = N + simd_double4x4(diagonal: SIMD4(repeating: shift))
        var v = SIMD4<Double>(1, 0, 0, 0)
        for _ in 0..<200 { v = M * v; v /= max(1e-30, simd_length(v)) }
        let q = simd_quatf(ix: Float(v.y), iy: Float(v.z), iz: Float(v.w), r: Float(v.x))
        let R = simd_float3x3(q)
        let t = cb - R * ca
        var T = m44(R); T.columns.3 = SIMD4(t, 1)
        return T
    }

    /// point-to-point ICP; returns the refined transform and the share of source points that found a partner
    static func icp(_ src: [SIMD3<Float>], _ ref: [SIMD3<Float>], _ g: GridIndex, start: simd_float4x4, maxDist: Float, iterations: Int) -> (simd_float4x4, Float) {
        var T = start, fit: Float = 0
        for _ in 0..<iterations {
            var a: [SIMD3<Float>] = [], b: [SIMD3<Float>] = []
            a.reserveCapacity(src.count); b.reserveCapacity(src.count)
            for q in src {
                let w = apply(T, q)
                if let (j, _) = g.nearest(w, in: ref, within: maxDist) { a.append(w); b.append(ref[Int(j)]) }
            }
            fit = Float(a.count) / Float(max(1, src.count))
            guard a.count >= 20 else { break }
            T = rigid(a, b) * T
        }
        return (T, fit)
    }

    static func sample(_ pts: [SIMD3<Float>], grid g: Float, max n: Int) -> [SIMD3<Float>] {
        // 4.6: never index millions of points to keep a few hundred thousand: thin the input first (memory)
        var seen = Set<SIMD3<Int32>>(), out: [SIMD3<Float>] = []
        let step = max(1, pts.count / max(1, n * 6))
        seen.reserveCapacity(min(pts.count / step, n * 6))
        for qi in stride(from: 0, to: pts.count, by: step) {
            let q = pts[qi]
            let k = SIMD3<Int32>(ik((q.x / g)), ik((q.y / g)), ik((q.z / g)))
            if seen.insert(k).inserted { out.append(q) }
        }
        if out.count > n { let s = out.count / n; out = stride(from: 0, to: out.count, by: s).map { out[$0] } }
        return out
    }

    /// aligns src to ref. Both are gravity-aligned (ARKit y up): try a turn every 10 degrees about y, then refine.
    static func align(_ src: Cloud, to ref: Cloud, progress: (Float) -> Void) -> (simd_float4x4, Float) {
        let r1 = sample(ref.p, grid: 0.025, max: 40000), s1 = sample(src.p, grid: 0.025, max: 6000)
        guard r1.count > 20, s1.count > 20 else { return (matrix_identity_float4x4, 0) }
        let g1 = GridIndex(r1, cell: 0.08)
        let rc = r1.reduce(SIMD3<Float>.zero, +) / Float(r1.count), sc = s1.reduce(SIMD3<Float>.zero, +) / Float(s1.count)
        var best = (matrix_identity_float4x4, Float(-1))
        for k in 0..<36 {
            let th = Float(k) * .pi / 18
            let R = simd_float3x3(simd_quatf(angle: th, axis: SIMD3(0, 1, 0)))
            var T0 = m44(R); T0.columns.3 = SIMD4(rc - R * sc, 1)
            let r = icp(s1, r1, g1, start: T0, maxDist: 0.08, iterations: 12)
            if r.1 > best.1 { best = r }
            progress(Float(k + 1) / 36 * 0.7)
        }
        var T = best.0, fit = best.1
        for (i, v) in [Float(0.008), 0.003].enumerated() {
            let r = sample(ref.p, grid: v, max: 400000), s = sample(src.p, grid: v, max: 30000)
            let g = GridIndex(r, cell: v * 3)
            (T, fit) = icp(s, r, g, start: T, maxDist: v * 3, iterations: 25)
            progress(0.7 + Float(i + 1) * 0.15)
        }
        return (T, fit)
    }

    /// adds src (already aligned) to base, skipping points where base already has surface (no doubles).
    /// 4.6: base is a set of dup-sized cells (8 bytes each), not a full point index - merging 3 big scans crashed on memory
    static func addNew(_ src: Cloud, to base: Cloud, dup: Float) -> (Cloud, Int) {
        @inline(__always) func key(_ q: SIMD3<Float>) -> Int64 { ik64((q.x / dup)) & 0x1FFFFF | ((ik64((q.y / dup)) & 0x1FFFFF) << 21) | ((ik64((q.z / dup)) & 0x1FFFFF) << 42) }
        var occ = Set<Int64>(); occ.reserveCapacity(base.count / 2)
        for q in base.p { occ.insert(key(q)) }
        var keep = [Bool](repeating: false, count: src.count)
        keep.withUnsafeMutableBufferPointer { kb in
            let kp = kb.baseAddress!
            parallelFor(src.count) { i in
                let q = src.p[i]; var hit = false
                search: for dx: Float in [-1, 0, 1] { for dy: Float in [-1, 0, 1] { for dz: Float in [-1, 0, 1] {
                    if occ.contains(key(q + SIMD3(dx, dy, dz) * dup)) { hit = true; break search }
                } } }
                kp[i] = !hit
            }
        }
        let add = src.keeping(keep)
        var o = base; o.p += add.p; o.c += add.c
        return (o, add.count)
    }
}

func m44(_ R: simd_float3x3) -> simd_float4x4 {
    simd_float4x4(columns: (SIMD4(R.columns.0, 0), SIMD4(R.columns.1, 0), SIMD4(R.columns.2, 0), SIMD4(0, 0, 0, 1)))
}
