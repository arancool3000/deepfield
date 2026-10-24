import Foundation
import simd

/// 5.5 Quick fill: instant, offline. A smooth surface is fitted (a plane per 2 cm cell, wider than the LiDAR noise, and
/// the planes of neighbouring cells blended by distance so there are no steps); the scan's points move most of the way
/// onto it (LiDAR fuzz halves) and new points are placed on it between them, spaced by the real local gap and coloured
/// from the nearest real points; then small gaps are filled. Corners and edges are left alone. ~4x the points.
/// 5.6 measured on a 4.26M-point desk scan: every point gets new neighbours (was 70%), 4.26x the points (was 2.6x).
extension CloudOps {
    static func quickFill(_ cl: Cloud, factor want: Float = 8, cap: Int = 24_000_000, flatMax: Float = 0.25, progress: (Float) -> Void = { _ in }) -> Cloud {
        guard cl.count > 100 else { return cl }
        let sp = max(0.0012, spacing(cl))
        let factor = max(1, min(want, Float(cap) / Float(cl.count)))
        let perPoint = Int((factor - 1).rounded())
        guard perPoint >= 1 else { return cl }
        let ref = cl.count > 5_000_000 ? thin(cl, grid: sp * 1.4) : cl
        let R: Float = max(sp * 6, 0.02)                     // the surface is fitted over this size
        // 1. one plane per R-cell, from the points in it and around it
        @inline(__always) func key(_ q: SIMD3<Float>, _ c: Float) -> Int64 { ik64(q.x / c) & 0x1FFFFF | ((ik64(q.y / c) & 0x1FFFFF) << 21) | ((ik64(q.z / c) & 0x1FFFFF) << 42) }
        var members: [Int64: [Int32]] = [:]
        for i in stride(from: 0, to: ref.count, by: max(1, ref.count / 1_500_000)) { members[key(ref.p[i], R), default: []].append(Int32(i)) }
        let keys = Array(members.keys)
        var planes = [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)?](repeating: nil, count: keys.count)      // normal, centre, e1, e2
        planes.withUnsafeMutableBufferPointer { pb in
            let pp = pb.baseAddress!
            parallelFor(keys.count) { ci in
                let k = keys[ci]
                var pts: [SIMD3<Float>] = []; pts.reserveCapacity(200)
                for dx: Int64 in -1...1 { for dy: Int64 in -1...1 { for dz: Int64 in -1...1 {
                    if let m = members[k &+ dx &+ (dy << 21) &+ (dz << 42)] { for j in stride(from: 0, to: m.count, by: max(1, m.count / 24)) { pts.append(ref.p[Int(m[j])]) } }
                } } }
                guard pts.count >= 8, let (n, c, flat) = planeNormal(pts), flat < flatMax else { return }
                let e1 = simd_normalize(simd_cross(n, abs(n.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)))
                pp[ci] = (n, c, e1, simd_cross(n, e1))
            }
        }
        var planeOf: [Int64: Int32] = [:]; planeOf.reserveCapacity(keys.count)
        for (ci, k) in keys.enumerated() where planes[ci] != nil { planeOf[k] = Int32(ci) }
        progress(0.35)
        /// where a point lies on the smooth surface: its projections onto the planes of the cells around it, blended by
        /// distance (one plane per cell gave visible steps between cells)
        @inline(__always) func onSurface(_ q: SIMD3<Float>) -> SIMD3<Float>? {
            let k = key(q, R)
            var sum = SIMD3<Float>.zero, w: Float = 0
            let r2 = R * R
            for dx: Int64 in -1...1 { for dy: Int64 in -1...1 { for dz: Int64 in -1...1 {
                guard let pi = planeOf[k &+ dx &+ (dy << 21) &+ (dz << 42)], let (nrm, ctr, _, _) = planes[Int(pi)] else { continue }
                let d2 = simd_length_squared(q - ctr)
                let ww = exp(-d2 / r2)
                sum += (q - nrm * simd_dot(q - ctr, nrm)) * ww; w += ww
            } } }
            return w > 1e-4 ? sum / w : nil
        }
        // 2. a fine grid for the small questions: the real gap here, and the colour of a new spot
        let fineCell = max(sp * 2, 0.004)
        let fine = GridIndex(ref.p, cell: fineCell)
        let offsets: [SIMD2<Float>] = [SIMD2(0.5, 0), SIMD2(0, 0.5), SIMD2(0.5, 0.5), SIMD2(-0.5, 0.5), SIMD2(0.25, -0.5), SIMD2(-0.25, -0.25)]
        let n = cl.count
        var newP = [SIMD3<Float>](repeating: .zero, count: n * perPoint), newC = [SIMD3<UInt8>](repeating: .zero, count: n * perPoint)
        var ok = [Bool](repeating: false, count: n * perPoint)
        newP.withUnsafeMutableBufferPointer { pb in newC.withUnsafeMutableBufferPointer { cb in ok.withUnsafeMutableBufferPointer { ob in
            let pp = pb.baseAddress!, cp = cb.baseAddress!, op = ob.baseAddress!
            parallelFor(n) { i in
                let q = cl.p[i]
                // the real gap between points here (points within one fine cell's radius, over that disc's area)
                var count = 0
                let rr = fineCell * fineCell
                var near: [SIMD3<Float>] = []; near.reserveCapacity(32)
                fine.around(q) { j in let p = ref.p[Int(j)]; if simd_length_squared(p - q) <= rr { count += 1; if near.count < 48 { near.append(p) } } }
                let ls = fineCell * (Float.pi / Float(max(1, count))).squareRoot()
                // 5.6: EVERY point is filled. Smooth surfaces use the blended big-surface fit; edges, corners, curves and
                // fine detail (no big flat surface: about two thirds of a real scan, left empty before) use their own
                // small neighbourhood's tangent, with new points kept closer (the surface bends there)
                let e1: SIMD3<Float>, e2: SIMD3<Float>, onPlane: SIMD3<Float>, big: Bool
                if let pi = planeOf[key(q, R)], let (pn, pc, pe1, pe2) = planes[Int(pi)] {
                    e1 = pe1; e2 = pe2; big = true
                    onPlane = onSurface(q) ?? (q - pn * simd_dot(q - pc, pn))
                } else if near.count >= 5, let (ln, lc, _) = planeNormal(near) {
                    big = false
                    e1 = simd_normalize(simd_cross(ln, abs(ln.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0))); e2 = simd_cross(ln, e1)
                    onPlane = q - ln * (simd_dot(q - lc, ln) * 0.5)
                } else { return }
                for k in 0..<perPoint {
                    let o = offsets[k % offsets.count]
                    let raw = onPlane + (e1 * o.x + e2 * o.y) * (big ? ls : ls * 0.7)
                    let np = big ? (onSurface(raw) ?? raw) : raw
                    var s = SIMD3<Float>.zero, w: Float = 0
                    let lim = ls * ls * 2.5
                    fine.around(np) { j in
                        let d2 = simd_length_squared(ref.p[Int(j)] - np)
                        guard d2 < lim else { return }
                        let ww = 1 / (d2 + ls * ls * 0.05), c = ref.c[Int(j)]
                        s += SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)) * ww; w += ww
                    }
                    guard w > 0 else { continue }          // no real point near the new spot: it would be inventing, so no
                    let c = s / w
                    pp[i * perPoint + k] = np; cp[i * perPoint + k] = SIMD3(u8(c.x), u8(c.y), u8(c.z)); op[i * perPoint + k] = true
                }
            }
        } } }
        progress(0.85)
        // smooth: every original point on a fitted surface moves 80% of the way onto it (LiDAR fuzz goes, edges stay -
        // corners and edges had no fitted surface)
        var base = cl
        base.p.withUnsafeMutableBufferPointer { bp in
            let pp = bp.baseAddress!
            parallelFor(n) { i in
                let q = cl.p[i]
                guard planeOf[key(q, R)] != nil, let target = onSurface(q) else { return }
                pp[i] = q + (target - q) * 0.8
            }
        }
        var add = Cloud(p: newP, c: newC).keeping(ok)
        newP = []; newC = []; ok = []
        add = thin(add, grid: max(0.0007, sp * 0.42))              // new points almost on top of each other merge
        // 5.6 coverage: any point still without new points beside it (a lone point, a thin wire, a spot whose new points
        // merged away) gets them now, along its own neighbourhood's tangent - every point in the scan is filled
        progress(0.88)
        let addIdx = GridIndex(add.p, cell: sp * 1.5)
        let lim2 = sp * sp * 2.25
        var extraP = [[SIMD3<Float>]](repeating: [], count: 64), extraC = [[SIMD3<UInt8>]](repeating: [], count: 64)
        extraP.withUnsafeMutableBufferPointer { epb in extraC.withUnsafeMutableBufferPointer { ecb in
            let ep = epb.baseAddress!, ec = ecb.baseAddress!
            DispatchQueue.concurrentPerform(iterations: 64) { chunk in
                let per = (n + 63) / 64, lo = chunk * per, hi = min(n, lo + per)
                guard lo < hi else { return }                          // 5.7 (48): the last chunks of a small cloud are empty (lo > hi trapped)
                for i in lo..<hi {
                    let q = base.p[i]
                    var have = 0
                    addIdx.around(q) { j in if have < 2 && simd_length_squared(add.p[Int(j)] - q) < lim2 { have += 1 } }
                    if have >= 2 { continue }
                    var near: [SIMD3<Float>] = []
                    fine.around(q) { j in if near.count < 24 { let p = ref.p[Int(j)]; if simd_length_squared(p - q) < fineCell * fineCell { near.append(p) } } }
                    let t1: SIMD3<Float>, t2: SIMD3<Float>
                    if near.count >= 4, let (ln, _, _) = planeNormal(near) {
                        t1 = simd_normalize(simd_cross(ln, abs(ln.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0))); t2 = simd_cross(ln, t1)
                    } else { t1 = SIMD3(1, 0, 0); t2 = SIMD3(0, 0, 1) }
                    let st = sp * 0.5
                    for o in [t1, -t1, t2, -t2] { ep[chunk].append(q + o * st); ec[chunk].append(cl.c[i]) }
                }
            }
        } }
        for k in 0..<64 { add.p += extraP[k]; add.c += extraC[k] }
        var out = base; out.p += add.p; out.c += add.c
        // gaps: surfaces continued across the small holes left, coloured from around them
        progress(0.92)
        let (_, gaps) = HoleFill.fill(out, cell: max(0.008, sp * 5), passes: 5, radius: 3, inside: nil, spacing: sp * 0.5)
        out.p += gaps.p; out.c += gaps.c
        progress(1)
        return out
    }
}
