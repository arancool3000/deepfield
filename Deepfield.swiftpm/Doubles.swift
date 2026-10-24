import Foundation
import simd

/// 5.0 Doubles, measured (no Gemini): a separate reference model is built from the photos' own LiDAR depth, with every
/// photo's drift corrected first (photo-to-photo matching + revisits, the same solver as Fix drift), so the reference
/// has ONE copy of each surface. The scan is then compared with it: a scan point that the reference does not have, in
/// space that corrected photos saw straight through, is a doubled copy and goes. Real thin parts are kept because the
/// reference has them too.
enum Doubles {
    static func fix(_ cl: Cloud, item: ScanItem, progress: @escaping (String) -> Void) -> (Cloud, String) {
        guard cl.count > 1000 else { return (cl, "Too few points.") }
        guard Library.captureURL(item) != nil else {
            let o = CloudOps.dedupe(cl, grid: Float(Prefs.mergeGridMM / 1000))
            return (o, "This scan has no photos, so there is no reference to compare with. Merged copies in the same cell: " + EditorModel.change(cl.count, o.count))
        }
        progress("Reading the photos' depth...")
        let kfs = Drift.load(item, maxViews: 70)
        guard kfs.count >= 4 else { return (cl, "Too few photos with depth to build a reference model.") }
        progress("Correcting the photos' drift...")
        let (X, edges, _) = Drift.solve(kfs)
        // corrected photos
        var views: [DepthView] = []
        for (i, k) in kfs.enumerated() {
            var kf = Keyframe(id: k.kf.id, jpg: k.kf.jpg, T: X[i] * k.kf.T, K: k.kf.K, w: k.kf.w, h: k.kf.h)
            kf.sharp = k.kf.sharp; kf.dw = k.dw; kf.dh = k.dh
            views.append(DepthView(kf: kf, depth: k.depth, dw: k.dw, dh: k.dh))
        }
        // the reference model: every corrected depth pixel, as 1.2 cm cells
        progress("Building the reference model from \(views.count) photos...")
        let cell: Float = 0.012
        @inline(__always) func key(_ q: SIMD3<Float>) -> Int64 { ik64(q.x / cell) & 0x1FFFFF | ((ik64(q.y / cell) & 0x1FFFFF) << 21) | ((ik64(q.z / cell) & 0x1FFFFF) << 42) }
        var ref = Set<Int64>(); ref.reserveCapacity(views.count * 20000)
        for v in views {
            let k = v.kf
            let fx = k.K[0][0] * Float(v.dw) / k.w, fy = k.K[1][1] * Float(v.dh) / k.h, cx = k.K[2][0] * Float(v.dw) / k.w, cy = k.K[2][1] * Float(v.dh) / k.h
            for y in stride(from: 0, to: v.dh, by: 2) { for x in stride(from: 0, to: v.dw, by: 2) {
                let d = v.depth[y * v.dw + x]
                guard d > 0.1 && d < 5 else { continue }
                ref.insert(key(CloudOps.apply(k.T, SIMD3((Float(x) - cx) * d / fx, -(Float(y) - cy) * d / fy, -d))))
            } }
        }
        // compare
        progress("Comparing your scan with the reference model...")
        let pick = AITools.pickViews(views, 32)
        var kill = [Bool](repeating: false, count: cl.count)
        kill.withUnsafeMutableBufferPointer { kb in
            let kp = kb.baseAddress!
            parallelFor(cl.count) { i in
                let p = cl.p[i], k0 = key(p)
                // the reference has surface here (or next to it): real
                for dx: Int64 in -1...1 { for dy: Int64 in -1...1 { for dz: Int64 in -1...1 {
                    if ref.contains(k0 &+ dx &+ (dy << 21) &+ (dz << 42)) { return }
                } } }
                var through = 0, agree = 0
                for v in pick {
                    guard let (u, vv, d) = v.kf.project(p) else { continue }
                    let md = v.measured(u, vv); guard md > 0 else { continue }
                    if d < md - 0.025 { through += 1 } else if abs(d - md) <= 0.025 { agree += 1 }
                }
                kp[i] = through >= 2 && through > agree
            }
        }
        // 5.7 (49): a point only goes if part of the scan STAYS within 6 cm of it - so removing doubles can thin out copies
        // but never leave a hole (a cupboard with 4 copies lost every copy in places, because none lined up with the
        // drift-corrected reference). Where every copy failed the test, all of them stay.
        let r: Float = 0.06, r2 = r * r
        @inline(__always) func gkey(_ q: SIMD3<Float>) -> SIMD3<Int32> { SIMD3(Int32((q.x / r).rounded(.down)), Int32((q.y / r).rounded(.down)), Int32((q.z / r).rounded(.down))) }
        var keptCells = [SIMD3<Int32>: [Int32]]()
        for i in 0..<cl.count where !kill[i] { keptCells[gkey(cl.p[i]), default: []].append(Int32(i)) }
        var spared = 0
        for i in 0..<cl.count where kill[i] {
            let p = cl.p[i], k = gkey(p)
            var covered = false
            search: for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 {
                for j in keptCells[k &+ SIMD3(dx, dy, dz)] ?? [] where simd_distance_squared(cl.p[Int(j)], p) <= r2 { covered = true; break search }
            } } }
            if !covered { kill[i] = false; spared += 1 }
        }
        var o = cl.keeping(kill.map { !$0 })
        let doubled = cl.count - o.count
        progress("Checking loose pieces...")
        let (fg, pieces) = CloudOps.floatingGhosts(o, views: pick)
        if pieces > 0 { o = o.keeping(fg.map { !$0 }) }
        let total = cl.count - o.count
        if total == 0 { return (cl, "No doubles found: every part of the scan matches the reference model built from \(views.count) photos.") }
        return (o, "Removed \(doubled.formatted()) doubled points\(pieces > 0 ? " and \(pieces) loose ghost piece\(pieces == 1 ? "" : "s")" : "") that the reference model from \(views.count) drift-corrected photos (\(edges) matches) does not have. \(o.count.formatted()) left\(spared > 0 ? "; \(spared.formatted()) kept where removing them would have left a hole" : ""). Undo if anything real went.")
    }
}
