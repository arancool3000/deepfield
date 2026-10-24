import Foundation
import UIKit
import simd

/// 5.0: the scan split into numbered parts that Gemini can name exactly ("remove part 7"), instead of drawing rough
/// boxes on photos. Big flat surfaces first (floor, walls, ceiling, table tops - RANSAC planes, split where they are
/// not connected), then every separate object (connected pieces of what is left). Each part is measured: size,
/// height above the floor, whether it floats, and how much of it the LiDAR saw straight THROUGH (a ghost or double).
struct Segments {
    struct Part {
        var id: Int
        var kind: String                 // floor | wall | ceiling | surface | object | specks
        var count = 0                    // points in the full scan
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var anchor = SIMD3<Float>.zero   // a real point of the part near its middle (where its number is drawn)
        var normal: SIMD3<Float>? = nil  // planes only
        var d: Float = 0
        var through: Float = 0           // share of its sampled points the photos saw through (0...1, -1 = unknown)
        var floating = false             // touches nothing else and is off the floor
        var size: SIMD3<Float> { hi - lo }
    }
    var parts: [Part] = []
    var cell: Float = 0.015
    var labelOf: [Int64: Int16] = [:]          // 1.5 cm cell -> part id
    var floorY: Float = 0

    @inline(__always) func key(_ q: SIMD3<Float>) -> Int64 { Segments.key(q, cell) }
    @inline(__always) static func key(_ q: SIMD3<Float>, _ c: Float) -> Int64 {
        ik64(q.x / c) & 0x1FFFFF | ((ik64(q.y / c) & 0x1FFFFF) << 21) | ((ik64(q.z / c) & 0x1FFFFF) << 42)
    }
    /// 5.7 (48): the key of a neighbouring cell. Adding dx + dy<<21 + dz<<42 to a packed key borrowed / carried between
    /// the fields, so cells either side of x, y or z = 0 were never neighbours (parts were cut at the z = 0 plane);
    /// each field now wraps in its own 21 bits, exactly as key() masks negative cells
    @inline(__always) static func neighbour(_ k: Int64, _ dx: Int64, _ dy: Int64, _ dz: Int64) -> Int64 {
        let m: Int64 = 0x1FFFFF
        return ((k & m) &+ dx) & m | ((((k >> 21) & m) &+ dy) & m) << 21 | ((((k >> 42) & m) &+ dz) & m) << 42
    }
    /// part of each point of a cloud (-1 = none)
    func labels(_ cl: Cloud) -> [Int16] {
        var out = [Int16](repeating: -1, count: cl.count)
        out.withUnsafeMutableBufferPointer { ob in
            let op = ob.baseAddress!
            parallelFor(cl.count) { i in
                let k = key(cl.p[i])
                if let l = labelOf[k] { op[i] = l; return }
                for dx: Int64 in -1...1 { for dy: Int64 in -1...1 { for dz: Int64 in -1...1 {
                    if let l = labelOf[Segments.neighbour(k, dx, dy, dz)] { op[i] = l; return }
                } } }
            }
        }
        return out
    }

    static func find(_ cl: Cloud, views: [DepthView]) -> Segments {
        var S = Segments()
        guard cl.count > 500 else { return S }
        let pts = CloudOps.sample(cl.p, grid: S.cell, max: 350_000)
        let n = pts.count
        var lab = [Int](repeating: -1, count: n)
        var rng = SystemRandomNumberGenerator()
        // ---- 1. planes ----
        var free = Array(0..<n)
        let ys = pts.map { $0.y }.sorted()
        let yLow = ys[ys.count * 2 / 100], yHigh = ys[ys.count * 98 / 100]
        var planes: [(SIMD3<Float>, Float, [Int])] = []
        for _ in 0..<10 {
            guard free.count > 2000 else { break }
            let probe = stride(from: 0, to: free.count, by: max(1, free.count / 20000)).map { free[$0] }
            var best = (SIMD3<Float>(0, 1, 0), Float(0), 0)
            for _ in 0..<350 {
                let a = pts[free[Int.random(in: 0..<free.count, using: &rng)]], b = pts[free[Int.random(in: 0..<free.count, using: &rng)]], c = pts[free[Int.random(in: 0..<free.count, using: &rng)]]
                var nr = simd_cross(b - a, c - a); let l = simd_length(nr); guard l > 1e-6 else { continue }
                nr /= l
                // only level or upright planes: rooms are built that way, and slanted fits are usually objects
                guard abs(nr.y) > 0.92 || abs(nr.y) < 0.2 else { continue }
                if nr.y < 0 || (abs(nr.y) < 0.2 && nr.x < 0) { nr = -nr }
                let d = -simd_dot(nr, a)
                var k = 0; for i in probe where abs(simd_dot(nr, pts[i]) + d) < 0.02 { k += 1 }
                if k > best.2 { best = (nr, d, k) }
            }
            guard best.2 > probe.count / 25 else { break }
            let (nr, d, _) = best
            let inl = free.filter { abs(simd_dot(nr, pts[$0]) + d) < 0.02 }
            guard inl.count > n / 40 else { break }
            planes.append((nr, d, inl))
            let used = Set(inl); free = free.filter { !used.contains($0) }
        }
        // each plane split into its connected pieces (two tables at one height are two parts)
        func components(_ idx: [Int], cell c: Float) -> [[Int]] {
            var cellOf: [Int64: [Int]] = [:]
            for i in idx { cellOf[key(pts[i], c), default: []].append(i) }
            var seen = Set<Int64>(), out: [[Int]] = []
            for (start, _) in cellOf where !seen.contains(start) {
                var stack = [start], members: [Int] = []; seen.insert(start)
                while let k = stack.popLast() {
                    members += cellOf[k] ?? []
                    for dx: Int64 in -1...1 { for dy: Int64 in -1...1 { for dz: Int64 in -1...1 {
                        let nb = Segments.neighbour(k, dx, dy, dz)
                        if cellOf[nb] != nil && !seen.contains(nb) { seen.insert(nb); stack.append(nb) }
                    } } }
                }
                out.append(members)
            }
            return out.sorted { $0.count > $1.count }
        }
        func key(_ q: SIMD3<Float>, _ c: Float) -> Int64 { Segments.key(q, c) }
        var parts: [Part] = []
        var leftovers: [Int] = free
        for (nr, d, inl) in planes {
            for comp in components(inl, cell: 0.05) {
                guard comp.count > max(300, n / 150) else { leftovers += comp; continue }
                let id = parts.count
                var p = Part(id: id, kind: "surface"); p.normal = nr; p.d = d
                for i in comp { lab[i] = id; p.lo = simd_min(p.lo, pts[i]); p.hi = simd_max(p.hi, pts[i]) }
                let y = -d / max(1e-4, nr.y)
                if abs(nr.y) < 0.2 { p.kind = "wall" }
                else if y < yLow + 0.15 { p.kind = "floor" }
                else if y > yHigh - 0.15 && p.size.x * p.size.z > 1.5 { p.kind = "ceiling" }
                parts.append(p)
            }
        }
        // ---- 2. objects: connected pieces of the rest ----
        let specksID = -2
        for comp in components(leftovers, cell: 0.03) {
            if comp.count < max(25, n / 2000) { for i in comp { lab[i] = specksID }; continue }
            let id = parts.count
            var p = Part(id: id, kind: "object")
            for i in comp { lab[i] = id; p.lo = simd_min(p.lo, pts[i]); p.hi = simd_max(p.hi, pts[i]) }
            parts.append(p)
        }
        // keep at most 40 numbered parts: the smallest objects join the specks
        if parts.count > 40 {
            let sizes = parts.map { p in lab.lazy.filter { $0 == p.id }.count }
            let order = parts.indices.sorted { sizes[$0] > sizes[$1] }
            let drop = Set(order.dropFirst(40).filter { parts[$0].kind == "object" })
            for i in 0..<n where lab[i] >= 0 && drop.contains(lab[i]) { lab[i] = specksID }
            var remap: [Int: Int] = [:], kept: [Part] = []
            for p in parts where !drop.contains(p.id) { remap[p.id] = kept.count; var q = p; q.id = kept.count; kept.append(q) }
            for i in 0..<n where lab[i] >= 0 { lab[i] = remap[lab[i]] ?? specksID }
            parts = kept
        }
        let specks = parts.count
        parts.append(Part(id: specks, kind: "specks"))
        for i in 0..<n where lab[i] == specksID { lab[i] = specks }
        // ---- 3. measure each part ----
        S.floorY = parts.first(where: { $0.kind == "floor" }).map { -$0.d / max(1e-4, $0.normal?.y ?? 1) } ?? yLow
        var members = [[Int]](repeating: [], count: parts.count)
        for i in 0..<n where lab[i] >= 0 { members[lab[i]].append(i) }
        let pick = AITools.pickViews(views, 24)
        for (pi, m) in members.enumerated() where !m.isEmpty {
            var p = parts[pi]
            p.lo = SIMD3(repeating: .greatestFiniteMagnitude); p.hi = -p.lo
            var c = SIMD3<Float>.zero
            for i in m { p.lo = simd_min(p.lo, pts[i]); p.hi = simd_max(p.hi, pts[i]); c += pts[i] }
            c /= Float(m.count)
            p.anchor = m.min { simd_length_squared(pts[$0] - c) < simd_length_squared(pts[$1] - c) }.map { pts[$0] } ?? c
            if !pick.isEmpty && p.kind != "specks" {
                var thr = 0, seen = 0
                for j in stride(from: 0, to: m.count, by: max(1, m.count / 150)) {
                    var through = 0, agree = 0
                    for v in pick {
                        guard let (u, vv, d) = v.kf.project(pts[m[j]]) else { continue }
                        let md = v.measured(u, vv); guard md > 0 else { continue }
                        if d < md - 0.03 { through += 1 } else if abs(d - md) <= 0.03 { agree += 1 }
                    }
                    if through + agree > 0 { seen += 1; if through >= 2 && through > agree { thr += 1 } }
                }
                p.through = seen > 0 ? Float(thr) / Float(seen) : -1
            } else { p.through = -1 }
            parts[pi] = p
        }
        // floating: an object off the floor whose box touches no other part's box
        for i in parts.indices where parts[i].kind == "object" {
            let a = parts[i]
            guard a.lo.y > S.floorY + 0.05 else { continue }
            let touches = parts.contains { b in
                b.id != a.id && b.kind != "specks" && a.lo.x <= b.hi.x + 0.03 && a.hi.x >= b.lo.x - 0.03 && a.lo.y <= b.hi.y + 0.03 && a.hi.y >= b.lo.y - 0.03 && a.lo.z <= b.hi.z + 0.03 && a.hi.z >= b.lo.z - 0.03
            }
            parts[i].floating = !touches
        }
        for i in 0..<n where lab[i] >= 0 { S.labelOf[key(pts[i], S.cell)] = Int16(lab[i]) }
        // real point counts in the full scan
        let full = S.labels(cl)
        var counts = [Int](repeating: 0, count: parts.count)
        for l in full where l >= 0 && Int(l) < counts.count { counts[Int(l)] += 1 }
        for i in parts.indices { parts[i].count = counts[i] }
        S.parts = parts
        return S
    }

    /// one line per part for Gemini
    func describe() -> String {
        parts.map { p in
            var t = "\(p.id): \(p.kind), \(p.count) points, \(String(format: "%.2fx%.2fx%.2f m", p.size.x, p.size.y, p.size.z)), from \(String(format: "%.2f", p.lo.y - floorY)) to \(String(format: "%.2f", p.hi.y - floorY)) m above the floor"
            if p.through >= 0 { t += String(format: ", LiDAR saw through %d%% of it", Int(p.through * 100)) }
            if p.floating { t += ", floating (touches nothing)" }
            return t
        }.joined(separator: "\n")
    }
    func name(_ id: Int) -> String {
        guard id >= 0, id < parts.count else { return "part \(id)" }
        let p = parts[id]
        return p.kind == "specks" ? "loose specks" : "\(p.kind) \(id)"
    }

    // ---- pictures for Gemini: the parts' numbers drawn where they are ----
    /// a camera looking at `target` from `pos` (no photo behind it): for scans without photos
    static func virtualCamera(_ pos: SIMD3<Float>, _ target: SIMD3<Float>, id: Int) -> Keyframe {
        let f = simd_normalize(target - pos)
        var r = simd_cross(f, SIMD3(0, 1, 0)); if simd_length(r) < 1e-3 { r = SIMD3(1, 0, 0) }; r = simd_normalize(r)
        let u = simd_cross(r, f)
        let T = simd_float4x4(columns: (SIMD4(r, 0), SIMD4(u, 0), SIMD4(-f, 0), SIMD4(pos, 1)))
        let w: Float = 1024, h: Float = 768
        let K = simd_float3x3(columns: (SIMD3(w * 0.75, 0, 0), SIMD3(0, w * 0.75, 0), SIMD3(w / 2, h / 2, 1)))
        var k = Keyframe(id: -id, jpg: URL(fileURLWithPath: "/dev/null"), T: T, K: K, w: w, h: h); k.sharp = 1
        return k
    }
    static func aroundViews(_ cl: Cloud) -> [Keyframe] {
        let b = cl.bounds, c = b.centre, r = max(0.6, simd_length(b.size) * 0.75)
        var out: [Keyframe] = []
        for (i, a) in [Float(0.5), 2.07, 3.64, 5.21].enumerated() {
            out.append(virtualCamera(c + SIMD3(cos(a) * r, b.size.y * 0.6 + 0.3, sin(a) * r), c, id: i + 1))
        }
        out.append(virtualCamera(c + SIMD3(0.01, r * 1.2 + b.size.y, 0), c, id: 9))      // from above
        return out
    }
    /// the scan drawn from a camera with a depth buffer (for visibility of the numbers)
    static func render(_ cl: Cloud, _ kf: Keyframe, width W: Int) -> (UIImage, [Float], Int, Int)? {
        let H = Int(Float(W) * kf.h / kf.w), sx = Float(W) / kf.w
        var z = [Float](repeating: .greatestFiniteMagnitude, count: W * H)
        var rgb = [UInt8](repeating: 18, count: W * H * 4)
        let step = max(1, cl.count / 900_000)
        var i = 0
        while i < cl.count {
            if let (u, v, d) = kf.project(cl.p[i]) {
                let x = Int(u * sx), y = Int(v * sx)
                for yy in y..<min(H, y + 2) { for xx in x..<min(W, x + 2) where d < z[yy * W + xx] {
                    z[yy * W + xx] = d; let o = (yy * W + xx) * 4
                    rgb[o] = cl.c[i].x; rgb[o + 1] = cl.c[i].y; rgb[o + 2] = cl.c[i].z; rgb[o + 3] = 255
                } }
            }
            i += step
        }
        guard let ctx = CGContext(data: &rgb, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let cg = ctx.makeImage() else { return nil }
        return (UIImage(cgImage: cg), z, W, H)
    }
    /// numbers on a picture taken by `kf`; visible(p) says whether that spot is really seen there
    /// 5.7 (48): drawn at scale 1 and at most `long` pixels on the long side. The default renderer drew at screen scale (3x),
    /// so a 1536 px photo became a 4608 x 3456 bitmap, two per view, four views at once
    func labelled(_ base: UIImage, _ kf: Keyframe, long: CGFloat = 640, visible: (SIMD3<Float>) -> Bool) -> (Data?, [Int]) {
        let f = min(1, long / max(1, max(base.size.width, base.size.height)))
        let size = CGSize(width: (base.size.width * f).rounded(), height: (base.size.height * f).rounded())
        let W = size.width, sx = Float(W) / kf.w
        var shown: [Int] = []
        let fmt = UIGraphicsImageRendererFormat.default(); fmt.scale = 1
        let img = UIGraphicsImageRenderer(size: size, format: fmt).image { _ in
            base.draw(in: CGRect(origin: .zero, size: size))
            for p in parts where p.kind != "specks" && p.count > 0 {
                guard let (u, v, _) = kf.project(p.anchor), visible(p.anchor) else { continue }
                shown.append(p.id)
                let x = CGFloat(u * sx), y = CGFloat(v * sx)
                let col: UIColor = p.through > 0.5 ? .systemRed : (p.floating ? .systemOrange : (p.kind == "object" ? .systemYellow : .systemTeal))
                let r = CGRect(x: x - 13, y: y - 13, width: 26, height: 26)
                col.setFill(); UIBezierPath(ovalIn: r).fill()
                let s = "\(p.id)" as NSString
                let at: [NSAttributedString.Key: Any] = [.font: UIFont.boldSystemFont(ofSize: 15), .foregroundColor: UIColor.black]
                let sz = s.size(withAttributes: at)
                s.draw(at: CGPoint(x: x - sz.width / 2, y: y - sz.height / 2), withAttributes: at)
            }
        }
        return (img.jpegData(compressionQuality: 0.75), shown)
    }

    /// up to 4 photo/render pairs (or, without photos, renders from around and above), numbered
    func pictures(_ cl: Cloud, views: [DepthView]) -> ([Data], String) {
        var out: [Data] = []
        if !views.isEmpty {
            // greedy: the photos that show the most numbered parts
            let cand = AITools.pickViews(views, 16)
            func vis(_ v: DepthView, _ p: SIMD3<Float>) -> Bool {
                guard let (u, vv, d) = v.kf.project(p) else { return false }
                let md = v.measured(u, vv); return md == 0 || abs(d - md) < 0.08
            }
            var covered = Set<Int>(), chosen: [DepthView] = []
            for _ in 0..<4 {
                let best = cand.max { a, b in
                    parts.filter { !covered.contains($0.id) && vis(a, $0.anchor) }.count < parts.filter { !covered.contains($0.id) && vis(b, $0.anchor) }.count
                }
                guard let v = best else { break }
                let new = parts.filter { !covered.contains($0.id) && vis(v, $0.anchor) }
                if new.isEmpty && !chosen.isEmpty { break }
                chosen.append(v); for p in new { covered.insert(p.id) }
            }
            var pairs = [[Data]](repeating: [], count: chosen.count)
            pairs.withUnsafeMutableBufferPointer { pb in
                let pp = pb.baseAddress!
                DispatchQueue.concurrentPerform(iterations: chosen.count) { i in
                    let v = chosen[i]
                    guard let photoData = v.kf.small(512), let photo = UIImage(data: photoData), let (scan, _, _, _) = Segments.render(cl, v.kf, width: min(640, Int(photo.size.width))) else { return }   // 5.7 (48): no wider than labelled() keeps
                    let a = labelled(photo, v.kf) { vis(v, $0) }.0, b = labelled(scan, v.kf) { vis(v, $0) }.0
                    if let a, let b { pp[i] = [a, b] }
                }
            }
            out = pairs.flatMap { $0 }
            return (out, "The images come in pairs: a PHOTO, then the SCAN drawn from the same camera, both with the same part numbers.")
        }
        for kf in Segments.aroundViews(cl) {
            guard let (img, z, W, _) = Segments.render(cl, kf, width: 640) else { continue }
            let sx = Float(W) / kf.w
            let (d, _) = labelled(img, kf) { p in
                guard let (u, v, dd) = kf.project(p) else { return false }
                let x = Int(u * sx), y = Int(v * sx); guard x >= 0, y >= 0, x < W, y * W + x < z.count else { return false }
                return dd < z[y * W + x] + 0.08
            }
            if let d { out.append(d) }
        }
        return (out, "This scan has no photos: the images are the scan itself drawn from around and from above, with the part numbers.")
    }
}
