import Foundation
import UIKit
import simd

/// AI gap filling for rooms: the walls, floor and ceiling are found as planes; holes inside each plane become candidate
/// gaps; Gemini looks at the gaps marked on the photos and decides which are surfaces the scanner missed (fill) and which
/// are real openings (doors, windows, doorways: leave open). Filled points are shown yellow until the user keeps them
/// (they then take the colour of the real surface around them) or removes them.
struct Gap {
    let id: Int
    let cells: [SIMD2<Int32>]
    let plane: Int
    var centre: SIMD3<Float>
    var area: Float                       // m²
}
struct RoomPlane {
    let n: SIMD3<Float>, d: Float, e1: SIMD3<Float>, e2: SIMD3<Float>
    var occ: [SIMD2<Int32>: (Int, SIMD3<Float>)] = [:]     // cell -> (points, colour sum)
    var clutter: Set<SIMD2<Int32>> = []                     // cells with things on BOTH sides of the plane: it passes through clutter there
    func world(_ c: SIMD2<Int32>, _ du: Float = 0.5, _ dv: Float = 0.5) -> SIMD3<Float> {
        let u = (Float(c.x) + du) * RoomFill.cell, v = (Float(c.y) + dv) * RoomFill.cell
        return e1 * u + e2 * v - n * d
    }
}

enum RoomFill {
    static let cell: Float = 0.03

    /// up to 8 big planes, walls (vertical) and floors / ceilings (horizontal) only
    static func planes(_ cl: Cloud) -> [RoomPlane] {
        var pts = CloudOps.sample(cl.p, grid: 0.04, max: 120_000)
        var out: [RoomPlane] = []
        let total = pts.count
        for _ in 0..<8 where pts.count > 200 {
            var bestN = 0, best: (SIMD3<Float>, Float)? = nil
            for _ in 0..<350 {
                let a = pts.randomElement()!, b = pts.randomElement()!, c = pts.randomElement()!
                var n = simd_cross(b - a, c - a); let l = simd_length(n); guard l > 1e-6 else { continue }
                n /= l
                guard abs(n.y) > 0.985 || abs(n.y) < 0.1 else { continue }        // level floors/ceilings (10°), upright walls (6°): ARKit knows gravity
                let d = -simd_dot(n, a)
                var k = 0; for q in pts where abs(simd_dot(n, q) + d) < 0.025 { k += 1 }
                if k > bestN { bestN = k; best = (n, d) }
            }
            guard let (n, d) = best, bestN > total / 25 else { break }
            pts.removeAll { abs(simd_dot(n, $0) + d) < 0.025 }
            let ref = abs(n.y) > 0.9 ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(0, 1, 0)
            let e1 = simd_normalize(simd_cross(n, ref)), e2 = simd_cross(n, e1)
            out.append(RoomPlane(n: n, d: d, e1: e1, e2: e2))
        }
        // occupancy of every plane from the FULL cloud (points within 2.5 cm of it)
        for i in out.indices {
            let pl = out[i]
            var occ: [SIMD2<Int32>: (Int, SIMD3<Float>)] = [:], mom: [SIMD2<Int32>: SIMD2<Float>] = [:], side: [SIMD2<Int32>: UInt8] = [:]
            let step = max(1, cl.count / 1_500_000)          // 4.5: a sample is plenty for 3 cm occupancy
            for j in stride(from: 0, to: cl.count, by: step) {
                let q = cl.p[j], sd = simd_dot(pl.n, q) + pl.d, a = abs(sd)
                guard a < 0.15 else { continue }
                let c = SIMD2<Int32>(ik((simd_dot(q, pl.e1) / cell)), ik((simd_dot(q, pl.e2) / cell)))
                if a > 0.03 { side[c, default: 0] |= sd > 0 ? 1 : 2; continue }
                guard a < 0.025 else { continue }
                let col = SIMD3<Float>(Float(cl.c[j].x), Float(cl.c[j].y), Float(cl.c[j].z))
                let o = occ[c] ?? (0, .zero); occ[c] = (o.0 + 1, o.1 + col)
                mom[c, default: .zero] += SIMD2(sd, sd * sd)
            }
            // 5.5: a cell of the real surface is FLAT (a few mm of spread); cables and clutter that merely cross the plane
            // fill its 5 cm band evenly (about 14 mm of spread) and are not surface
            out[i].occ = occ.filter { kv in
                guard kv.value.0 >= 2, let m = mom[kv.key] else { return false }
                let n = Float(kv.value.0), mean = m.x / n, sdv = (max(0, m.y / n - mean * mean)).squareRoot()
                return sdv < 0.008
            }
            // things 3-15 cm in front AND behind the plane (any of the 3x3 cells around): the plane runs through clutter
            for (c, b) in side where b != 0 {
                var all: UInt8 = 0
                for dy in -1...1 { for dx in -1...1 { all |= side[SIMD2(c.x + Int32(dx), c.y + Int32(dy))] ?? 0 } }
                if all == 3 { out[i].clutter.insert(c) }
            }
        }
        return out
    }

    /// 5.5: the holes of every plane by a morphological CLOSING (grow the surface 24 cm, shrink it back): every hole and
    /// bay up to that size, also at the plane's edges (behind furniture, where wall meets floor). A cell the LiDAR saw
    /// straight THROUGH from two photos is an opening (doorway, window) and stays open.
    static func gaps(_ planes: [RoomPlane], views: [DepthView] = []) -> [Gap] {
        var gaps: [Gap] = []
        let rc: Int32 = 8                                                  // 8 cells of 3 cm
        for (pi, pl) in planes.enumerated() where pl.occ.count > 50 {
            var lo = SIMD2<Int32>(Int32.max, Int32.max), hi = SIMD2<Int32>(Int32.min, Int32.min)
            for c in pl.occ.keys { lo = simd_min(lo, c); hi = simd_max(hi, c) }
            lo &-= rc; hi &+= rc
            let W = Int(hi.x - lo.x + 1), H = Int(hi.y - lo.y + 1)
            guard W > 2, H > 2, W * H < 4_000_000 else { continue }
            var occ = [Bool](repeating: false, count: W * H)
            for c in pl.occ.keys { occ[Int(c.y - lo.y) * W + Int(c.x - lo.x)] = true }
            // distance (in cells, chessboard) to the nearest occupied cell, then to the nearest cell beyond the grown area
            func distance(_ seed: [Bool]) -> [Int32] {
                var d = [Int32](repeating: Int32.max / 2, count: W * H)
                var queue: [Int] = []; queue.reserveCapacity(W * H)
                for i in 0..<(W * H) where seed[i] { d[i] = 0; queue.append(i) }
                var h = 0
                while h < queue.count {
                    let i = queue[h]; h += 1
                    let x = i % W, y = i / W, nd = d[i] + 1
                    for dy in -1...1 { for dx in -1...1 where dx != 0 || dy != 0 {
                        let xx = x + dx, yy = y + dy; guard xx >= 0, yy >= 0, xx < W, yy < H else { continue }
                        let j = yy * W + xx; if d[j] > nd { d[j] = nd; queue.append(j) }
                    } }
                }
                return d
            }
            let dOcc = distance(occ)
            let grown = dOcc.map { $0 <= rc }
            let dOut = distance(grown.map { !$0 })
            var hole = [Bool](repeating: false, count: W * H)
            for i in 0..<(W * H) where !occ[i] && grown[i] && dOut[i] > rc {
                hole[i] = !pl.clutter.contains(SIMD2(Int32(i % W) + lo.x, Int32(i / W) + lo.y))
            }
            // openings: spots the cameras saw through
            if !views.isEmpty {
                for i in 0..<(W * H) where hole[i] {
                    let c = SIMD2<Int32>(Int32(i % W) + lo.x, Int32(i / W) + lo.y)
                    if SmartFill.seenThrough(pl.world(c), views) >= 2 { hole[i] = false }
                }
            }
            // connected pieces
            var seen = [Bool](repeating: false, count: W * H)
            for s0 in 0..<(W * H) where hole[s0] && !seen[s0] {
                var stack = [s0], cells: [SIMD2<Int32>] = []; seen[s0] = true
                var rim = 0, rimReal = 0
                while let i = stack.popLast() {
                    cells.append(SIMD2(Int32(i % W) + lo.x, Int32(i / W) + lo.y))
                    let x = i % W, y = i / W
                    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let xx = x + dx, yy = y + dy; guard xx >= 0, yy >= 0, xx < W, yy < H else { rim += 1; continue }
                        let j = yy * W + xx
                        if hole[j] { if !seen[j] { seen[j] = true; stack.append(j) } } else { rim += 1; if occ[j] { rimReal += 1 } }
                    }
                }
                let area = Float(cells.count) * cell * cell
                // 5.5: a gap is filled only where REAL surface surrounds it (3/4 of its rim). Scattered points with nothing
                // scanned between them (clutter, a floor glimpsed past the desk) are not a surface with a hole in it
                guard area <= 6, rimReal * 4 >= rim * 3 else { continue }
                var m = SIMD3<Float>.zero; for c in cells { m += pl.world(c) }
                gaps.append(Gap(id: gaps.count, cells: cells, plane: pi, centre: m / Float(cells.count), area: area))
            }
        }
        return gaps
    }

    /// photos with the bigger gaps outlined and numbered, for Gemini
    static func annotated(_ gaps: [Gap], _ planes: [RoomPlane], _ frames: [Keyframe], maxPhotos: Int = 6) -> ([Data], [Int: Int]) {
        var byFrame: [Int: [Int]] = [:]
        for g in gaps where g.area >= 0.03 {
            var best = -1, bd: Float = .greatestFiniteMagnitude
            for (fi, f) in frames.enumerated() {
                guard let (u, v, d) = f.project(g.centre) else { continue }
                let off = abs(u / f.w - 0.5) + abs(v / f.h - 0.5)
                let score = d + off * 2
                if score < bd { bd = score; best = fi }
            }
            if best >= 0 { byFrame[best, default: []].append(g.id) }
        }
        var imgs: [Data] = [], photoOf: [Int: Int] = [:]
        for (fi, ids) in byFrame.sorted(by: { $0.value.count > $1.value.count }).prefix(maxPhotos) {
            let f = frames[fi]
            guard let photo = UIImage(contentsOfFile: f.jpg.path) else { continue }
            let W: CGFloat = 640, s = W / CGFloat(f.w), H = CGFloat(f.h) * s
            let img = UIGraphicsImageRenderer(size: CGSize(width: W, height: H)).image { ctx in
                photo.draw(in: CGRect(x: 0, y: 0, width: W, height: H))
                for id in ids {
                    let g = gaps[id], pl = planes[g.plane]
                    var lo = CGPoint(x: 1e9, y: 1e9), hi = CGPoint(x: -1e9, y: -1e9)
                    for c in g.cells where c.x % 2 == 0 {
                        guard let (u, v, _) = f.project(pl.world(c)) else { continue }
                        lo.x = min(lo.x, CGFloat(u) * s); lo.y = min(lo.y, CGFloat(v) * s); hi.x = max(hi.x, CGFloat(u) * s); hi.y = max(hi.y, CGFloat(v) * s)
                    }
                    guard hi.x > lo.x else { continue }
                    let r = CGRect(x: lo.x, y: lo.y, width: hi.x - lo.x, height: hi.y - lo.y)
                    ctx.cgContext.setStrokeColor(UIColor.yellow.cgColor); ctx.cgContext.setLineWidth(3); ctx.cgContext.stroke(r)
                    ("\(id)" as NSString).draw(at: CGPoint(x: r.minX + 4, y: r.minY + 2), withAttributes: [.font: UIFont.boldSystemFont(ofSize: 22), .foregroundColor: UIColor.yellow, .backgroundColor: UIColor.black.withAlphaComponent(0.6)])
                }
            }
            if let d = img.jpegData(compressionQuality: 0.7) { imgs.append(d); for id in ids { photoOf[id] = imgs.count - 1 } }
        }
        return (imgs, photoOf)
    }

    /// 5.5: points for the chosen gaps at the scan's own spacing (a 1 m2 gap is ~150k points), coloured from the plane's
    /// real texture: around EACH gap the plane is laid flat as a sharp image (4 mm pixels, 25 cm margin) and the hole is
    /// grown from matching patches of the real surface there (floorboards and grain carry on), sampled smoothly; yellow
    /// copies for display until kept.
    static func points(_ gaps: [Gap], _ planes: [RoomPlane], cloud cl: Cloud, spacing: Float = 0.003) -> (Cloud, Cloud) {
        var yellow = Cloud(), real = Cloud()
        let px: Float = 0.004, margin: Float = 0.25
        for g in gaps {
            let pl = planes[g.plane]
            var lo = SIMD2<Int32>(Int32.max, Int32.max), hi = SIMD2<Int32>(Int32.min, Int32.min)
            for c in g.cells { lo = simd_min(lo, c); hi = simd_max(hi, c) }
            let u0 = Float(lo.x) * cell - margin, v0 = Float(lo.y) * cell - margin
            let W = min(900, Int((Float(hi.x - lo.x + 1) * cell + 2 * margin) / px) + 1), H = min(900, Int((Float(hi.y - lo.y + 1) * cell + 2 * margin) / px) + 1)
            // the real surface around the gap
            var img = [SIMD3<Float>](repeating: .zero, count: W * H), cnt = [Float](repeating: 0, count: W * H)
            let uMax = u0 + Float(W) * px, vMax = v0 + Float(H) * px
            let step = max(1, cl.count / 4_000_000)
            for j in stride(from: 0, to: cl.count, by: step) where abs(simd_dot(pl.n, cl.p[j]) + pl.d) < 0.02 {
                let q = cl.p[j], u = simd_dot(q, pl.e1), v = simd_dot(q, pl.e2)
                guard u >= u0, v >= v0, u < uMax, v < vMax else { continue }
                let x = Int((u - u0) / px), y = Int((v - v0) / px)
                img[y * W + x] += SIMD3(Float(cl.c[j].x), Float(cl.c[j].y), Float(cl.c[j].z)); cnt[y * W + x] += 1
            }
            var known = [Bool](repeating: false, count: W * H), hole = [Bool](repeating: false, count: W * H)
            for i in 0..<(W * H) where cnt[i] > 0 { img[i] /= cnt[i]; known[i] = true }
            // the hole: the gap's cells, plus the unknown pixels in between known ones the gap touches
            let per = Int((cell / px).rounded(.up)) + 1          // cells overlap by a pixel: no unmarked strips between them (they drew black lines)
            for c in g.cells {
                let x0 = Int((Float(c.x) * cell - u0) / px), y0 = Int((Float(c.y) * cell - v0) / px)
                for y in max(0, y0)..<min(H, y0 + per) { for x in max(0, x0)..<min(W, x0 + per) where !known[y * W + x] { hole[y * W + x] = true } }
            }
            guard known.contains(true) else { continue }
            let syn = GenFill.synthesize(target: img, known: known, hole: hole, W: W, H: H, src: img, srcKnown: known, SW: W, SH: H)
            // 5.5: copied patches alone made a patchwork of the wrong things (a cable's colour on the floor). The colour is
            // a smooth membrane from the real surface at the gap's EDGE (no seam), and only the fine grain (under ~3 cm)
            // comes from the grown texture, limited so it can never paint an object
            let base = RoomFill.pushPull(img, known, W, H), fine = RoomFill.boxBlur(syn, W, H, 4)
            var tex = img
            for i in 0..<(W * H) where !known[i] {
                let d = simd_clamp(syn[i] - fine[i], SIMD3(repeating: -18), SIMD3(repeating: 18))
                tex[i] = simd_clamp(base[i] + d, .zero, SIMD3(repeating: 255))
            }
            @inline(__always) func colour(_ u: Float, _ v: Float) -> SIMD3<Float> {
                let fx = min(Float(W - 1), max(0, (u - u0) / px - 0.5)), fy = min(Float(H - 1), max(0, (v - v0) / px - 0.5))
                let x0 = Int(fx), y0 = Int(fy), x1 = min(W - 1, x0 + 1), y1 = min(H - 1, y0 + 1), tx = fx - Float(x0), ty = fy - Float(y0)
                return RoomFill.bilerp(tex[y0 * W + x0], tex[y0 * W + x1], tex[y1 * W + x0], tex[y1 * W + x1], tx, ty)
            }
            let steps = max(1, Int(cell / spacing))
            for c in g.cells {
                for a in 0..<steps { for b in 0..<steps {
                    // the whole cell, edge to edge (jitter only in the middle left lines between cells that read as a grid)
                    let du = (Float(a) + Float.random(in: 0..<1)) / Float(steps), dv = (Float(b) + Float.random(in: 0..<1)) / Float(steps)
                    let q = pl.world(c, du, dv)
                    let k = colour(simd_dot(q, pl.e1), simd_dot(q, pl.e2))
                    yellow.p.append(q); yellow.c.append(SIMD3(255, 214, 10))
                    real.p.append(q); real.c.append(SIMD3(u8(k.x), u8(k.y), u8(k.z)))
                } }
            }
        }
        return (yellow, real)
    }

    /// smooth fill of the unknown pixels from the known ones (push-pull: averaged down a pyramid, filled back up)
    static func pushPull(_ img: [SIMD3<Float>], _ known: [Bool], _ W: Int, _ H: Int) -> [SIMD3<Float>] {
        var out = img
        for i in 0..<(W * H) where !known[i] { out[i] = .zero }
        var w = known.map { $0 ? Float(1) : 0 }
        if W <= 1 && H <= 1 { return out }
        let w2 = (W + 1) / 2, h2 = (H + 1) / 2
        var c = [SIMD3<Float>](repeating: .zero, count: w2 * h2), cw = [Float](repeating: 0, count: w2 * h2)
        for y in 0..<H { for x in 0..<W { let i = y * W + x, j = (y / 2) * w2 + x / 2; c[j] += out[i] * w[i]; cw[j] += w[i] } }
        var ck = [Bool](repeating: false, count: w2 * h2)
        for j in 0..<(w2 * h2) where cw[j] > 0 { c[j] /= cw[j]; ck[j] = true }
        if !ck.contains(true) { return out }
        let up = (w2 == W && h2 == H) ? c : pushPull(c, ck, w2, h2)
        for y in 0..<H { for x in 0..<W where !known[y * W + x] {
            // bilinear from the coarser level
            let fx = min(Float(w2 - 1), max(0, (Float(x) - 0.5) / 2)), fy = min(Float(h2 - 1), max(0, (Float(y) - 0.5) / 2))
            let x0 = Int(fx), y0 = Int(fy), x1 = min(w2 - 1, x0 + 1), y1 = min(h2 - 1, y0 + 1), tx = fx - Float(x0), ty = fy - Float(y0)
            out[y * W + x] = RoomFill.bilerp(up[y0 * w2 + x0], up[y0 * w2 + x1], up[y1 * w2 + x0], up[y1 * w2 + x1], tx, ty)
        } }
        w = []
        return out
    }

    @inline(__always) static func bilerp(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, _ d: SIMD3<Float>, _ tx: Float, _ ty: Float) -> SIMD3<Float> {
        let top: SIMD3<Float> = a + (b - a) * tx, bot: SIMD3<Float> = c + (d - c) * tx
        return top + (bot - top) * ty
    }

    static func boxBlur(_ img: [SIMD3<Float>], _ W: Int, _ H: Int, _ r: Int) -> [SIMD3<Float>] {
        var a = img, b = img
        for y in 0..<H { var s = SIMD3<Float>.zero, n: Float = 0
            for x in 0..<min(W, r) { s += a[y * W + x]; n += 1 }
            for x in 0..<W {
                if x + r < W { s += a[y * W + x + r]; n += 1 }
                if x - r - 1 >= 0 { s -= a[y * W + x - r - 1]; n -= 1 }
                b[y * W + x] = s / n
            } }
        for x in 0..<W { var s = SIMD3<Float>.zero, n: Float = 0
            for y in 0..<min(H, r) { s += b[y * W + x]; n += 1 }
            for y in 0..<H {
                if y + r < H { s += b[(y + r) * W + x]; n += 1 }
                if y - r - 1 >= 0 { s -= b[(y - r - 1) * W + x]; n -= 1 }
                a[y * W + x] = s / n
            } }
        return a
    }

    /// the whole thing: planes, gaps, Gemini's decision, points. Without Gemini, gaps under 0.6 m² are filled.
    static func run(_ cl: Cloud, item: ScanItem, deep: Bool = false) async -> (Cloud, Cloud, String) {
        let views = await Task.detached { DepthView.load(item, max: 40) }.value
        let found: ([RoomPlane], [Gap]) = await Task.detached { () -> ([RoomPlane], [Gap]) in
            let p = RoomFill.planes(cl); return (p, RoomFill.gaps(p, views: views))
        }.value
        let planes = found.0, gaps = found.1
        guard !gaps.isEmpty else { return (Cloud(), Cloud(), "No missing parts found in the walls, floor or ceiling.") }
        let frames = Library.captureURL(item).map { ModelBuilder.pick(Keyframe.all(in: $0), 120) } ?? []
        let ann: ([Data], [Int: Int]) = await Task.detached { RoomFill.annotated(gaps, planes, frames) }.value
        let imgs = ann.0, photoOf = ann.1
        var fill = Set(gaps.filter { $0.area < 0.03 }.map { $0.id })          // tiny holes: always filled
        var how = ""
        let asked = gaps.filter { photoOf[$0.id] != nil }
        if !imgs.isEmpty {
            // 5.7 (48): n·q + d = 0, so the plane sits at y = -d/n.y and n.y*d = -y: positive below the start (a floor)
            let list = asked.map { "gap \($0.id) (photo \(photoOf[$0.id]!), \(String(format: "%.2f", $0.area)) m², on a \(abs(planes[$0.plane].n.y) > 0.9 ? (planes[$0.plane].n.y * planes[$0.plane].d > 0 ? "floor" : "ceiling") : "wall"))" }.joined(separator: "; ")
            let prompt = """
            These photos come from a 3D room scan. Yellow numbered boxes mark places where the scan of a wall, floor or ceiling \
            has a hole. For each gap decide: "fill" if it is a surface the scanner simply missed (hidden behind furniture, \
            not looked at, a shadow, a dark or shiny patch of the same wall/floor/ceiling), or "open" if the real room really has \
            an opening or a different thing there (a door, doorway, window, open shelf, a hole, a staircase). Gaps: \(list). \
            Reply JSON only: {"fill": [gap numbers], "open": [gap numbers], "note": one short sentence for the user}.
            """
            if let t = try? await Gemini.ask(prompt, images: imgs, json: true, deep: deep), let j = try? JSONSerialization.jsonObject(with: Data(t.utf8)) as? [String: Any] {
                for n in (j["fill"] as? [Any] ?? []) { if let i = (n as? Int) ?? Int("\(n)") { fill.insert(i) } }
                how = (j["note"] as? String).map { " Gemini: " + $0 } ?? ""
            } else {
                for g in asked where g.area < 0.6 { fill.insert(g.id) }
                how = " (Gemini unavailable: filled the small gaps only)"
            }
        } else { for g in gaps where g.area < 0.6 { fill.insert(g.id) } }
        let chosen = gaps.filter { fill.contains($0.id) }
        let sp = await Task.detached { max(0.002, CloudOps.spacing(cl)) }.value
        let pts: (Cloud, Cloud) = await Task.detached { RoomFill.points(chosen, planes, cloud: cl, spacing: sp) }.value
        let yellow = pts.0, real = pts.1
        let area = chosen.reduce(Float(0)) { $0 + $1.area }
        return (yellow, real, "AI filled \(chosen.count) missing part\(chosen.count == 1 ? "" : "s") (\(String(format: "%.1f", area)) m², \(yellow.count.formatted()) points, yellow) with texture grown from the real surfaces.\(how)")
    }
}
