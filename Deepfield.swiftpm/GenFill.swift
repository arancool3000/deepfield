import Foundation
import UIKit
import simd

/// Generative fill for point clouds, the way image generators/inpainting fill a hole: the surface around the gap is laid
/// flat as an image, the missing pixels are GROWN from matching patches of real texture (exemplar synthesis with
/// coherence, so stripes, grain and carpet patterns carry on instead of blurring), photos that saw a spot win, reference
/// photos the user adds can be the texture source, and the user's own words (read by Gemini) steer the choices.
enum GenFill {
    struct Guide: Decodable { var source: String?; var tint: [Float]?; var note: String? }   // source: surroundings | reference | photos

    /// colours the new points; `fill` positions come from the dense geometric fill
    static func texture(_ fill: Cloud, scan: Cloud, item: ScanItem, refs: [UIImage], guide: String, progress: @escaping (String) -> Void) async -> (Cloud, String) {
        guard fill.count > 0 else { return (fill, "") }
        // 1. the plane of the area: fitted to the real points around the new ones
        var lo = fill.p[0], hi = fill.p[0]; for q in fill.p { lo = simd_min(lo, q); hi = simd_max(hi, q) }
        let pad: Float = 0.15
        let around = scan.p.indices.filter { i in let q = scan.p[i]; return q.x > lo.x - pad && q.y > lo.y - pad && q.z > lo.z - pad && q.x < hi.x + pad && q.y < hi.y + pad && q.z < hi.z + pad }
        let sample = stride(from: 0, to: around.count, by: max(1, around.count / 20000)).map { scan.p[around[$0]] } + CloudOps.downsample(fill, to: 3000).p
        guard let (n, centre, _) = planeNormal(sample) else { return (fill, "") }
        let e1 = simd_normalize(simd_cross(n, abs(n.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0))), e2 = simd_cross(n, e1)
        func uv(_ p: SIMD3<Float>) -> SIMD2<Float> { let d = p - centre; return SIMD2(simd_dot(d, e1), simd_dot(d, e2)) }
        var umin = SIMD2<Float>(repeating: .greatestFiniteMagnitude), umax = -umin
        for q in fill.p { let t = uv(q); umin = simd_min(umin, t); umax = simd_max(umax, t) }
        umin -= pad; umax += pad
        let span = umax - umin
        let px = max(0.002, max(span.x, span.y) / 420)                     // at most ~420 px a side
        let W = max(8, Int(span.x / px) + 1), H = max(8, Int(span.y / px) + 1)
        // 2. the raster: real colours where the scan has surface on this plane; the new points are the holes
        var img = [SIMD3<Float>](repeating: .zero, count: W * H), cnt = [Float](repeating: 0, count: W * H)
        for i in around {
            let q = scan.p[i]; guard abs(simd_dot(q - centre, n)) < 0.03 else { continue }
            let t = (uv(q) - umin) / px; let x = Int(t.x), y = Int(t.y); guard x >= 0, y >= 0, x < W, y < H else { continue }
            img[y * W + x] += SIMD3<Float>(Float(scan.c[i].x), Float(scan.c[i].y), Float(scan.c[i].z)); cnt[y * W + x] += 1
        }
        var known = [Bool](repeating: false, count: W * H)
        for i in 0..<(W * H) where cnt[i] > 0 { img[i] /= cnt[i]; known[i] = true }
        var hole = [Bool](repeating: false, count: W * H)
        var pixOf = [Int](repeating: -1, count: fill.count)
        for i in 0..<fill.count { let t = (uv(fill.p[i]) - umin) / px; let x = Int(t.x), y = Int(t.y); if x >= 0, y >= 0, x < W, y < H { pixOf[i] = y * W + x; if !known[y * W + x] { hole[y * W + x] = true } } }
        // 3. Gemini reads the user's words (and sees the area) to choose the source and any colour shift
        var g = Guide(source: refs.isEmpty ? "surroundings" : "reference", tint: nil, note: nil)
        if !guide.trimmingCharacters(in: .whitespaces).isEmpty {
            progress("Gemini is reading your description...")
            var imgs: [Data] = []
            if let d = rasterJPEG(img, known, hole, W, H) { imgs.append(d) }
            for r in refs.prefix(2) { if let d = r.jpegData(compressionQuality: 0.6) { imgs.append(d) } }
            let prompt = """
            A 3D scanning app is filling a missing patch of a surface. Image 1 shows the surface laid flat (magenta = missing). \
            \(refs.isEmpty ? "" : "The next images are reference photos the user added.") The user says: "\(guide)". \
            Reply JSON only: {"source": "surroundings" (grow the texture from the real surface around it) | "reference" (from the reference photos) , \
            "tint": [r, g, b] multipliers around 1.0 if the user asked for a colour change, else null, "note": one short sentence for the user}.
            """
            if let t = try? await Gemini.ask(prompt, images: imgs, json: true), let d = t.data(using: .utf8), let gg = try? JSONDecoder().decode(Guide.self, from: d) { g = gg }
        }
        // 4. the texture source: the real surroundings on the plane, or a reference photo at the same scale
        progress("Growing the texture into the gap...")
        var src = img, srcKnown = known, SW = W, SH = H
        if g.source == "reference", let r = refs.first, let cg = r.cgImage {
            let w = min(360, cg.width), h = max(8, Int(Float(cg.height) * Float(w) / Float(cg.width)))
            var buf = [UInt8](repeating: 0, count: w * h * 4)
            let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            ctx?.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            src = (0..<(w * h)).map { SIMD3(Float(buf[$0 * 4]), Float(buf[$0 * 4 + 1]), Float(buf[$0 * 4 + 2])) }
            srcKnown = [Bool](repeating: true, count: w * h); SW = w; SH = h
        }
        let filled = synthesize(target: img, known: known, hole: hole, W: W, H: H, src: src, srcKnown: srcKnown, SW: SW, SH: SH)
        // 5. back onto the points; photos that really saw a spot win over the synthesis
        var out = fill
        for i in 0..<fill.count where pixOf[i] >= 0 {
            var c = filled[pixOf[i]]
            if let t = g.tint, t.count == 3 { c *= SIMD3(t[0], t[1], t[2]) }
            out.c[i] = SIMD3(u8(c.x), u8(c.y), u8(c.z))
        }
        progress("Using photo colours where a photo saw the spot...")
        let photos = await Task.detached { PhotoView.load(DepthView.load(item, max: 40)) }.value
        var fromPhotos = 0
        for i in 0..<out.count {
            var best: (SIMD3<UInt8>, Float)? = nil
            for ph in photos { if let c = ph.colour(out.p[i], tol: 0.02), c.1 > (best?.1 ?? 0) { best = c } }
            if let b = best { out.c[i] = b.0; fromPhotos += 1 }
        }
        let how = g.source == "reference" ? "from your reference photo" : "from the real surface around it"
        return (out, "Texture grown \(how)\(fromPhotos > 0 ? ", \(fromPhotos.formatted()) points from your scan photos" : "").\(g.note.map { " " + $0 } ?? "")")
    }

    /// pixel-by-pixel exemplar synthesis from the hole's edge inwards: each pixel copies the centre of the best-matching
    /// 7x7 patch, searched randomly plus around where its neighbours copied from (coherence keeps patterns continuous)
    static func synthesize(target: [SIMD3<Float>], known kIn: [Bool], hole: [Bool], W: Int, H: Int,
                           src: [SIMD3<Float>], srcKnown: [Bool], SW: Int, SH: Int) -> [SIMD3<Float>] {
        var img = target, known = kIn
        let r = 3
        // source centres whose whole patch is known
        var centres: [Int] = []
        for y in r..<max(r, SH - r) { for x in r..<max(r, SW - r) {
            var ok = true
            for dy in -r...r { for dx in -r...r where !srcKnown[(y + dy) * SW + x + dx] { ok = false; break } ; if !ok { break } }
            if ok { centres.append(y * SW + x) }
        } }
        if centres.isEmpty {                                                   // too little real surface: plain average
            var s = SIMD3<Float>.zero, n: Float = 0; for i in 0..<src.count where srcKnown[i] { s += src[i]; n += 1 }
            let avg = n > 0 ? s / n : SIMD3(128, 128, 128)
            for i in 0..<img.count where hole[i] { img[i] = avg }
            return img
        }
        var from = [Int](repeating: -1, count: W * H)
        var todo = Set((0..<(W * H)).filter { hole[$0] && !known[$0] })
        var rng = SystemRandomNumberGenerator()
        while !todo.isEmpty {
            // the edge: unknown pixels with the most known neighbours first
            var edge: [(Int, Int)] = []
            for i in todo {
                let x = i % W, y = i / W; var k = 0
                for dy in -1...1 { for dx in -1...1 { let xx = x + dx, yy = y + dy; if xx >= 0, yy >= 0, xx < W, yy < H, known[yy * W + xx] { k += 1 } } }
                if k > 0 { edge.append((i, k)) }
            }
            if edge.isEmpty { for i in todo { img[i] = img.indices.first(where: { known[$0] }).map { img[$0] } ?? SIMD3(128, 128, 128); known[i] = true }; break }
            edge.sort { $0.1 > $1.1 }
            for (i, _) in edge {
                let x = i % W, y = i / W
                var cands: [Int] = (0..<120).map { _ in centres[Int.random(in: 0..<centres.count, using: &rng)] }
                for dy in -1...1 { for dx in -1...1 {
                    let xx = x + dx, yy = y + dy
                    guard xx >= 0, yy >= 0, xx < W, yy < H, let f = Optional(from[yy * W + xx]), f >= 0 else { continue }
                    let sx = f % SW - dx, sy = f / SW - dy
                    if sx >= r, sy >= r, sx < SW - r, sy < SH - r { cands.append(sy * SW + sx) }
                } }
                var best = -1, bestCost = Float.greatestFiniteMagnitude
                for c in cands {
                    let cx = c % SW, cy = c / SW; var cost: Float = 0, used = 0
                    for dy in -r...r { for dx in -r...r {
                        let tx = x + dx, ty = y + dy
                        guard tx >= 0, ty >= 0, tx < W, ty < H, known[ty * W + tx] else { continue }
                        let s = cy + dy, sx = cx + dx
                        guard s >= 0, sx >= 0, s < SH, sx < SW else { cost += 1e5; continue }
                        let d = img[ty * W + tx] - src[s * SW + sx]; cost += simd_dot(d, d); used += 1
                    } }
                    if used > 0 { cost /= Float(used) }
                    if cost < bestCost { bestCost = cost; best = c }
                }
                if best >= 0 { img[i] = src[best]; from[i] = best }
                known[i] = true; todo.remove(i)
            }
        }
        return img
    }

    static func rasterJPEG(_ img: [SIMD3<Float>], _ known: [Bool], _ hole: [Bool], _ W: Int, _ H: Int) -> Data? {
        var buf = [UInt8](repeating: 0, count: W * H * 4)
        for i in 0..<(W * H) {
            let c = hole[i] ? SIMD3<Float>(255, 0, 255) : (known[i] ? img[i] : SIMD3(20, 20, 20))
            buf[i * 4] = u8(c.x); buf[i * 4 + 1] = u8(c.y); buf[i * 4 + 2] = u8(c.z); buf[i * 4 + 3] = 255
        }
        guard let ctx = CGContext(data: &buf, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue), let cg = ctx.makeImage() else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.7)
    }
}
