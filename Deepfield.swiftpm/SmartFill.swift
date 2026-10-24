import Foundation
import UIKit
import CoreImage
import simd

/// A photo decoded small enough to keep in memory, with its depth view, for colour look-ups.
struct PhotoView {
    let dv: DepthView
    let rgba: [UInt8]
    let pw: Int, ph: Int
    static func load(_ views: [DepthView], max n: Int = 30, long: CGFloat = 960) -> [PhotoView] {
        AITools.pickViews(views, n).compactMap { v in
            guard var ci = CIImage(contentsOf: v.kf.jpg) else { return nil }
            let s = long / max(ci.extent.width, ci.extent.height); ci = ci.transformed(by: CGAffineTransform(scaleX: s, y: s))
            let w = Int(ci.extent.width), h = Int(ci.extent.height)
            var buf = [UInt8](repeating: 0, count: w * h * 4)
            sharedCI.render(ci, toBitmap: &buf, rowBytes: w * 4, bounds: CGRect(x: 0, y: 0, width: w, height: h), format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
            return PhotoView(dv: v, rgba: buf, pw: w, ph: h)
        }
    }
    /// colour of a world point in this photo, if the photo actually sees that point (LiDAR depth agrees)
    func colour(_ p: SIMD3<Float>, tol: Float = 0.03) -> (SIMD3<UInt8>, Float)? {
        guard let (u, v, d) = dv.kf.project(p) else { return nil }
        let md = dv.measured(u, v)
        guard md == 0 || abs(d - md) < tol else { return nil }
        let x = Int(u / dv.kf.w * Float(pw)), y = Int(v / dv.kf.h * Float(ph))
        guard x >= 0, y >= 0, x < pw, y < ph else { return nil }
        let o = (y * pw + x) * 4                            // 5.7 (47): CoreImage bitmaps run top-down (this read every photo upside down)
        let centre = 1 - (abs(u / dv.kf.w - 0.5) + abs(v / dv.kf.h - 0.5))   // middle of the photo is sharpest
        return (SIMD3(rgba[o], rgba[o + 1], rgba[o + 2]), dv.kf.sharp * max(0.1, centre) / max(0.3, d))
    }
}

enum SmartFill {
    /// how many depth maps saw straight THROUGH this point (it is open space there, not a missing surface)
    static func seenThrough(_ p: SIMD3<Float>, _ views: [DepthView]) -> Int {
        var n = 0
        for v in views {
            guard let (u, vv, d) = v.kf.project(p) else { continue }
            let md = v.measured(u, vv)
            if md > 0 && md > d + 0.06 { n += 1 }
        }
        return n
    }

    /// best photo colour for each point (falls back to the given colour)
    static func recolour(_ cl: inout Cloud, _ photos: [PhotoView]) -> Int {
        guard !photos.isEmpty else { return 0 }
        let pts = cl.p, np = photos.count
        // 5.0: each photo's exposure and white balance, measured where two photos see the same point, so colours from
        // different photos match (no patchwork); then every point blends its three best views
        var sums = [SIMD3<Float>](repeating: .zero, count: np * np), cnt = [Float](repeating: 0, count: np * np)
        let probe = stride(from: 0, to: pts.count, by: max(1, pts.count / 30000)).map { $0 }
        for i in probe {
            var seen: [(Int, SIMD3<Float>)] = []
            for (k, ph) in photos.enumerated() { if let c = ph.colour(pts[i]) { let v = SIMD3<Float>(Float(c.0.x), Float(c.0.y), Float(c.0.z)); if v.x + v.y + v.z > 45 && v.x + v.y + v.z < 735 { seen.append((k, v)) } } }
            guard seen.count >= 2 else { continue }
            for a in 0..<seen.count { for b2 in (a + 1)..<seen.count {
                let (ka, va) = seen[a], (kb, vb) = seen[b2]
                // 5.7 (49): both sides floored at 1 - a 0 in one channel (deep reds and blues) gave log(0) = -inf, every gain went NaN,
                // and the clamp turned NaN into +-0.6 per channel: AI fill points kept as bright green on white walls
                let r = SIMD3<Float>(log(max(1, vb.x) / max(1, va.x)), log(max(1, vb.y) / max(1, va.y)), log(max(1, vb.z) / max(1, va.z)))      // x_a - x_b
                sums[ka * np + kb] += r; cnt[ka * np + kb] += 1
            } }
        }
        var x = [SIMD3<Float>](repeating: .zero, count: np)
        for _ in 0..<80 {
            for p in 0..<np {
                var s = SIMD3<Float>.zero, w: Float = 1                                   // a pull towards "no change"
                for q in 0..<np where q != p {
                    let n1 = cnt[p * np + q], n2 = cnt[q * np + p]
                    if n1 >= 8 { s += (x[q] + sums[p * np + q] / n1) * n1; w += n1 }
                    if n2 >= 8 { s += (x[q] - sums[q * np + p] / n2) * n2; w += n2 }
                }
                x[p] = s / w
            }
        }
        var mean = SIMD3<Float>.zero; for v in x { mean += v }; mean /= Float(np)
        let gain = x.map { v -> SIMD3<Float> in
            let d = v - mean
            guard d.x.isFinite, d.y.isFinite, d.z.isFinite else { return SIMD3(repeating: 1) }       // never apply a broken gain
            let g = simd_clamp(d, SIMD3(repeating: -0.6), SIMD3(repeating: 0.6)); return SIMD3(exp(g.x), exp(g.y), exp(g.z))
        }
        cl.c.withUnsafeMutableBufferPointer { cb in
            let cp = cb.baseAddress!
            parallelFor(pts.count) { i in
                var top: [(SIMD3<Float>, Float)] = []
                for (k, ph) in photos.enumerated() {
                    guard let c = ph.colour(pts[i]), c.1 > 0 else { continue }
                    let v = SIMD3<Float>(Float(c.0.x), Float(c.0.y), Float(c.0.z)) * gain[k]
                    top.append((v, c.1))
                }
                guard !top.isEmpty else { return }
                top.sort { $0.1 > $1.1 }
                var s = SIMD3<Float>.zero, w: Float = 0
                for (v, sc) in top.prefix(3) { let ww = sc * sc; s += v * ww; w += ww }      // the best view dominates, the next two smooth the joins
                let c = s / w
                cp[i] = SIMD3(u8(c.x), u8(c.y), u8(c.z))
            }
        }
        return pts.count
    }

    /// real LiDAR readings the fusion dropped (edges, low counts), taken back out of the saved depth maps, where the
    /// scan has nothing within 1 cm and no other depth map saw through the spot
    static func rescue(_ cl: Cloud, _ item: ScanItem, _ views: [DepthView], box: (SIMD3<Float>, SIMD3<Float>)? = nil) -> Cloud {
        // 4.5: "is there a point within ~1 cm" is a lookup in a set of 1 cm cells (was a nearest-point search)
        var occ = Set<SIMD3<Int32>>(); occ.reserveCapacity(cl.count / 2)
        for q in cl.p { occ.insert(SIMD3(ik((q.x / 0.01)), ik((q.y / 0.01)), ik((q.z / 0.01)))) }
        func near(_ p: SIMD3<Float>) -> Bool {
            let k = SIMD3<Int32>(ik((p.x / 0.01)), ik((p.y / 0.01)), ik((p.z / 0.01)))
            for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 { if occ.contains(k &+ SIMD3(dx, dy, dz)) { return true } } } }
            return false
        }
        var out = Cloud()
        for v in AITools.pickViews(views, 36) {
            let k = v.kf
            let fx = k.K[0][0] * Float(v.dw) / k.w, fy = k.K[1][1] * Float(v.dh) / k.h, cx = k.K[2][0] * Float(v.dw) / k.w, cy = k.K[2][1] * Float(v.dh) / k.h
            var y = 0
            while y < v.dh {
                var x = 0
                while x < v.dw {
                    let d = v.depth[y * v.dw + x]
                    if d > 0.1 && d < MAX_DEPTH {
                        let pc = k.T * SIMD4<Float>((Float(x) - cx) * d / fx, -(Float(y) - cy) * d / fy, -d, 1)
                        let p = SIMD3<Float>(pc.x, pc.y, pc.z)
                        let inBox = box.map { b in p.x >= b.0.x && p.y >= b.0.y && p.z >= b.0.z && p.x <= b.1.x && p.y <= b.1.y && p.z <= b.1.z } ?? true
                        if inBox && !near(p) { out.p.append(p); out.c.append(SIMD3(128, 128, 128)) }
                    }
                    x += 3
                }
                y += 3
            }
        }
        guard out.count > 0 else { return out }
        // keep readings that several maps agree on (one-off noise goes), and that nobody saw through
        out = CloudOps.dedupe(out, grid: 0.006)
        let keep = out.p.map { seenThrough($0, views) == 0 }
        out = out.keeping(keep)
        return CloudOps.removeFloaters(out, radius: 0.015, minN: 3)
    }

    /// the whole fill: rescued LiDAR, surfaces continued across holes, walls/floor/ceiling with Gemini's opening check;
    /// nothing where a depth map saw open space; colours from the photos that see each spot
    static func run(_ cl: Cloud, item: ScanItem, deep: Bool, box: (SIMD3<Float>, SIMD3<Float>)? = nil, region: ((SIMD3<Float>) -> Bool)? = nil,
                    progress: @escaping (Double, String) -> Void = { _, _ in }) async -> (Cloud, Cloud, String) {
        progress(0.02, "Reading the photos' depth...")
        let views = await Task.detached { DepthView.load(item, max: 48) }.value
        let b = cl.bounds, ext = max(b.size.x, max(b.size.y, b.size.z))
        let s = max(0.008, min(0.02, ext / 400))
        // 4.5: gaps are found on a thinned copy (about 10x fewer points); the new points are made at full detail
        // 5.0: in a box, only the box (plus a margin) is looked at; the thinning keeps one point per cell (fast)
        progress(0.12, "Looking at the area...")
        let lite: Cloud = await Task.detached {
            var src = cl
            if let bx = box {
                let m: Float = 0.3, lo = bx.0 - m, hi = bx.1 + m
                src = cl.keeping(cl.p.map { $0.x >= lo.x && $0.y >= lo.y && $0.z >= lo.z && $0.x <= hi.x && $0.y <= hi.y && $0.z <= hi.z })
            }
            return CloudOps.thin(src, grid: s / 2)
        }.value
        // 1. real data first
        progress(0.25, "Recovering real LiDAR readings...")
        let rescued = await Task.detached { rescue(lite, item, views, box: box) }.value
        var base = lite; base.p += rescued.p; base.c += rescued.c
        // 2. any surface: continue it across small holes
        let base2 = base
        progress(0.4, "Measuring the scan's point spacing...")
        let sp = await Task.detached { CloudOps.spacing(cl) }.value
        progress(0.5, box == nil ? "Continuing the surfaces across gaps..." : "Generating points in the area...")
        let inside: (SIMD3<Float>) -> Bool = region ?? { p in box.map { b in p.x >= b.0.x && p.y >= b.0.y && p.z >= b.0.z && p.x <= b.1.x && p.y <= b.1.y && p.z <= b.1.z } ?? true }
        let (hy, hr): (Cloud, Cloud) = await Task.detached {
            if let b = box { return HoleFill.fillRegion(base2, box: b, inside: inside, spacing: sp) }      // a place the user chose: fill it all, densely
            return HoleFill.fill(base2, cell: s, passes: 5, radius: 3, inside: nil, spacing: sp)
        }.value
        // 3. walls / floor / ceiling (Gemini decides openings) - whole-scan fills only
        var ry = Cloud(), rr = Cloud(), note = ""
        if box == nil { progress(0.7, "Walls, floor and ceiling (Gemini checks openings)..."); let rf = await RoomFill.run(cl, item: item, deep: deep); ry = rf.0; rr = rf.1; note = rf.2 }
        progress(0.88, "Checking against open space and colouring from the photos...")
        var yellow = Cloud(), real = Cloud()
        yellow.p = rescued.p + hy.p + ry.p; yellow.c = Array(repeating: SIMD3(255, 214, 10), count: rescued.count) + hy.c + ry.c
        real.p = rescued.p + hr.p + rr.p; real.c = rescued.c + hr.c + rr.c
        // never fill open space; colour from the photos
        let (y2, r2): (Cloud, Cloud) = await Task.detached {
            let keep = real.p.map { seenThrough($0, views) < 2 }
            var r = real.keeping(keep); let y = yellow.keeping(keep)
            let photos = PhotoView.load(views)
            _ = recolour(&r, photos)
            return (y, r)
        }.value
        let msg = "AI generated \(y2.count.formatted()) points (yellow): \(rescued.count.formatted()) recovered from the LiDAR, the rest continued from the surfaces around the gaps, coloured from your photos.\(note.isEmpty ? "" : " " + note)"
        return (y2, r2, msg)
    }

    /// Align to photos: Gemini marks where the scan does not line up with the photos; ghost layers there are carved
    /// away with the LiDAR depth maps, then every point is re-coloured from the photo that sees it best
    static func align(_ cl: Cloud, item: ScanItem, deep: Bool) async -> (Cloud, String) {
        let views = await Task.detached { DepthView.load(item, max: 60) }.value
        guard !views.isEmpty else { return (cl, "This scan has no photos to align to.") }
        // 4.6: geometric drift fix first (photo-to-photo matching and loop closure), then Gemini looks at what is left
        let (cl, driftMsg) = await Task.detached { Drift.fix(cl, item: item) { _, _ in } }.value
        Log.add(driftMsg)
        var task = "Find every place where the scan does NOT line up with the photo: shifted or doubled edges and objects, smeared parts, colours painted onto the wrong surface. Use op \"doubles\" with a box for each, strength by how bad it is. Protect real thin parts. Do not use other ops."
        if deep { task += " Look very carefully." }
        var o = cl, how = ""
        if let (p, refs, vs) = try? await AITools.smartPlan(task, cloud: cl, item: item, deep: deep) {
            let grid = Float(Prefs.mergeGridMM / 1000)
            let r = await Task.detached { AITools.apply(p, to: cl, refs: refs, views: vs, grid: grid) }.value
            o = r.0; how = "Gemini: " + p.summary
        } else {
            let g = await Task.detached { Vision.ghosts(cl, views, tol: 0.02, only: nil) }.value
            o = cl.keeping(g.map { !$0 }); how = "Gemini unavailable: carved ghost layers everywhere, gently"
            let (fg, nChunks) = CloudOps.floatingGhosts(o, views: views)
            if nChunks > 0 { o = o.keeping(fg.map { !$0 }); how += ", \(nChunks) floating ghost pieces" }
        }
        let removed = cl.count - o.count
        let o2 = o
        let recoloured: Cloud = await Task.detached { var x = o2; _ = recolour(&x, PhotoView.load(views)); return x }.value
        return (recoloured, "\(driftMsg). Then \(removed.formatted()) misplaced points removed and colours re-projected from the photos. \(how)")
    }
}

// ---------- AI-painted fill: Gemini's image model paints what no photo saw ----------
enum AIGen {
    /// renders the scan from the photo that best faces the gap, marks the gap in magenta, asks Gemini's image model to
    /// paint it in, and colours the new points from the painting. Needs an image-capable Gemini plan.
    static func paint(_ fill: Cloud, scan: Cloud, item: ScanItem) async -> (Cloud, String) {
        guard fill.count > 0 else { return (fill, "") }
        let views = await Task.detached { DepthView.load(item, max: 40) }.value
        let c = fill.p.reduce(SIMD3<Float>.zero, +) / Float(fill.count)
        guard let v = views.min(by: { a, b in
            func score(_ x: DepthView) -> Float {
                guard let (u, vv, d) = x.kf.project(c) else { return .greatestFiniteMagnitude }
                return d + (abs(u / x.kf.w - 0.5) + abs(vv / x.kf.h - 0.5)) * 2
            }
            return score(a) < score(b)
        }), v.kf.project(c) != nil else { return (fill, "No photo faces this spot, so the AI painter was skipped.") }
        let W = 640, H = Int(Float(W) * v.kf.h / v.kf.w), sx = Float(W) / v.kf.w
        let (png, map): (Data?, [Int]) = await Task.detached {
            guard let base = Vision.render(scan, v.kf, width: W) else { return (nil, []) }
            var map = [Int](repeating: -1, count: fill.count)
            let img = UIGraphicsImageRenderer(size: CGSize(width: W, height: H)).image { ctx in
                base.draw(in: CGRect(x: 0, y: 0, width: W, height: H))
                UIColor.magenta.setFill()
                for i in 0..<fill.count {
                    guard let (u, vv, _) = v.kf.project(fill.p[i]) else { continue }
                    let x = Int(u * sx), y = Int(vv * sx); guard x >= 0, y >= 0, x < W, y < H else { continue }
                    map[i] = y * W + x
                    ctx.fill(CGRect(x: x - 2, y: y - 2, width: 5, height: 5))
                }
            }
            return (img.pngData(), map)
        }.value
        guard let p = png else { return (fill, "Could not draw the scan for the AI painter.") }
        let prompt = "This is a render of a 3D scan of a real place. The MAGENTA area is missing. Repaint ONLY the magenta area so it continues the real surfaces, colours and textures around it naturally and photo-realistically. Keep every other pixel the same. Return the full image at the same size."
        do {
            let out = try await Gemini.askImage(prompt, image: p)
            guard let ui = UIImage(data: out), let cg = ui.cgImage else { return (fill, "The AI painter sent nothing back.") }
            var rgba = [UInt8](repeating: 0, count: W * H * 4)
            let ctx = CGContext(data: &rgba, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            ctx?.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))
            var o = fill, n = 0
            for i in 0..<fill.count where map[i] >= 0 {
                let px = map[i] % W, py = map[i] / W, k = (py * W + px) * 4                  // 5.7 (47): CGContext memory runs top-down, like map (it read the painting mirrored)
                let col = SIMD3<UInt8>(rgba[k], rgba[k + 1], rgba[k + 2])
                if !(col.x > 200 && col.y < 60 && col.z > 200) { o.c[i] = col; n += 1 }      // never keep leftover magenta
            }
            return (o, "Gemini painted \(n.formatted()) of the new points.")
        } catch { return (fill, error.localizedDescription) }
    }
}

extension Gemini {
    static let imageModels = ["gemini-3.1-flash-image", "gemini-2.5-flash-image"]
    /// an image back from Gemini's image models (inpainting). The free plan does not include them.
    static func askImage(_ prompt: String, image: Data) async throws -> Data {
        guard !key.isEmpty else { throw Failure(errorDescription: "The AI painter is Gemini's image model: add a Gemini key in Settings > AI.") }
        var last: Error = Failure(errorDescription: "Gemini's image painter did not answer.")
        for m in imageModels {
            var req = URLRequest(url: URL(string: "https://generativelanguage.googleapis.com/v1beta/models/\(m):generateContent")!)
            req.httpMethod = "POST"; req.timeoutInterval = 120
            req.setValue("application/json", forHTTPHeaderField: "Content-Type"); req.setValue(key, forHTTPHeaderField: "x-goog-api-key")
            let body: [String: Any] = ["contents": [["parts": [["inline_data": ["mime_type": "image/png", "data": image.base64EncodedString()]], ["text": prompt]]]],
                                       "generationConfig": ["responseModalities": ["IMAGE", "TEXT"]]]
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (data, resp) = try await URLSession.shared.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            if code == 429 {
                let msg = String(data: data, encoding: .utf8) ?? ""
                last = Failure(errorDescription: msg.contains("limit: 0") ? "Gemini's image painter is not on the free plan. Add billing to your Gemini key to use AI-painted fill - photo colours were used instead." : "Gemini's image painter is busy or out of allowance - photo colours were used instead.")
                continue
            }
            guard code == 200, let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let cands = j["candidates"] as? [[String: Any]],
                  let content = cands.first?["content"] as? [String: Any], let parts = content["parts"] as? [[String: Any]] else {
                last = Failure(errorDescription: "Gemini's image painter failed (HTTP \(code)) - photo colours were used instead."); continue
            }
            for p in parts { if let inl = (p["inlineData"] ?? p["inline_data"]) as? [String: Any], let b = inl["data"] as? String, let d = Data(base64Encoded: b) { return d } }
        }
        Log.add("image painter: \(last.localizedDescription)")
        throw last
    }
}
