import Foundation
import simd
import CoreGraphics
import ImageIO
import SceneKit
import UIKit

/// a photo for texturing: where it was taken, its lens, its depth (for "does it really see this spot")
struct TexPhoto { let url: URL; let T: simd_float4x4; let Ti: simd_float4x4; let fx: Float; let fy: Float; let cx: Float; let cy: Float; let w: Int; let h: Int; let sharp: Float
    let depth: [Float]; let dw: Int; let dh: Int }

/// a textured mesh: positions, UVs (0...1, top-left origin), triangles, and one texture atlas (RGBA)
struct TexturedMesh { var v: [SIMD3<Float>] = []; var uv: [SIMD2<Float>] = []; var n: [SIMD3<Float>] = []; var f: [UInt32] = []; var atlas: CGImage? = nil }

/// 5.0: lifelike models. Each triangle of the LiDAR surface is painted from the ONE photo that sees it best (facing it,
/// close, sharp, and confirmed by that photo's own depth that nothing is in the way). Neighbouring triangles prefer the
/// same photo so there are few seams; each patch of one photo is cut out at full resolution and packed into an atlas.
enum Texturer {
    static func project(_ p: TexPhoto, _ x: SIMD3<Float>) -> (Float, Float, Float)? {
        let c = p.Ti * SIMD4(x, 1); let z = -c.z
        guard z > 0.05 else { return nil }
        let u = p.cx + p.fx * c.x / z, v = p.cy - p.fy * c.y / z
        guard u >= 1, v >= 1, u < Float(p.w - 1), v < Float(p.h - 1) else { return nil }
        return (u, v, z)
    }
    /// how well photo p shows a triangle (0 = not at all)
    static func score(_ p: TexPhoto, _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, _ n: SIMD3<Float>) -> Float {
        let ctr = (a + b + c) / 3
        let cam = SIMD3<Float>(p.T.columns.3.x, p.T.columns.3.y, p.T.columns.3.z)
        let toCam = cam - ctr, dist = simd_length(toCam)
        let cosA = abs(simd_dot(n, toCam / dist))            // the depth check below decides which side is seen
        guard cosA > 0.2 else { return 0 }
        guard let (u, v, z) = project(p, ctr), project(p, a) != nil, project(p, b) != nil, project(p, c) != nil else { return 0 }
        // visibility: the photo's own LiDAR depth at that pixel must agree (nothing in front of the triangle)
        let dx = Int(u / Float(p.w) * Float(p.dw)), dy = Int(v / Float(p.h) * Float(p.dh))
        guard dx >= 0, dy >= 0, dx < p.dw, dy < p.dh else { return 0 }
        let md = p.depth[dy * p.dw + dx]
        guard md > 0, abs(md - z) < max(0.03, 0.03 * z) else { return 0 }
        let centre = 1 - 0.6 * max(abs(u / Float(p.w) - 0.5), abs(v / Float(p.h) - 0.5))   // the middle of a photo is sharpest
        return cosA * cosA * centre * p.sharp / max(0.25, z * z)
    }

    static func build(v: [SIMD3<Float>], f: [UInt32], photos: [TexPhoto], atlasSize S: Int, seenOnly: Bool = false, progress: (Float, String) -> Void) -> TexturedMesh {
        let nt = f.count / 3
        // face normals
        var fn = [SIMD3<Float>](repeating: .zero, count: nt)
        for t in 0..<nt { let a = v[Int(f[3 * t])], b = v[Int(f[3 * t + 1])], c = v[Int(f[3 * t + 2])]; let x = simd_cross(b - a, c - a); let l = simd_length(x); fn[t] = l > 0 ? x / l : .zero }
        // 1. best photo for every triangle (and the runner-ups, for seam smoothing)
        progress(0.05, "Choosing the best photo for every part...")
        var best = [Int32](repeating: -1, count: nt), bestS = [Float](repeating: 0, count: nt)
        best.withUnsafeMutableBufferPointer { bb in bestS.withUnsafeMutableBufferPointer { sb in
            let bp = bb.baseAddress!, sp = sb.baseAddress!
            DispatchQueue.concurrentPerform(iterations: 64) { chunk in
                let per = (nt + 63) / 64
                let lo = chunk * per, hi = min(nt, (chunk + 1) * per)
                guard lo < hi else { return }                       // 5.7 (48): small meshes leave the last chunks empty (lo > hi traps)
                for t in lo..<hi {
                    let a = v[Int(f[3 * t])], b = v[Int(f[3 * t + 1])], c = v[Int(f[3 * t + 2])]
                    for (pi, p) in photos.enumerated() { let s = score(p, a, b, c, fn[t]); if s > sp[t] { sp[t] = s; bp[t] = Int32(pi) } }
                }
            }
        } }
        // 2. fewer seams: a triangle takes its neighbours' photo when that photo shows it nearly as well
        progress(0.35, "Joining neighbouring parts...")
        var vt = [[Int32]](repeating: [], count: v.count)                     // vertex -> triangles
        for t in 0..<nt { for k in 0..<3 { vt[Int(f[3 * t + k])].append(Int32(t)) } }
        for _ in 0..<3 {
            var next = best
            for t in 0..<nt where best[t] >= 0 {
                var votes: [Int32: Int] = [:]
                for k in 0..<3 { for o in vt[Int(f[3 * t + k])] where o != Int32(t) && best[Int(o)] >= 0 { votes[best[Int(o)], default: 0] += 1 } }
                guard let (lab, cnt) = votes.max(by: { $0.value < $1.value }), lab != best[t], cnt >= 4 else { continue }
                let a = v[Int(f[3 * t])], b = v[Int(f[3 * t + 1])], c = v[Int(f[3 * t + 2])]
                if score(photos[Int(lab)], a, b, c, fn[t]) >= bestS[t] * 0.7 { next[t] = lab }
            }
            best = next
        }
        // 3. charts: connected triangles that use the same photo
        progress(0.45, "Cutting the photo patches...")
        var chart = [Int32](repeating: -1, count: nt)
        var charts: [(photo: Int32, tris: [Int32])] = []
        for t0 in 0..<nt where chart[t0] < 0 && best[t0] >= 0 {
            let id = Int32(charts.count), lab = best[t0]
            var stack = [Int32(t0)], members: [Int32] = []; chart[t0] = id
            while let t = stack.popLast() {
                members.append(t)
                for k in 0..<3 { for o in vt[Int(f[3 * Int(t) + k])] where chart[Int(o)] < 0 && best[Int(o)] == lab { chart[Int(o)] = id; stack.append(o) } }
            }
            charts.append((lab, members))
        }
        // pixel boxes of each chart in its photo
        struct Box { var x0: Float; var y0: Float; var x1: Float; var y1: Float }
        var boxes: [Box] = []
        for c in charts {
            let p = photos[Int(c.photo)]
            var bx = Box(x0: .greatestFiniteMagnitude, y0: .greatestFiniteMagnitude, x1: 0, y1: 0)
            for t in c.tris { for k in 0..<3 { if let (u, vv, _) = project(p, v[Int(f[3 * Int(t) + k])]) { bx.x0 = min(bx.x0, u); bx.y0 = min(bx.y0, vv); bx.x1 = max(bx.x1, u); bx.y1 = max(bx.y1, vv) } } }
            bx.x0 = max(0, bx.x0 - 2); bx.y0 = max(0, bx.y0 - 2); bx.x1 = min(Float(p.w), bx.x1 + 2); bx.y1 = min(Float(p.h), bx.y1 + 2)
            boxes.append(bx)
        }
        // 4. pack: shrink everything by one factor until it fits (shelf packing, tallest first)
        let fallback = 256                                                       // a strip at the bottom for unseen triangles
        var scale: Float = 1
        let area = boxes.reduce(Float(0)) { $0 + ($1.x1 - $1.x0 + 2) * ($1.y1 - $1.y0 + 2) }
        scale = min(1, (Float(S * (S - fallback)) * 0.85 / max(1, area)).squareRoot())
        let order = boxes.indices.sorted { (boxes[$0].y1 - boxes[$0].y0) > (boxes[$1].y1 - boxes[$1].y0) }
        var place = [SIMD2<Float>](repeating: .zero, count: boxes.count)
        while true {
            var x: Float = 0, y: Float = 0, rowH: Float = 0, ok = true
            for i in order {
                let w = (boxes[i].x1 - boxes[i].x0) * scale + 2, h = (boxes[i].y1 - boxes[i].y0) * scale + 2
                if x + w > Float(S) { x = 0; y += rowH; rowH = 0 }
                if y + h > Float(S - fallback) { ok = false; break }
                place[i] = SIMD2(x + 1, y + 1); x += w; rowH = max(rowH, h)
            }
            if ok { break }
            scale *= 0.92
        }
        // 5. the atlas: each photo decoded once, its patches drawn in
        progress(0.55, "Painting the photos into the texture...")
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: nil, width: S, height: S, bitsPerComponent: 8, bytesPerRow: S * 4, space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: S, height: S))
        ctx.interpolationQuality = .high
        var byPhoto: [Int32: [Int]] = [:]
        for (i, c) in charts.enumerated() { byPhoto[c.photo, default: []].append(i) }
        var done = 0
        for (pi, list) in byPhoto {
            guard let src = CGImageSourceCreateWithURL(photos[Int(pi)].url as CFURL, nil), let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
            let iw = Float(img.width) / Float(photos[Int(pi)].w)                 // the stored photo may be smaller than the lens size
            for i in list {
                let b = boxes[i]
                guard let crop = img.cropping(to: CGRect(x: CGFloat(b.x0 * iw), y: CGFloat(b.y0 * iw), width: CGFloat((b.x1 - b.x0) * iw), height: CGFloat((b.y1 - b.y0) * iw))) else { continue }
                let w = CGFloat((b.x1 - b.x0) * scale), h = CGFloat((b.y1 - b.y0) * scale)
                // CoreGraphics draws bottom-up: flip so atlas row 0 is the top
                ctx.draw(crop, in: CGRect(x: CGFloat(place[i].x), y: CGFloat(S) - CGFloat(place[i].y) - h, width: w, height: h))
            }
            done += 1
            progress(0.55 + 0.4 * Float(done) / Float(max(1, byPhoto.count)), "Painting the photos into the texture...")
        }
        // 6. the mesh with its own vertices per chart (a vertex on a seam has one UV per side)
        var out = TexturedMesh()
        var vertChart: [Int32] = [], vertOrig: [Int32] = []
        var vertexCharts = [[Int32]](repeating: [], count: v.count)
        for (i, c) in charts.enumerated() { for tt in c.tris { for k in 0..<3 { let vi = Int(f[3 * Int(tt) + k]); if !vertexCharts[vi].contains(Int32(i)) { vertexCharts[vi].append(Int32(i)) } } } }
        for (i, c) in charts.enumerated() {
            let p = photos[Int(c.photo)], b = boxes[i]
            var local: [UInt32: UInt32] = [:]
            for t in c.tris {
                for k in 0..<3 {
                    let vi = f[3 * Int(t) + k]
                    if let m = local[vi] { out.f.append(m); continue }
                    let (u, vv, _) = project(p, v[Int(vi)]) ?? (b.x0, b.y0, 0)
                    let ax = place[i].x + (u - b.x0) * scale, ay = place[i].y + (vv - b.y0) * scale
                    let m = UInt32(out.v.count); local[vi] = m
                    out.v.append(v[Int(vi)]); out.uv.append(SIMD2(ax / Float(S), ay / Float(S))); out.f.append(m); vertChart.append(Int32(i)); vertOrig.append(Int32(vi))
                }
            }
        }
        let seamTris = out.f.count / 3
        levelSeams(out, vertChart: vertChart, vertOrig: vertOrig, chartPhoto: charts.map { $0.photo }, vertexCharts: vertexCharts, photos: photos, atlas: ctx, S: S, nTris: seamTris, progress: progress)
        // triangles no photo saw: plain light grey from the fallback strip (they get the LiDAR shape, not a guessed colour)
        // seenOnly (3D photo): those are left out - the result is only what the camera really saw in colour
        let fy = (Float(S) - Float(fallback) / 2) / Float(S)
        for t in 0..<nt where best[t] < 0 && !seenOnly {
            for k in 0..<3 { out.v.append(v[Int(f[3 * t + k])]); out.uv.append(SIMD2(0.5, fy)); out.f.append(UInt32(out.v.count - 1)) }
        }
        out.n = [SIMD3<Float>](repeating: .zero, count: out.v.count)
        var t = 0
        while t < out.f.count { let a = Int(out.f[t]), b = Int(out.f[t + 1]), c = Int(out.f[t + 2]); let x = simd_cross(out.v[b] - out.v[a], out.v[c] - out.v[a]); out.n[a] += x; out.n[b] += x; out.n[c] += x; t += 3 }
        for i in 0..<out.n.count { let l = simd_length(out.n[i]); out.n[i] = l > 0 ? out.n[i] / l : SIMD3(0, 1, 0) }
        out.atlas = ctx.makeImage()
        progress(1, "Done: \(charts.count) photo patches, \(byPhoto.count) photos used")
        return out
    }

    /// small decoded copies of photos, for colour comparisons between photos
    static func thumbnails(_ ids: Set<Int32>, photos: [TexPhoto]) -> [Int32: (rgb: [UInt8], w: Int, h: Int)] {
        var small: [Int32: (rgb: [UInt8], w: Int, h: Int)] = [:]
        for pi in ids {
            guard let src = CGImageSourceCreateWithURL(photos[Int(pi)].url as CFURL, nil),
                  let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 400, kCGImageSourceCreateThumbnailWithTransform: false] as CFDictionary) else { continue }
            let w = img.width, h = img.height
            var buf = [UInt8](repeating: 0, count: w * h * 4)
            guard let c = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { continue }
            c.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
            small[pi] = (buf, w, h)
        }
        return small
    }

    /// Seamless colour: at every vertex shared by patches of different photos, each side is pulled to the average of the
    /// photos' (soft) colours there; inside a patch the correction spreads smoothly from its edges (harmonic fill), and
    /// is painted over the atlas pixel by pixel. Seams disappear; each photo keeps its own fine detail.
    static func levelSeams(_ out: TexturedMesh, vertChart: [Int32], vertOrig: [Int32], chartPhoto: [Int32], vertexCharts: [[Int32]], photos: [TexPhoto],
                           atlas ctx: CGContext, S: Int, nTris: Int, progress: (Float, String) -> Void) {
        let nv = nTris == 0 ? 0 : vertChart.count
        let small = thumbnails(Set(chartPhoto), photos: photos)
        func sample(_ pi: Int32, _ x: SIMD3<Float>) -> SIMD3<Float>? {
            guard let s = small[pi], let (u, vv, _) = project(photos[Int(pi)], x) else { return nil }
            let px = Int(u / Float(photos[Int(pi)].w) * Float(s.w)), py = Int(vv / Float(photos[Int(pi)].h) * Float(s.h))
            guard px >= 0, py >= 0, px < s.w, py < s.h else { return nil }
            let o = (py * s.w + px) * 4
            return SIMD3(max(8, Float(s.rgb[o])), max(8, Float(s.rgb[o + 1])), max(8, Float(s.rgb[o + 2])))
        }
        // fixed corrections at seam vertices
        var delta = [SIMD3<Float>](repeating: .zero, count: nv)
        var fixed = [Bool](repeating: false, count: nv)
        for i in 0..<nv {
            let orig = Int(vertOrig[i]); let cs = vertexCharts[orig]
            guard cs.count >= 2 else { continue }
            let mine = chartPhoto[Int(vertChart[i])]
            var sum = SIMD3<Float>.zero, n: Float = 0
            for c in cs { if let s = sample(chartPhoto[Int(c)], out.v[i]) { sum += s; n += 1 } }
            guard n >= 2, let own = sample(mine, out.v[i]) else { continue }
            let avg = sum / n
            let d = SIMD3<Float>(log(avg.x / own.x), log(avg.y / own.y), log(avg.z / own.z))
            delta[i] = simd_clamp(d, SIMD3(repeating: -0.6), SIMD3(repeating: 0.6)); fixed[i] = true
        }
        // neighbours inside each patch
        progress(0.9, "Blending the seams...")
        var nb = [[Int32]](repeating: [], count: nv)
        var t = 0
        while t < out.f.count && t / 3 < nTris { for k in 0..<3 { let a = out.f[t + k], b = out.f[t + (k + 1) % 3]; nb[Int(a)].append(Int32(b)); nb[Int(b)].append(Int32(a)) }; t += 3 }
        for _ in 0..<250 {
            var next = delta
            for i in 0..<nv where !fixed[i] && !nb[i].isEmpty {
                var s = SIMD3<Float>.zero; for j in nb[i] { s += delta[Int(j)] }
                next[i] = s / Float(nb[i].count)
            }
            delta = next
        }
        // paint: multiply each pixel by exp(interpolated correction)
        guard let base = ctx.data?.assumingMemoryBound(to: UInt8.self) else { return }
        var done = [Bool](repeating: false, count: S * S)
        let Sf = Float(S)
        t = 0
        while t < nTris * 3 {
            let a = Int(out.f[t]), b = Int(out.f[t + 1]), c = Int(out.f[t + 2]); t += 3
            let pa = out.uv[a] * Sf, pb = out.uv[b] * Sf, pc = out.uv[c] * Sf
            if simd_length(delta[a]) + simd_length(delta[b]) + simd_length(delta[c]) < 0.01 { continue }
            let x0 = max(0, Int(min(pa.x, min(pb.x, pc.x)) - 1)), x1 = min(S - 1, Int(max(pa.x, max(pb.x, pc.x)) + 1))
            let y0 = max(0, Int(min(pa.y, min(pb.y, pc.y)) - 1)), y1 = min(S - 1, Int(max(pa.y, max(pb.y, pc.y)) + 1))
            let den = (pb.y - pc.y) * (pa.x - pc.x) + (pc.x - pb.x) * (pa.y - pc.y)
            guard abs(den) > 1e-6 else { continue }
            for y in y0...y1 { for x in x0...x1 {
                let px = Float(x) + 0.5, py = Float(y) + 0.5
                let l1 = ((pb.y - pc.y) * (px - pc.x) + (pc.x - pb.x) * (py - pc.y)) / den
                let l2 = ((pc.y - pa.y) * (px - pc.x) + (pa.x - pc.x) * (py - pc.y)) / den
                let l3 = 1 - l1 - l2
                guard l1 >= -0.02, l2 >= -0.02, l3 >= -0.02, !done[y * S + x] else { continue }
                done[y * S + x] = true
                let d = delta[a] * l1 + delta[b] * l2 + delta[c] * l3
                let o = (y * S + x) * 4
                base[o] = u8(Float(base[o]) * exp(d.x)); base[o + 1] = u8(Float(base[o + 1]) * exp(d.y)); base[o + 2] = u8(Float(base[o + 2]) * exp(d.z))
            } }
        }
    }

    /// OBJ + MTL + JPEG (Blender, MeshLab, slicers)
    static func writeOBJ(_ m: TexturedMesh, to dir: URL, name: String) throws {
        // 5.7 (48): streamed to the file in 4 MB pieces; one String of a multi-million-triangle model (plus its encoded
        // copy) was hundreds of MB on top of the mesh and atlas
        let url = dir.appendingPathComponent("\(name).obj")
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
        let fh = try FileHandle(forWritingTo: url)
        var ok = false
        defer { try? fh.close(); if !ok { try? FileManager.default.removeItem(at: url) } }
        var s = "mtllib \(name).mtl\nusemtl tex\n"
        s.reserveCapacity(4 << 20 + 256)
        func flush(_ force: Bool = false) throws {
            guard force || s.utf8.count >= 4 << 20 else { return }
            try fh.write(contentsOf: Data(s.utf8)); s.removeAll(keepingCapacity: true)
        }
        for p in m.v { s += "v \(p.x) \(p.y) \(p.z)\n"; try flush() }
        for u in m.uv { s += "vt \(u.x) \(1 - u.y)\n"; try flush() }
        for n in m.n { s += "vn \(n.x) \(n.y) \(n.z)\n"; try flush() }
        var t = 0
        while t < m.f.count { let a = m.f[t] + 1, b = m.f[t + 1] + 1, c = m.f[t + 2] + 1; s += "f \(a)/\(a)/\(a) \(b)/\(b)/\(b) \(c)/\(c)/\(c)\n"; t += 3; try flush() }
        try flush(true)
        ok = true
        try "newmtl tex\nKd 1 1 1\nmap_Kd \(name).jpg\n".write(to: dir.appendingPathComponent("\(name).mtl"), atomically: true, encoding: .utf8)
        if let a = m.atlas, let dest = CGImageDestinationCreateWithURL(dir.appendingPathComponent("\(name).jpg") as CFURL, "public.jpeg" as CFString, 1, nil) {
            CGImageDestinationAddImage(dest, a, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary); CGImageDestinationFinalize(dest)
        }
    }
}

extension TexturedMesh {
    /// SceneKit geometry (centred), with the atlas as its texture, shown unlit so the photo colours are exact
    func geometry(centre: SIMD3<Float>) -> SCNGeometry {
        let vs = SCNGeometrySource(vertices: v.map { let q = $0 - centre; return SCNVector3(q.x, q.y, q.z) })
        let ns = SCNGeometrySource(normals: n.map { SCNVector3($0.x, $0.y, $0.z) })
        let ts = SCNGeometrySource(textureCoordinates: uv.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) })
        let el = SCNGeometryElement(data: Data(bytes: f, count: f.count * 4), primitiveType: .triangles, primitiveCount: f.count / 3, bytesPerIndex: 4)
        let g = SCNGeometry(sources: [vs, ns, ts], elements: [el])
        let m = SCNMaterial(); m.lightingModel = .constant; m.isDoubleSided = true
        if let a = atlas { m.diffuse.contents = UIImage(cgImage: a) }
        m.diffuse.wrapS = .clamp; m.diffuse.wrapT = .clamp; m.diffuse.mipFilter = .linear
        g.materials = [m]
        return g
    }
}

/// 5.0 the lifelike model: LiDAR depth from all photos gives the shape (no noise-guessed surface), the photos give the
/// look. Runs on the phone from the saved capture.
enum PhotoModel {
    static func frames(_ item: ScanItem, max n: Int) -> ([DepthFrame], [TexPhoto]) {
        guard let cap = Library.captureURL(item) else { return ([], []) }
        let all = ModelBuilder.pick(Keyframe.all(in: cap), n)
        var df: [DepthFrame] = [], tp: [TexPhoto] = []
        let med = all.map { $0.sharp }.sorted()
        let ms = max(1, med.isEmpty ? 1 : med[med.count / 2])
        for k in all where k.dw > 0 {
            let base = k.jpg.deletingPathExtension()
            guard let d = try? Data(contentsOf: base.appendingPathExtension("depth")), d.count >= k.dw * k.dh * 4 else { continue }
            var dep = [Float](repeating: 0, count: k.dw * k.dh)
            d.withUnsafeBytes { b in for i in 0..<dep.count { dep[i] = b.loadUnaligned(fromByteOffset: i * 4, as: Float.self) } }
            let raw = dep
            // 5.7 (47): the confidence also goes to the mesher, which weights medium readings lower
            var cf: [UInt8]? = nil
            if let c = try? Data(contentsOf: base.appendingPathExtension("conf")), c.count == dep.count { for i in 0..<dep.count where c[i] < 1 { dep[i] = 0 }; cf = [UInt8](c) }
            let sx = Float(k.dw) / k.w, sy = Float(k.dh) / k.h
            df.append(DepthFrame(T: k.T, fx: k.K[0][0] * sx, fy: k.K[1][1] * sy, cx: k.K[2][0] * sx, cy: k.K[2][1] * sy, depth: dep, dw: k.dw, dh: k.dh, conf: cf))
            tp.append(TexPhoto(url: k.jpg, T: k.T, Ti: k.Ti, fx: k.K[0][0], fy: k.K[1][1], cx: k.K[2][0], cy: k.K[2][1], w: Int(k.w), h: Int(k.h), sharp: min(2, k.sharp / ms), depth: raw, dw: k.dw, dh: k.dh))
        }
        // 5.0: the depth-only frames (saved between photos) add shape where no photo frame looked
        let dd = cap.appendingPathComponent("depth")
        if let names = try? FileManager.default.contentsOfDirectory(atPath: dd.path) {
            let jsons = names.filter { $0.hasSuffix(".json") }.sorted()
            let step = max(1, jsons.count / n)
            for jn in stride(from: 0, to: jsons.count, by: step).map({ jsons[$0] }) {
                let base = dd.appendingPathComponent(String(jn.dropLast(5)))
                guard let j = try? JSONSerialization.jsonObject(with: Data(contentsOf: base.appendingPathExtension("json"))) as? [String: Any],
                      let t = j["transform"] as? [Double], t.count == 16, let k = j["intrinsics"] as? [Double], k.count == 9,
                      let dw = j["depthWidth"] as? Int, let dh = j["depthHeight"] as? Int,
                      let d16 = try? Data(contentsOf: base.appendingPathExtension("depth16")), d16.count >= dw * dh * 2 else { continue }
                let iw = Float(j["imageWidth"] as? Double ?? 1920), ih = Float(j["imageHeight"] as? Double ?? 1440)
                var dep = [Float](repeating: 0, count: dw * dh)
                d16.withUnsafeBytes { b in for i in 0..<dep.count { dep[i] = Float(b.loadUnaligned(fromByteOffset: i * 2, as: Float16.self)) } }
                var cf: [UInt8]? = nil
                if let c = try? Data(contentsOf: base.appendingPathExtension("conf")), c.count == dep.count { for i in 0..<dep.count where c[i] < 1 { dep[i] = 0 }; cf = [UInt8](c) }
                let f = t.map { Float($0) }
                let T = simd_float4x4(columns: (SIMD4(f[0], f[1], f[2], f[3]), SIMD4(f[4], f[5], f[6], f[7]), SIMD4(f[8], f[9], f[10], f[11]), SIMD4(f[12], f[13], f[14], f[15])))
                let sx = Float(dw) / iw, sy = Float(dh) / ih
                df.append(DepthFrame(T: T, fx: Float(k[0]) * sx, fy: Float(k[4]) * sy, cx: Float(k[6]) * sx, cy: Float(k[7]) * sy, depth: dep, dw: dw, dh: dh, conf: cf))
            }
        }
        return (df, tp)
    }
    static func build(_ item: ScanItem, extent ext: Float, detail: Int, minWeight: Float = 1.0, atlas atlasIn: Int? = nil, seenOnly: Bool = false, progress: @escaping (Float, String) -> Void) -> TexturedMesh? {
        progress(0.02, "Reading the photos' depth...")
        let (df, tp) = frames(item, max: detail == 0 ? 60 : (detail == 1 ? 100 : 140))
        guard df.count >= 6 else { return nil }
        // small objects get fine cells: up close the LiDAR is good to 1-2 mm (a 4 mm floor made them lumpy)
        let h = max(ext < 0.8 ? 0.002 : 0.004, min(0.02, ext / Float([200, 320, 450][detail])))
        // minWeight: how much seeing a surface needs before it is kept (about one clear view at 1 m is 1.0)
        guard let m = DepthMesher.build(df, h: h, minWeight: minWeight, progress: { p, s in progress(0.05 + p * 0.4, s) }) else { return nil }
        let atlas = atlasIn ?? (detail == 0 ? 4096 : 6144)
        return Texturer.build(v: m.v, f: m.f, photos: tp, atlasSize: atlas, seenOnly: seenOnly) { p, s in progress(0.45 + p * 0.55, s) }
    }
}
