import Foundation
import UIKit
import Vision
import CoreImage
import simd

/// People and faces. Remove people: the person outlines the phone saves with each photo (or, without them, Gemini's
/// boxes) mark which points are a person; ghost trails of people who moved are carved away using the LiDAR's free
/// space; then the gaps are offered as yellow AI fill. Blur faces: Apple's on-device face detector, applied to every
/// stored photo and to the colours of the points on those faces.
enum People {
    /// person mask of a view (dw x dh, 255 = person), if the phone recorded one
    static func mask(_ v: DepthView) -> [UInt8]? {
        let u = v.kf.jpg.deletingPathExtension().appendingPathExtension("person")
        guard let d = try? Data(contentsOf: u), d.count == v.dw * v.dh else { return nil }
        return [UInt8](d)
    }

    static func remove(_ cl: Cloud, item: ScanItem) async -> (Cloud, String) {
        let views = await Task.detached { DepthView.load(item, max: 90) }.value
        guard !views.isEmpty else { return (cl, "This scan has no photos, so people cannot be found.") }
        let masks = views.map { mask($0) }
        var hit = [Bool](repeating: false, count: cl.count)
        var how = ""
        if masks.contains(where: { $0 != nil }) {
            hit.withUnsafeMutableBufferPointer { hb in
                let hp = hb.baseAddress!
                parallelFor(cl.count) { i in
                    for (vi, v) in views.enumerated() {
                        guard let m = masks[vi], let (u, vv, d) = v.kf.project(cl.p[i]) else { continue }
                        let x = Int(u / v.kf.w * Float(v.dw)), y = Int(vv / v.kf.h * Float(v.dh))
                        guard x >= 0, y >= 0, x < v.dw, y < v.dh, m[y * v.dw + x] > 127 else { continue }
                        let md = v.measured(u, vv)
                        if md > 0 && abs(d - md) < 0.06 { hp[i] = true; return }     // this point IS the person in that photo
                    }
                }
            }
            how = "person outlines"
        } else {
            // no outlines on this phone: Gemini finds the people in a few photos
            // 5.7 (48): a photo that failed to load shifted every later image, so Gemini's "photo i" went through the
            // wrong camera and real points were removed - only the views whose photo is in the prompt are kept, in order
            let loaded = AITools.pickViews(views, 8).compactMap { v in v.kf.small(512).map { (v, $0) } }
            let refs = loaded.map { $0.0 }, imgs = loaded.map { $0.1 }
            let prompt = "These photos come from a 3D scan. Find every person or part of a person (hands, arms, legs). Reply JSON only: {\"people\": [{\"photo\": index, \"box\": [ymin, xmin, ymax, xmax] 0-1000}]}. [] if none."
            if let t = try? await Gemini.ask(prompt, images: imgs, json: true),
               let j = try? JSONSerialization.jsonObject(with: Data(t.utf8)) as? [String: Any], let ps = j["people"] as? [[String: Any]] {
                for p in ps {
                    guard let i = p["photo"] as? Int, i >= 0, i < refs.count, let b = p["box"] as? [Double], b.count == 4 else { continue }
                    let m = Vision.inBox(cl, refs[i], b.map { Float($0) }, tol: 0.06)
                    for k in 0..<hit.count where m[k] { hit[k] = true }
                }
                how = "Gemini"
            } else { how = "none found (Gemini unavailable)" }
        }
        let people = hit.filter { $0 }.count
        var o = cl.keeping(hit.map { !$0 })
        // people who moved leave ghost trails in places other photos saw as empty
        let ghosts = Vision.ghosts(o, views, tol: 0.03, only: nil)
        let g = ghosts.filter { $0 }.count
        if g > 0 { o = o.keeping(ghosts.map { !$0 }) }
        return (o, "Removed \(people.formatted()) person points (\(how)) and \(g.formatted()) ghost points.")
    }

    // ---------- faces ----------
    /// face boxes in a photo, in its own (sensor) pixel coordinates, whichever way the phone was held
    static func faces(_ url: URL) -> [CGRect] {
        guard let ci = CIImage(contentsOf: url) else { return [] }
        let W = ci.extent.width, H = ci.extent.height
        var out: [CGRect] = []
        for orient in [CGImagePropertyOrientation.right, .up, .left] {
            let req = VNDetectFaceRectanglesRequest()
            try? VNImageRequestHandler(ciImage: ci, orientation: orient).perform([req])
            for f in req.results ?? [] {
                let b = f.boundingBox             // normalised, origin bottom-left, in the ORIENTED image
                // back to sensor pixels (origin top-left)
                var r: CGRect
                switch orient {
                case .right: r = CGRect(x: (1 - b.maxY) * W, y: (1 - b.maxX) * H, width: b.height * W, height: b.width * H)
                case .left:  r = CGRect(x: b.minY * W, y: b.minX * H, width: b.height * W, height: b.width * H)
                default:     r = CGRect(x: b.minX * W, y: (1 - b.maxY) * H, width: b.width * W, height: b.height * H)
                }
                r = r.insetBy(dx: -r.width * 0.25, dy: -r.height * 0.3)
                if !out.contains(where: { $0.intersects(r) }) { out.append(r) }
            }
        }
        return out
    }

    /// pixelates the faces in a photo file, in place
    static func pixelate(_ url: URL, _ rects: [CGRect]) {
        guard !rects.isEmpty, let ci = CIImage(contentsOf: url) else { return }
        let H = ci.extent.height
        var img = ci
        for r in rects {
            let cr = CGRect(x: r.minX, y: H - r.maxY, width: r.width, height: r.height).intersection(ci.extent)   // CI is bottom-left
            guard !cr.isEmpty else { continue }
            let px = ci.cropped(to: cr).applyingFilter("CIPixellate", parameters: [kCIInputScaleKey: max(12, cr.width / 10), kCIInputCenterKey: CIVector(x: cr.midX, y: cr.midY)]).cropped(to: cr)
            img = px.composited(over: img)
        }
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        if let d = sharedCI.jpegRepresentation(of: img, colorSpace: cs, options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.88]) {
            try? d.write(to: url, options: .atomic)
        }
    }

    /// every photo of the scan gets its faces pixelated; points on those faces get coarse block colours
    static func blurFaces(_ cl: Cloud, item: ScanItem) -> (Cloud, String) {
        guard let cap = Library.captureURL(item) else { return (cl, "This scan has no photos to look for faces in.") }
        let kfs = Keyframe.all(in: cap)
        // 5.7 (46): deleted scan photos keep the capture name, so an empty folder said "No faces found." without checking any
        guard !kfs.isEmpty else { return (cl, "This scan has no photos to look for faces in.") }
        var o = cl, photos = 0
        var faceMask = [Bool](repeating: false, count: cl.count)
        for k in kfs {
            let rects = faces(k.jpg)
            guard !rects.isEmpty else { continue }
            photos += 1
            pixelate(k.jpg, rects)
            let base = k.jpg.deletingPathExtension()
            let dv: DepthView? = {
                guard k.dw > 0, let d = try? Data(contentsOf: base.appendingPathExtension("depth")), d.count >= k.dw * k.dh * 4 else { return nil }
                var a = [Float](repeating: 0, count: k.dw * k.dh)
                d.withUnsafeBytes { b in for i in 0..<a.count { a[i] = b.loadUnaligned(fromByteOffset: i * 4, as: Float.self) } }
                return DepthView(kf: k, depth: a, dw: k.dw, dh: k.dh)
            }()
            faceMask.withUnsafeMutableBufferPointer { fb in
                let fp = fb.baseAddress!
                parallelFor(cl.count) { i in
                    guard let (u, v, d) = k.project(cl.p[i]) else { return }
                    for r in rects where r.contains(CGPoint(x: CGFloat(u), y: CGFloat(v))) {
                        let md = dv?.measured(u, v) ?? 0
                        if md == 0 || abs(d - md) < 0.12 { fp[i] = true }
                    }
                }
            }
        }
        // face points: one colour per 2.5 cm block (a 3D pixelation)
        var blocks: [SIMD3<Int32>: (SIMD3<Float>, Float)] = [:]
        @inline(__always) func bk(_ p: SIMD3<Float>) -> SIMD3<Int32> { SIMD3(ik((p.x / 0.025)), ik((p.y / 0.025)), ik((p.z / 0.025))) }
        for i in 0..<o.count where faceMask[i] {
            let k = bk(o.p[i]); let c = SIMD3<Float>(Float(o.c[i].x), Float(o.c[i].y), Float(o.c[i].z))
            let b = blocks[k] ?? (.zero, 0); blocks[k] = (b.0 + c, b.1 + 1)
        }
        var n = 0
        for i in 0..<o.count where faceMask[i] {
            if let b = blocks[bk(o.p[i])] { let c = b.0 / b.1; o.c[i] = SIMD3(u8(c.x), u8(c.y), u8(c.z)); n += 1 }
        }
        return (o, photos == 0 ? "No faces found." : "Blurred faces in \(photos) photo\(photos == 1 ? "" : "s") and \(n.formatted()) points.")
    }
}

/// a 3D model preview with a Close button (Quick Look inside a sheet hides its own)
import SwiftUI
/// every model opens in the model screen (view, AR, edit, save, share, rename, delete)
struct ModelPreview: View {
    let url: URL
    var body: some View { ModelStudio(url: url) }
}
