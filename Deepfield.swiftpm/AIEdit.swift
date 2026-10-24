import Foundation
import UIKit
import simd

/// Gemini-led editing. Gemini is shown each reference photo next to a render of the scan from the SAME camera, so it
/// can compare them: what is real, what is doubled, what is missing, what the user wants changed. It answers with a plan
/// of steps tied to boxes on those photos; the app turns each box into 3D points using that photo's LiDAR depth.
struct SmartPlan: Decodable, Identifiable {
    struct Step: Decodable {
        var op: String                   // protect | remove | keep_only | doubles | floaters | floor | smooth | reflection | brightness | auto_colour | level
        var photo: Int?
        var box: [Float]?                // [ymin, xmin, ymax, xmax], 0-1000
        var strength: String?            // light | normal | strong
        var amount: Float?
        var what: String?                // what Gemini sees there, for the user
    }
    var summary: String
    var name: String?
    var steps: [Step]
    var id: String { summary }
}

/// one reference photo with its LiDAR depth, ready for projection
struct DepthView {
    let kf: Keyframe
    let depth: [Float]                   // dw x dh metres, 0 = unknown
    let dw: Int, dh: Int

    static func load(_ item: ScanItem, max n: Int) -> [DepthView] {
        guard let cap = Library.captureURL(item) else { return [] }
        return ModelBuilder.pick(Keyframe.all(in: cap), n).compactMap { k in
            guard k.dw > 0, let d = try? Data(contentsOf: k.jpg.deletingPathExtension().appendingPathExtension("depth")), d.count >= k.dw * k.dh * 4 else { return nil }
            let conf = try? Data(contentsOf: k.jpg.deletingPathExtension().appendingPathExtension("conf"))
            var a = [Float](repeating: 0, count: k.dw * k.dh)
            d.withUnsafeBytes { b in for i in 0..<a.count { a[i] = b.loadUnaligned(fromByteOffset: i * 4, as: Float.self) } }
            if let c = conf, c.count == a.count { for i in 0..<a.count where c[i] == 0 { a[i] = 0 } }
            return DepthView(kf: k, depth: a, dw: k.dw, dh: k.dh)
        }
    }
    /// the nearest measured surface around a photo pixel (3x3 minimum: thin parts in front win over the wall behind)
    func measured(_ u: Float, _ v: Float) -> Float {
        let x = Int(u / kf.w * Float(dw)), y = Int(v / kf.h * Float(dh))
        var best: Float = 0
        for yy in max(0, y - 1)...min(dh - 1, y + 1) { for xx in max(0, x - 1)...min(dw - 1, x + 1) {
            let m = depth[yy * dw + xx]; if m > 0 && (best == 0 || m < best) { best = m }
        } }
        return best
    }
}

enum Vision {
    /// the cloud drawn from a photo's camera (nearest point wins), to set beside the photo
    static func render(_ cl: Cloud, _ kf: Keyframe, width W: Int = 448) -> UIImage? {
        let H = Int(Float(W) * kf.h / kf.w), sx = Float(W) / kf.w
        var z = [Float](repeating: .greatestFiniteMagnitude, count: W * H)
        var rgb = [UInt8](repeating: 18, count: W * H * 4)
        let step = max(1, cl.count / 800_000)
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
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &rgb, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let cg = ctx.makeImage() else { return nil }
        return UIImage(cgImage: cg)
    }
    /// photo on the left, the scan from the same camera on the right
    static func pair(_ cl: Cloud, _ kf: Keyframe) -> Data? {
        guard let photo = UIImage(contentsOfFile: kf.jpg.path), let scan = render(cl, kf) else { return nil }
        let w: CGFloat = 448, h = w * CGFloat(kf.h / kf.w)
        let img = UIGraphicsImageRenderer(size: CGSize(width: w * 2 + 8, height: h)).image { c in
            UIColor.black.setFill(); c.fill(CGRect(x: 0, y: 0, width: w * 2 + 8, height: h))
            photo.draw(in: CGRect(x: 0, y: 0, width: w, height: h)); scan.draw(in: CGRect(x: w + 8, y: 0, width: w, height: h))
        }
        return img.jpegData(compressionQuality: 0.7)
    }

    /// points that are the visible surface inside a box of a photo (checked against that photo's LiDAR depth)
    static func inBox(_ cl: Cloud, _ v: DepthView, _ box: [Float], tol: Float = 0.05) -> [Bool] {
        let y0 = box[0] / 1000 * v.kf.h, x0 = box[1] / 1000 * v.kf.w, y1 = box[2] / 1000 * v.kf.h, x1 = box[3] / 1000 * v.kf.w
        var m = [Bool](repeating: false, count: cl.count)
        m.withUnsafeMutableBufferPointer { mb in
            let mp = mb.baseAddress!
            parallelFor(cl.count) { i in
                guard let (u, vv, d) = v.kf.project(cl.p[i]), u >= x0, u <= x1, vv >= y0, vv <= y1 else { return }
                let md = v.measured(u, vv)
                mp[i] = md == 0 || abs(d - md) < tol
            }
        }
        return m
    }

    /// doubles: a point floating in space that the LiDAR saw as EMPTY, in more views than see it as surface, is a ghost copy
    static func ghosts(_ cl: Cloud, _ views: [DepthView], tol: Float, only: [Bool]?) -> [Bool] {
        var g = [Bool](repeating: false, count: cl.count)
        g.withUnsafeMutableBufferPointer { gb in
            let gp = gb.baseAddress!
            parallelFor(cl.count) { i in
                if let o = only, !o[i] { return }
                var front = 0, ok = 0
                for v in views {
                    guard let (u, vv, d) = v.kf.project(cl.p[i]) else { continue }
                    let md = v.measured(u, vv)
                    guard md > 0 else { continue }
                    if d < md - tol { front += 1 } else if abs(d - md) <= tol { ok += 1 }
                }
                gp[i] = front >= 2 && front > ok
            }
        }
        return g
    }
}

extension AITools {
    /// asks Gemini for a plan; `task` is the tool tapped or the user's own words
    static func smartPlan(_ task: String, cloud cl: Cloud, item: ScanItem, deep: Bool = false, only: [String]? = nil) async throws -> (SmartPlan, [DepthView], [DepthView]) {
        let views = await Task.detached { DepthView.load(item, max: 32) }.value
        let picked = AITools.pickViews(views, 4)
        // photo i and the scan rendered from the same camera, as separate images (boxes are then on the photo itself);
        // 5.0: the renders are made side by side on all cores
        let pairs: [[Data]] = await Task.detached { () -> [[Data]] in
            let refs = picked
            var out = [[Data]](repeating: [], count: refs.count)
            out.withUnsafeMutableBufferPointer { ob in
                let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: refs.count) { i in
                    let v = refs[i]
                    if let ph = v.kf.small(448), let sc = Vision.render(cl, v.kf)?.jpegData(compressionQuality: 0.7) { op[i] = [ph, sc] }
                }
            }
            return out
        }.value
        // 5.7 (46): a view whose photo or render failed left a hole, so Gemini's "photo i" was refs[i+1] and its boxes
        // went through the wrong camera - only the views that made it into the prompt are kept, in the same order
        let refs = zip(picked, pairs).filter { !$0.1.isEmpty }.map { $0.0 }
        let imgs = pairs.flatMap { $0 }
        let b = cl.bounds
        let prompt = """
        You edit 3D scans in a LiDAR scanning app. The scan is a coloured point cloud, \(cl.count) points, \
        \(String(format: "%.2f x %.2f x %.2f", b.size.x, b.size.y, b.size.z)) m (x wide, y up, z deep).
        \(imgs.isEmpty ? "No photos are available." : "The images come in pairs: image 2i is reference PHOTO i, image 2i+1 is the SCAN rendered from exactly the same camera; there are \(imgs.count / 2) photos, photo 0 to \(imgs.count / 2 - 1). Compare them: doubled or ghost copies of edges and objects, stray specks in the air, reflections that became fake geometry, parts that are real in the photo.")
        The request: \(task)
        Reply with JSON only: {"summary": one or two plain sentences for the user, "name": 2-4 word name of the subject, \
        "steps": [ {"op": ..., "photo": index, "box": [ymin, xmin, ymax, xmax] 0-1000 within PHOTO i itself, "strength": "light"|"normal"|"strong", "amount": number, "what": short description} ] }.
        Ops: "protect" (real parts that must never be removed, ALWAYS add these for thin or important parts such as stands, legs, cables, handles), \
        "remove" (delete what is in the box), "keep_only" (keep only the object in the box), "doubles" (a box where the scan shows a doubled/ghost copy; strength), \
        "floaters" (stray points; strength, no box), "floor" (remove the floor or table under the subject, only if the user wants the object on its own), \
        "smooth" (amount 0-1), "reflection" (box on a mirror/glass/screen whose reflection became geometry), "brightness" (amount -0.3 to 0.3), "auto_colour", "level".
        Be conservative: never remove a real part of the subject. Only include steps that are needed.
        """
        let text = try await Gemini.ask(prompt + (only.map { " Use ONLY these ops: " + $0.joined(separator: ", ") + "." } ?? ""), images: imgs, json: true, deep: deep)
        let clean = text.replacingOccurrences(of: "```json", with: "").replacingOccurrences(of: "```", with: "")
        guard let d = clean.data(using: .utf8), var p = try? JSONDecoder().decode(SmartPlan.self, from: d) else {
            throw Gemini.Failure(errorDescription: "The AI's answer could not be read. Try again.")
        }
        if let o = only { p.steps = p.steps.filter { o.contains($0.op) } }        // a tool does only its own job
        return (p, refs, views)
    }

    static func pickViews(_ v: [DepthView], _ n: Int) -> [DepthView] {
        guard v.count > n else { return v }
        return (0..<n).map { v[$0 * (v.count - 1) / max(1, n - 1)] }
    }

    /// runs a plan. Protected points survive every step. `refs` are the photos the boxes refer to, `views` all depth views.
    nonisolated static func apply(_ p: SmartPlan, to c: Cloud, refs: [DepthView], views: [DepthView], grid: Float) -> (Cloud, String) {
        var o = c, notes: [String] = []
        let n0 = c.count
        func view(_ s: SmartPlan.Step) -> DepthView? { guard let i = s.photo, i >= 0, i < refs.count else { return nil }; return refs[i] }
        func boxMask(_ s: SmartPlan.Step) -> [Bool]? { guard let v = view(s), let b = s.box, b.count == 4 else { return nil }; return Vision.inBox(o, v, b) }
        // 5.0: the protected parts are measured ONCE and carried along with the points (was re-measured on every step)
        var prot = [Bool](repeating: false, count: o.count)
        for s in p.steps where s.op == "protect" { if let bm = boxMask(s) { for i in 0..<prot.count where bm[i] { prot[i] = true } } }
        func drop(_ kill: [Bool], _ label: String) {
            let keep = (0..<o.count).map { !kill[$0] || prot[$0] }
            let before = o.count; o = o.keeping(keep)
            prot = zip(prot, keep).filter { $0.1 }.map { $0.0 }
            if before != o.count { notes.append("\(label) \((before - o.count).formatted())") }
        }
        for s in p.steps {
            switch s.op {
            case "remove": if let m = boxMask(s) { drop(m, s.what ?? "removed") }
            case "keep_only":
                if let m = boxMask(s) {
                    var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo, any = false
                    for i in 0..<o.count where m[i] { lo = simd_min(lo, o.p[i]); hi = simd_max(hi, o.p[i]); any = true }
                    if any { lo -= 0.03; hi += 0.03; drop(o.p.map { q in !(q.x >= lo.x && q.y >= lo.y && q.z >= lo.z && q.x <= hi.x && q.y <= hi.y && q.z <= hi.z) }, "around the subject") }
                }
            case "doubles":
                let tol: Float = s.strength == "strong" ? 0.01 : (s.strength == "light" ? 0.025 : 0.015)
                drop(Vision.ghosts(o, views, tol: tol, only: boxMask(s)), "ghost points")
            case "floaters":
                let (r, k): (Float, Int) = s.strength == "strong" ? (max(0.012, grid * 5), 4) : (s.strength == "light" ? (max(0.02, grid * 6), 1) : (max(0.012, grid * 5), 2))
                drop(CloudOps.floaterKeep(o, radius: r, minN: k).map { !$0 }, "stray points")
                let (fg, nChunks) = CloudOps.floatingGhosts(o, views: views)
                if nChunks > 0 { drop(fg, "\(nChunks) floating ghost piece\(nChunks == 1 ? "" : "s")") }
            case "floor":
                if let (n, d) = CloudOps.floorPlane(o) { drop(o.p.map { simd_dot(n, $0) + d <= 0.012 }, "floor") }
            case "reflection":
                if let v = view(s), let b = s.box, b.count == 4 {
                    drop(reflectionMask(o, [AIPlan.Reflection(photo: 0, box: b, what: s.what)], frames: [v.kf]), "reflection")
                }
            case "smooth": o = CloudOps.smooth(o, radius: max(0.004, grid * 2), amount: max(0, min(0.8, s.amount ?? 0.4))); notes.append("smoothed")
            case "brightness": o = CloudOps.colour(o, brightness: max(-0.3, min(0.3, s.amount ?? 0)), contrast: 1, saturation: 1); notes.append("brightness")
            case "auto_colour": o = CloudOps.autoColour(o); notes.append("colours balanced")
            case "level": o = CloudOps.level(o).0; notes.append("levelled")
            default: break
            }
        }
        if !p.steps.contains(where: { $0.op == "floaters" }) && p.steps.contains(where: { $0.op == "doubles" }) {
            let (fg, nChunks) = CloudOps.floatingGhosts(o, views: views)
            if nChunks > 0 { drop(fg, "\(nChunks) floating ghost piece\(nChunks == 1 ? "" : "s")") }
        }
        // 5.0: no whole-scan merge at the end (it took seconds and hundreds of MB on big scans, and removals never make copies)
        return (o, EditorModel.change(n0, o.count) + (notes.isEmpty ? "" : " - " + notes.joined(separator: ", ")))
    }

    /// the steps each tool may use (Smooth only smooths, etc.)
    static func allowedOps(_ tool: String) -> [String]? {
        switch tool {
        case "doubles": return ["doubles", "protect"]
        case "floaters": return ["floaters", "protect"]
        case "floor": return ["floor", "protect"]
        case "smooth": return ["smooth", "protect"]
        default: return nil
        }
    }
    /// what each tool asks Gemini for
    static func taskText(_ tool: String) -> String {
        switch tool {
        case "cleanup": return "Make this scan clean and professional: remove ghost or doubled parts (mostly seen through by the LiDAR), floating junk and stray specks, fix doubles and drift if the measurements show them, smooth grainy surfaces lightly and balance the colours. Keep EVERY real part (protect thin parts near anything removed). Do not remove the floor or walls."
        case "doubles": return "Find every place where the scan shows a doubled or ghost copy of a surface or edge compared with the photo, and remove the copies. Protect the real thin parts."
        case "floaters": return "Remove stray points and loose floating pieces that are not in the photo (a misplaced copy of part of a cupboard, a ghost of a person). Protect every thin real part (stands, cables, legs) and real floating things (lamps)."
        case "floor": return "Remove the floor or table surface under the subject, keeping the subject whole including its feet or stand."
        case "smooth": return "Decide how much smoothing this scan needs without losing real detail (amount), and protect fine details."
        default: return tool
        }
    }
}
