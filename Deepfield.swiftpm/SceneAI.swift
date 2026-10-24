import Foundation
import UIKit
import simd

/// 5.0 Gemini editing by numbered parts. The app splits the scan into parts and measures them (Segments); Gemini sees
/// the photos (or renders) with the numbers on, reads the measurements and the request, and answers with steps on
/// exact parts. Every step is shown in words, can be switched off, and is applied by measured geometry - never by a
/// box guessed on a photo.
struct PartPlan: Identifiable {
    struct Step: Decodable {
        var op: String
        var parts: [Int]?
        var rgb: [Float]?
        var amount: Float?
        var why: String?
    }
    struct Answer: Decodable { var summary: String?; var name: String?; var steps: [Step]? }
    let id = UUID()
    var summary: String
    var name: String?
    var steps: [Step]
}

enum SceneAI {
    static let ops: [String: String] = [
        "remove": "delete these parts",
        "keep_only": "keep only these parts, delete everything else",
        "protect": "these parts must survive every other step",
        "crop": "keep only what is inside the box around these parts (plus 5 cm)",
        "recolour": "paint these parts this colour, keeping their shading (rgb 0-255, amount 0-1 how strongly)",
        "brightness": "lighten (amount up to 0.3) or darken (down to -0.3) these parts, or all if no parts",
        "saturation": "colour strength: amount 0 = grey, 1 = unchanged, 2 = vivid; these parts or all",
        "photo_colour": "re-colour these parts (or all) from the photos that see them best",
        "auto_colour": "balance the whole scan's colours (white balance + contrast)",
        "smooth": "smooth grainy surfaces of these parts (or all), amount 0-1, edges stay sharp",
        "flatten": "press these flat parts (walls, floor, table tops) exactly flat, amount 0-1",
        "fill_holes": "fill holes and gaps in and around these parts with new points",
        "doubles": "remove doubled/ghost copies everywhere, measured against a reference model from the photos",
        "strays": "remove loose specks and floating junk the LiDAR saw through",
        "drift": "fix drift: realign the photos to each other and move the points with them",
        "level": "turn the scan so the floor is exactly level",
    ]

    static func plan(_ request: String, cloud cl: Cloud, item: ScanItem, deep: Bool, progress: @escaping (String) -> Void) async throws -> (PartPlan, Segments) {
        progress("Splitting the scan into parts...")
        let views = await Task.detached { DepthView.load(item, max: 32) }.value
        let seg = await Task.detached { Segments.find(cl, views: views) }.value
        progress(views.isEmpty ? "Drawing views of the parts for Gemini..." : "Marking the parts on the photos...")
        let (imgs, how) = await Task.detached { seg.pictures(cl, views: views) }.value
        let b = cl.bounds
        let opList = ops.sorted { $0.key < $1.key }.map { "\"\($0.key)\": \($0.value)" }.joined(separator: "\n")
        let prompt = """
        You are the editor inside a 3D LiDAR scanning app. The scan is a coloured point cloud of \(cl.count) points, \
        \(String(format: "%.2f x %.2f x %.2f", b.size.x, b.size.y, b.size.z)) m. The app has split it into numbered parts and measured them:
        \(seg.describe())
        \(how) Numbers are drawn on the parts (red = mostly seen through by the LiDAR, orange = floating, yellow = object, teal = surface).
        What the measurements mean: a part the LiDAR saw through more than 50% is almost always a ghost or doubled copy (from drift \
        or something that moved). A small floating part that is also seen through is junk. A floating part the LiDAR did NOT see \
        through can be real (a hanging lamp, a shelf).
        The request: \(request)
        Steps you can use (op: what it does):
        \(opList)
        Reply with JSON only: {"summary": one or two plain sentences for the user saying what you will do and why, \
        "name": 2-4 word name of what was scanned, "steps": [{"op": ..., "parts": [numbers] (omit for whole-scan ops), "rgb": [r,g,b] (recolour only), \
        "amount": number (where it applies), "why": a few words naming the thing, e.g. "the doubled cupboard door"}]}.
        Rules: do exactly what was asked and what is clearly needed for it, nothing else. Never remove the floor, walls or a real \
        object unless asked. Protect thin real parts near anything you remove. Refer to parts only by the numbers above. \
        If nothing needs doing, return no steps and say why.
        """
        let text = try await Gemini.ask(prompt, images: imgs, json: true, deep: deep)
        let clean = text.replacingOccurrences(of: "```json", with: "").replacingOccurrences(of: "```", with: "")
        guard let d = clean.data(using: .utf8), let a = try? JSONDecoder().decode(PartPlan.Answer.self, from: d) else {
            throw Gemini.Failure(errorDescription: "Gemini's answer could not be read. Try again.")
        }
        let valid = (a.steps ?? []).filter { s in ops[s.op] != nil }.map { s -> PartPlan.Step in
            var t = s; t.parts = s.parts?.filter { $0 >= 0 && $0 < seg.parts.count }; return t
        }
        return (PartPlan(summary: a.summary ?? "", name: a.name, steps: valid), seg)
    }

    static func title(_ s: PartPlan.Step, _ seg: Segments?) -> String {
        let names = (s.parts ?? []).map { seg?.name($0) ?? "part \($0)" }
        let on = names.isEmpty ? "" : " " + names.joined(separator: ", ")
        var t: String
        switch s.op {
        case "remove": t = "Remove" + on
        case "keep_only": t = "Keep only" + on
        case "protect": t = "Protect" + on
        case "crop": t = "Crop to" + on
        case "recolour": t = "Recolour" + on
        case "brightness": t = ((s.amount ?? 0) >= 0 ? "Brighten" : "Darken") + (on.isEmpty ? " everything" : on)
        case "saturation": t = "Colour strength" + (on.isEmpty ? " everything" : on)
        case "photo_colour": t = "Colour from photos" + on
        case "auto_colour": t = "Balance colours"
        case "smooth": t = "Smooth" + on
        case "flatten": t = "Flatten" + on
        case "fill_holes": t = "Fill holes" + on
        case "doubles": t = "Remove doubles (measured)"
        case "strays": t = "Remove stray specks and floating junk"
        case "drift": t = "Fix drift"
        case "level": t = "Level the floor"
        default: t = s.op
        }
        if let w = s.why, !w.isEmpty { t += " - " + w }
        return t
    }
    static func icon(_ op: String) -> String {
        ["remove": "trash", "keep_only": "scope", "protect": "lock.shield", "crop": "crop", "recolour": "paintbrush", "brightness": "sun.max",
         "saturation": "drop.halffull", "photo_colour": "photo", "auto_colour": "paintpalette", "smooth": "drop", "flatten": "square.split.diagonal",
         "fill_holes": "square.dashed.inset.filled", "doubles": "square.on.square", "strays": "aqi.low", "drift": "arrow.triangle.merge", "level": "level"][op] ?? "wand.and.stars"
    }

    /// applies the switched-on steps. Order: protect, removals, colour and shape per part, fills, then whole-scan fixes.
    static func apply(_ plan: PartPlan, skip: Set<Int>, to cl: Cloud, seg: Segments, item: ScanItem, grid: Float) -> (Cloud, String) {
        var o = cl
        var lab = seg.labels(o)
        var notes: [String] = []
        let steps = plan.steps.enumerated().filter { !skip.contains($0.offset) }.map { $0.element }
        let protected = Set(steps.filter { $0.op == "protect" }.flatMap { $0.parts ?? [] }.map { Int16($0) })
        func keep(_ k: [Bool], _ label: String) {
            let kk = (0..<o.count).map { k[$0] || protected.contains(lab[$0]) }
            let before = o.count
            o = o.keeping(kk); lab = zip(lab, kk).filter { $0.1 }.map { $0.0 }
            if before != o.count { notes.append("\(label): \((before - o.count).formatted()) points") }
        }
        func sel(_ s: PartPlan.Step) -> Set<Int16>? { guard let p = s.parts, !p.isEmpty else { return nil }; return Set(p.map { Int16($0) }) }
        // 1. removals
        for s in steps {
            switch s.op {
            case "remove": if let set = sel(s) { keep(lab.map { !set.contains($0) }, "removed " + (s.why ?? "parts")) }
            case "keep_only": if let set = sel(s) { keep(lab.map { set.contains($0) }, "kept only " + (s.why ?? "the chosen parts")) }
            case "crop":
                if let set = sel(s) {
                    var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
                    for p in seg.parts where set.contains(Int16(p.id)) { lo = simd_min(lo, p.lo); hi = simd_max(hi, p.hi) }
                    lo -= 0.05; hi += 0.05
                    keep(o.p.map { q in q.x >= lo.x && q.y >= lo.y && q.z >= lo.z && q.x <= hi.x && q.y <= hi.y && q.z <= hi.z }, "cropped")
                }
            case "strays":
                let junk = Set(seg.parts.filter { $0.kind == "specks" || ($0.floating && ($0.through > 0.4 || $0.count < max(200, cl.count / 400))) }.map { Int16($0.id) })
                keep(lab.map { !junk.contains($0) }, "stray specks and floating junk")
            default: break
            }
        }
        // 2. colour and shape, part by part
        func indices(_ set: Set<Int16>?) -> [Int] { set == nil ? Array(0..<o.count) : (0..<o.count).filter { set!.contains(lab[$0]) } }
        for s in steps {
            switch s.op {
            case "recolour":
                guard let rgb = s.rgb, rgb.count == 3 else { continue }
                let idx = indices(sel(s)); guard !idx.isEmpty else { continue }
                let t = SIMD3<Float>(rgb[0], rgb[1], rgb[2]), a = max(0, min(1, s.amount ?? 0.85))
                var mean: Float = 0; for i in idx { mean += Float(o.c[i].x) * 0.299 + Float(o.c[i].y) * 0.587 + Float(o.c[i].z) * 0.114 }; mean /= Float(idx.count)
                for i in idx {
                    let c = SIMD3<Float>(Float(o.c[i].x), Float(o.c[i].y), Float(o.c[i].z))
                    let lum = c.x * 0.299 + c.y * 0.587 + c.z * 0.114
                    let shaded = t * (lum / max(1, mean))
                    let v = c + (shaded - c) * a
                    o.c[i] = SIMD3(u8(v.x), u8(v.y), u8(v.z))
                }
                notes.append("recoloured " + (s.why ?? "parts"))
            case "brightness", "saturation":
                let idx = indices(sel(s))
                let b: Float = s.op == "brightness" ? max(-0.3, min(0.3, s.amount ?? 0)) : 0
                let sat: Float = s.op == "saturation" ? max(0, min(2, s.amount ?? 1)) : 1
                for i in idx {
                    var v = SIMD3<Float>(Float(o.c[i].x), Float(o.c[i].y), Float(o.c[i].z)) / 255
                    let g = simd_dot(v, SIMD3(0.299, 0.587, 0.114))
                    v = SIMD3(repeating: g) + (v - SIMD3(repeating: g)) * sat + b
                    v = simd_clamp(v, .zero, SIMD3(repeating: 1)) * 255
                    o.c[i] = SIMD3(u8(v.x), u8(v.y), u8(v.z))
                }
                notes.append(s.op == "brightness" ? (b >= 0 ? "brightened" : "darkened") : "colour strength")
            case "photo_colour":
                let idx = indices(sel(s)); guard !idx.isEmpty else { continue }
                let photos = PhotoView.load(DepthView.load(item, max: 60))
                guard !photos.isEmpty else { notes.append("no photos to colour from"); continue }
                var sub = Cloud(p: idx.map { o.p[$0] }, c: idx.map { o.c[$0] })
                _ = SmartFill.recolour(&sub, photos)
                for (j, i) in idx.enumerated() { o.c[i] = sub.c[j] }
                notes.append("coloured from \(photos.count) photos")
            case "smooth":
                let idx = indices(sel(s)); guard idx.count > 20 else { continue }
                let sub = Cloud(p: idx.map { o.p[$0] }, c: idx.map { o.c[$0] })
                let sm = CloudOps.smooth(sub, radius: max(0.004, grid * 2.5), amount: max(0.1, min(0.9, s.amount ?? 0.5)))
                for (j, i) in idx.enumerated() { o.p[i] = sm.p[j] }
                notes.append("smoothed " + (s.why ?? ""))
            case "flatten":
                for pid in s.parts ?? [] {
                    guard pid >= 0, pid < seg.parts.count, let n = seg.parts[pid].normal else { continue }
                    let d = seg.parts[pid].d, a = max(0, min(1, s.amount ?? 0.8)), l = Int16(pid)
                    for i in 0..<o.count where lab[i] == l {
                        let off = simd_dot(n, o.p[i]) + d
                        if abs(off) < 0.04 { o.p[i] -= n * off * a }
                    }
                }
                notes.append("flattened " + (s.why ?? "surfaces"))
            default: break
            }
        }
        // 3. fills
        for s in steps where s.op == "fill_holes" {
            var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
            let set = sel(s)
            for p in seg.parts where set == nil || set!.contains(Int16(p.id)) { lo = simd_min(lo, p.lo); hi = simd_max(hi, p.hi) }
            guard lo.x < hi.x else { continue }
            lo -= 0.1; hi += 0.1
            let sp = CloudOps.spacing(o)
            let (_, real) = HoleFill.fill(o, cell: max(0.008, sp * 3), passes: 4, radius: 3, inside: (lo, hi), spacing: sp)
            o.p += real.p; o.c += real.c; lab += [Int16](repeating: -1, count: real.count)
            notes.append("filled \(real.count.formatted()) points")
        }
        // 4. whole-scan fixes (they move or re-measure everything, so they come last)
        for s in steps {
            switch s.op {
            case "doubles": let r = Doubles.fix(o, item: item) { _ in }; if r.0.count != o.count { notes.append("doubles: \((o.count - r.0.count).formatted()) points") }; o = r.0
            case "drift":
                // 5.7 (48): Drift.fix hands the scan back untouched when there is nothing to fix (or no photos) - say why
                let r = Drift.fix(o, item: item) { _, _ in }
                let moved = r.0.count != o.count || r.0.p != o.p
                o = r.0; notes.append(moved ? "drift fixed" : "drift not changed: " + r.1.trimmingCharacters(in: CharacterSet(charactersIn: ". ")))
            case "level": let r = CloudOps.level(o); o = r.0; if r.1 { notes.append("levelled") }
            case "auto_colour": o = CloudOps.autoColour(o); notes.append("colours balanced")
            default: break
            }
        }
        return (o, EditorModel.change(cl.count, o.count) + (notes.isEmpty ? "" : ". " + notes.joined(separator: "; ")))
    }
}
