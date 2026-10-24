import Foundation
import Security
import UIKit
import simd

/// Gemini (free models: Flash Lite by default). 5.8: no key is built in - each user adds their own in Settings > AI,
/// kept in the iOS Keychain; every AI tool goes to the provider chosen there (AIProviders.swift).
enum Gemini {
    private static let builtIn = ""
    private static let account = "gemini-key"

    static var key: String {
        get { keychainGet() ?? builtIn }
        set { let v = newValue.trimmingCharacters(in: .whitespacesAndNewlines); if v.isEmpty || v == builtIn { keychainDelete() } else { keychainSet(v) } }
    }

    struct Failure: LocalizedError { let errorDescription: String? }

    /// one request: text plus optional JPEG images; returns the model's text
    /// tries the chosen model first, then the other free ones if Google says a model is gone (404)
    static let deepModels = ["gemini-3.8-flash", "gemini-3.6-flash"]
    static func ask(_ prompt: String, images: [Data], json: Bool = false, deep: Bool = false) async throws -> String {
        // 5.8: ChatGPT, Claude, Grok, DeepSeek or OpenRouter when chosen in Settings > AI
        let p = AIProvider.current
        if p != .gemini { return try await AIProviders.ask(p, prompt, images: images, json: json) }
        guard !key.isEmpty else { throw Failure(errorDescription: "Add your Gemini key in Settings > AI (free at aistudio.google.com/apikey), or pick another AI there.") }
        let order = deep ? deepModels + Prefs.models.filter { !deepModels.contains($0) } : [Prefs.model] + Prefs.models.filter { $0 != Prefs.model }
        var last: Error = Failure(errorDescription: "Gemini did not answer.")
        for m in order {
            for attempt in 0..<2 {
                do { return try await askOne(m, prompt, images: images, json: json) }
                catch let e as Gone { last = Failure(errorDescription: "Gemini model \(m) is not available: \(e.msg)"); break }
                // 5.7 (43): a timeout ended the whole request; now the next (quicker) model gets a go
                catch let e as URLError where e.code == .timedOut { last = Failure(errorDescription: "Gemini took too long to answer."); Log.add("gemini \(m) timed out"); break }
                catch let e as Busy { last = Failure(errorDescription: "Gemini is busy right now (\(e.code)). Try again in a moment."); if attempt == 0 { try? await Task.sleep(nanoseconds: 1_500_000_000) } }
            }
        }
        Log.add("gemini failed: \(last.localizedDescription)")
        throw last
    }
    struct Gone: Error { let msg: String }
    struct Busy: Error { let code: Int }

    private static func askOne(_ model: String, _ prompt: String, images: [Data], json: Bool) async throws -> String {
        let k = key
        guard !k.isEmpty else { throw Failure(errorDescription: "Add a Gemini API key in Settings.") }
        var parts: [[String: Any]] = images.map { ["inline_data": ["mime_type": "image/jpeg", "data": $0.base64EncodedString()]] }
        parts.append(["text": prompt])
        var body: [String: Any] = ["contents": [["parts": parts]]]
        if json { body["generationConfig"] = ["responseMimeType": "application/json", "temperature": 0.2] }
        var req = URLRequest(url: URL(string: "https://generativelanguage.googleapis.com/v1beta/models/\(model):generateContent")!)
        req.httpMethod = "POST"; req.timeoutInterval = deepModels.contains(model) ? 150 : 75
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue(k, forHTTPHeaderField: "x-goog-api-key")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if code == 404 {
            let msg = ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any]).flatMap { ($0["error"] as? [String: Any])?["message"] as? String } ?? "not found"
            throw Gone(msg: msg)
        }
        if code == 500 || code == 502 || code == 503 || code == 504 { throw Busy(code: code) }
        if code == 429 { throw Failure(errorDescription: "Gemini's free allowance is used up for now. Try again in a minute, or tomorrow.") }
        if code == 400 || code == 403 {
            let msg = ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any]).flatMap { ($0["error"] as? [String: Any])?["message"] as? String } ?? "the key was refused"
            throw Failure(errorDescription: "Gemini: \(msg)")
        }
        guard code == 200, let j = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let cands = j["candidates"] as? [[String: Any]], let content = cands.first?["content"] as? [String: Any],
              let ps = content["parts"] as? [[String: Any]] else {
            throw Failure(errorDescription: "Gemini did not answer (HTTP \(code)).")
        }
        return ps.compactMap { $0["text"] as? String }.joined()
    }

    // ---- keychain ----
    private static func query() -> [String: Any] { [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "KindleHubScan", kSecAttrAccount as String: account] }
    private static func keychainGet() -> String? {
        var q = query(); q[kSecReturnData as String] = true; q[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: AnyObject?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        return String(data: d, encoding: .utf8)
    }
    private static func keychainSet(_ v: String) {
        keychainDelete()
        var q = query(); q[kSecValueData as String] = Data(v.utf8); q[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(q as CFDictionary, nil)
    }
    private static func keychainDelete() { SecItemDelete(query() as CFDictionary) }
}

/// A keyframe saved during the scan: photo + pose + intrinsics (the same files the Mac uses)
struct Keyframe {
    let id: Int
    let jpg: URL
    let T: simd_float4x4          // camera -> world (ARKit axes)
    let K: simd_float3x3
    let w: Float, h: Float
    let Ti: simd_float4x4         // world -> camera, worked out once (projecting used to invert T for every point)
    var sharp: Float = 0
    var dw = 0, dh = 0
    init(id: Int, jpg: URL, T: simd_float4x4, K: simd_float3x3, w: Float, h: Float) {
        self.id = id; self.jpg = jpg; self.T = T; self.K = K; self.w = w; self.h = h; self.Ti = T.inverse
    }

    static func all(in capture: URL) -> [Keyframe] {
        let dir = capture.appendingPathComponent("frames")
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return [] }
        var out: [Keyframe] = []
        for n in names.sorted() where n.hasSuffix(".json") {
            let base = String(n.dropLast(5))
            guard let d = try? Data(contentsOf: dir.appendingPathComponent(n)),
                  let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                  let t = j["transform"] as? [Double], t.count == 16, let k = j["intrinsics"] as? [Double], k.count == 9 else { continue }
            let f = t.map { Float($0) }, kf = k.map { Float($0) }
            let T = simd_float4x4(columns: (SIMD4(f[0], f[1], f[2], f[3]), SIMD4(f[4], f[5], f[6], f[7]), SIMD4(f[8], f[9], f[10], f[11]), SIMD4(f[12], f[13], f[14], f[15])))
            let K = simd_float3x3(columns: (SIMD3(kf[0], kf[1], kf[2]), SIMD3(kf[3], kf[4], kf[5]), SIMD3(kf[6], kf[7], kf[8])))
            let w = Float(j["imageWidth"] as? Double ?? 1920), h = Float(j["imageHeight"] as? Double ?? 1440)
            var key = Keyframe(id: Int(base) ?? out.count, jpg: dir.appendingPathComponent(base + ".jpg"), T: T, K: K, w: w, h: h)
            key.sharp = Float(j["sharpness"] as? Double ?? 0); key.dw = j["depthWidth"] as? Int ?? 0; key.dh = j["depthHeight"] as? Int ?? 0
            out.append(key)
        }
        return out
    }
    /// a small JPEG of the photo, for sending to Gemini
    func small(_ long: CGFloat = 640) -> Data? {
        guard let img = UIImage(contentsOfFile: jpg.path) else { return nil }
        let s = long / max(img.size.width, img.size.height)
        let size = CGSize(width: img.size.width * s, height: img.size.height * s)
        return UIGraphicsImageRenderer(size: size).image { _ in img.draw(in: CGRect(origin: .zero, size: size)) }.jpegData(compressionQuality: 0.75)
    }
    /// pixel position and distance of a world point in this photo (nil if behind the camera or off the photo)
    func project(_ p: SIMD3<Float>) -> (Float, Float, Float)? {
        let c = Ti * SIMD4(p, 1)
        let d = -c.z
        guard d > 0.05 else { return nil }
        let u = K[2][0] + K[0][0] * c.x / d, v = K[2][1] - K[1][1] * c.y / d
        guard u >= 0, v >= 0, u < w, v < h else { return nil }
        return (u, v, d)
    }
}

/// What the AI suggests; the user sees it before anything is applied.
struct AIPlan: Decodable {
    struct Crop: Decodable { var x0: Float; var x1: Float; var y0: Float; var y1: Float; var z0: Float; var z1: Float }
    struct Reflection: Decodable { var photo: Int; var box: [Float]; var what: String? }   // box = [ymin, xmin, ymax, xmax] in 0...1000
    var summary: String
    var name: String?
    var remove_floor: Bool?
    var floaters: String?            // none | light | strong
    var smooth: Float?               // 0...1
    var crop: Crop?
    var reflections: [Reflection]?
    var brightness: Float?           // -0.3...0.3
}

enum AITools {
    static func pickFrames(_ all: [Keyframe], _ n: Int) -> [Keyframe] {
        guard all.count > n else { return all }
        return (0..<n).map { all[$0 * (all.count - 1) / max(1, n - 1)] }
    }

    static func plan(for cl: Cloud, item: ScanItem) async throws -> (AIPlan, [Keyframe]) {
        let frames = Library.captureURL(item).map { pickFrames(Keyframe.all(in: $0), 6) } ?? []
        let imgs = frames.compactMap { $0.small() }
        let b = cl.bounds
        let prompt = """
        You are the clean-up assistant in a LiDAR 3D scanning app. The scan is a coloured point cloud of \(cl.count) points, \
        size \(String(format: "%.2f x %.2f x %.2f", b.size.x, b.size.y, b.size.z)) metres (x wide, y up, z deep). \
        \(imgs.isEmpty ? "No photos are available." : "The \(imgs.count) photos attached were taken during the scan; they are photo 0 to \(imgs.count - 1) in order.")
        Decide how to make this scan look clean and professional. Reply with JSON only, these keys:
        "summary": one or two plain sentences for the user saying what you will do and why;
        "name": a short name for what was scanned (2-4 words);
        "remove_floor": true if the subject sits on a floor or table that should be removed;
        "floaters": "none", "light" or "strong" (stray points in the air);
        "smooth": 0 to 1 (0 = keep detail, 0.5 = gentle);
        "brightness": -0.3 to 0.3 (0 if the colours look right);
        "crop": null, or the box to keep as fractions 0-1 of the size: {"x0","x1","y0","y1","z0","z1"} (only if there is obvious clutter around the subject);
        "reflections": a list of mirrors, glass, windows, screens or shiny surfaces seen in the photos, each {"photo": index, "box": [ymin, xmin, ymax, xmax] 0-1000, "what": "..."}; [] if none.
        Be conservative: never crop away part of the subject.
        """
        let text = try await Gemini.ask(prompt, images: imgs, json: true)
        let clean = text.replacingOccurrences(of: "```json", with: "").replacingOccurrences(of: "```", with: "")
        guard let d = clean.data(using: .utf8), let p = try? JSONDecoder().decode(AIPlan.self, from: d) else {
            throw Gemini.Failure(errorDescription: "The AI's answer could not be read. Try again.")
        }
        return (p, frames)
    }

    /// points seen through a reflective region are phantoms: in that photo they sit FARTHER than the surface itself
    static func removeReflections(_ cl: Cloud, _ refl: [AIPlan.Reflection], frames: [Keyframe]) -> (Cloud, Int) {
        let drop = reflectionMask(cl, refl, frames: frames)
        return (cl.keeping(drop.map { !$0 }), drop.filter { $0 }.count)
    }
    static func reflectionMask(_ cl: Cloud, _ refl: [AIPlan.Reflection], frames: [Keyframe]) -> [Bool] {
        var drop = [Bool](repeating: false, count: cl.count)
        for r in refl where r.photo >= 0 && r.photo < frames.count && r.box.count == 4 {
            let f = frames[r.photo]
            let y0 = r.box[0] / 1000 * f.h, x0 = r.box[1] / 1000 * f.w, y1 = r.box[2] / 1000 * f.h, x1 = r.box[3] / 1000 * f.w
            var inside: [(Int, Float)] = []
            for i in 0..<cl.count {
                if let (u, v, d) = f.project(cl.p[i]), u >= x0, u <= x1, v >= y0, v <= y1 { inside.append((i, d)) }
            }
            guard inside.count > 30 else { continue }
            let ds = inside.map { $0.1 }.sorted()
            let surface = ds[ds.count / 10]                 // the glass / mirror itself (or its frame)
            for (i, d) in inside where d > surface + 0.06 { drop[i] = true }
        }
        return drop
    }

    static func describe(_ item: ScanItem) async throws -> String {
        let frames = Library.captureURL(item).map { pickFrames(Keyframe.all(in: $0), 3) } ?? []
        let imgs = frames.compactMap { $0.small(512) }
        guard !imgs.isEmpty else { throw Gemini.Failure(errorDescription: "This scan has no photos to look at.") }
        return try await Gemini.ask("These photos come from a 3D scan. Reply with JSON {\"name\": 2-4 word name of what was scanned, \"description\": one sentence}.", images: imgs, json: true)
    }
}

extension AITools {
    /// applies an AI plan to a cloud (any thread)
    nonisolated static func apply(_ p: AIPlan, to c: Cloud, frames fr: [Keyframe], grid: Float) -> (Cloud, String) {
        let n0 = c.count; var o = c; var notes: [String] = []
        if let r = p.reflections, !r.isEmpty, !fr.isEmpty { let (x, k) = removeReflections(o, r, frames: fr); o = x; notes.append("\(k.formatted()) reflection points") }
        if let cr = p.crop, cr.x1 > cr.x0, cr.y1 > cr.y0, cr.z1 > cr.z0 {
            o = CloudOps.crop(o, lo: SIMD3(cr.x0, cr.y0, cr.z0) - 0.02, hi: SIMD3(cr.x1, cr.y1, cr.z1) + 0.02, keepInside: true); notes.append("cropped")
        }
        if p.remove_floor == true { let (x, ok) = CloudOps.removeFloor(o); o = x; if ok { notes.append("floor") } }
        switch p.floaters ?? "light" {
        case "strong": o = CloudOps.removeFloaters(o, radius: max(0.008, grid * 4), minN: 6); notes.append("floaters")
        case "none": break
        default: o = CloudOps.removeFloaters(o, radius: max(0.006, grid * 3), minN: 3); notes.append("floaters")
        }
        o = CloudOps.dedupe(o, grid: grid)
        if let s = p.smooth, s > 0.05 { o = CloudOps.smooth(o, radius: max(0.004, grid * 2), amount: min(0.8, s)); notes.append("smoothed") }
        if let b = p.brightness, abs(b) > 0.01 { o = CloudOps.colour(o, brightness: max(-0.3, min(0.3, b)), contrast: 1, saturation: 1) }
        return (o, EditorModel.change(n0, o.count) + (notes.isEmpty ? "" : " (" + notes.joined(separator: ", ") + ")"))
    }
}
