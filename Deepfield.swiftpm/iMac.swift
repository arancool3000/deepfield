import Foundation

/// 5.5 iMac processing: the iMac's M1 runs Apple's photogrammetry at FULL detail (an iPhone can only make "reduced": about
/// 4x the triangles and an 8K texture instead of 2K). On home Wi-Fi the phone sends the photos, the iMac builds, the
/// phone gets the model back. Away from home, or if the iMac is off, everything builds on the phone as before.
enum MacWorker {
    static let host = "http://Home-Computer.local:8766"
    struct Job: Decodable { let state: String; let progress: Double?; let stage: String?; let error: String?; let ahead: Int? }
    enum Fail: LocalizedError { case msg(String); var errorDescription: String? { if case .msg(let m) = self { return m }; return nil } }

    /// is the iMac there (answers within 2 s)
    static func available() async -> Bool {
        guard UserDefaults.standard.object(forKey: "useMac") as? Bool ?? true, let u = URL(string: host + "/ping") else { return false }
        var r = URLRequest(url: u, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 2)
        r.httpMethod = "GET"
        guard let (d, resp) = try? await URLSession.shared.data(for: r), (resp as? HTTPURLResponse)?.statusCode == 200 else { return false }
        return (try? JSONSerialization.jsonObject(with: d) as? [String: Any])?["ok"] as? Bool == true
    }

    /// a folder as one zip file (iOS zips a folder for upload by itself)
    static func zip(_ dir: URL) throws -> URL {
        var out: URL?, err: NSError?, inner: Error?
        NSFileCoordinator().coordinate(readingItemAt: dir, options: .forUploading, error: &err) { z in
            let dst = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".zip")
            do { try FileManager.default.copyItem(at: z, to: dst); out = dst } catch { inner = error }
        }
        if let e = err ?? inner { throw e }
        guard let o = out else { throw Fail.msg("could not pack the photos") }
        return o
    }

    /// photos -> model on the iMac. Returns the downloaded USDZ.
    static func build(folder: URL, name: String, mask: Bool, detail: String = "full", progress: @escaping @MainActor (Double, String) -> Void) async throws -> URL {
        await progress(0, "Packing the photos for the iMac...")
        let z = try await Task.detached { try zip(folder) }.value
        defer { try? FileManager.default.removeItem(at: z) }
        let mb = ((try? FileManager.default.attributesOfItem(atPath: z.path)[.size] as? Int) ?? 0) / 1_000_000
        await progress(0.01, "Sending \(mb) MB of photos to the iMac...")
        var q = URLComponents(string: host + "/job")!
        q.queryItems = [URLQueryItem(name: "detail", value: detail), URLQueryItem(name: "mask", value: mask ? "1" : "0"), URLQueryItem(name: "name", value: name)]
        var r = URLRequest(url: q.url!, timeoutInterval: 600)
        r.httpMethod = "POST"; r.setValue("application/zip", forHTTPHeaderField: "Content-Type")
        let (d, resp) = try await URLSession.shared.upload(for: r, fromFile: z)
        guard (resp as? HTTPURLResponse)?.statusCode == 200, let id = (try? JSONSerialization.jsonObject(with: d) as? [String: Any])?["id"] as? String else { throw Fail.msg("the iMac did not take the photos") }
        // follow the build (a missed answer or two is fine: Wi-Fi blips)
        var misses = 0
        while true {
            try await Task.sleep(nanoseconds: 2_000_000_000)
            var g = URLRequest(url: URL(string: host + "/job/" + id)!, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 6)
            g.httpMethod = "GET"
            guard let (jd, _) = try? await URLSession.shared.data(for: g), let j = try? JSONDecoder().decode(Job.self, from: jd) else {
                misses += 1; if misses > 30 { throw Fail.msg("lost the iMac") }; continue
            }
            misses = 0
            switch j.state {
            case "done":
                await progress(0.98, "Bringing the model back...")
                let (tmp, dr) = try await URLSession.shared.download(from: URL(string: host + "/job/\(id)/model.usdz")!)
                guard (dr as? HTTPURLResponse)?.statusCode == 200 else { throw Fail.msg("could not download the model") }
                let dst = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".usdz")
                try FileManager.default.moveItem(at: tmp, to: dst)
                return dst
            case "failed": throw Fail.msg(j.error ?? "the iMac could not build it")
            case "queued": await progress(0.02, "Waiting for the iMac (\(j.ahead ?? 0) ahead)...")
            default: await progress(max(0.02, j.progress ?? 0), (j.stage ?? "Building on the iMac") + "...")
            }
        }
    }
}
