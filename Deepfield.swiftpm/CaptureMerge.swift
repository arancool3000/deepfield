import Foundation
import simd

/// 5.6: a merged scan's photos. A merge kept only the FIRST scan's photos, so the photoreal model, textures, doubles and
/// the AI tools never saw what the other scans photographed. Now every scan's photos and depth frames join one capture,
/// each moved by the same line-up as its points. Files are hard links (no extra space); only the small .json poses
/// are rewritten.
enum CaptureMerge {
    /// parts: each scan's capture folder and the transform that lined its points up. Returns the new capture's name.
    static func combine(_ parts: [(URL, simd_float4x4)]) -> String? {
        let fm = FileManager.default
        // 5.7 (46): one part is enough when the base scan has no photos of its own (its frames still need moving)
        guard !parts.isEmpty, let base = parts.first?.0.deletingLastPathComponent() else { return nil }
        let name = "merge-" + UUID().uuidString, dst = base.appendingPathComponent(name)
        var wrote = 0
        for (pi, (src, M)) in parts.enumerated() {
            for sub in ["frames", "depth"] {
                let sd = src.appendingPathComponent(sub), dd = dst.appendingPathComponent(sub)
                guard let names = try? fm.contentsOfDirectory(atPath: sd.path), !names.isEmpty else { continue }
                try? fm.createDirectory(at: dd, withIntermediateDirectories: true)
                for f in names {
                    // the scan's number in front of the frame's own (still all digits: Keyframe ids are numbers, and a continued
                    // scan's frames already carry a stamp in front, so no arithmetic that could make two names meet)
                    guard let dot = f.firstIndex(of: "."), Int(f[..<dot]) != nil else { continue }
                    // 5.7 (46): a fixed-width scan number - a bare one let scan 1's frame 12 and scan 11's frame 2 both become "112"
                    let ext = String(f[f.index(after: dot)...]), out = dd.appendingPathComponent(String(format: "%03d", pi + 1) + String(f[..<dot]) + "." + ext)
                    if ext == "json" {
                        guard var j = (try? JSONSerialization.jsonObject(with: Data(contentsOf: sd.appendingPathComponent(f)))) as? [String: Any],
                              let t = j["transform"] as? [Double], t.count == 16 else { continue }
                        let a = t.map { Float($0) }
                        let T = simd_float4x4(columns: (SIMD4(a[0], a[1], a[2], a[3]), SIMD4(a[4], a[5], a[6], a[7]), SIMD4(a[8], a[9], a[10], a[11]), SIMD4(a[12], a[13], a[14], a[15])))
                        let T2 = M * T
                        var tm: [Double] = []; for c in 0..<4 { for r in 0..<4 { tm.append(Double(T2[c][r])) } }
                        j["transform"] = tm
                        if let d = try? JSONSerialization.data(withJSONObject: j) { try? d.write(to: out); wrote += 1 }
                    } else if (try? fm.linkItem(at: sd.appendingPathComponent(f), to: out)) == nil {
                        try? fm.copyItem(at: sd.appendingPathComponent(f), to: out)
                    }
                }
            }
        }
        guard wrote > 0 else { try? fm.removeItem(at: dst); return nil }
        return name
    }
}
