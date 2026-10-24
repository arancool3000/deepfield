import SwiftUI
import UIKit

/// A small activity log (memory + a file that is trimmed), used by bug reports.
enum Log {
    private static let q = DispatchQueue(label: "log")
    private static var lines: [String] = []
    static var url: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("deepfield-log.txt") }
    static func add(_ s: String) {
        let line = "\(ISO8601DateFormatter().string(from: Date())) \(s)"
        q.async {
            lines.append(line); if lines.count > 400 { lines.removeFirst(lines.count - 400) }
            if let h = try? FileHandle(forWritingTo: url) { h.seekToEndOfFile(); h.write(Data((line + "\n").utf8)); try? h.close() }
            else { try? Data((line + "\n").utf8).write(to: url) }
            if let a = try? FileManager.default.attributesOfItem(atPath: url.path), (a[.size] as? Int ?? 0) > 400_000 {
                try? Data(lines.joined(separator: "\n").utf8).write(to: url)
            }
        }
    }
    static func recent(_ n: Int = 200) -> String {
        let all = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        return all.split(separator: "\n").suffix(n).joined(separator: "\n")
    }
}

/// Did the app end without going to the background first (a crash or a kill while open)?
enum CrashWatch {
    static let key = "df.cleanExit"
    static var crashedLastTime = false
    static func launch() {
        let d = UserDefaults.standard
        crashedLastTime = d.object(forKey: key) != nil && !d.bool(forKey: key)
        d.set(false, forKey: key)
        Log.add("launch \(Updates.currentVersion) (\(Updates.currentBuild)) crashedLastTime=\(crashedLastTime)")
    }
    static func background() { UserDefaults.standard.set(true, forKey: key) }
    static func active() { UserDefaults.standard.set(false, forKey: key) }
}

enum Diagnostics {
    @MainActor static func text() -> String {
        let p = ProcessInfo.processInfo
        let v = Updates.currentVersion, b = String(Updates.currentBuild)
        var sys = utsname(); uname(&sys)
        let model = withUnsafePointer(to: &sys.machine) { $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) } }
        let free = (try? URL(fileURLWithPath: NSHomeDirectory()).resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage) ?? 0
        let f = ByteCountFormatter()
        return """
        Deepfield \(v) (\(b)) on \(model), iOS \(UIDevice.current.systemVersion)
        Memory \(f.string(fromByteCount: Int64(p.physicalMemory))), free storage \(f.string(fromByteCount: free ?? 0)), scans use \(Storage.usedText())
        Library: \(Library.shared.items.count) items. Crashed last time: \(CrashWatch.crashedLastTime ? "yes" : "no")
        """
    }
}

struct BugReportView: View {
    @Environment(\.dismiss) private var dismiss
    var crashed = false
    @State private var what = ""
    @State private var steps = ""
    @State private var includeLog = true
    @State private var tidy = true
    @State private var busy = false
    @State private var share: URL?
    @State private var note = ""

    var body: some View {
        NavigationStack {
            Form {
                if crashed { Section { Label("Deepfield closed unexpectedly last time. A report helps fix it.", systemImage: "exclamationmark.triangle").foregroundStyle(.orange) } }
                Section("What went wrong?") {
                    TextField("e.g. it crashed when I pressed Save on a 10M point ULTRA scan", text: $what, axis: .vertical).lineLimit(3...6)
                }
                Section("What were you doing? (optional)") {
                    TextField("Mode, how long you scanned, what you tapped", text: $steps, axis: .vertical).lineLimit(2...5)
                }
                Section {
                    Toggle("Include the activity log and phone details", isOn: $includeLog)
                    Toggle("Let Gemini tidy it into a clear report", isOn: $tidy)
                } footer: { Text("The log has what the app did (modes, point counts, saves, errors). It never includes your photos or scans. To get it to Claude: AirDrop it to your Mac (it lands in Downloads) and say \"read the bug report\".") }
                Section {
                    Button(busy ? "Preparing..." : "Send report") { Task { await make() } }.disabled(busy || what.trimmingCharacters(in: .whitespaces).isEmpty)
                    if !note.isEmpty { Text(note).font(.caption).foregroundStyle(.secondary) }
                }
            }
            .navigationTitle("Report a bug")
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } } }
            .sheet(item: $share) { ShareSheet(items: [$0]) }
        }
    }

    @MainActor private func make() async {
        busy = true; defer { busy = false }
        let diag = includeLog ? Diagnostics.text() : ""
        let log = includeLog ? Log.recent(200) : ""
        var summary = ""
        if tidy {
            let prompt = """
            Turn this user's bug report for an iPhone 3D scanning app (Deepfield) into a clear developer bug report: \
            Title, What happened, Expected, Steps to reproduce, Likely area (from the log). Plain text, short.
            User: \(what)
            Doing: \(steps)
            Device: \(diag)
            Recent log:
            \(log.suffix(6000))
            """
            summary = (try? await Gemini.ask(prompt, images: [])) ?? ""
            if summary.isEmpty { note = "Gemini was unavailable, so the report is sent as written." }
        }
        let body = """
        DEEPFIELD BUG REPORT
        ====================
        \(summary.isEmpty ? "" : summary + "\n\n---- as written by the user ----\n")\(what)
        \(steps.isEmpty ? "" : "\nDoing: " + steps)

        \(diag)

        \(log.isEmpty ? "" : "---- recent log ----\n" + log)
        """
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd-HHmm"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("Deepfield-bug-\(f.string(from: Date())).txt")
        try? body.write(to: url, atomically: true, encoding: .utf8)
        Log.add("bug report made")
        share = url
    }
}
