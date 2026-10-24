import SwiftUI

/// Settings, stored in UserDefaults. Views read them with @AppStorage under the same keys.
enum PrefKey {
    static let pointSize = "pointSize", background = "background", colourMode = "colourMode", mergeGrid = "mergeGridMM"
    static let maxDepth = "maxDepth", strictness = "strictness", haptics = "haptics", units = "units", model = "geminiModel"
}

enum Prefs {
    static var d: UserDefaults { .standard }
    static var pointSize: Double { d.object(forKey: PrefKey.pointSize) as? Double ?? 3 }
    static var background: String { d.string(forKey: PrefKey.background) ?? "black" }
    static var colourMode: String { d.string(forKey: PrefKey.colourMode) ?? "photo" }
    static var mergeGridMM: Double { d.object(forKey: PrefKey.mergeGrid) as? Double ?? 2 }
    static var maxDepth: Double { d.object(forKey: PrefKey.maxDepth) as? Double ?? 3.5 }
    static var strictness: Int { d.object(forKey: PrefKey.strictness) as? Int ?? 1 }     // 0 loose, 1 normal, 2 strict
    static var haptics: Bool { d.object(forKey: PrefKey.haptics) as? Bool ?? true }
    static var autoTorch: Bool { d.object(forKey: "autoTorch") as? Bool ?? true }
    static var scanMesh: Bool { d.object(forKey: "scanMesh") as? Bool ?? true }
    static var metric: Bool { (d.string(forKey: PrefKey.units) ?? "metric") == "metric" }
    /// free models that answer this key (Oct 2026); 2.5 was retired for new keys
    static let models = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"]
    static var model: String {
        let m = d.string(forKey: PrefKey.model) ?? models[0]
        return models.contains(m) ? m : models[0]
    }

    /// copies the settings into the scanner's tuning values; called when a scan starts
    static func applyToScanner() {
        ScanKind.resetTuning()
        MAX_DEPTH = Float(maxDepth)
        MERGE_VOXEL = Float(mergeGridMM / 1000)
        ALERTS_ON = haptics
        // 5.7 (47): no "Strict" at capture any more - a stored 2 gets Balanced, so a scan never throws points away for it
        switch strictness {
        case 0: MIN_HITS = 1; EDGE_REL = 0.14; BLUR_REL = 0.12
        default: MIN_HITS = 2; EDGE_REL = 0.10; BLUR_REL = 0.2
        }
    }

    /// photo frequency and size, on top of the mode's own tuning (0 fewer - the default, 1 normal, 2 more)
    static var photoFrequency: Int { d.object(forKey: "photoFrequency") as? Int ?? 0 }
    static var photoSize: Double { d.object(forKey: "photoSize") as? Double ?? 0 }      // 0 = the mode decides
    static func applyPhotoPrefs(_ kind: ScanKind? = nil) {
        if let k = kind, k.splats { if photoSize > 0 { PHOTO_LONG = max(1280, CGFloat(photoSize)) }; return }   // splats need their photos
        switch photoFrequency {
        case 0: KF_MOVE *= 1.8; KF_TURN *= 1.8; MAX_KEYFRAMES = MAX_KEYFRAMES / 2
        case 2: KF_MOVE *= 0.6; KF_TURN *= 0.6; MAX_KEYFRAMES = MAX_KEYFRAMES * 3 / 2
        default: break
        }
        if photoSize > 0 { PHOTO_LONG = CGFloat(photoSize); if photoSize <= 1920 { VIDEO_PREF = VIDEO_PREF == 2 ? 2 : 1 } }
    }
    /// a reach set past a mode's own lifts it for the big-space modes (close-up modes keep theirs)
    static func applyReach(_ kind: ScanKind?) {
        if let k = kind, [.object, .surface, .person].contains(k) { return }
        MAX_DEPTH = max(MAX_DEPTH, Float(maxDepth))
    }
    static func bgColor(_ s: String) -> UIColor {
        switch s { case "grey": return UIColor(white: 0.18, alpha: 1); case "white": return .white; default: return .black }
    }

    static func length(_ metres: Float) -> String {
        if metric { return metres < 1 ? String(format: "%.0f cm", metres * 100) : String(format: "%.2f m", metres) }
        let inch = metres / 0.0254
        return inch < 36 ? String(format: "%.1f in", inch) : String(format: "%.1f ft", inch / 12)
    }
}

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage(PrefKey.pointSize) private var pointSize = 3.0
    @AppStorage(PrefKey.background) private var background = "black"
    @AppStorage(PrefKey.colourMode) private var colourMode = "photo"
    @AppStorage(PrefKey.mergeGrid) private var grid = 2.0
    @AppStorage(PrefKey.maxDepth) private var maxDepth = 3.5
    @AppStorage(PrefKey.strictness) private var strictness = 1
    @AppStorage(PrefKey.haptics) private var haptics = true
    @AppStorage("autoTorch") private var autoTorch = true
    @AppStorage("scanMesh") private var scanMesh = true
    @AppStorage("useMac") private var useMac = true
    @State private var macHere: Bool?
    @AppStorage("photoFrequency") private var photoFrequency = 0
    @AppStorage("photoSize") private var photoSize = 0.0
    @AppStorage(PrefKey.units) private var units = "metric"
    @AppStorage(PrefKey.model) private var model = "gemini-3.5-flash-lite"
    @State private var key = Gemini.key
    @AppStorage("ai.provider") private var aiProvider = "gemini"
    @State private var otherKey = ""
    @State private var otherModel = ""
    @State private var modelList: [String] = []
    @State private var testing = false
    @State private var testResult = ""
    @State private var storageText = "..."
    @State private var askFree = false
    @State private var showBug = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Scanning") {
                    // 5.7 (47): plain words, and the numbers tucked away; the defaults are what most scans want
                    DisclosureGroup("Fine-tuning") {
                        Picker("Keep points", selection: $strictness) {
                            Text("Keep more").tag(0); Text("Balanced").tag(1)
                        }.pickerStyle(.segmented)
                        Text(["Keeps nearly everything, a little more fuzz.", "Balanced (recommended)."][max(0, min(1, strictness))])
                            .font(.caption).foregroundStyle(.secondary)
                        LabeledContent("How far to scan") {
                            Stepper(Prefs.length(Float(maxDepth)), value: $maxDepth, in: 0.5...8, step: 0.25)
                        }
                        if maxDepth > 5 { Text("The iPhone's LiDAR is rated to about 5 m: beyond that readings are sparse and less exact.").font(.caption).foregroundStyle(.orange) }
                        Picker("Point spacing", selection: $grid) {
                            Text("1 mm").tag(1.0); Text("2 mm").tag(2.0); Text("4 mm").tag(4.0)
                        }
                        // 5.7 (47): every scan mode sets its own spacing (ScanKind.apply), so say where this one counts
                        Text("Finer = bigger file. Scan modes pick their own spacing; this is used when you add to an old scan.").font(.caption).foregroundStyle(.secondary)
                        Picker("Photos per scan", selection: $photoFrequency) { Text("Fewer (smaller files)").tag(0); Text("Normal").tag(1); Text("More (better models)").tag(2) }
                        Picker("Photo size", selection: $photoSize) {
                            Text("Automatic").tag(0.0); Text("Small (1280)").tag(1280.0); Text("Medium (1920)").tag(1920.0); Text("Large (2560)").tag(2560.0); Text("Max (4096)").tag(4096.0)
                        }
                        Text("Photos are for colour, the AI tools and textured models. LiDAR does the shape, so fewer, smaller photos save a lot of space.").font(.caption).foregroundStyle(.secondary)
                        // 5.7 (47): matches applyPhotoPrefs (splat modes skip the frequency, size has a 1280 floor) and applyReach
                        Text("Gaussian and Aran splat scans always keep all their photos (Photo size still applies). How far to scan only reaches further than a mode's own range; Object, Surface and Person scans keep their close range.").font(.caption).foregroundStyle(.secondary)
                        Button("Back to recommended") { strictness = 1; grid = 2.0; maxDepth = 3.5; photoFrequency = 0; photoSize = 0.0 }
                    }
                    Toggle("Haptics and sounds", isOn: $haptics)
                    Toggle("Auto torch in the dark", isOn: $autoTorch)
                    Toggle("White mesh over scanned areas", isOn: $scanMesh)
                }
                .onAppear { if strictness > 1 { strictness = 1 } }      // 5.7 (47): a stored Strict shows as Balanced
                Section {
                    Toggle("Build models on my iMac", isOn: $useMac)
                    LabeledContent("iMac") { Text(macHere == nil ? "Checking..." : (macHere! ? "Ready on this Wi-Fi" : "Not reachable")).foregroundStyle(macHere == true ? .green : .secondary) }
                } header: { Text("iMac processing") } footer: {
                    Text("On home Wi-Fi, photo models (Pro object, Pro room, Photos to 3D, OMNI) are built by the iMac's M1 at full detail: about 4 times the detail of the phone, with an 8K texture. Away from home they build on the phone.")
                }
                .task(id: useMac) { macHere = nil; macHere = await MacWorker.available() }
                Section("Viewer") {
                    LabeledContent("Point size") { Slider(value: $pointSize, in: 1...8).frame(width: 170) }
                    Picker("Background", selection: $background) {
                        Text("Black").tag("black"); Text("Grey").tag("grey"); Text("White").tag("white")
                    }
                    Picker("Colours", selection: $colourMode) {
                        Text("Photo").tag("photo"); Text("Height").tag("height")
                    }
                    Picker("Units", selection: $units) { Text("Metric").tag("metric"); Text("Imperial").tag("imperial") }
                }
                Section {
                    // 5.8: any of six AIs, each with the user's own key (nothing built in)
                    Picker("AI", selection: $aiProvider) { ForEach(AIProvider.allCases) { Text($0.title).tag($0.rawValue) } }
                    let p = AIProvider(rawValue: aiProvider) ?? .gemini
                    if p == .gemini {
                        SecureField("Gemini API key", text: $key)
                            .textInputAutocapitalization(.never).autocorrectionDisabled()
                            .onChange(of: key) { _, v in Gemini.key = v }
                        Picker("Model", selection: $model) {
                            Text("Gemini 3.5 Flash Lite (fastest)").tag("gemini-3.5-flash-lite")
                            Text("Gemini 3.1 Flash Lite").tag("gemini-3.1-flash-lite")
                            Text("Gemini 3.6 Flash").tag("gemini-3.6-flash")
                            Text("Gemini 3.8 Flash (smartest)").tag("gemini-3.8-flash")
                        }
                    } else {
                        SecureField("\(p.title) API key", text: $otherKey)
                            .textInputAutocapitalization(.never).autocorrectionDisabled()
                            .onChange(of: otherKey) { _, v in p.key = v }
                        HStack {
                            TextField("Model", text: $otherModel).textInputAutocapitalization(.never).autocorrectionDisabled()
                                .onChange(of: otherModel) { _, v in p.model = v }
                            Menu("Load models") { ForEach(modelList, id: \.self) { m in Button(m) { otherModel = m } } }
                                .onTapGesture { }
                                .task(id: aiProvider + otherKey) { modelList = (try? await AIProviders.models(p)) ?? [] }
                        }
                        if !p.seesImages { Text("\(p.title) cannot see photos: tools that look at your scan's photos send words only.").font(.caption).foregroundStyle(.orange) }
                    }
                    Button(testing ? "Testing..." : "Test connection") {
                        testing = true; testResult = ""
                        Task {
                            do { let r = try await Gemini.ask("Reply with the single word OK.", images: []); testResult = "Works: " + r.prefix(30) }
                            catch { testResult = error.localizedDescription }
                            testing = false
                        }
                    }.disabled(testing)
                    if !testResult.isEmpty { Text(testResult).font(.caption).foregroundStyle(.secondary) }
                } header: { Text("AI") } footer: {
                    Text("Used only when you tap an AI tool. Your own key (get one at \((AIProvider(rawValue: aiProvider) ?? .gemini).keyURL)); photos from the scan are sent to that AI for that request.")
                }
                .onChange(of: aiProvider) { _, v in let p = AIProvider(rawValue: v) ?? .gemini; otherKey = p.key; otherModel = p.model; testResult = ""; modelList = [] }
                .onAppear { let p = AIProvider(rawValue: aiProvider) ?? .gemini; otherKey = p.key; otherModel = p.model }
                Section {
                    LabeledContent("Used by scans", value: storageText)
                    Button("Delete scan photos (keeps every point cloud and model)", role: .destructive) { askFree = true }
                } header: { Text("Storage") } footer: { Text("Photos are what take the space. Without them a scan cannot be AI-cleaned or turned into a textured model again.") }
                Section("Help") {
                    Button { showBug = true } label: { Label("Report a bug", systemImage: "ladybug") }
                }
                Section("About") {
                    LabeledContent("Version", value: "\(Updates.currentVersion) (\(Updates.currentBuild))")
                }
            }
            .navigationTitle("Settings")
            .task { storageText = Storage.usedText() }
            .sheet(isPresented: $showBug) { BugReportView() }
            .confirmationDialog("Delete the photos of every scan?", isPresented: $askFree, titleVisibility: .visible) {
                Button("Delete photos", role: .destructive) { Storage.deletePhotos(); storageText = Storage.usedText() }
            }
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}

enum Storage {
    static var docs: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    // 5.7 (46): merged captures hard-link their photos, so each file is counted once by its file identity
    static func size(_ u: URL, seen: inout Set<NSObject>) -> Int64 {
        guard let e = FileManager.default.enumerator(at: u, includingPropertiesForKeys: [.fileSizeKey, .fileResourceIdentifierKey]) else { return 0 }
        var n: Int64 = 0
        for case let f as URL in e {
            guard let v = try? f.resourceValues(forKeys: [.fileSizeKey, .fileResourceIdentifierKey]) else { continue }
            if let id = v.fileResourceIdentifier as? NSObject, !seen.insert(id).inserted { continue }
            n += Int64(v.fileSize ?? 0)
        }
        return n
    }
    static func size(_ u: URL) -> Int64 { var seen = Set<NSObject>(); return size(u, seen: &seen) }
    static func usedText() -> String {
        var seen = Set<NSObject>()
        let photos = size(docs.appendingPathComponent("captures"), seen: &seen) + size(docs.appendingPathComponent("objects"), seen: &seen)
        let lib = size(docs.appendingPathComponent("library"), seen: &seen)
        let f = ByteCountFormatter(); f.countStyle = .file
        return "\(f.string(fromByteCount: photos + lib)) (photos \(f.string(fromByteCount: photos)))"
    }
    static func deletePhotos() {
        for d in ["captures", "objects"] { try? FileManager.default.removeItem(at: docs.appendingPathComponent(d)) }
    }
}
