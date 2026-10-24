import SwiftUI
import SceneKit
import RealityKit
import ARKit

/// SUPERPRO: one guided LiDAR + photo capture tuned for what is being scanned, then every tool in a row:
/// clean the cloud, build a textured model from the photos on the phone, and name it with AI.
enum ScanKind: String, CaseIterable, Identifiable {
    case room, area, object, surface, person, ultra, omni, lite, lidar, cloud, blast, gaussian, aran
    var id: String { rawValue }
    var title: String { ["room": "Room", "area": "Area", "object": "Object", "surface": "Surface", "person": "Person", "ultra": "ULTRA", "omni": "OMNI",
                         "lite": "Lite", "lidar": "LiDAR", "cloud": "Point cloud", "blast": "PowerBlast", "gaussian": "Gaussian", "aran": "Aran"][rawValue]! }
    /// ULTRA and OMNI: the autopilot runs the scan (identifies it, tunes, torch, voice guide)
    var auto: Bool { self == .ultra || self == .omni }
    /// OMNI's dense capture, kept whatever the autopilot re-tunes: 4K photos every ~10 cm / 9 degrees, depth frames between
    static func omniPhotos() {
        KF_MOVE = min(KF_MOVE, 0.10); KF_TURN = min(KF_TURN, 9 * .pi / 180); MAX_KEYFRAMES = max(MAX_KEYFRAMES, 180); VIDEO_PREF = 0; PHOTO_LONG = max(PHOTO_LONG, 2560)
    }
    /// these finish by making splats straight away
    var splats: Bool { self == .gaussian || self == .aran }
    /// the quick modes go straight to the editor (no finishing pipeline)
    var quick: Bool { [.lite, .lidar, .cloud, .blast].contains(self) }
    var icon: String { ["room": "house", "area": "square.dashed", "object": "cube", "surface": "rectangle.dashed", "person": "person.crop.circle", "ultra": "bolt.circle", "omni": "circle.hexagongrid.circle.fill",
                        "lite": "leaf", "lidar": "dot.radiowaves.left.and.right", "cloud": "circle.grid.3x3.fill", "blast": "bolt.horizontal.fill", "gaussian": "sparkles", "aran": "wand.and.stars"][rawValue]! }
    static var picks: [ScanKind] { [.room, .area, .object, .surface, .person] }
    var hint: String {
        switch self {
        case .room: return "Walk the room slowly: walls first at chest height, then low and high. Keep 1-3 m away."
        case .area: return "Cover a desk, a corner or a shelf from several heights, 50 cm-1.5 m away."
        case .object: return "Circle the object twice, once level and once from above, 25-60 cm away. Plain floor helps."
        case .surface: return "Sweep slowly across the surface, 20-40 cm away, keeping the phone parallel to it."
        case .person: return "Ask them to stay still. Walk around them at arm's length: head height first, then chest height."
        case .ultra: return "Just scan. ULTRA works out what this is, tunes itself, lights it, guides you and finishes everything."
        case .omni: return "OMNI: walk slowly round everything, at two heights. Every method runs - LiDAR, dense photos, depth frames - and is fused into a photoreal model."
        case .lite: return "Quick and small: sweep once at a steady pace. Few photos, small file."
        case .lidar: return "LiDAR only - no camera, no photos. Points are coloured by height. Move slowly for the cleanest surfaces."
        case .cloud: return "Dense colour point cloud. Move slowly and give each area a second look."
        case .blast: return "PowerBlast: sweep big rooms as fast as you like, up to 5 m away. Points appear instantly."
        case .gaussian: return "Gaussian: walk slowly all the way round, at two heights. Sharp photos from every side make the best splats."
        case .aran: return "Aran: go round twice - once level, once from higher up - and get close to the details. Every spot needs several sharp photos."
        }
    }
    /// scanner tuning: reach, detail grid, how often a photo is kept and how many
    /// the defaults every mode starts from (the globals outlive a scan)
    /// the defaults every mode starts from (the globals outlive a scan). 4.5: far fewer photos (LiDAR does the work,
    /// photos are for colour, AI and models), plain video frames (full-resolution stills pause the LiDAR feed), fusion
    /// on every frame, and no "too fast" pause until the phone is really swinging.
    static func resetTuning() {
        STRIDE = 1; MAX_VOXELS = 60_000_000; MIN_CONF = UInt8(ARConfidenceLevel.medium.rawValue); VIDEO_PREF = 1
        FUSE_GAP = 0; PREVIEW_GAP = 1.0; PHOTO_LONG = 1920; MAX_KEYFRAMES = 100; KF_MOVE = 0.30; KF_TURN = 18 * .pi / 180
        FAST_MOVE = 1.3; FAST_TURN = 160 * .pi / 180; USE_CAMERA = true; PREVIEW_SOFT = false
    }
    func apply() {
        switch self {
        case .room:    MAX_DEPTH = 5.0; MERGE_VOXEL = 0.002; KF_MOVE = 0.45; KF_TURN = 22 * .pi / 180; MAX_KEYFRAMES = 100
        case .area:    MAX_DEPTH = 3.0; MERGE_VOXEL = 0.002; KF_MOVE = 0.25; KF_TURN = 16 * .pi / 180; MAX_KEYFRAMES = 100
        case .object:  MAX_DEPTH = 1.2; MERGE_VOXEL = 0.002; KF_MOVE = 0.10; KF_TURN = 11 * .pi / 180; MAX_KEYFRAMES = 80; EDGE_REL = 0.08; VIDEO_PREF = 0; PHOTO_LONG = 2560
        case .surface: MAX_DEPTH = 0.9; MERGE_VOXEL = 0.002; KF_MOVE = 0.12; KF_TURN = 10 * .pi / 180; MAX_KEYFRAMES = 60
        case .person:  MAX_DEPTH = 1.6; MERGE_VOXEL = 0.002; KF_MOVE = 0.12; KF_TURN = 11 * .pi / 180; MAX_KEYFRAMES = 80; EDGE_REL = 0.08; VIDEO_PREF = 0; PHOTO_LONG = 2560
        case .lite:    MAX_DEPTH = 3.0; MERGE_VOXEL = 0.005; STRIDE = 2; MAX_VOXELS = 3_000_000; KF_MOVE = 0.7; KF_TURN = 35 * .pi / 180; MAX_KEYFRAMES = 30; PHOTO_LONG = 1280
                       FAST_MOVE = 1.8; FAST_TURN = 220 * .pi / 180
        case .lidar:   MAX_DEPTH = 3.5; MERGE_VOXEL = 0.002; MIN_CONF = UInt8(ARConfidenceLevel.high.rawValue); MIN_HITS = 2; EDGE_REL = 0.07; MAX_KEYFRAMES = 0; USE_CAMERA = false
        case .cloud:   MAX_DEPTH = 3.0; MERGE_VOXEL = 0.002; KF_MOVE = 0.4; KF_TURN = 25 * .pi / 180; MAX_KEYFRAMES = 60
        case .blast:   MAX_DEPTH = 6.0; MERGE_VOXEL = 0.003; MIN_HITS = 1; EDGE_REL = 0.12; PREVIEW_GAP = 0.6; KF_MOVE = 1.0; KF_TURN = 40 * .pi / 180
                       MAX_KEYFRAMES = 60; VIDEO_PREF = 2; FAST_MOVE = 3.0; FAST_TURN = 320 * .pi / 180
        case .gaussian: PREVIEW_SOFT = true; MAX_DEPTH = 3.5; MERGE_VOXEL = 0.002; KF_MOVE = 0.12; KF_TURN = 10 * .pi / 180; MAX_KEYFRAMES = 140; VIDEO_PREF = 0; PHOTO_LONG = 1920
        case .aran:    PREVIEW_SOFT = true; MAX_DEPTH = 3.0; MERGE_VOXEL = 0.002; KF_MOVE = 0.07; KF_TURN = 7 * .pi / 180; MAX_KEYFRAMES = 200; VIDEO_PREF = 0; PHOTO_LONG = 1920
        case .omni:    MAX_DEPTH = 3.5; MERGE_VOXEL = 0.002; KF_MOVE = 0.10; KF_TURN = 9 * .pi / 180; MAX_KEYFRAMES = 180; VIDEO_PREF = 0; PHOTO_LONG = 2560
        case .ultra:   MAX_DEPTH = 3.0; MERGE_VOXEL = 0.002; KF_MOVE = 0.15; KF_TURN = 12 * .pi / 180; MAX_KEYFRAMES = 120; VIDEO_PREF = 0; PHOTO_LONG = 2560
        }
    }
}

struct SuperProFinish: View {
    let item: ScanItem
    let kind: ScanKind
    var resolved: ScanKind? = nil          // what ULTRA decided it was
    @Environment(\.dismiss) private var dismiss
    @StateObject private var builder = ModelBuilder()
    @StateObject private var apple = ModelBuilder()          // 5.5 OMNI: Apple's photogrammetry model from the same capture
    @State private var step = 0                    // 0 cleaning, 1 model, 2 naming, 3 done
    @State private var cleanText = ""
    @State private var nameText = ""
    @State private var openCloud: ScanItem?
    @State private var preview: URL?
    @State private var started = false
    @State private var photoreal: ScanItem?

    var body: some View {
        NavigationStack {
            List {
                Section {
                    row(0, "Clean the point cloud", cleanText)
                    row(1, kind == .omni ? "Photoreal model (LiDAR shape, your photos)" : "Make the 3D model", step == 1 ? "\(Int(builder.progress * 100))%" : builder.status)
                    if kind == .omni || PROCESS_CHOICE == .imac {
                        HStack(spacing: 12) {
                            Group {
                                if apple.result != nil { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                                else if apple.running { ProgressView() }
                                else { Image(systemName: "circle").foregroundStyle(.secondary) }
                            }.frame(width: 24)
                            VStack(alignment: .leading) {
                                Text("Photogrammetry model (Apple, from the same photos)")
                                Text(apple.running ? "\(Int(apple.progress * 100))% - keep the app open" : apple.status).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    row(2, "Name it with AI", nameText)
                } header: { Text(kind.auto ? kind.title : "SUPERPRO \(kind.title.lowercased())") } footer: {
                    Text("Keep the app open while the model builds. You can close this screen: the point cloud is already saved.")
                }
                if step >= 3 {
                    Section {
                        Button { openCloud = item } label: { Label("Open the point cloud", systemImage: "circle.grid.3x3") }
                        if let m = builder.result { Button { preview = Library.modelURL(m.id) } label: { Label("View the 3D model", systemImage: "cube") } }
                        if let m = apple.result { Button { preview = Library.modelURL(m.id) } label: { Label("View the photogrammetry model", systemImage: "cube.transparent") } }
                        if Library.captureURL(item) != nil { Button { photoreal = item } label: { Label("View photoreal", systemImage: "camera.filters") } }
                    }
                }
            }
            .navigationTitle("Finishing").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button(step < 3 ? "Close" : "Done") { dismiss() } } }
            // 5.7 (48): run() in its own task, not the view's - closing the sheet cancels .task, and with it the
            // cancellation-aware Gemini calls (the plan fell back to the safe clean-up, the AI name failed)
            .task { if !started { started = true; Task { @MainActor in await run() } } }
            .onAppear { Awake.set("finish", true) }.onDisappear { Awake.set("finish", false) }
            .fullScreenCover(item: $openCloud) { EditorView(item: $0, autoFill: [.room, .area].contains(resolved ?? kind)) }
            .fullScreenCover(item: $preview) { ModelPreview(url: $0) }
            .fullScreenCover(item: $photoreal) { AranView(item: $0) }
        }
    }

    private func row(_ i: Int, _ title: String, _ detail: String) -> some View {
        HStack(spacing: 12) {
            Group {
                if step > i { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                else if step == i { ProgressView() }
                else { Image(systemName: "circle").foregroundStyle(.secondary) }
            }.frame(width: 24)
            VStack(alignment: .leading) {
                Text(title)
                if !detail.isEmpty { Text(detail).font(.caption).foregroundStyle(.secondary) }
            }
        }
    }

    private func run() async {
        // 1. clean: floaters and doubles out, floor off for an object, gentle smoothing except on a surface
        let it = item, k = resolved ?? kind, ultra = kind.auto, omni = kind == .omni, choice = PROCESS_CHOICE, dense = omni || choice == .dense, mac = omni || choice == .imac, grid = Float(max(0.002, Double(MERGE_VOXEL)))
        // Gemini compares the photos with the scan and plans the clean-up (protecting real parts); safe fallback without it
        cleanText = "Gemini is comparing the scan with the photos..."
        var task = AITools.taskText("cleanup")
        if k == .object || k == .person { task += " This is a scan of a single \(k == .person ? "person" : "object"): remove the floor or table under it and anything around it that is not part of it." }
        var plan: (SmartPlan, [DepthView], [DepthView])? = nil
        if let c0 = await Task.detached(operation: { Library.load(it) }).value { plan = try? await AITools.smartPlan(task, cloud: c0, item: it) }
        let pl = plan
        let result: (Cloud, String)? = await Task.detached(priority: .userInitiated) {
            guard let c = Library.load(it) else { return nil }
            // 5.6: the scan as it came is kept first ("Restore the unpolished scan" in the editor brings it back)
            if !FileManager.default.fileExists(atPath: Library.originalURL(it.id).path) { try? FileManager.default.copyItem(at: Library.pointsURL(it.id), to: Library.originalURL(it.id)) }
            var o = c, how = ""
            if let (p, refs, views) = pl { let r = AITools.apply(p, to: o, refs: refs, views: views, grid: grid); o = r.0; how = " - Gemini: " + p.summary }
            else { o = CloudOps.cleanUp(o, grid: grid, smoothIt: false); how = " (safe clean-up: Gemini unavailable)" }
            // 5.6 a clean-up never takes away a large part of a scan by itself: an OMNI room scan lost 91% when Gemini
            // read it as "a robot" and removed the room as background. Past the limit the plan is refused and only
            // the careful clean-up (strays and copies) runs.
            let limit: Float = (k == .object || k == .person) ? 0.75 : 0.15
            if Float(c.count - o.count) > Float(c.count) * limit {
                let pct = Int(100 * Float(c.count - o.count) / Float(max(1, c.count)))
                Log.add("finish clean-up refused: would remove \(c.count - o.count) of \(c.count)")
                o = CloudOps.cleanUp(c, grid: grid, smoothIt: false)
                how = " (careful clean-up: Gemini's plan would have removed \(pct)% - too much)"
            }
            if ultra { o = CloudOps.autoColour(o) }
            return (o, EditorModel.change(c.count, o.count) + how)
        }.value
        // 5.7 (48): the cloud carries on in memory - update() only queues the write on Library.io, so reading points.bin
        // back here got the file from before the clean-up (or the fuse), and the next write put that back
        var cur: Cloud? = nil
        if let (o, t) = result { Library.shared.update(it, cloud: o); cur = o; cleanText = t } else { cleanText = "Could not open the scan" }
        // OMNI: every measured method in turn - drift fixed (only when it helps), doubles against the photo reference,
        // then Quick fill for a dense, even surface
        if omni, let c0 = cur {
            cleanText = "Fixing drift and doubles..."
            let fused: (Cloud, String) = await Task.detached(priority: .userInitiated) {
                var c = c0, notes: [String] = []
                let (d, m1) = Drift.fix(c, item: it) { _, _ in }; if !m1.hasPrefix("No drift") { c = d; notes.append("drift fixed") }
                let (e, _) = Doubles.fix(c, item: it) { _ in }; if e.count < c.count { notes.append("\((c.count - e.count).formatted()) doubled points out") }; c = e
                let f = CloudOps.quickFill(c); notes.append("Quick fill \(c.count.formatted()) -> \(f.count.formatted())")
                return (f, notes.joined(separator: ", "))
            }.value
            Library.shared.update(it, cloud: fused.0); cur = fused.0; Library.shared.setPolished(it)
            cleanText = (cleanText.isEmpty ? "" : cleanText + ". ") + fused.1
        }
        step = 1

        // 2. the lifelike model: LiDAR depth from every photo gives the shape, the photos give the look (from points if there are no photos)
        builder.status = "Making the 3D model..."
        var loaded = cur
        if loaded == nil { loaded = await Task.detached(operation: { Library.load(it) }).value }
        if let cl = loaded {
            let ext = max(cl.bounds.size.x, max(cl.bounds.size.y, cl.bounds.size.z)), centre = cl.bounds.centre
            var geo: SCNGeometry? = nil, note = ""
            if Library.captureURL(it) != nil {
                let t0 = Date(), photos = ((try? FileManager.default.contentsOfDirectory(atPath: Library.captureURL(it)!.appendingPathComponent("frames").path)) ?? []).filter { $0.hasSuffix(".json") }.count
                let t = await Task.detached(priority: .userInitiated) { PhotoModel.build(it, extent: ext, detail: dense ? 2 : 1, atlas: dense ? 8192 : nil) { p, s in Task { @MainActor in builder.progress = Double(p); builder.status = s } } }.value
                ProcessChoice.learn(dense ? .dense : .standard, seconds: Date().timeIntervalSince(t0), photos: photos)
                if let t = t { geo = t.geometry(centre: centre); note = "\((t.f.count / 3).formatted()) triangles, painted from your photos" }
            }
            if geo == nil {
                let cams = Mesher.cameras(it)
                let mesh = await Task.detached { Mesher.build(cl, cams: cams) { p, _ in Task { @MainActor in builder.progress = Double(p) } } }.value
                if let m = mesh { geo = m.geometry(); note = "\(m.triangles.formatted()) triangles, coloured from the scan" }
            }
            if let g = geo {
                let scene = SCNScene(); scene.rootNode.addChildNode(SCNNode(geometry: g))
                let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(it.id.uuidString)-mesh.usdz")
                if scene.write(to: url, options: nil, delegate: nil, progressHandler: nil), let saved = Library.shared.addModel(from: url, name: it.name + " model") {
                    builder.result = saved; builder.status = note
                } else { builder.status = "The model was made but could not be saved." }
            } else { builder.status = "Not enough points for a model." }
        }
        step = 2

        // 3. AI name (skipped quietly if there is no key or no answer)
        do {
            let t = try await AITools.describe(it)
            if let d = t.data(using: .utf8), let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any], let n = j["name"] as? String, !n.isEmpty {
                Library.shared.rename(it, to: n)
                if let m = builder.result { Library.shared.rename(m, to: n + " model") }
                if let s = j["description"] as? String { Library.shared.setNotes(it, s) }
                nameText = n
            }
        } catch { nameText = error.localizedDescription }
        // 5.5 OMNI: then Apple's photogrammetry from the same 4K photos and depth (the engine of Pro object) - after the
        // photoreal model, so the two never compete for memory, and after the name, so the model gets it too
        if mac {
            let name = (Library.shared.items.first { $0.id == it.id }?.name ?? it.name) + " photogrammetry"
            if let cap = Library.captureURL(it) { apple.buildFromCapture(cap, name: name, masking: false, maxPhotos: 120) }
            else { apple.status = "This scan kept no photos, so there is nothing to build from"; Log.add("photogrammetry skipped: no capture folder") }
        }
        step = 3
        // the polish waited for this: OMNI already fixed drift and doubles; the other modes get it now, on the finished scan
        Library.shared.finishing.remove(it.id)
        if omni { Library.shared.setPolished(it) }
        // 5.7 (48): only once the queued writes have landed, or the polish reads points.bin from before the clean-up
        else { Library.io.async { Task { @MainActor in if let now = Library.shared.items.first(where: { $0.id == it.id }) { Polish.start(now) } } } }
    }
}
