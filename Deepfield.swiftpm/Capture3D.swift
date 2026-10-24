import SwiftUI
import RealityKit
import PhotosUI
import QuickLook
import os
import ImageIO

/// Turns photos into a textured USDZ on the phone (Apple's Object Capture reconstruction). It asks for the highest
/// detail first and steps down only if this phone refuses it: full, then medium, then reduced.
@MainActor
final class ModelBuilder: ObservableObject {
    @Published var progress: Double = 0
    @Published var status = ""
    // 5.6 (38): a running build keeps itself alive. It lived only as long as the screen that started it, so closing
    // OMNI's Finishing screen dropped the job and its phone fallback (a weak reference) silently did nothing
    private static var alive: [ObjectIdentifier: ModelBuilder] = [:]
    @Published var running = false { didSet { Awake.set(Awake.key(self, "model"), running); ModelBuilder.alive[ObjectIdentifier(self)] = running ? self : nil } }
    deinit { Awake.set(Awake.key(self, "model"), false) }
    @Published var result: ScanItem?
    @Published var detailUsed = ""

    // iOS builds models at "reduced" detail only (full / medium exist on macOS); the quality comes from the photos and depth
    static let details: [(PhotogrammetrySession.Request.Detail, String)] = [(.reduced, "phone")]

    /// a folder of photos (Photos to 3D, Pro Object Scan: their HEICs carry depth and gravity already)
    /// 5.7 (51): the photos' own folder under Documents/objects, linked to the model so they are kept with it
    private var photoSet: String?
    func build(images: URL, name: String, masking: Bool = false, macImages: URL? = nil) {
        let parent = images.deletingLastPathComponent()
        photoSet = parent.deletingLastPathComponent().lastPathComponent == "objects" ? parent.lastPathComponent : (images.deletingLastPathComponent().lastPathComponent == "objects" ? images.lastPathComponent : nil)
        // 5.6: never build from a handful of photos (a Pro room capture with 8 made a useless model) - say so instead
        let n = ((try? FileManager.default.contentsOfDirectory(atPath: images.path)) ?? []).filter { ["heic", "jpg", "jpeg", "png"].contains(($0 as NSString).pathExtension.lowercased()) }.count
        guard n >= 10 else { status = "Only \(n) photos were taken - too few for a model. Capture again and keep moving until the photo bar is full."; Log.add("model refused: \(n) photos"); return }
        viaMac(macImages ?? images, name: name, masking: masking) { [weak self] in
            self?.start(name: name, masking: masking) { cfg in try PhotogrammetrySession(input: images, configuration: cfg) }
        }
    }

    @Published var onMac = false
    /// 5.5: on home Wi-Fi with the iMac on, build there at full detail; anything going wrong builds on the phone instead
    private func viaMac(_ folder: URL, name: String, masking: Bool, cleanup: URL? = nil, phone: @escaping () -> Void) {
        // 5.7 (46): a new build forgets the last one's model - a stale result kept screens on "building" after a failure
        running = true; progress = 0; result = nil; status = "Looking for the iMac..."
        Task {
            // running stays true through the hand-over to the phone (screens read a false as "it failed")
            guard await MacWorker.available() else { if let c = cleanup { try? FileManager.default.removeItem(at: c) }; phone(); return }
            onMac = true
            do {
                let u = try await MacWorker.build(folder: folder, name: name, mask: masking) { [weak self] p, s in self?.progress = p; self?.status = s }
                if let c = cleanup { try? FileManager.default.removeItem(at: c) }
                detailUsed = "full (iMac)"
                if let it = Library.shared.addModel(from: u, name: name, photoSet: photoSet) { result = it; status = "Done - full detail, built on the iMac" }
                else { status = "The model was made but could not be saved." }
                Log.add("model built on the iMac"); running = false; onMac = false
            } catch {
                Log.add("iMac build failed: \(error.localizedDescription)")
                if let c = cleanup { try? FileManager.default.removeItem(at: c) }
                onMac = false
                status = "The iMac could not build it (\(error.localizedDescription)) - building on the phone instead"
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                phone()
            }
        }
    }

    /// a LiDAR scan's capture: the sharpest, best-spread photos, each with its LiDAR depth map
    func buildFromCapture(_ cap: URL, name: String, masking: Bool, maxPhotos: Int = 50) {
        // 5.6: shown at once, and the reading of hundreds of photos happens off the main thread (OMNI showed this step
        // as an empty circle with no word of what happened)
        running = true; progress = 0; result = nil; status = "Getting the photos ready..."
        Task {
            let found: ([Keyframe], Int) = await Task.detached { let all = Keyframe.all(in: cap); return (ModelBuilder.pick(all, maxPhotos), all.count) }.value
            self.buildFromPicks(found.0, all: found.1, name: name, masking: masking)
        }
    }
    private func buildFromPicks(_ picks: [Keyframe], all: Int, name: String, masking: Bool) {
        Log.add("photogrammetry: \(picks.count) of \(all) photos picked")
        guard picks.count >= 12 else { running = false; status = all == 0 ? "No photos in this scan - photogrammetry needs the scan's photos" : "Only \(picks.count) usable photos - scan longer for a model"; Log.add("photogrammetry skipped: \(picks.count) of \(all) photos usable"); return }
        let phone: () -> Void = { [weak self] in
            self?.start(name: name, masking: masking) { cfg in
                try PhotogrammetrySession(input: picks.enumerated().lazy.compactMap { ModelBuilder.sample($0.offset, $0.element) }, configuration: cfg)
            }
        }
        // the iMac gets the same photos as a folder (links, no copies), each with its LiDAR depth
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mac-" + UUID().uuidString)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for (i, k) in picks.enumerated() {
                let n = String(format: "%04d", i)
                try FileManager.default.linkItem(at: k.jpg, to: dir.appendingPathComponent(n + ".jpg"))
                // the LiDAR depth, confidence and pose go too: the iMac builds with depth, so the model has its real size
                for ext in ["depth", "conf", "json"] {
                    let f = k.jpg.deletingPathExtension().appendingPathExtension(ext)
                    if FileManager.default.fileExists(atPath: f.path) { try? FileManager.default.linkItem(at: f, to: dir.appendingPathComponent(n + "." + ext)) }
                }
            }
        } catch { try? FileManager.default.removeItem(at: dir); phone(); return }
        viaMac(dir, name: name, masking: masking, cleanup: dir, phone: phone)
    }

    private func start(name: String, masking: Bool, make: @escaping (PhotogrammetrySession.Configuration) throws -> PhotogrammetrySession) {
        guard PhotogrammetrySession.isSupported else { status = "This iPhone cannot build 3D models on the device."; running = false; return }
        running = true; progress = 0; status = "Building the 3D model..."
        Task {
            // 5.5: building needs room. Right after another model (OMNI builds the photoreal one first) memory comes back
            // over a few seconds: WAIT for it instead of giving up at once (that read as "skipped immediately")
            var waited = 0
            while os_proc_available_memory() < 900_000_000 && waited < 90 {
                status = "Waiting for memory to free up (\(os_proc_available_memory() / 1_000_000) MB free)..."
                try? await Task.sleep(nanoseconds: 1_000_000_000); waited += 1
            }
            if os_proc_available_memory() < 900_000_000 {
                status = "Not enough free memory to build a model right now - close other apps and try again."
                Log.add("photogrammetry skipped: \(os_proc_available_memory() / 1_000_000) MB free"); running = false; return
            }
            var cfg = PhotogrammetrySession.Configuration()
            cfg.featureSensitivity = .normal          // .high roughly doubles the time
            cfg.sampleOrdering = .sequential
            cfg.isObjectMaskingEnabled = masking
            for (detail, label) in ModelBuilder.details {
                let out = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).usdz")
                status = "Building the 3D model (\(label) detail)..."; progress = 0
                if await run(make, cfg, detail, out) {
                    detailUsed = label
                    if let it = Library.shared.addModel(from: out, name: name, photoSet: photoSet) { result = it; status = "Done - \(label) detail" }
                    else { status = "The model was made but could not be saved." }
                    Log.add("model built (\(label))"); running = false; return
                }
            }
            running = false
            if result == nil && !status.hasPrefix("Could not") { status = "Could not build the model." }
        }
    }

    /// one attempt at one detail level; false if the phone refused it or it failed
    private func run(_ make: (PhotogrammetrySession.Configuration) throws -> PhotogrammetrySession, _ cfg: PhotogrammetrySession.Configuration,
                     _ detail: PhotogrammetrySession.Request.Detail, _ out: URL) async -> Bool {
        do {
            let s = try make(cfg)
            try s.process(requests: [.modelFile(url: out, detail: detail)])
            var ok = false
            for try await o in s.outputs {
                switch o {
                case .requestProgress(_, let f): progress = f
                case .requestComplete(_, _): ok = true
                case .requestError(_, let e): status = "Could not build at this detail: \(e.localizedDescription)"; Log.add("photogrammetry error: \(e)")
                case .processingComplete, .processingCancelled: return ok && FileManager.default.fileExists(atPath: out.path)
                default: break
                }
            }
            return ok && FileManager.default.fileExists(atPath: out.path)
        } catch { status = "Could not build the model: \(error.localizedDescription)"; Log.add("photogrammetry failed: \(error)"); return false }
    }

    /// sharpest first, but never two photos from almost the same spot; then back in capture order
    nonisolated static func pick(_ all: [Keyframe], _ n: Int) -> [Keyframe] {
        let sorted = all.sorted { $0.sharp > $1.sharp }
        let median = sorted.isEmpty ? 0 : sorted[sorted.count / 2].sharp
        var out: [Keyframe] = []
        for k in sorted where k.sharp >= median * 0.5 {
            let p = SIMD3<Float>(k.T.columns.3.x, k.T.columns.3.y, k.T.columns.3.z), f = -SIMD3<Float>(k.T.columns.2.x, k.T.columns.2.y, k.T.columns.2.z)
            let close = out.contains { o in
                let q = SIMD3<Float>(o.T.columns.3.x, o.T.columns.3.y, o.T.columns.3.z), g = -SIMD3<Float>(o.T.columns.2.x, o.T.columns.2.y, o.T.columns.2.z)
                return simd_length(p - q) < 0.02 && simd_dot(f, g) > 0.9986      // 2 cm and 3 degrees
            }
            if !close { out.append(k) }
            if out.count >= n { break }
        }
        return out.sorted { $0.id < $1.id }
    }

    /// one photo + its LiDAR depth as an Object Capture sample
    nonisolated static func sample(_ id: Int, _ k: Keyframe) -> PhotogrammetrySample? {
        guard var ci = CIImage(contentsOf: k.jpg) else { return nil }
        let long = max(ci.extent.width, ci.extent.height)
        if long > 1600 { let sc = 1600 / long; ci = ci.transformed(by: CGAffineTransform(scaleX: sc, y: sc)) }     // plenty for "reduced" detail, far quicker
        let w = Int(ci.extent.width), h = Int(ci.extent.height)
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess, let img = pb else { return nil }
        sharedCI.render(ci, to: img)
        var s = PhotogrammetrySample(id: id, image: img)
        let base = k.jpg.deletingPathExtension()
        if k.dw > 0, let d = try? Data(contentsOf: base.appendingPathExtension("depth")), d.count >= k.dw * k.dh * 4 {
            let conf = try? Data(contentsOf: base.appendingPathExtension("conf"))
            var dpb: CVPixelBuffer?
            if CVPixelBufferCreate(nil, k.dw, k.dh, kCVPixelFormatType_DepthFloat32, attrs, &dpb) == kCVReturnSuccess, let dm = dpb {
                CVPixelBufferLockBaseAddress(dm, [])
                if let base = CVPixelBufferGetBaseAddress(dm) {
                    let row = CVPixelBufferGetBytesPerRow(dm)
                    d.withUnsafeBytes { src in
                        for y in 0..<k.dh {
                            let dst = base.advanced(by: y * row).assumingMemoryBound(to: Float32.self)
                            for x in 0..<k.dw {
                                var v = src.loadUnaligned(fromByteOffset: (y * k.dw + x) * 4, as: Float32.self)
                                if let c = conf, c.count == k.dw * k.dh, c[y * k.dw + x] == 0 { v = 0 }     // low LiDAR confidence: unknown
                                dst[x] = v
                            }
                        }
                    }
                }
                CVPixelBufferUnlockBaseAddress(dm, [])
                s.depthDataMap = dm
            }
        }
        return s
    }
}

// ---------- Pro Object Scan: Apple's guided capture, then the on-device model ----------
struct ProScanView: View {
    var area = false                          // 5.0 Pro room: Apple's Object Capture in area mode (iOS 18) for rooms and spaces
    @Environment(\.dismiss) private var dismiss
    @State private var session: ObjectCaptureSession?
    @StateObject private var builder = ModelBuilder()
    @State private var folder: URL?
    @State private var stateText = "Starting..."
    @State private var phase = 0          // 0 capture, 1 building, 2 done
    @State private var preview: URL?
    @State private var pass = 0              // 0 level, 1 high, 2 low, 3 bottom (after a flip)
    @State private var passStart = [0, 0, 0, 0]   // photo count when each ring began
    static let roomMin = 120                       // a room needs far more photos than Apple's auto-shots gave (it built from too few)
    static let roomTarget = 250
    /// 5.6 Pro room: Apple takes a photo only when it judges one new enough, which left rooms short. While capturing, ask
    /// for one every 0.6 s whenever the session can take it - Apple still refuses blurry or duplicate ones.
    private func extraShots(_ s: ObjectCaptureSession) async {
        // 5.7 (41): objects too - waiting for Apple's own shots made object rings slow
        guard s.state == .capturing else { return }
        while !Task.isCancelled, s.state == .capturing {
            if s.canRequestImageCapture { s.requestImageCapture() }
            try? await Task.sleep(nanoseconds: 600_000_000)
        }
    }
    static let perRing = 30                        // Apple's engine needs ~80-200 photos for a small object; 30 in total made a potato
    static let passTitle = ["Ring 1 of 3: level", "Ring 2 of 3: high", "Ring 3 of 3: low", "The bottom"]
    static let passHint = ["Walk all the way round with the phone level with the middle of the object.",
                           "Raise the phone and look down at it (about 45 degrees) all the way round.",
                           "Crouch and look up at it slightly, all the way round.",
                           "Walk round once more: the underside is facing you now."]
    static let nextText = ["Level ring done. Next: raise the phone and look down at it.",
                           "High ring done. Next: crouch and look up at it.",
                           "All three rings done. For a complete model, turn the object onto its side so the bottom shows, then scan once more.",
                           "Bottom done - the full object is captured."]

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if !ObjectCaptureSession.isSupported {
                unsupported
            } else if phase == 0, let s = session {
                ObjectCaptureView(session: s).ignoresSafeArea()
                VStack {
                    HStack {
                        Button("Cancel") { s.cancel(); dismiss() }.buttonStyle(.bordered)
                        Spacer()
                        Text("\(s.numberOfShotsTaken) photos").font(.headline.monospacedDigit()).padding(8).background(.ultraThinMaterial, in: Capsule())
                    }.padding()
                    Spacer()
                    // 5.6: while capturing, only a slim bar - the big panels covered the live model
                    if s.state != .capturing { Text(stateText).font(.subheadline).padding(10).background(.ultraThinMaterial, in: Capsule()) }
                    controls(s).padding(.bottom, 16)
                }
                .task(id: s.state == .capturing) { await extraShots(s) }
            } else if phase >= 1 {
                buildingView
            }
        }
        .task { start() }
        .onAppear { Awake.set("procapture", true) }.onDisappear { Awake.set("procapture", false) }
        .fullScreenCover(item: $preview) { ModelPreview(url: $0) }
    }

    private var unsupported: some View {
        VStack(spacing: 14) {
            Image(systemName: "cube.transparent").font(.system(size: 50))
            Text("Pro Object Scan needs an iPhone with LiDAR running iOS 17 or later.").multilineTextAlignment(.center)
            Button("Close") { dismiss() }.buttonStyle(.borderedProminent)
        }.padding()
    }

    @ViewBuilder private func controls(_ s: ObjectCaptureSession) -> some View {
        switch s.state {
        case .ready:
            if area {
                if #available(iOS 18.0, *) {
                    Button { s.startCapturing() } label: { Text("Start capturing the room").frame(minWidth: 220) }.buttonStyle(.borderedProminent).controlSize(.large)
                } else {
                    Text("Pro room needs iOS 18 or later. Use OMNI or Room for rooms on this phone.").font(.caption).multilineTextAlignment(.center)
                }
            } else {
                Button { _ = s.startDetecting() } label: { Text("Find the object").frame(minWidth: 200) }.buttonStyle(.borderedProminent).controlSize(.large)
            }
        case .detecting:
            HStack {
                Button("Reset") { _ = s.resetDetection() }.buttonStyle(.bordered)
                Button { s.startCapturing() } label: { Text("Start capture").frame(minWidth: 160) }.buttonStyle(.borderedProminent).controlSize(.large)
            }
        case .capturing where area:
            // a slim bar: the room's live model stays visible above it
            let got = s.numberOfShotsTaken, enough = got >= ProScanView.roomMin
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(enough ? "Enough photos - more is better" : "Walls at chest height, then low and high").font(.caption.weight(.medium)).lineLimit(1)
                    ProgressView(value: Double(min(got, ProScanView.roomTarget)), total: Double(ProScanView.roomTarget)).tint(enough ? .green : .yellow)
                    Text("\(got) of \(ProScanView.roomTarget) photos").font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
                Button { s.finish() } label: { Text("Build") }.buttonStyle(.borderedProminent).disabled(!enough)
            }
            .padding(.horizontal, 12).padding(.vertical, 8).frame(maxWidth: 360).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14)).padding(.horizontal)
        case .capturing:
            // 5.0 guided: level, then high looking down, then low looking up, then (optionally) the bottom after a flip
            // 5.7 (41): a short ring never blocks. "Go round again" started a NEW scan pass, which makes Apple line the
            // object up afresh - it drifted and the model came out doubled. Photos keep coming in the same pass instead.
            if s.userCompletedScanPass {
                VStack(spacing: 10) {
                    Text(ProScanView.nextText[min(pass, 3)]).font(.subheadline.weight(.medium)).multilineTextAlignment(.center)
                    let ringGot = s.numberOfShotsTaken - passStart[min(pass, 3)]
                    if ringGot < ProScanView.perRing { Text("\(ringGot) photos in this ring - fine to carry on; photos still add while you move.").font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center) }
                    HStack {
                        if pass < 2 {
                            Button { pass += 1; passStart[pass] = s.numberOfShotsTaken; s.beginNewScanPass() } label: { Text(pass == 0 ? "Start the high ring" : "Start the low ring").frame(minWidth: 170) }.buttonStyle(.borderedProminent)
                        } else if pass == 2 {
                            Button("Skip the bottom") { s.finish() }.buttonStyle(.bordered)
                            Button { pass += 1; passStart[pass] = s.numberOfShotsTaken; s.beginNewScanPassAfterFlip() } label: { Text("I've flipped it").frame(minWidth: 140) }.buttonStyle(.borderedProminent)
                        } else {
                            Button { s.finish() } label: { Text("Finish and build").frame(minWidth: 170) }.buttonStyle(.borderedProminent)
                        }
                    }
                    if pass < 3 && s.numberOfShotsTaken >= 40 { Button("Finish now") { s.finish() }.font(.caption) }
                }
                .padding(12).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            } else {
                VStack(spacing: 4) {
                    Text(ProScanView.passTitle[min(pass, 3)]).font(.subheadline.bold())
                    Text(ProScanView.passHint[min(pass, 3)]).font(.caption2).multilineTextAlignment(.center).lineLimit(2)
                    let got = s.numberOfShotsTaken - passStart[min(pass, 3)]
                    ProgressView(value: Double(min(got, ProScanView.perRing)), total: Double(ProScanView.perRing)).frame(width: 200)
                    Text(got >= ProScanView.perRing ? "\(got) photos in this ring - enough" : "\(got) of \(ProScanView.perRing) photos in this ring").font(.caption2.monospacedDigit())
                    if s.numberOfShotsTaken >= 40 { Button("Finish early") { s.finish() }.buttonStyle(.bordered).font(.caption) }
                }.padding(.horizontal, 12).padding(.vertical, 8).frame(maxWidth: 360).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            }
        default: EmptyView()
        }
    }

    private var buildingView: some View {
        VStack(spacing: 18) {
            Image(systemName: phase == 2 ? "checkmark.circle.fill" : "cube.transparent").font(.system(size: 54)).foregroundStyle(phase == 2 ? .green : .white)
            Text(builder.status).multilineTextAlignment(.center)
            if builder.running { ProgressView(value: builder.progress).frame(width: 240); Text("\(Int(builder.progress * 100))% - keep the app open").font(.caption).foregroundStyle(.secondary) }
            if let r = builder.result {
                Button("View model") { preview = Library.modelURL(r.id) }.buttonStyle(.borderedProminent)
            }
            if !builder.running { Button("Close") { dismiss() }.buttonStyle(.bordered) }
        }
        .padding()
        .onChange(of: builder.result) { _, r in if r != nil { phase = 2 } }
    }

    private func start() {
        guard ObjectCaptureSession.isSupported, session == nil else { return }
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("objects").appendingPathComponent(UUID().uuidString)
        let imgs = base.appendingPathComponent("images"), ck = base.appendingPathComponent("checkpoint")
        try? FileManager.default.createDirectory(at: imgs, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: ck, withIntermediateDirectories: true)
        folder = imgs
        let s = ObjectCaptureSession()
        var cfg = ObjectCaptureSession.Configuration()
        cfg.checkpointDirectory = ck
        cfg.isOverCaptureEnabled = true
        s.start(imagesDirectory: imgs, configuration: cfg)
        session = s
        Task {
            for await st in s.stateUpdates {
                switch st {
                case .initializing: stateText = "Starting the camera..."
                case .ready: stateText = area ? "Stand in the room and tap Start" : "Point at the object and tap Find the object"
                case .detecting: stateText = "Adjust the box to fit the object, then Start capture"
                case .capturing: stateText = area ? "Walk slowly round the room" : "Walk slowly around the object"
                case .finishing: stateText = "Saving the photos..."
                case .completed:
                    phase = 1
                    // masking keeps only the object: the floor or table under it is cut away automatically
                    builder.build(images: imgs, name: (area ? "Room " : "Object ") + Date().formatted(date: .abbreviated, time: .shortened), masking: !area)
                case .failed(let e): stateText = "Capture failed: \(e.localizedDescription)"
                @unknown default: break
                }
            }
        }
    }
}

// ---------- Photos to 3D: any photos from the library ----------
struct PhotosTo3DView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var picks: [PhotosPickerItem] = []
    @StateObject private var builder = ModelBuilder()
    @State private var copying = false
    @State private var preview: URL?

    var body: some View {
        NavigationStack {
            VStack(spacing: 18) {
                Image(systemName: "photo.stack").font(.system(size: 50)).padding(.top, 30)
                Text("Pick 20 to 200 photos of one object, taken all the way around it with lots of overlap.")
                    .multilineTextAlignment(.center).foregroundStyle(.secondary).padding(.horizontal)
                PhotosPicker(selection: $picks, maxSelectionCount: 200, matching: .images) {
                    Label(picks.isEmpty ? "Choose photos" : "\(picks.count) photos chosen", systemImage: "photo.on.rectangle")
                        .frame(minWidth: 220)
                }.buttonStyle(.bordered).controlSize(.large)
                Button { Task { await go() } } label: { Text("Build 3D model").frame(minWidth: 220) }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(picks.count < 10 || builder.running || copying)
                if copying { ProgressView("Copying photos...") }
                if builder.running { ProgressView(value: builder.progress).frame(width: 240) }
                if !builder.status.isEmpty { Text(builder.status).font(.footnote).multilineTextAlignment(.center) }
                if let r = builder.result { Button("View model") { preview = Library.modelURL(r.id) }.buttonStyle(.borderedProminent) }
                Spacer()
            }
            .padding()
            .navigationTitle("Photos to 3D").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() }.disabled(builder.running) } }
            .fullScreenCover(item: $preview) { ModelPreview(url: $0) }
        }
    }

    private func go() async {
        copying = true
        let top = FileManager.default.temporaryDirectory.appendingPathComponent("photos-\(UUID().uuidString)")
        let dir = top.appendingPathComponent("small")            // the phone's copies; the originals sit beside them, never inside
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // 5.6: the iMac takes the ORIGINAL photos (full resolution, and the depth a Portrait or LiDAR photo carries)
        let mac = await MacWorker.available()
        let orig = top.appendingPathComponent("original")
        if mac { try? FileManager.default.createDirectory(at: orig, withIntermediateDirectories: true) }
        var n = 0
        for (i, p) in picks.enumerated() {
            // 5.0: a 48 MP photo given to the model builder at full size ran the phone out of memory (the app was
            // killed). Each photo is downsampled straight from the file to 2400 px (rotation applied) as a real JPEG.
            guard let d = try? await p.loadTransferable(type: Data.self) else { continue }
            if mac {
                let ext = p.supportedContentTypes.first?.preferredFilenameExtension ?? "heic"
                try? d.write(to: orig.appendingPathComponent(String(format: "%04d.", i) + ext))
            }
            let ok: Bool = await Task.detached(priority: .userInitiated) {
                guard let src = CGImageSourceCreateWithData(d as CFData, nil),
                      let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 2400, kCGImageSourceCreateThumbnailWithTransform: true] as CFDictionary),
                      let dst = CGImageDestinationCreateWithURL(dir.appendingPathComponent(String(format: "%04d.jpg", i)) as CFURL, "public.jpeg" as CFString, 1, nil) else { return false }
                CGImageDestinationAddImage(dst, img, [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary)
                return CGImageDestinationFinalize(dst)
            }.value
            if ok { n += 1 }
        }
        copying = false
        guard n >= 10 else { builder.status = "Only \(n) photos could be read - at least 10 are needed (20 or more works far better)."; return }
        if n < 20 { builder.status = "\(n) photos is very few - trying anyway. 20 to 60 photos with lots of overlap work best." }
        builder.build(images: dir, name: "Photo model " + Date().formatted(date: .abbreviated, time: .shortened), macImages: mac ? orig : nil)
    }
}

struct QuickLookView: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> QLPreviewController {
        let c = QLPreviewController(); c.dataSource = context.coordinator; return c
    }
    func updateUIViewController(_ c: QLPreviewController, context: Context) {}
    func makeCoordinator() -> Coord { Coord(url) }
    final class Coord: NSObject, QLPreviewControllerDataSource {
        let url: URL
        init(_ u: URL) { url = u }
        func numberOfPreviewItems(in c: QLPreviewController) -> Int { 1 }
        func previewController(_ c: QLPreviewController, previewItemAt i: Int) -> QLPreviewItem { url as NSURL }
    }
}
