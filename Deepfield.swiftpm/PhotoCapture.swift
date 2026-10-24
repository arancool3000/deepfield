import SwiftUI
import ARKit
import SceneKit
import CoreImage

/// 5.0 Photos -> 3D, captured in the app: manual, automatic (by how far the phone moved) or timed photos at a chosen
/// quality. Every photo keeps its LiDAR depth and exact position, so the model is built on the phone with no upload:
/// Apple's Object Capture (with depth) for objects, or Deepfield's LiDAR-shaped, photo-painted model for rooms.
final class PhotoCaptureSession: NSObject, ObservableObject, ARSessionDelegate {
    enum Mode: String, CaseIterable { case manual = "Manual", auto = "Auto", timed = "Timed" }
    @Published var count = 0
    @Published var ready = false
    @Published var hint = "Point at what you want to model"
    @Published var lastThumb: UIImage?
    @Published var running = false { didSet { Awake.set(Awake.key(self, "photos"), running) } }
    deinit { Awake.set(Awake.key(self, "photos"), false) }
    var mode: Mode = .auto
    var spacing: Float = 0.08                     // auto: metres moved between photos (or ~8 degrees turned)
    var interval: TimeInterval = 1                // timed: seconds between photos
    var longSide: CGFloat = 2560                  // quality
    var maxPhotos = 200
    private(set) var folder: URL?
    private var lastT: simd_float4x4?
    private var lastShot: TimeInterval = 0
    private var sharpHist: [Float] = []
    private var pendingManual = false
    private let io = DispatchQueue(label: "photo3d.io", qos: .userInitiated)
    private var writing = 0
    weak var session: ARSession?

    func start(_ s: ARSession) {
        session = s; s.delegate = self
        let c = ARWorldTrackingConfiguration()
        var sem: ARConfiguration.FrameSemantics = []
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) { sem.insert(.smoothedSceneDepth) }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) { sem.insert(.sceneDepth) }
        c.frameSemantics = sem
        if #available(iOS 16.0, *), let f = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution { c.videoFormat = f }
        s.run(c)
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("captures").appendingPathComponent("photos-" + UUID().uuidString)
        try? FileManager.default.createDirectory(at: d.appendingPathComponent("frames"), withIntermediateDirectories: true)
        folder = d
    }
    func shutter() { pendingManual = true }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let ok = (frame.smoothedSceneDepth ?? frame.sceneDepth) != nil
        if ok != ready { DispatchQueue.main.async { self.ready = ok } }
        guard ok, count < maxPhotos, writing < 3 else { return }
        if case .limited = frame.camera.trackingState { set("Move a little slower - the phone is finding its place"); return }
        let t = frame.timestamp, T = frame.camera.transform
        var due = false
        switch mode {
        case .manual: due = pendingManual
        case .timed: due = running && t - lastShot >= interval
        case .auto:
            guard running else { break }
            if let L = lastT {
                let moved = simd_length(SIMD3(T.columns.3.x - L.columns.3.x, T.columns.3.y - L.columns.3.y, T.columns.3.z - L.columns.3.z))
                let f1 = -SIMD3(T.columns.2.x, T.columns.2.y, T.columns.2.z), f0 = -SIMD3(L.columns.2.x, L.columns.2.y, L.columns.2.z)
                let turned = acos(max(-1, min(1, simd_dot(f1, f0))))
                due = moved >= spacing || turned >= spacing * 1.7          // 8 cm ~ 8 degrees
                if !due { set("Keep moving around it slowly") }
            } else { due = true }
        }
        guard due else { return }
        // blur check: a photo far softer than the recent ones is skipped (the next sharp frame is taken instead)
        let sharp = frameSharpness(frame.capturedImage)
        sharpHist.append(sharp); if sharpHist.count > 60 { sharpHist.removeFirst() }
        let med = sharpHist.sorted()[sharpHist.count / 2]
        if sharpHist.count >= 8 && sharp < med * 0.55 { set("Hold steady - that one was blurry"); return }
        pendingManual = false; lastShot = t; lastT = T
        save(frame, sharp: sharp)
    }
    private func set(_ s: String) { if hint != s { DispatchQueue.main.async { self.hint = s } } }

    private func save(_ f: ARFrame, sharp: Float) {
        guard let dir = folder, let dep = f.smoothedSceneDepth ?? f.sceneDepth,
              let dData = copyPlane(dep.depthMap, bytesPerPixel: 4), let cData = dep.confidenceMap.flatMap({ copyPlane($0, bytesPerPixel: 1) }) else { return }
        let idx = count
        DispatchQueue.main.async { self.count += 1; self.hint = "\(self.count) photo\(self.count == 1 ? "" : "s")" }
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        let img = f.capturedImage, T = f.camera.transform, K = f.camera.intrinsics
        let dw = CVPixelBufferGetWidth(dep.depthMap), dh = CVPixelBufferGetHeight(dep.depthMap)
        let long = longSide
        writing += 1
        io.async {
            defer { DispatchQueue.main.async { self.writing -= 1 } }
            let name = String(format: "%06d", idx), frames = dir.appendingPathComponent("frames")
            var ci = CIImage(cvPixelBuffer: img)
            let iw = ci.extent.width, ih = ci.extent.height
            let s = min(1, long / max(iw, ih))
            if s < 1 { ci = ci.transformed(by: CGAffineTransform(scaleX: s, y: s)) }
            let q = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
            guard let jpg = sharedCI.jpegRepresentation(of: ci, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, options: [q: 0.9]) else { return }
            try? jpg.write(to: frames.appendingPathComponent(name + ".jpg"))
            try? dData.write(to: frames.appendingPathComponent(name + ".depth"))
            try? cData.write(to: frames.appendingPathComponent(name + ".conf"))
            var tm: [Double] = []; for c in 0..<4 { for r in 0..<4 { tm.append(Double(T[c][r])) } }
            var km: [Double] = []; for c in 0..<3 { for r in 0..<3 { km.append(Double(K[c][r])) } }
            let fs = Double(s); km[0] *= fs; km[4] *= fs; km[6] *= fs; km[7] *= fs
            let meta: [String: Any] = ["transform": tm, "intrinsics": km, "imageWidth": Int((iw * s).rounded()), "imageHeight": Int((ih * s).rounded()),
                                       "depthWidth": dw, "depthHeight": dh, "timestamp": f.timestamp, "sharpness": Double(sharp)]
            if let j = try? JSONSerialization.data(withJSONObject: meta) { try? j.write(to: frames.appendingPathComponent(name + ".json")) }
            let th = UIImage(ciImage: ci.transformed(by: CGAffineTransform(scaleX: 160 / max(ci.extent.width, 1), y: 160 / max(ci.extent.width, 1))))
            let r = UIGraphicsImageRenderer(size: th.size).image { _ in th.draw(at: .zero) }
            DispatchQueue.main.async { self.lastThumb = r }
        }
    }
    func discard() { if let d = folder { try? FileManager.default.removeItem(at: d) } }
}

struct PhotoCaptureARView: UIViewRepresentable {
    let cap: PhotoCaptureSession
    func makeUIView(context: Context) -> ARSCNView { let v = ARSCNView(frame: .zero); v.antialiasingMode = .none; cap.start(v.session); return v }
    func updateUIView(_ v: ARSCNView, context: Context) {}
    static func dismantleUIView(_ v: ARSCNView, coordinator: ()) { v.session.pause() }
}

struct PhotoCaptureView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var cap = PhotoCaptureSession()
    @StateObject private var builder = ModelBuilder()
    @State private var mode: PhotoCaptureSession.Mode = .auto
    @State private var quality = 1                 // 0 fast 1600, 1 high 2560, 2 max (full 4K frame)
    @State private var spacing = 1                 // 0 dense, 1 normal, 2 sparse
    @State private var interval = 1.0
    @State private var building = false
    @State private var buildNote = ""
    @State private var buildProgress: Double = 0
    @State private var preview: URL?
    @State private var askEngine = false

    var body: some View {
        ZStack {
            PhotoCaptureARView(cap: cap).ignoresSafeArea()
            VStack(spacing: 10) {
                HStack {
                    Button { if !building { cap.discard(); dismiss() } } label: { Image(systemName: "xmark").font(.title3.bold()).frame(width: 44, height: 44).background(.ultraThinMaterial, in: Circle()) }
                    Spacer()
                    Text("\(cap.count) photo\(cap.count == 1 ? "" : "s")").font(.headline.monospacedDigit()).padding(.horizontal, 12).padding(.vertical, 8).background(.ultraThinMaterial, in: Capsule())
                    Spacer()
                    if let t = cap.lastThumb { Image(uiImage: t).resizable().scaledToFill().frame(width: 44, height: 44).clipShape(RoundedRectangle(cornerRadius: 8)) }
                }.padding(.horizontal)
                Spacer()
                if building {
                    VStack(spacing: 8) { Text(buildNote).font(.subheadline); ProgressView(value: buildProgress).frame(width: 240) }
                        .padding(14).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                } else {
                    Text(cap.hint).font(.footnote).padding(10).background(.ultraThinMaterial, in: Capsule())
                    settings
                    HStack(spacing: 26) {
                        Color.clear.frame(width: 64, height: 1)
                        Button { shutterTapped() } label: {
                            ZStack {
                                Circle().stroke(.white, lineWidth: 4).frame(width: 78, height: 78)
                                if mode != .manual && cap.running { RoundedRectangle(cornerRadius: 7).fill(Color.red).frame(width: 30, height: 30) }
                                else { Circle().fill(mode == .manual ? Color.white : Color.red).frame(width: 62, height: 62) }
                            }
                        }.disabled(!cap.ready)
                        Button { cap.running = false; askEngine = true } label: { Text("Build").bold().frame(width: 64, height: 44) }
                            .buttonStyle(.borderedProminent).disabled(cap.count < 12)
                    }.padding(.bottom, 26)
                }
            }
        }
        .background(Color.black)
        .confirmationDialog("Make the 3D model", isPresented: $askEngine, titleVisibility: .visible) {
            Button("Object - Apple Object Capture with depth") { buildApple() }
            Button("Room or scene - LiDAR shape, photo texture") { buildDeepfield() }
        } message: { Text("\(cap.count) photos. On home Wi-Fi the object model is built by your iMac at full detail.") }
        .fullScreenCover(item: $preview) { ModelPreview(url: $0) }
        .onChange(of: builder.progress) { _, p in buildProgress = p }
        .onChange(of: builder.status) { _, s in if building { buildNote = s } }
        .onChange(of: builder.result) { _, r in if let r { building = false; preview = Library.modelURL(r.id) } }
        // 5.6: a build that ends without a model gives the screen back (it stayed on "building" with every button off)
        .onChange(of: builder.running) { _, r in if !r && builder.result == nil && building { building = false; cap.hint = builder.status.isEmpty ? "The model could not be built" : builder.status } }
    }

    private var settings: some View {
        VStack(spacing: 8) {
            Picker("", selection: $mode) { ForEach(PhotoCaptureSession.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented)
                .onChange(of: mode) { _, m in cap.mode = m; cap.running = false }
            HStack {
                Text("Quality").font(.caption)
                Picker("", selection: $quality) { Text("Fast").tag(0); Text("High").tag(1); Text("Max").tag(2) }.pickerStyle(.segmented)
                    .onChange(of: quality) { _, q in cap.longSide = [1600, 2560, 4096][q] }
            }
            if mode == .auto {
                HStack {
                    Text("Spacing").font(.caption)
                    Picker("", selection: $spacing) { Text("Dense").tag(0); Text("Normal").tag(1); Text("Sparse").tag(2) }.pickerStyle(.segmented)
                        .onChange(of: spacing) { _, s in cap.spacing = [0.04, 0.08, 0.15][s] }
                }
            } else if mode == .timed {
                HStack {
                    Text("Every").font(.caption)
                    Picker("", selection: $interval) { Text("0.5 s").tag(0.5); Text("1 s").tag(1.0); Text("2 s").tag(2.0); Text("3 s").tag(3.0) }.pickerStyle(.segmented)
                        .onChange(of: interval) { _, v in cap.interval = v }
                }
            }
            Text(mode == .manual ? "Tap the shutter for each photo. Overlap each photo with the last by about two thirds." :
                 (mode == .auto ? "Tap to start, then walk slowly around it: a photo is taken every time you have moved enough." : "Tap to start: a photo every \(interval == 0.5 ? "half second" : "\(Int(interval)) s"). Move slowly between them."))
                .font(.caption2).multilineTextAlignment(.center)
        }
        .padding(12).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16)).padding(.horizontal, 12)
    }

    private func shutterTapped() {
        if mode == .manual { cap.shutter() } else { cap.running.toggle() }
    }
    private func buildApple() {
        guard let dir = cap.folder else { return }
        building = true; buildNote = "Getting the photos ready..."
        builder.buildFromCapture(dir, name: "Photo model " + Date().formatted(date: .abbreviated, time: .shortened), masking: true, maxPhotos: min(200, cap.count))
        if !builder.running { building = false; cap.hint = builder.status }          // refused at once (too few usable photos)
    }
    private func buildDeepfield() {
        guard let dir = cap.folder else { return }
        building = true; buildNote = "Building the model..."
        let it = ScanItem(name: "Photo model", capture: dir.lastPathComponent)
        Task.detached(priority: .userInitiated) {
            // the extent of the scene from the photos' own depth
            var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
            for k in Keyframe.all(in: dir) { let p = SIMD3<Float>(k.T.columns.3.x, k.T.columns.3.y, k.T.columns.3.z); lo = simd_min(lo, p - 1); hi = simd_max(hi, p + 1) }
            let ext = max(0.5, simd_reduce_max(hi - lo))
            let t = PhotoModel.build(it, extent: ext, detail: 1) { p, s in Task { @MainActor in buildProgress = Double(p); buildNote = s } }
            let url: URL? = t.flatMap { tm in
                let scene = SCNScene(); scene.rootNode.addChildNode(SCNNode(geometry: tm.geometry(centre: (lo + hi) / 2)))
                let u = FileManager.default.temporaryDirectory.appendingPathComponent("photo-model.usdz")
                try? FileManager.default.removeItem(at: u)
                return scene.write(to: u, options: nil, delegate: nil, progressHandler: nil) ? u : nil
            }
            await MainActor.run {
                building = false
                guard let u = url, let saved = Library.shared.addModel(from: u, name: "Photo model " + Date().formatted(date: .abbreviated, time: .shortened)) else { cap.hint = "Could not build a model from these photos - take more, with more overlap"; return }
                preview = Library.modelURL(saved.id)
            }
        }
    }
}
