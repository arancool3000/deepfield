import SwiftUI
import SceneKit
import simd

/// 5.6 (38) how a finished scan is processed (picked on the review screen after Done)
enum ProcessChoice: String, CaseIterable, Identifiable {
    case fast, standard, dense, imac
    var id: String { rawValue }
    var title: String { ["fast": "Fast", "standard": "Default", "dense": "Dense", "imac": "On the iMac"][rawValue]! }
    var detail: String {
        switch self {
        case .fast: return "Open the point cloud now. No model, nothing removed."
        case .standard: return "This mode's usual finishing: clean-up, photoreal model, AI name."
        case .dense: return "Finer model and an 8K photo texture. Slower."
        case .imac: return "Default, plus a full-detail photogrammetry model built on the iMac (phone if it is away)."
        }
    }
    /// seconds per photo, learned from the last real runs on this phone (a rough first guess until then)
    private var guess: Double { ["fast": 0, "standard": 0.8, "dense": 2.2, "imac": 4][rawValue]! }
    var secondsPerPhoto: Double { UserDefaults.standard.object(forKey: "secPerPhoto." + rawValue) as? Double ?? guess }
    var measured: Bool { UserDefaults.standard.object(forKey: "secPerPhoto." + rawValue) != nil }
    /// a running average, so one odd run does not throw the estimate
    static func learn(_ c: ProcessChoice, seconds: Double, photos: Int) {
        guard photos >= 5, seconds > 1 else { return }
        let s = seconds / Double(photos), old = c.measured ? c.secondsPerPhoto : s
        UserDefaults.standard.set(old * 0.6 + s * 0.4, forKey: "secPerPhoto." + c.rawValue)
    }
    func estimate(photos: Int) -> String {
        if self == .fast { return "a few seconds" }
        var s = 20 + secondsPerPhoto * Double(photos)                       // + clean-up and naming
        if self == .imac { s += ProcessChoice.standard.secondsPerPhoto * Double(photos) }
        let t = s < 90 ? "\(Int((s / 10).rounded(.up)) * 10) s" : "\(Int((s / 60).rounded())) min"
        return (measured ? "about " : "roughly ") + t
    }
}
/// read by the finishing screen (set just before a scan is saved)
var PROCESS_CHOICE: ProcessChoice = .standard

/// the scan as it is, with a small camera at every photo; Extend goes back to scanning, Process saves it
struct ScanReview: View {
    @ObservedObject var scanner: Scanner
    let photos: Bool                 // this mode makes models from its photos
    let extending: Bool              // carrying on a saved scan: it merges, there is nothing to pick
    let done: (ProcessChoice?) -> Void
    @State private var choice: ProcessChoice = .standard
    @State private var showCams = true

    var body: some View {
        let n = scanner.keyframes
        ZStack(alignment: .bottom) {
            ReviewView(scanner: scanner, showCams: showCams).ignoresSafeArea()
            VStack(spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Review").font(.headline)
                        Text("\(scanner.count.formatted()) points · \(n) photos · \(String(format: "%.1f", scanner.areaM2)) m²").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if n > 0 { Toggle(isOn: $showCams) { Image(systemName: "camera") }.toggleStyle(.button) }
                }
                if !extending {
                    VStack(spacing: 6) {
                        ForEach(ProcessChoice.allCases) { c in
                            let off = c != .fast && c != .standard && (!photos || n < 12)
                            Button { choice = c } label: {
                                HStack(alignment: .top, spacing: 10) {
                                    Image(systemName: choice == c ? "largecircle.fill.circle" : "circle").foregroundStyle(choice == c ? Color.accentColor : .secondary)
                                    VStack(alignment: .leading, spacing: 1) {
                                        HStack { Text(c.title).font(.subheadline.bold()); Spacer(); Text(c.estimate(photos: n)).font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                                        Text(off ? (photos ? "Needs at least 12 photos from this scan" : "Not available for this scan mode") : c.detail)   // 5.7 (46): say why, not always photos
                                            .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.leading)   // 5.7 (48): these modifiers were inside the comment
                                    }
                                }.contentShape(Rectangle())
                            }.buttonStyle(.plain).disabled(off).opacity(off ? 0.45 : 1)
                        }
                    }
                }
                HStack(spacing: 10) {
                    Button { done(nil) } label: { Label("Extend", systemImage: "plus.viewfinder").frame(maxWidth: .infinity) }
                        .buttonStyle(.bordered).controlSize(.large)
                    Button { done(extending ? .standard : choice) } label: { Label(extending ? "Save" : "Process", systemImage: "checkmark").frame(maxWidth: .infinity) }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                }
                Text(scanner.count > 1_000_000 ? "Big scan: the points appear in a few seconds. Extend carries on scanning; nothing is lost." : "Extend carries on scanning; nothing is lost.").font(.caption2).foregroundStyle(.secondary)
            }
            .padding(16).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22)).padding()
        }
        .background(Color.black)
    }
}

/// points plus a small white camera (frustum) at every photo, both centred the same way
private struct ReviewView: UIViewRepresentable {
    let scanner: Scanner
    let showCams: Bool
    final class Coord { let cloud = SCNNode(), cams = SCNNode(); var built = false; var loading = false; var camCentre = SIMD3<Float>.zero }
    func makeCoordinator() -> Coord { Coord() }
    func makeUIView(context: Context) -> SCNView {
        let v = SCNView(frame: .zero); v.backgroundColor = .black; v.allowsCameraControl = true; v.antialiasingMode = .multisampling4X
        let scene = SCNScene(); v.scene = scene
        scene.rootNode.addChildNode(context.coordinator.cloud); scene.rootNode.addChildNode(context.coordinator.cams)
        let cam = SCNNode(); cam.camera = SCNCamera(); cam.camera?.zNear = 0.01; cam.position = SCNVector3(0, 0.6, 2.6)
        cam.look(at: SCNVector3Zero); scene.rootNode.addChildNode(cam); v.pointOfView = cam
        let co = context.coordinator, sc = scanner, poses = scanner.photoPoses
        // 5.8 (53): the photo cameras show at once - the points wait behind the pause's merge (11 s on a big scan, which
        // left the review empty unless the scan was small, e.g. a continued one)
        if !poses.isEmpty {
            var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
            for T in poses { let p = SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z); lo = simd_min(lo, p); hi = simd_max(hi, p) }
            let c = (lo + hi) / 2, reach = max(0.8, simd_length(hi - lo) * 1.5)
            co.cams.geometry = ReviewView.frusta(poses, centre: c, size: max(0.03, reach * 0.02)); co.camCentre = c
            cam.position = SCNVector3(0, reach * 0.35, reach * 1.1); cam.look(at: SCNVector3Zero)
        }
        co.loading = true
        DispatchQueue.global(qos: .userInitiated).async {
            let (all, cols) = sc.previewPoints(max: 300_000)
            guard !all.isEmpty else { return }
            var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
            for p in all { lo = simd_min(lo, p); hi = simd_max(hi, p) }
            let c = (lo + hi) / 2, step = 1
            var verts: [SCNVector3] = [], cf: [Float] = []
            verts.reserveCapacity(all.count / step + 1); cf.reserveCapacity(3 * (all.count / step + 1))
            for i in stride(from: 0, to: all.count, by: step) {
                let p = all[i] - c, q = cols[i]
                verts.append(SCNVector3(p.x, p.y, p.z)); cf += [Float(q.x) / 255, Float(q.y) / 255, Float(q.z) / 255]
            }
            let g = pointGeometry(verts, cf, size: 2)
            let f = ReviewView.frusta(poses, centre: c, size: max(0.03, simd_length(hi - lo) * 0.02))
            // the view starts back from the scan, whatever its size
            let reach = max(0.5, simd_length(hi - lo))
            DispatchQueue.main.async {
                co.loading = false
                co.cloud.geometry = g; co.cams.geometry = f
                cam.position = SCNVector3(0, reach * 0.35, reach * 1.1); cam.look(at: SCNVector3Zero)
            }
        }
        return v
    }
    func updateUIView(_ v: SCNView, context: Context) { context.coordinator.cams.isHidden = !showCams }
    /// the preview's memory goes back before the save starts
    static func dismantleUIView(_ v: SCNView, coordinator: Coord) { coordinator.cloud.geometry = nil; coordinator.cams.geometry = nil; v.scene = nil }

    /// one pyramid of lines per photo: apex at the camera, a 4:3 frame in front (ARKit cameras look down -z)
    static func frusta(_ poses: [simd_float4x4], centre: SIMD3<Float>, size: Float) -> SCNGeometry? {
        guard !poses.isEmpty else { return nil }
        var v: [SCNVector3] = [], idx: [UInt32] = []
        let w = size * 0.67, h = size * 0.5, d = size
        let local: [SIMD3<Float>] = [.zero, SIMD3(-w, -h, -d), SIMD3(w, -h, -d), SIMD3(w, h, -d), SIMD3(-w, h, -d)]
        let edges: [(UInt32, UInt32)] = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        for T in poses {
            let b = UInt32(v.count)
            for p in local { let q = T * SIMD4(p, 1); v.append(SCNVector3(q.x - centre.x, q.y - centre.y, q.z - centre.z)) }
            for (a, e) in edges { idx += [b + a, b + e] }
        }
        let g = SCNGeometry(sources: [SCNGeometrySource(vertices: v)], elements: [SCNGeometryElement(indices: idx, primitiveType: .line)])
        let m = SCNMaterial(); m.diffuse.contents = UIColor.white; m.lightingModel = .constant; g.materials = [m]
        return g
    }
}
