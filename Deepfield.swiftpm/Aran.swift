import SwiftUI
import SceneKit

/// 5.0 Aran: Deepfield's own scene format. The base is the lifelike model - a solid surface shaped by the LiDAR depth of
/// every photo and painted seamlessly from the photos - so walls, floors and objects are solid and photo-sharp. What a
/// solid surface cannot hold (wires, leaves, hair, thin edges: scan points with no surface near them) is added as real
/// Gaussian splats, soft and true to the scan. Solid where things are solid, soft where they are fine.
struct AranScene {
    let mesh: SCNGeometry
    let detail: SCNGeometry?
    let detailCount: Int
    let triangles: Int
    let splatRadius: Float
    let textured: TexturedMesh
    let detailCloud: Cloud
    let photos: [RealPhoto]
    let centre: SIMD3<Float>
}

enum AranBuilder {
    /// 5.7 (48): one Aran build at a time in the whole app. PhotoModel.build cannot be stopped part way, so a closed and
    /// reopened Aran used to start a second multi-GB build beside the first and iOS killed the app for memory
    private final class Gate: @unchecked Sendable { let lock = NSLock(); var busy = false }
    private static let gate = Gate()
    static func claim() -> Bool { gate.lock.lock(); defer { gate.lock.unlock() }; if gate.busy { return false }; gate.busy = true; return true }
    static func release() { gate.lock.lock(); gate.busy = false; gate.lock.unlock() }
    static func build(_ item: ScanItem, cloud cl: Cloud, progress: @escaping (Float, String) -> Void) -> AranScene? {
        guard cl.count > 1000 else { return nil }
        let b = cl.bounds, ext = max(0.4, max(b.size.x, max(b.size.y, b.size.z))), centre = b.centre
        guard let t = PhotoModel.build(item, extent: ext, detail: 2, minWeight: 0.8, atlas: 8192, progress: { p, s in progress(p * 0.8, s) }) else { return nil }
        if Task.isCancelled { return nil }      // 5.7 (48): closed while the surface was built - skip the rest
        // the fine detail: scan points with no surface near them
        progress(0.82, "Finding the fine detail the surface cannot hold...")
        let h = max(0.004, min(0.02, ext / 450))
        // strict: only points well clear of the surface (2.5 cm) that form dense structures - tested on a real scan,
        // a looser rule let LiDAR noise smear over the sharp photo texture
        let near = max(0.025, h * 4)
        let g = GridIndex(t.v, cell: near)
        let src = cl.count > 2_000_000 ? CloudOps.downsample(cl, to: 2_000_000) : cl
        var off = [Bool](repeating: false, count: src.count)
        off.withUnsafeMutableBufferPointer { ob in
            let op = ob.baseAddress!
            parallelFor(src.count) { i in op[i] = g.nearest(src.p[i], in: t.v, within: near) == nil }
        }
        var detail = src.keeping(off)
        // lone specks are not detail: a detail point needs neighbours (a wire is a line of points)
        if detail.count > 0 { detail = detail.keeping(CloudOps.floaterKeep(detail, radius: 0.012, minN: 8)) }
        detail = CloudOps.thin(detail, grid: 0.0025)
        if detail.count > 600_000 { detail = CloudOps.downsample(detail, to: 600_000) }
        progress(0.95, "Putting it together...")
        let meshGeo = t.geometry(centre: centre)
        var detailGeo: SCNGeometry? = nil
        if detail.count > 0 {
            let verts = detail.p.map { let q = $0 - centre; return SCNVector3(q.x, q.y, q.z) }
            var cols: [Float] = []; cols.reserveCapacity(detail.count * 3)
            for c in detail.c { cols += [Float(c.x) / 255, Float(c.y) / 255, Float(c.z) / 255] }
            detailGeo = pointGeometry(verts, cols, size: 3)
        }
        progress(1, "Done")
        return AranScene(mesh: meshGeo, detail: detailGeo, detailCount: detail.count, triangles: t.f.count / 3, splatRadius: 0.003, textured: t, detailCloud: detail, photos: RealPhoto.load(item), centre: centre)
    }
}

struct AranSceneView: UIViewRepresentable {
    let scene: AranScene
    var showDetail = false
    final class Coord { var splat: SCNMaterial?; weak var view: SCNView?; var detailNode: SCNNode?; var projector: PhotoProjector? }
    func makeCoordinator() -> Coord { Coord() }
    func makeUIView(context: Context) -> SCNView {
        let v = SCNView(frame: .zero); v.backgroundColor = .black; v.allowsCameraControl = true; v.antialiasingMode = .multisampling2X
        let s = SCNScene()
        s.rootNode.addChildNode(SCNNode(geometry: scene.mesh))
        if let d = scene.detail {
            if let m = SplatLook.material(radius: scene.splatRadius, pxScale: 1500) { d.materials = [m]; context.coordinator.splat = m }
            let n = SCNNode(geometry: d); n.isHidden = !showDetail; s.rootNode.addChildNode(n); context.coordinator.detailNode = n
        }
        let cam = SCNNode(); cam.camera = SCNCamera(); cam.camera?.zNear = 0.01
        let (lo, hi) = scene.mesh.boundingBox
        let size = max(Float(hi.x - lo.x), Float(hi.y - lo.y), Float(hi.z - lo.z))
        cam.position = SCNVector3(0, size * 0.15, size * 1.1)
        s.rootNode.addChildNode(cam); v.pointOfView = cam
        v.scene = s; context.coordinator.view = v
        if !scene.photos.isEmpty, let mat = scene.mesh.materials.first {
            let p = PhotoProjector(photos: scene.photos, material: mat, centre: scene.centre); context.coordinator.projector = p
            v.delegate = p; v.rendersContinuously = true; v.preferredFramesPerSecond = 30      // 30 is smooth for turning a model, half the battery
        }
        return v
    }
    func updateUIView(_ v: SCNView, context: Context) {
        if let m = context.coordinator.splat { SplatLook.set(m, radius: scene.splatRadius, pxScale: SplatLook.pxScale(v)) }
        context.coordinator.detailNode?.isHidden = !showDetail
    }
}

struct AranView: View {
    let item: ScanItem
    var cloud: Cloud? = nil
    @Environment(\.dismiss) private var dismiss
    @State private var scene: AranScene?
    @State private var busy = false
    @State private var progress: Float = 0
    @State private var status = ""
    @State private var share: URL?
    @State private var started = false
    @State private var showDetail = false
    @State private var job: Task<Void, Never>?       // 5.7 (48): cancelled on close, like SplatSheet's
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                if let s = scene {
                    AranSceneView(scene: s, showDetail: showDetail).ignoresSafeArea()
                    VStack { Spacer()
                        if s.detailCount > 0 { Toggle("Detail splats (\(s.detailCount.formatted()))", isOn: $showDetail).font(.caption).padding(.horizontal, 12).padding(.vertical, 6).background(.ultraThinMaterial, in: Capsule()).fixedSize() }
                        // 5.7 (46): Save / Share report here - status was drawn only before a scene existed, so nobody saw it
                        if !status.isEmpty { Text(status).font(.caption).padding(8).background(.ultraThinMaterial, in: Capsule()) }
                        Text("\(s.triangles.formatted()) triangles painted from your photos - drag to turn, pinch to zoom")
                            .font(.caption).padding(8).background(.ultraThinMaterial, in: Capsule()).padding(.bottom, 20)
                    }
                } else {
                    VStack(spacing: 14) {
                        Image(systemName: "wand.and.stars").font(.system(size: 52)).foregroundStyle(.orange)
                        Text("Aran: the finest model Deepfield makes - every photo's LiDAR depth for the shape, every photo for the look, seams levelled, the largest texture. Optional splats for detail that stands clear of the surfaces.")
                            .multilineTextAlignment(.center).font(.subheadline).padding(.horizontal)
                        if busy { ProgressView(value: progress).frame(width: 240); Text(status).font(.caption).foregroundStyle(.secondary) }
                        else if Library.captureURL(item) == nil { Text("Aran needs the scan's photos. Scan in Aran mode, or with photos on.").font(.caption).foregroundStyle(.orange).multilineTextAlignment(.center) }
                        else { Button("Make Aran") { make() }.buttonStyle(.borderedProminent).controlSize(.large) }
                        if !status.isEmpty && !busy { Text(status).font(.caption).foregroundStyle(.secondary) }
                    }.foregroundStyle(.white)
                }
            }
            .navigationTitle("Aran").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button { dismiss() } label: { Label("Close", systemImage: "xmark") }.labelStyle(.titleAndIcon) }
                if let s = scene {
                    ToolbarItem(placement: .confirmationAction) {
                        Menu {
                            Button("Save the model to the library") { saveModel(s) }
                            Button("Share the model (USDZ)") { share = usdz(s) }
                            Button("Share the model (OBJ + texture)") { shareOBJ(s) }
                            Button("Share the detail splats (.ply)") { let u = Export.tmp("\(item.name) detail.ply"); try? s.detailCloud.ply().write(to: u); share = u }
                        } label: { Image(systemName: "square.and.arrow.up") }
                    }
                }
            }
            .sheet(item: $share) { ShareSheet(items: [$0]) }
            .onAppear { if !started { started = true; if Library.captureURL(item) != nil { make() } } }
            .onDisappear { job?.cancel(); job = nil }
        }
    }
    private func make() {
        // 5.7 (48): a build from an Aran that was closed may still be running; never start a second one beside it
        guard AranBuilder.claim() else { status = "The last Aran build is still finishing - try again in a moment."; return }
        busy = true
        let given = cloud, it = item
        job = Task.detached(priority: .userInitiated) {
            defer { AranBuilder.release() }
            guard let c = given ?? Library.load(it) else { await MainActor.run { busy = false; status = "Could not open the scan" }; return }
            if Task.isCancelled { return }
            let s = AranBuilder.build(it, cloud: c) { p, t in Task { @MainActor in progress = p; status = t } }
            if Task.isCancelled { return }
            await MainActor.run { busy = false; scene = s; status = s == nil ? "Not enough photo depth to build Aran from this scan." : ""; Log.add("aran \(s?.triangles ?? 0) tris \(s?.detailCount ?? 0) detail") }
        }
    }
    private func usdz(_ s: AranScene) -> URL? {
        let sc = SCNScene(); sc.rootNode.addChildNode(SCNNode(geometry: s.mesh))
        let u = Export.tmp("\(item.name.replacingOccurrences(of: "/", with: "-")) Aran.usdz"); try? FileManager.default.removeItem(at: u)
        return sc.write(to: u, options: nil, delegate: nil, progressHandler: nil) ? u : nil
    }
    private func shareOBJ(_ s: AranScene) {
        let name = item.name.replacingOccurrences(of: "/", with: "-") + " Aran"
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("aran-" + UUID().uuidString)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // 5.7 (46): a failed write said nothing and shared an .obj that did not exist
        do { try Texturer.writeOBJ(s.textured, to: dir, name: name); status = ""; share = dir.appendingPathComponent("\(name).obj") }
        catch { status = "Could not write the OBJ: \(error.localizedDescription)"; Log.add("aran obj failed: \(error)") }
    }
    private func saveModel(_ s: AranScene) {
        if let u = usdz(s), Library.shared.addModel(from: u, name: item.name + " Aran") != nil { status = "Saved to the library" } else { status = "Could not save" }
    }
}
