import SwiftUI
import RealityKit
import ARKit
import SceneKit
import ModelIO
import SceneKit.ModelIO

/// 5.0 the model screen: view, place in AR, edit (turn, tip, sit on the floor, real size, cut away below a height),
/// save, share in common formats, rename, delete. Every model in the app opens here.
struct ModelStudio: View {
    let url: URL
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var lib = Library.shared
    @State private var scene: SCNScene?
    @State private var root = SCNNode()
    @State private var undo: [(simd_float4x4, [SCNGeometry?])] = []
    @State private var dirty = false
    @State private var editing = false
    @State private var cut = 0.0                 // 0...1 of the height: everything below goes
    @State private var heightCM = ""
    @State private var showAR: URL?
    @State private var share: URL?
    @State private var note = ""
    @State private var renaming = false
    @State private var newName = ""
    @State private var askDelete = false
    private var item: ScanItem? { lib.items.first { Library.modelURL($0.id).standardizedFileURL == url.standardizedFileURL } }

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                Color.black.ignoresSafeArea()
                if let s = scene { StudioView(scene: s, cutPlane: editing ? cutHeight : nil).ignoresSafeArea() }
                else { ProgressView("Opening...").tint(.white).foregroundStyle(.white) }
                VStack(spacing: 8) {
                    if !note.isEmpty { Text(note).font(.footnote).padding(8).background(.ultraThinMaterial, in: Capsule()).onTapGesture { note = "" } }
                    if editing { editPanel } else { mainBar }
                }.padding(.horizontal, 10).padding(.bottom, 8)
            }
            .navigationTitle(item?.name ?? "3D model").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button { dismiss() } label: { Label("Close", systemImage: "xmark") }.labelStyle(.titleAndIcon) }
                if let it = item {
                    ToolbarItem(placement: .primaryAction) {
                        Menu {
                            Button { newName = it.name; renaming = true } label: { Label("Rename", systemImage: "pencil") }
                            Button(role: .destructive) { askDelete = true } label: { Label("Delete", systemImage: "trash") }
                        } label: { Image(systemName: "ellipsis.circle") }
                    }
                }
            }
            .alert("Rename", isPresented: $renaming) {
                TextField("Name", text: $newName)
                Button("Save") { if let it = item { lib.rename(it, to: newName) } }
                Button("Cancel", role: .cancel) {}
            }
            .confirmationDialog("Delete this model?", isPresented: $askDelete, titleVisibility: .visible) {
                Button("Delete", role: .destructive) { if let it = item { lib.delete(it); dismiss() } }
            }
            .sheet(item: $share) { ShareSheet(items: [$0]) }
            .fullScreenCover(item: $showAR) { u in ARModelView(url: u) }
            .task { load() }
        }
    }

    // ---- the bars ----
    private var mainBar: some View {
        HStack(spacing: 10) {
            bar("AR", "arkit") { if let u = currentFile() { showAR = u } }
            bar("Edit", "slider.horizontal.3") { withAnimation { editing = true } }
            Menu {
                Button("USDZ (iPhone, Mac, Reality)") { exportAs("usdz") }
                Button("OBJ (Blender, most apps)") { exportAs("obj") }
                Button("STL (3D printing)") { exportAs("stl") }
                Button("PLY") { exportAs("ply") }
            } label: { barLabel("Share", "square.and.arrow.up") }
            bar(item == nil ? "Save" : (dirty ? "Save" : "Saved"), "tray.and.arrow.down") { save() }.disabled(item != nil && !dirty)
        }
    }
    private var editPanel: some View {
        VStack(spacing: 10) {
            HStack(spacing: 8) {
                small("Turn left", "rotate.left") { apply { $0.simdRotate(by: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0)), aroundTarget: .zero) } }
                small("Turn right", "rotate.right") { apply { $0.simdRotate(by: simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 1, 0)), aroundTarget: .zero) } }
                small("Tip over", "arrow.uturn.down") { apply { $0.simdRotate(by: simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0)), aroundTarget: .zero) } }
                small("On floor", "square.bottomhalf.filled") { apply { _ in sitOnFloor() } }
            }
            HStack {
                Text("Real height").font(.caption)
                TextField("cm", text: $heightCM).keyboardType(.decimalPad).textFieldStyle(.roundedBorder).frame(width: 80)
                Button("Set") { setHeight() }.buttonStyle(.bordered).disabled(Double(heightCM) == nil)
                Spacer()
                Text(sizeText).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            HStack {
                Text("Cut below").font(.caption)
                Slider(value: $cut, in: 0...0.5)
                Button("Cut") { cutBelow() }.buttonStyle(.bordered).disabled(cut < 0.005)
            }
            HStack {
                Button("Undo") { doUndo() }.buttonStyle(.bordered).disabled(undo.isEmpty)
                Spacer()
                Button("Done") { withAnimation { editing = false; cut = 0 } }.buttonStyle(.borderedProminent)
            }
        }
        .padding(12).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }
    private func bar(_ t: String, _ i: String, _ go: @escaping () -> Void) -> some View { Button(action: go) { barLabel(t, i) } }
    private func barLabel(_ t: String, _ i: String) -> some View {
        VStack(spacing: 3) { Image(systemName: i).font(.title3); Text(t).font(.caption2) }
            .frame(maxWidth: .infinity, minHeight: 50).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14)).foregroundStyle(.white)
    }
    private func small(_ t: String, _ i: String, _ go: @escaping () -> Void) -> some View {
        Button(action: go) { VStack(spacing: 2) { Image(systemName: i); Text(t).font(.caption2).lineLimit(1).minimumScaleFactor(0.6) }.frame(maxWidth: .infinity, minHeight: 40) }.buttonStyle(.bordered)
    }

    // ---- the model ----
    private func load() {
        guard scene == nil, let s = try? SCNScene(url: url, options: [.checkConsistency: true]) else { if scene == nil { note = "Could not open this model" }; return }
        let r = SCNNode()
        for c in s.rootNode.childNodes { c.removeFromParentNode(); r.addChildNode(c) }
        s.rootNode.addChildNode(r)
        root = r; scene = s
        centre()
    }
    /// the model's box in its parent's space
    private var box: (SIMD3<Float>, SIMD3<Float>) {
        let (lo, hi) = root.boundingBox
        let pts = [SIMD3<Float>(Float(lo.x), Float(lo.y), Float(lo.z)), SIMD3<Float>(Float(hi.x), Float(hi.y), Float(hi.z))]
        var a = SIMD3<Float>(repeating: .greatestFiniteMagnitude), b = -a
        for x in [pts[0].x, pts[1].x] { for y in [pts[0].y, pts[1].y] { for z in [pts[0].z, pts[1].z] {
            let w = root.simdConvertPosition(SIMD3(x, y, z), to: nil); a = simd_min(a, w); b = simd_max(b, w)
        } } }
        return (a, b)
    }
    private var sizeText: String { let (a, b) = box, s = b - a; return String(format: "%.0f x %.0f x %.0f cm", s.x * 100, s.y * 100, s.z * 100) }
    private var cutHeight: Float? { cut < 0.005 ? nil : { let (a, b) = box; return a.y + (b.y - a.y) * Float(cut) }() }
    private func centre() { let (a, b) = box; root.simdPosition -= SIMD3((a.x + b.x) / 2, a.y, (a.z + b.z) / 2) }
    private func sitOnFloor() { let (a, b) = box; root.simdPosition -= SIMD3((a.x + b.x) / 2, a.y, (a.z + b.z) / 2) }
    private func snapshot() -> (simd_float4x4, [SCNGeometry?]) {
        var gs: [SCNGeometry?] = []; root.enumerateHierarchy { n, _ in gs.append(n.geometry) }
        return (root.simdTransform, gs)
    }
    private func apply(_ f: (SCNNode) -> Void) {
        undo.append(snapshot()); if undo.count > 10 { undo.removeFirst() }
        f(root); dirty = true
    }
    private func doUndo() {
        guard let (t, gs) = undo.popLast() else { return }
        root.simdTransform = t
        var i = 0; root.enumerateHierarchy { n, _ in if i < gs.count { n.geometry = gs[i] }; i += 1 }
        dirty = true
    }
    private func setHeight() {
        guard let cm = Float(heightCM.replacingOccurrences(of: ",", with: ".")), cm > 0 else { return }
        let (a, b) = box; let h = b.y - a.y; guard h > 1e-5 else { return }
        apply { $0.simdScale *= SIMD3(repeating: (cm / 100) / h) }
        sitOnFloor()
        note = "Now \(sizeText)"
    }
    /// removes every triangle that lies wholly below the cut height (the floor or table under a scan)
    private func cutBelow() {
        guard let y = cutHeight else { return }
        undo.append(snapshot()); if undo.count > 10 { undo.removeFirst() }
        var removed = 0
        root.enumerateHierarchy { n, _ in
            guard let g = n.geometry, let vs = g.sources(for: .vertex).first else { return }
            let stride = vs.dataStride, off = vs.dataOffset, cnt = vs.vectorCount
            guard vs.bytesPerComponent == 4, vs.usesFloatComponents else { return }
            var world = [SIMD3<Float>](repeating: .zero, count: cnt)
            vs.data.withUnsafeBytes { b in
                for i in 0..<cnt {
                    let o = off + i * stride
                    let p = SIMD3<Float>(b.loadUnaligned(fromByteOffset: o, as: Float.self), b.loadUnaligned(fromByteOffset: o + 4, as: Float.self), b.loadUnaligned(fromByteOffset: o + 8, as: Float.self))
                    world[i] = n.simdConvertPosition(p, to: nil)
                }
            }
            var elements: [SCNGeometryElement] = []
            var kept: [[UInt32]?] = []                 // 5.7 (48): per element, the triangles left (nil = not triangles, kept as is)
            let removed0 = removed
            for e in g.elements {
                guard e.primitiveType == .triangles else { kept.append(nil); continue }
                var keep: [UInt32] = []
                e.data.withUnsafeBytes { b in
                    for t in 0..<e.primitiveCount {
                        var idx = [UInt32](repeating: 0, count: 3)
                        for k in 0..<3 {
                            let o = (t * 3 + k) * e.bytesPerIndex
                            idx[k] = e.bytesPerIndex == 2 ? UInt32(b.loadUnaligned(fromByteOffset: o, as: UInt16.self)) : b.loadUnaligned(fromByteOffset: o, as: UInt32.self)
                        }
                        if idx.allSatisfy({ Int($0) < cnt && world[Int($0)].y < y }) { removed += 1; continue }
                        keep += idx
                    }
                }
                kept.append(keep)
            }
            guard removed > removed0 else { return }
            // 5.7 (48): drop the vertices no triangle uses any more - SceneKit takes the box from the vertex source, so the
            // cut-away floor still set On floor, the size, Real height and the next cut (the model floated by the cut height)
            var sources = g.sources
            let plain = kept.allSatisfy { $0?.allSatisfy { Int($0) < cnt } ?? false } 
                && g.sources.allSatisfy { $0.vectorCount == cnt && $0.data.count >= $0.dataOffset + max(0, cnt - 1) * $0.dataStride + $0.bytesPerComponent * $0.componentsPerVector }
            var map = [Int32](repeating: -1, count: cnt), used: [Int] = []
            if plain {
                for k in kept { for i in k! where map[Int(i)] < 0 { map[Int(i)] = Int32(used.count); used.append(Int(i)) } }
                sources = g.sources.map { src in
                    let w = src.bytesPerComponent * src.componentsPerVector, so = src.dataOffset, ss = src.dataStride
                    var d = Data(count: used.count * w)
                    d.withUnsafeMutableBytes { o in src.data.withUnsafeBytes { b in
                        for (j, i) in used.enumerated() { o.baseAddress!.advanced(by: j * w).copyMemory(from: b.baseAddress!.advanced(by: so + i * ss), byteCount: w) }
                    } }
                    return SCNGeometrySource(data: d, semantic: src.semantic, vectorCount: used.count, usesFloatComponents: src.usesFloatComponents,
                                             componentsPerVector: src.componentsPerVector, bytesPerComponent: src.bytesPerComponent, dataOffset: 0, dataStride: w)
                }
            }
            for (e, k) in zip(g.elements, kept) {
                guard let k else { elements.append(e); continue }
                let idx = plain ? k.map { UInt32(map[Int($0)]) } : k
                elements.append(SCNGeometryElement(data: Data(bytes: idx, count: idx.count * 4), primitiveType: .triangles, primitiveCount: idx.count / 3, bytesPerIndex: 4))
            }
            let ng = SCNGeometry(sources: sources, elements: elements)
            ng.materials = g.materials
            n.geometry = ng
        }
        sitOnFloor(); cut = 0; dirty = true
        note = removed > 0 ? "Cut \(removed.formatted()) triangles from the bottom" : "Nothing below that height"
    }

    // ---- files ----
    /// 5.6: the model alone for writing - the viewer's scene also holds the red cut-preview plane (4 m square), which was
    /// written into every edited save, export and AR view
    private func clean() -> SCNScene { let o = SCNScene(); o.rootNode.addChildNode(root.clone()); return o }
    /// the model as it is now, as a USDZ file (the original if nothing changed)
    private func currentFile() -> URL? {
        guard dirty, scene != nil else { return url }
        let u = Export.tmp("model-edit-\(UUID().uuidString.prefix(6)).usdz")
        return clean().write(to: u, options: nil, delegate: nil, progressHandler: nil) ? u : nil
    }
    private func save() {
        guard let u = currentFile() else { note = "Could not save"; return }
        if let it = item {
            let dst = Library.modelURL(it.id)
            if u != dst { _ = try? FileManager.default.replaceItemAt(dst, withItemAt: u) }
            dirty = false; undo = []; note = "Saved"
            // 5.7 (48): re-draw the library picture - Home kept showing the model as it was before the edit, for good
            let id = it.id
            Library.io.async { Library.modelThumb(id); DispatchQueue.main.async { Library.shared.objectWillChange.send() } }
        } else if Library.shared.addModel(from: u, name: "Model " + Date().formatted(date: .abbreviated, time: .shortened)) != nil {
            note = "Saved to the library"
        }
    }
    private func exportAs(_ ext: String) {
        guard scene != nil else { return }
        let s = clean()
        let name = (item?.name ?? "model").replacingOccurrences(of: "/", with: "-")
        if ext == "usdz" { if let u = currentFile() { let d = Export.tmp("\(name).usdz"); try? FileManager.default.removeItem(at: d); try? FileManager.default.copyItem(at: u, to: d); share = d }; return }
        let asset = MDLAsset(scnScene: s)
        guard MDLAsset.canExportFileExtension(ext) else { note = "This phone cannot write .\(ext)"; return }
        let d = Export.tmp("\(name).\(ext)"); try? FileManager.default.removeItem(at: d)
        do { try asset.export(to: d); share = d } catch { note = "Could not write .\(ext): \(error.localizedDescription)" }
    }
}

/// the scene with soft lighting; a translucent plane shows where a cut will go
struct StudioView: UIViewRepresentable {
    let scene: SCNScene
    var cutPlane: Float?
    final class Coord { let plane = SCNNode() }
    func makeCoordinator() -> Coord { Coord() }
    func makeUIView(context: Context) -> SCNView {
        let v = SCNView(frame: .zero); v.backgroundColor = UIColor(white: 0.06, alpha: 1); v.allowsCameraControl = true; v.autoenablesDefaultLighting = true
        v.antialiasingMode = .multisampling4X; v.scene = scene
        let pl = SCNPlane(width: 4, height: 4); let m = SCNMaterial(); m.diffuse.contents = UIColor.systemRed.withAlphaComponent(0.25); m.isDoubleSided = true; m.lightingModel = .constant
        pl.materials = [m]; context.coordinator.plane.geometry = pl; context.coordinator.plane.eulerAngles.x = -.pi / 2; context.coordinator.plane.isHidden = true
        scene.rootNode.addChildNode(context.coordinator.plane)
        return v
    }
    func updateUIView(_ v: SCNView, context: Context) {
        if let y = cutPlane { context.coordinator.plane.isHidden = false; context.coordinator.plane.simdPosition = SIMD3(0, y, 0) } else { context.coordinator.plane.isHidden = true }
    }
}

/// AR Quick Look: put the model in the room at real size
struct ARModelView: View {
    let url: URL
    @Environment(\.dismiss) private var dismiss
    @State private var placed = false
    @State private var note: String?
    @AppStorage("arOcclusion") private var occlusion = false
    var body: some View {
        ZStack {
            ARPlacer(url: url, placed: $placed, note: $note).id(occlusion).ignoresSafeArea()
            VStack {
                HStack {
                    Button { dismiss() } label: { Label("Close", systemImage: "xmark") }.buttonStyle(.borderedProminent).tint(.black.opacity(0.6))
                    Spacer()
                    Button { occlusion.toggle(); placed = false } label: { Label(occlusion ? "Hidden by real things" : "Always in front", systemImage: occlusion ? "person.crop.rectangle.stack" : "cube") }
                        .buttonStyle(.borderedProminent).tint(.black.opacity(0.6)).font(.caption)
                }.padding()
                Spacer()
                Text(note ?? (placed ? "Drag to move, pinch to resize, twist to turn. Tap elsewhere to move it there." : "Point at the floor or a table, then tap to place the model."))
                    .font(.subheadline).multilineTextAlignment(.center).padding(10).background(.ultraThinMaterial, in: Capsule()).padding(.bottom, 40)
            }
        }
    }
}

/// 5.5: the camera view with the model placed in the room (real size), RealityKit. It was Quick Look, whose AR switch did
/// not show inside the app's sheet, so AR showed only the model on a white screen.
struct ARPlacer: UIViewRepresentable {
    let url: URL
    @Binding var placed: Bool
    @Binding var note: String?
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeUIView(context: Context) -> ARView {
        let v = ARView(frame: .zero)
        let cfg = ARWorldTrackingConfiguration()
        cfg.planeDetection = [.horizontal, .vertical]
        cfg.environmentTexturing = .automatic
        // 5.6: the room mesh's floor sits a centimetre or two above the real floor, so with occlusion on a shrunk
        // model sank "under the floor". Occlusion is a choice now (off by default) and the model sits 1.5 cm up.
        if UserDefaults.standard.bool(forKey: "arOcclusion"), ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            cfg.sceneReconstruction = .mesh
            v.environment.sceneUnderstanding.options.insert(.occlusion)      // real things in front hide the model
        }
        v.session.run(cfg)
        let coach = ARCoachingOverlayView(); coach.session = v.session; coach.goal = .anyPlane; coach.activatesAutomatically = true
        coach.frame = v.bounds; coach.autoresizingMask = [.flexibleWidth, .flexibleHeight]; v.addSubview(coach)
        v.addGestureRecognizer(UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.tap(_:))))
        v.addGestureRecognizer(UILongPressGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.longPress(_:))))
        context.coordinator.view = v
        DispatchQueue.main.async { context.coordinator.load() }      // not during the view update: load() changes the bound note
        return v
    }
    func updateUIView(_ v: ARView, context: Context) {}
    static func dismantleUIView(_ v: ARView, coordinator: Coordinator) { v.session.pause() }

    final class Coordinator: NSObject {
        let parent: ARPlacer
        weak var view: ARView?
        var holder: ModelEntity?
        var anchor: AnchorEntity?
        init(_ p: ARPlacer) { parent = p }
        func load() {
            parent.note = "Opening the model..."
            if #available(iOS 18.0, *) {
                // big photoreal models (8K textures) load off the main thread: the camera keeps running meanwhile
                Task { @MainActor in
                    do { self.ready(try await Entity(contentsOf: self.parent.url)) }
                    catch { self.parent.note = "Could not open the model: \(error.localizedDescription)" }
                }
            } else {
                do { ready(try Entity.load(contentsOf: parent.url)) }
                catch { parent.note = "Could not open the model: \(error.localizedDescription)" }
            }
        }
        private func ready(_ e: Entity) {
            let h = ModelEntity(); h.addChild(e)
            // sit it on the surface: its lowest point at the anchor, centred over it
            let b = h.visualBounds(relativeTo: h)
            e.position = SIMD3(-b.center.x, -b.min.y, -b.center.z)
            // the gestures hit-test the holder itself: give IT the shape (a box round the model)
            h.collision = CollisionComponent(shapes: [ShapeResource.generateBox(size: b.extents).offsetBy(translation: SIMD3(0, b.extents.y / 2, 0))])
            holder = h
            parent.note = nil
        }
        /// 5.7 (44): real detected surfaces first (the floor's actual geometry); ARKit's guess only when there is none.
        /// The guess was often far too deep, so the model sat beyond the floor and seemed to run away as you walked up.
        private func surface(_ v: ARView, _ pt: CGPoint) -> ARRaycastResult? {
            v.raycast(from: pt, allowing: .existingPlaneGeometry, alignment: .horizontal).first
                ?? v.raycast(from: pt, allowing: .existingPlaneInfinite, alignment: .horizontal).first
                ?? v.raycast(from: pt, allowing: .estimatedPlane, alignment: .horizontal).first
        }
        /// the model's anchor is an ARKit anchor, so ARKit keeps correcting its place as it learns the room
        private func put(_ v: ARView, _ h: ModelEntity, at hit: ARRaycastResult) {
            var wt = matrix_identity_float4x4                          // upright, at the hit point, 1.5 cm up
            wt.columns.3 = hit.worldTransform.columns.3; wt.columns.3.y += 0.015
            let keepTurn = anchor.map { _ in h.transform.rotation } ?? simd_quatf(angle: 0, axis: [0, 1, 0])
            if let old = anchor { v.scene.removeAnchor(old) }
            if let oa = arAnchor { v.session.remove(anchor: oa) }
            let aa = ARAnchor(name: "model", transform: wt); v.session.add(anchor: aa); arAnchor = aa
            let a = AnchorEntity(anchor: aa)
            h.transform.rotation = keepTurn; h.position = .zero              // a drag before would offset the new spot
            a.addChild(h); v.scene.addAnchor(a); anchor = a
        }
        var arAnchor: ARAnchor?
        @objc func tap(_ g: UITapGestureRecognizer) {
            guard let v = view, let h = holder else { return }
            // 5.7 (44): once placed, taps never move it (a tap just beside it threw it to wherever ARKit guessed - often far
            // behind, so it "disappeared"). Drag to move, or long-press a spot to send it there.
            if anchor != nil { parent.note = "Drag to move it, two fingers to turn or resize. Long-press a spot to move it there."; return }
            guard let hit = surface(v, g.location(in: v)) else { parent.note = "No floor found there yet - move the phone a little so it can see the floor"; return }
            put(v, h, at: hit)
            v.installGestures([.translation, .rotation, .scale], for: h)
            parent.placed = true; parent.note = nil
        }
        @objc func longPress(_ g: UILongPressGestureRecognizer) {
            guard g.state == .began, let v = view, let h = holder, anchor != nil else { return }
            guard let hit = surface(v, g.location(in: v)) else { parent.note = "No floor found there yet"; return }
            put(v, h, at: hit); parent.note = nil
        }
    }
}
