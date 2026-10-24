import SwiftUI
import SceneKit
import ARKit
import simd

// ---------- AR checkpoints: glowing stations around the subject; walk to each one and it turns green ----------
final class Checkpoints {
    let node = SCNNode()
    private(set) var points: [SIMD3<Float>] = []
    private(set) var done: [Bool] = []
    private var nodes: [SCNNode] = []
    var placed: Bool { !points.isEmpty }
    var progress: Double { done.isEmpty ? 0 : Double(done.filter { $0 }.count) / Double(done.count) }
    var remaining: Int { done.filter { !$0 }.count }

    func clear() { for n in node.childNodes { n.removeFromParentNode() }; points = []; done = []; nodes = [] }

    /// stations on rings around the subject: where to stand (and how high to hold the phone)
    func place(centre c: SIMD3<Float>, radius r: Float, rings: [(dy: Float, count: Int)]) {
        clear()
        for ring in rings {
            for i in 0..<ring.count {
                let a = Float(i) / Float(ring.count) * 2 * .pi
                points.append(SIMD3(c.x + r * cos(a), c.y + ring.dy, c.z + r * sin(a)))
            }
        }
        done = Array(repeating: false, count: points.count)
        for (i, p) in points.enumerated() {
            let n = SCNNode(); n.position = SCNVector3(p.x, p.y, p.z)
            let orb = SCNSphere(radius: 0.035); orb.segmentCount = 16
            let m = SCNMaterial(); m.lightingModel = .constant; m.diffuse.contents = UIColor.cyan; m.emission.contents = UIColor.cyan
            orb.materials = [m]
            let o = SCNNode(geometry: orb)
            o.runAction(.repeatForever(.sequence([.scale(to: 1.35, duration: 0.7), .scale(to: 1.0, duration: 0.7)])))
            n.addChildNode(o)
            let ringG = SCNTorus(ringRadius: 0.09, pipeRadius: 0.004)
            let rm = SCNMaterial(); rm.lightingModel = .constant; rm.diffuse.contents = UIColor.cyan.withAlphaComponent(0.7); ringG.materials = [rm]
            let rn = SCNNode(geometry: ringG); rn.eulerAngles.x = .pi / 2; n.addChildNode(rn)
            n.addChildNode(label("\(i + 1)", color: .white, size: 0.05, y: 0.08))
            node.addChildNode(n); nodes.append(n)
        }
    }

    /// marks stations the phone has reached; returns true if one was just reached
    func update(phone: SIMD3<Float>) -> Bool {
        var hit = false
        for i in points.indices where !done[i] && simd_length(points[i] - phone) < 0.3 {
            done[i] = true; hit = true
            let n = nodes[i]
            for c in n.childNodes {
                if let g = c.geometry { g.firstMaterial?.diffuse.contents = UIColor.green; g.firstMaterial?.emission.contents = UIColor.green }
            }
            n.runAction(.sequence([.scale(to: 1.6, duration: 0.25), .fadeOut(duration: 1.2), .removeFromParentNode()]))
        }
        return hit
    }
    /// the nearest station still to visit
    func next(from phone: SIMD3<Float>) -> (Int, SIMD3<Float>)? {
        var best: (Int, SIMD3<Float>)? = nil, bd: Float = .greatestFiniteMagnitude
        for i in points.indices where !done[i] { let d = simd_length(points[i] - phone); if d < bd { bd = d; best = (i, points[i]) } }
        return best
    }
}

/// a camera-facing 3D text label
func label(_ s: String, color: UIColor, size: CGFloat, y: Float) -> SCNNode {
    let t = SCNText(string: s, extrusionDepth: 0); t.font = UIFont.systemFont(ofSize: 10, weight: .semibold); t.flatness = 0.2
    let m = SCNMaterial(); m.lightingModel = .constant; m.diffuse.contents = color; m.isDoubleSided = true; t.materials = [m]
    let n = SCNNode(geometry: t)
    let (lo, hi) = t.boundingBox
    n.pivot = SCNMatrix4MakeTranslation((hi.x - lo.x) / 2 + lo.x, lo.y, 0)
    let k = Float(size) / 10; n.scale = SCNVector3(k, k, k)
    let holder = SCNNode(); holder.position = SCNVector3(0, y, 0); holder.addChildNode(n)
    let bb = SCNBillboardConstraint(); bb.freeAxes = .all; holder.constraints = [bb]
    return holder
}

// ---------- ULTRA: everything automatic ----------
/// Runs beside the live scan: picks the mode (AI from the camera, then the measured distance), re-tunes, torch,
/// voice guide, Smart check, checkpoints.
@MainActor
final class UltraPilot: ObservableObject {
    @Published var mode: ScanKind = .area
    @Published var subject = ""
    @Published var note = "ULTRA: looking at what you are scanning..."
    private var asked = false
    private var lastTune: TimeInterval = 0
    private var started: TimeInterval = 0
    private var started15 = false            // the voice guide was started automatically once
    private var closeCap: Float?             // 5.7 (48): the close-up mode's own reach (object 1.2, surface 0.9, person 1.6)
    var autoTorch = false

    func tick(_ s: Scanner, torch: Torch, guide: LiveGuide, ultra: Bool) {
        let t = Date().timeIntervalSince1970
        // auto torch (in every mode when Settings say so, always in ULTRA); hysteresis because the torch lifts the reading
        if USE_CAMERA && (ultra || Prefs.autoTorch) {
            if !torch.on && s.ambient < 220 && s.running { torch.level = s.ambient < 90 ? 1.0 : 0.6; torch.on = true; torch.apply(); autoTorch = true }
            else if torch.on && autoTorch && s.ambient > 1500 { torch.off(); autoTorch = false }
        }
        guard ultra else { return }
        if started == 0 { started = t }
        if !s.smartOn && !s.smartUserOff { s.smartOn = true }        // 5.6: never back on after you turned it off
        if !guide.on && !guide.userOff && !started15 && t - started > 1.5 { started15 = true; guide.kind = .ultra; guide.start() }   // once, and never after you turn it off
        // the AI decides what this is once, from a camera frame
        if !asked && s.running && t - started > 2.5 {
            asked = true
            s.requestJPEG { d in
                Task { @MainActor in
                    let prompt = "This camera frame is the start of a 3D scan. Reply JSON only: {\"mode\": one of room|area|object|surface|person, \"subject\": what is being scanned in 2-6 words}."
                    if let r = try? await Gemini.ask(prompt, images: [d], json: true),
                       let j = try? JSONSerialization.jsonObject(with: Data(r.utf8)) as? [String: Any] {
                        if let m = j["mode"] as? String, let k = ScanKind(rawValue: m), k != .ultra { self.set(k, s, guide: guide, why: "AI") }
                        if let sub = j["subject"] as? String { self.subject = sub; guide.subject = sub }
                        self.note = "ULTRA: \(self.subject.isEmpty ? self.mode.title.lowercased() : self.subject) - \(self.mode.title) tuning"
                    }
                }
            }
        }
        // fine tuning from the measured distance: reach and detail follow how far away the surface actually is
        if s.running && t - lastTune > 4 {
            lastTune = t
            let d = s.medianDepth
            let reach = max(1.0, min(6.0, d * 2.4 + 0.3))
            // 5.7 (48): close-up modes keep their own short reach (as Prefs.applyReach does) - the settings floor fused the
            // walls and floor up to 3.5 m into an object or person scan and dragged the checkpoint centre off it
            if let cap = closeCap { MAX_DEPTH = min(cap, reach) } else { MAX_DEPTH = max(Float(Prefs.maxDepth), reach) }
            KF_MOVE = max(0.10, min(0.35, d * 0.18))
            if !s.checkpoints.placed && s.count > 150_000 && (mode == .object || mode == .person) { s.placeCheckpoints(mode) }
        }
    }

    func set(_ k: ScanKind, _ s: Scanner, guide: LiveGuide, why: String) {
        guard k != mode else { return }
        // 5.7 (46): start from the settings baseline, so the last mode's EDGE_REL / PHOTO_LONG do not carry over
        mode = k; Prefs.applyToScanner(); k.apply(); Prefs.applyPhotoPrefs(k); Prefs.applyReach(k); MAX_KEYFRAMES = 120; VIDEO_PREF = 0
        closeCap = [.object, .surface, .person].contains(k) ? MAX_DEPTH : nil     // 5.7 (48)
        s.setPeopleMasking(k != .person)
        guide.say("The app switched to \(k.title) mode (\(why)).")
    }
}

// ---------- more editing tools ----------
extension CloudOps {
    /// grey-world white balance, then stretch each channel between its 1st and 99th percentile
    static func autoColour(_ cl: Cloud) -> Cloud {
        guard cl.count > 100 else { return cl }
        var sum = SIMD3<Double>.zero
        var hist = [[Int]](repeating: [Int](repeating: 0, count: 256), count: 3)
        for k in cl.c { sum += SIMD3<Double>(Double(k.x), Double(k.y), Double(k.z)); hist[0][Int(k.x)] += 1; hist[1][Int(k.y)] += 1; hist[2][Int(k.z)] += 1 }
        let mean = sum / Double(cl.count), grey = (mean.x + mean.y + mean.z) / 3
        let gain = SIMD3<Float>(Float(grey / max(1, mean.x)), Float(grey / max(1, mean.y)), Float(grey / max(1, mean.z)))
        func pct(_ h: [Int], _ q: Double) -> Float { var acc = 0; let target = Int(Double(cl.count) * q); for i in 0..<256 { acc += h[i]; if acc >= target { return Float(i) } }; return 255 }
        let lo = SIMD3<Float>(pct(hist[0], 0.01), pct(hist[1], 0.01), pct(hist[2], 0.01)) * gain
        let hi = SIMD3<Float>(pct(hist[0], 0.99), pct(hist[1], 0.99), pct(hist[2], 0.99)) * gain
        var o = cl
        for i in 0..<o.count {
            var v = SIMD3<Float>(Float(o.c[i].x), Float(o.c[i].y), Float(o.c[i].z)) * gain
            v = (v - lo) / simd_max(hi - lo, SIMD3(repeating: 1)) * 255
            v = simd_clamp(v, .zero, SIMD3(repeating: 255))
            o.c[i] = SIMD3(u8(v.x), u8(v.y), u8(v.z))
        }
        return o
    }

    /// the biggest flat near-horizontal plane, if any (normal pointing up, d)
    static func floorPlane(_ cl: Cloud) -> (SIMD3<Float>, Float)? {
        guard cl.count > 100 else { return nil }
        let b = cl.bounds, cutY = b.lo.y + b.size.y * 0.35
        var low: [SIMD3<Float>] = []
        let step = max(1, cl.count / 60000)
        for i in stride(from: 0, to: cl.count, by: step) where cl.p[i].y <= cutY { low.append(cl.p[i]) }
        guard low.count > 50 else { return nil }
        var bestN = 0, best: (SIMD3<Float>, Float)? = nil
        for _ in 0..<400 {
            let a = low.randomElement()!, b2 = low.randomElement()!, c2 = low.randomElement()!
            var nrm = simd_cross(b2 - a, c2 - a); let len = simd_length(nrm)
            guard len > 1e-6 else { continue }
            nrm /= len; if nrm.y < 0 { nrm = -nrm }
            guard nrm.y > 0.85 else { continue }
            let d = -simd_dot(nrm, a)
            var n = 0; for q in low where abs(simd_dot(nrm, q) + d) < 0.012 { n += 1 }
            if n > bestN { bestN = n; best = (nrm, d) }
        }
        return bestN > low.count / 10 ? best : nil
    }

    /// turns the cloud so its floor is exactly level
    static func level(_ cl: Cloud) -> (Cloud, Bool) {
        guard let (n, _) = floorPlane(cl) else { return (cl, false) }
        let q = simd_quatf(from: n, to: SIMD3(0, 1, 0))
        let c = cl.bounds.centre
        var o = cl
        for i in 0..<o.count { o.p[i] = q.act(o.p[i] - c) + c }
        return (o, true)
    }

    /// keeps only the biggest connected piece (1.5 cm grid flood fill)
    static func largestPiece(_ cl: Cloud, cell: Float = 0.015) -> Cloud {
        guard cl.count > 0 else { return cl }
        var cells: [SIMD3<Int32>: Int] = [:]
        var cellOf = [SIMD3<Int32>](repeating: .zero, count: cl.count)
        for i in 0..<cl.count {
            let q = cl.p[i], k = SIMD3<Int32>(ik((q.x / cell)), ik((q.y / cell)), ik((q.z / cell)))
            cellOf[i] = k; cells[k, default: 0] += 1
        }
        var comp: [SIMD3<Int32>: Int] = [:], sizes: [Int] = []
        for (start, _) in cells where comp[start] == nil {
            let id = sizes.count; var stack = [start]; comp[start] = id; var total = 0
            while let k = stack.popLast() {
                total += cells[k] ?? 0
                for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 {
                    let nb = k &+ SIMD3(dx, dy, dz)
                    if cells[nb] != nil && comp[nb] == nil { comp[nb] = id; stack.append(nb) }
                } } }
            }
            sizes.append(total)
        }
        guard let big = sizes.max() else { return cl }
        // 4.5: every piece at least 3% the size of the biggest stays (a chair leg that came apart is not junk)
        let keepIDs = Set(sizes.indices.filter { sizes[$0] * 33 >= big })
        return cl.keeping(cellOf.map { keepIDs.contains(comp[$0] ?? -1) })
    }
}

// ---------- AR guidance: a 3D arrow in front of the phone pointing at the next target, with a dotted trail to it ----------
final class GuideArrow {
    let node = SCNNode()
    private let arrow = SCNNode()
    private var dots: [SCNNode] = []
    private let tag = SCNNode()
    init() {
        let cone = SCNCone(topRadius: 0, bottomRadius: 0.022, height: 0.06)
        let shaft = SCNCylinder(radius: 0.007, height: 0.05)
        let m = SCNMaterial(); m.lightingModel = .constant; m.diffuse.contents = UIColor.systemYellow; m.emission.contents = UIColor.systemYellow
        cone.materials = [m]; shaft.materials = [m]
        let c = SCNNode(geometry: cone); c.eulerAngles.x = -.pi / 2; c.position = SCNVector3(0, 0, -0.04)
        let sh = SCNNode(geometry: shaft); sh.eulerAngles.x = -.pi / 2; sh.position = SCNVector3(0, 0, 0.01)
        arrow.addChildNode(c); arrow.addChildNode(sh)
        node.addChildNode(arrow)
        for _ in 0..<10 {
            let s = SCNSphere(radius: 0.008); let dm = SCNMaterial(); dm.lightingModel = .constant; dm.diffuse.contents = UIColor.systemYellow.withAlphaComponent(0.8); s.materials = [dm]
            let d = SCNNode(geometry: s); node.addChildNode(d); dots.append(d)
        }
        node.isHidden = true
    }
    /// cam: the camera transform; target: where to go (nil hides the guide)
    func update(cam: simd_float4x4, target: SIMD3<Float>?) {
        guard let t = target else { node.isHidden = true; return }
        let pos = SIMD3<Float>(cam.columns.3.x, cam.columns.3.y, cam.columns.3.z)
        let fwd = -SIMD3<Float>(cam.columns.2.x, cam.columns.2.y, cam.columns.2.z), up = SIMD3<Float>(cam.columns.1.x, cam.columns.1.y, cam.columns.1.z)
        let a = pos + fwd * 0.45 - up * 0.12
        let dist = simd_length(t - pos)
        if dist < 0.3 { node.isHidden = true; return }
        node.isHidden = false
        arrow.simdPosition = a
        arrow.simdLook(at: t)
        let pulse = 1 + 0.15 * Float(sin(Date().timeIntervalSince1970 * 6))
        arrow.simdScale = SIMD3(repeating: pulse)
        for (i, d) in dots.enumerated() {
            let f = Float(i + 1) / Float(dots.count + 1)
            d.simdPosition = a + (t - a) * f
        }
    }
}

// ---------- AR viewer: put a saved scan back into the room and walk around it ----------
struct ARPlaceView: UIViewRepresentable {
    let cloud: Cloud
    let miniature: Bool
    let turn: Double
    final class Coord: NSObject {
        var view: ARSCNView?
        let holder = SCNNode()
        let model = SCNNode()
        var placed = false
        var size: SIMD3<Float> = .zero
        @objc func tap(_ g: UITapGestureRecognizer) {
            guard let v = view else { return }
            let p = g.location(in: v)
            guard let q = v.raycastQuery(from: p, allowing: .estimatedPlane, alignment: .any), let hit = v.session.raycast(q).first else { return }
            holder.simdTransform = hit.worldTransform
            if !placed { v.scene.rootNode.addChildNode(holder); placed = true }
        }
    }
    func makeCoordinator() -> Coord { Coord() }
    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView(frame: .zero)
        let c = ARWorldTrackingConfiguration(); c.planeDetection = [.horizontal, .vertical]
        v.session.run(c)
        let co = context.coordinator; co.view = v
        co.holder.addChildNode(co.model)
        v.addGestureRecognizer(UITapGestureRecognizer(target: co, action: #selector(Coord.tap(_:))))
        let cl = cloud
        DispatchQueue.global(qos: .userInitiated).async {
            let g = cl.geometry(maxPoints: 700_000, mode: "photo", size: 2)
            let size = cl.bounds.size
            DispatchQueue.main.async {
                co.model.geometry = g?.0; co.size = size
                co.model.simdPosition = SIMD3(0, size.y / 2, 0)         // the scan's bottom sits on the surface tapped
                v.setNeedsLayout()
            }
        }
        return v
    }
    func updateUIView(_ v: ARSCNView, context: Context) {
        let co = context.coordinator
        let longest = max(0.01, max(co.size.x, max(co.size.y, co.size.z)))
        let s: Float = miniature ? 0.35 / longest : 1
        co.model.simdScale = SIMD3(repeating: s)
        co.model.simdPosition = SIMD3(0, co.size.y / 2 * s, 0)
        co.model.simdEulerAngles = SIMD3(0, Float(turn) * .pi / 180, 0)
    }
    static func dismantleUIView(_ v: ARSCNView, coordinator: Coord) { v.session.pause() }
}

struct ARSceneView: View {
    let item: ScanItem
    @Environment(\.dismiss) private var dismiss
    @State private var cloud: Cloud?
    @State private var mini = false
    @State private var turn = 0.0
    var body: some View {
        ZStack {
            if let c = cloud { ARPlaceView(cloud: c, miniature: mini, turn: turn).ignoresSafeArea() } else { Color.black.ignoresSafeArea(); ProgressView("Opening...") }
            VStack {
                HStack {
                    Button { dismiss() } label: { Image(systemName: "xmark").font(.title3.bold()).frame(width: 40, height: 40).background(.ultraThinMaterial, in: Circle()) }
                    Spacer()
                    Picker("", selection: $mini) { Text("Real size").tag(false); Text("Miniature").tag(true) }.pickerStyle(.segmented).frame(width: 220)
                }.padding()
                Spacer()
                Text("Tap the floor (or a table) to place your scan, then walk around it.").font(.footnote).padding(10).background(.ultraThinMaterial, in: Capsule())
                HStack { Image(systemName: "rotate.right"); Slider(value: $turn, in: -180...180) }.padding().background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16)).padding()
            }
        }
        .task { cloud = await Task.detached { Library.load(item).map { CloudOps.downsample($0, to: 1_500_000) } }.value }
    }
}

extension CloudOps {
    /// Separate pieces of the scan (3 cm grid); a SMALL piece that most depth maps saw straight through is a ghost copy
    /// (a cupboard edge that drifted, a person who walked off) and goes. A real floating thing (a hanging lamp) is
    /// confirmed by the maps that saw it as surface, so it stays.
    static func floatingGhosts(_ cl: Cloud, views: [DepthView]) -> ([Bool], Int) {
        var drop = [Bool](repeating: false, count: cl.count)
        guard !views.isEmpty, cl.count > 1000 else { return (drop, 0) }
        let cell: Float = 0.03
        var cellPts: [SIMD3<Int32>: Int] = [:]
        var cellOf = [SIMD3<Int32>](repeating: .zero, count: cl.count)
        for i in 0..<cl.count { let q = cl.p[i]; let k = SIMD3<Int32>(ik((q.x / cell)), ik((q.y / cell)), ik((q.z / cell))); cellOf[i] = k; cellPts[k, default: 0] += 1 }
        var comp: [SIMD3<Int32>: Int] = [:], sizes: [Int] = [], lo: [SIMD3<Int32>] = [], hi: [SIMD3<Int32>] = []
        for (start, _) in cellPts where comp[start] == nil {
            let id = sizes.count; var stack = [start]; comp[start] = id; var total = 0; var l = start, h = start
            while let k = stack.popLast() {
                total += cellPts[k] ?? 0; l = simd_min(l, k); h = simd_max(h, k)
                for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 {
                    let nb = k &+ SIMD3(dx, dy, dz); if cellPts[nb] != nil && comp[nb] == nil { comp[nb] = id; stack.append(nb) }
                } } }
            }
            sizes.append(total); lo.append(l); hi.append(h)
        }
        let n = cl.count
        let cand = Set(sizes.indices.filter { sizes[$0] >= 30 && sizes[$0] < n * 4 / 100 && simd_reduce_max(hi[$0] &- lo[$0]) < 40 })   // under 1.2 m
        guard !cand.isEmpty else { return (drop, 0) }
        var members: [Int: [Int]] = [:]
        for i in 0..<n { if let c = comp[cellOf[i]], cand.contains(c) { members[c, default: []].append(i) } }
        var removed = 0
        for (_, idx) in members {
            let step = max(1, idx.count / 200)
            var ghost = 0, real = 0, tried = 0
            for j in stride(from: 0, to: idx.count, by: step) {
                var front = 0, ok = 0
                for v in views {
                    guard let (u, vv, d) = v.kf.project(cl.p[idx[j]]) else { continue }
                    let md = v.measured(u, vv); guard md > 0 else { continue }
                    if d < md - 0.04 { front += 1 } else if abs(d - md) <= 0.04 { ok += 1 }
                }
                tried += 1
                if front >= 2 && front > ok { ghost += 1 } else if ok >= 2 { real += 1 }
            }
            if tried > 0 && ghost * 10 >= tried * 4 && ghost > real { for i in idx { drop[i] = true }; removed += 1 }
        }
        return (drop, removed)
    }
}
