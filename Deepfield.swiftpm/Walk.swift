import SwiftUI
import ARKit
import SceneKit
import CoreMotion
import CoreImage
import simd

// ---------- 5.6 Walk: Street View for your own places ----------
// At each spot you turn round once: ARKit knows exactly where the phone points for every frame, so the frames are
// painted straight onto a sphere (no guessing which photo goes where). Walk a few steps, do the next spot. The viewer
// puts you inside each sphere; arrows on the floor point at the other spots (their real positions) and take you there.

/// one frame kept while turning round: a small copy of the picture and where the camera pointed
struct WalkFrame {
    let rgba: [UInt8]; let w: Int; let h: Int
    let R: simd_float3x3                 // camera -> world rotation
    let fx: Float, fy: Float, cx: Float, cy: Float
    var pos = SIMD3<Float>.zero          // where the camera was (it moves as you turn: the cause of ghosts)
    var depth: [Float] = []; var dw = 0; var dh = 0      // LiDAR depth (0 = unknown)
    var fwd: SIMD3<Float> { -R.columns.2 }
}

/// 5.6 how much each spot captures (like Matterport's capture types)
enum WalkMode: String, CaseIterable {
    case quick = "Quick 180°", standard = "360°", full = "Full sphere", lidar = "LiDAR 3D"
    static var current: WalkMode { WalkMode(rawValue: UserDefaults.standard.string(forKey: "walk.mode") ?? "") ?? .standard }
    var detail: String {
        switch self {
        case .quick: return "7 shots - the half in front of you. Fastest."
        case .standard: return "24 shots - all the way round, up and down."
        case .full: return "45 shots - the whole sphere, ceiling and floor too. Cleanest joins."
        case .lidar: return "24 shots with LiDAR depth - spheres plus a measurable 3D dollhouse and floor plan."
        }
    }
    /// the dot directions, the first one straight ahead (yaw = where you face at the start)
    func targets(yaw: Float) -> [SIMD3<Float>] {
        func d(_ lat: Float, _ lon: Float) -> SIMD3<Float> {
            let la = lat * .pi / 180, lo = yaw + lon * .pi / 180
            return SIMD3(cos(la) * sin(lo), sin(la), -cos(la) * cos(lo))
        }
        func ring(_ lat: Float, _ n: Int, from: Float = 0, span: Float = 360) -> [SIMD3<Float>] {
            (0..<n).map { k in d(lat, from + span * Float(k) / Float(span >= 360 ? n : max(1, n - 1))) }
        }
        switch self {
        case .quick: return ring(0, 7, from: -90, span: 180)
        case .standard, .lidar: return ring(0, 12) + ring(38, 6, from: 30) + ring(-38, 6, from: 30)
        case .full: return ring(0, 12) + ring(32, 10, from: 18) + ring(-32, 10, from: 18) + ring(62, 6, from: 30) + ring(-62, 6, from: 30) + [d(88, 0)]
        }
    }
}

/// 5.6 the walk's capture options (Settings in the capture screen)
enum WalkOptions {
    static var d: UserDefaults { .standard }
    static var high: Bool { d.bool(forKey: "walk.high") }                     // 6K spheres from sharper frames
    static var dense: Bool { d.bool(forKey: "walk.dense") }                   // more, smaller tiles: more overlap, cleaner seams
    static var house: Bool { d.object(forKey: "walk.house") as? Bool ?? true } // the 3D dollhouse
    static var sphereW: Int { high ? 6144 : 4096 }
    static var frameW: Int { high ? 1440 : 1024 }
}

struct WalkNode: Codable { var x: Float; var y: Float; var z: Float; var file: String; var yaw: Float }

enum WalkStitch {
    /// rows of the coverage grid: latitude (degrees) and how many cells round
    static var rows: [(Float, Int)] { WalkOptions.dense ? [(-66, 8), (-36, 14), (-6, 18), (24, 16), (52, 11), (78, 4)] : [(-62, 6), (-28, 10), (5, 12), (38, 10), (70, 5)] }
    /// the direction of a cell's middle
    static func dirOf(_ r: Int, _ c: Int) -> SIMD3<Float> {
        let lat = rows[r].0 * .pi / 180, lon = Float(c) / Float(rows[r].1) * 2 * .pi - .pi
        return SIMD3(cos(lat) * sin(lon), sin(lat), -cos(lat) * cos(lon))
    }
    static func cell(_ fwd: SIMD3<Float>) -> (Int, Int) {
        let lat = asin(max(-1, min(1, fwd.y))) * 180 / .pi, lon = atan2(fwd.x, -fwd.z)
        var best = 0, bd = Float.greatestFiniteMagnitude
        for (i, r) in rows.enumerated() where abs(r.0 - lat) < bd { bd = abs(r.0 - lat); best = i }
        let n = rows[best].1
        let c = Int(((lon + .pi) / (2 * .pi) * Float(n)).rounded()) % n
        return (best, c)
    }
    static var cellCount: Int { rows.reduce(0) { $0 + $1.1 } }

    /// 5.6: the camera changes its exposure as you turn (window vs wall), which made bands. One brightness gain per frame,
    /// solved so frames agree where they overlap (sampled on a coarse sphere), averaging 1
    static func gains(_ frames: [WalkFrame]) -> [Float] {
        let n = frames.count
        guard n > 1 else { return [Float](repeating: 1, count: n) }
        var samples: [[(Int, Float)]] = []
        for y in 0..<48 { for x in 0..<96 {
            let lat = Float.pi / 2 - (Float(y) + 0.5) / 48 * .pi, lon = (Float(x) + 0.5) / 96 * 2 * .pi - .pi
            let d = SIMD3<Float>(cos(lat) * sin(lon), sin(lat), -cos(lat) * cos(lon))
            var here: [(Int, Float)] = []
            for (i, f) in frames.enumerated() {
                let c = f.R.transpose * d, z = -c.z
                guard z > 0.05 else { continue }
                let u = f.cx + f.fx * c.x / z, v = f.cy - f.fy * c.y / z
                guard u >= Float(f.w) * 0.1, v >= Float(f.h) * 0.1, u < Float(f.w) * 0.9, v < Float(f.h) * 0.9 else { continue }
                // a small patch's brightness (one pixel is too noisy)
                var l: Float = 0, k: Float = 0
                for dy in stride(from: -4, through: 4, by: 2) { for dx in stride(from: -4, through: 4, by: 2) {
                    let o = ((Int(v) + dy) * f.w + Int(u) + dx) * 4
                    l += 0.3 * Float(f.rgba[o]) + 0.59 * Float(f.rgba[o + 1]) + 0.11 * Float(f.rgba[o + 2]); k += 1
                } }
                here.append((i, l / k))
            }
            if here.count >= 2 { samples.append(here) }
        } }
        var g = [Float](repeating: 1, count: n)
        for _ in 0..<6 {
            var num = [Float](repeating: 0, count: n), den = [Float](repeating: 0, count: n)
            for sm in samples {
                let total = sm.reduce(Float(0)) { $0 + g[$1.0] * $1.1 }
                for (i, l) in sm where l > 8 && l < 247 {          // not clipped
                    let others = (total - g[i] * l) / Float(sm.count - 1)
                    num[i] += others; den[i] += l
                }
            }
            for i in 0..<n where den[i] > 0 { g[i] = max(0.6, min(1.6, num[i] / den[i])) }
            let mean = g.reduce(0, +) / Float(n); g = g.map { $0 / mean }
        }
        return g
    }

    /// the frames painted onto an equirectangular panorama (W x W/2), feathered where they overlap; what no frame saw
    /// (straight down, where you stood) is filled softly from around it
    static func stitch(_ frames: [WalkFrame], centre: SIMD3<Float>? = nil, W: Int = 4096) -> CGImage? {
        let H = W / 2
        guard !frames.isEmpty else { return nil }
        var out = [SIMD3<Float>](repeating: .zero, count: W * H)
        var known = [Bool](repeating: false, count: W * H)
        let fw = frames.map { $0.fwd }
        let gain = gains(frames)
        // 5.6 DEPTH (no ghosts): the phone moves 30-50 cm while you turn, so pointing alone puts near things in several
        // places. (1) The sphere's own depth: every frame's LiDAR points, seen from the spot's centre, nearest wins, holes
        // filled. (2) Every sphere pixel becomes its real 3D point, and takes its colour from the frames that saw THAT
        // point unblocked (their own depth agrees) - so all frames agree on where things are. What has no depth (far,
        // glass, sky) falls back to pointing alone, where distance makes the movement harmless.
        if let ctr = centre, frames.contains(where: { $0.dw > 0 }) {
            let DW = W / 4, DH = H / 4
            var D = [Float](repeating: .greatestFiniteMagnitude, count: DW * DH)
            for f in frames where f.dw > 0 && f.depth.count == f.dw * f.dh {
                for dy in 0..<f.dh { for dx in 0..<f.dw {
                    let z = f.depth[dy * f.dw + dx]; guard z > 0.1 else { continue }
                    let u = (Float(dx) + 0.5) / Float(f.dw) * Float(f.w), v = (Float(dy) + 0.5) / Float(f.dh) * Float(f.h)
                    let d = f.R * SIMD3<Float>((u - f.cx) / f.fx * z, -(v - f.cy) / f.fy * z, -z) + f.pos - ctr
                    let dist = simd_length(d); guard dist > 0.05 else { continue }
                    let lon = atan2(d.x, -d.z), lat = asin(max(-1, min(1, d.y / dist)))
                    let px = Int((lon + .pi) / (2 * .pi) * Float(DW)), py = Int((.pi / 2 - lat) / .pi * Float(DH))
                    for oy in 0...1 { for ox in 0...1 {
                        let k = min(DH - 1, max(0, py + oy)) * DW + ((px + ox) % DW + DW) % DW
                        if dist < D[k] { D[k] = dist }
                    } }
                } }
            }
            // holes in the sphere's depth: smooth fill from around (unknown stays unknown far from any depth)
            var Dk = D.map { $0 < .greatestFiniteMagnitude }
            if Dk.contains(true) {
                let filled = RoomFill.pushPull(D.map { SIMD3<Float>($0 < .greatestFiniteMagnitude ? $0 : 0, 0, 0) }, Dk, DW, DH)
                // fill only small holes (within ~6 cells of real depth); the rest is "far"
                var near = Dk
                for _ in 0..<6 { var nx = near; for y in 0..<DH { for x in 0..<DW where !near[y * DW + x] {
                    if near[y * DW + (x + 1) % DW] || near[y * DW + (x + DW - 1) % DW] || (y > 0 && near[(y - 1) * DW + x]) || (y < DH - 1 && near[(y + 1) * DW + x]) { nx[y * DW + x] = true }
                } }; near = nx }
                for k in 0..<(DW * DH) where !Dk[k] && near[k] { D[k] = filled[k].x; Dk[k] = true }
            }
            out.withUnsafeMutableBufferPointer { ob in known.withUnsafeMutableBufferPointer { kb in
                let op = ob.baseAddress!, kp = kb.baseAddress!
                DispatchQueue.concurrentPerform(iterations: H) { y in
                    let lat = Float.pi / 2 - (Float(y) + 0.5) / Float(H) * .pi
                    let cl = cos(lat), sl = sin(lat)
                    for x in 0..<W {
                        let k = min(DH - 1, y / 4) * DW + min(DW - 1, x / 4)
                        guard Dk[k] else { continue }
                        // smooth depth inside a surface (bilinear), a clean step at an edge (the nearest cell)
                        let fx = (Float(x) + 0.5) / 4 - 0.5, fy = (Float(y) + 0.5) / 4 - 0.5
                        let x0 = (Int(floor(fx)) + DW) % DW, y0 = max(0, min(DH - 1, Int(floor(fy)))), x1 = (x0 + 1) % DW, y1 = min(DH - 1, y0 + 1)
                        let tx = fx - floor(fx), ty = max(0, min(1, fy - Float(y0)))
                        let q = [y0 * DW + x0, y0 * DW + x1, y1 * DW + x0, y1 * DW + x1]
                        var depth = D[k]
                        if q.allSatisfy({ Dk[$0] }) {
                            let a = D[q[0]], b = D[q[1]], c = D[q[2]], d = D[q[3]]
                            if max(max(a, b), max(c, d)) < min(min(a, b), min(c, d)) * 1.08 { depth = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty }
                        }
                        let lon = (Float(x) + 0.5) / Float(W) * 2 * .pi - .pi
                        let dir = SIMD3<Float>(cl * sin(lon), sl, -cl * cos(lon))
                        let P = ctr + dir * depth
                        var sum = SIMD3<Float>.zero, ws: Float = 0
                        for (i, f) in frames.enumerated() where simd_dot(dir, fw[i]) > 0.35 && f.dw > 0 {
                            let c = f.R.transpose * (P - f.pos), z = -c.z
                            guard z > 0.05 else { continue }
                            let u = f.cx + f.fx * c.x / z, v = f.cy - f.fy * c.y / z
                            guard u >= 0, v >= 0, u < Float(f.w - 1), v < Float(f.h - 1) else { continue }
                            // seen unblocked: this frame's own depth there agrees with the point
                            let fz = f.depth[min(f.dh - 1, Int(v / Float(f.h) * Float(f.dh))) * f.dw + min(f.dw - 1, Int(u / Float(f.w) * Float(f.dw)))]
                            if fz > 0.1 && fz < z - max(0.06, z * 0.06) { continue }
                            let e = min(min(u, Float(f.w - 1) - u) / Float(f.w), min(v, Float(f.h - 1) - v) / Float(f.h)) * 2
                            let e2 = e * e, wgt = e2 * e2 + 1e-9
                            let o = (Int(v) * f.w + Int(u)) * 4
                            sum += SIMD3(Float(f.rgba[o]), Float(f.rgba[o + 1]), Float(f.rgba[o + 2])) * (wgt * gain[i]); ws += wgt
                        }
                        if ws > 0 { op[y * W + x] = sum / ws; kp[y * W + x] = true }
                    }
                }
            } }
        }
        out.withUnsafeMutableBufferPointer { ob in known.withUnsafeMutableBufferPointer { kb in
            let op = ob.baseAddress!, kp = kb.baseAddress!
            DispatchQueue.concurrentPerform(iterations: H) { y in
                let lat = Float.pi / 2 - (Float(y) + 0.5) / Float(H) * .pi
                let cl = cos(lat), sl = sin(lat)
                for x in 0..<W where !kp[y * W + x] {
                    let lon = (Float(x) + 0.5) / Float(W) * 2 * .pi - .pi
                    let d = SIMD3<Float>(cl * sin(lon), sl, -cl * cos(lon))
                    var sum = SIMD3<Float>.zero, wsum: Float = 0
                    for (i, f) in frames.enumerated() where simd_dot(d, fw[i]) > 0.45 {
                        let c = f.R.transpose * d, z = -c.z
                        guard z > 0.05 else { continue }
                        let u = f.cx + f.fx * c.x / z, v = f.cy - f.fy * c.y / z
                        guard u >= 0, v >= 0, u < Float(f.w - 1), v < Float(f.h - 1) else { continue }
                        // feather: weight falls to 0 at the picture's edges
                        let e = min(min(u, Float(f.w - 1) - u) / Float(f.w), min(v, Float(f.h - 1) - v) / Float(f.h)) * 2
                        // the frame that sees a spot most squarely wins it; others only blend in at seams (averaging many blurred)
                        let e2 = e * e, e4 = e2 * e2, wgt = e4 * e4 + 1e-9
                        // 5.7 (47): CoreImage's bitmaps run TOP-down. Reading them bottom-up pasted every photo upside down into its own
                        // place (things "in the bed", flipped pictures): 9 dB against a true panorama, 18 dB fixed
                        let o = (Int(v) * f.w + Int(u)) * 4
                        sum += SIMD3(Float(f.rgba[o]), Float(f.rgba[o + 1]), Float(f.rgba[o + 2])) * (wgt * gain[i]); wsum += wgt
                    }
                    if wsum > 0 { op[y * W + x] = sum / wsum; kp[y * W + x] = true }
                }
            }
        } }
        if known.contains(false) { out = RoomFill.pushPull(out, known, W, H) }
        var px = [UInt8](repeating: 255, count: W * H * 4)
        for i in 0..<(W * H) { let c = out[i]; px[i * 4] = u8(c.x); px[i * 4 + 1] = u8(c.y); px[i * 4 + 2] = u8(c.z) }
        guard let prov = CGDataProvider(data: Data(px) as CFData) else { return nil }
        return CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: W * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: prov, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

final class WalkCapture: NSObject, ObservableObject, ARSessionDelegate {
    @Published var ready = false
    @Published var turning = false               // capturing the current spot
    @Published var covered = Set<Int>()          // cells done at this spot (row * 100 + cell)
    @Published var spots = 0
    @Published var hint = "Stand where the first view should be, then tap Capture spot"
    @Published var lastPos = SIMD3<Float>.zero
    private(set) var nodes: [WalkNode] = []
    private var frames: [WalkFrame] = []
    private var spotPos = SIMD3<Float>.zero, spotYaw: Float = 0
    private var lastFwd = SIMD3<Float>(0, 0, -1), lastShot: TimeInterval = 0
    let dir: URL
    let capName = "walk-" + UUID().uuidString            // 5.6: every frame also with its LiDAR depth, for the 3D dollhouse
    private var capDir: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("captures").appendingPathComponent(capName) }
    private var keyIdx = 0
    private var spotIdx = 0                      // 5.7 (48): next sphere file number, taken at finish (nodes.count lags the stitch)
    private let work = DispatchQueue(label: "walk.stitch", qos: .userInitiated)
    private let io = DispatchQueue(label: "walk.io", qos: .userInitiated)

    override init() {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("walk-" + UUID().uuidString)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        super.init()
        try? FileManager.default.createDirectory(at: capDir.appendingPathComponent("frames"), withIntermediateDirectories: true)
    }
    func start(_ s: ARSession) {
        s.delegate = self
        let c = ARWorldTrackingConfiguration()
        c.worldAlignment = .gravity
        var sem: ARConfiguration.FrameSemantics = []
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) { sem.insert(.sceneDepth) }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) { sem.insert(.smoothedSceneDepth) }
        c.frameSemantics = sem
        s.run(c)
    }
    // 5.7 (48): the capture folder is kept only once a saved walk points at it; closing with X (spots or not) left
    // hundreds of MB of LiDAR frames nothing could reach. Removed on io, after any frame still being written.
    private var saved = false
    deinit { if !saved { let c = capDir; io.async { try? FileManager.default.removeItem(at: c) } } }
    // ---- 5.6 Matterport-style capture: dots in the room; aim the ring at a dot, hold still, it captures ----
    @Published var done = 0
    @Published var total = 0
    @Published var aim: Double = 0               // how far through holding still on a dot (0...1)
    @Published var wander = false                // drifted a long way from the spot (a gentle tip, never a block)
    weak var root: SCNNode?
    weak var view: ARSCNView?
    // 5.7 (45): where the next dot is on screen, for the guide line from the ring (nil = none left); off = it is off-screen
    @Published var nextDot: CGPoint?
    @Published var nextOff = false
    private var dotPos: [SIMD3<Float>] = []
    private var targets: [SIMD3<Float>] = [], taken: [Bool] = [], dots: [SCNNode] = []
    private var aimStart: TimeInterval = 0, aimed = -1
    private var started = false

    func beginSpot() {
        frames = []; turning = true; started = false; done = 0; aim = 0; wander = false
        hint = "Aim the ring at a dot and hold still"
    }
    private func placeDots(_ pos: SIMD3<Float>, yaw: Float) {
        dots.forEach { $0.removeFromParentNode() }
        targets = WalkMode.current.targets(yaw: yaw)
        taken = [Bool](repeating: false, count: targets.count); total = targets.count
        dots = targets.map { d in
            let n = SCNNode(geometry: SCNSphere(radius: 0.045))
            let m = SCNMaterial(); m.diffuse.contents = UIColor.white; m.lightingModel = .constant; m.readsFromDepthBuffer = false
            n.geometry?.materials = [m]; n.renderingOrder = 50
            n.simdPosition = pos + d * 1.6
            root?.addChildNode(n)
            return n
        }
        dotPos = targets.map { pos + $0 * 1.6 }
    }
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if !ready { DispatchQueue.main.async { self.ready = true } }
        let T = frame.camera.transform, pos = SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z)
        if simd_distance(pos, lastPos) > 0.05 { DispatchQueue.main.async { self.lastPos = pos } }
        guard turning, case .normal = frame.camera.trackingState else { return }
        let R = simd_float3x3(SIMD3(T.columns.0.x, T.columns.0.y, T.columns.0.z), SIMD3(T.columns.1.x, T.columns.1.y, T.columns.1.z), SIMD3(T.columns.2.x, T.columns.2.y, T.columns.2.z))
        let fwd = -R.columns.2
        if !started {
            started = true; spotPos = pos
            // the dots start where you are facing, level
            spotYaw = atan2(fwd.x, -fwd.z)
            DispatchQueue.main.async { self.placeDots(pos, yaw: self.spotYaw) }
            return
        }
        let w = simd_distance(pos, spotPos) > 0.15          // 5.7 (46): moving the phone is what makes seams; tip sooner (still never a block)
        if w != wander { DispatchQueue.main.async { self.wander = w } }
        guard !targets.isEmpty else { return }
        // the dot the ring is on: within 5 degrees, measured from where the phone IS (5.7 (45): it used the direction from
        // where the spot began, so a step aside made the ring take a shot beside the dot and refuse the dot itself)
        var on = -1, next = -1, best: Float = -2
        for i in targets.indices where !taken[i] && i < dotPos.count {
            let c = simd_dot(simd_normalize(dotPos[i] - pos), fwd)
            if c > 0.9962 && on < 0 { on = i }
            if c > best { best = c; next = i }
        }
        // the guide line: to the nearest dot still to take
        if let v = view, next >= 0 {
            let p = dotPos[next], sp = v.projectPoint(SCNVector3(p.x, p.y, p.z))
            let ahead = simd_dot(p - pos, fwd) > 0
            var pt = CGPoint(x: CGFloat(sp.x), y: CGFloat(sp.y))
            let b = v.bounds, mid = CGPoint(x: b.midX, y: b.midY)
            if !ahead { pt = CGPoint(x: 2 * mid.x - pt.x, y: 2 * mid.y - pt.y) }      // behind you: point the other way
            let off = !ahead || !b.insetBy(dx: 24, dy: 60).contains(pt)
            if off {                                                   // clamp to the screen edge along the same direction
                let dx = pt.x - mid.x, dy = pt.y - mid.y, k = min(abs((b.width / 2 - 30) / max(1, abs(dx))), abs((b.height / 2 - 80) / max(1, abs(dy))))
                pt = CGPoint(x: mid.x + dx * min(1, k), y: mid.y + dy * min(1, k))
            }
            if on >= 0 { pt = mid }
            DispatchQueue.main.async { self.nextDot = pt; self.nextOff = off && on < 0 }
        } else if next < 0 { DispatchQueue.main.async { self.nextDot = nil } }
        let steady = acos(max(-1, min(1, simd_dot(fwd, lastFwd)))) < 0.012
        lastFwd = fwd
        if on != aimed { aimed = on; aimStart = frame.timestamp }
        let held = on >= 0 && steady ? min(1, (frame.timestamp - aimStart) / 0.45) : 0
        if abs(held - aim) > 0.05 || (held == 0 && aim != 0) { DispatchQueue.main.async { self.aim = held } }
        if on < 0 || !steady { if !steady { aimStart = frame.timestamp }; return }
        guard held >= 1, frame.timestamp - lastShot > 0.25 else { return }
        lastShot = frame.timestamp
        taken[on] = true; aimed = -1
        let i = on
        let ci = CIImage(cvPixelBuffer: frame.capturedImage)
        let W = WalkOptions.frameW, H = Int(Float(W) * Float(ci.extent.height) / Float(ci.extent.width)), sc = CGFloat(W) / ci.extent.width
        var px = [UInt8](repeating: 0, count: W * H * 4)
        sharedCI.render(ci.transformed(by: CGAffineTransform(scaleX: sc, y: sc)), toBitmap: &px, rowBytes: W * 4, bounds: CGRect(x: 0, y: 0, width: W, height: H), format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        let K = frame.camera.intrinsics, s = Float(sc)
        var wf = WalkFrame(rgba: px, w: W, h: H, R: R, fx: K[0][0] * s, fy: K[1][1] * s, cx: K[2][0] * s, cy: K[2][1] * s)
        wf.pos = pos
        // 5.7 (47): every spot keeps its LiDAR depth, so the sphere puts near things where they really are as the phone
        // moves round you (measured on a test house: 18 dB pointing-only, 24 dB with depth)
        if let dep = frame.smoothedSceneDepth ?? frame.sceneDepth, let dd = copyPlane(dep.depthMap, bytesPerPixel: 4) {
            let dw = CVPixelBufferGetWidth(dep.depthMap), dh = CVPixelBufferGetHeight(dep.depthMap)
            var d = [Float](repeating: 0, count: dw * dh)
            dd.withUnsafeBytes { b in for k in 0..<d.count { d[k] = b.loadUnaligned(fromByteOffset: k * 4, as: Float.self) } }
            if let cm = dep.confidenceMap, let cd = copyPlane(cm, bytesPerPixel: 1), cd.count >= d.count { for k in 0..<d.count where cd[k] < 1 { d[k] = 0 } }
            wf.depth = d; wf.dw = dw; wf.dh = dh
        }
        frames.append(wf)
        // LiDAR 3D: every shot also with its depth, for the dollhouse
        if WalkMode.current == .lidar, let dep = frame.smoothedSceneDepth ?? frame.sceneDepth, let dd = copyPlane(dep.depthMap, bytesPerPixel: 4), let cm = dep.confidenceMap, let cd = copyPlane(cm, bytesPerPixel: 1) {
            let img = frame.capturedImage, dw = CVPixelBufferGetWidth(dep.depthMap), dh = CVPixelBufferGetHeight(dep.depthMap), ts = frame.timestamp, idx = keyIdx, cdir = capDir
            keyIdx += 1
            let sharp = frameSharpness(img)
            io.async { Photo3DCapture.write(img, dd, cd, dw, dh, T, K, sharp, ts, idx, cdir) }
        }
        DispatchQueue.main.async {
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            if i < self.dots.count {
                let n = self.dots[i]
                SCNTransaction.begin(); SCNTransaction.animationDuration = 0.25; n.opacity = 0; n.scale = SCNVector3(2, 2, 2)
                SCNTransaction.completionBlock = { n.removeFromParentNode() }; SCNTransaction.commit()
            }
            self.done += 1; self.aim = 0
            if self.done >= self.total { self.finishSpot() }
        }
    }
    var coverage: Double { total == 0 ? 0 : Double(done) / Double(total) }

    /// stitches this spot's sphere in the background and adds it to the walk
    func finishSpot() {
        guard turning else { return }
        turning = false
        dots.forEach { $0.removeFromParentNode() }; dots = []; targets = []; dotPos = []; nextDot = nil
        // the spot is where the phone was on average while turning (the middle of its circle round you)
        let fr = frames, pos = frames.isEmpty ? spotPos : frames.reduce(SIMD3<Float>.zero) { $0 + $1.pos } / Float(frames.count), yaw = spotYaw, idx = spotIdx, d = dir
        spotIdx += 1; frames = []
        hint = "Making the sphere..."
        work.async {
            // 5.7 (47): depth-based stitching. It was switched off in 5.6 as "a mess", but that test ran with every photo read
            // upside down; with the rows right, depth beats pointing-only by 6 dB on the test house
            let img = WalkStitch.stitch(fr, centre: pos, W: WalkOptions.sphereW)
            let name = String(format: "spot%03d.jpg", idx)
            if let im = img, let jpg = UIImage(cgImage: im).jpegData(compressionQuality: 0.88) { try? jpg.write(to: d.appendingPathComponent(name)) }
            DispatchQueue.main.async {
                if img != nil { self.nodes.append(WalkNode(x: pos.x, y: pos.y, z: pos.z, file: name, yaw: yaw)); self.spots = self.nodes.count }
                self.hint = img == nil ? "That spot did not work - try again" : "Spot \(self.nodes.count) done. Walk 2-4 steps and capture the next, or Save"
            }
        }
    }
    @Published var building: Double?
    /// waits for any sphere still being made, writes the walk, then builds its 3D dollhouse from the LiDAR frames
    func save(done: @escaping (ScanItem?) -> Void) {
        work.async { self.io.async {
            DispatchQueue.main.async {
                guard !self.nodes.isEmpty, let d = try? JSONEncoder().encode(self.nodes) else { done(nil); return }
                try? d.write(to: self.dir.appendingPathComponent("walk.json"))
                guard let it = Library.shared.addWalk(from: self.dir, name: "Walk " + Date().formatted(date: .abbreviated, time: .shortened), spots: self.nodes.count) else { done(nil); return }
                Library.shared.setCapture(it, self.capName); self.saved = true
                var item = it; item.capture = self.capName
                // 5.7 (48): build from what was captured (any LiDAR frames written), not the mode picked at Save time
                guard self.keyIdx > 0 else { done(item); return }
                self.building = 0; self.hint = "Building the 3D dollhouse..."
                let ns = self.nodes
                DispatchQueue.global(qos: .userInitiated).async {
                    WalkHouse.build(item, nodes: ns) { p, s in DispatchQueue.main.async { self.building = Double(p); self.hint = s } }
                    DispatchQueue.main.async { self.building = nil; done(item) }
                }
            }
        } }
    }
}

struct WalkARView: UIViewRepresentable {
    let cap: WalkCapture
    func makeUIView(context: Context) -> ARSCNView { let v = ARSCNView(frame: .zero); cap.root = v.scene.rootNode; cap.view = v; cap.start(v.session); return v }
    func updateUIView(_ v: ARSCNView, context: Context) {}
    static func dismantleUIView(_ v: ARSCNView, coordinator: ()) { v.session.pause() }
}

struct WalkCaptureScreen: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var cap = WalkCapture()
    @State private var saving = false
    @State private var open: ScanItem?
    @State private var showHelp = !UserDefaults.standard.bool(forKey: "walk.seenHelp")
    @State private var showOptions = false
    var body: some View {
        ZStack {
            WalkARView(cap: cap).ignoresSafeArea()
            if cap.turning, let t = cap.nextDot {
                // 5.7 (45) the guide line: from the ring to the next dot (an arrow at the edge when it is off-screen)
                GeometryReader { g in
                    let c = CGPoint(x: g.size.width / 2, y: g.size.height / 2)
                    let d = hypot(t.x - c.x, t.y - c.y)
                    if d > 40 {
                        let ux = (t.x - c.x) / d, uy = (t.y - c.y) / d
                        Path { p in p.move(to: CGPoint(x: c.x + ux * 36, y: c.y + uy * 36)); p.addLine(to: t) }
                            .stroke(.white.opacity(0.85), style: StrokeStyle(lineWidth: 3, lineCap: .round, dash: [8, 7]))
                        if cap.nextOff {
                            Image(systemName: "chevron.right.circle.fill").font(.title).foregroundStyle(.white)
                                .rotationEffect(.radians(atan2(uy, ux))).position(t)
                        }
                    }
                }.ignoresSafeArea().allowsHitTesting(false).shadow(radius: 2)
            }
            if cap.turning {
                // the aiming ring (Matterport-style): put a dot inside it and hold still - it fills, then captures
                ZStack {
                    Circle().stroke(.white.opacity(0.9), lineWidth: 3).frame(width: 64, height: 64)
                    Circle().trim(from: 0, to: cap.aim).stroke(Color.green, style: StrokeStyle(lineWidth: 5, lineCap: .round)).rotationEffect(.degrees(-90)).frame(width: 64, height: 64)
                }.shadow(radius: 3).allowsHitTesting(false)
            }
            VStack(spacing: 10) {
                HStack {
                    Button { dismiss() } label: { Image(systemName: "xmark").font(.title3.bold()).frame(width: 44, height: 44).background(.ultraThinMaterial, in: Circle()) }
                    Spacer()
                    Text("Walk - \(cap.spots) spot\(cap.spots == 1 ? "" : "s")").font(.headline).padding(.horizontal, 12).padding(.vertical, 8).background(.ultraThinMaterial, in: Capsule())
                    Spacer()
                    Button { showOptions = true } label: { Image(systemName: "gearshape").font(.title3).frame(width: 44, height: 44).background(.ultraThinMaterial, in: Circle()) }.disabled(cap.turning)
                    Button { showHelp = true } label: { Image(systemName: "questionmark").font(.title3.bold()).frame(width: 44, height: 44).background(.ultraThinMaterial, in: Circle()) }
                }.padding(.horizontal)
                Spacer()
                if cap.turning {
                    Text("\(cap.done) of \(cap.total)").font(.headline.monospacedDigit()).foregroundStyle(.white).shadow(radius: 3)
                    if cap.wander { Text("Tip: turn the phone on the spot - keep it in one place, like turning a camera on a tripod").font(.caption).foregroundStyle(.white).padding(6).background(.ultraThinMaterial, in: Capsule()) }
                }
                if let b = cap.building { ProgressView(value: b).frame(width: 240).tint(.white) }
                Text(cap.hint).font(.footnote).multilineTextAlignment(.center).padding(10).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12)).padding(.horizontal)
                HStack(spacing: 14) {
                    if cap.turning {
                        Button { cap.finishSpot() } label: { Text(cap.done >= cap.total ? "Done" : "Finish spot now").frame(minWidth: 170) }
                            .buttonStyle(.bordered).disabled(cap.done < 2)
                    } else {
                        Button { cap.beginSpot() } label: { Text(cap.spots == 0 ? "Start - \(WalkMode.current.rawValue)" : "Next spot").frame(minWidth: 150) }
                            .buttonStyle(.borderedProminent).disabled(!cap.ready || saving)
                        if cap.spots > 0 {
                            Button { saving = true; cap.save { it in saving = false; if let it { open = it } } } label: { Text(saving ? "Building..." : "Save") }
                                .buttonStyle(.bordered).disabled(saving)
                        }
                    }
                }.padding(.bottom, 30)
            }
        }
        .background(Color.black)
        .onAppear { Awake.set("walk", true) }.onDisappear { Awake.set("walk", false) }
        .fullScreenCover(item: $open, onDismiss: { dismiss() }) { WalkViewer(item: $0) }
        .sheet(isPresented: $showHelp, onDismiss: { UserDefaults.standard.set(true, forKey: "walk.seenHelp") }) { WalkHelp() }
        .sheet(isPresented: $showOptions) { WalkOptionsView() }
    }
}

/// how to capture a walk well (shown the first time, and from the ? button)
struct WalkHelp: View {
    @Environment(\.dismiss) private var dismiss
    private let steps: [(String, String, String)] = [
        ("circle.circle", "Aim at the dots", "Dots appear round you. Put a dot inside the ring in the middle and hold still - the ring fills and it takes the shot. Then the next dot."),
        ("arrow.triangle.2.circlepath", "Turn on the spot", "Turning your feet round, phone close to you, gives the cleanest joins. A little movement is fine."),
        ("figure.walk", "Walk a few steps, Next spot", "1-3 metres, keeping the last spot in view. Doorways: one spot each side."),
        ("slider.horizontal.3", "Pick how much to capture", "Quick 180°, 360°, Full sphere, or LiDAR 3D for a dollhouse you can measure - in the gear menu."),
        ("cube.transparent", "Save", "Then walk through it, or see the dollhouse and floor plan (LiDAR 3D).")]
    var body: some View {
        NavigationStack {
            List {
                ForEach(steps, id: \.1) { s in
                    HStack(alignment: .top, spacing: 14) {
                        Image(systemName: s.0).font(.title2).foregroundStyle(.teal).frame(width: 34)
                        VStack(alignment: .leading, spacing: 3) { Text(s.1).font(.headline); Text(s.2).font(.subheadline).foregroundStyle(.secondary) }
                    }.padding(.vertical, 4)
                }
            }
            .navigationTitle("How to capture a walk").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Got it") { dismiss() }.bold() } }
        }
    }
}

struct WalkOptionsView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage("walk.mode") private var mode = WalkMode.standard.rawValue
    @AppStorage("walk.high") private var high = false
    var body: some View {
        NavigationStack {
            Form {
                Section("Capture type") {
                    ForEach(WalkMode.allCases, id: \.self) { m in
                        Button { mode = m.rawValue } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) { Text(m.rawValue).foregroundStyle(.primary); Text(m.detail).font(.caption).foregroundStyle(.secondary) }
                                Spacer()
                                if mode == m.rawValue { Image(systemName: "checkmark").foregroundStyle(.teal) }
                            }
                        }
                    }
                }
                Section { Picker("Sphere quality", selection: $high) { Text("Standard (4K)").tag(false); Text("High (6K)").tag(true) } } footer: { Text("High: sharper shots and a 6K sphere, a little slower to make.") }
            }
            .navigationTitle("Walk options").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() }.bold() } }
        }
    }
}

/// the cells of the sphere: filled as they are captured (rings from low to high)
struct CoverageRing: View {
    let covered: Set<Int>
    var body: some View {
        Canvas { ctx, size in
            let c = CGPoint(x: size.width / 2, y: size.height / 2), R = min(size.width, size.height) / 2
            for (ri, row) in WalkStitch.rows.enumerated() {
                let r0 = R * CGFloat(ri) / CGFloat(WalkStitch.rows.count), r1 = R * CGFloat(ri + 1) / CGFloat(WalkStitch.rows.count)
                for k in 0..<row.1 {
                    let a0 = Angle.degrees(Double(k) / Double(row.1) * 360 - 90), a1 = Angle.degrees(Double(k + 1) / Double(row.1) * 360 - 90)
                    var p = Path(); p.addArc(center: c, radius: r1 - 2, startAngle: a0, endAngle: a1, clockwise: false); p.addArc(center: c, radius: r0 + 2, startAngle: a1, endAngle: a0, clockwise: true); p.closeSubpath()
                    ctx.fill(p, with: .color(covered.contains(ri * 100 + k) ? .green : .white.opacity(0.25)))
                }
            }
        }
        .background(Circle().fill(.black.opacity(0.35)))
    }
}

/// 5.6 the dollhouse: a photoreal 3D model of the whole walk (shape from every frame's LiDAR depth, look from the
/// photos), in the same world as the spheres - so a move between spots can fly through it, and distances are real
enum WalkHouse {
    static func url(_ id: UUID) -> URL { Library.walkURL(id).appendingPathComponent("house.scn") }
    static func build(_ item: ScanItem, nodes: [WalkNode], progress: @escaping (Float, String) -> Void) {
        let ps = nodes.map { SIMD3<Float>($0.x, $0.y, $0.z) }
        var lo = ps[0], hi = ps[0]; for p in ps { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let ext = max(4, simd_reduce_max(hi - lo) + 6)
        guard let t = PhotoModel.build(item, extent: ext, detail: 1, minWeight: 0.8, progress: progress) else { return }
        let g = t.geometry(centre: .zero)                      // world coordinates, like the spots
        let scene = SCNScene(); let n = SCNNode(geometry: g); n.name = "house"; scene.rootNode.addChildNode(n)
        _ = scene.write(to: url(item.id), options: nil, delegate: nil, progressHandler: nil)
    }
}

// ---------- the viewer ----------
// Three ways to see a walk: Inside (stand in a sphere, arrows to the other spots), Dollhouse (the 3D model from
// outside, cut away so you see into the rooms, every spot marked) and Plan (straight down, to scale). Moving between
// spots flies through the 3D model; Measure gives real distances off the LiDAR model.
final class WalkWorld: NSObject, ObservableObject {
    enum Mode: String, CaseIterable { case inside = "Inside", dollhouse = "Dollhouse", plan = "Plan" }
    @Published var mode: Mode = .inside { didSet { apply() } }
    @Published var measuring = false { didSet { clearMeasure() } }
    @Published var gyro = false { didSet { gyroChanged() } }
    @Published var readout = ""
    @Published var hasHouse = false
    var nodes: [WalkNode] = []
    var at = 0
    var dir: URL!
    let scene = SCNScene()
    let cam = SCNNode(), outCam = SCNNode(), planCam = SCNNode()
    let sphere = SCNNode(), arrows = SCNNode(), markers = SCNNode(), measure = SCNNode()
    var house: SCNNode?
    var yaw: Float = 0, pitch: Float = 0, fov: CGFloat = 70
    weak var view: SCNView?
    private var flying = false
    private var mark: SIMD3<Float>?
    private let motion = CMMotionManager()
    private var gyroBase: (Float, Double) = (0, 0)
    private var centre = SIMD3<Float>.zero, size: Float = 5

    func spot(_ i: Int) -> SIMD3<Float> { SIMD3(nodes[i].x, nodes[i].y, nodes[i].z) }
    func load(_ dir: URL) {
        self.dir = dir
        nodes = (try? JSONDecoder().decode([WalkNode].self, from: Data(contentsOf: dir.appendingPathComponent("walk.json")))) ?? []
        sphere.geometry = WalkSceneView.insideSphere(); sphere.renderingOrder = -10
        sphere.geometry?.firstMaterial?.readsFromDepthBuffer = false; sphere.geometry?.firstMaterial?.writesToDepthBuffer = false
        for n in [sphere, arrows, markers, measure] { scene.rootNode.addChildNode(n) }
        cam.camera = SCNCamera(); cam.camera?.zNear = 0.03; cam.camera?.zFar = 60
        outCam.camera = SCNCamera(); outCam.camera?.zNear = 0.05; outCam.camera?.zFar = 200; outCam.camera?.fieldOfView = 50
        planCam.camera = SCNCamera(); planCam.camera?.usesOrthographicProjection = true; planCam.camera?.zNear = 0.05; planCam.camera?.zFar = 200
        for c in [cam, outCam, planCam] { scene.rootNode.addChildNode(c) }
        let hu = dir.appendingPathComponent("house.scn")
        if let hs = try? SCNScene(url: hu, options: nil), let h = hs.rootNode.childNode(withName: "house", recursively: true) ?? hs.rootNode.childNodes.first {
            h.removeFromParentNode(); scene.rootNode.addChildNode(h); house = h; hasHouse = true
            // seen from outside, faces turned away are not drawn: ceilings and the near walls vanish (the dollhouse cut)
            h.geometry?.materials.forEach { $0.isDoubleSided = false; $0.cullMode = .back }
            let (lo, hi) = h.boundingBox
            centre = SIMD3(Float(lo.x + hi.x) / 2, Float(lo.y + hi.y) / 2, Float(lo.z + hi.z) / 2)
            size = max(2, simd_length(SIMD3<Float>(Float(hi.x - lo.x), Float(hi.y - lo.y), Float(hi.z - lo.z))))
        } else if !nodes.isEmpty {
            var lo = spot(0), hi = spot(0); for i in nodes.indices { lo = simd_min(lo, spot(i)); hi = simd_max(hi, spot(i)) }
            centre = (lo + hi) / 2; size = max(4, simd_length(hi - lo) + 4)
        }
        if let f = nodes.first { yaw = -f.yaw }
        buildMarkers()
        show(0, fly: false)
        apply()
    }
    /// the floor under a spot: straight down onto the model (1.35 m below the eye without one)
    func floor(_ p: SIMD3<Float>) -> Float {
        if let h = house {
            let hits = scene.rootNode.hitTestWithSegment(from: SCNVector3(p.x, p.y - 0.3, p.z), to: SCNVector3(p.x, p.y - 2.6, p.z), options: [SCNHitTestOption.ignoreHiddenNodes.rawValue: false, SCNHitTestOption.backFaceCulling.rawValue: false])
            if let f = hits.first(where: { $0.node === h || $0.node.parent === h }) { return Float(f.worldCoordinates.y) }
        }
        return p.y - 1.35
    }
    private func buildMarkers() {
        markers.childNodes.forEach { $0.removeFromParentNode() }
        for i in nodes.indices {
            let p = spot(i)
            let disc = SCNNode(geometry: SCNCylinder(radius: 0.16, height: 0.012))
            let m = SCNMaterial(); m.diffuse.contents = UIColor.systemBlue; m.lightingModel = .constant      // 5.7 (46): was drawn white under "tap a blue spot"
            disc.geometry?.materials = [m]
            disc.simdPosition = SIMD3(p.x, floor(p) + 0.02, p.z); disc.name = "spot\(i)"
            markers.addChildNode(disc)
        }
    }
    func show(_ i: Int, fly: Bool) {
        guard i < nodes.count, !flying else { return }
        let from = cam.simdPosition, to = spot(i)
        let img = UIImage(contentsOfFile: dir.appendingPathComponent(nodes[i].file).path)
        at = i
        clearMeasure()
        if fly, let h = house, mode == .inside {
            // Matterport-style: the sphere fades to the 3D model, the camera glides there, the next sphere fades in
            flying = true; arrows.isHidden = true
            h.isHidden = false
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0.25
            sphere.opacity = 0
            SCNTransaction.completionBlock = {
                SCNTransaction.begin(); SCNTransaction.animationDuration = min(1.4, 0.5 + Double(simd_distance(from, to)) * 0.25)
                SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                self.cam.simdPosition = to
                SCNTransaction.completionBlock = {
                    self.sphere.simdPosition = to; self.sphere.geometry?.firstMaterial?.diffuse.contents = img
                    self.placeArrows()
                    SCNTransaction.begin(); SCNTransaction.animationDuration = 0.3
                    self.sphere.opacity = 1
                    // 5.7 (48): only if still Inside - switching to Dollhouse/Plan mid-flight must keep the model and hide the arrows
                    SCNTransaction.completionBlock = { if self.mode == .inside { h.isHidden = true; self.arrows.isHidden = false }; self.flying = false }
                    SCNTransaction.commit()
                }
                SCNTransaction.commit()
            }
            SCNTransaction.commit()
        } else {
            cam.simdPosition = to; sphere.simdPosition = to
            SCNTransaction.begin(); SCNTransaction.animationDuration = fly ? 0.35 : 0
            sphere.geometry?.firstMaterial?.diffuse.contents = img
            SCNTransaction.commit()
            placeArrows()
        }
    }
    private func placeArrows() {
        arrows.childNodes.forEach { $0.removeFromParentNode() }
        let here = spot(at), fl = floor(here)
        let others = nodes.indices.filter { $0 != at }.sorted { simd_distance(spot($0), here) < simd_distance(spot($1), here) }.prefix(6)
        for j in others {
            var d = spot(j) - here; d.y = 0
            let dist = simd_length(d); guard dist > 0.1 else { continue }
            d /= dist
            let a = SCNNode(geometry: WalkSceneView.arrow()); a.name = "spot\(j)"
            let r = min(2.2, max(1.0, dist * 0.6))
            a.simdPosition = SIMD3(here.x + d.x * r, fl + 0.03, here.z + d.z * r)
            a.simdOrientation = simd_quatf(angle: atan2(-d.x, -d.z), axis: SIMD3(0, 1, 0)) * simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
            arrows.addChildNode(a)
        }
    }
    func aim() {
        cam.simdOrientation = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)) * simd_quatf(angle: pitch, axis: SIMD3(1, 0, 0))
        cam.camera?.fieldOfView = fov
    }
    func apply() {
        guard let v = view else { return }
        let inside = mode == .inside
        sphere.isHidden = !inside; arrows.isHidden = !inside; markers.isHidden = inside
        house?.isHidden = inside
        v.allowsCameraControl = mode == .dollhouse            // inside and plan use their own drag and pinch
        switch mode {
        case .inside: v.pointOfView = cam; aim()
        case .dollhouse:
            outCam.simdPosition = centre + SIMD3(size * 0.45, size * 0.75, size * 0.85); outCam.simdLook(at: centre)
            v.pointOfView = outCam
            v.defaultCameraController.target = SCNVector3(centre.x, centre.y, centre.z)
        case .plan:
            planCam.simdPosition = centre + SIMD3(0, size * 2, 0); planCam.simdLook(at: centre, up: SIMD3(0, 0, -1), localFront: SIMD3(0, 0, -1))
            planCam.camera?.orthographicScale = Double(size) * 0.55
            v.pointOfView = planCam
        }
        readout = inside ? "" : (hasHouse ? "Tap a blue spot to go inside" : "No 3D model for this walk (made before 5.6)")
    }
    // ---- measuring: two taps on the real (LiDAR) surfaces ----
    private func clearMeasure() { mark = nil; measure.childNodes.forEach { $0.removeFromParentNode() }; if measuring { readout = hasHouse ? "Tap the first point" : "Measuring needs the 3D model" } else if mode == .inside { readout = "" } }
    private func surfacePoint(_ pt: CGPoint) -> SIMD3<Float>? {
        guard let v = view, let h = house else { return nil }
        let was = h.isHidden; h.isHidden = false; defer { h.isHidden = was }
        let hits = v.hitTest(pt, options: [.ignoreHiddenNodes: false, .searchMode: SCNHitTestSearchMode.all.rawValue, .backFaceCulling: false])
        return hits.first(where: { $0.node === h || $0.node.parent === h }).map { SIMD3(Float($0.worldCoordinates.x), Float($0.worldCoordinates.y), Float($0.worldCoordinates.z)) }
    }
    private func dot(_ p: SIMD3<Float>) {
        let n = SCNNode(geometry: SCNSphere(radius: 0.02)); n.geometry?.firstMaterial?.diffuse.contents = UIColor.systemYellow
        n.geometry?.firstMaterial?.readsFromDepthBuffer = false; n.renderingOrder = 10; n.simdPosition = p; measure.addChildNode(n)
    }
    func tap(_ pt: CGPoint) {
        guard let v = view, !flying else { return }
        if measuring && mode == .inside {
            guard let p = surfacePoint(pt) else { readout = "No surface there"; return }
            if let a = mark {
                dot(p)
                let line = SCNNode(geometry: SCNGeometry(sources: [SCNGeometrySource(vertices: [SCNVector3(a.x, a.y, a.z), SCNVector3(p.x, p.y, p.z)])], elements: [SCNGeometryElement(indices: [UInt16(0), 1], primitiveType: .line)]))
                line.geometry?.firstMaterial?.diffuse.contents = UIColor.systemYellow; line.geometry?.firstMaterial?.readsFromDepthBuffer = false; line.renderingOrder = 10
                measure.addChildNode(line)
                let d = simd_distance(a, p)
                readout = !Prefs.metric ? String(format: "%.1f in  (%.2f ft)", d * 39.37, d * 3.281) : (d < 1 ? String(format: "%.1f cm", d * 100) : String(format: "%.2f m", d))
                mark = nil
            } else { measure.childNodes.forEach { $0.removeFromParentNode() }; mark = p; dot(p); readout = "Now the second point" }
            return
        }
        for h in v.hitTest(pt, options: [.searchMode: SCNHitTestSearchMode.all.rawValue]) {
            var n: SCNNode? = h.node
            while let x = n, !(x.name ?? "").hasPrefix("spot") { n = x.parent }
            if let s = n?.name, let i = Int(s.dropFirst(4)) {
                if mode == .inside { show(i, fly: true) }
                else {
                    // fly from the dollhouse down into that spot
                    let p = spot(i)
                    cam.simdPosition = v.pointOfView?.simdWorldPosition ?? p
                    cam.simdOrientation = v.pointOfView?.simdWorldOrientation ?? cam.simdOrientation
                    v.pointOfView = cam; v.allowsCameraControl = false; markers.isHidden = true
                    SCNTransaction.begin(); SCNTransaction.animationDuration = 1.1
                    SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                    cam.simdPosition = p; cam.simdOrientation = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))
                    SCNTransaction.completionBlock = { self.pitch = 0; self.show(i, fly: false); self.mode = .inside }
                    SCNTransaction.commit()
                }
                return
            }
        }
    }
    func pan(_ t: CGPoint) {
        // 5.7 (46): the plan is a flat top-down view; SceneKit's own orbit/zoom did nothing useful there - it slides and zooms here
        if mode == .plan, let s = planCam.camera?.orthographicScale, let v = view {
            let k = Float(2 * s) / Float(max(1, v.bounds.height))
            planCam.simdPosition += SIMD3(-Float(t.x) * k, 0, -Float(t.y) * k); return
        }
        guard mode == .inside, !gyro else { return }
        let k = Float(fov / 70) * 0.005
        yaw += Float(t.x) * k; pitch = max(-1.45, min(1.45, pitch + Float(t.y) * k)); aim()
    }
    func pinch(_ s: CGFloat) {
        if mode == .plan, let c = planCam.camera { c.orthographicScale = max(0.5, min(Double(size) * 2, c.orthographicScale / Double(s))); return }
        guard mode == .inside else { return }; fov = max(30, min(100, fov / s)); aim()
    }
    // ---- look round by moving the phone ----
    private func gyroChanged() {
        if gyro, motion.isDeviceMotionAvailable {
            // 5.7 (46): straight from the attitude quaternion. Euler yaw/pitch flip over when the phone tilts past
            // upright, so looking above the horizon spun the view sideways instead of up.
            var off: simd_quatf? = nil
            motion.deviceMotionUpdateInterval = 1 / 60
            motion.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] m, _ in
                guard let self, let q = m?.attitude.quaternion, self.mode == .inside, !self.flying else { return }
                // the motion frame is Z-up; SceneKit is Y-up. The phone's back camera looks down its own -z, like SceneKit's camera.
                let dev = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0)) * simd_quatf(ix: Float(q.x), iy: Float(q.y), iz: Float(q.z), r: Float(q.w))
                let f0 = dev.act(SIMD3(0, 0, -1))
                if off == nil { off = simd_quatf(angle: self.yaw - atan2(-f0.x, -f0.z), axis: SIMD3(0, 1, 0)) }
                let o = off! * dev
                self.cam.simdOrientation = o
                let f = o.act(SIMD3(0, 0, -1))
                self.yaw = atan2(-f.x, -f.z); self.pitch = asin(max(-1, min(1, f.y)))      // dragging carries on from here when gyro is off
            }
        } else { motion.stopDeviceMotionUpdates() }
    }
    deinit { motion.stopDeviceMotionUpdates() }
}

struct WalkViewer: View {
    let item: ScanItem
    @Environment(\.dismiss) private var dismiss
    @StateObject private var world = WalkWorld()
    var body: some View {
        ZStack {
            WalkSceneView(world: world, dir: Library.walkURL(item.id)).ignoresSafeArea()
            VStack(spacing: 8) {
                HStack {
                    Button { dismiss() } label: { Image(systemName: "xmark").font(.title3.bold()).frame(width: 44, height: 44).background(.ultraThinMaterial, in: Circle()) }
                    Spacer()
                    if world.mode == .inside {
                        Button { world.gyro.toggle() } label: { Image(systemName: world.gyro ? "gyroscope" : "hand.draw").font(.title3).frame(width: 44, height: 44).background(.ultraThinMaterial, in: Circle()) }
                        Button { world.measuring.toggle() } label: { Image(systemName: "ruler").font(.title3).frame(width: 44, height: 44).background(world.measuring ? AnyShapeStyle(Color.yellow.opacity(0.8)) : AnyShapeStyle(.ultraThinMaterial), in: Circle()) }
                            .disabled(!world.hasHouse)
                    }
                }.padding(.horizontal).padding(.top, 6)
                Spacer()
                if !world.readout.isEmpty { Text(world.readout).font(.subheadline.monospacedDigit().bold()).padding(.horizontal, 12).padding(.vertical, 8).background(.ultraThinMaterial, in: Capsule()) }
                Picker("", selection: $world.mode) { ForEach(WalkWorld.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) } }
                    .pickerStyle(.segmented).frame(maxWidth: 320).padding(6).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                    .disabled(!world.hasHouse)
                Text(world.mode == .inside ? (world.gyro ? "Move the phone to look round. Tap an arrow to walk." : "Drag to look round, pinch to zoom, tap an arrow to walk.") : (world.mode == .plan ? "Drag to move, pinch to zoom, tap a blue spot to go in." : "Drag to turn, pinch to zoom, tap a blue spot to go in."))
                    .font(.caption2).foregroundStyle(.secondary).padding(.bottom, 14)
            }
        }
        .background(Color.black)
    }
}

struct WalkSceneView: UIViewRepresentable {
    let world: WalkWorld
    let dir: URL
    final class Coord: NSObject {
        let world: WalkWorld
        init(_ w: WalkWorld) { world = w }
        @objc func pan(_ g: UIPanGestureRecognizer) { let t = g.translation(in: g.view); g.setTranslation(.zero, in: g.view); world.pan(t) }
        @objc func pinch(_ g: UIPinchGestureRecognizer) { world.pinch(g.scale); g.scale = 1 }
        @objc func tap(_ g: UITapGestureRecognizer) { world.tap(g.location(in: g.view)) }
    }
    func makeCoordinator() -> Coord { Coord(world) }
    func makeUIView(context: Context) -> SCNView {
        let v = SCNView(frame: .zero); v.backgroundColor = UIColor(white: 0.04, alpha: 1); v.antialiasingMode = .multisampling2X
        v.scene = world.scene; v.autoenablesDefaultLighting = false
        world.view = v
        world.load(dir)
        let pan = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coord.pan(_:)))
        let pinch = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coord.pinch(_:)))
        // inside, these drive the look; in the dollhouse the view's own camera control takes the gestures
        pan.cancelsTouchesInView = false; pinch.cancelsTouchesInView = false
        v.addGestureRecognizer(pan); v.addGestureRecognizer(pinch)
        v.addGestureRecognizer(UITapGestureRecognizer(target: context.coordinator, action: #selector(Coord.tap(_:))))
        return v
    }
    func updateUIView(_ v: SCNView, context: Context) {}

    static func arrow() -> SCNGeometry {
        let p = UIBezierPath(); p.move(to: CGPoint(x: 0, y: 0.32)); p.addLine(to: CGPoint(x: 0.22, y: 0.02)); p.addLine(to: CGPoint(x: 0.08, y: 0.02))
        p.addLine(to: CGPoint(x: 0.08, y: -0.22)); p.addLine(to: CGPoint(x: -0.08, y: -0.22)); p.addLine(to: CGPoint(x: -0.08, y: 0.02)); p.addLine(to: CGPoint(x: -0.22, y: 0.02)); p.close()
        let s = SCNShape(path: p, extrusionDepth: 0.02)
        let m = SCNMaterial(); m.diffuse.contents = UIColor.white.withAlphaComponent(0.92); m.lightingModel = .constant; m.isDoubleSided = true
        m.readsFromDepthBuffer = false
        s.materials = [m]
        return s
    }

    /// a sphere seen from inside whose texture coordinates are exactly the stitcher's (longitude, latitude) mapping
    static func insideSphere(radius r: Float = 10) -> SCNGeometry {
        let nLon = 96, nLat = 48
        var v: [SCNVector3] = [], uv: [CGPoint] = [], idx: [UInt32] = []
        for j in 0...nLat {
            let lat = Float.pi / 2 - Float(j) / Float(nLat) * .pi
            for i in 0...nLon {
                let lon = Float(i) / Float(nLon) * 2 * .pi - .pi
                v.append(SCNVector3(r * cos(lat) * sin(lon), r * sin(lat), -r * cos(lat) * cos(lon)))
                uv.append(CGPoint(x: CGFloat(i) / CGFloat(nLon), y: CGFloat(j) / CGFloat(nLat)))
            }
        }
        for j in 0..<nLat { for i in 0..<nLon {
            let a = UInt32(j * (nLon + 1) + i), b = a + 1, c = a + UInt32(nLon + 1), d = c + 1
            idx += [a, c, b, b, c, d]
        } }
        let g = SCNGeometry(sources: [SCNGeometrySource(vertices: v), SCNGeometrySource(textureCoordinates: uv)],
                            elements: [SCNGeometryElement(indices: idx, primitiveType: .triangles)])
        let m = SCNMaterial(); m.lightingModel = .constant; m.isDoubleSided = true; m.diffuse.contents = UIColor.darkGray
        m.diffuse.mipFilter = .linear
        g.materials = [m]
        return g
    }
}
