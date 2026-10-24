import SwiftUI
import ARKit
import SceneKit
import CoreMotion
import AVFoundation
import simd

// ---------- 3D photo: one photo + its LiDAR depth -> a textured 3D scene with real parallax ----------
final class Photo3DCapture: NSObject, ObservableObject, ARSessionDelegate {
    @Published var ready = false
    weak var session: ARSession?
    var last: ARFrame?
    func start(_ s: ARSession) {
        session = s; s.delegate = self
        let c = ARWorldTrackingConfiguration()
        var sem: ARConfiguration.FrameSemantics = []
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) { sem.insert(.sceneDepth) }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) { sem.insert(.smoothedSceneDepth) }
        c.frameSemantics = sem
        // 5.0: 4K video frames (no shutter sound, no pause)
        if #available(iOS 16.0, *), let f = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution { c.videoFormat = f }
        s.run(c)
    }
    // 5.0: after the shot, a short burst while the phone sways: these frames saw what is behind the edges
    @Published var collecting: Double? = nil                 // 0...1 while gathering side frames
    private var mainT = matrix_identity_float4x4
    private var sides: [Photo3DBuilder.Side] = []
    private var collectStart: TimeInterval = 0, lastSide: TimeInterval = 0
    private var onBurst: (([Photo3DBuilder.Side]) -> Void)?
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        last = frame
        let ok = (frame.smoothedSceneDepth ?? frame.sceneDepth) != nil
        if ok != ready { DispatchQueue.main.async { self.ready = ok } }
        if onScene != nil { sceneFrame(frame); return }
        guard onBurst != nil else { return }
        let t = frame.timestamp
        if collectStart == 0 { collectStart = t }
        let moved = simd_length(SIMD3<Float>(frame.camera.transform.columns.3.x - mainT.columns.3.x, frame.camera.transform.columns.3.y - mainT.columns.3.y, frame.camera.transform.columns.3.z - mainT.columns.3.z))
        if t - lastSide > 0.15, moved > 0.01, sides.count < 12, case .normal = frame.camera.trackingState, let side = Photo3DBuilder.Side(frame) { sides.append(side); lastSide = t }
        let f = min(1, (t - collectStart) / 2.2)
        DispatchQueue.main.async { self.collecting = f }
        if f >= 1 || sides.count >= 12 { let cb = onBurst, got = sides; onBurst = nil; DispatchQueue.main.async { self.collecting = nil; cb?(got) } }
    }
    // 5.0: the 3D photo is a short scan - ~3 s of moving the phone in a small arc, every frame with its photo, LiDAR
    // depth and pose, written as a capture so the lifelike model engine builds the real scene
    private var sceneDir: URL?
    private var sceneFrames = 0
    private var sceneLast: simd_float4x4?
    private var sceneMain: (CVPixelBuffer, CVPixelBuffer, CVPixelBuffer?, simd_float3x3, simd_float4x4, CGSize)?
    private var onScene: ((URL?, (CVPixelBuffer, CVPixelBuffer, CVPixelBuffer?, simd_float3x3, simd_float4x4, CGSize)?) -> Void)?
    private var sceneStart: TimeInterval = 0
    private var sceneTaken: [SIMD3<Float>] = []
    /// 5.5 the guided cross: move LEFT, RIGHT, UP, DOWN from where the shutter was tapped (in the camera's own axes)
    struct Guide: Equatable { var step = 0; var pos = SIMD2<Float>.zero; var done = [false, false, false, false]; var frames = 0 }
    @Published var guide: Guide?
    static let reach: [Float] = [0.10, 0.10, 0.07, 0.07]           // metres: left, right, up, down
    private var g = Guide(), lastGuidePush: TimeInterval = 0
    private var screenView = matrix_identity_float4x4         // the shutter moment's view, in the SCREEN's axes (left/right/up/down as held)
    private let io = DispatchQueue(label: "photo3d.scene", qos: .userInitiated)
    func shootScene(_ done: @escaping (URL?, (CVPixelBuffer, CVPixelBuffer, CVPixelBuffer?, simd_float3x3, simd_float4x4, CGSize)?) -> Void) {
        // 5.7 (48): no depth on this frame - still answer, so the shutter is not left busy for good
        guard let f = last, let dep = f.smoothedSceneDepth ?? f.sceneDepth else { DispatchQueue.main.async { done(nil, nil) }; return }
        closed = false
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("captures").appendingPathComponent("p3d-" + UUID().uuidString)
        try? FileManager.default.createDirectory(at: d.appendingPathComponent("frames"), withIntermediateDirectories: true)
        sceneDir = d; sceneFrames = 0; sceneLast = nil; sceneStart = 0; sceneTaken = []; g = Guide(); lastGuidePush = 0
        // ARKit's camera axes are the sensor's (landscape): held upright, its x runs DOWN the screen. The guide works in
        // the screen's own axes, so "left" is left however the phone is held.
        let o = (UIApplication.shared.connectedScenes.first as? UIWindowScene)?.interfaceOrientation ?? .portrait
        screenView = f.camera.viewMatrix(for: o)
        DispatchQueue.main.async { self.guide = Guide() }
        sceneMain = (f.capturedImage, dep.depthMap, dep.confidenceMap, f.camera.intrinsics, f.camera.transform,
                     CGSize(width: CVPixelBufferGetWidth(f.capturedImage), height: CVPixelBufferGetHeight(f.capturedImage)))
        onScene = done
    }
    private func sceneFrame(_ frame: ARFrame) {
        guard let dir = sceneDir, let dep = frame.smoothedSceneDepth ?? frame.sceneDepth, let m = sceneMain else { return }
        let t = frame.timestamp
        if sceneStart == 0 { sceneStart = t }
        let T = frame.camera.transform
        let pos = SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z)
        // where the phone is now, in the shutter moment's screen axes (x right, y up)
        let local = screenView * SIMD4(pos, 1)
        g.pos = SIMD2(local.x, local.y)
        let want: [Bool] = [local.x <= -Self.reach[0], local.x >= Self.reach[1], local.y >= Self.reach[2], local.y <= -Self.reach[3]]
        for k in 0..<4 where want[k] && !g.done[k] { g.done[k] = true; DispatchQueue.main.async { UIImpactFeedbackGenerator(style: .light).impactOccurred() } }
        while g.step < 4 && g.done[g.step] { g.step += 1 }
        // a new frame wherever the phone is somewhere new (1.5 cm from every frame so far) or turned 2 degrees
        var take = sceneLast == nil
        if !take, let L = sceneLast {
            let f1 = -SIMD3(T.columns.2.x, T.columns.2.y, T.columns.2.z), f0 = -SIMD3(L.columns.2.x, L.columns.2.y, L.columns.2.z)
            take = !sceneTaken.contains { simd_length($0 - pos) < 0.015 } || acos(max(-1, min(1, simd_dot(f1, f0)))) > 0.035
        }
        if take, sceneFrames < 64, case .normal = frame.camera.trackingState,
           let dd = copyPlane(dep.depthMap, bytesPerPixel: 4), let cm = dep.confidenceMap, let cd = copyPlane(cm, bytesPerPixel: 1) {
            sceneLast = T; sceneTaken.append(pos)
            let idx = sceneFrames; sceneFrames += 1; g.frames = sceneFrames
            let img = frame.capturedImage, K = frame.camera.intrinsics, dw = CVPixelBufferGetWidth(dep.depthMap), dh = CVPixelBufferGetHeight(dep.depthMap)
            let sharp = frameSharpness(img)
            io.async { Photo3DCapture.write(img, dd, cd, dw, dh, T, K, sharp, t, idx, dir) }
        }
        if t - lastGuidePush > 0.066 { lastGuidePush = t; let gg = g; DispatchQueue.main.async { self.guide = gg } }
        // done: all four directions reached (or 25 s, with whatever was gathered)
        if g.step >= 4 || t - sceneStart > 25 { finishScene() }
    }
    /// ends the guided capture now (also the Done button) and builds from what was gathered
    func finishScene() {
        guard let dir = sceneDir, onScene != nil else { return }
        let cb = onScene, m = sceneMain, n = sceneFrames
        onScene = nil; sceneDir = nil
        io.async {                                                                                                          // after the last write
            if n < 6 { try? FileManager.default.removeItem(at: dir) }          // 5.7 (48): too few frames to use: not left behind
            DispatchQueue.main.async { self.guide = nil; self.collecting = nil; cb?(n >= 6 ? dir : nil, m) }
        }
    }
    /// 5.7 (48): the screen closed - a capture in progress is dropped with its frames, and a build still running throws its result away
    private(set) var closed = false
    func cancelScene() {
        closed = true
        guard let dir = sceneDir else { return }
        onScene = nil; sceneDir = nil
        io.async { try? FileManager.default.removeItem(at: dir) }           // after the last write
        DispatchQueue.main.async { self.guide = nil; self.collecting = nil }
    }
    static func write(_ img: CVPixelBuffer, _ depth: Data, _ conf: Data, _ dw: Int, _ dh: Int, _ T: simd_float4x4, _ K: simd_float3x3, _ sharp: Float, _ ts: TimeInterval, _ idx: Int, _ dir: URL) {
        let name = String(format: "%06d", idx), frames = dir.appendingPathComponent("frames")
        var ci = CIImage(cvPixelBuffer: img)
        let iw = ci.extent.width, ih = ci.extent.height, s = min(1, 2560 / max(iw, ih))
        if s < 1 { ci = ci.transformed(by: CGAffineTransform(scaleX: s, y: s)) }
        let q = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let jpg = sharedCI.jpegRepresentation(of: ci, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, options: [q: 0.9]) else { return }
        try? jpg.write(to: frames.appendingPathComponent(name + ".jpg"))
        try? depth.write(to: frames.appendingPathComponent(name + ".depth"))
        try? conf.write(to: frames.appendingPathComponent(name + ".conf"))
        var tm: [Double] = []; for c in 0..<4 { for r in 0..<4 { tm.append(Double(T[c][r])) } }
        var km: [Double] = []; for c in 0..<3 { for r in 0..<3 { km.append(Double(K[c][r])) } }
        let fs = Double(s); km[0] *= fs; km[4] *= fs; km[6] *= fs; km[7] *= fs
        let meta: [String: Any] = ["transform": tm, "intrinsics": km, "imageWidth": Int((iw * s).rounded()), "imageHeight": Int((ih * s).rounded()),
                                   "depthWidth": dw, "depthHeight": dh, "timestamp": ts, "sharpness": Double(sharp)]
        if let j = try? JSONSerialization.data(withJSONObject: meta) { try? j.write(to: frames.appendingPathComponent(name + ".json")) }
    }
    /// the depth of this moment, then the side frames from the sway
    func shoot(_ done: @escaping (CVPixelBuffer, CVPixelBuffer, CVPixelBuffer?, simd_float3x3, simd_float4x4, CGSize, [Photo3DBuilder.Side]) -> Void) {
        guard let f = last, let dep = f.smoothedSceneDepth ?? f.sceneDepth else { return }
        let dmap = dep.depthMap, conf = dep.confidenceMap, img = f.capturedImage, K = f.camera.intrinsics, T = f.camera.transform
        let size = CGSize(width: CVPixelBufferGetWidth(img), height: CVPixelBufferGetHeight(img))
        mainT = T; sides = []; collectStart = 0; lastSide = 0
        onBurst = { got in done(img, dmap, conf, K, T, size, got) }
    }
}

struct Photo3DARView: UIViewRepresentable {
    let cap: Photo3DCapture
    func makeUIView(context: Context) -> ARSCNView { let v = ARSCNView(frame: .zero); cap.start(v.session); return v }
    func updateUIView(_ v: ARSCNView, context: Context) {}
    static func dismantleUIView(_ v: ARSCNView, coordinator: ()) { v.session.pause() }
}

enum Photo3DBuilder {
    /// 5.0: the real scene from a short capture - shape from every frame's LiDAR depth, painted from the photos - placed
    /// in the camera space of the moment the shutter was tapped (so the viewer, video and export work as before)
    static func scene(_ dir: URL, mainT: simd_float4x4, progress: @escaping (Float, String) -> Void) -> (SCNGeometry, Float, [RealPhoto])? {
        let all = Keyframe.all(in: dir)
        guard all.count >= 6 else { return nil }
        // how deep the scene is: the main frame's depths
        var ds: [Float] = []
        if let k = all.first, let d = try? Data(contentsOf: k.jpg.deletingPathExtension().appendingPathExtension("depth")) {
            d.withUnsafeBytes { b in for i in stride(from: 0, to: d.count / 4, by: 7) { let z = b.loadUnaligned(fromByteOffset: i * 4, as: Float.self); if z > 0.1 && z.isFinite { ds.append(z) } } }
        }
        ds.sort()
        let mid = ds.isEmpty ? 1.5 : ds[ds.count / 2], far = ds.isEmpty ? 3 : ds[ds.count * 95 / 100]
        let item = ScanItem(name: "3D photo", capture: dir.lastPathComponent)
        // 5.5: only what the photos saw in colour (no grey filler, nothing weakly seen)
        guard var t = PhotoModel.build(item, extent: max(0.8, far * 1.6), detail: 2, minWeight: 0.8, seenOnly: true, progress: progress), t.f.count >= 300 else { return nil }
        let Ti = mainT.inverse
        let R = simd_float3x3(SIMD3(Ti.columns.0.x, Ti.columns.0.y, Ti.columns.0.z), SIMD3(Ti.columns.1.x, Ti.columns.1.y, Ti.columns.1.z), SIMD3(Ti.columns.2.x, Ti.columns.2.y, Ti.columns.2.z))
        t.v = t.v.map { CloudOps.apply(Ti, $0) }
        t.n = t.n.map { R * $0 }
        return (t.geometry(centre: .zero), mid * 2, RealPhoto.load(item))
    }

    /// a frame from the sway after the shot: a small copy of the picture, its depth, where it was taken
    struct Side {
        let rgba: [UInt8]; let w: Int; let h: Int
        let depth: [Float]; let dw: Int; let dh: Int
        let Ti: simd_float4x4; let fx: Float; let fy: Float; let cx: Float; let cy: Float
        init?(_ f: ARFrame) {
            guard let dep = f.smoothedSceneDepth ?? f.sceneDepth else { return nil }
            let ci = CIImage(cvPixelBuffer: f.capturedImage)
            let W = 480, H = Int(Float(W) * Float(ci.extent.height) / Float(ci.extent.width))
            let sc = CGFloat(W) / ci.extent.width
            var px = [UInt8](repeating: 0, count: W * H * 4)
            sharedCI.render(ci.transformed(by: CGAffineTransform(scaleX: sc, y: sc)), toBitmap: &px, rowBytes: W * 4, bounds: CGRect(x: 0, y: 0, width: W, height: H), format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
            let dm = dep.depthMap, dw = CVPixelBufferGetWidth(dm), dh = CVPixelBufferGetHeight(dm)
            CVPixelBufferLockBaseAddress(dm, .readOnly); defer { CVPixelBufferUnlockBaseAddress(dm, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(dm) else { return nil }
            let row = CVPixelBufferGetBytesPerRow(dm)
            var d = [Float](repeating: 0, count: dw * dh)
            for y in 0..<dh { let r = base.advanced(by: y * row).assumingMemoryBound(to: Float32.self); for x in 0..<dw { d[y * dw + x] = r[x] } }
            let K = f.camera.intrinsics, s = Float(W) / Float(ci.extent.width)
            rgba = px; w = W; h = H; depth = d; self.dw = dw; self.dh = dh; Ti = f.camera.transform.inverse
            fx = K[0][0] * s; fy = K[1][1] * s; cx = K[2][0] * s; cy = K[2][1] * s
        }
        /// 5.7 (38) a saved frame of the guided capture (photo, LiDAR depth, pose), for the background layer
        init?(_ k: Keyframe) {
            guard let ci = CIImage(contentsOf: k.jpg), k.dw > 0,
                  let dd = try? Data(contentsOf: k.jpg.deletingPathExtension().appendingPathExtension("depth")), dd.count >= k.dw * k.dh * 4 else { return nil }
            let W = 480, H = Int(Float(W) * Float(ci.extent.height) / Float(ci.extent.width))
            var px = [UInt8](repeating: 0, count: W * H * 4)
            sharedCI.render(ci.transformed(by: CGAffineTransform(scaleX: CGFloat(W) / ci.extent.width, y: CGFloat(H) / ci.extent.height)), toBitmap: &px, rowBytes: W * 4, bounds: CGRect(x: 0, y: 0, width: W, height: H), format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
            var d = [Float](repeating: 0, count: k.dw * k.dh)
            dd.withUnsafeBytes { b in for i in 0..<d.count { d[i] = b.loadUnaligned(fromByteOffset: i * 4, as: Float.self) } }
            let s = Float(W) / k.w
            rgba = px; w = W; h = H; depth = d; dw = k.dw; dh = k.dh; Ti = k.T.inverse
            fx = k.K[0][0] * s; fy = k.K[1][1] * s; cx = k.K[2][0] * s; cy = k.K[2][1] * s
        }
        /// the colour this frame saw at a world point, if it really saw that point (its depth agrees)
        func colour(_ p: SIMD3<Float>) -> SIMD3<Float>? {
            let c = Ti * SIMD4(p, 1); let z = -c.z
            guard z > 0.05 else { return nil }
            let u = cx + fx * c.x / z, v = cy - fy * c.y / z
            guard u >= 0, v >= 0, u < Float(w - 1), v < Float(h - 1) else { return nil }
            let dx = min(dw - 1, Int(u / Float(w) * Float(dw))), dy = min(dh - 1, Int(v / Float(h) * Float(dh)))
            let md = depth[dy * dw + dx]
            guard md > 0.05, abs(md - z) < max(0.03, 0.04 * z) else { return nil }
            let o = (Int(v) * w + Int(u)) * 4                            // 5.7 (47): CoreImage bitmaps run top-down
            return SIMD3(Float(rgba[o]), Float(rgba[o + 1]), Float(rgba[o + 2]))
        }
    }

    /// the textured depth mesh, in camera space (x right, y up, looking down -z), with jumps in depth cut open
    static func build(image: CVPixelBuffer, depth: CVPixelBuffer, conf: CVPixelBuffer?, K: simd_float3x3, T: simd_float4x4 = matrix_identity_float4x4, size: CGSize, sides: [Side] = []) -> (SCNGeometry, UIImage, Float)? {
        let ci = CIImage(cvPixelBuffer: image)
        guard let cg = sharedCI.createCGImage(ci, from: ci.extent) else { return nil }
        let photo = UIImage(cgImage: cg)
        let dw = CVPixelBufferGetWidth(depth), dh = CVPixelBufferGetHeight(depth)
        CVPixelBufferLockBaseAddress(depth, .readOnly); defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard let db = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let dRow = CVPixelBufferGetBytesPerRow(depth)
        var cf: [UInt8]? = nil
        if let c = conf {
            CVPixelBufferLockBaseAddress(c, .readOnly)
            if let cb = CVPixelBufferGetBaseAddress(c) { let r = CVPixelBufferGetBytesPerRow(c); cf = (0..<(dw * dh)).map { cb.advanced(by: ($0 / dw) * r).assumingMemoryBound(to: UInt8.self)[$0 % dw] } }
            CVPixelBufferUnlockBaseAddress(c, .readOnly)
        }
        func dAt(_ x: Int, _ y: Int) -> Float { db.advanced(by: y * dRow).assumingMemoryBound(to: Float32.self)[x] }
        // 5.0: the raw depth, with low-confidence and missing readings filled in from their neighbours (no holes)
        var R = [Float](repeating: 0, count: dw * dh)
        for y in 0..<dh { for x in 0..<dw {
            let z = dAt(x, y)
            if z > 0.05, z.isFinite, cf.map({ $0[y * dw + x] > 0 }) ?? true { R[y * dw + x] = z }
        } }
        for _ in 0..<40 {
            var any = false, next = R
            for y in 0..<dh { for x in 0..<dw where R[y * dw + x] == 0 {
                var s: Float = 0, n: Float = 0
                for (ox, oy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] { let qx = x + ox, qy = y + oy; if qx >= 0, qy >= 0, qx < dw, qy < dh, R[qy * dw + qx] > 0 { s += R[qy * dw + qx]; n += 1 } }
                if n > 0 { next[y * dw + x] = s / n } else { any = true }
            } }
            R = next; if !any { break }
        }
        // a small copy of the photo to guide the upsampling (depth edges follow colour edges)
        let GW = 640, GH = Int(Float(GW) * Float(size.height) / Float(size.width))
        var rgba = [UInt8](repeating: 0, count: GW * GH * 4)
        let small = ci.transformed(by: CGAffineTransform(scaleX: CGFloat(GW) / ci.extent.width, y: CGFloat(GH) / ci.extent.height))
        sharedCI.render(small, toBitmap: &rgba, rowBytes: GW * 4, bounds: CGRect(x: 0, y: 0, width: GW, height: GH), format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        func col(_ x: Int, _ y: Int) -> SIMD3<Float> { let o = (y * GW + x) * 4;      // 5.7 (47): top-down (depth edges followed the mirrored photo)
        return SIMD3(Float(rgba[o]), Float(rgba[o + 1]), Float(rgba[o + 2])) }
        // joint bilateral upsampling: each grid point blends the 4x4 nearest depth readings, by distance and by how
        // alike their colours are (smooth surfaces, sharp edges - no staircase pixels)
        var D = [Float](repeating: 0, count: GW * GH)
        D.withUnsafeMutableBufferPointer { db2 in
            let dp = db2.baseAddress!
            DispatchQueue.concurrentPerform(iterations: GH) { y in
                for x in 0..<GW {
                    let fx = (Float(x) + 0.5) / Float(GW) * Float(dw) - 0.5, fy = (Float(y) + 0.5) / Float(GH) * Float(dh) - 0.5
                    let me = col(x, y)
                    var s: Float = 0, w: Float = 0
                    for oy in -1...2 { for ox in -1...2 {
                        let qx = min(dw - 1, max(0, Int(fx.rounded(.down)) + ox)), qy = min(dh - 1, max(0, Int(fy.rounded(.down)) + oy))
                        let z = R[qy * dw + qx]; guard z > 0 else { continue }
                        let ddx = fx - Float(qx), ddy = fy - Float(qy)
                        let px = min(GW - 1, Int((Float(qx) + 0.5) / Float(dw) * Float(GW))), py = min(GH - 1, Int((Float(qy) + 0.5) / Float(dh) * Float(GH)))
                        let dc = simd_length(col(px, py) - me) / 255
                        let wt = exp(-(ddx * ddx + ddy * ddy) / 1.2) * exp(-dc * dc / 0.02)
                        s += z * wt; w += wt
                    } }
                    dp[y * GW + x] = w > 1e-6 ? s / w : 0
                }
            }
        }
        let sx = Float(size.width) / Float(GW)
        let fx = K[0][0] / sx, fy = K[1][1] / sx, cx = K[2][0] / sx, cy = K[2][1] / sx
        // 5.0: more depth - distances are stretched around the middle depth of the picture (35% more relief)
        let valid = D.filter { $0 > 0.05 }.sorted()
        let zMid = valid.isEmpty ? 1 : valid[valid.count / 2]
        // 5.7 (38): true depth (it was stretched 35%, which threw the projected real photos out of place)
        @inline(__always) func ex(_ z: Float) -> Float { max(0.05, z) }
        var verts: [SCNVector3] = [], uvs: [CGPoint] = [], idx: [UInt32] = []
        verts.reserveCapacity(GW * GH); uvs.reserveCapacity(GW * GH)
        var maxD: Float = 0
        for y in 0..<GH { for x in 0..<GW {
            var z = D[y * GW + x]; if z <= 0 { z = 0.01 }
            maxD = max(maxD, z)
            let ze = ex(z)
            verts.append(SCNVector3((Float(x) - cx) * ze / fx, -(Float(y) - cy) * ze / fy, -ze))
            uvs.append(CGPoint(x: (CGFloat(x) + 0.5) / CGFloat(GW), y: (CGFloat(y) + 0.5) / CGFloat(GH)))
        } }
        // cut only where the surface really breaks: a jump that does not continue the slope around it (a floor or wall
        // seen at an angle changes depth smoothly and stays joined; the edge of a chair in front of a wall is cut)
        func bend(_ i: Int, _ step: Int, _ limit: Int, _ pos: Int) -> Float {
            guard pos > 0, pos < limit - 1 else { return 0 }
            let a = D[i - step], b = D[i], c = D[i + step]
            return abs(a - 2 * b + c) / max(0.05, b)
        }
        for y in 0..<(GH - 1) { for x in 0..<(GW - 1) {
            let a = y * GW + x, b = a + 1, c = a + GW, d = c + 1
            let zs = [D[a], D[b], D[c], D[d]]
            guard let lo = zs.min(), let hi = zs.max(), lo > 0.05 else { continue }
            if hi / lo > 1.03 {
                let k = max(bend(a, 1, GW, x), bend(b, 1, GW, x + 1), bend(a, GW, GH, y), bend(c, GW, GH, y + 1))
                if k > 0.035 { continue }
            }
            idx += [UInt32(a), UInt32(c), UInt32(b), UInt32(b), UInt32(c), UInt32(d)]
        } }
        guard !idx.isEmpty else { return nil }
        // 5.0: the background layer (what spatial photos show when you look round an edge): near things pushed back to
        // the depth just around them. Its colour where the photo only shows the near thing comes first from the frames
        // taken while the phone swayed (they really saw behind the edge), then from real texture copied patch by patch
        // from the background around it - never a blur.
        let Rr = 6
        var H1 = [Float](repeating: 0, count: GW * GH), Bd = [Float](repeating: 0, count: GW * GH)
        for y in 0..<GH { for x in 0..<GW { var m: Float = 0; for k in max(0, x - Rr)...min(GW - 1, x + Rr) { m = max(m, D[y * GW + k]) }; H1[y * GW + x] = m } }
        for y in 0..<GH { for x in 0..<GW { var m: Float = 0; for k in max(0, y - Rr)...min(GH - 1, y + Rr) { m = max(m, H1[k * GW + x]) }; Bd[y * GW + x] = m } }
        var target = [SIMD3<Float>](repeating: .zero, count: GW * GH), known = [Bool](repeating: false, count: GW * GH), hole = known
        var fromSides = 0
        for y in 0..<GH { for x in 0..<GW {
            let i = y * GW + x
            target[i] = col(x, y)
            if D[i] > 0.05 && D[i] >= Bd[i] * 0.95 { known[i] = true; continue }
            // what is behind: the background point, in the world, looked up in the side frames
            let z = Bd[i]
            if z > 0.05 && !sides.isEmpty {
                let pc = SIMD4<Float>((Float(x) - cx) * z / fx, -(Float(y) - cy) * z / fy, -z, 1), pw = T * pc
                var sum = SIMD3<Float>.zero, n: Float = 0
                for sd in sides { if let c = sd.colour(SIMD3(pw.x, pw.y, pw.z)) { sum += c; n += 1 } }
                if n > 0 { target[i] = sum / n; known[i] = true; fromSides += 1; continue }
            }
            hole[i] = true
        } }
        let filled = GenFill.synthesize(target: target, known: known, hole: hole, W: GW, H: GH, src: target, srcKnown: known, SW: GW, SH: GH)
        Log.add("3d photo background: \(fromSides) pixels seen by the sway frames, \(hole.filter { $0 }.count) synthesised")
        let fill = (filled.map { $0.x }, filled.map { $0.y }, filled.map { $0.z })
        var bgPix = [UInt8](repeating: 255, count: GW * GH * 4)
        for i in 0..<(GW * GH) { bgPix[i * 4] = u8(fill.0[i]); bgPix[i * 4 + 1] = u8(fill.1[i]); bgPix[i * 4 + 2] = u8(fill.2[i]) }
        var bgImage: UIImage? = nil
        if let ctx = CGContext(data: &bgPix, width: GW, height: GH, bitsPerComponent: 8, bytesPerRow: GW * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue), let cgb = ctx.makeImage() { bgImage = UIImage(cgImage: cgb) }
        let base = UInt32(verts.count)
        for y in 0..<GH { for x in 0..<GW {
            let z = ex(max(0.06, Bd[y * GW + x]) * 1.015)
            verts.append(SCNVector3((Float(x) - cx) * z / fx, -(Float(y) - cy) * z / fy, -z))
            uvs.append(CGPoint(x: (CGFloat(x) + 0.5) / CGFloat(GW), y: (CGFloat(y) + 0.5) / CGFloat(GH)))
        } }
        var bidx: [UInt32] = []
        for y in 0..<(GH - 1) { for x in 0..<(GW - 1) {
            let a = y * GW + x, b = a + 1, c = a + GW, d = c + 1
            let zs = [Bd[a], Bd[b], Bd[c], Bd[d]]
            guard let lo = zs.min(), let hi = zs.max(), lo > 0.05, hi / lo < 1.25 else { continue }
            bidx += [base + UInt32(a), base + UInt32(c), base + UInt32(b), base + UInt32(b), base + UInt32(c), base + UInt32(d)]
        } }
        let g = SCNGeometry(sources: [SCNGeometrySource(vertices: verts), SCNGeometrySource(textureCoordinates: uvs)],
                            elements: [SCNGeometryElement(indices: idx, primitiveType: .triangles), SCNGeometryElement(indices: bidx, primitiveType: .triangles)])
        let m = SCNMaterial(); m.diffuse.contents = photo; m.lightingModel = .constant; m.isDoubleSided = true
        let mb = SCNMaterial(); mb.diffuse.contents = bgImage ?? photo; mb.lightingModel = .constant; mb.isDoubleSided = true
        g.materials = [m, mb]
        _ = maxD
        return (g, photo, zMid * 2)
    }
}

/// shows the 3D photo; tilting the phone moves the camera a few centimetres (real parallax), dragging looks around
struct Photo3DView: UIViewRepresentable {
    let geometry: SCNGeometry
    let depth: Float
    var photos: [RealPhoto] = []                 // 5.0 photoreal scene
    var mainT = matrix_identity_float4x4         // the scene is in this camera's space
    final class Coord: NSObject {
        let motion = CMMotionManager(); var cam = SCNNode(); var base: CMAttitude?; var drag = CGPoint.zero; var depth: Float = 2
        var projector: PhotoProjector?
        @objc func pan(_ g: UIPanGestureRecognizer) { drag = g.translation(in: g.view); if g.state == .ended || g.state == .cancelled { UIView.animate(withDuration: 0.3) { self.drag = .zero } } }
    }
    func makeCoordinator() -> Coord { Coord() }
    func makeUIView(context: Context) -> SCNView {
        let v = SCNView(frame: .zero); v.backgroundColor = .black; v.antialiasingMode = .multisampling4X
        let scene = SCNScene()
        let root = SCNNode(geometry: geometry); root.eulerAngles.z = -.pi / 2             // the sensor is landscape; the phone was upright
        scene.rootNode.addChildNode(root)
        let co = context.coordinator; co.depth = depth
        co.cam.camera = SCNCamera(); co.cam.camera?.zNear = 0.01; co.cam.camera?.fieldOfView = 62
        scene.rootNode.addChildNode(co.cam); v.scene = scene; v.pointOfView = co.cam; v.isPlaying = true
        if !photos.isEmpty, let mat = geometry.materials.first {
            // SceneKit world = the turned root of camera-space geometry; photo world = mainT * camera space
            let p = PhotoProjector(photos: photos, material: mat, worldFromScene: mainT * root.simdTransform.inverse)
            co.projector = p; v.delegate = p; v.rendersContinuously = true
        }
        v.addGestureRecognizer(UIPanGestureRecognizer(target: co, action: #selector(Coord.pan(_:))))
        let focus = SCNVector3(0, 0, -max(0.3, depth * 0.5))          // the middle depth stays put, near things move most
        var sm = SIMD2<Float>.zero
        co.motion.deviceMotionUpdateInterval = 1 / 60
        co.motion.startDeviceMotionUpdates(to: .main) { m, _ in
            guard let m else { return }
            if co.base == nil { co.base = m.attitude.copy() as? CMAttitude }
            let a = m.attitude.copy() as! CMAttitude; if let b = co.base { a.multiply(byInverseOf: b) }
            let r = Float(0.15) * max(0.6, min(1.6, co.depth / 2))      // 5.0: up to ~15 cm of parallax, scaled to the scene
            let dx = Float(max(-0.5, min(0.5, a.roll))) * r * 2 + Float(co.drag.x) * 0.0011
            let dy = Float(max(-0.5, min(0.5, a.pitch))) * r * 2 - Float(co.drag.y) * 0.0011
            sm += (SIMD2(dx, dy) - sm) * 0.18                                       // smoothed: glides instead of jittering
            co.cam.simdPosition = SIMD3(sm.x, sm.y, 0.02)
            co.cam.look(at: focus)
        }
        return v
    }
    func updateUIView(_ v: SCNView, context: Context) {}
    static func dismantleUIView(_ v: SCNView, coordinator: Coord) { coordinator.motion.stopDeviceMotionUpdates() }
}

struct Photo3DScreen: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var cap = Photo3DCapture()
    @State private var built: (SCNGeometry, UIImage, Float)?
    @State private var realPhotos: [RealPhoto] = []
    @State private var realT = matrix_identity_float4x4
    @State private var keepDir: URL?
    @State private var busy = false
    @State private var share: URL?
    @State private var note = ""
    var body: some View {
        ZStack {
            if let b = built {
                Photo3DView(geometry: b.0, depth: b.2, photos: realPhotos, mainT: realT).ignoresSafeArea()
                    // the spatial-photo window: soft rounded edges and a gentle dark vignette
                    .mask(RoundedRectangle(cornerRadius: 34).padding(10).blur(radius: 14).ignoresSafeArea())
                    .overlay(RadialGradient(colors: [.clear, .black.opacity(0.35)], center: .center, startRadius: 260, endRadius: 620).allowsHitTesting(false).ignoresSafeArea())
                VStack {
                    HStack {
                        Button { built = nil; note = ""; realPhotos = []; if let d = keepDir { try? FileManager.default.removeItem(at: d) }; keepDir = nil } label: { Label("Retake", systemImage: "camera").padding(10).background(.ultraThinMaterial, in: Capsule()) }
                        Spacer()
                        Menu {
                            Button("Save to library") { save(b) }
                            Button("Export USDZ") { share = usdz(Self.plain(b.0)) }
                            Button("Make a 3D sway video") { video(b) }
                        } label: { Image(systemName: "square.and.arrow.up").font(.title3).frame(width: 44, height: 44).background(.ultraThinMaterial, in: Circle()) }
                    }.padding()
                    Spacer()
                    Text(note.isEmpty ? "Tilt your phone to look around the 3D photo, or drag." : note).font(.footnote).padding(10).background(.ultraThinMaterial, in: Capsule()).padding(.bottom, 30)
                }
            } else {
                Photo3DARView(cap: cap).ignoresSafeArea()
                VStack {
                    HStack {
                        Button { dismiss() } label: { Image(systemName: "xmark").font(.title3.bold()).frame(width: 44, height: 44).background(.ultraThinMaterial, in: Circle()) }
                        Spacer()
                        Text("3D photo").font(.headline).padding(.horizontal, 12).padding(.vertical, 8).background(.ultraThinMaterial, in: Capsule())
                    }.padding()
                    Spacer()
                    if let g = cap.guide {
                        CrossGuide(g: g) { cap.finishScene() }
                    } else if busy && cap.collecting == nil {
                        VStack(spacing: 8) { ProgressView(); Text(note.isEmpty ? "Building the 3D scene..." : note).font(.subheadline) }
                            .padding(14).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                    } else if let f = cap.collecting {
                        VStack(spacing: 8) {
                            Text("Move the phone slowly in a small arc").font(.headline)
                            ProgressView(value: f).tint(.white).frame(width: 220)
                        }.padding(14).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                    } else {
                        Text(note.isEmpty ? "Frame it and tap. Then, keeping it in the middle, move the phone left, right, up and down as shown: every part it sees in colour becomes 3D. 0.5-4 m away." : note).font(.footnote).padding(10).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12)).padding(.horizontal)
                    }
                    Button { shoot() } label: {
                        ZStack { Circle().stroke(.white, lineWidth: 4).frame(width: 78, height: 78); Circle().fill(busy ? Color.gray : Color.white).frame(width: 62, height: 62) }
                    }.disabled(!cap.ready || busy).padding(.bottom, 34)
                }
            }
        }
        .background(Color.black)
        .onAppear { Awake.set("photo3d", true) }
        .onDisappear { Awake.set("photo3d", false); cap.cancelScene(); if let d = keepDir { try? FileManager.default.removeItem(at: d) } }
        .sheet(item: $share) { ShareSheet(items: [$0]) }
    }
    private func shoot() {
        busy = true
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        cap.shootScene { dir, main in
            note = "Building the 3D scene..."
            DispatchQueue.global(qos: .userInitiated).async {
                var b: (SCNGeometry, UIImage, Float)? = nil, rp: [RealPhoto] = []
                // 5.7 (38): the shutter photo's own depth mesh is the 3D photo - whole, sharp, no holes (the fused scene
                // mesh of 5.5 kept only what every frame agreed on and showed the rest as black holes). The guided
                // capture's frames colour what is behind edges, so that is still what the camera really saw.
                if let m = main {
                    let kfs = dir.map { Keyframe.all(in: $0) } ?? []
                    let step = max(1, kfs.count / 24)
                    let sides = stride(from: 0, to: kfs.count, by: step).compactMap { Photo3DBuilder.Side(kfs[$0]) }
                    b = Photo3DBuilder.build(image: m.0, depth: m.1, conf: m.2, K: m.3, T: m.4, size: m.5, sides: sides)
                    if let d = dir { rp = RealPhoto.load(ScanItem(name: "3D photo", capture: d.lastPathComponent)) }
                }
                let keep = rp.isEmpty ? nil : dir
                if keep == nil, let d = dir { try? FileManager.default.removeItem(at: d) }       // the photoreal view still reads the photos
                DispatchQueue.main.async {
                    if cap.closed { if let d = keep { try? FileManager.default.removeItem(at: d) }; return }    // 5.7 (48): closed while building
                    busy = false; note = ""; realPhotos = rp; realT = main?.4 ?? matrix_identity_float4x4; keepDir = keep; built = b; if b == nil { note = dir == nil ? "No depth here - try somewhere closer" : "Not enough was seen - move further each way (follow the cross), 0.5-4 m from it" }; Log.add("3d photo \(b != nil) scene=\(dir != nil)") }
            }
        }
    }
    /// 5.7 (48): the on-screen geometry's material is rewritten every frame by the photo projector on SceneKit's render
    /// thread; exports and the video draw a copy with fresh plain materials (same photos), never the shared one
    static func plain(_ g: SCNGeometry) -> SCNGeometry {
        let c = g.copy() as! SCNGeometry
        c.materials = g.materials.map { o in
            let m = SCNMaterial(); m.diffuse.contents = o.diffuse.contents; m.lightingModel = o.lightingModel; m.isDoubleSided = o.isDoubleSided; return m
        }
        return c
    }
    private func usdz(_ g: SCNGeometry) -> URL? {
        let scene = SCNScene(); let n = SCNNode(geometry: g); n.eulerAngles.z = -.pi / 2; scene.rootNode.addChildNode(n)
        let u = Export.tmp("3D photo.usdz"); try? FileManager.default.removeItem(at: u)
        return scene.write(to: u, options: nil, delegate: nil, progressHandler: nil) ? u : nil
    }
    private func save(_ b: (SCNGeometry, UIImage, Float)) {
        guard let u = usdz(Self.plain(b.0)), let it = Library.shared.addModel(from: u, name: "3D photo " + Date().formatted(date: .abbreviated, time: .shortened)) else { note = "Could not save"; return }
        let r = UIGraphicsImageRenderer(size: CGSize(width: 240, height: 240))
        let thumb = r.image { _ in
            let up = UIImage(cgImage: b.1.cgImage!, scale: 1, orientation: .right)
            let s = max(240 / up.size.width, 240 / up.size.height)
            up.draw(in: CGRect(x: (240 - up.size.width * s) / 2, y: (240 - up.size.height * s) / 2, width: up.size.width * s, height: up.size.height * s))
        }
        try? thumb.jpegData(compressionQuality: 0.8)?.write(to: Library.thumbURL(it.id))
        Thumbs.forget(it.id)
        note = "Saved to the library"
    }
    /// a short video swaying side to side, so the depth shows anywhere it is shared
    private func video(_ b: (SCNGeometry, UIImage, Float)) {
        note = "Making the video..."
        let g = Self.plain(b.0), depth = b.2                    // 5.7 (48): not the geometry the projector is drawing
        DispatchQueue.global(qos: .userInitiated).async {
            guard let dev = MTLCreateSystemDefaultDevice() else { return }
            let scene = SCNScene(); scene.background.contents = UIColor.black
            let n = SCNNode(geometry: g); n.eulerAngles.z = -.pi / 2; scene.rootNode.addChildNode(n)
            let cam = SCNNode(); cam.camera = SCNCamera(); cam.camera?.zNear = 0.01; cam.camera?.fieldOfView = 62; scene.rootNode.addChildNode(cam)
            let ren = SCNRenderer(device: dev, options: nil); ren.scene = scene; ren.pointOfView = cam
            let size = CGSize(width: 1080, height: 1440)
            let url = Export.tmp("3D photo.mp4"); try? FileManager.default.removeItem(at: url)
            guard let w = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return }
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 1080, AVVideoHeightKey: 1440])
            let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: 1080, kCVPixelBufferHeightKey as String: 1440])
            w.add(input); w.startWriting(); w.startSession(atSourceTime: .zero)
            let focus = SCNVector3(0, 0, -min(1.5, depth * 0.5)), frames = 150
            for f in 0..<frames {
                let a = Float(f) / Float(frames) * 2 * .pi
                cam.simdPosition = SIMD3(sin(a) * 0.07, cos(a * 2) * 0.02, 0.02); cam.look(at: focus)
                let img = ren.snapshot(atTime: Double(f) / 30, with: size, antialiasingMode: .multisampling4X)
                while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.005) }
                guard let pool = ad.pixelBufferPool, let cgi = img.cgImage else { continue }
                var pb: CVPixelBuffer?; CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb); guard let px = pb else { continue }
                CVPixelBufferLockBaseAddress(px, [])
                if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(px), width: 1080, height: 1440, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(px), space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) { ctx.draw(cgi, in: CGRect(origin: .zero, size: size)) }
                CVPixelBufferUnlockBaseAddress(px, [])
                ad.append(px, withPresentationTime: CMTime(value: CMTimeValue(f), timescale: 30))
            }
            input.markAsFinished()
            let sem = DispatchSemaphore(value: 0); w.finishWriting { sem.signal() }; sem.wait()
            DispatchQueue.main.async { note = ""; if w.status == .completed { share = url } else { note = "Could not make the video" } }
        }
    }
}

/// 5.5 the guided cross for 3D photo: where the phone is (the dot), the four targets, the one to go to next
struct CrossGuide: View {
    let g: Photo3DCapture.Guide
    let done: () -> Void
    static let words = ["Move the phone LEFT", "Now move it RIGHT", "Now move it UP", "Now move it DOWN"]
    var body: some View {
        VStack(spacing: 10) {
            Text(g.step < 4 ? CrossGuide.words[g.step] : "Done").font(.title3.bold())
            Text("Slowly, keeping what you photographed in the middle").font(.caption).foregroundStyle(.secondary)
            ZStack {
                let R: CGFloat = 70
                Path { p in p.move(to: CGPoint(x: 0, y: R)); p.addLine(to: CGPoint(x: 2 * R, y: R)); p.move(to: CGPoint(x: R, y: 0)); p.addLine(to: CGPoint(x: R, y: 2 * R)) }
                    .stroke(.white.opacity(0.35), style: StrokeStyle(lineWidth: 3, lineCap: .round)).frame(width: 2 * R, height: 2 * R)
                ForEach(0..<4, id: \.self) { k in
                    let off: [CGSize] = [CGSize(width: -R, height: 0), CGSize(width: R, height: 0), CGSize(width: 0, height: -R), CGSize(width: 0, height: R)]
                    Circle().fill(g.done[k] ? Color.green : (k == g.step ? Color.yellow : Color.white.opacity(0.3))).frame(width: 22, height: 22)
                        .overlay(Image(systemName: g.done[k] ? "checkmark" : ["arrow.left", "arrow.right", "arrow.up", "arrow.down"][k]).font(.caption2.bold()).foregroundStyle(.black))
                        .offset(off[k])
                }
                // the phone: its sideways position scaled so a target sits on its circle
                let px = CGFloat(g.pos.x / Photo3DCapture.reach[g.pos.x < 0 ? 0 : 1]) * R, py = CGFloat(-g.pos.y / Photo3DCapture.reach[g.pos.y > 0 ? 2 : 3]) * R
                Circle().stroke(.white, lineWidth: 3).frame(width: 18, height: 18).offset(x: max(-R - 10, min(R + 10, px)), y: max(-R - 10, min(R + 10, py)))
            }.frame(width: 170, height: 170)
            HStack {
                Text("\(g.frames) views").font(.caption.monospacedDigit())
                Spacer()
                if g.frames >= 10 { Button("Done", action: done).buttonStyle(.bordered).font(.caption) }
            }.frame(width: 220)
        }
        .padding(14).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }
}
