import SwiftUI
import MetalKit
import Accelerate
import simd

/// Gaussian splats made on the phone: each surface point of the LiDAR scan becomes a soft, see-through oval lying on
/// its surface, sized to its neighbours; colours come from the sharpest photo that sees each one (checked against that
/// photo's LiDAR depth). High detail splits every splat into four smaller photo-coloured ones. Drawn by a Metal splat
/// renderer (sorted, blended Gaussians) and exported as a standard 3D Gaussian Splatting .ply or a .splat file.
let SPLAT_SHADER = """
#include <metal_stdlib>
using namespace metal;
// one splat = 7 float4: pos+opacity, scale, rotation quaternion (x y z w), colour, then the view-dependent colour terms
struct Splat { float4 pos; float4 scale; float4 rot; float4 c0; float4 bx; float4 by; float4 bz; };
struct U { float4x4 view; float4x4 proj; float2 viewport; float2 focal; float4 cam; };
struct VOut { float4 position [[position]]; float2 d; float3 conic; float4 color; };

vertex VOut splat_vs(uint vid [[vertex_id]], uint iid [[instance_id]], constant Splat* S [[buffer(0)]], constant uint* order [[buffer(1)]], constant U& u [[buffer(2)]]) {
    Splat s = S[order[iid]];
    VOut o; o.color = float4(0); o.conic = float3(0); o.d = float2(0);
    float4 pv = u.view * float4(s.pos.xyz, 1);
    if (pv.z > -0.03) { o.position = float4(0, 0, 2, 1); return o; }
    float x = s.rot.x, y = s.rot.y, z = s.rot.z, w = s.rot.w;
    float3x3 R = float3x3(float3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
                          float3(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
                          float3(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)));
    float3x3 M = R * float3x3(float3(s.scale.x, 0, 0), float3(0, s.scale.y, 0), float3(0, 0, s.scale.z));
    float3x3 Sig = M * transpose(M);
    float3x3 W = float3x3(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    float tz = -pv.z;
    float3x3 J = float3x3(float3(u.focal.x / tz, 0, 0), float3(0, u.focal.y / tz, 0), float3(u.focal.x * pv.x / (tz * tz), u.focal.y * pv.y / (tz * tz), 0));
    float3x3 T = J * W;
    float3x3 cov = T * Sig * transpose(T);
    float a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;
    float det = a * c - b * b;
    if (det <= 1e-9) { o.position = float4(0, 0, 2, 1); return o; }
    o.conic = float3(c / det, -b / det, a / det);
    float mid = 0.5 * (a + c);
    float lam = mid + sqrt(max(0.1, mid * mid - det));
    float r = min(ceil(3.0 * sqrt(lam)), 256.0);
    float2 corner = float2((vid & 1) ? 1.0 : -1.0, (vid & 2) ? 1.0 : -1.0) * r;
    float4 pc = u.proj * pv;
    float2 ndc = pc.xy / pc.w + corner * 2.0 / u.viewport;
    o.position = float4(ndc, pc.z / pc.w, 1);
    o.d = corner;
    float3 dir = normalize(u.cam.xyz - s.pos.xyz);
    float3 col = clamp(s.c0.rgb + s.bx.rgb * dir.x + s.by.rgb * dir.y + s.bz.rgb * dir.z, 0.0, 1.0);
    o.color = float4(col, s.pos.w);
    return o;
}
fragment float4 splat_fs(VOut i [[stage_in]]) {
    float2 d = i.d;
    float power = -0.5 * (i.conic.x * d.x * d.x + i.conic.z * d.y * d.y) - i.conic.y * d.x * d.y;
    if (power > 0) discard_fragment();
    float alpha = min(0.99, i.color.a * exp(power));
    if (alpha < 1.0 / 255.0) discard_fragment();
    return float4(i.color.rgb * alpha, alpha);
}
"""

/// 7 float4 per splat: position + opacity, scale, rotation (x y z w), colour, and three view-dependent colour terms
struct SplatData { var buf: [SIMD4<Float>] = []; var count: Int { buf.count / 7 }; var centre = SIMD3<Float>.zero; var radius: Float = 1 }

enum SplatBuilder {
    static func build(_ clIn: Cloud, item: ScanItem, high: Bool, progress: @escaping (Float, String) -> Void) -> SplatData? {
        guard clIn.count > 500 else { return nil }
        let b0 = clIn.bounds, ext = max(b0.size.x, max(b0.size.y, b0.size.z))
        let g = max(0.003, min(0.01, ext / 1200))
        progress(0.02, "Spacing the splats...")
        var cl = CloudOps.dedupe(clIn, grid: g)
        if cl.count > 1_300_000 { cl = CloudOps.downsample(cl, to: 1_300_000) }
        let P = cl.p, n = P.count
        // surface direction and local spacing of each splat
        progress(0.1, "Finding each splat's surface...")
        let gi = GridIndex(P, cell: g * 3)
        var N = [SIMD3<Float>](repeating: SIMD3(0, 1, 0), count: n), sp = [Float](repeating: g, count: n), flat = [Float](repeating: 0, count: n)
        N.withUnsafeMutableBufferPointer { nb in sp.withUnsafeMutableBufferPointer { sb in flat.withUnsafeMutableBufferPointer { fb in
            let np = nb.baseAddress!, spp = sb.baseAddress!, fp = fb.baseAddress!
            parallelFor(n) { i in
                var nbh: [SIMD3<Float>] = []; nbh.reserveCapacity(40)
                let r = g * 3, r2 = r * r
                gi.around(P[i]) { j in if simd_length_squared(P[Int(j)] - P[i]) <= r2 { nbh.append(P[Int(j)]) } }
                spp[i] = max(g * 0.6, min(g * 3, r * (Float.pi / Float(max(1, nbh.count))).squareRoot()))   // sparse areas get bigger splats
                if let (nv, _, fl) = planeNormal(nbh) { np[i] = nv; fp[i] = fl } else { fp[i] = 1 }
            }
        } } }
        // children: 4 per splat in high detail (photo texture finer than the LiDAR)
        // 5.7 (39): at most ~1.6M splats in all (a 1.1M-point Gaussian scan at 4x made 4.4M - over 1 GB with the working
        // arrays - and iOS killed the app); High thins the points first so it still gets the finer photo colour
        if high && n * 4 > 1_600_000 {
            cl = CloudOps.downsample(cl, to: 400_000)
            return build(cl, item: item, high: true, progress: progress)
        }
        let subs: [SIMD2<Float>] = high ? [SIMD2(-1, -1), SIMD2(1, -1), SIMD2(-1, 1), SIMD2(1, 1)] : [SIMD2(0, 0)]
        let m = n * subs.count
        var pos = [SIMD3<Float>](repeating: .zero, count: m), nor = [SIMD3<Float>](repeating: .zero, count: m), scl = [Float](repeating: 0, count: m), fl2 = [Float](repeating: 0, count: m)
        for i in 0..<n {
            let nv = N[i], e1 = simd_normalize(simd_cross(nv, abs(nv.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0))), e2 = simd_cross(nv, e1)
            for (k, o) in subs.enumerated() {
                let j = i * subs.count + k
                pos[j] = P[i] + (e1 * o.x + e2 * o.y) * sp[i] / 4; nor[j] = nv; scl[j] = sp[i] * (high ? 0.45 : 0.8); fl2[j] = flat[i]
            }
        }
        // colours: the sharpest, most head-on, closest photo that really sees each splat
        progress(0.3, "Colouring from your photos...")
        var col = [SIMD3<Float>](repeating: .zero, count: m)
        for j in 0..<m { let c = cl.c[j / subs.count]; col[j] = SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)) / 255 }
        let views = DepthView.load(item, max: 50)
        if !views.isEmpty {
            let photos = PhotoView.load(views, max: 40, long: 1280)
            var best = [Float](repeating: 0, count: m), sure = [Bool](repeating: false, count: m)
            for (pi, ph) in photos.enumerated() {
                let camP = SIMD3<Float>(ph.dv.kf.T.columns.3.x, ph.dv.kf.T.columns.3.y, ph.dv.kf.T.columns.3.z)
                col.withUnsafeMutableBufferPointer { cb in best.withUnsafeMutableBufferPointer { bb in sure.withUnsafeMutableBufferPointer { ub in
                    let cp = cb.baseAddress!, bp = bb.baseAddress!, up = ub.baseAddress!
                    parallelFor(m) { j in
                        guard let (c, s) = ph.colour(pos[j], tol: 0.03) else { return }
                        // 5.7 (46): colour() lets a photo through where its LiDAR has no reading (no occlusion check), so a
                        // photo whose depth confirms the splat always beats one that only might see it
                        let ok = ph.dv.kf.project(pos[j]).map { ph.dv.measured($0.0, $0.1) > 0 } ?? false
                        if up[j] && !ok { return }
                        let face = abs(simd_dot(simd_normalize(camP - pos[j]), nor[j]))
                        let sc = s * max(0.05, face)
                        if sc > bp[j] || (ok && !up[j]) { bp[j] = sc; up[j] = ok; cp[j] = SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)) / 255 }
                    }
                } } }
                progress(0.3 + 0.6 * Float(pi + 1) / Float(photos.count), "Colouring from your photos (\(pi + 1)/\(photos.count))...")
            }
        }
        // pack
        var out = SplatData(); out.buf.reserveCapacity(m * 7)
        for j in 0..<m {
            let nv = nor[j]
            let q = simd_quatf(from: SIMD3(0, 0, 1), to: nv)
            let thin = fl2[j] < 0.05 ? scl[j] * 0.12 : scl[j] * 0.6          // flat surfaces: thin discs; corners: rounder
            out.buf.append(SIMD4(pos[j], 0.92))
            out.buf.append(SIMD4(scl[j], scl[j], thin, 0))
            out.buf.append(SIMD4(q.imag, q.real))
            out.buf.append(SIMD4(col[j], 0))
            out.buf.append(.zero); out.buf.append(.zero); out.buf.append(.zero)
        }
        out.centre = b0.centre; out.radius = max(0.3, simd_length(b0.size) / 2)
        progress(1, "\(m.formatted()) splats")
        return out
    }

    /// the standard 3D Gaussian Splatting .ply (SuperSplat, Polycam, Luma, Scaniverse web viewers, most splat tools)
    static func ply(_ s: SplatData) -> Data {
        let n = s.count
        // 5.7 (46): the trained view-dependent colour goes out as degree-1 SH (f_rest_0-8, channel-major), so the file
        // matches the viewer. Ours: c0 + bx*d.x + by*d.y + bz*d.z with d towards the camera; 3DGS uses -d and
        // -C1*y*sh1 + C1*z*sh2 - C1*x*sh3, so sh1 = by/C1, sh2 = -bz/C1, sh3 = bx/C1
        let rest = (0..<9).map { "property float f_rest_\($0)\n" }.joined()
        let head = Data("ply\nformat binary_little_endian 1.0\nelement vertex \(n)\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n\(rest)property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n".utf8)
        let C0: Float = 0.28209479, C1: Float = 0.48860251, F = 26
        var d = head; d.append(Data(count: n * F * 4))
        // written in place: a fresh [Float] per splat was millions of allocations on big scans
        d.withUnsafeMutableBytes { raw in
            let o = (raw.baseAddress! + head.count).bindMemory(to: Float.self, capacity: n * F)
            for i in 0..<n {
                let p = s.buf[i * 7], sc = s.buf[i * 7 + 1], q = s.buf[i * 7 + 2], c = s.buf[i * 7 + 3]
                let bx = s.buf[i * 7 + 4], by = s.buf[i * 7 + 5], bz = s.buf[i * 7 + 6]
                let op = min(0.999, max(0.001, p.w)), r = o + i * F
                r[0] = p.x; r[1] = p.y; r[2] = p.z; r[3] = 0; r[4] = 0; r[5] = 0
                r[6] = (c.x - 0.5) / C0; r[7] = (c.y - 0.5) / C0; r[8] = (c.z - 0.5) / C0
                for ch in 0..<3 { r[9 + ch * 3] = by[ch] / C1; r[10 + ch * 3] = -bz[ch] / C1; r[11 + ch * 3] = bx[ch] / C1 }
                r[18] = log(op / (1 - op)); r[19] = log(max(1e-6, sc.x)); r[20] = log(max(1e-6, sc.y)); r[21] = log(max(1e-6, sc.z))
                r[22] = q.w; r[23] = q.x; r[24] = q.y; r[25] = q.z
            }
        }
        return d
    }
    /// the compact .splat format (32 bytes each; antimatter15 / many web viewers)
    static func splatFile(_ s: SplatData) -> Data {
        var d = Data(); d.reserveCapacity(s.count * 32)
        for i in 0..<s.count {
            let p = s.buf[i * 7], sc = s.buf[i * 7 + 1], q = s.buf[i * 7 + 2], c = s.buf[i * 7 + 3]
            var f: [Float] = [p.x, p.y, p.z, sc.x, sc.y, sc.z]
            f.withUnsafeBytes { d.append(contentsOf: $0) }
            d.append(contentsOf: [u8(c.x * 255), u8(c.y * 255), u8(c.z * 255), u8(p.w * 255)])
            d.append(contentsOf: [q.w, q.x, q.y, q.z].map { u8($0 * 128 + 128) })
        }
        return d
    }
}

// ---------- the Metal splat renderer ----------
final class SplatRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private var pipe: MTLRenderPipelineState?
    private var splatBuf: MTLBuffer?
    private var orderBuf: MTLBuffer?
    private var count = 0
    private var positions: [Float] = []           // x, y, z for sorting
    private var sorting = false, sortedFor = SIMD3<Float>(repeating: .infinity)
    var target = SIMD3<Float>.zero, dist: Float = 2, yaw: Float = 0.6, pitch: Float = 0.35
    // 5.7 (47): the view only draws when asked (on-demand MTKView), so a still model costs no GPU or battery
    weak var view: MTKView?
    func kick() { view?.setNeedsDisplay() }
    struct U { var view: simd_float4x4; var proj: simd_float4x4; var viewport: SIMD2<Float>; var focal: SIMD2<Float>; var cam: SIMD4<Float> }

    init?(_ s: SplatData) {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        device = d; queue = q
        super.init()
        do {
            let lib = try d.makeLibrary(source: SPLAT_SHADER, options: nil)
            let pd = MTLRenderPipelineDescriptor()
            pd.vertexFunction = lib.makeFunction(name: "splat_vs"); pd.fragmentFunction = lib.makeFunction(name: "splat_fs")
            let ca = pd.colorAttachments[0]!; ca.pixelFormat = .bgra8Unorm; ca.isBlendingEnabled = true
            ca.sourceRGBBlendFactor = .one; ca.destinationRGBBlendFactor = .oneMinusSourceAlpha; ca.sourceAlphaBlendFactor = .one; ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            pipe = try d.makeRenderPipelineState(descriptor: pd)
        } catch { Log.add("splat shader: \(error)"); return nil }
        count = s.count
        splatBuf = s.buf.withUnsafeBytes { d.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        orderBuf = d.makeBuffer(length: max(4, count * 4), options: .storageModeShared)
        if let ob = orderBuf { let o = ob.contents().assumingMemoryBound(to: UInt32.self); for i in 0..<count { o[i] = UInt32(i) } }
        positions = [Float](repeating: 0, count: count * 3)
        for i in 0..<count { let p = s.buf[i * 7]; positions[i * 3] = p.x; positions[i * 3 + 1] = p.y; positions[i * 3 + 2] = p.z }
        target = s.centre; dist = s.radius * 2.2
    }
    var eye: SIMD3<Float> { target + dist * SIMD3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw)) }

    /// back-to-front order for this camera, worked out off the main thread (vDSP sorts a million keys quickly)
    private func resort() {
        guard !sorting else { return }
        let e = eye
        if simd_length(e - sortedFor) < dist * 0.03 { return }
        sorting = true
        let n = count, pos = positions, f = simd_normalize(target - e)
        DispatchQueue.global(qos: .userInitiated).async {
            var keys = [Float](repeating: 0, count: n)
            for i in 0..<n { keys[i] = (pos[i * 3] - e.x) * f.x + (pos[i * 3 + 1] - e.y) * f.y + (pos[i * 3 + 2] - e.z) * f.z }
            var idx = [vDSP_Length](0..<vDSP_Length(n))
            vDSP_vsorti(keys, &idx, nil, vDSP_Length(n), -1)       // farthest first
            // 5.7 (46): a fresh buffer, swapped in on the main thread, so frames already queued keep reading a whole old order
            guard let nb = self.device.makeBuffer(length: max(4, n * 4), options: .storageModeShared) else { DispatchQueue.main.async { self.sorting = false }; return }   // 5.7 (47): no kick here, or a failing makeBuffer would re-sort forever; the next gesture retries
            let o = nb.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<n { o[i] = UInt32(idx[i]) }
            // 5.7 (47): one more frame with the new order; its resort() sees sortedFor = eye and stops, unless the camera moved meanwhile
            DispatchQueue.main.async { self.orderBuf = nb; self.sorting = false; self.sortedFor = e; self.kick() }
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { kick() }
    func draw(in view: MTKView) {
        guard let pipe, let sb = splatBuf, let ob = orderBuf, let rp = view.currentRenderPassDescriptor, let dr = view.currentDrawable,
              let cb = queue.makeCommandBuffer(), let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { return }
        resort()
        let W = Float(view.drawableSize.width), H = Float(view.drawableSize.height)
        let e = eye, fz = simd_normalize(e - target), xa = simd_normalize(simd_cross(SIMD3(0, 1, 0), fz)), ya = simd_cross(fz, xa)
        var vm = matrix_identity_float4x4
        vm.columns = (SIMD4(xa.x, ya.x, fz.x, 0), SIMD4(xa.y, ya.y, fz.y, 0), SIMD4(xa.z, ya.z, fz.z, 0), SIMD4(-simd_dot(xa, e), -simd_dot(ya, e), -simd_dot(fz, e), 1))
        let fovy: Float = 50 * .pi / 180, f = 1 / tan(fovy / 2), near: Float = 0.01, far: Float = 100
        let pm = simd_float4x4(columns: (SIMD4(f * H / W, 0, 0, 0), SIMD4(0, f, 0, 0), SIMD4(0, 0, far / (near - far), -1), SIMD4(0, 0, near * far / (near - far), 0)))
        let focal = H / 2 / tan(fovy / 2)
        var u = U(view: vm, proj: pm, viewport: SIMD2(W, H), focal: SIMD2(focal, focal), cam: SIMD4(e, 1))
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(sb, offset: 0, index: 0); enc.setVertexBuffer(ob, offset: 0, index: 1)
        enc.setVertexBytes(&u, length: MemoryLayout<U>.stride, index: 2)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: count)
        enc.endEncoding(); cb.present(dr); cb.commit()
    }
}

struct SplatView: UIViewRepresentable {
    let renderer: SplatRenderer
    final class Coord: NSObject, UIGestureRecognizerDelegate {
        var r: SplatRenderer; var lastPan = CGPoint.zero; var lastPinch: CGFloat = 1; var lastTwo = CGPoint.zero
        init(_ r: SplatRenderer) { self.r = r }
        // 5.7 (46): pinch and the two-finger move run together, or whichever starts first blocks the other
        func gestureRecognizer(_ g: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith o: UIGestureRecognizer) -> Bool {
            func two(_ x: UIGestureRecognizer) -> Bool { (x as? UIPanGestureRecognizer)?.minimumNumberOfTouches == 2 }
            return (g is UIPinchGestureRecognizer && two(o)) || (o is UIPinchGestureRecognizer && two(g))
        }
        @objc func pan(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view); if g.state == .began { lastPan = .zero }
            let dx = Float(t.x - lastPan.x), dy = Float(t.y - lastPan.y); lastPan = t
            r.yaw -= dx * 0.008; r.pitch = max(-1.4, min(1.4, r.pitch + dy * 0.008)); r.kick()
        }
        @objc func pinch(_ g: UIPinchGestureRecognizer) {
            if g.state == .began { lastPinch = 1 }
            r.dist = max(0.05, r.dist * Float(lastPinch / g.scale)); lastPinch = g.scale; r.kick()
        }
        @objc func two(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view); if g.state == .began { lastTwo = .zero }
            let dx = Float(t.x - lastTwo.x), dy = Float(t.y - lastTwo.y); lastTwo = t
            let e = r.eye, fz = simd_normalize(e - r.target), xa = simd_normalize(simd_cross(SIMD3(0, 1, 0), fz)), ya = simd_cross(fz, xa)
            r.target += (-xa * dx + ya * dy) * r.dist * 0.0015; r.kick()
        }
    }
    func makeCoordinator() -> Coord { Coord(renderer) }
    func makeUIView(context: Context) -> MTKView {
        let v = MTKView(frame: .zero, device: renderer.device)
        v.colorPixelFormat = .bgra8Unorm; v.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        v.delegate = renderer; v.preferredFramesPerSecond = 60
        v.isPaused = true; v.enableSetNeedsDisplay = true      // 5.7 (47): draws on kick() only - gestures, sizing, a finished sort
        renderer.view = v
        let r = renderer; DispatchQueue.main.async { r.kick() }   // first frame after layout, so the model shows on open
        let p = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coord.pan(_:))); p.maximumNumberOfTouches = 1
        let t = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coord.two(_:))); t.minimumNumberOfTouches = 2
        let z = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coord.pinch(_:)))
        t.delegate = context.coordinator; z.delegate = context.coordinator
        v.addGestureRecognizer(p); v.addGestureRecognizer(t); v.addGestureRecognizer(z)
        return v
    }
    func updateUIView(_ v: MTKView, context: Context) {
        // 5.7 (46): a trained / retrained renderer must replace the old one, or the view keeps drawing the LiDAR-init splats
        let c = context.coordinator; guard c.r !== renderer else { return }
        renderer.yaw = c.r.yaw; renderer.pitch = c.r.pitch; renderer.dist = c.r.dist; renderer.target = c.r.target
        c.r = renderer; v.delegate = renderer; renderer.view = v; renderer.kick()
    }
}

struct SplatSheet: View {
    let item: ScanItem
    var cloud: Cloud? = nil
    var startDetail = 0
    var autoStart = false
    @Environment(\.dismiss) private var dismiss
    @State private var detail = 0                 // 0 normal, 1 high, 2 Aran splat
    @State private var busy = false
    @State private var progress: Float = 0
    @State private var status = ""
    @State private var data: SplatData?
    @State private var renderer: SplatRenderer?
    @State private var share: URL?
    @State private var started = false
    @State private var trainer: SplatTrainer?        // 5.7 (41) kept, so "Train more" carries on where it stopped
    @State private var trainedFor: Double = 0
    @State private var job: Task<Void, Never>?       // 5.7 (46): cancelled on close, so a closed sheet stops building
    @State private var exporting = false
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                if let r = renderer { SplatView(renderer: r).ignoresSafeArea() }
                else {
                    VStack(spacing: 14) {
                        Image(systemName: "sparkles").font(.system(size: 52)).foregroundStyle(.purple)
                        Text("Gaussian splats: every surface point becomes a soft oval coloured from your photos - smooth and photo-like, made on the phone.")
                            .multilineTextAlignment(.center).font(.subheadline).padding(.horizontal)
                        // 5.7 (46): with photos High trains longer at the same count (4x splats would not fit training memory),
                        // and the choice is read once at the start, so it is locked while building
                        Picker("Detail", selection: $detail) { Text("Normal").tag(0); Text(Library.captureURL(item) == nil ? "High (4x)" : "High (longer training)").tag(1) }
                            .pickerStyle(.segmented).padding(.horizontal, 30).disabled(busy)
                        if busy { ProgressView(value: progress).frame(width: 240); Text(status).font(.caption).foregroundStyle(.secondary) }
                        else {
                            Button("Make splats") { make() }.buttonStyle(.borderedProminent).controlSize(.large)
                            if !status.isEmpty { Text(status).font(.caption).foregroundStyle(.orange) }   // 5.7 (46): why the last try failed
                        }
                        if Library.captureURL(item) == nil { Text("This scan has no photos, so the splats use the scan's own colours.").font(.caption).foregroundStyle(.orange) }
                    }.foregroundStyle(.white)
                }
                if renderer != nil {
                    VStack(spacing: 8) {
                        Spacer()
                        if busy { ProgressView(value: progress).frame(width: 240) }
                        Text(busy ? status : "\(status) - drag to turn, pinch to zoom, two fingers to move").font(.caption).padding(8).background(.ultraThinMaterial, in: Capsule())
                        if trainer != nil && !busy { Button { train(60) } label: { Label("Train more (1 min)", systemImage: "sparkles") }.buttonStyle(.borderedProminent) }
                    }.padding(.bottom, 20)
                }
            }
            .navigationTitle("Gaussian splats").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button { dismiss() } label: { Label("Close", systemImage: "xmark") }.labelStyle(.titleAndIcon) }
                // 5.7 (46): hidden while training, which would export the untrained / previous round's splats
                if let d = data, !busy {
                    ToolbarItem(placement: .confirmationAction) {
                        if exporting { ProgressView() } else {
                            Menu {
                                Button("Export .ply (3D Gaussian Splatting)") { export(d, "\(item.name) splats.ply", SplatBuilder.ply) }
                                Button("Export .splat (web viewers)") { export(d, "\(item.name).splat", SplatBuilder.splatFile) }
                            } label: { Image(systemName: "square.and.arrow.up") }
                        }
                    }
                }
            }
            .sheet(item: $share) { ShareSheet(items: [$0]) }
            .onAppear { if !started { started = true; detail = min(1, startDetail); if autoStart { make() } } }
            .onDisappear { job?.cancel(); job = nil }
        }
    }
    /// 5.7 (46): builds and writes the file off the main thread, and only shares it if the write worked
    private func export(_ d: SplatData, _ name: String, _ make: @escaping (SplatData) -> Data) {
        exporting = true
        Task.detached(priority: .userInitiated) {
            let u = Export.tmp(name)
            let ok = (try? make(d).write(to: u)) != nil
            await MainActor.run { exporting = false; if ok { share = u } else { status = "Could not save the export (is the phone full?)" } }
        }
    }
    private func make() {
        busy = true; status = ""
        let given = cloud, it = item, dt = detail
        job = Task.detached(priority: .userInitiated) {
            guard let c = given ?? Library.load(it) else { await MainActor.run { busy = false; status = "Could not open the scan" }; return }
            let report: (Float, String) -> Void = { p, s in Task { @MainActor in progress = p; status = s } }
            // 5.7 (41): with photos, the splats are TRAINED against them (Scaniverse-style): they start on the LiDAR
            // points (thinned so the phone trains them quickly) and are optimised until each photo's view matches
            if dt != 2, let cap = Library.captureURL(it) {
                let views = SplatTrainer.loadViews(frames: cap.appendingPathComponent("frames"), long: 480, max: 160)
                if views.count >= 8, let d0 = SplatBuilder.build(c.count > 400_000 ? CloudOps.downsample(c, to: 400_000) : c, item: it, high: false, progress: { p, s in report(p * 0.2, s) }),
                   !Task.isCancelled, let tr = SplatTrainer(params: SplatTrainer.params(fromViewer: d0.buf), views: views, extent: d0.radius * 2) {
                    if Task.isCancelled { return }
                    await MainActor.run { trainer = tr; data = d0; renderer = SplatRenderer(d0) }
                    await MainActor.run { train(dt == 1 ? 150 : 90) }
                    return
                }
            }
            if Task.isCancelled { return }
            let d = dt == 2 ? SplatBuilder.aran(c, item: it, progress: report) : SplatBuilder.build(c, item: it, high: dt == 1, progress: report)
            if Task.isCancelled { return }
            let r = d.flatMap { SplatRenderer($0) }
            await MainActor.run { busy = false; data = d; renderer = r; if d == nil { status = "Not enough points for splats." } else if r == nil { status = "This phone's graphics could not start the splat viewer." }; Log.add("splats \(d?.count ?? 0)") }
        }
    }
    /// trains for `secs` more seconds, then shows the result
    private func train(_ secs: Double) {
        guard let tr = trainer, let d0 = data else { return }
        busy = true; progress = 0; status = "Training the splats on your photos..."
        let all = Array(0..<tr.views.count), before = trainedFor
        job = Task.detached(priority: .userInitiated) {
            // 5.7 (46): a closed sheet stops the run early and skips the export and viewer
            if Task.isCancelled { return }
            tr.train(views: all, seconds: secs, stop: { Task.isCancelled }) { f, e in Task { @MainActor in progress = f; status = String(format: "Training on your photos - %.0f%% (error %.1f%%)", f * 100, e * 100) } }
            if Task.isCancelled { return }
            var d = SplatData(); d.buf = tr.export(); d.centre = d0.centre; d.radius = d0.radius
            let r = SplatRenderer(d)
            await MainActor.run {
                trainedFor = before + secs; busy = false
                // 5.7 (46): only claim (and export) the trained splats if the viewer could show them
                if let r { renderer = r; data = d; status = "\(d.count.formatted()) splats, trained \(Int(trainedFor)) s" }
                else { status = "Trained, but this phone could not show the result - showing the earlier splats" }
                Log.add("splats trained \(Int(trainedFor)) s, \(tr.iteration) steps, \(d.count)")
            }
        }
    }
}

// ---------- Aran splat: multi-view-consistent photo-pixel splats on top of the LiDAR splats ----------
extension SplatBuilder {
    /// 1. for each sharp photo, the LiDAR depth is lifted to photo resolution along the photo's own edges (each pixel takes
    ///    the nearby LiDAR depth whose colour matches its own best), 2. every pixel becomes a pixel-sized splat in 3D,
    /// 3. it is kept only if two or more OTHER photos agree on both its depth and its colour (reflections, flicker and
    ///    floating junk fail), 4. its colour is the middle of the agreeing photos, 5. LiDAR splats fill where photos are thin.
    static func aran(_ cl: Cloud, item: ScanItem, progress: @escaping (Float, String) -> Void) -> SplatData? {
        guard var base = build(cl, item: item, high: false, progress: { p, s in progress(p * 0.3, s) }) else { return nil }
        let views = DepthView.load(item, max: 70)
        guard views.count >= 4 else { progress(1, "Aran splat needs the scan's photos; made plain splats"); return base }
        progress(0.32, "Loading photos for Aran splat...")
        let photos = PhotoView.load(views, max: 70, long: 480)
        let sources = Array(photos.indices.sorted { photos[$0].dv.kf.sharp > photos[$1].dv.kf.sharp }.prefix(24))
        // one photo splat per 1.5 mm (several photos saw the same spot)
        // 5.7 (48): filled photo by photo (no all-pixels arrays next to it), and at most ~1.6M splats in all like
        // build's 5.7 (39) cap - a room's photos made up to ~5M (over 1.5 GB with the viewer's copy) and iOS killed the app
        let maxSplats = 1_600_000
        var cells: [SIMD3<Int32>: (SIMD3<Float>, SIMD3<Float>, Float, Float)] = [:], nPhoto = 0
        for (si, s) in sources.enumerated() {
            let ph = photos[s], v = ph.dv, k = v.kf
            let W = ph.pw, H = ph.ph, sx = k.w / Float(W)
            let fx = k.K[0][0] / sx, fy = k.K[1][1] / sx, cx = k.K[2][0] / sx, cy = k.K[2][1] / sx
            // 5.7 (48): top-down like the other photos below (5.7 (47)) - this read the mirrored row
            @inline(__always) func pix(_ x: Int, _ y: Int) -> SIMD3<Float> { let o = (y * W + x) * 4; return SIMD3(Float(ph.rgba[o]), Float(ph.rgba[o + 1]), Float(ph.rgba[o + 2])) }
            var outP = [SIMD3<Float>?](repeating: nil, count: W * H), outC = [SIMD3<Float>](repeating: .zero, count: W * H), outS = [Float](repeating: 0, count: W * H)
            outP.withUnsafeMutableBufferPointer { pb in outC.withUnsafeMutableBufferPointer { cb in outS.withUnsafeMutableBufferPointer { sb in
                let pp = pb.baseAddress!, cp = cb.baseAddress!, sp = sb.baseAddress!
                parallelFor(W * H) { idx in
                    let x = idx % W, y = idx / W
                    let me = pix(x, y)
                    // edge-aware depth: the nearby LiDAR sample whose colour matches this pixel best
                    let fxd = Float(x) / Float(W) * Float(v.dw), fyd = Float(y) / Float(H) * Float(v.dh)
                    var best: Float = .greatestFiniteMagnitude, d: Float = 0
                    for oy in -1...1 { for ox in -1...1 {
                        let qx = min(v.dw - 1, max(0, Int(fxd.rounded()) + ox)), qy = min(v.dh - 1, max(0, Int(fyd.rounded()) + oy))
                        let z = v.depth[qy * v.dw + qx]; guard z > 0.1 else { continue }
                        let px = min(W - 1, Int((Float(qx) + 0.5) / Float(v.dw) * Float(W))), py = min(H - 1, Int((Float(qy) + 0.5) / Float(v.dh) * Float(H)))
                        let cd = simd_reduce_add(abs(pix(px, py) - me)) / 255
                        let dd = (fxd - Float(qx) - 0.5) * (fxd - Float(qx) - 0.5) + (fyd - Float(qy) - 0.5) * (fyd - Float(qy) - 0.5)
                        let cost = cd * 4 + dd * 0.15
                        if cost < best { best = cost; d = z }
                    } }
                    guard d > 0.1, d < 3.5 else { return }
                    let pc = k.T * SIMD4<Float>((Float(x) - cx) * d / fx, -(Float(y) - cy) * d / fy, -d, 1)
                    let p = SIMD3<Float>(pc.x, pc.y, pc.z)
                    // the other photos must agree on depth AND colour
                    var agree = 0, seen = 0, cols: [SIMD3<Float>] = [me]
                    for (oi, o) in photos.enumerated() where oi != s {
                        guard let (u, vv, dz) = o.dv.kf.project(p) else { continue }
                        let md = o.dv.measured(u, vv); guard md > 0 else { continue }
                        seen += 1
                        let ox = Int(u / o.dv.kf.w * Float(o.pw)), oy = Int(vv / o.dv.kf.h * Float(o.ph))
                        guard ox >= 0, oy >= 0, ox < o.pw, oy < o.ph else { continue }
                        let oo = (oy * o.pw + ox) * 4                       // 5.7 (47): CoreImage bitmaps run top-down
                        let oc = SIMD3<Float>(Float(o.rgba[oo]), Float(o.rgba[oo + 1]), Float(o.rgba[oo + 2]))
                        if abs(dz - md) < 0.025 && simd_reduce_add(abs(oc - me)) < 90 { agree += 1; if cols.count < 9 { cols.append(oc) } }
                    }
                    guard agree >= 2, Float(agree) >= Float(seen) * 0.4 else { return }
                    // middle colour per channel (a reflection in one photo cannot win)
                    func med(_ a: [Float]) -> Float { let s = a.sorted(); return s[s.count / 2] }
                    pp[idx] = p; cp[idx] = SIMD3(med(cols.map { $0.x }), med(cols.map { $0.y }), med(cols.map { $0.z })) / 255
                    sp[idx] = d / fx * 0.75
                }
            } } }
            for i in 0..<(W * H) {
                guard let q = outP[i] else { continue }
                let key = SIMD3<Int32>(ik((q.x / 0.0015)), ik((q.y / 0.0015)), ik((q.z / 0.0015)))
                if let o = cells[key] { cells[key] = (o.0 + q, o.1 + outC[i], o.2 + outS[i], o.3 + 1) }
                else if cells.count < maxSplats { cells[key] = (q, outC[i], outS[i], 1) }   // sharpest photos come first
                else { continue }
                nPhoto += 1
            }
            progress(0.35 + 0.55 * Float(si + 1) / Float(sources.count), "Aran splat: photo \(si + 1) of \(sources.count), \(nPhoto.formatted()) photo splats")
        }
        // LiDAR splats where the photo splats are go (no blur under sharp detail)
        var covered = Set<SIMD3<Int32>>()
        for (_, v) in cells { let q = v.0 / v.3; covered.insert(SIMD3(ik((q.x / 0.006)), ik((q.y / 0.006)), ik((q.z / 0.006)))) }
        var keep = [Bool](repeating: false, count: base.count), kept = 0
        for i in 0..<base.count {
            let p = base.buf[i * 7]
            if !covered.contains(SIMD3(ik((p.x / 0.006)), ik((p.y / 0.006)), ik((p.z / 0.006)))) { keep[i] = true; kept += 1 }
        }
        covered = []
        // 5.7 (48): photo splats over the budget left by the LiDAR fill are evenly thinned (cells come in hash order,
        // so spatially mixed) and the survivors grow to cover the same area
        let photoKeep = cells.count > maxSplats - kept ? Float(max(0, maxSplats - kept)) / Float(cells.count) : 1
        let grow = photoKeep < 1 ? (1 / max(photoKeep, 0.01)).squareRoot() : 1
        var out = SplatData(); out.centre = base.centre; out.radius = base.radius
        out.buf.reserveCapacity((kept + Int(Float(cells.count) * photoKeep) + 1) * 7)
        for i in 0..<base.count where keep[i] { out.buf.append(contentsOf: base.buf[(i * 7)..<(i * 7 + 7)]) }
        keep = []
        var acc: Float = 0, nOut = 0
        for (_, v) in cells {
            acc += photoKeep; guard acc >= 1 else { continue }; acc -= 1; nOut += 1
            let q = v.0 / v.3, c = v.1 / v.3, s = v.2 / v.3 * grow
            out.buf.append(SIMD4(q, 0.9)); out.buf.append(SIMD4(s, s, s, 0)); out.buf.append(SIMD4(0, 0, 0, 1)); out.buf.append(SIMD4(c, 0))
            out.buf.append(.zero); out.buf.append(.zero); out.buf.append(.zero)
        }
        base = out
        progress(1, "Aran splat: \(nOut.formatted()) photo splats + \(kept.formatted()) LiDAR splats")   // 5.7 (46): the ones actually kept
        return base
    }
}
