import Foundation
import Metal
import simd
import ImageIO
import CoreGraphics

/// 5.7 Gaussian splat TRAINING on the phone (Scaniverse-style): the splats start on the LiDAR points and are then
/// optimised against the scan's own photos on the GPU - position, size, shape, opacity and view-dependent colour - so
/// what you see from where a photo was taken is that photo. A tile rasteriser renders each photo's view, the L1 error
/// is pushed back through the same blending (forward and backward in one pass per tile), and Adam steps every splat.
let SPLAT_TRAIN_SHADER = """
#include <metal_stdlib>
using namespace metal;
#define TS 16
#define PS 24
// params per splat (24 floats): 0-2 pos, 3-5 log scale, 6-9 rotation (x y z w), 10 opacity logit, 11-13 colour,
// 14-16 / 17-19 / 20-22 colour change along the view direction's x / y / z
struct View { float3x3 R; float3 t; float4 K; uint2 size; uint2 tiles; float3 cam; uint img; };
struct Opt { float lrPos; float lrScale; float lrRot; float lrOp; float lrC0; float lrC1; float b1t; float b2t; float maxLogScale; uint N; };

inline float3x3 quatR(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(float3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
                    float3(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
                    float3(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)));
}
inline float3x3 outer(float3 a, float3 b) { return float3x3(a * b.x, a * b.y, a * b.z); }

kernel void project(device const float* G [[buffer(0)]], constant View& V [[buffer(1)]], device float4* mean [[buffer(2)]],
                    device float4* conop [[buffer(3)]], device float4* col [[buffer(4)]], device atomic_uint* tileCount [[buffer(5)]],
                    device uint4* rect [[buffer(6)]], constant uint& N [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    device const float* g = G + i * PS;
    mean[i] = float4(0); rect[i] = uint4(0);
    float3 p = float3(g[0], g[1], g[2]);
    float3 pc = V.R * p + V.t;
    float z = -pc.z;
    if (z < 0.05) return;
    float fx = V.K.x, fy = V.K.y, cx = V.K.z, cy = V.K.w;
    float u = cx + fx * pc.x / z, v = cy - fy * pc.y / z;
    float3 s = exp(float3(g[3], g[4], g[5]));
    float4 q = normalize(float4(g[6], g[7], g[8], g[9]));
    float3x3 M = quatR(q) * float3x3(float3(s.x, 0, 0), float3(0, s.y, 0), float3(0, 0, s.z));
    float3x3 Sig = M * transpose(M);
    float3 t0 = transpose(V.R) * float3(fx / z, 0, fx * pc.x / (z * z));
    float3 t1 = transpose(V.R) * float3(0, -fy / z, -fy * pc.y / (z * z));
    float a = dot(t0, Sig * t0) + 0.3, b = dot(t0, Sig * t1), c = dot(t1, Sig * t1) + 0.3;
    float det = a * c - b * b;
    if (det <= 1e-12) return;
    float mid = 0.5 * (a + c), lam = mid + sqrt(max(0.1, mid * mid - det));
    float r = min(ceil(3.0 * sqrt(lam)), 160.0);
    float W = float(V.size.x), H = float(V.size.y);
    if (u + r < 0 || v + r < 0 || u - r >= W || v - r >= H) return;
    uint x0 = uint(max(0.0, floor((u - r) / TS))), y0 = uint(max(0.0, floor((v - r) / TS)));
    uint x1 = min(V.tiles.x, uint(floor((u + r) / TS)) + 1), y1 = min(V.tiles.y, uint(floor((v + r) / TS)) + 1);
    if (x1 <= x0 || y1 <= y0) return;
    float op = 1.0 / (1.0 + exp(-g[10]));
    float3 dir = normalize(V.cam - p);
    float3 raw = float3(g[11], g[12], g[13]) + float3(g[14], g[15], g[16]) * dir.x + float3(g[17], g[18], g[19]) * dir.y + float3(g[20], g[21], g[22]) * dir.z;
    mean[i] = float4(u, v, z, r);
    conop[i] = float4(c / det, -b / det, a / det, op);
    col[i] = float4(clamp(raw, 0.0, 1.0), 0);
    rect[i] = uint4(x0, y0, x1, y1);
    for (uint ty = y0; ty < y1; ty++) for (uint tx = x0; tx < x1; tx++) atomic_fetch_add_explicit(&tileCount[ty * V.tiles.x + tx], 1u, memory_order_relaxed);
}

kernel void emit(device const float4* mean [[buffer(0)]], device const uint4* rect [[buffer(1)]], constant View& V [[buffer(2)]],
                 device atomic_uint* cursor [[buffer(3)]], device uint2* entries [[buffer(4)]], constant uint& N [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    uint4 rc = rect[i];
    if (rc.z <= rc.x) return;
    uint d = as_type<uint>(mean[i].z);
    for (uint ty = rc.y; ty < rc.w; ty++) for (uint tx = rc.x; tx < rc.z; tx++) {
        uint s = atomic_fetch_add_explicit(&cursor[ty * V.tiles.x + tx], 1u, memory_order_relaxed);
        entries[s] = uint2(d, i);
    }
}

// one threadgroup per 16x16 tile: blend front to back, then (training) walk back to front for the gradients
kernel void render(device const uint2* entries [[buffer(0)]], device const uint2* ranges [[buffer(1)]], device const float4* mean [[buffer(2)]],
                   device const float4* conop [[buffer(3)]], device const float4* col [[buffer(4)]], constant View& V [[buffer(5)]],
                   device const uchar4* images [[buffer(6)]], device atomic_float* g2 [[buffer(7)]], device float4* outImg [[buffer(8)]],
                   device atomic_float* lossSum [[buffer(9)]], constant uint& backward [[buffer(10)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint2 lt [[thread_position_in_threadgroup]],
                   uint li [[thread_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]]) {
    threadgroup uint sid[256]; threadgroup float4 smean[256]; threadgroup float4 scon[256]; threadgroup float4 scol[256];
    uint tile = tg.y * V.tiles.x + tg.x;
    uint2 rg = ranges[tile];
    uint n = rg.y - rg.x;
    uint2 pix = tg * TS + lt;
    bool inside = pix.x < V.size.x && pix.y < V.size.y;
    float2 pf = float2(pix);
    float T = 1; float3 C = float3(0); uint last = 0; bool done = !inside;
    for (uint b0 = 0; b0 < n; b0 += 256) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint k = b0 + li;
        if (k < n) { uint id = entries[rg.x + k].y; sid[li] = id; smean[li] = mean[id]; scon[li] = conop[id]; scol[li] = col[id]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint m = min(256u, n - b0);
        for (uint j = 0; j < m && !done; j++) {
            float2 d = smean[j].xy - pf; float4 co = scon[j];
            float power = -0.5 * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            if (power > 0) continue;
            float alpha = min(0.99, co.w * exp(power));
            if (alpha < 1.0 / 255.0) continue;
            float tt = T * (1 - alpha);
            if (tt < 1e-4) { done = true; break; }
            C += scol[j].rgb * alpha * T; T = tt; last = b0 + j + 1;
        }
    }
    float3 dC = float3(0); float e1 = 0;
    if (inside) {
        uint pi = pix.y * V.size.x + pix.x;
        outImg[pi] = float4(C, 1);
        float3 gt = float3(images[V.img + pi].rgb) / 255.0;
        float3 e = C - gt;
        e1 = abs(e.x) + abs(e.y) + abs(e.z);
        dC = sign(e) / float(V.size.x * V.size.y * 3);
    }
    e1 = simd_sum(e1);
    if (sl == 0) atomic_fetch_add_explicit(lossSum, e1, memory_order_relaxed);
    if (backward == 0) return;
    float Tb = T; float3 acc = float3(0); float lastA = 0; float3 lastC = float3(0);
    uint nb = (n + 255) / 256;
    for (int bi = int(nb) - 1; bi >= 0; bi--) {
        uint b0 = uint(bi) * 256;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint k = b0 + li;
        if (k < n) { uint id = entries[rg.x + k].y; sid[li] = id; smean[li] = mean[id]; scon[li] = conop[id]; scol[li] = col[id]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint m = min(256u, n - b0);
        for (int j = int(m) - 1; j >= 0; j--) {
            uint gi = b0 + uint(j);
            float gm0 = 0, gm1 = 0, gA = 0, gB = 0, gC = 0, gop = 0; float3 gc = float3(0);
            if (inside && gi < last) {
                float2 d = smean[j].xy - pf; float4 co = scon[j];
                float power = -0.5 * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
                float G = exp(power);
                float alpha = min(0.99, co.w * G);
                if (power <= 0 && alpha >= 1.0 / 255.0) {
                    Tb = Tb / (1 - alpha);
                    float3 c = scol[j].rgb;
                    gc = alpha * Tb * dC;
                    acc = lastA * lastC + (1 - lastA) * acc;
                    lastC = c;
                    float dLda = dot(c - acc, dC) * Tb;
                    lastA = alpha;
                    float dLdG = co.w * dLda;
                    float gdx = G * d.x, gdy = G * d.y;
                    gm0 = dLdG * (-gdx * co.x - gdy * co.y);
                    gm1 = dLdG * (-gdy * co.z - gdx * co.y);
                    gA = dLdG * G * (-0.5 * d.x * d.x); gB = dLdG * G * (-d.x * d.y); gC = dLdG * G * (-0.5 * d.y * d.y);
                    gop = G * dLda;
                }
            }
            gm0 = simd_sum(gm0); gm1 = simd_sum(gm1); gA = simd_sum(gA); gB = simd_sum(gB); gC = simd_sum(gC); gop = simd_sum(gop);
            gc = float3(simd_sum(gc.x), simd_sum(gc.y), simd_sum(gc.z));
            if (sl == 0 && (gm0 != 0 || gm1 != 0 || gop != 0 || gc.x != 0 || gc.y != 0 || gc.z != 0)) {
                device atomic_float* o = g2 + sid[j] * 10;
                atomic_fetch_add_explicit(o + 0, gm0, memory_order_relaxed); atomic_fetch_add_explicit(o + 1, gm1, memory_order_relaxed);
                atomic_fetch_add_explicit(o + 2, gA, memory_order_relaxed); atomic_fetch_add_explicit(o + 3, gB, memory_order_relaxed);
                atomic_fetch_add_explicit(o + 4, gC, memory_order_relaxed); atomic_fetch_add_explicit(o + 5, gop, memory_order_relaxed);
                atomic_fetch_add_explicit(o + 6, gc.x, memory_order_relaxed); atomic_fetch_add_explicit(o + 7, gc.y, memory_order_relaxed);
                atomic_fetch_add_explicit(o + 8, gc.z, memory_order_relaxed);
            }
        }
    }
}

inline void adam(device float* g, device float* m, device float* v, uint k, float grad, float lr, constant Opt& O) {
    float mm = 0.9 * m[k] + 0.1 * grad, vv = 0.999 * v[k] + 0.001 * grad * grad;
    m[k] = mm; v[k] = vv;
    g[k] -= lr * (mm / O.b1t) / (sqrt(vv / O.b2t) + 1e-15);
}

// the screen-space gradients back to every parameter, then one Adam step (only splats this photo saw)
kernel void step(device float* G [[buffer(0)]], device float* Mo [[buffer(1)]], device float* Ve [[buffer(2)]], device float* g2 [[buffer(3)]],
                 device const float4* mean [[buffer(4)]], constant View& V [[buffer(5)]], constant Opt& O [[buffer(6)]],
                 device float* stats [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= O.N) return;
    device float* gg = g2 + i * 10;
    if (mean[i].w == 0) return;
    float gm0 = gg[0], gm1 = gg[1], gA = gg[2], gB = gg[3], gC = gg[4], gop = gg[5];
    float3 gcol = float3(gg[6], gg[7], gg[8]);
    for (int k = 0; k < 10; k++) gg[k] = 0;
    device float* g = G + i * PS; device float* m = Mo + i * PS; device float* v = Ve + i * PS;
    float3 p = float3(g[0], g[1], g[2]);
    float3 pc = V.R * p + V.t;
    float z = -pc.z;
    float fx = V.K.x, fy = V.K.y;
    float3 s = exp(float3(g[3], g[4], g[5]));
    float4 qr = float4(g[6], g[7], g[8], g[9]); float ql = max(1e-8, length(qr)); float4 q = qr / ql;
    float3x3 R = quatR(q);
    float3x3 M = R * float3x3(float3(s.x, 0, 0), float3(0, s.y, 0), float3(0, 0, s.z));
    float3x3 Sig = M * transpose(M);
    float3 t0 = transpose(V.R) * float3(fx / z, 0, fx * pc.x / (z * z));
    float3 t1 = transpose(V.R) * float3(0, -fy / z, -fy * pc.y / (z * z));
    float a = dot(t0, Sig * t0) + 0.3, b = dot(t0, Sig * t1), c = dot(t1, Sig * t1) + 0.3;
    float det = a * c - b * b, d2 = det * det;
    float ga = gA * (-c * c / d2) + gB * (b * c / d2) + gC * (-b * b / d2);
    float gb = gA * (2 * b * c / d2) + gB * (-(det + 2 * b * b) / d2) + gC * (2 * a * b / d2);
    float gc = gA * (-b * b / d2) + gB * (b * a / d2) + gC * (-a * a / d2);
    float3x3 GS = outer(t0, t0) * ga + outer(t0, t1) * gb + outer(t1, t1) * gc;
    float3x3 dM = (GS + transpose(GS)) * M;
    float3 gs = float3(dot(R[0], dM[0]), dot(R[1], dM[1]), dot(R[2], dM[2])) * s;
    float3x3 GR = float3x3(dM[0] * s.x, dM[1] * s.y, dM[2] * s.z);
    // G_ij = row i, column j = GR[j][i]
    float G00 = GR[0][0], G01 = GR[1][0], G02 = GR[2][0], G10 = GR[0][1], G11 = GR[1][1], G12 = GR[2][1], G20 = GR[0][2], G21 = GR[1][2], G22 = GR[2][2];
    float x = q.x, y = q.y, zq = q.z, w = q.w;
    float gw = 2 * (-zq * G01 + y * G02 + zq * G10 - x * G12 - y * G20 + x * G21);
    float gx = 2 * (y * G01 + zq * G02 + y * G10 - 2 * x * G11 - w * G12 + zq * G20 + w * G21 - 2 * x * G22);
    float gy = 2 * (-2 * y * G00 + x * G01 + w * G02 + x * G10 + zq * G12 - w * G20 + zq * G21 - 2 * y * G22);
    float gz = 2 * (-2 * zq * G00 - w * G01 + x * G02 + w * G10 - 2 * zq * G11 + y * G12 + x * G20 + y * G21);
    float4 gq = float4(gx, gy, gz, gw);
    gq = (gq - q * dot(q, gq)) / ql;
    float3 gpc = float3(gm0 * fx / z, -gm1 * fy / z, gm0 * fx * pc.x / (z * z) - gm1 * fy * pc.y / (z * z));
    float3 gp = transpose(V.R) * gpc;
    float3 dir = normalize(V.cam - p);
    float3 raw = float3(g[11], g[12], g[13]) + float3(g[14], g[15], g[16]) * dir.x + float3(g[17], g[18], g[19]) * dir.y + float3(g[20], g[21], g[22]) * dir.z;
    float3 graw = select(float3(0), gcol, raw > 0.0 && raw < 1.0);
    float op = 1.0 / (1.0 + exp(-g[10]));
    float gol = gop * op * (1 - op);
    for (int k = 0; k < 3; k++) adam(g, m, v, k, gp[k], O.lrPos, O);
    for (int k = 0; k < 3; k++) { adam(g, m, v, 3 + k, gs[k], O.lrScale, O); g[3 + k] = min(g[3 + k], O.maxLogScale); }
    for (int k = 0; k < 4; k++) adam(g, m, v, 6 + k, gq[k], O.lrRot, O);
    adam(g, m, v, 10, gol, O.lrOp, O);
    for (int k = 0; k < 3; k++) adam(g, m, v, 11 + k, graw[k], O.lrC0, O);
    for (int k = 0; k < 3; k++) { adam(g, m, v, 14 + k, graw[k] * dir.x, O.lrC1, O); adam(g, m, v, 17 + k, graw[k] * dir.y, O.lrC1, O); adam(g, m, v, 20 + k, graw[k] * dir.z, O.lrC1, O); }
    stats[i * 2] += length(float2(gm0, gm1)); stats[i * 2 + 1] += 1;
}
"""

/// one training photo: small RGBA copy, intrinsics at that size, camera -> world pose (ARKit axes)
struct SplatTrainView { var rgba: [UInt8]; var w: Int; var h: Int; var fx: Float; var fy: Float; var cx: Float; var cy: Float; var T: simd_float4x4 }

final class SplatTrainer {
    static let PS = 24
    struct ViewU { var R: simd_float3x3; var t: SIMD3<Float>; var K: SIMD4<Float>; var size: SIMD2<UInt32>; var tiles: SIMD2<UInt32>; var cam: SIMD3<Float>; var img: UInt32 }
    struct OptU { var lrPos: Float; var lrScale: Float; var lrRot: Float; var lrOp: Float; var lrC0: Float; var lrC1: Float; var b1t: Float; var b2t: Float; var maxLogScale: Float; var N: UInt32 }

    let dev: MTLDevice
    private let q: MTLCommandQueue
    private let pProject, pEmit, pRender, pStep: MTLComputePipelineState
    private(set) var N: Int
    private var G, Mo, Ve, g2, mean, conop, col, rect, stats: MTLBuffer
    private var images: MTLBuffer, imgOffset: [Int] = []
    private var tileCount, cursor, ranges, outImg, lossBuf: MTLBuffer
    private var entries: MTLBuffer
    let views: [SplatTrainView]
    let W: Int, H: Int, TX: Int, TY: Int
    var extent: Float
    private(set) var iteration = 0

    /// params: N x 24 floats (see the shader); every view must be the same size
    init?(params: [Float], views: [SplatTrainView], extent: Float) {
        guard let d = MTLCreateSystemDefaultDevice(), let cq = d.makeCommandQueue(), let v0 = views.first, !params.isEmpty else { return nil }
        dev = d; q = cq; self.views = views; self.extent = extent
        W = v0.w; H = v0.h; TX = (W + 15) / 16; TY = (H + 15) / 16
        N = params.count / SplatTrainer.PS
        do {
            let o = MTLCompileOptions(); o.languageVersion = .version3_0
            let lib = try d.makeLibrary(source: SPLAT_TRAIN_SHADER, options: o)
            pProject = try d.makeComputePipelineState(function: lib.makeFunction(name: "project")!)
            pEmit = try d.makeComputePipelineState(function: lib.makeFunction(name: "emit")!)
            pRender = try d.makeComputePipelineState(function: lib.makeFunction(name: "render")!)
            pStep = try d.makeComputePipelineState(function: lib.makeFunction(name: "step")!)
        } catch { print("splat train shader: \(error)"); return nil }
        func buf(_ n: Int) -> MTLBuffer? { d.makeBuffer(length: max(16, n), options: .storageModeShared) }
        let n = N, F = 4
        guard let g = params.withUnsafeBytes({ d.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let mo = buf(n * 24 * F), let ve = buf(n * 24 * F), let gr = buf(n * 10 * F), let me = buf(n * 16), let co = buf(n * 16), let cl = buf(n * 16),
              let re = buf(n * 16), let st = buf(n * 2 * F), let tc = buf(TX * TY * 4), let cu = buf(TX * TY * 4), let ra = buf(TX * TY * 8),
              let oi = buf(W * H * 16), let lb = buf(4), let en = buf(n * 4 * 8) else { return nil }
        G = g; Mo = mo; Ve = ve; g2 = gr; mean = me; conop = co; col = cl; rect = re; stats = st
        tileCount = tc; cursor = cu; ranges = ra; outImg = oi; lossBuf = lb; entries = en
        memset(mo.contents(), 0, mo.length); memset(ve.contents(), 0, ve.length); memset(gr.contents(), 0, gr.length); memset(st.contents(), 0, st.length)
        memset(tc.contents(), 0, tc.length)
        var all = [UInt8](); all.reserveCapacity(views.count * W * H * 4)
        for v in views { imgOffset.append(all.count / 4); all += v.rgba }
        guard let im = all.withUnsafeBytes({ d.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }) else { return nil }
        images = im
    }

    private func viewU(_ i: Int) -> ViewU {
        let v = views[i], Ti = v.T.inverse
        let R = simd_float3x3(SIMD3(Ti.columns.0.x, Ti.columns.0.y, Ti.columns.0.z), SIMD3(Ti.columns.1.x, Ti.columns.1.y, Ti.columns.1.z), SIMD3(Ti.columns.2.x, Ti.columns.2.y, Ti.columns.2.z))
        return ViewU(R: R, t: SIMD3(Ti.columns.3.x, Ti.columns.3.y, Ti.columns.3.z), K: SIMD4(v.fx, v.fy, v.cx, v.cy), size: SIMD2(UInt32(W), UInt32(H)),
                     tiles: SIMD2(UInt32(TX), UInt32(TY)), cam: SIMD3(v.T.columns.3.x, v.T.columns.3.y, v.T.columns.3.z), img: UInt32(imgOffset[i]))
    }
    private func groups(_ n: Int, _ p: MTLComputePipelineState) -> (MTLSize, MTLSize) {
        let t = p.threadExecutionWidth * 4
        return (MTLSize(width: (n + t - 1) / t, height: 1, depth: 1), MTLSize(width: t, height: 1, depth: 1))
    }

    /// renders view i; with `train`, also takes one optimisation step on it. Returns the mean absolute error (0-1)
    @discardableResult
    func pass(_ i: Int, train: Bool, lr: OptU? = nil) -> Float {
        var V = viewU(i), n32 = UInt32(N)
        // 1. project every splat, count how many land on each tile
        guard let c1 = q.makeCommandBuffer(), let e1 = c1.makeComputeCommandEncoder() else { return 1 }
        e1.setComputePipelineState(pProject)
        e1.setBuffer(G, offset: 0, index: 0); e1.setBytes(&V, length: MemoryLayout<ViewU>.stride, index: 1)
        e1.setBuffer(mean, offset: 0, index: 2); e1.setBuffer(conop, offset: 0, index: 3); e1.setBuffer(col, offset: 0, index: 4)
        e1.setBuffer(tileCount, offset: 0, index: 5); e1.setBuffer(rect, offset: 0, index: 6); e1.setBytes(&n32, length: 4, index: 7)
        let (g1, t1) = groups(N, pProject); e1.dispatchThreadgroups(g1, threadsPerThreadgroup: t1); e1.endEncoding()
        c1.commit(); c1.waitUntilCompleted()
        // 2. where each tile's list starts
        let nt = TX * TY
        let cnt = tileCount.contents().assumingMemoryBound(to: UInt32.self), cur = cursor.contents().assumingMemoryBound(to: UInt32.self)
        let rg = ranges.contents().assumingMemoryBound(to: SIMD2<UInt32>.self)
        var total = 0
        for t in 0..<nt { let c = Int(cnt[t]); rg[t] = SIMD2(UInt32(total), UInt32(total + c)); cur[t] = UInt32(total); total += c; cnt[t] = 0 }
        if total * 8 > entries.length { guard let e = dev.makeBuffer(length: total * 8 * 3 / 2, options: .storageModeShared) else { return 1 }; entries = e }
        // 3. every splat writes itself into its tiles' lists
        guard let c2 = q.makeCommandBuffer(), let e2 = c2.makeComputeCommandEncoder() else { return 1 }
        e2.setComputePipelineState(pEmit)
        e2.setBuffer(mean, offset: 0, index: 0); e2.setBuffer(rect, offset: 0, index: 1); e2.setBytes(&V, length: MemoryLayout<ViewU>.stride, index: 2)
        e2.setBuffer(cursor, offset: 0, index: 3); e2.setBuffer(entries, offset: 0, index: 4); e2.setBytes(&n32, length: 4, index: 5)
        let (g2s, t2) = groups(N, pEmit); e2.dispatchThreadgroups(g2s, threadsPerThreadgroup: t2); e2.endEncoding()
        c2.commit(); c2.waitUntilCompleted()
        // 4. each tile's list front to back (depth bits sort as integers)
        let en = entries.contents().assumingMemoryBound(to: SIMD2<UInt32>.self)
        DispatchQueue.concurrentPerform(iterations: nt) { t in
            let r = rg[t]; let c = Int(r.y - r.x); guard c > 1 else { return }
            var b = UnsafeMutableBufferPointer(start: en + Int(r.x), count: c); b.sort { $0.x < $1.x }
        }
        // 5. render (+ gradients), 6. step
        lossBuf.contents().assumingMemoryBound(to: Float.self).pointee = 0
        guard let c3 = q.makeCommandBuffer(), let e3 = c3.makeComputeCommandEncoder() else { return 1 }
        var bw: UInt32 = train ? 1 : 0
        e3.setComputePipelineState(pRender)
        e3.setBuffer(entries, offset: 0, index: 0); e3.setBuffer(ranges, offset: 0, index: 1); e3.setBuffer(mean, offset: 0, index: 2)
        e3.setBuffer(conop, offset: 0, index: 3); e3.setBuffer(col, offset: 0, index: 4); e3.setBytes(&V, length: MemoryLayout<ViewU>.stride, index: 5)
        e3.setBuffer(images, offset: 0, index: 6); e3.setBuffer(g2, offset: 0, index: 7); e3.setBuffer(outImg, offset: 0, index: 8)
        e3.setBuffer(lossBuf, offset: 0, index: 9); e3.setBytes(&bw, length: 4, index: 10)
        e3.dispatchThreadgroups(MTLSize(width: TX, height: TY, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        if train, var o = lr {
            iteration += 1
            o.b1t = 1 - pow(0.9, Float(iteration)); o.b2t = 1 - pow(0.999, Float(iteration)); o.N = n32
            e3.setComputePipelineState(pStep)
            e3.setBuffer(G, offset: 0, index: 0); e3.setBuffer(Mo, offset: 0, index: 1); e3.setBuffer(Ve, offset: 0, index: 2); e3.setBuffer(g2, offset: 0, index: 3)
            e3.setBuffer(mean, offset: 0, index: 4); e3.setBytes(&V, length: MemoryLayout<ViewU>.stride, index: 5); e3.setBytes(&o, length: MemoryLayout<OptU>.stride, index: 6)
            e3.setBuffer(stats, offset: 0, index: 7)
            let (g4, t4) = groups(N, pStep); e3.dispatchThreadgroups(g4, threadsPerThreadgroup: t4)
        }
        e3.endEncoding(); c3.commit(); c3.waitUntilCompleted()
        return lossBuf.contents().assumingMemoryBound(to: Float.self).pointee / Float(W * H * 3)
    }

    /// the last rendered image, RGB 0-1
    var lastImage: UnsafeBufferPointer<SIMD4<Float>> { UnsafeBufferPointer(start: outImg.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: W * H) }

    /// learning rates (position decays over the run, as in the original method)
    func rates(progress f: Float) -> OptU {
        let pos = 1.6e-4 * extent * pow(0.01, min(1, f))
        return OptU(lrPos: pos, lrScale: 5e-3, lrRot: 1e-3, lrOp: 5e-2, lrC0: 5e-3, lrC1: 1e-3, b1t: 1, b2t: 1, maxLogScale: log(max(0.01, extent * 0.05)), N: UInt32(N))
    }

    /// trains on `train` views until the time is up (or `stop` says so); progress gets (0-1, mean error)
    func train(views idx: [Int], seconds: Double, stop: () -> Bool = { false }, progress: (Float, Float) -> Void = { _, _ in }) {
        let t0 = Date(); var order: [Int] = [], k = 0, errSum: Float = 0, errN = 0
        while true {
            let f = Float(Date().timeIntervalSince(t0) / seconds)
            if f >= 1 || stop() { break }     // 5.7 (46): a closed sheet stops the run instead of training on for up to 150 s
            if k >= order.count { order = idx.shuffled(); k = 0 }
            errSum += pass(order[k], train: true, lr: rates(progress: f)); errN += 1; k += 1
            if errN == 25 { progress(f, errSum / 25); errSum = 0; errN = 0 }
        }
    }

    /// the trained splats in the viewer's layout (7 float4 each), faint ones left out
    func export() -> [SIMD4<Float>] {
        let g = G.contents().assumingMemoryBound(to: Float.self)
        var out: [SIMD4<Float>] = []; out.reserveCapacity(N * 7)
        for i in 0..<N {
            let p = g + i * SplatTrainer.PS
            let op = 1 / (1 + exp(-p[10])); if op < 0.02 { continue }
            var qv = SIMD4<Float>(p[6], p[7], p[8], p[9]); qv /= max(1e-8, simd_length(qv))
            out.append(SIMD4(p[0], p[1], p[2], op))
            out.append(SIMD4(exp(p[3]), exp(p[4]), exp(p[5]), 0))
            out.append(qv)
            out.append(SIMD4(p[11], p[12], p[13], 0)); out.append(SIMD4(p[14], p[15], p[16], 0)); out.append(SIMD4(p[17], p[18], p[19], 0)); out.append(SIMD4(p[20], p[21], p[22], 0))
        }
        return out
    }

    /// the starting splats from the viewer's layout (7 float4 each: pos+opacity, scale, rotation, colour, 3 terms)
    static func params(fromViewer b: [SIMD4<Float>]) -> [Float] {
        let n = b.count / 7
        var out = [Float](repeating: 0, count: n * PS)
        for i in 0..<n {
            let s = b[i * 7 + 1], r = b[i * 7 + 2], c = b[i * 7 + 3], o = min(0.99, max(0.01, b[i * 7].w))
            let base = i * PS
            out[base] = b[i * 7].x; out[base + 1] = b[i * 7].y; out[base + 2] = b[i * 7].z
            out[base + 3] = log(max(1e-4, s.x)); out[base + 4] = log(max(1e-4, s.y)); out[base + 5] = log(max(1e-4, s.z))
            out[base + 6] = r.x; out[base + 7] = r.y; out[base + 8] = r.z; out[base + 9] = r.w
            out[base + 10] = log(o / (1 - o))
            out[base + 11] = c.x; out[base + 12] = c.y; out[base + 13] = c.z
        }
        return out
    }

    /// the scan's photos as training views, every one `long` pixels wide (ARKit frames are all the same size)
    static func loadViews(frames dir: URL, long: Int = 480, max n: Int = 160) -> [SplatTrainView] {
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).filter { $0.hasSuffix(".json") }.sorted()
        let step = Swift.max(1, names.count / n)
        var out: [SplatTrainView] = []
        for name in stride(from: 0, to: names.count, by: step).map({ names[$0] }) {
            let base = dir.appendingPathComponent(String(name.dropLast(5)))
            guard let d = try? Data(contentsOf: dir.appendingPathComponent(name)), let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                  let t = j["transform"] as? [Double], t.count == 16, let k = j["intrinsics"] as? [Double], k.count == 9,
                  let src = CGImageSourceCreateWithURL(base.appendingPathExtension("jpg") as CFURL, nil),
                  let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: long] as CFDictionary) else { continue }
            let iw = Float(j["imageWidth"] as? Double ?? 1920)
            let w = long, h = Int((Float(long) * Float(img.height) / Float(img.width)).rounded())
            if let f = out.first, f.w != w || f.h != h { continue }
            var px = [UInt8](repeating: 0, count: w * h * 4)
            guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { continue }
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
            let f = t.map { Float($0) }, s = Float(w) / iw
            let T = simd_float4x4(columns: (SIMD4(f[0], f[1], f[2], f[3]), SIMD4(f[4], f[5], f[6], f[7]), SIMD4(f[8], f[9], f[10], f[11]), SIMD4(f[12], f[13], f[14], f[15])))
            out.append(SplatTrainView(rgba: px, w: w, h: h, fx: Float(k[0]) * s, fy: Float(k[4]) * s, cx: (Float(k[6]) + 0.5) * s - 0.5, cy: (Float(k[7]) + 0.5) * s - 0.5, T: T))
        }
        return out
    }
}
