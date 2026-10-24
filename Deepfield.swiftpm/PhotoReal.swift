import SwiftUI
import SceneKit
import Metal
import UIKit

/// 5.0 photoreal viewing (unstructured lumigraph rendering): while you move round a model, the photos taken nearest to
/// your view are projected onto the 3D shape on the GPU and blended by how well each matches where you are. A photo
/// only paints what it really saw (its own LiDAR depth must agree), and the seamless baked texture shows wherever no
/// photo applies. At any spot you photographed you see your photo; between them it blends.
let PHOTOREAL_SHADER = """
#pragma arguments
float4x4 aWorldToCam;
float4x4 bWorldToCam;
float4 aK;
float4 bK;
float4 blend;          // x: weight of a, y: weight of b, z: depth tolerance (fraction), w: on (1) / off (0)
texture2d<float> aTex;
texture2d<float> bTex;
texture2d<float> aDepth;
texture2d<float> bDepth;
#pragma body
if (blend.w > 0.5) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    constexpr sampler sd(filter::nearest, address::clamp_to_edge);
    float4 world = scn_frame.inverseViewTransform * float4(_surface.position, 1.0);
    float3 sum = float3(0.0); float wsum = 0.0;
    float4 ca = aWorldToCam * world; float za = -ca.z;
    if (za > 0.05 && blend.x > 0.001) {
        float2 uv = float2(aK.z + aK.x * ca.x / za, aK.w - aK.y * ca.y / za);
        if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) {
            float md = aDepth.sample(sd, uv).r;
            // only a surface hidden BEHIND what the photo saw is skipped (thin parts in front stay)
            if (md <= 0.05 || za < md + max(0.03, blend.z * za)) { sum += aTex.sample(s, uv).rgb * blend.x; wsum += blend.x; }
        }
    }
    float4 cb = bWorldToCam * world; float zb = -cb.z;
    if (zb > 0.05 && blend.y > 0.001) {
        float2 uv = float2(bK.z + bK.x * cb.x / zb, bK.w - bK.y * cb.y / zb);
        if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) {
            float md = bDepth.sample(sd, uv).r;
            // only a surface hidden BEHIND what the photo saw is skipped (thin parts in front stay)
            if (md <= 0.05 || zb < md + max(0.03, blend.z * zb)) { sum += bTex.sample(s, uv).rgb * blend.y; wsum += blend.y; }
        }
    }
    if (wsum > 0.0) { _output.color = float4(sum / wsum, _output.color.a); }
}
"""

/// one captured photo for photoreal viewing
struct RealPhoto {
    let jpg: URL
    let T: simd_float4x4
    let Ti: simd_float4x4
    let k: SIMD4<Float>                 // fx/w, fy/h, cx/w, cy/h
    let depth: [Float]; let dw: Int; let dh: Int
    var pos: SIMD3<Float> { SIMD3(T.columns.3.x, T.columns.3.y, T.columns.3.z) }
    var fwd: SIMD3<Float> { -SIMD3(T.columns.2.x, T.columns.2.y, T.columns.2.z) }

    static func load(_ item: ScanItem, max n: Int = 160) -> [RealPhoto] {
        guard let cap = Library.captureURL(item) else { return [] }
        return ModelBuilder.pick(Keyframe.all(in: cap), n).compactMap { k in
            guard k.dw > 0, FileManager.default.fileExists(atPath: k.jpg.path),
                  let d = try? Data(contentsOf: k.jpg.deletingPathExtension().appendingPathExtension("depth")), d.count >= k.dw * k.dh * 4 else { return nil }
            var a = [Float](repeating: 0, count: k.dw * k.dh)
            d.withUnsafeBytes { b in for i in 0..<a.count { a[i] = b.loadUnaligned(fromByteOffset: i * 4, as: Float.self) } }
            return RealPhoto(jpg: k.jpg, T: k.T, Ti: k.Ti, k: SIMD4(k.K[0][0] / k.w, k.K[1][1] / k.h, k.K[2][0] / k.w, k.K[2][1] / k.h), depth: a, dw: k.dw, dh: k.dh)
        }
    }
}

/// drives the shader: picks the two photos that best match the current view each frame and keeps their pictures loaded
final class PhotoProjector: NSObject, SCNSceneRendererDelegate {
    let photos: [RealPhoto]
    let material: SCNMaterial
    let worldFromModel: simd_float4x4          // the model is drawn centred: model -> world
    private let device = MTLCreateSystemDefaultDevice()
    private var images: [Int: UIImage] = [:], depths: [Int: MTLTexture] = [:], loading = Set<Int>()
    private var shown: (Int, Int) = (-1, -1)
    // 5.7 (46): images/depths/shown are written on main (load, evict) and read on SceneKit's render thread; guard them
    private let lock = NSLock()
    private var weights = SIMD2<Float>(0, 0)
    private let q = DispatchQueue(label: "photoreal.load", qos: .userInitiated)
    var enabled = true

    /// centre: the model is drawn shifted by -centre (scene = world - centre)
    convenience init(photos: [RealPhoto], material: SCNMaterial, centre: SIMD3<Float>) {
        var T = matrix_identity_float4x4; T.columns.3 = SIMD4(centre, 1)
        self.init(photos: photos, material: material, worldFromScene: T)
    }
    /// worldFromScene: SceneKit's world -> the photos' (ARKit) world
    init(photos: [RealPhoto], material: SCNMaterial, worldFromScene: simd_float4x4) {
        self.photos = photos; self.material = material; worldFromModel = worldFromScene
        super.init()
        var mods = material.shaderModifiers ?? [:]
        mods[.fragment] = PHOTOREAL_SHADER
        material.shaderModifiers = mods
        material.setValue(NSValue(scnVector4: SCNVector4(0, 0, 0.04, 0)), forKey: "blend")
    }
    private func depthTexture(_ p: RealPhoto) -> MTLTexture? {
        guard let dev = device else { return nil }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: p.dw, height: p.dh, mipmapped: false)
        guard let t = dev.makeTexture(descriptor: d) else { return nil }
        p.depth.withUnsafeBytes { t.replace(region: MTLRegionMake2D(0, 0, p.dw, p.dh), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: p.dw * 4) }
        return t
    }
    private func ensure(_ i: Int) {
        guard i >= 0, images[i] == nil, !loading.contains(i) else { return }
        loading.insert(i)
        let p = photos[i]
        q.async {
            let img = UIImage(contentsOfFile: p.jpg.path)
            let dt = self.depthTexture(p)
            DispatchQueue.main.async {
                self.loading.remove(i)
                self.lock.lock(); defer { self.lock.unlock() }
                if let img { self.images[i] = img }
                if let dt { self.depths[i] = dt }
                // 5.7 (48): never evict the photo just loaded or the pair asked for - ensure is not called again for them
                let keep = self.asked
                if self.images.count > 8 { for k in self.images.keys where k != i && k != keep.0 && k != keep.1 && k != self.shown.0 && k != self.shown.1 && self.images.count > 8 { self.images[k] = nil; self.depths[k] = nil } }
            }
        }
    }
    /// how well a photo matches a view (position and direction)
    private func score(_ p: RealPhoto, _ pos: SIMD3<Float>, _ fwd: SIMD3<Float>) -> Float {
        let ang = acos(max(-1, min(1, simd_dot(p.fwd, fwd))))
        return simd_length(p.pos - pos) + ang * 0.6                          // 0.6 m per radian
    }
    private var asked = (-2, -2)
    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        guard enabled, !photos.isEmpty, let pov = renderer.pointOfView else { return }
        let camW = worldFromModel * pov.simdWorldTransform
        let pos = SIMD3<Float>(camW.columns.3.x, camW.columns.3.y, camW.columns.3.z), fwd = -SIMD3<Float>(camW.columns.2.x, camW.columns.2.y, camW.columns.2.z)
        var best = (Float.greatestFiniteMagnitude, -1), second = (Float.greatestFiniteMagnitude, -1)
        for (i, p) in photos.enumerated() {
            let s = score(p, pos, fwd)
            if s < best.0 { second = best; best = (s, i) } else if s < second.0 { second = (s, i) }
        }
        // 5.7 (48): asked is written under the lock - ensure (main) reads it to keep that pair from eviction
        // ask for the two photos only when they change (this ran twice a frame, 120 main-thread jobs a second)
        if asked.0 != best.1 || asked.1 != second.1 { lock.lock(); asked = (best.1, second.1); lock.unlock(); let a = best.1, b = second.1; DispatchQueue.main.async { self.ensure(a); self.ensure(b) } }
        // weights: the closer photo dominates; far from every photo the baked texture takes over
        var wa = 1 / max(0.02, best.0), wb = second.1 >= 0 ? 1 / max(0.02, second.0) : 0
        let sum = wa + wb; wa /= sum; wb /= sum
        let fade = max(0, min(1, 1.6 - best.0 * 2))                          // within ~0.3 m / 30 degrees of a photo: full
        lock.lock(); defer { lock.unlock() }
        let ready = (images[best.1] != nil && depths[best.1] != nil, second.1 >= 0 && images[second.1] != nil && depths[second.1] != nil)
        if !ready.0 && !ready.1 { return }
        let mfw = worldFromModel                                              // world -> photo camera, for model-space points
        func put(_ slot: String, _ i: Int) {
            let p = photos[i]
            material.setValue(NSValue(scnMatrix4: SCNMatrix4(p.Ti * mfw)), forKey: slot + "WorldToCam")
            material.setValue(NSValue(scnVector4: SCNVector4(p.k.x, p.k.y, p.k.z, p.k.w)), forKey: slot + "K")
            if let img = images[i] { material.setValue(SCNMaterialProperty(contents: img), forKey: slot + "Tex") }
            if let d = depths[i] { material.setValue(SCNMaterialProperty(contents: d), forKey: slot + "Depth") }
        }
        if ready.0 && shown.0 != best.1 { put("a", best.1); shown.0 = best.1 }
        if ready.1 && shown.1 != second.1 { put("b", second.1); shown.1 = second.1 }
        weights = SIMD2(ready.0 ? wa * fade : 0, ready.1 ? wb * fade : 0)
        material.setValue(NSValue(scnVector4: SCNVector4(weights.x, weights.y, 0.04, 1)), forKey: "blend")
    }
}
