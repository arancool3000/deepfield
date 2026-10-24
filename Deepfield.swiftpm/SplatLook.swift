import SceneKit
import Metal

/// 5.0: points drawn as real Gaussian splats on the GPU - soft round blobs of real-world size that fade at the rim and
/// blend into one continuous surface (the look of a Gaussian splat scan). The solid core writes depth so surfaces never
/// show through each other; the rim blends. Used live while scanning in Gaussian / Aran modes and as an editor look.
enum SplatLook {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;
    struct NodeBuf { float4x4 modelViewProjectionTransform; };
    struct SplatParams { float radius; float pxScale; };
    struct VIn { float3 position [[attribute(0)]]; float4 color [[attribute(3)]]; };
    struct VOut { float4 position [[position]]; float size [[point_size]]; float4 color; };
    vertex VOut splat_vertex(VIn in [[stage_in]], constant NodeBuf& scn_node [[buffer(1)]], constant SplatParams& splat [[buffer(2)]]) {
        VOut o;
        o.position = scn_node.modelViewProjectionTransform * float4(in.position, 1.0);
        o.size = clamp(splat.radius * splat.pxScale / max(o.position.w, 0.01), 1.5, 96.0);
        o.color = in.color;
        return o;
    }
    fragment half4 splat_fragment(VOut in [[stage_in]], float2 pc [[point_coord]]) {
        float2 d = pc * 2.0 - 1.0;
        float r2 = dot(d, d);
        float a = exp(-r2 * 3.0);
        if (r2 > 1.0 || a < 0.3) discard_fragment();
        return half4(half3(in.color.rgb) * a, a);
    }
    """
    private static let program: SCNProgram? = {
        guard let dev = MTLCreateSystemDefaultDevice(), let lib = try? dev.makeLibrary(source: source, options: nil) else { Log.add("splat look: no Metal library"); return nil }
        let p = SCNProgram(); p.library = lib; p.vertexFunctionName = "splat_vertex"; p.fragmentFunctionName = "splat_fragment"; p.isOpaque = false
        return p
    }()
    static var available: Bool { program != nil }
    /// a material drawing a point geometry as splats; pxScale = projection[1][1] x viewport height in pixels
    static func material(radius: Float, pxScale: Float) -> SCNMaterial? {
        guard let p = program else { return nil }
        let m = SCNMaterial(); m.program = p; m.blendMode = .alpha; m.writesToDepthBuffer = true
        set(m, radius: radius, pxScale: pxScale)
        return m
    }
    static func set(_ m: SCNMaterial, radius: Float, pxScale: Float) {
        var v: [Float] = [radius, pxScale]
        m.setValue(Data(bytes: &v, count: 8), forKey: "splat")
    }
    /// projection[1][1] x height in pixels of a view
    static func pxScale(_ v: SCNView) -> Float {
        let p = v.pointOfView?.camera?.projectionTransform.m22 ?? 1.73
        return Float(p) * Float(v.bounds.height * v.contentScaleFactor)
    }
}
