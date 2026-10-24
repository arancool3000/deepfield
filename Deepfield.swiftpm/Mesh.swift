import Foundation
import SceneKit
import SwiftUI
import simd

// ---------- small linear algebra ----------
/// normal of a point neighbourhood: eigenvector of the smallest eigenvalue of its covariance
func planeNormal(_ pts: [SIMD3<Float>]) -> (n: SIMD3<Float>, c: SIMD3<Float>, flat: Float)? {
    guard pts.count >= 3 else { return nil }
    var c = SIMD3<Double>.zero
    for p in pts { c += SIMD3<Double>(p) }
    c /= Double(pts.count)
    var a00 = 0.0, a01 = 0.0, a02 = 0.0, a11 = 0.0, a12 = 0.0, a22 = 0.0
    for p in pts { let d = SIMD3<Double>(p) - c; a00 += d.x * d.x; a01 += d.x * d.y; a02 += d.x * d.z; a11 += d.y * d.y; a12 += d.y * d.z; a22 += d.z * d.z }
    let lmin = minEigenSym3(a00, a01, a02, a11, a12, a22)
    let r0 = SIMD3(a00 - lmin, a01, a02), r1 = SIMD3(a01, a11 - lmin, a12), r2 = SIMD3(a02, a12, a22 - lmin)
    let c01 = simd_cross(r0, r1), c02 = simd_cross(r0, r2), c12 = simd_cross(r1, r2)
    var best = c01, bl = simd_length_squared(c01)
    if simd_length_squared(c02) > bl { best = c02; bl = simd_length_squared(c02) }
    if simd_length_squared(c12) > bl { best = c12; bl = simd_length_squared(c12) }
    guard bl > 1e-30 else { return nil }
    let tr = a00 + a11 + a22
    return (SIMD3<Float>(simd_normalize(best)), SIMD3<Float>(c), Float(lmin / max(1e-12, tr)))
}

// ---------- hole filling on ANY surface ----------
/// Empty cells next to the surface are filled when the surface around them is flat enough and surrounds them
/// (neighbours on every side when seen along the surface). Repeated passes close holes from the edge inwards.
enum HoleFill {
    struct Acc { var n: Int = 0; var p = SIMD3<Float>.zero; var c = SIMD3<Float>.zero; var ai = false }

    static func fill(_ cl: Cloud, cell s: Float, passes: Int, radius R: Int = 3, inside box: (SIMD3<Float>, SIMD3<Float>)? = nil, spacing: Float = 0.003) -> (Cloud, Cloud) {
        var occ: [SIMD3<Int32>: Acc] = [:]
        @inline(__always) func key(_ p: SIMD3<Float>) -> SIMD3<Int32> { SIMD3(ik((p.x / s)), ik((p.y / s)), ik((p.z / s))) }
        for i in 0..<cl.count {
            let k = key(cl.p[i]); var a = occ[k] ?? Acc()
            a.n += 1; a.p += cl.p[i]; a.c += SIMD3<Float>(Float(cl.c[i].x), Float(cl.c[i].y), Float(cl.c[i].z)); occ[k] = a
        }
        var added: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []      // point on the surface, normal, colour
        for _ in 0..<passes {
            var cand = Set<SIMD3<Int32>>()
            for k in occ.keys {
                for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 {
                    let nb = k &+ SIMD3(dx, dy, dz); if occ[nb] == nil { cand.insert(nb) }
                } } }
            }
            // 4.5: a quick test first (surface on both sides along two axes), then the full check on every core
            let list = Array(cand)
            var results = [(SIMD3<Int32>, Acc, SIMD3<Float>)?](repeating: nil, count: list.count)
            let occView = occ
            results.withUnsafeMutableBufferPointer { rb in
                let rp = rb.baseAddress!
                parallelFor(list.count) { li in
                    let c = list[li]
                    let ctr = (SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)) + 0.5) * s
                    if let b = box, !(ctr.x >= b.0.x && ctr.y >= b.0.y && ctr.z >= b.0.z && ctr.x <= b.1.x && ctr.y <= b.1.y && ctr.z <= b.1.z) { return }
                    let r = Int32(R)
                    var axesBoth = 0
                    for ax in [SIMD3<Int32>(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)] {
                        var plus = false, minus = false
                        for t in 1...r { if !plus, occView[c &+ ax &* t] != nil { plus = true }; if !minus, occView[c &- ax &* t] != nil { minus = true } }
                        if plus && minus { axesBoth += 1 }
                    }
                    guard axesBoth >= 2 else { return }
                    var nb: [SIMD3<Float>] = [], cols: [(SIMD3<Float>, Float)] = []
                    for dx in -r...r { for dy in -r...r { for dz in -r...r {
                        if let a = occView[c &+ SIMD3(dx, dy, dz)] { let q = a.p / Float(a.n); nb.append(q); cols.append((a.c / Float(a.n), 1 / max(1e-4, simd_length(q - ctr)))) }
                    } } }
                    guard nb.count >= 8, let (n, pc, flat) = planeNormal(nb), flat < 0.04 else { return }
                    let off = simd_dot(ctr - pc, n)
                    guard abs(off) < s * 0.8 else { return }
                    let e1 = simd_normalize(simd_cross(n, abs(n.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0))), e2 = simd_cross(n, e1)
                    var ang: [Float] = []
                    for q in nb { let d = q - ctr; let u = simd_dot(d, e1), v = simd_dot(d, e2); if u * u + v * v > s * s * 0.25 { ang.append(atan2(v, u)) } }
                    guard ang.count >= 6 else { return }
                    ang.sort()
                    var gap: Float = (ang[0] + 2 * .pi) - ang[ang.count - 1]
                    for j in 1..<ang.count { gap = max(gap, ang[j] - ang[j - 1]) }
                    guard gap < 110 * .pi / 180 else { return }
                    var cs = SIMD3<Float>.zero, ws: Float = 0
                    for (cc, w) in cols { cs += cc * w; ws += w }
                    var a = Acc(); a.n = 1; a.p = ctr - n * off; a.c = cs / ws; a.ai = true
                    rp[li] = (c, a, n)
                }
            }
            let newCells = results.compactMap { $0 }
            if newCells.isEmpty { break }
            for (k, a, n) in newCells { occ[k] = a; added.append((a.p, n, a.c)) }
        }
        // points: a small patch on the surface per filled cell
        var yellow = Cloud(), real = Cloud()
        let steps = max(2, min(8, Int((s / max(0.0014, spacing)).rounded())))
        for (p, n, c) in added {
            let e1 = simd_normalize(simd_cross(n, abs(n.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0))), e2 = simd_cross(n, e1)
            let col = SIMD3<UInt8>(u8(c.x), u8(c.y), u8(c.z))
            for a in 0..<steps { for b in 0..<steps {
                let u = (Float(a) + Float.random(in: 0.2...0.8)) / Float(steps) - 0.5, v = (Float(b) + Float.random(in: 0.2...0.8)) / Float(steps) - 0.5
                let q = p + e1 * u * s + e2 * v * s
                yellow.p.append(q); yellow.c.append(SIMD3(255, 214, 10)); real.p.append(q); real.c.append(col)
            } }
        }
        return (yellow, real)
    }
}

extension HoleFill {
    /// Fill everything missing inside a region the user chose: starting from the surface around the region and growing
    /// inwards pass by pass, every empty cell that lies on the local surface is filled - no "is it surrounded" test,
    /// because the user said this is where it is missing. Points are made at the scan's own spacing (dense).
    static func fillRegion(_ cl: Cloud, box: (SIMD3<Float>, SIMD3<Float>), inside: @escaping (SIMD3<Float>) -> Bool, spacing sp: Float) -> (Cloud, Cloud) {
        let s: Float = 0.01
        let lo = box.0 - 0.12, hi = box.1 + 0.12
        var occ: [SIMD3<Int32>: Acc] = [:]
        @inline(__always) func key(_ p: SIMD3<Float>) -> SIMD3<Int32> { SIMD3(ik((p.x / s)), ik((p.y / s)), ik((p.z / s))) }
        for i in 0..<cl.count {
            let q = cl.p[i]
            guard q.x >= lo.x, q.y >= lo.y, q.z >= lo.z, q.x <= hi.x, q.y <= hi.y, q.z <= hi.z else { continue }
            let k = key(q); var a = occ[k] ?? Acc()
            a.n += 1; a.p += q; a.c += SIMD3<Float>(Float(cl.c[i].x), Float(cl.c[i].y), Float(cl.c[i].z)); occ[k] = a
        }
        var added: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []
        let ext = simd_reduce_max(box.1 - box.0)
        let passes = max(6, min(60, Int(ext / s / 2) + 4))
        for _ in 0..<passes {
            var cand = Set<SIMD3<Int32>>()
            for k in occ.keys {
                for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 {
                    let nb = k &+ SIMD3(dx, dy, dz)
                    guard occ[nb] == nil else { continue }
                    let c = (SIMD3<Float>(Float(nb.x), Float(nb.y), Float(nb.z)) + 0.5) * s
                    if inside(c) { cand.insert(nb) }
                } } }
            }
            if cand.isEmpty { break }
            let list = Array(cand), occView = occ
            var res = [(SIMD3<Int32>, Acc, SIMD3<Float>)?](repeating: nil, count: list.count)
            res.withUnsafeMutableBufferPointer { rb in
                let rp = rb.baseAddress!
                parallelFor(list.count) { li in
                    let c = list[li], ctr = (SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)) + 0.5) * s
                    var nb: [SIMD3<Float>] = [], cols: [(SIMD3<Float>, Float)] = []
                    for dx: Int32 in -5...5 { for dy: Int32 in -5...5 { for dz: Int32 in -5...5 {
                        if let a = occView[c &+ SIMD3(dx, dy, dz)] { let q = a.p / Float(a.n); nb.append(q); cols.append((a.c / Float(a.n), 1 / max(1e-4, simd_length(q - ctr)))) }
                    } } }
                    guard nb.count >= 6, let (n, pc, flat) = planeNormal(nb), flat < 0.12 else { return }
                    let off = simd_dot(ctr - pc, n)
                    guard abs(off) < s * 0.9 else { return }
                    var cs = SIMD3<Float>.zero, ws: Float = 0
                    for (cc, w) in cols { cs += cc * w; ws += w }
                    var a = Acc(); a.n = 1; a.p = ctr - n * off; a.c = cs / ws; a.ai = true
                    rp[li] = (c, a, n)
                }
            }
            let got = res.compactMap { $0 }
            if got.isEmpty { break }
            for (k, a, n) in got { occ[k] = a; added.append((a.p, n, a.c)) }
        }
        var yellow = Cloud(), real = Cloud()
        let steps = max(2, min(7, Int((s / max(0.0014, sp)).rounded())))
        for (p, n, c) in added {
            let e1 = simd_normalize(simd_cross(n, abs(n.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0))), e2 = simd_cross(n, e1)
            let col = SIMD3<UInt8>(u8(c.x), u8(c.y), u8(c.z))
            for a in 0..<steps { for b in 0..<steps {
                let u = (Float(a) + Float.random(in: 0.2...0.8)) / Float(steps) - 0.5, v = (Float(b) + Float.random(in: 0.2...0.8)) / Float(steps) - 0.5
                let q = p + e1 * u * s + e2 * v * s
                guard inside(q) else { continue }
                yellow.p.append(q); yellow.c.append(SIMD3(255, 214, 10)); real.p.append(q); real.c.append(col)
            } }
        }
        return (yellow, real)
    }
}

// ---------- point cloud -> solid coloured mesh (surface nets over a signed distance field) ----------
struct MeshData {
    var v: [SIMD3<Float>] = []
    var c: [SIMD3<Float>] = []           // 0...1
    var n: [SIMD3<Float>] = []
    var f: [UInt32] = []                 // triangles
    var triangles: Int { f.count / 3 }

    func geometry() -> SCNGeometry {
        let vs = SCNGeometrySource(vertices: v.map { SCNVector3($0.x, $0.y, $0.z) })
        let ns = SCNGeometrySource(normals: n.map { SCNVector3($0.x, $0.y, $0.z) })
        var flat: [Float] = []; flat.reserveCapacity(c.count * 3); for k in c { flat += [linear(k.x), linear(k.y), linear(k.z)] }
        let cd = Data(bytes: flat, count: flat.count * 4)
        let cs = SCNGeometrySource(data: cd, semantic: .color, vectorCount: c.count, usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let el = SCNGeometryElement(data: Data(bytes: f, count: f.count * 4), primitiveType: .triangles, primitiveCount: f.count / 3, bytesPerIndex: 4)
        let g = SCNGeometry(sources: [vs, ns, cs], elements: [el])
        let m = SCNMaterial(); m.lightingModel = .physicallyBased; m.roughness.contents = 0.85; m.metalness.contents = 0.0; m.isDoubleSided = true
        g.materials = [m]
        return g
    }

    /// OBJ with vertex colours (Blender, MeshLab, most slicers read them)
    func obj() -> Data {
        var s = "# Deepfield mesh from LiDAR points\n"; s.reserveCapacity(v.count * 60 + f.count * 12)
        for i in 0..<v.count { s += String(format: "v %.5f %.5f %.5f %.4f %.4f %.4f\n", v[i].x, v[i].y, v[i].z, c[i].x, c[i].y, c[i].z) }
        for i in 0..<n.count { s += String(format: "vn %.4f %.4f %.4f\n", n[i].x, n[i].y, n[i].z) }
        var t = 0
        while t < f.count { let a = f[t] + 1, b = f[t + 1] + 1, d = f[t + 2] + 1; s += "f \(a)//\(a) \(b)//\(b) \(d)//\(d)\n"; t += 3 }
        return Data(s.utf8)
    }
}

enum Mesher {
    /// builds the mesh; cams = where the phone was (normals face them), progress 0...1
    static func build(_ cl: Cloud, cams: [SIMD3<Float>], cell hIn: Float? = nil, progress: @escaping (Float, String) -> Void) -> MeshData? {
        guard cl.count > 1000 else { return nil }
        let b = cl.bounds, ext = max(b.size.x, max(b.size.y, b.size.z))
        let h = hIn ?? max(0.005, min(0.025, ext / 380))
        progress(0.02, "Thinning to a \(Int(h * 1000)) mm grid...")
        var pts = CloudOps.dedupe(cl, grid: h * 0.8)
        if pts.count > 900_000 { pts = CloudOps.downsample(pts, to: 900_000) }
        let P = pts.p, n = P.count
        // normals (PCA of neighbours), facing the nearest camera position (or away from the middle)
        progress(0.1, "Working out surface directions...")
        let g = GridIndex(P, cell: h * 2)
        var N = [SIMD3<Float>](repeating: .zero, count: n)
        let ctr = b.centre
        let camIdx = cams.isEmpty ? nil : GridIndex(cams, cell: 0.5)
        N.withUnsafeMutableBufferPointer { nb in
            let np = nb.baseAddress!
            parallelFor(n) { i in
                var nbh: [SIMD3<Float>] = []; nbh.reserveCapacity(48)
                let r2 = (h * 2) * (h * 2)
                g.around(P[i]) { j in if simd_length_squared(P[Int(j)] - P[i]) <= r2 { nbh.append(P[Int(j)]) } }
                guard let (nv, _, _) = planeNormal(nbh) else { return }
                var out = nv
                var toward = P[i] - ctr                                   // default: outwards from the middle
                if !cams.isEmpty {
                    var best = cams[0], bd = Float.greatestFiniteMagnitude
                    if let ci = camIdx, let (k, _) = ci.nearest(P[i], in: cams, within: 1.5) { best = cams[Int(k)] }
                    else { for c in cams { let d = simd_length_squared(c - P[i]); if d < bd { bd = d; best = c } } }
                    toward = best - P[i]
                }
                if simd_dot(out, toward) < 0 { out = -out }
                np[i] = out
            }
        }
        // signed distance at grid corners near the points (Gaussian-weighted plane distances), with colour
        progress(0.35, "Building the surface...")
        struct V { var sd: Float = 0; var w: Float = 0; var c = SIMD3<Float>.zero }
        var field: [SIMD3<Int32>: V] = [:]; field.reserveCapacity(n * 4)
        let sig2 = (h * 1.5) * (h * 1.5)
        for i in 0..<n where N[i] != .zero {
            let p = P[i], nv = N[i]
            let col = SIMD3<Float>(Float(pts.c[i].x), Float(pts.c[i].y), Float(pts.c[i].z)) / 255
            let base = SIMD3<Int32>(ik((p.x / h)), ik((p.y / h)), ik((p.z / h)))
            for dx: Int32 in 0...1 { for dy: Int32 in 0...1 { for dz: Int32 in 0...1 {      // the 8 corners of its cell
                let k = base &+ SIMD3(dx, dy, dz)
                let vp = SIMD3<Float>(Float(k.x), Float(k.y), Float(k.z)) * h
                let d2 = simd_length_squared(vp - p)
                let w = exp(-d2 / sig2)
                guard w > 0.02 else { continue }
                var e = field[k] ?? V()
                e.sd += simd_dot(vp - p, nv) * w; e.w += w; e.c += col * w
                field[k] = e
            } } }
            if i % 200_000 == 0 { progress(0.35 + 0.35 * Float(i) / Float(n), "Building the surface...") }
        }
        // surface nets: one vertex per cell the surface crosses, quads across sign changes
        progress(0.72, "Making triangles...")
        var cellVert: [SIMD3<Int32>: UInt32] = [:]
        var mesh = MeshData()
        let corners: [SIMD3<Int32>] = [SIMD3(0,0,0), SIMD3(1,0,0), SIMD3(0,1,0), SIMD3(1,1,0), SIMD3(0,0,1), SIMD3(1,0,1), SIMD3(0,1,1), SIMD3(1,1,1)]
        let edges: [(Int, Int)] = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
        func sd(_ k: SIMD3<Int32>) -> (Float, SIMD3<Float>)? { guard let e = field[k], e.w > 0.15 else { return nil }; return (e.sd / e.w, e.c / e.w) }
        func vertex(_ cellK: SIMD3<Int32>) -> UInt32? {
            if let v = cellVert[cellK] { return v }
            var vals: [(Float, SIMD3<Float>)] = []; vals.reserveCapacity(8)
            for c in corners { guard let s = sd(cellK &+ c) else { return nil }; vals.append(s) }
            var sum = SIMD3<Float>.zero, cnt: Float = 0, col = SIMD3<Float>.zero
            for (a, bb) in edges {
                let va = vals[a].0, vb = vals[bb].0
                guard (va < 0) != (vb < 0) else { continue }
                let t = va / (va - vb)
                let pa = SIMD3<Float>(Float(corners[a].x), Float(corners[a].y), Float(corners[a].z)), pb = SIMD3<Float>(Float(corners[bb].x), Float(corners[bb].y), Float(corners[bb].z))
                sum += pa + (pb - pa) * t; cnt += 1
                col += vals[a].1 * (1 - t) + vals[bb].1 * t
            }
            guard cnt > 0 else { return nil }
            let local = sum / cnt
            let wp = (SIMD3<Float>(Float(cellK.x), Float(cellK.y), Float(cellK.z)) + local) * h
            let idx = UInt32(mesh.v.count)
            mesh.v.append(wp); mesh.c.append(col / cnt); mesh.n.append(.zero)
            cellVert[cellK] = idx
            return idx
        }
        let axes: [SIMD3<Int32>] = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
        var done = 0
        for (k, e) in field where e.w > 0.15 {
            let s0 = e.sd / e.w
            for (ai, ax) in axes.enumerated() {
                guard let (s1, _) = sd(k &+ ax), (s0 < 0) != (s1 < 0) else { continue }
                // the 4 cells around this edge
                let u = axes[(ai + 1) % 3], v = axes[(ai + 2) % 3]
                guard let a = vertex(k), let b2 = vertex(k &- u), let c2 = vertex(k &- u &- v), let d = vertex(k &- v) else { continue }
                if s0 < 0 { mesh.f += [a, b2, c2, a, c2, d] } else { mesh.f += [a, c2, b2, a, d, c2] }
            }
            done += 1
            if done % 300_000 == 0 { progress(0.72 + 0.2 * Float(done) / Float(max(1, field.count)), "Making triangles...") }
        }
        guard !mesh.f.isEmpty else { return nil }
        // smooth normals
        progress(0.94, "Finishing...")
        var t = 0
        while t < mesh.f.count {
            let a = Int(mesh.f[t]), b2 = Int(mesh.f[t + 1]), c2 = Int(mesh.f[t + 2])
            let fn = simd_cross(mesh.v[b2] - mesh.v[a], mesh.v[c2] - mesh.v[a])
            mesh.n[a] += fn; mesh.n[b2] += fn; mesh.n[c2] += fn
            t += 3
        }
        for i in 0..<mesh.n.count { let l = simd_length(mesh.n[i]); mesh.n[i] = l > 0 ? mesh.n[i] / l : SIMD3(0, 1, 0) }
        progress(1, "Done")
        return mesh
    }

    /// where the phone was during the scan (for facing the surfaces the right way)
    static func cameras(_ item: ScanItem) -> [SIMD3<Float>] {
        guard let cap = Library.captureURL(item) else { return [] }
        return Keyframe.all(in: cap).map { SIMD3($0.T.columns.3.x, $0.T.columns.3.y, $0.T.columns.3.z) }
    }
}

/// builds and shows the mesh, exports OBJ / USDZ, saves it to the library
struct MeshSheet: View {
    let item: ScanItem
    let cloud: Cloud
    @Environment(\.dismiss) private var dismiss
    @State private var progress: Float = 0
    @State private var status = "Ready"
    @State private var mesh: MeshData?
    @State private var tex: TexturedMesh?
    @State private var texGeo: SCNGeometry?
    @State private var texPhotos: [RealPhoto] = []
    @State private var geo: SCNGeometry?
    @State private var busy = false
    @State private var share: URL?
    @State private var detail = 1                 // 0 fast, 1 normal, 2 fine

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                if let g = texGeo { MeshView(geometry: g, photos: texPhotos, centre: cloud.bounds.centre).ignoresSafeArea(edges: .bottom) }
                else if let g = geo { MeshView(geometry: g).ignoresSafeArea(edges: .bottom) }
                else {
                    Spacer()
                    Image(systemName: "cube.fill").font(.system(size: 56)).foregroundStyle(.secondary)
                    Text(Library.captureURL(item) != nil ? "A solid, lifelike model: LiDAR depth from every photo gives the shape, and each part is painted from the photo that sees it best. On the phone." : "A solid coloured mesh made from your LiDAR points, on the phone.")
                        .multilineTextAlignment(.center).font(.subheadline).padding(.horizontal)
                    Picker("Detail", selection: $detail) { Text("Fast").tag(0); Text("Normal").tag(1); Text("Fine").tag(2) }.pickerStyle(.segmented).padding(.horizontal, 40)
                    if busy { ProgressView(value: progress).frame(width: 240); Text(status).font(.caption).foregroundStyle(.secondary) }
                    else { Button("Make 3D model") { make() }.buttonStyle(.borderedProminent).controlSize(.large) }
                    Spacer()
                }
            }
            .navigationTitle("3D model from points").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button { dismiss() } label: { Label("Close", systemImage: "xmark") }.labelStyle(.titleAndIcon) }
                if let t = tex {
                    ToolbarItem(placement: .confirmationAction) {
                        Menu {
                            Button("Save to library") { if let u = usdzTex(t) , Library.shared.addModel(from: u, name: item.name + " model") != nil { status = "Saved to the library" } else { status = "Could not save" } }
                            Button("Share USDZ") { share = usdzTex(t) }
                            Button("Share OBJ + texture") { shareOBJ(t) }
                        } label: { Image(systemName: "square.and.arrow.up") }
                    }
                }
                if let m = mesh {
                    ToolbarItem(placement: .confirmationAction) {
                        Menu {
                            Button("Export OBJ (with colours)") { export(m, "obj") }
                            Button("Export USDZ") { export(m, "usdz") }
                            Button("Export GLB (glTF)") { share = Export.tmp("\(item.name).glb"); try? Export.glb(m).write(to: share!) }
                            Button("Export STL (3D printing)") { share = Export.tmp("\(item.name).stl"); try? Export.stl(m).write(to: share!) }
                            Button("Save to library") { saveModel(m) }
                        } label: { Image(systemName: "square.and.arrow.up") }
                    }
                }
            }
            .sheet(item: $share) { ShareSheet(items: [$0]) }
        }
    }

    private func usdzTex(_ t: TexturedMesh) -> URL? {
        let c = cloud.bounds.centre
        let scene = SCNScene(); scene.rootNode.addChildNode(SCNNode(geometry: t.geometry(centre: c)))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(item.name.replacingOccurrences(of: "/", with: "-")).usdz")
        return scene.write(to: url, options: nil, delegate: nil, progressHandler: nil) ? url : nil
    }
    private func shareOBJ(_ t: TexturedMesh) {
        let name = item.name.replacingOccurrences(of: "/", with: "-")
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("obj-" + UUID().uuidString)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? Texturer.writeOBJ(t, to: dir, name: name)
        share = dir.appendingPathComponent("\(name).obj")
    }
    private func make() {
        busy = true
        let cl = cloud, it = item, d = detail
        if Library.captureURL(it) != nil {
            let ext = max(cl.bounds.size.x, max(cl.bounds.size.y, cl.bounds.size.z)), centre = cl.bounds.centre
            Task.detached(priority: .userInitiated) {
                let t = PhotoModel.build(it, extent: ext, detail: d) { p, s in Task { @MainActor in progress = p; status = s } }
                let g = t?.geometry(centre: centre), rp = t == nil ? [] : RealPhoto.load(it)
                await MainActor.run {
                    if let t = t, let g = g { busy = false; tex = t; texPhotos = rp; texGeo = g; status = "\(t.f.count / 3) triangles, photo-textured"; Log.add("photo model \(t.f.count / 3) triangles"); return }
                    status = "Not enough photo depth - building from the points instead..."
                    makeFromPoints()
                }
            }
            return
        }
        makeFromPoints()
    }
    private func makeFromPoints() {
        busy = true
        let cl = cloud, it = item, d = detail
        Task.detached(priority: .userInitiated) {
            let b = cl.bounds, ext = max(b.size.x, max(b.size.y, b.size.z))
            let h = max(0.004, min(0.03, ext / [220, 380, 600][d]))
            let cams = Mesher.cameras(it)
            let m = Mesher.build(cl, cams: cams, cell: h) { p, s in Task { @MainActor in progress = p; status = s } }
            let g = m?.geometry()
            await MainActor.run {
                busy = false; mesh = m; geo = g
                status = m.map { "\($0.triangles.formatted()) triangles" } ?? "Could not make a surface from these points."
                Log.add("mesh \(m?.triangles ?? 0) triangles")
            }
        }
    }
    private func usdz(_ m: MeshData) -> URL? {
        let scene = SCNScene(); scene.rootNode.addChildNode(SCNNode(geometry: m.geometry()))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(item.name.replacingOccurrences(of: "/", with: "-")).usdz")
        return scene.write(to: url, options: nil, delegate: nil, progressHandler: nil) ? url : nil
    }
    private func export(_ m: MeshData, _ kind: String) {
        if kind == "usdz" { share = usdz(m); return }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(item.name.replacingOccurrences(of: "/", with: "-")).obj")
        try? m.obj().write(to: url); share = url
    }
    private func saveModel(_ m: MeshData) {
        if let u = usdz(m), Library.shared.addModel(from: u, name: item.name + " mesh") != nil { status = "Saved to the library" }
    }
}

struct MeshView: UIViewRepresentable {
    let geometry: SCNGeometry
    var photos: [RealPhoto] = []                 // 5.0 photoreal
    var centre = SIMD3<Float>.zero               // the geometry was centred on this (photo world)
    final class Coord { var projector: PhotoProjector? }
    func makeCoordinator() -> Coord { Coord() }
    func makeUIView(context: Context) -> SCNView {
        let v = SCNView(frame: .zero); v.backgroundColor = .black; v.allowsCameraControl = true; v.antialiasingMode = .multisampling4X
        v.autoenablesDefaultLighting = true
        let scene = SCNScene(); let node = SCNNode(geometry: geometry)
        let (lo, hi) = geometry.boundingBox
        node.position = SCNVector3(-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -(lo.z + hi.z) / 2)
        scene.rootNode.addChildNode(node)
        let amb = SCNNode(); amb.light = SCNLight(); amb.light?.type = .ambient; amb.light?.intensity = 700; scene.rootNode.addChildNode(amb)
        v.scene = scene
        if !photos.isEmpty, let mat = geometry.materials.first {
            // scene = model + node offset, model = photo world - centre
            var T = matrix_identity_float4x4; T.columns.3 = SIMD4(centre - SIMD3(Float(node.position.x), Float(node.position.y), Float(node.position.z)), 1)
            let p = PhotoProjector(photos: photos, material: mat, worldFromScene: T); context.coordinator.projector = p
            v.delegate = p; v.rendersContinuously = true; v.preferredFramesPerSecond = 30
        }
        return v
    }
    func updateUIView(_ v: SCNView, context: Context) {}
}
