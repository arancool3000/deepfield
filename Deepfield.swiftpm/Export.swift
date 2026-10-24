import Foundation
import UIKit
import SceneKit
import AVFoundation
import RoomPlan
import simd

/// Every export format: point clouds (PLY, XYZ, LAS), meshes (OBJ, STL, GLB, USDZ), floor plans (PNG, PDF, DXF), video.
enum Export {
    static func tmp(_ name: String) -> URL { FileManager.default.temporaryDirectory.appendingPathComponent(name.replacingOccurrences(of: "/", with: "-")) }

    // ---- point clouds ----
    // 5.7 (48): XYZ and LAS are made in 1M-point pieces like Cloud.write. xyz(_:to:) / las(_:to:) stream to a file and never hold
    // the whole export; xyz(_:) / las(_:) still return Data but no longer keep a full String or Double copy alive beside it.
    /// XYZ text: "x y z r g b" per line (CloudCompare, MeshLab, most CAD tools)
    static func xyz(_ cl: Cloud) -> Data { var d = Data(); d.reserveCapacity(cl.count * 40); xyzChunks(cl) { d.append($0) }; return d }
    static func xyz(_ cl: Cloud, to url: URL) throws { try stream(to: url) { try xyzChunks(cl, $0) } }
    private static func xyzChunks(_ cl: Cloud, _ sink: (Data) throws -> Void) rethrows {
        var i = 0
        while i < cl.count {
            let e = min(cl.count, i + 1_000_000)
            var s = ""; s.reserveCapacity((e - i) * 40)
            for j in i..<e { let p = cl.p[j], c = cl.c[j]; s += String(format: "%.4f %.4f %.4f %d %d %d\n", p.x, p.y, p.z, c.x, c.y, c.z) }
            try sink(Data(s.utf8)); i = e
        }
    }
    /// LAS 1.2, point format 2 (with colour), z up, 0.1 mm units - the survey / BIM standard
    static func las(_ cl: Cloud) -> Data { var d = Data(); d.reserveCapacity(227 + cl.count * 26); lasChunks(cl) { d.append($0) }; return d }
    static func las(_ cl: Cloud, to url: URL) throws { try stream(to: url) { try lasChunks(cl, $0) } }
    private static func lasChunks(_ cl: Cloud, _ sink: (Data) throws -> Void) rethrows {
        var d = Data(); d.reserveCapacity(227)
        func u8(_ v: UInt8) { d.append(v) }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        func i32(_ v: Int32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        func f64(_ v: Double) { var x = v; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        func text(_ s: String, _ n: Int) { var b = Array(s.utf8.prefix(n)); b += [UInt8](repeating: 0, count: n - b.count); d.append(contentsOf: b) }
        // ARKit is y up; LAS is z up: X = x, Y = -z, Z = y (converted per point, not into a 32 B/point copy)
        func las(_ q: SIMD3<Float>) -> SIMD3<Double> { SIMD3<Double>(Double(q.x), Double(-q.z), Double(q.y)) }
        let n = cl.count
        var lo = SIMD3<Double>(repeating: .greatestFiniteMagnitude), hi = -lo
        for q in cl.p { let p = las(q); lo = simd_min(lo, p); hi = simd_max(hi, p) }
        if n == 0 { lo = .zero; hi = .zero }
        let scale = 0.0001, off = lo
        let cal = Calendar(identifier: .gregorian), now = Date()
        text("LASF", 4); u16(0); u16(0); u32(0); u16(0); u16(0); text("", 8)
        u8(1); u8(2); text("Deepfield", 32); text("Deepfield 5.5", 32)
        u16(UInt16(cal.ordinality(of: .day, in: .year, for: now) ?? 1)); u16(UInt16(cal.component(.year, from: now)))
        u16(227); u32(227); u32(0); u8(2); u16(26); u32(UInt32(n))
        u32(UInt32(n)); u32(0); u32(0); u32(0); u32(0)
        f64(scale); f64(scale); f64(scale); f64(off.x); f64(off.y); f64(off.z)
        f64(hi.x); f64(lo.x); f64(hi.y); f64(lo.y); f64(hi.z); f64(lo.z)
        try sink(d); d = Data(); d.reserveCapacity(min(n, 1_000_000) * 26)
        for i in 0..<n {
            let q = (las(cl.p[i]) - off) / scale
            i32(Int32(clamping: Int(q.x.rounded()))); i32(Int32(clamping: Int(q.y.rounded()))); i32(Int32(clamping: Int(q.z.rounded())))
            u16(0); u8(0x09); u8(1); u8(0); u8(0); u16(0)          // intensity, return 1 of 1, class 1 (unclassified)
            let c = cl.c[i]; u16(UInt16(c.x) * 257); u16(UInt16(c.y) * 257); u16(UInt16(c.z) * 257)
            if (i + 1) % 1_000_000 == 0 { try sink(d); d.removeAll(keepingCapacity: true) }
        }
        if !d.isEmpty { try sink(d) }
    }
    /// writes the pieces a chunk generator hands over straight to a file
    private static func stream(to url: URL, _ body: ((Data) throws -> Void) throws -> Void) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let h = try FileHandle(forWritingTo: url)
        defer { try? h.close() }
        try body { try h.write(contentsOf: $0) }
    }

    // ---- meshes ----
    static func stl(_ m: MeshData) -> Data {
        var d = Data(count: 80); var n = UInt32(m.triangles); withUnsafeBytes(of: &n) { d.append(contentsOf: $0) }
        d.reserveCapacity(84 + m.triangles * 50)
        func f(_ v: SIMD3<Float>) { var x = v.x, y = v.y, z = v.z; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) }; withUnsafeBytes(of: &y) { d.append(contentsOf: $0) }; withUnsafeBytes(of: &z) { d.append(contentsOf: $0) } }
        var t = 0
        while t < m.f.count {
            let a = m.v[Int(m.f[t])], b = m.v[Int(m.f[t + 1])], c = m.v[Int(m.f[t + 2])]
            let nn = simd_cross(b - a, c - a); f(simd_length(nn) > 0 ? simd_normalize(nn) : .zero); f(a); f(b); f(c); d.append(contentsOf: [0, 0])
            t += 3
        }
        return d
    }
    /// binary glTF with vertex colours (Blender, three.js, Sketchfab, Android, the web)
    static func glb(_ m: MeshData) -> Data {
        func bytes<T>(_ a: [T]) -> Data { a.withUnsafeBufferPointer { Data(buffer: $0) } }
        let pos = bytes(m.v.flatMap { [$0.x, $0.y, $0.z] }), nor = bytes(m.n.flatMap { [$0.x, $0.y, $0.z] }), col = bytes(m.c.flatMap { [linear($0.x), linear($0.y), linear($0.z)] }), idx = bytes(m.f)
        var bin = Data(); let views = [pos, nor, col, idx]; var offsets: [Int] = []
        for v in views { offsets.append(bin.count); bin.append(v); while bin.count % 4 != 0 { bin.append(0) } }
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo; for v in m.v { lo = simd_min(lo, v); hi = simd_max(hi, v) }
        let json: [String: Any] = [
            "asset": ["version": "2.0", "generator": "Deepfield"], "scene": 0, "scenes": [["nodes": [0]]], "nodes": [["mesh": 0]],
            "meshes": [["primitives": [["attributes": ["POSITION": 0, "NORMAL": 1, "COLOR_0": 2], "indices": 3, "material": 0]]]],
            "materials": [["pbrMetallicRoughness": ["metallicFactor": 0, "roughnessFactor": 0.9], "doubleSided": true]],
            "buffers": [["byteLength": bin.count]],
            "bufferViews": (0..<4).map { i -> [String: Any] in ["buffer": 0, "byteOffset": offsets[i], "byteLength": views[i].count, "target": i == 3 ? 34963 : 34962] },
            "accessors": [
                ["bufferView": 0, "componentType": 5126, "count": m.v.count, "type": "VEC3", "min": [lo.x, lo.y, lo.z], "max": [hi.x, hi.y, hi.z]],
                ["bufferView": 1, "componentType": 5126, "count": m.n.count, "type": "VEC3"],
                ["bufferView": 2, "componentType": 5126, "count": m.c.count, "type": "VEC3"],
                ["bufferView": 3, "componentType": 5125, "count": m.f.count, "type": "SCALAR"]]
        ]
        var js = (try? JSONSerialization.data(withJSONObject: json)) ?? Data("{}".utf8)
        while js.count % 4 != 0 { js.append(0x20) }
        var out = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { out.append(contentsOf: $0) } }
        u32(0x46546C67); u32(2); u32(UInt32(12 + 8 + js.count + 8 + bin.count))
        u32(UInt32(js.count)); u32(0x4E4F534A); out.append(js)
        u32(UInt32(bin.count)); u32(0x004E4942); out.append(bin)
        return out
    }

    // ---- floor plans ----
    static func pdf(_ img: UIImage, title: String) -> Data {
        let page = CGRect(x: 0, y: 0, width: 595, height: 842)               // A4
        return UIGraphicsPDFRenderer(bounds: page).pdfData { ctx in
            ctx.beginPage()
            (title as NSString).draw(at: CGPoint(x: 40, y: 36), withAttributes: [.font: UIFont.systemFont(ofSize: 20, weight: .bold)])
            ("Made with Deepfield - \(Date().formatted(date: .abbreviated, time: .omitted))" as NSString).draw(at: CGPoint(x: 40, y: 64), withAttributes: [.font: UIFont.systemFont(ofSize: 11), .foregroundColor: UIColor.gray])
            let s = min(515 / img.size.width, 720 / img.size.height)
            img.draw(in: CGRect(x: 40, y: 92, width: img.size.width * s, height: img.size.height * s))
        }
    }
    /// DXF (AutoCAD, SketchUp, any CAD): walls, doors, windows and furniture as lines on their own layers, metres
    static func dxf(_ room: CapturedRoom) -> Data {
        var s = "0\nSECTION\n2\nENTITIES\n"
        func line(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ layer: String) {
            s += "0\nLINE\n8\n\(layer)\n10\n\(a.x)\n20\n\(-a.y)\n30\n0\n11\n\(b.x)\n21\n\(-b.y)\n31\n0\n"
        }
        for w in room.walls { let g = PlanDraw.seg(w.transform, w.dimensions.x); line(g.a, g.b, "WALLS") }
        for d in room.doors { let g = PlanDraw.seg(d.transform, d.dimensions.x); line(g.a, g.b, "DOORS") }
        for w in room.windows { let g = PlanDraw.seg(w.transform, w.dimensions.x); line(g.a, g.b, "WINDOWS") }
        for o in room.openings { let g = PlanDraw.seg(o.transform, o.dimensions.x); line(g.a, g.b, "OPENINGS") }
        for o in room.objects {
            let c = SIMD2<Float>(o.transform.columns.3.x, o.transform.columns.3.z), ax = simd_normalize(SIMD2<Float>(o.transform.columns.0.x, o.transform.columns.0.z)), az = SIMD2<Float>(-ax.y, ax.x)
            let hx = ax * o.dimensions.x / 2, hz = az * o.dimensions.z / 2
            let q = [c - hx - hz, c + hx - hz, c + hx + hz, c - hx + hz]
            for i in 0..<4 { line(q[i], q[(i + 1) % 4], "FURNITURE") }
        }
        s += "0\nENDSEC\n0\nEOF\n"
        return Data(s.utf8)
    }

    // ---- turntable video: the scan turning slowly, as an MP4 ----
    static func turntable(_ cl: Cloud, seconds: Double = 8, size: CGSize = CGSize(width: 1080, height: 1080), solid: Bool, progress: @escaping (Float) -> Void) -> URL? {
        guard cl.count > 0, let dev = MTLCreateSystemDefaultDevice(), let (geo, _) = CloudOps.downsample(cl, to: 1_200_000).geometry(maxPoints: 1_200_000, mode: "photo", size: 2) else { return nil }
        if solid, let e = geo.elements.first { e.pointSize = 0.006; e.minimumPointScreenSpaceRadius = 1; e.maximumPointScreenSpaceRadius = 40 }
        let scene = SCNScene(); scene.background.contents = UIColor.black
        scene.rootNode.addChildNode(SCNNode(geometry: geo))
        let b = cl.bounds, r = max(0.4, simd_length(b.size) * 0.9)
        let cam = SCNNode(); cam.camera = SCNCamera(); cam.camera?.zNear = 0.01; cam.camera?.fieldOfView = 45; scene.rootNode.addChildNode(cam)
        let ren = SCNRenderer(device: dev, options: nil); ren.scene = scene; ren.pointOfView = cam
        let url = tmp("Deepfield turntable.mp4"); try? FileManager.default.removeItem(at: url)
        guard let w = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return nil }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: Int(size.width), AVVideoHeightKey: Int(size.height)])
        let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: Int(size.width), kCVPixelBufferHeightKey as String: Int(size.height)])
        w.add(input); w.startWriting(); w.startSession(atSourceTime: .zero)
        let fps = 30, n = Int(seconds * Double(fps))
        for f in 0..<n {
            let a = Float(f) / Float(n) * 2 * .pi
            cam.simdPosition = SIMD3(sin(a) * r, r * 0.35, cos(a) * r); cam.simdLook(at: .zero)
            let img = ren.snapshot(atTime: Double(f) / Double(fps), with: size, antialiasingMode: .multisampling4X)
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.005) }
            guard let pool = ad.pixelBufferPool else { break }
            var pb: CVPixelBuffer?; CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
            guard let px = pb, let cg = img.cgImage else { continue }
            CVPixelBufferLockBaseAddress(px, [])
            if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(px), width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(px), space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) {
                ctx.draw(cg, in: CGRect(origin: .zero, size: size))
            }
            CVPixelBufferUnlockBaseAddress(px, [])
            ad.append(px, withPresentationTime: CMTime(value: CMTimeValue(f), timescale: CMTimeScale(fps)))
            progress(Float(f + 1) / Float(n))
        }
        input.markAsFinished()
        let sem = DispatchSemaphore(value: 0); w.finishWriting { sem.signal() }; sem.wait()
        return w.status == .completed ? url : nil
    }

    // ---- import ----
    /// PLY (binary little-endian or ASCII, x y z [+ red green blue]) and XYZ/TXT ("x y z [r g b]") files
    static func importCloud(_ url: URL) -> Cloud? {
        guard let d = try? Data(contentsOf: url) else { return nil }
        if url.pathExtension.lowercased() == "ply" { return ply(d) }
        guard let s = String(data: d, encoding: .utf8) else { return nil }
        var o = Cloud()
        for line in s.split(whereSeparator: \.isNewline) {
            let f = line.split(whereSeparator: { $0 == " " || $0 == "," || $0 == "\t" }).compactMap { Float($0) }
            guard f.count >= 3 else { continue }
            o.p.append(SIMD3(f[0], f[1], f[2]))
            // 5.7 (48): u8() clamps and maps nan/inf to grey; Int(Float) trapped on "nan", "inf" or 1e30 colour columns
            let unit: Float = f.count >= 6 && f[3] <= 1 && f[4] <= 1 && f[5] <= 1 ? 255 : 1
            o.c.append(f.count >= 6 ? SIMD3(u8(f[3] * unit), u8(f[4] * unit), u8(f[5] * unit)) : SIMD3(200, 200, 200))
        }
        return o.count > 0 ? o.sane() : nil
    }
    static func ply(_ d: Data) -> Cloud? {
        guard let end = d.range(of: Data("end_header".utf8)) else { return nil }
        var hEnd = end.upperBound; while hEnd < d.count && (d[hEnd] == 0x0A || d[hEnd] == 0x0D) { hEnd += 1 }
        guard let header = String(data: d.subdata(in: 0..<end.lowerBound), encoding: .ascii) else { return nil }
        let lines = header.split(whereSeparator: \.isNewline).map(String.init)
        let binary = lines.contains { $0.hasPrefix("format binary_little_endian") }
        var count = 0, props: [(String, String)] = [], inVertex = false
        for l in lines {
            let w = l.split(separator: " ").map(String.init)
            if w.first == "element" { inVertex = w.count > 2 && w[1] == "vertex"; if inVertex { count = Int(w[2]) ?? 0 } }
            else if w.first == "property", inVertex, w.count >= 3, w[1] != "list" { props.append((w[1], w[2])) }
        }
        let size: [String: Int] = ["float": 4, "float32": 4, "double": 8, "float64": 8, "uchar": 1, "uint8": 1, "char": 1, "int8": 1, "short": 2, "ushort": 2, "int16": 2, "uint16": 2, "int": 4, "uint": 4, "int32": 4, "uint32": 4]
        let names = props.map { $0.1 }
        guard let ix = names.firstIndex(of: "x"), let iy = names.firstIndex(of: "y"), let iz = names.firstIndex(of: "z"), count > 0 else { return nil }
        let ir = names.firstIndex(of: "red"), ig = names.firstIndex(of: "green"), ib = names.firstIndex(of: "blue")
        var offs: [Int] = [], stride = 0
        for (t, _) in props { offs.append(stride); stride += size[t] ?? 4 }
        // 5.7 (48): the header's count is untrusted - check it against the file before reserving (a bad count asked for GBs, and count * stride could overflow)
        let avail = d.count - hEnd
        if binary { guard count <= avail / stride else { return nil } }
        let cap = binary ? count : min(count, avail / 6)          // an ASCII vertex line is at least "0 0 0\n"
        var o = Cloud(); o.p.reserveCapacity(cap); o.c.reserveCapacity(cap)
        if binary {
            d.withUnsafeBytes { b in
                // 5.7 (46): every integer type is read at its own width; 16-bit is rescaled to 0-255 only for colour channels
                func val(_ base: Int, _ k: Int, colour: Bool = false) -> Float {
                    let o = base + offs[k]
                    switch props[k].0 {
                    case "double", "float64": return Float(b.loadUnaligned(fromByteOffset: o, as: Double.self))
                    case "uchar", "uint8": return Float(b[o])
                    case "char", "int8": return Float(Int8(bitPattern: b[o]))
                    case "ushort", "uint16": let v = Float(b.loadUnaligned(fromByteOffset: o, as: UInt16.self)); return colour ? v / 257 : v
                    case "short", "int16": return Float(b.loadUnaligned(fromByteOffset: o, as: Int16.self))
                    case "int", "int32": return Float(b.loadUnaligned(fromByteOffset: o, as: Int32.self))
                    case "uint", "uint32": return Float(b.loadUnaligned(fromByteOffset: o, as: UInt32.self))
                    default: return b.loadUnaligned(fromByteOffset: o, as: Float.self)
                    }
                }
                for i in 0..<count {
                    let base = hEnd + i * stride
                    o.p.append(SIMD3(val(base, ix), val(base, iy), val(base, iz)))
                    if let r = ir, let g = ig, let bb = ib { o.c.append(SIMD3(u8(val(base, r, colour: true)), u8(val(base, g, colour: true)), u8(val(base, bb, colour: true)))) } else { o.c.append(SIMD3(200, 200, 200)) }
                }
            }
        } else {
            guard let body = String(data: d.subdata(in: hEnd..<d.count), encoding: .ascii) else { return nil }
            for line in body.split(whereSeparator: \.isNewline).prefix(count) {
                let f = line.split(separator: " ").compactMap { Float($0) }
                guard f.count > max(ix, iy, iz) else { continue }
                o.p.append(SIMD3(f[ix], f[iy], f[iz]))
                if let r = ir, let g = ig, let bb = ib, f.count > max(r, g, bb) { o.c.append(SIMD3(u8(f[r]), u8(f[g]), u8(f[bb]))) } else { o.c.append(SIMD3(200, 200, 200)) }
            }
        }
        return o.count > 0 ? o.sane() : nil
    }
}
