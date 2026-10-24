import SwiftUI
import RoomPlan
import simd

// ---------- Floor plan mode: Apple RoomPlan captures walls, doors, windows and furniture; we draw the plan ----------
struct FloorPlanScanView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var room: CapturedRoom?
    @State private var capturing = true
    @State private var finishing = false
    @State private var problem = ""
    @StateObject private var ctl = RoomController()

    var body: some View {
        ZStack {
            if !RoomCaptureSession.isSupported {
                VStack(spacing: 14) {
                    Image(systemName: "square.split.bottomrightquarter").font(.system(size: 50))
                    Text("Floor plan mode needs an iPhone with LiDAR.").multilineTextAlignment(.center)
                    Button("Close") { dismiss() }.buttonStyle(.borderedProminent)
                }.padding()
            } else if let r = room {
                FloorPlanResult(room: r) { dismiss() }
            } else {
                RoomCaptureRep(ctl: ctl).ignoresSafeArea()
                VStack {
                    HStack {
                        Button { ctl.stop(); dismiss() } label: { Image(systemName: "xmark").font(.title3.bold()).frame(width: 44, height: 44).background(.ultraThinMaterial, in: Circle()) }
                        Spacer()
                        Text("Floor plan").font(.headline).padding(.horizontal, 12).padding(.vertical, 8).background(.ultraThinMaterial, in: Capsule())
                    }.padding()
                    Spacer()
                    if !problem.isEmpty { Text(problem).font(.footnote).padding(10).background(.ultraThinMaterial, in: Capsule()) }
                    Text("Walk slowly around the room, pointing at every wall, door and window.").font(.footnote).padding(10).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12)).padding(.horizontal)
                    Button { finishing = true; ctl.stop() } label: {
                        Text(finishing ? "Making the plan..." : "Done").font(.headline).frame(minWidth: 200).padding(.vertical, 6)
                    }.buttonStyle(.borderedProminent).controlSize(.large).disabled(finishing).padding(.bottom, 30)
                }
            }
        }
        .background(Color.black)
        .onAppear {
            ctl.onRoom = { r in room = r; Log.add("floor plan: \(r.walls.count) walls, \(r.doors.count) doors, \(r.windows.count) windows, \(r.objects.count) objects") }
            ctl.onError = { e in problem = e; finishing = false }
        }
    }
}

final class RoomController: NSObject, ObservableObject, RoomCaptureViewDelegate, RoomCaptureSessionDelegate {
    var view: RoomCaptureView?
    var onRoom: ((CapturedRoom) -> Void)?
    var onError: ((String) -> Void)?
    override init() { super.init() }
    required init?(coder: NSCoder) { super.init() }
    func encode(with coder: NSCoder) {}
    func start(_ v: RoomCaptureView) {
        view = v; v.delegate = self
        v.captureSession.run(configuration: RoomCaptureSession.Configuration())
    }
    func stop() { view?.captureSession.stop() }
    func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool { true }
    func captureView(didPresent processedResult: CapturedRoom, error: Error?) {
        // 5.7 (46): stop at the error - the result after it went on to replace the message with an empty plan
        // 5.7 (48): the session is stopped by now, so start a fresh one - otherwise the next Done stops a stopped session,
        // no result ever comes and the button hangs on "Making the plan..."
        if let e = error {
            DispatchQueue.main.async {
                self.view?.captureSession.run(configuration: RoomCaptureSession.Configuration())
                self.onError?("RoomPlan: \(e.localizedDescription) Scan the room again, then tap Done.")
            }
            return
        }
        DispatchQueue.main.async { self.onRoom?(processedResult) }
    }
}

struct RoomCaptureRep: UIViewRepresentable {
    let ctl: RoomController
    func makeUIView(context: Context) -> RoomCaptureView { let v = RoomCaptureView(frame: .zero); ctl.start(v); return v }
    func updateUIView(_ v: RoomCaptureView, context: Context) {}
}

enum PlanDraw {
    struct Seg { var a: SIMD2<Float>; var b: SIMD2<Float> }
    /// a surface (wall, door, window) as a line on the floor: centre +- half its width along its own x axis
    static func seg(_ t: simd_float4x4, _ width: Float) -> Seg {
        let c = SIMD2<Float>(t.columns.3.x, t.columns.3.z), d = simd_normalize(SIMD2<Float>(t.columns.0.x, t.columns.0.z))
        return Seg(a: c - d * width / 2, b: c + d * width / 2)
    }
    static func area(_ room: CapturedRoom) -> Float? {
        if #available(iOS 17.0, *), let f = room.floors.first, f.polygonCorners.count >= 3 {
            let pts = f.polygonCorners.map { SIMD2<Float>($0.x, $0.y) }
            var a: Float = 0
            for i in 0..<pts.count { let p = pts[i], q = pts[(i + 1) % pts.count]; a += p.x * q.y - q.x * p.y }
            return abs(a) / 2
        }
        return nil
    }

    static func image(_ room: CapturedRoom, size S: CGFloat = 2000) -> UIImage {
        var all: [SIMD2<Float>] = []
        let walls = room.walls.map { seg($0.transform, $0.dimensions.x) }
        for w in walls { all += [w.a, w.b] }
        for o in room.objects { all.append(SIMD2(o.transform.columns.3.x, o.transform.columns.3.z)) }
        guard !all.isEmpty else { return UIImage() }
        var lo = all[0], hi = all[0]; for p in all { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let span = max(1, max(hi.x - lo.x, hi.y - lo.y)), margin: CGFloat = S * 0.1
        let scale = (S - 2 * margin) / CGFloat(span)
        func pt(_ p: SIMD2<Float>) -> CGPoint { CGPoint(x: margin + CGFloat(p.x - lo.x) * scale, y: margin + CGFloat(p.y - lo.y) * scale) }
        return UIGraphicsImageRenderer(size: CGSize(width: S, height: S)).image { ctx in
            let g = ctx.cgContext
            UIColor.white.setFill(); g.fill(CGRect(x: 0, y: 0, width: S, height: S))
            // floor
            if #available(iOS 17.0, *), let f = room.floors.first, f.polygonCorners.count >= 3 {
                let t = f.transform
                let path = UIBezierPath()
                for (i, c) in f.polygonCorners.enumerated() {
                    let w = t * SIMD4<Float>(c.x, c.y, c.z, 1)
                    let p = pt(SIMD2(w.x, w.z)); if i == 0 { path.move(to: p) } else { path.addLine(to: p) }
                }
                path.close(); UIColor(white: 0.95, alpha: 1).setFill(); path.fill()
            }
            // furniture
            for o in room.objects {
                let c = SIMD2<Float>(o.transform.columns.3.x, o.transform.columns.3.z), ax = simd_normalize(SIMD2<Float>(o.transform.columns.0.x, o.transform.columns.0.z)), az = SIMD2<Float>(-ax.y, ax.x)
                let hx = ax * o.dimensions.x / 2, hz = az * o.dimensions.z / 2
                let p = UIBezierPath(); p.move(to: pt(c - hx - hz)); p.addLine(to: pt(c + hx - hz)); p.addLine(to: pt(c + hx + hz)); p.addLine(to: pt(c - hx + hz)); p.close()
                UIColor(white: 0.86, alpha: 1).setFill(); p.fill(); UIColor(white: 0.55, alpha: 1).setStroke(); p.lineWidth = 3; p.stroke()
                let name = "\(o.category)".replacingOccurrences(of: "Category.", with: "")
                let a: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 26, weight: .medium), .foregroundColor: UIColor.darkGray]
                let sz = (name as NSString).size(withAttributes: a); let cp = pt(c)
                (name as NSString).draw(at: CGPoint(x: cp.x - sz.width / 2, y: cp.y - sz.height / 2), withAttributes: a)
            }
            // walls with their lengths
            g.setLineCap(.square)
            for (i, w) in walls.enumerated() {
                g.setStrokeColor(UIColor.black.cgColor); g.setLineWidth(max(10, scale * 0.1))
                g.move(to: pt(w.a)); g.addLine(to: pt(w.b)); g.strokePath()
                let len = room.walls[i].dimensions.x
                guard len > 0.4 else { continue }
                let mid = (w.a + w.b) / 2, d = simd_normalize(w.b - w.a), nrm = SIMD2<Float>(-d.y, d.x)
                let lp = pt(mid + nrm * Float(52 / scale))
                let txt = Prefs.length(len) as NSString
                let a: [NSAttributedString.Key: Any] = [.font: UIFont.monospacedDigitSystemFont(ofSize: 30, weight: .semibold), .foregroundColor: UIColor.systemBlue]
                let sz = txt.size(withAttributes: a)
                txt.draw(at: CGPoint(x: lp.x - sz.width / 2, y: lp.y - sz.height / 2), withAttributes: a)
            }
            // windows (blue double line) and openings (gap), doors (gap + swing)
            for wnd in room.windows {
                let s = seg(wnd.transform, wnd.dimensions.x)
                g.setStrokeColor(UIColor.white.cgColor); g.setLineWidth(max(10, scale * 0.1)); g.move(to: pt(s.a)); g.addLine(to: pt(s.b)); g.strokePath()
                g.setStrokeColor(UIColor.systemTeal.cgColor); g.setLineWidth(4); g.move(to: pt(s.a)); g.addLine(to: pt(s.b)); g.strokePath()
            }
            for op in room.openings {
                let s = seg(op.transform, op.dimensions.x)
                g.setStrokeColor(UIColor.white.cgColor); g.setLineWidth(max(12, scale * 0.12)); g.move(to: pt(s.a)); g.addLine(to: pt(s.b)); g.strokePath()
            }
            for dr in room.doors {
                let s = seg(dr.transform, dr.dimensions.x)
                g.setStrokeColor(UIColor.white.cgColor); g.setLineWidth(max(12, scale * 0.12)); g.move(to: pt(s.a)); g.addLine(to: pt(s.b)); g.strokePath()
                let r = CGFloat(dr.dimensions.x) * scale, a0 = pt(s.a), dir = atan2(CGFloat(s.b.y - s.a.y), CGFloat(s.b.x - s.a.x))
                g.setStrokeColor(UIColor.systemOrange.cgColor); g.setLineWidth(3)
                g.addArc(center: a0, radius: r, startAngle: dir, endAngle: dir + .pi / 2, clockwise: false); g.strokePath()
                g.move(to: a0); g.addLine(to: CGPoint(x: a0.x + r * cos(dir + .pi / 2), y: a0.y + r * sin(dir + .pi / 2))); g.strokePath()
            }
            // title, area, scale bar
            var title = "Floor plan"
            if let a = area(room) { title += String(format: "  -  %.1f m²", a) + (Prefs.metric ? "" : String(format: " (%.0f sq ft)", a * 10.7639)) }
            (title as NSString).draw(at: CGPoint(x: margin, y: margin * 0.3), withAttributes: [.font: UIFont.systemFont(ofSize: 44, weight: .bold), .foregroundColor: UIColor.black])
            let bar = CGFloat(1) * scale, by = S - margin * 0.45
            g.setStrokeColor(UIColor.black.cgColor); g.setLineWidth(6); g.move(to: CGPoint(x: margin, y: by)); g.addLine(to: CGPoint(x: margin + bar, y: by)); g.strokePath()
            (Prefs.length(1) as NSString).draw(at: CGPoint(x: margin + bar + 14, y: by - 20), withAttributes: [.font: UIFont.systemFont(ofSize: 30), .foregroundColor: UIColor.black])
            ("Deepfield" as NSString).draw(at: CGPoint(x: S - margin - 150, y: by - 20), withAttributes: [.font: UIFont.systemFont(ofSize: 28), .foregroundColor: UIColor.gray])
        }
    }
}

struct FloorPlanResult: View {
    let room: CapturedRoom
    let close: () -> Void
    @State private var image: UIImage?
    @State private var share: URL?
    @State private var saved = false
    var body: some View {
        NavigationStack {
            VStack {
                if let img = image {
                    ScrollView([.horizontal, .vertical]) { Image(uiImage: img).resizable().scaledToFit().frame(width: 380) }
                    HStack(spacing: 16) {
                        Label("\(room.walls.count) walls", systemImage: "square.split.2x1")
                        Label("\(room.doors.count) doors", systemImage: "door.left.hand.open")
                        Label("\(room.windows.count) windows", systemImage: "window.casement")
                    }.font(.caption).foregroundStyle(.secondary)
                } else { ProgressView("Drawing the plan...") }
            }
            .navigationTitle("Floor plan").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close", action: close) }
                ToolbarItem(placement: .confirmationAction) {
                    Menu {
                        Button("Share plan image") { share = png() }
                        Button("Share as PDF") { if let i = image { let u = Export.tmp("Floor plan.pdf"); try? Export.pdf(i, title: "Floor plan").write(to: u); share = u } }
                        Button("Share as DXF (CAD)") { let u = Export.tmp("Floor plan.dxf"); try? Export.dxf(room).write(to: u); share = u }
                        Button("Share 3D model (USDZ)") { share = usdz() }
                        Button(saved ? "Saved to library" : "Save to library") { save() }.disabled(saved)
                    } label: { Image(systemName: "square.and.arrow.up") }
                }
            }
            .sheet(item: $share) { ShareSheet(items: [$0]) }
            .task { image = PlanDraw.image(room) }
        }
    }
    private func png() -> URL? {
        guard let d = image?.pngData() else { return nil }
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("Floor plan.png"); try? d.write(to: u); return u
    }
    private func usdz() -> URL? {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("Room.usdz")
        try? FileManager.default.removeItem(at: u)
        do { try room.export(to: u, exportOptions: .parametric); return u } catch { return nil }
    }
    private func save() {
        guard let u = usdz(), let it = Library.shared.addModel(from: u, name: "Floor plan " + Date().formatted(date: .abbreviated, time: .shortened)) else { return }
        if let img = image, let small = UIGraphicsImageRenderer(size: CGSize(width: 240, height: 240)).image(actions: { _ in img.draw(in: CGRect(x: 0, y: 0, width: 240, height: 240)) }).jpegData(compressionQuality: 0.8) {
            // 5.7 (46): queued behind addModel's own render on Library.io, which otherwise wrote over the plan picture
            let id = it.id
            Library.io.async {
                try? small.write(to: Library.thumbURL(id))
                Thumbs.forget(id)
                DispatchQueue.main.async { Library.shared.objectWillChange.send() }
            }
        }
        if let d = image?.pngData() { try? d.write(to: Library.folder(it.id).appendingPathComponent("plan.png")) }
        saved = true
    }
}

// ---------- a floor plan from any point cloud: the walls sliced out between knee and head height ----------
enum CloudPlan {
    static func image(_ cl: Cloud, size S: CGFloat = 1600) -> UIImage? {
        guard cl.count > 1000 else { return nil }
        let floorY = CloudOps.floorHeight(cl)
        var pts: [SIMD2<Float>] = []
        let step = max(1, cl.count / 2_000_000)
        for i in stride(from: 0, to: cl.count, by: step) { let q = cl.p[i]; if q.y > floorY + 0.5 && q.y < floorY + 1.9 { pts.append(SIMD2(q.x, q.z)) } }
        guard pts.count > 200 else { return nil }
        var lo = pts[0], hi = pts[0]; for p in pts { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        // the floor's footprint for the area
        var floorCells = Set<SIMD2<Int32>>()
        for i in stride(from: 0, to: cl.count, by: step) { let q = cl.p[i]; if abs(q.y - floorY) < 0.04 { floorCells.insert(SIMD2(ik((q.x / 0.1)), ik((q.z / 0.1)))) } }
        let area = Float(floorCells.count) * 0.01
        let span = max(0.5, max(hi.x - lo.x, hi.y - lo.y)), margin: CGFloat = S * 0.1, scale = (S - 2 * margin) / CGFloat(span)
        return UIGraphicsImageRenderer(size: CGSize(width: S, height: S)).image { ctx in
            let g = ctx.cgContext
            UIColor.white.setFill(); g.fill(CGRect(x: 0, y: 0, width: S, height: S))
            UIColor(white: 0.93, alpha: 1).setFill()
            for c in floorCells {
                let x = margin + CGFloat(Float(c.x) * 0.1 - lo.x) * scale, y = margin + CGFloat(Float(c.y) * 0.1 - lo.y) * scale
                g.fill(CGRect(x: x, y: y, width: 0.1 * scale + 1, height: 0.1 * scale + 1))
            }
            UIColor.black.withAlphaComponent(0.35).setFill()
            for p in pts { g.fill(CGRect(x: margin + CGFloat(p.x - lo.x) * scale, y: margin + CGFloat(p.y - lo.y) * scale, width: 2, height: 2)) }
            let s = cl.bounds.size
            let title = String(format: "Floor plan  -  %@ x %@, floor about %.1f m²", Prefs.length(s.x), Prefs.length(s.z), area)
            (title as NSString).draw(at: CGPoint(x: margin, y: margin * 0.3), withAttributes: [.font: UIFont.systemFont(ofSize: 34, weight: .bold), .foregroundColor: UIColor.black])
            let bar = scale, by = S - margin * 0.45
            g.setStrokeColor(UIColor.black.cgColor); g.setLineWidth(5); g.move(to: CGPoint(x: margin, y: by)); g.addLine(to: CGPoint(x: margin + bar, y: by)); g.strokePath()
            (Prefs.length(1) as NSString).draw(at: CGPoint(x: margin + bar + 12, y: by - 18), withAttributes: [.font: UIFont.systemFont(ofSize: 26), .foregroundColor: UIColor.black])
        }
    }
}
