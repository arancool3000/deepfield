import SwiftUI
import PhotosUI
import SceneKit
import simd
import os

/// The editor's state: the cloud, undo/redo, and every tool run off the main thread.
@MainActor
final class EditorModel: ObservableObject {
    @Published var cloud = Cloud() { didSet { rebuildShown() } }
    @Published var busy: String? { didSet { if busy == nil { progress = nil }; Awake.set(Awake.key(self, "edit"), busy != nil) } }
    deinit { Awake.set(Awake.key(self, "edit"), false) }
    @Published var progress: Double?                // 5.0: a real progress bar for long jobs (nil = spinner)
    private func step(_ f: Double, _ s: String) { Task { @MainActor in if self.busy != nil { self.progress = f; self.busy = s } } }
    @Published var message = ""
    @Published var stamp = 0
    @Published var plan: AIPlan?
    @Published var smart: SmartPlan?
    @Published var aiFill = Cloud() { didSet { rebuildShown() } }   // AI-made points for missing parts, shown yellow until kept or removed
    @Published var shown = Cloud()
    var bounds = Cloud.Box(lo: .zero, hi: .zero)          // of the cloud, worked out once per edit (never per frame)
    private func rebuildShown() { shown = aiFill.count > 0 ? Cloud(p: cloud.p + aiFill.p, c: cloud.c + aiFill.c) : cloud; bounds = cloud.bounds }
    var aiReal = Cloud()                     // the same points with the colour of the real surface around them
    // 5.0 solid view: a real surface built from the points (the mesher), rebuilt after each edit while it is shown
    @Published var mesh: SCNGeometry?
    var realPhotos: [RealPhoto] = []            // 5.0 photoreal: the scan's photos, projected live onto the model
    var meshCentre = SIMD3<Float>.zero
    @Published var meshNote = ""
    var meshStamp = -1
    var meshFor = -1                      // the edit the surface on screen was made for
    private var meshing = false
    // 5.7 (43): the photo surface is built from the capture's depth, not from the points, so edits never reached it -
    // after a cut the old surface stayed on screen (38 points left, a whole speaker drawn) and tools worked on parts
    // you could not see. It is built once and kept; every edit trims it to the points that are left.
    private var fullMesh: TexturedMesh?
    nonisolated static func trim(_ t: TexturedMesh, to c: Cloud) -> TexturedMesh {
        let ext = max(c.bounds.size.x, max(c.bounds.size.y, c.bounds.size.z)), s = max(0.008, ext / 250)
        var cells = Set<SIMD3<Int32>>(); cells.reserveCapacity(c.count / 2)
        @inline(__always) func key(_ p: SIMD3<Float>) -> SIMD3<Int32> { SIMD3(Int32((p.x / s).rounded(.down)), Int32((p.y / s).rounded(.down)), Int32((p.z / s).rounded(.down))) }
        for p in c.p { cells.insert(key(p)) }
        func near(_ p: SIMD3<Float>) -> Bool {
            let k = key(p)
            for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 where cells.contains(k &+ SIMD3(dx, dy, dz)) { return true } } }
            return false
        }
        var o = t; o.f = []; o.f.reserveCapacity(t.f.count)
        for i in stride(from: 0, to: t.f.count, by: 3) {
            let ctr = (t.v[Int(t.f[i])] + t.v[Int(t.f[i + 1])] + t.v[Int(t.f[i + 2])]) / 3
            if near(ctr) { o.f += [t.f[i], t.f[i + 1], t.f[i + 2]] }
        }
        return o
    }
    func buildMesh() {
        guard !meshing, loaded, cloud.count > 1000 else { if cloud.count <= 1000 { mesh = nil; meshStamp = stamp; meshFor = stamp; meshNote = "Too few points for a surface" }; return }
        meshing = true; meshStamp = stamp
        let c = cloud, it = item, photos = hasPhotos, kept = fullMesh, forStamp = stamp, gen = loadGen
        meshNote = "Building the surface..."
        Task.detached(priority: .userInitiated) {
            let ctr = c.bounds.centre, ext = max(c.bounds.size.x, max(c.bounds.size.y, c.bounds.size.z))
            var geo: SCNGeometry? = nil
            // 5.0: with photos, the lifelike model (shape from every photo's LiDAR depth, look from the photos)
            var rp: [RealPhoto] = []
            var full = kept
            if photos, full == nil { full = PhotoModel.build(it, extent: ext, detail: 1, progress: { p, s in Task { @MainActor in self.meshNote = "\(s) \(Int(p * 100))%" } }) }
            if photos, let t = full { geo = EditorModel.trim(t, to: c).geometry(centre: ctr); if kept == nil { rp = RealPhoto.load(it) } }
            let fullOut = full
            if geo == nil {
                let cams = Mesher.cameras(it)
                let md = Mesher.build(c, cams: cams) { p, s in Task { @MainActor in self.meshNote = "\(s) \(Int(p * 100))%" } }
                geo = md.map { PointCloudView.meshGeometry($0, centre: ctr) }
            }
            let g = geo, photosLoaded = rp
            await MainActor.run {
                // 5.7 (46): a surface made before a reload (continue scanning) belongs to the old capture - never keep it
                if gen == self.loadGen {
                    if !photosLoaded.isEmpty { self.realPhotos = photosLoaded }
                    self.fullMesh = fullOut
                }
                self.meshCentre = ctr
                self.mesh = g; self.meshFor = forStamp; self.meshing = false
                self.meshNote = g == nil ? "Too few points for a surface" : ""
                if self.meshStamp != self.stamp { self.buildMesh() }        // edited while building
            }
        }
    }

    var smartRefs: [DepthView] = [], smartViews: [DepthView] = []
    @Published var loaded = false
    var frames: [Keyframe] = []
    /// 5.7 (47): one undo step - the whole cloud from before an edit, named after that edit
    struct UndoStep { let label: String; let cloud: Cloud; var bytes: Int { cloud.count * 20 } }   // 16 B position + 4 B colour
    private var undoStack: [UndoStep] = [], redoStack: [UndoStep] = []
    var item: ScanItem
    var dirty = false
    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }
    /// 5.7 (47): the undo steps' names, newest first - what long-pressing Undo lists
    var history: [String] { undoStack.reversed().map { $0.label } }

    init(item: ScanItem) { self.item = item }

    private var loadGen = 0
    func load() {
        busy = "Opening..."
        let it = item, again = loaded
        Task.detached(priority: .userInitiated) {
            // 5.7 (48): continue scanning queues the merged points on Library.io moments before this reload - reading
            // first got the old file, and the next save wrote it back over the new points. Wait for the queued writes.
            await withCheckedContinuation { (k: CheckedContinuation<Void, Never>) in Library.io.async { k.resume() } }
            let c = Library.load(it) ?? Cloud()
            await MainActor.run {
                // 5.7 (46): continue scanning reloads a bigger capture - the kept photo surface, its photos, the photo
                // check, the pending AI fill and the undo steps all belonged to the old one (new parts had no surface)
                if again {
                    self.loadGen += 1; self.fullMesh = nil; self.realPhotos = []; self.mesh = nil
                    self.hasPhotos = EditorModel.photosExist(it)
                    self.aiFill = Cloud(); self.aiReal = Cloud()
                    self.undoStack.removeAll(); self.redoStack.removeAll()
                }
                self.cloud = c; self.busy = nil; self.loaded = true; self.stamp += 1
            }
        }
    }

    /// 5.7 (46): an edit made while yellow AI fill is waiting would leave the fill (made for the old cloud) on screen,
    /// and Keep would put back points from areas just cut - the unkept fill is dropped instead
    private func dropFill() -> String {
        guard aiFill.count > 0 || aiReal.count > 0 else { return "" }
        aiFill = Cloud(); aiReal = Cloud(); return " (the unkept AI fill was removed)"
    }

    /// runs an edit in the background; 5.7 (47): keeps as many named undo steps (each a whole copy of the cloud) as fit
    /// in a memory budget - a twelfth of the phone's memory, at most 900 MB and 30 steps, never fewer than one
    func run(_ label: String, _ f: @escaping @Sendable (Cloud) -> (Cloud, String), then: (@MainActor () -> Void)? = nil) {
        guard busy == nil else { return }
        busy = label; editing = true
        let gen = loadGen
        // 5.7 (48): before a big edit, old undo copies go first if memory is short (a Quick fill on 9M points, with undo
        // copies kept, ran the phone out of memory and the app was killed)
        while undoStack.count > 0 && os_proc_available_memory() < 1_500_000_000 { undoStack.removeFirst() }
        redoStack.removeAll()
        let before = cloud, dropped = dropFill()
        Task.detached(priority: .userInitiated) {
            let (raw, msg) = f(before)
            let after = raw.sane()
            await MainActor.run {
                self.editing = false
                // 5.7 (48): made from the cloud before a reload (continue scanning) - landing it would drop the new points
                guard gen == self.loadGen else { return }        // (the reload already cleared busy)
                let name = label.hasSuffix("...") ? String(label.dropLast(3)) : label
                self.undoStack.append(UndoStep(label: name, cloud: before))
                // 5.7 (48): also never more than a quarter of what is free right now
                let budget = min(900_000_000, Int(ProcessInfo.processInfo.physicalMemory / 12), os_proc_available_memory() / 4)   // a 10M-point copy is ~200 MB
                while self.undoStack.count > 30 || (self.undoStack.count > 1 && self.undoStack.reduce(0, { $0 + $1.bytes }) > budget) { self.undoStack.removeFirst() }
                self.redoStack.removeAll()
                self.cloud = after; self.dirty = true
                self.message = msg + dropped; self.busy = nil; self.stamp += 1
                then?()
            }
        }
    }
    /// 5.7 (48): an edit is running (Back would leave before its result is saved)
    @Published private(set) var editing = false
    func undo() { guard let s = undoStack.popLast() else { return }; let d = dropFill(); redoStack.append(UndoStep(label: s.label, cloud: cloud)); cloud = s.cloud; dirty = true; stamp += 1; message = "Undone" + d }
    func redo() { guard let s = redoStack.popLast() else { return }; let d = dropFill(); undoStack.append(UndoStep(label: s.label, cloud: cloud)); cloud = s.cloud; dirty = true; stamp += 1; message = "Redone" + d }
    /// 5.7 (47): jumps back to before the step at `index` in `history` (0 = newest) - every step passed goes onto
    /// redo in order, so Redo walks forward again one step at a time
    func undo(to index: Int) {
        guard index >= 0, index < undoStack.count else { return }
        let d = dropFill()
        var label = ""
        for _ in 0...index {
            let s = undoStack.removeLast()
            redoStack.append(UndoStep(label: s.label, cloud: cloud)); cloud = s.cloud; label = s.label
        }
        dirty = true; stamp += 1; message = "Back to before: " + label + d
    }

    func save() { if dirty { Library.shared.update(item, cloud: cloud); dirty = false } }

    nonisolated static func change(_ a: Int, _ b: Int) -> String {
        let d = a - b
        return d >= 0 ? "Removed \(d.formatted()) points, \(b.formatted()) left" : "Added \((-d).formatted()) points"
    }

    // ---- Gemini-led tools: Gemini compares the photos with the scan, plans, the user approves ----
    @Published var deep = false                // Gemini 3.8 Flash (then 3.6) for the smart tools
    // 5.0: Gemini edits by numbered parts (SceneAI); the measured tools run straight away
    @Published var partPlan: PartPlan?
    var seg: Segments?
    func smartTool(_ tool: String) {
        switch tool {
        case "floaters":
            let it = item
            run("Finding stray specks and floating junk...") { c in
                let seg = Segments.find(c, views: DepthView.load(it, max: 24))
                return SceneAI.apply(PartPlan(summary: "", name: nil, steps: [PartPlan.Step(op: "strays")]), skip: [], to: c, seg: seg, item: it, grid: Float(Prefs.mergeGridMM / 1000))
            }
        case "floor":
            run("Removing the floor...") { c in
                guard let (n, d) = CloudOps.floorPlane(c) else { return (c, "No floor found") }
                let o = c.keeping(c.p.map { simd_dot(n, $0) + d > 0.012 }); return (o, "Floor removed: " + EditorModel.change(c.count, o.count))
            }
        case "smooth":
            let g = Float(Prefs.mergeGridMM / 1000)
            run("Smoothing...") { c in (CloudOps.smooth(c, radius: max(0.004, g * 2.5), amount: 0.5), "Smoothed - surfaces flatter, edges kept. Undo if it is too much.") }
        default: askParts(AITools.taskText(tool))
        }
    }
    func askParts(_ request: String) {
        guard busy == nil else { return }
        busy = "Splitting the scan into parts..."
        let c = cloud, it = item, dp = deep
        Task {
            do {
                let (p, sg) = try await SceneAI.plan(request, cloud: c, item: it, deep: dp) { s in Task { @MainActor in if self.busy != nil { self.busy = s } } }
                busy = "Gemini is planning..."
                seg = sg; busy = nil
                if p.steps.isEmpty { message = p.summary.isEmpty ? "Gemini found nothing to change." : p.summary } else { partPlan = p }
            } catch { busy = nil; message = error.localizedDescription }
        }
    }
    func applyParts(_ p: PartPlan, skip: Set<Int>) {
        guard let sg = seg else { return }
        if let n = p.name, !n.isEmpty, item.name.hasPrefix("Scan ") { Library.shared.rename(item, to: n); item.name = n }
        partPlan = nil
        let it = item, grid = Float(Prefs.mergeGridMM / 1000)
        run("Applying the plan...") { c in SceneAI.apply(p, skip: skip, to: c, seg: sg, item: it, grid: grid) }
    }
    func ask(_ task: String, fallback: String? = nil, only: [String]? = nil) {
        guard busy == nil else { return }
        busy = hasPhotos ? (deep ? "Gemini 3.8 is studying your scan and photos..." : "Gemini is comparing your scan with the photos...") : (deep ? "Gemini 3.8 is studying views of your scan..." : "Gemini is looking at views of your scan...")
        let c = cloud, it = item, dp = deep
        Task {
            do {
                let (p, refs, views) = try await AITools.smartPlan(task, cloud: c, item: it, deep: dp, only: only)
                smartRefs = refs; smartViews = views; busy = nil
                if p.steps.isEmpty { message = p.summary.isEmpty ? "Gemini found nothing to change." : p.summary } else { smart = p }
            } catch {
                busy = nil
                guard let f = fallback else { message = error.localizedDescription; return }
                // Gemini unreachable: the careful version that never removes surfaces
                let g = Float(Prefs.mergeGridMM / 1000)
                message = "Gemini unavailable (\(error.localizedDescription)). Used the safe version."
                run("Safe clean-up...") { c in
                    switch f {
                    case "floaters": let o = CloudOps.removeFloaters(c, radius: max(0.02, g * 6), minN: 1); return (o, "Safe: " + EditorModel.change(c.count, o.count))
                    case "smooth": return (CloudOps.smooth(c, radius: max(0.004, g * 2), amount: 0.3), "Safe: smoothed lightly")
                    case "floor": let (o, ok) = CloudOps.removeFloor(c); return (o, ok ? "Safe: floor removed" : "No floor found")
                    default: let o = CloudOps.dedupe(c, grid: g); return (o, "Safe: merged copies in the same cell, " + EditorModel.change(c.count, o.count))
                    }
                }
            }
        }
    }
    func applySmart(_ p: SmartPlan) {
        let refs = smartRefs, views = smartViews, grid = Float(Prefs.mergeGridMM / 1000)
        if let n = p.name, !n.isEmpty, item.name.hasPrefix("Scan ") { Library.shared.rename(item, to: n); item.name = n }
        smart = nil
        run("Applying Gemini's plan...") { c in AITools.apply(p, to: c, refs: refs, views: views, grid: grid) }
    }

    // ---- AI fill of missing room parts ----
    func fillGaps() {
        guard busy == nil else { return }
        busy = "Recovering LiDAR and generating the missing parts..."
        let c = cloud, it = item, dp = deep
        Task {
            let (y, r, msg) = await SmartFill.run(c, item: it, deep: dp, progress: step)
            aiFill = y; aiReal = r; busy = nil; message = msg; stamp += 1
        }
    }
    /// generate inside a box (fractions of the bounds)
    func generate(lo: SIMD3<Float>, hi: SIMD3<Float>) {
        guard busy == nil else { return }
        busy = "Generating points in the box..."
        let c = cloud, it = item, b = cloud.bounds
        let box = (b.lo + b.size * lo, b.lo + b.size * hi)
        Task {
            let (y, r, msg) = await SmartFill.run(c, item: it, deep: false, box: box, progress: step)
            aiFill = y; aiReal = r; busy = nil; message = msg; stamp += 1
        }
    }
    /// generate inside a box given in scan coordinates (the tapped spot)
    func generate(world lo: SIMD3<Float>, _ hi: SIMD3<Float>) {
        guard busy == nil else { return }
        busy = "Generating points at the spot you picked..."
        let c = cloud, it = item
        Task {
            let (y, r, msg) = await SmartFill.run(c, item: it, deep: false, box: (lo, hi), progress: step)
            aiFill = y; aiReal = r; busy = nil; message = msg; stamp += 1
        }
    }
    /// 5.7 (48): generate inside the cut box as shown - turned about up by `yaw` (the wand used the unturned box, so
    /// with Turn set it filled outside the yellow box and missed its corners)
    func generate(centre: SIMD3<Float>, half: SIMD3<Float>, yaw: Float) {
        guard busy == nil else { return }
        busy = "Generating points in the box..."
        let c = cloud, it = item, cy = cos(yaw), sy = sin(yaw)
        let inside: (SIMD3<Float>) -> Bool = { p in
            let d = p - centre, q = SIMD3<Float>(cy * d.x - sy * d.z, d.y, sy * d.x + cy * d.z)     // as CloudOps.cropBox
            return abs(q.x) <= half.x && abs(q.y) <= half.y && abs(q.z) <= half.z
        }
        let ax = abs(cy) * half.x + abs(sy) * half.z, az = abs(sy) * half.x + abs(cy) * half.z    // the turned box's bounds
        let box = (centre - SIMD3(ax, half.y, az), centre + SIMD3(ax, half.y, az))
        Task {
            let (y, r, msg) = await SmartFill.run(c, item: it, deep: false, box: box, region: inside, progress: step)
            // recovered LiDAR is gathered over the whole bounds - only what lies in the turned box is offered
            let yi = y.keeping(y.p.map(inside)), ri = r.keeping(r.p.map(inside))
            aiFill = yi; aiReal = ri; busy = nil; stamp += 1
            message = yi.count == y.count ? msg : "AI generated \(yi.count.formatted()) points (yellow) inside the box, coloured from your photos."
        }
    }
    func fixDrift() {
        guard busy == nil else { return }
        let it = item
        busy = "Measuring drift from your photos..."
        let c = cloud
        Task.detached(priority: .userInitiated) {
            let (o, msg) = Drift.fix(c, item: it) { p, s in Task { @MainActor in self.busy = s + " \(Int(p * 100))%" } }
            await MainActor.run { self.busy = nil; self.run("Applying the drift fix...") { _ in (o, msg) } }
        }
    }
    /// dense fill inside spheres around the tapped spot or the drawn stroke
    func fillRegion(centres: [SIMD3<Float>], radius r: Float, guide: String = "", refs: [UIImage] = []) {
        guard busy == nil, !centres.isEmpty else { return }
        busy = "Generating points where you marked..."
        var lo = centres[0], hi = centres[0]; for c in centres { lo = simd_min(lo, c); hi = simd_max(hi, c) }
        let box = (lo - r, hi + r)
        let g = GridIndex(centres, cell: max(0.01, r)), cs = centres, r2 = r * r
        let inside: (SIMD3<Float>) -> Bool = { p in var hit = false; g.around(p) { j in if !hit && simd_length_squared(cs[Int(j)] - p) <= r2 { hit = true } }; return hit }
        let c = cloud, it = item
        Task {
            let (y, rr, msg) = await SmartFill.run(c, item: it, deep: false, box: box, region: inside) { f, t in self.step(f * 0.7, t) }
            var real = rr, note = msg
            if rr.count > 0 {
                // generative texture: grown from the real surroundings (or the reference photos), steered by the user's words
                let (pc, pm) = await GenFill.texture(rr, scan: c, item: it, refs: refs, guide: guide) { s in self.step(s.hasPrefix("Gemini") ? 0.75 : (s.hasPrefix("Growing") ? 0.85 : 0.95), s) }
                real = pc; if !pm.isEmpty { note = "\(rr.count.formatted()) points generated. " + pm }
            }
            aiFill = y; aiReal = real; busy = nil; message = note; stamp += 1
        }
    }
    func alignToPhotos() {
        guard busy == nil else { return }
        busy = deep ? "Gemini 3.8 is checking the scan against the photos..." : "Checking the scan against the photos..."
        let c = cloud, it = item, dp = deep
        Task {
            let (o, msg) = await SmartFill.align(c, item: it, deep: dp)
            busy = nil
            run("Applying the alignment...") { _ in (o, msg) }
        }
    }
    func keepFill() {
        guard aiReal.count > 0 else { return }
        let add = aiReal
        aiFill = Cloud(); aiReal = Cloud()
        run("Colouring the AI parts...") { c in var o = c; o.p += add.p; o.c += add.c; return (o, "Kept \(add.count.formatted()) AI points, coloured from the real surfaces") }
    }
    /// 5.0: worked out once (it was a folder listing on every redraw - every slider movement)
    lazy var hasPhotos: Bool = EditorModel.photosExist(item)
    nonisolated static func photosExist(_ item: ScanItem) -> Bool {
        guard let cap = Library.captureURL(item) else { return false }
        return ((try? FileManager.default.contentsOfDirectory(atPath: cap.appendingPathComponent("frames").path)) ?? []).contains { $0.hasSuffix(".jpg") }
    }
    func recolourFromPhotos() {
        let it = item
        run("Re-colouring from the photos...") { c in
            var o = c
            let photos = PhotoView.load(DepthView.load(it, max: 60))
            _ = SmartFill.recolour(&o, photos)
            return (o, "Colours re-projected from \(photos.count) photos")
        }
    }
    func measure() {
        let c = cloud
        busy = "Measuring..."
        Task.detached {
            let s = c.bounds.size
            var txt = "\(Prefs.length(s.x)) wide x \(Prefs.length(s.y)) tall x \(Prefs.length(s.z)) deep"
            if let (n, d) = CloudOps.floorPlane(c), n.y > 0.95 {
                var cells = Set<SIMD2<Int32>>()
                for q in c.p where abs(simd_dot(n, q) + d) < 0.03 { cells.insert(SIMD2(ik((q.x / 0.1)), ik((q.z / 0.1)))) }
                let area = Float(cells.count) * 0.01
                let top = c.p.reduce(-Float.greatestFiniteMagnitude) { max($0, $1.y) }
                txt += String(format: " - floor %.1f m², height %@", area, Prefs.length(top + d / n.y))
            }
            let msg = "Size: " + txt
            await MainActor.run { self.busy = nil; self.message = msg }
        }
    }
    func removePeople() {
        guard busy == nil else { return }
        busy = "Finding people in your scan..."
        let c = cloud, it = item
        Task {
            let (o, msg) = await People.remove(c, item: it)
            guard o.count != c.count else { busy = nil; message = msg; return }
            // 5.7 (48): the places they were standing are offered as yellow AI fill once the removal has landed (a fixed
            // 0.6 s wait found the removal still busy and skipped it, and with nobody found it filled the whole scan)
            let spots = await Task.detached(priority: .userInitiated) { EditorModel.emptied(c, o, cell: 0.15) }.value
            busy = nil
            run("Removing people...", { _ in (o, msg) }, then: { if !spots.isEmpty { self.fillRegion(centres: spots, radius: 0.25) } })
        }
    }
    /// 5.7 (48): centres of the cells that had points in `a` and have none left in `b` (what an edit removed), at most 2000
    nonisolated static func emptied(_ a: Cloud, _ b: Cloud, cell s: Float) -> [SIMD3<Float>] {
        @inline(__always) func key(_ p: SIMD3<Float>) -> SIMD3<Int32> { SIMD3(Int32((p.x / s).rounded(.down)), Int32((p.y / s).rounded(.down)), Int32((p.z / s).rounded(.down))) }
        var left = Set<SIMD3<Int32>>(); left.reserveCapacity(b.count / 8)
        for p in b.p { left.insert(key(p)) }
        var seen = Set<SIMD3<Int32>>(), out: [SIMD3<Float>] = []
        for p in a.p {
            let k = key(p)
            if !left.contains(k), seen.insert(k).inserted { out.append((SIMD3(Float(k.x), Float(k.y), Float(k.z)) + 0.5) * s) }
        }
        guard out.count > 2000 else { return out }
        let st = Double(out.count) / 2000
        return (0..<2000).map { out[Int(Double($0) * st)] }
    }
    /// 5.0: measured against a reference model made from the photos (no Gemini plan)
    func removeDoubles() {
        guard busy == nil else { return }
        let it = item, c = cloud
        busy = "Finding doubles..."
        Task.detached(priority: .userInitiated) {
            let (o, msg) = Doubles.fix(c, item: it) { s in Task { @MainActor in self.busy = s } }
            await MainActor.run { self.busy = nil; if o.count != c.count { self.run("Removing doubles...") { _ in (o, msg) } } else { self.message = msg } }
        }
    }
    /// 5.0: instant offline densification (no network): ~4x the points on the local surface
    func quickFill() {
        // 5.7 (52): room first - the photo surface (rebuilt afterwards) and all but the newest undo copy go, or the memory cap
        // left no room and Quick fill added 1.0x
        fullMesh = nil; mesh = nil
        if undoStack.count > 1 { undoStack.removeFirst(undoStack.count - 1) }
        Log.add("quick fill: \(cloud.count) points, \(os_proc_available_memory() / 1_000_000) MB free")
        run("Quick fill: adding points on the surfaces...") { c in
            let cap = EditorModel.fillCap(c.count)
            let o = CloudOps.quickFill(c, cap: cap)
            let x = Double(o.count) / Double(max(1, c.count))
            let note = x < 1.5 ? " Memory only allowed a small fill - close other apps, or fill inside the cut box (smaller areas fill fully)." : ""
            return (o, "Quick fill: \(c.count.formatted()) -> \(o.count.formatted()) points (\(String(format: "%.1f", x))x), surfaces smoothed and gaps filled, offline. Undo to go back." + note)
        }
    }
    /// 5.6: Quick fill only inside the box (the rest of the scan stays exactly as it is)
    func quickFill(centre: SIMD3<Float>, half: SIMD3<Float>, yaw: Float) {
        run("Quick fill inside the box...") { c in
            let inside = CloudOps.cropBox(c, centre: centre, half: half, yaw: yaw, keepInside: true)
            guard inside.count > 100 else { return (c, "Too few points inside the box to fill") }
            let outside = CloudOps.cropBox(c, centre: centre, half: half, yaw: yaw, keepInside: false)
            let filled = CloudOps.quickFill(inside, cap: EditorModel.fillCap(c.count) - outside.count)
            var o = outside; o.p += filled.p; o.c += filled.c
            return (o, "Quick fill inside the box: \(inside.count.formatted()) -> \(filled.count.formatted()) points. Undo to go back.")
        }
    }
    /// 5.7 (48): how many points Quick fill may end with - what free memory allows (measured: 1.93 GB peak for 4.3M -> 17.7M
    /// points, ~105 bytes per point it ends with; 120 kept as margin), never more than 24M, never fewer than the scan has
    nonisolated static func fillCap(_ have: Int) -> Int {
        let free = Int(os_proc_available_memory()) - 700_000_000 - have * 20
        return max(have, min(24_000_000, free / 120))
    }
    func blurFaces() {
        let it = item
        run("Blurring faces...") { c in People.blurFaces(c, item: it) }
    }
    func removeFill() { aiFill = Cloud(); aiReal = Cloud(); message = "AI parts removed"; stamp += 1 }

    // ---- AI ----
    func askAI() {
        guard busy == nil else { return }
        busy = "Asking Gemini..."
        let c = cloud, it = item
        Task {
            do { let (p, fr) = try await AITools.plan(for: c, item: it); frames = fr; plan = p; busy = nil }
            catch { busy = nil; message = error.localizedDescription }
        }
    }
    func applyPlan(_ p: AIPlan) {
        let fr = frames, grid = Float(Prefs.mergeGridMM / 1000)
        if let n = p.name, !n.isEmpty, item.name.hasPrefix("Scan ") { Library.shared.rename(item, to: n); item.name = n }
        run("Applying the AI plan...") { c in AITools.apply(p, to: c, frames: fr, grid: grid) }
        plan = nil
    }
}

struct EditorView: View {
    @StateObject private var m: EditorModel
    @Environment(\.dismiss) private var dismiss
    @AppStorage(PrefKey.pointSize) private var pointSize = 3.0
    @AppStorage(PrefKey.background) private var background = "black"
    @AppStorage(PrefKey.colourMode) private var colourMode = "photo"
    @AppStorage("pointStyle") private var style = "points"
    @AppStorage(PrefKey.mergeGrid) private var gridMM = 2.0
    @State private var tool = 0
    @State private var cutLow = 0.0
    @State private var cutHigh = 1.0
    @State private var boxX = 0.0...1.0
    @State private var boxZ = 0.0...1.0
    @State private var bright = 0.0
    @State private var contrast = 1.0
    @State private var sat = 1.0
    @State private var share: URL?
    @State private var renaming = false
    @State private var newName = ""
    @State private var instruction = ""
    @State private var showModel = false
    @State private var showAR = false
    @State private var showMesh = false
    @State private var showSplat = false
    @State private var showAran = false
    @State private var showContinue = false
    @State private var picking = false
    @State private var fillMode = 0                   // 0 off, 1 tap a spot, 2 draw an area
    @State private var stroke: [SIMD3<Float>] = []
    @State private var askFill = false
    @State private var fillGuide = ""
    @State private var refPicks: [PhotosPickerItem] = []
    @State private var refImages: [UIImage] = []
    @State private var pick: SIMD3<Float>?
    @State private var fillRadius = 0.25
    @State private var showPlan = false
    @State private var measure: [SIMD3<Float>] = []
    @State private var closeShape = false
    @State private var noteMode = false
    @State private var notePoint: SIMD3<Float>?
    @State private var noteText = ""
    @State private var section = 1.0
    @State private var videoBusy = false
    @State private var toolsHidden = false            // 5.0: look at the scan with nothing over it
    @State private var cutKeep = true                 // Cut preview: keep inside (outside dimmed) or remove inside
    @State private var cutYaw = 0.0                   // the box turned about up, degrees

    var autoFill = false
    init(item: ScanItem, autoFill: Bool = false) { _m = StateObject(wrappedValue: EditorModel(item: item)); self.autoFill = autoFill }
    private var grid: Float { Float(gridMM / 1000) }

    /// the cut box in scan coordinates (centre, half size), from the sliders
    private var cutBox: (SIMD3<Float>, SIMD3<Float>) {
        let b = m.bounds
        let lo = b.lo + b.size * SIMD3(Float(boxX.lowerBound), Float(cutLow), Float(boxZ.lowerBound))
        let hi = b.lo + b.size * SIMD3(Float(boxX.upperBound), Float(cutHigh), Float(boxZ.upperBound))
        return ((lo + hi) / 2, simd_max((hi - lo) / 2, SIMD3(repeating: 0.001)))
    }
    private var cutActive: Bool { tool == 1 && !toolsHidden && m.loaded && (cutLow > 0.001 || cutHigh < 0.999 || boxX != 0...1 || boxZ != 0...1 || abs(cutYaw) > 0.01) }
    private var gradeActive: Bool { tool == 2 && !toolsHidden && (abs(bright) > 0.001 || abs(contrast - 1) > 0.001 || abs(sat - 1) > 0.001) }

    var body: some View {
        GeometryReader { geo in
            let landscape = geo.size.width > geo.size.height
            ZStack(alignment: .bottom) {
                PointCloudView(cloud: m.shown, stamp: m.stamp, pointSize: pointSize, background: background, mode: colourMode, solid: style == "solid", grid: Double(gridMM / 1000),
                               onTap: (picking || tool == 4) ? { p in tapped(p) } : nil, marker: picking ? pick.map { ($0, Float(fillRadius)) } : nil,
                               lines: tool == 4 ? measure : (fillMode == 2 ? stroke : []), closed: closeShape, pins: m.item.pins ?? [], section: tool == 1 && !toolsHidden ? section : 1,   // 5.7 (43): Section only hides while cutting - other tools edit everything, so they show everything
                               
                               onStroke: fillMode == 2 ? { p in if stroke.last.map({ simd_length($0 - p) > Float(fillRadius) * 0.25 }) ?? true { stroke.append(p) } } : nil,
                               cut: cutActive ? CutPreview(centre: cutBox.0, half: cutBox.1, yaw: Float(cutYaw * .pi / 180), keepInside: cutKeep) : nil,
                               grade: gradeActive ? SIMD3(Float(bright), Float(contrast), Float(sat)) : nil,
                               mesh: style == "solid" && m.meshFor == m.stamp && m.aiFill.count == 0 ? m.mesh : nil,     // a surface from before the last edit is never shown; 5.7 (48): nor while yellow AI fill waits (the surface hid it)
                               splats: style == "splats",
                               photos: style == "solid" ? m.realPhotos : [], meshCentre: m.meshCentre)
                    .ignoresSafeArea()
                    .onChange(of: style) { _, v in if v == "solid" { m.buildMesh() } }
                    .onChange(of: m.stamp) { _, _ in if style == "solid" && m.meshStamp != m.stamp && m.busy == nil { m.buildMesh() } }
                    .onChange(of: m.loaded) { _, l in if l && style == "solid" { m.buildMesh() } }
                VStack(spacing: 0) {
                    topBar
                    Spacer(minLength: 0)
                    HStack(alignment: .bottom, spacing: 0) {
                        if landscape { Spacer(minLength: 0) }
                        VStack(spacing: 6) {
                            if let b = m.busy {
                                VStack(spacing: 6) {
                                    HStack(spacing: 8) { if m.progress == nil { ProgressView() }; Text(b).font(.footnote).lineLimit(2); if let f = m.progress { Text("\(Int(f * 100))%").font(.footnote.monospacedDigit().bold()) } }
                                    if let f = m.progress { ProgressView(value: f).tint(.yellow).frame(width: 220) }
                                }
                                .padding(10).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                            }
                            else if style == "solid" && !m.meshNote.isEmpty { HStack(spacing: 8) { ProgressView(); Text(m.meshNote).font(.footnote).lineLimit(1) }.padding(8).background(.ultraThinMaterial, in: Capsule()) }
                            if m.aiFill.count > 0 && m.busy == nil {
                                HStack(spacing: 10) {
                                    Button { m.keepFill() } label: { Label("Keep & colour", systemImage: "checkmark").lineLimit(1) }.buttonStyle(.borderedProminent).tint(.yellow).foregroundStyle(.black)
                                    Button { m.removeFill() } label: { Label("Remove AI parts", systemImage: "xmark").lineLimit(1) }.buttonStyle(.bordered)
                                }
                            }
                            if m.busy == nil && !m.message.isEmpty && !toolsHidden {
                                Text(m.message).font(.footnote).lineLimit(3).padding(.horizontal, 12).padding(.vertical, 6).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                                    .onTapGesture { m.message = "" }
                            }
                            if toolsHidden {
                                Button { withAnimation { toolsHidden = false } } label: { Label("Tools", systemImage: "slider.horizontal.3").font(.subheadline.weight(.semibold)).padding(.horizontal, 14).padding(.vertical, 9).background(.ultraThinMaterial, in: Capsule()) }
                                    .padding(.bottom, 10)
                            } else {
                                panel.frame(maxHeight: landscape ? geo.size.height - 60 : geo.size.height * 0.46)
                            }
                        }
                        .frame(width: landscape ? min(360, geo.size.width * 0.44) : geo.size.width)
                    }
                }
                .frame(width: geo.size.width)
            }
        }
        .background(Color.black)
        .onAppear {
            if !m.loaded { m.load() }
            // 5.0: photoreal is the look for scans with photos (switched once; your later choice is kept)
            if !UserDefaults.standard.bool(forKey: "photorealDefault5") { UserDefaults.standard.set(true, forKey: "photorealDefault5"); style = "solid" }
        }
        .confirmationDialog("Fill missing parts", isPresented: $askFill, titleVisibility: .visible) {
            Button("Quick fill - 4x denser and smoother, offline") { m.quickFill() }
            Button("Whole scan, automatically") { m.fillGaps() }
            Button("Tap where to fill") { tool = 0; fillMode = 1; picking = true; pick = nil }
            Button("Draw where to fill") { tool = 0; fillMode = 2; stroke = [] }
        }
        .onChange(of: m.loaded) { _, l in if l && autoFill { m.fillGaps() } }
        .sheet(item: $share) { ShareSheet(items: [$0]) }
        .fullScreenCover(isPresented: $showAR) { ARSceneView(item: m.item) }
        .fullScreenCover(isPresented: $showMesh) { MeshSheet(item: m.item, cloud: m.cloud) }
        .fullScreenCover(isPresented: $showSplat) { SplatSheet(item: m.item, cloud: m.cloud) }
        .fullScreenCover(isPresented: $showAran) { AranView(item: m.item, cloud: m.cloud) }
        .fullScreenCover(isPresented: $showContinue) { ContentView(kind: nil, continuing: m.item) { _ in showContinue = false; DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { m.load() } } }
        .sheet(isPresented: $showPlan) { CloudPlanSheet(cloud: m.cloud, name: m.item.name) }
        .sheet(isPresented: $showModel) { ModelSheet(item: m.item).presentationDetents([.medium]) }
        .sheet(item: $m.partPlan) { p in PartPlanSheet(plan: p, seg: m.seg, apply: { skip in m.applyParts(p, skip: skip) }, cancel: { m.partPlan = nil }).presentationDetents([.medium, .large]) }
        .sheet(item: $m.smart) { p in SmartPlanSheet(plan: p, apply: { m.applySmart(p) }, cancel: { m.smart = nil }).presentationDetents([.medium, .large]) }
        .sheet(item: $m.plan) { p in PlanSheet(plan: p, apply: { m.applyPlan(p) }, cancel: { m.plan = nil }).presentationDetents([.medium]) }
        .alert("Rename", isPresented: $renaming) {
            TextField("Name", text: $newName)
            Button("Save") { Library.shared.rename(m.item, to: newName); m.item.name = newName }
            Button("Cancel", role: .cancel) {}
        }
    }

    private func barButton(_ icon: String, disabled: Bool = false, _ go: @escaping () -> Void) -> some View {
        Button(action: go) {
            Image(systemName: icon).font(.callout.weight(.semibold)).frame(width: 32, height: 32).background(.thinMaterial, in: Circle())
        }.disabled(disabled)
    }
    private var topBar: some View {
        HStack(spacing: 6) {
            // 5.7 (48): not while an edit runs - its result landed on a closed editor and was never saved
            barButton("chevron.left", disabled: m.editing) { m.save(); dismiss() }
            VStack(alignment: .leading, spacing: 1) {
                Text(m.item.name).font(.subheadline.weight(.semibold)).lineLimit(1).truncationMode(.tail)
                Text("\(m.cloud.count.formatted()) points").font(.caption2.monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)      // the name gives way, never the buttons
            .layoutPriority(-1)
            .contentShape(Rectangle())
            .onTapGesture { newName = m.item.name; renaming = true }
            barButton(toolsHidden ? "eye.fill" : "eye") { withAnimation { toolsHidden.toggle() } }
            // 5.7 (47): tap undoes one step; long-press lists the named steps to jump back several at once
            Menu {
                ForEach(Array(m.history.enumerated()), id: \.offset) { i, l in Button(l) { m.undo(to: i) } }
            } label: {
                Image(systemName: "arrow.uturn.backward").font(.callout.weight(.semibold)).frame(width: 32, height: 32).background(.thinMaterial, in: Circle())
            } primaryAction: { m.undo() }
            .disabled(!m.canUndo || m.busy != nil)
            barButton("arrow.uturn.forward", disabled: !m.canRedo || m.busy != nil) { m.redo() }
            Menu {
                Picker("Look", selection: $style) {
                    Label("Points", systemImage: "circle.grid.3x3").tag("points")
                    Label("Splats (smooth surface)", systemImage: "circle.hexagongrid.fill").tag("splats")
                    Label(m.hasPhotos ? "Photoreal (your photos on the 3D shape)" : "Solid model", systemImage: "cube.fill").tag("solid")
                }
                // 5.7 (48): not while a job runs - its result (made from the old cloud) replaced the continued scan
                Button("Continue scanning") { m.save(); showContinue = true }.disabled(m.busy != nil)
                Button("Aran (solid model + detail splats)") { showAran = true }
                Button("Gaussian splats") { showSplat = true }
                Button("Make 3D model from points") { showMesh = true }
                Button("Floor plan") { showPlan = true }
                Button("View in AR (walk around it)") { showAR = true }
                Button("Apple textured mesh (basic)") { showModel = true }.disabled(Library.captureURL(m.item) == nil)
                // 5.7 (48): exports share `busy` - one finishing mid-edit cleared it and let a second edit start on the old cloud
                Menu("Export point cloud") {
                    Button("PLY (with colour)") { exportPLY() }
                    Button("LAS (survey / BIM)") { exportCloud("las") }
                    Button("XYZ text") { exportCloud("xyz") }
                }.disabled(m.busy != nil)
                Button(videoBusy ? "Making the video..." : "Turntable video") { turntable() }.disabled(videoBusy || m.busy != nil)
                if FileManager.default.fileExists(atPath: Library.originalURL(m.item.id).path) {
                    Button("Restore the unpolished scan") {
                        let it = m.item
                        m.run("Restoring...") { cur in
                            // 5.7 (46): an unreadable original used to replace the scan with 0 points (and Back saved that)
                            guard let d = try? Data(contentsOf: Library.originalURL(it.id)), let c = Cloud(data: d), c.count > 0 else { return (cur, "The unpolished scan could not be read - nothing was changed.") }
                            return (c, "The scan as it was before the automatic polish. Save to keep it.")
                        }
                    }
                }
                Button("Save changes") { m.save(); m.message = "Saved" }
                Button("Rename") { newName = m.item.name; renaming = true }
            } label: { Image(systemName: "ellipsis").font(.callout.weight(.semibold)).frame(width: 32, height: 32).background(.thinMaterial, in: Circle()) }
        }
        .padding(.horizontal, 10).padding(.vertical, 5)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial)
    }

    private var panel: some View {
        VStack(spacing: 8) {
            Picker("", selection: $tool) {
                Text("Clean").tag(0); Text("Cut").tag(1); Text("Colour").tag(2); Text("Measure").tag(4); Text("AI").tag(3)
            }.pickerStyle(.segmented)
            ScrollView(.vertical, showsIndicators: false) {
                Group {
                    switch tool {
                    case 0: if fillMode != 0 { fillPanel } else { cleanTools }
                    case 1: cutTools
                    case 2: colourTools
                    case 4: measureTools
                    default: aiTools
                    }
                }
                .disabled(m.busy != nil || !m.loaded)
            }
            .scrollBounceBehavior(.basedOnSize)
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
        .padding(.horizontal, 8).padding(.bottom, 6)
    }

    private func toolButton(_ title: String, _ icon: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 3) { Image(systemName: icon).font(.body); Text(title).font(.caption2).lineLimit(1).minimumScaleFactor(0.5) }
                .frame(minWidth: 0, maxWidth: .infinity, minHeight: 42)
        }
        .buttonStyle(.bordered)
    }

    /// the fill tools: tap a spot, or draw over the gap; then Fill
    private var fillPanel: some View {
        VStack(spacing: 8) {
            Text(fillMode == 1 ? (pick == nil ? "Tap the gap on the scan" : "Set the size, then Fill") : (stroke.isEmpty ? "Draw over the gap with your finger" : "Draw more, or Fill"))
                .font(.subheadline.weight(.medium))
            HStack { Text(fillMode == 2 ? "Brush" : "Size").font(.caption); Slider(value: $fillRadius, in: 0.03...1); Text(Prefs.length(Float(fillRadius * 2))).font(.caption.monospacedDigit()).frame(width: 52) }
            TextField("Describe what belongs there (optional), e.g. wooden floor, stripes left to right", text: $fillGuide, axis: .vertical)
                .lineLimit(1...2).textFieldStyle(.roundedBorder).font(.caption)
            PhotosPicker(selection: $refPicks, maxSelectionCount: 3, matching: .images) {
                Label(refImages.isEmpty ? "Add reference photos (optional)" : "\(refImages.count) reference photo\(refImages.count == 1 ? "" : "s")", systemImage: "photo.badge.plus").font(.caption)
            }
            .onChange(of: refPicks) { _, items in
                Task { var out: [UIImage] = []; for i in items { if let d = try? await i.loadTransferable(type: Data.self), let im = UIImage(data: d) { out.append(im) } }; refImages = out }
            }
            HStack {
                Button("Cancel") { fillMode = 0; picking = false; pick = nil; stroke = [] }.buttonStyle(.bordered)
                if fillMode == 2 { Button("Clear") { stroke = [] }.buttonStyle(.bordered).disabled(stroke.isEmpty) }
                Button { doFill() } label: { Label("Fill", systemImage: "wand.and.rays").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).tint(.yellow).foregroundStyle(.black).disabled(fillMode == 1 ? pick == nil : stroke.isEmpty)
            }
        }
    }
    private func doFill() {
        let r = Float(fillRadius)
        if fillMode == 1, let p = pick { m.fillRegion(centres: [p], radius: r, guide: fillGuide, refs: refImages) }
        else if fillMode == 2, !stroke.isEmpty { m.fillRegion(centres: stroke, radius: r, guide: fillGuide, refs: refImages) }
        fillMode = 0; picking = false; pick = nil; stroke = []
    }
    private var cleanTools: some View {
        VStack(spacing: 10) {
            Button { m.smartTool("cleanup") } label: { Label("Smart clean up (Gemini)", systemImage: "sparkles").frame(maxWidth: .infinity) }
                .buttonStyle(.borderedProminent).controlSize(.large)
            HStack(spacing: 8) {
                toolButton("Doubles", "square.on.square") { m.removeDoubles() }
                toolButton("Floaters", "aqi.low") { m.smartTool("floaters") }
                toolButton("Floor", "square.bottomhalf.filled") { m.smartTool("floor") }
                toolButton("Smooth", "drop") { m.smartTool("smooth") }
                toolButton("Align", "scope") { m.alignToPhotos() }.disabled(!m.hasPhotos)
                toolButton("Drift", "arrow.triangle.merge") { m.fixDrift() }.disabled(!m.hasPhotos)
            }
            if !m.hasPhotos { Text("This scan has no photos (LiDAR only): Gemini sees renders only, and Align, People, Faces and textured models need photos.").font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center) }
            if Prefs.model != "gemini-3.8-flash" {
                Toggle(isOn: $m.deep) { Label("Deep (Gemini 3.8 Flash, smarter, slower)", systemImage: "brain") }.font(.caption).tint(.purple)
            }
            Text("Gemini compares each photo with your scan, protects real parts, and shows its plan before anything changes.")
                .font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center)
            HStack(spacing: 8) {
                toolButton("Main parts", "circle.circle") { m.run("Keeping the main parts...") { c in let o = CloudOps.largestPiece(c); return (o, "Main parts kept: " + EditorModel.change(c.count, o.count)) } }
                toolButton("Level", "level") { m.run("Levelling...") { c in let (o, ok) = CloudOps.level(c); return (o, ok ? "Floor is level now" : "No floor found to level by") } }
                toolButton("Auto colour", "paintpalette") { m.run("Balancing colours...") { c in (CloudOps.autoColour(c), "Colours balanced") } }
                toolButton("Quick fill", "circle.grid.3x3.fill") { m.quickFill() }
                toolButton("AI fill", "square.dashed.inset.filled") { askFill = true }
                toolButton("People", "person.slash") { m.removePeople() }.disabled(!m.hasPhotos)
                toolButton("Faces", "eye.slash") { m.blurFaces() }.disabled(!m.hasPhotos)
                toolButton("Size", "ruler") { m.measure() }
            }
        }
    }

    private var cutTools: some View {
        VStack(spacing: 6) {
            Picker("", selection: $cutKeep) { Text("Keep inside the box").tag(true); Text("Remove inside").tag(false) }.pickerStyle(.segmented)
            Text(cutActive ? (cutKeep ? "Dark red points will be removed. Drag the sliders until the box holds just what you want." : "Dark red points inside the box will be removed.") : "Move the sliders: a yellow box appears on the scan and shows live what goes.")
                .font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center)
            row("Floor", Slider(value: $cutLow, in: 0...0.9))
            row("Top", Slider(value: $cutHigh, in: 0.1...1))
            row("Width", RangeSlider(range: $boxX))
            row("Depth", RangeSlider(range: $boxZ))
            row("Turn", Slider(value: $cutYaw, in: -90...90))
            row("Section", Slider(value: $section, in: 0.15...1))
            Text("Section only hides the top while you place the box - it removes nothing.").font(.caption2).foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Button { cut(inside: cutKeep) } label: { Label(cutKeep ? "Keep inside" : "Remove inside", systemImage: "scissors").frame(maxWidth: .infinity) }.buttonStyle(.borderedProminent).disabled(!cutActive)
                Button("Reset") { cutLow = 0; cutHigh = 1; boxX = 0...1; boxZ = 0...1; cutYaw = 0; section = 1 }.buttonStyle(.bordered)
                Button { let (c, h) = cutBox; m.generate(centre: c, half: h, yaw: Float(cutYaw * .pi / 180)) } label: { Image(systemName: "wand.and.rays") }.buttonStyle(.bordered).tint(.yellow).disabled(!cutActive)
            }
            Button { let (c, h) = cutBox; m.quickFill(centre: c, half: h, yaw: Float(cutYaw * .pi / 180)) } label: { Label("Quick fill inside the box", systemImage: "circle.grid.3x3.fill").frame(maxWidth: .infinity) }
                .buttonStyle(.bordered).disabled(!cutActive)
            Text("Section only changes the view (look into a room from above). The wand generates missing points inside the box.").font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
    }
    private func cut(inside: Bool) {
        let (c, h) = cutBox, yaw = Float(cutYaw * .pi / 180)
        m.run("Cutting...") { cl in let o = CloudOps.cropBox(cl, centre: c, half: h, yaw: yaw, keepInside: inside); return (o, EditorModel.change(cl.count, o.count)) }
        cutLow = 0; cutHigh = 1; boxX = 0...1; boxZ = 0...1; cutYaw = 0
    }

    private var colourTools: some View {
        VStack(spacing: 8) {
            row("Brightness", Slider(value: $bright, in: -0.3...0.3))
            row("Contrast", Slider(value: $contrast, in: 0.6...1.6))
            row("Saturation", Slider(value: $sat, in: 0...2))
            Text(gradeActive ? "This is a live preview. Apply keeps it." : "Slide to preview; nothing changes until Apply.").font(.caption2).foregroundStyle(.secondary)
            HStack {
                Button("From photos") { m.recolourFromPhotos() }.buttonStyle(.bordered).disabled(!m.hasPhotos)
                Button("Apply") {
                    let b = Float(bright), c2 = Float(contrast), s = Float(sat)
                    m.run("Recolouring...") { c in (CloudOps.colour(c, brightness: b, contrast: c2, saturation: s), "Colours changed") }
                    bright = 0; contrast = 1; sat = 1
                }.buttonStyle(.borderedProminent)
                Picker("", selection: $colourMode) { Text("Photo").tag("photo"); Text("Height").tag("height") }.pickerStyle(.segmented)
            }
        }
    }

    private var aiTools: some View {
        VStack(spacing: 10) {
            HStack {
                TextField("Tell Gemini what to change...", text: $instruction, axis: .vertical)
                    .lineLimit(1...3).textFieldStyle(.roundedBorder)
                Button { let t = instruction; instruction = ""; m.askParts(t) } label: { Image(systemName: "arrow.up.circle.fill").font(.title) }
                    .disabled(instruction.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            Text("e.g. \"remove the desk but keep the monitor stand\", \"fix the doubled edge on the left\", \"make the colours warmer\"")
                .font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center)
            if Prefs.model != "gemini-3.8-flash" {
                Toggle(isOn: $m.deep) { Label("Deep (Gemini 3.8 Flash)", systemImage: "brain") }.font(.caption).tint(.purple)
            }
            HStack {
                Button { m.smartTool("cleanup") } label: { Label("AI fix", systemImage: "wand.and.stars") }.buttonStyle(.bordered)
                Button("Name it with AI") {
                    m.busy = "Asking Gemini..."
                    let it = m.item
                    Task {
                        do {
                            let t = try await AITools.describe(it)
                            if let d = t.data(using: .utf8), let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any], let n = j["name"] as? String {
                                Library.shared.rename(it, to: n); m.item.name = n
                                if let s = j["description"] as? String { Library.shared.setNotes(it, s); m.message = s }
                            }
                        } catch { m.message = error.localizedDescription }
                        m.busy = nil
                    }
                }.buttonStyle(.bordered)
            }
        }
    }

    private func row<V: View>(_ t: String, _ v: V) -> some View {
        HStack { Text(t).font(.subheadline).frame(width: 84, alignment: .leading); v }
    }

    // ---- measuring and notes ----
    private func tapped(_ p: SIMD3<Float>) {
        if picking { pick = p; return }
        if noteMode { notePoint = p; noteText = ""; return }
        measure.append(p)
    }
    private var measureTools: some View {
        VStack(spacing: 8) {
            Picker("", selection: $noteMode) { Text("Measure").tag(false); Text("Add notes").tag(true) }.pickerStyle(.segmented)
            if noteMode {
                Text(notePoint == nil ? "Tap a spot on the scan to pin a note there." : "Write the note for the spot you tapped:").font(.caption).foregroundStyle(.secondary)
                if notePoint != nil {
                    HStack {
                        TextField("e.g. crack in the plaster", text: $noteText).textFieldStyle(.roundedBorder)
                        Button("Pin") {
                            guard let p = notePoint, !noteText.isEmpty else { return }
                            var pins = m.item.pins ?? []; pins.append(Pin(x: p.x, y: p.y, z: p.z, text: noteText))
                            m.item.pins = pins; Library.shared.setPins(m.item, pins); notePoint = nil; noteText = ""
                        }.buttonStyle(.borderedProminent)
                    }
                }
                if let pins = m.item.pins, !pins.isEmpty {
                    HStack { Text("\(pins.count) note\(pins.count == 1 ? "" : "s")").font(.caption); Spacer(); Button("Remove all", role: .destructive) { m.item.pins = []; Library.shared.setPins(m.item, []) }.font(.caption) }
                }
            } else {
                Text(measureText).font(.subheadline.monospacedDigit()).multilineTextAlignment(.center).frame(maxWidth: .infinity)
                HStack {
                    Button("Undo point") { if !measure.isEmpty { measure.removeLast() } }.disabled(measure.isEmpty)
                    Toggle("Closed", isOn: $closeShape).lineLimit(1)
                    Button("Clear", role: .destructive) { measure = [] }.disabled(measure.isEmpty)
                }.font(.caption).buttonStyle(.bordered)
            }
        }
    }
    private var measureText: String {
        guard !measure.isEmpty else { return "Tap points on the scan to measure between them." }
        guard measure.count > 1 else { return "Tap the next point." }
        var total: Float = 0
        for i in 1..<measure.count { total += simd_length(measure[i] - measure[i - 1]) }
        let last = simd_length(measure[measure.count - 1] - measure[measure.count - 2])
        let ys = measure.map { $0.y }
        var t = measure.count == 2 ? "Distance \(Prefs.length(last))" : "Last \(Prefs.length(last)) - total \(Prefs.length(total + (closeShape ? simd_length(measure[0] - measure[measure.count - 1]) : 0)))"
        t += "\nHeight difference \(Prefs.length((ys.max() ?? 0) - (ys.min() ?? 0)))"
        if closeShape && measure.count >= 3 {
            var n = SIMD3<Float>.zero
            for i in 0..<measure.count { n += simd_cross(measure[i], measure[(i + 1) % measure.count]) }
            let a = simd_length(n) / 2
            t += Prefs.metric ? String(format: " - area %.2f m²", a) : String(format: " - area %.1f sq ft", a * 10.7639)
        }
        return t
    }
    private func exportCloud(_ kind: String) {
        guard m.busy == nil else { return }
        let c = m.cloud, name = m.item.name
        m.busy = "Exporting..."
        Task.detached {
            let url = Export.tmp("\(name).\(kind)")
            try? (kind == "las" ? Export.las(c, to: url) : Export.xyz(c, to: url))      // 5.7 (48): streamed to the file, no whole-file buffer
            await MainActor.run { m.busy = nil; share = url }
        }
    }
    private func turntable() {
        guard m.busy == nil else { return }
        videoBusy = true; m.busy = "Making the turntable video..."
        let c = m.cloud, sol = style == "solid"
        Task.detached {
            let url = Export.turntable(c, solid: sol) { p in Task { @MainActor in if videoBusy { m.busy = "Making the turntable video... \(Int(p * 100))%" } } }
            await MainActor.run { videoBusy = false; m.busy = nil; if let u = url { share = u } else { m.message = "Could not make the video" } }
        }
    }
    private func exportPLY() {
        guard m.busy == nil else { return }
        let c = m.cloud, name = m.item.name.replacingOccurrences(of: "/", with: "-")
        m.busy = "Exporting..."
        Task.detached {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(name).ply")
            try? c.ply().write(to: url)
            await MainActor.run { m.busy = nil; share = url }
        }
    }
}

extension URL: Identifiable { public var id: String { absoluteString } }
extension AIPlan: Identifiable { var id: String { summary } }

struct PlanSheet: View {
    let plan: AIPlan
    let apply: () -> Void
    let cancel: () -> Void
    var body: some View {
        NavigationStack {
            List {
                Section { Text(plan.summary) }
                Section("Steps") {
                    if plan.remove_floor == true { Label("Remove the floor", systemImage: "square.bottomhalf.filled") }
                    if (plan.floaters ?? "light") != "none" { Label("Remove floaters (\(plan.floaters ?? "light"))", systemImage: "aqi.low") }
                    Label("Merge doubles", systemImage: "square.on.square")
                    if let s = plan.smooth, s > 0.05 { Label("Smooth \(Int(s * 100))%", systemImage: "drop") }
                    if plan.crop != nil { Label("Crop to the subject", systemImage: "crop") }
                    if let r = plan.reflections, !r.isEmpty { Label("Remove \(r.count) reflection\(r.count == 1 ? "" : "s"): " + r.compactMap { $0.what }.joined(separator: ", "), systemImage: "sparkle.magnifyingglass") }
                    if let b = plan.brightness, abs(b) > 0.01 { Label(b > 0 ? "Brighten" : "Darken", systemImage: "sun.max") }
                    if let n = plan.name { Label("Name: \(n)", systemImage: "character.cursor.ibeam") }
                }
            }
            .navigationTitle("AI plan").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel", action: cancel) }
                ToolbarItem(placement: .confirmationAction) { Button("Apply", action: apply).bold() }
            }
        }
    }
}

/// two-thumb slider for 0...1
struct RangeSlider: View {
    @Binding var range: ClosedRange<Double>
    var body: some View {
        GeometryReader { g in
            let w = g.size.width
            ZStack(alignment: .leading) {
                Capsule().fill(Color.secondary.opacity(0.3)).frame(height: 4)
                Capsule().fill(Color.accentColor).frame(width: max(0, CGFloat(range.upperBound - range.lowerBound) * w), height: 4)
                    .offset(x: CGFloat(range.lowerBound) * w)
                thumb.offset(x: CGFloat(range.lowerBound) * w - 11)
                    .gesture(DragGesture().onChanged { v in range = min(max(0, Double(v.location.x / w)), range.upperBound - 0.05)...range.upperBound })
                thumb.offset(x: CGFloat(range.upperBound) * w - 11)
                    .gesture(DragGesture().onChanged { v in range = range.lowerBound...max(min(1, Double(v.location.x / w)), range.lowerBound + 0.05) })
            }
            .frame(height: 30)
        }
        .frame(height: 30)
    }
    private var thumb: some View { Circle().fill(.white).frame(width: 22, height: 22).shadow(radius: 2) }
}

/// a cut box shown live on the scan: the points that would go are drawn dark red
struct CutPreview: Equatable { var centre: SIMD3<Float>; var half: SIMD3<Float>; var yaw: Float; var keepInside: Bool }

/// 5.0: the GPU does the previews. Cut box (keep / remove), the section (points above a height hidden) and colour
/// grading are shader uniforms, so sliding a slider never rebuilds a million-point geometry.
/// 5.7 (48) the points under the photo surface: each pushed a few millimetres further from the camera
let PUSHBACK_SHADER = """
#pragma body
float4 vp = scn_node.modelViewTransform * _geometry.position;
float dist = max(0.05, -vp.z);
vp.xyz *= (dist + 0.006) / dist;
_geometry.position = scn_node.inverseModelViewTransform * vp;
"""

let PREVIEW_SHADER = """
#pragma arguments
float4 boxLo;
float4 boxHi;
float4 boxCfg;
float4 boxCtr;
float4 grade;
#pragma body
float3 p = _geometry.position.xyz;
if (p.y > boxCfg.w) { _geometry.position = float4(0.0, 1.0e6, 0.0, 1.0); }
if (boxCfg.x > 0.5) {
    float3 d = p - boxCtr.xyz;
    float3 q = float3(boxCfg.y * d.x - boxCfg.z * d.z, d.y, boxCfg.z * d.x + boxCfg.y * d.z);
    bool inside = q.x >= boxLo.x && q.x <= boxHi.x && q.y >= boxLo.y && q.y <= boxHi.y && q.z >= boxLo.z && q.z <= boxHi.z;
    bool gone = (boxCfg.x < 1.5) ? !inside : inside;
    if (gone) { _geometry.color = float4(_geometry.color.rgb * 0.18 + float3(0.30, 0.02, 0.02), 1.0); }
}
if (grade.w > 0.5) {
    float3 v = pow(max(_geometry.color.rgb, float3(0.0)), float3(1.0 / 2.2));
    float g = dot(v, float3(0.299, 0.587, 0.114));
    v = float3(g) + (v - float3(g)) * grade.z;
    v = (v - 0.5) * grade.y + 0.5 + grade.x;
    _geometry.color = float4(pow(clamp(v, 0.0, 1.0), float3(2.2)), 1.0);
}
"""

/// SceneKit view of a Cloud; the geometry is built off the main thread and drawn at up to 1M points
struct PointCloudView: UIViewRepresentable {
    let cloud: Cloud
    let stamp: Int
    let pointSize: Double
    let background: String
    let mode: String
    var solid = false                 // points drawn as real-size patches: reads as a solid model
    var grid: Double = 0.002
    var onTap: ((SIMD3<Float>) -> Void)? = nil        // set while picking a spot: the tapped point, in scan coordinates
    var marker: (SIMD3<Float>, Float)? = nil           // the picked spot and radius, drawn as a yellow sphere
    var lines: [SIMD3<Float>] = []                     // measurement points, joined in order
    var closed = false
    var pins: [Pin] = []
    var section: Double = 1                            // 1 = all; less = points above that share of the height hidden (view only)
    var onStroke: ((SIMD3<Float>) -> Void)? = nil      // drawing: each point the finger passes over
    var cut: CutPreview? = nil                          // live cut box
    var grade: SIMD3<Float>? = nil                      // live brightness, contrast, saturation
    var mesh: SCNGeometry? = nil                        // solid view: shown instead of the points (centred like them)
    var splats = false                                  // 5.0: points drawn as Gaussian splats (a smooth continuous surface)
    var photos: [RealPhoto] = []                        // 5.0 photoreal: photos projected live onto the mesh
    var meshCentre = SIMD3<Float>.zero
    /// the mesher's output as centred SceneKit geometry with vertex colours, lit
    static func meshGeometry(_ m: MeshData, centre: SIMD3<Float>) -> SCNGeometry {
        let vs = SCNGeometrySource(vertices: m.v.map { let q = $0 - centre; return SCNVector3(q.x, q.y, q.z) })
        let ns = SCNGeometrySource(normals: m.n.map { SCNVector3($0.x, $0.y, $0.z) })
        var flat: [Float] = []; flat.reserveCapacity(m.c.count * 3); for k in m.c { flat += [linear(k.x), linear(k.y), linear(k.z)] }
        let cs = SCNGeometrySource(data: Data(bytes: flat, count: flat.count * 4), semantic: .color, vectorCount: m.c.count, usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let el = SCNGeometryElement(data: Data(bytes: m.f, count: m.f.count * 4), primitiveType: .triangles, primitiveCount: m.f.count / 3, bytesPerIndex: 4)
        let g = SCNGeometry(sources: [vs, ns, cs], elements: [el])
        let mat = SCNMaterial(); mat.lightingModel = .physicallyBased; mat.roughness.contents = 0.9; mat.metalness.contents = 0.0; mat.isDoubleSided = true
        mat.shaderModifiers = [.geometry: PREVIEW_SHADER]
        g.materials = [mat]
        return g
    }
    final class Coord: NSObject {
        var key = ""; let node = SCNNode(); let markerNode = SCNNode(); let overlay = SCNNode(); let pinNode = SCNNode(); let boxNode = SCNNode(); let meshNode = SCNNode()
        var overlayKey = "", pinKey = ""
        var centre = SIMD3<Float>.zero; var sample: [SIMD3<Float>] = []; var bounds = Cloud.Box(lo: .zero, hi: .zero)
        var plainMat: SCNMaterial?; var splatGrow: Float = 1      // splat look: the normal material, and how much bigger splats are when not every point is drawn
        var projector: PhotoProjector?; weak var projGeo: SCNGeometry?
        weak var view: SCNView?
        var onTap: ((SIMD3<Float>) -> Void)?
        var onStroke: ((SIMD3<Float>) -> Void)?
        var drawPan: UIPanGestureRecognizer?
        func nearest(_ t: CGPoint, _ v: SCNView) -> SIMD3<Float>? {
            var best: SIMD3<Float>? = nil, bd: CGFloat = 44 * 44
            for q in sample {
                let s = v.projectPoint(SCNVector3(q.x, q.y, q.z))
                guard s.z > 0, s.z < 1 else { continue }
                let dx = CGFloat(s.x) - t.x, dy = CGFloat(s.y) - t.y, d = dx * dx + dy * dy
                if d < bd { bd = d; best = q }
            }
            return best.map { $0 + centre }
        }
        /// 5.6 where the finger points ON THE SURFACE, also over a gap: the scan points round the finger, the nearest
        /// layer of them (not the wall seen through the hole), a plane through them, and the finger's line of sight hits
        /// that plane. The nearest-point snap jumped to the edges of the gap or to whatever was behind it.
        func surfacePoint(_ t: CGPoint, _ v: SCNView) -> SIMD3<Float>? {
            let n0 = v.unprojectPoint(SCNVector3(Float(t.x), Float(t.y), 0)), n1 = v.unprojectPoint(SCNVector3(Float(t.x), Float(t.y), 1))
            let o = SIMD3<Float>(Float(n0.x), Float(n0.y), Float(n0.z)), dir = simd_normalize(SIMD3<Float>(Float(n1.x), Float(n1.y), Float(n1.z)) - o)
            var near: [(Float, SIMD3<Float>)] = []
            let R: CGFloat = 70
            for q in sample {
                let s = v.projectPoint(SCNVector3(q.x, q.y, q.z))
                guard s.z > 0, s.z < 1 else { continue }
                let dx = CGFloat(s.x) - t.x, dy = CGFloat(s.y) - t.y
                if dx * dx + dy * dy < R * R { near.append((simd_dot(q - o, dir), q)) }
            }
            guard near.count >= 3 else { return nearest(t, v) }
            let dmin = near.map { $0.0 }.min()!
            let layer = near.filter { $0.0 < dmin + 0.15 }.map { $0.1 }            // the front surface only
            guard layer.count >= 5, let (n, c, _) = planeNormal(layer) else { return (layer.reduce(.zero, +) / Float(layer.count)) + centre }
            let den = simd_dot(n, dir)
            guard abs(den) > 0.05 else { return (layer.reduce(.zero, +) / Float(layer.count)) + centre }
            let tt = simd_dot(c - o, n) / den
            guard tt > 0 else { return nearest(t, v) }
            return o + dir * tt + centre
        }
        @objc func drew(_ g: UIPanGestureRecognizer) {
            guard let v = view, let cb = onStroke, let p = surfacePoint(g.location(in: v), v) else { return }
            cb(p)
        }
        @objc func tapped(_ g: UITapGestureRecognizer) {
            guard let v = view, let cb = onTap, !sample.isEmpty else { return }
            if let p = surfacePoint(g.location(in: v), v) { cb(p) }
        }
    }
    func makeCoordinator() -> Coord { Coord() }
    func makeUIView(context: Context) -> SCNView {
        let v = SCNView(frame: .zero); v.allowsCameraControl = true; v.antialiasingMode = .multisampling2X
        let scene = SCNScene(); scene.rootNode.addChildNode(context.coordinator.node); v.scene = scene
        let co = context.coordinator; co.view = v
        let sph = SCNSphere(radius: 1); let mm = SCNMaterial(); mm.diffuse.contents = UIColor.systemYellow.withAlphaComponent(0.35); mm.lightingModel = .constant; mm.isDoubleSided = true
        sph.materials = [mm]; co.markerNode.geometry = sph; co.markerNode.isHidden = true; scene.rootNode.addChildNode(co.markerNode)
        scene.rootNode.addChildNode(co.overlay); scene.rootNode.addChildNode(co.pinNode)
        // the cut box: 12 yellow edges of a unit cube, scaled and placed each update
        let cv: [SCNVector3] = [SCNVector3(-0.5, -0.5, -0.5), SCNVector3(0.5, -0.5, -0.5), SCNVector3(0.5, -0.5, 0.5), SCNVector3(-0.5, -0.5, 0.5),
                                SCNVector3(-0.5, 0.5, -0.5), SCNVector3(0.5, 0.5, -0.5), SCNVector3(0.5, 0.5, 0.5), SCNVector3(-0.5, 0.5, 0.5)]
        let ci: [UInt32] = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
        let bg = SCNGeometry(sources: [SCNGeometrySource(vertices: cv)], elements: [SCNGeometryElement(data: Data(bytes: ci, count: ci.count * 4), primitiveType: .line, primitiveCount: 12, bytesPerIndex: 4)])
        let bm = SCNMaterial(); bm.lightingModel = .constant; bm.diffuse.contents = UIColor.systemYellow; bm.emission.contents = UIColor.systemYellow; bg.materials = [bm]
        co.boxNode.geometry = bg; co.boxNode.isHidden = true; scene.rootNode.addChildNode(co.boxNode)
        scene.rootNode.addChildNode(co.meshNode); v.autoenablesDefaultLighting = true
        v.addGestureRecognizer(UITapGestureRecognizer(target: co, action: #selector(Coord.tapped(_:))))
        let dp = UIPanGestureRecognizer(target: co, action: #selector(Coord.drew(_:))); dp.maximumNumberOfTouches = 1; dp.isEnabled = false
        v.addGestureRecognizer(dp); co.drawPan = dp
        let cam = SCNNode(); cam.camera = SCNCamera(); cam.camera?.zNear = 0.01; cam.position = SCNVector3(0, 0, 2.2)
        scene.rootNode.addChildNode(cam); v.pointOfView = cam
        return v
    }
    func updateUIView(_ v: SCNView, context: Context) {
        v.backgroundColor = Prefs.bgColor(background)
        let key = "\(stamp)|\(mode)"
        let co = context.coordinator, ps = CGFloat(pointSize)
        // solid view: the mesh replaces the points
        if co.meshNode.geometry !== mesh { co.meshNode.geometry = mesh }
        // photoreal: the photos nearest the view are projected onto the mesh every frame
        if let mg = mesh, !photos.isEmpty, let mat = mg.materials.first {
            if co.projGeo !== mg { co.projector = PhotoProjector(photos: photos, material: mat, centre: meshCentre); co.projGeo = mg; v.delegate = co.projector; v.rendersContinuously = true; v.preferredFramesPerSecond = 30 }
        } else if co.projector != nil { v.delegate = nil; co.projector = nil; co.projGeo = nil; v.rendersContinuously = false }
        // 5.7 (48): the photo surface only exists where the photos' depth reached; the scan's points cover more. The points
        // stay on under it, pushed 6 mm back so the surface wins wherever it exists - its holes show the real scan, not black
        co.node.isHidden = false; co.meshNode.isHidden = mesh == nil
        if let pm = co.node.geometry?.materials.first, pm.program == nil {
            var mods = pm.shaderModifiers ?? [:]
            let want = mesh != nil ? PUSHBACK_SHADER : nil
            if mods[.geometry] != want && (mods[.geometry] == nil || mods[.geometry] == PUSHBACK_SHADER || want != nil) { mods[.geometry] = want; pm.shaderModifiers = mods }
        }
        // splat look (the cut box and colour previews use the normal points: they need its shader)
        if let g = co.node.geometry {
            let wantSplat = splats && mesh == nil && cut == nil && grade == nil && SplatLook.available
            if wantSplat {
                if g.materials.first?.program == nil { co.plainMat = g.materials.first; if let sm = SplatLook.material(radius: Float(grid) * 1.6, pxScale: SplatLook.pxScale(v)) { g.materials = [sm] } }
                else if let sm = g.materials.first { SplatLook.set(sm, radius: Float(grid) * 1.6 * co.splatGrow, pxScale: SplatLook.pxScale(v)) }
            } else if g.materials.first?.program != nil, let pm = co.plainMat { g.materials = [pm] }
        }
        // live previews: shader uniforms (local coordinates: the geometry is centred on co.centre)
        if let mat = (mesh ?? co.node.geometry)?.materials.first, mat.program == nil {
            if mat.shaderModifiers?[.geometry] == nil { var mods = mat.shaderModifiers ?? [:]; mods[.geometry] = PREVIEW_SHADER; mat.shaderModifiers = mods }
            func set(_ k: String, _ a: Float, _ b: Float, _ c: Float, _ d: Float) { mat.setValue(NSValue(scnVector4: SCNVector4(a, b, c, d)), forKey: k) }
            let b = co.bounds
            let top: Float = section < 0.999 && cloud.count > 0 ? b.lo.y + b.size.y * Float(section) - co.centre.y : 1e9
            if let c = cut {
                let ctr = c.centre - co.centre
                set("boxCtr", ctr.x, ctr.y, ctr.z, 0); set("boxLo", -c.half.x, -c.half.y, -c.half.z, 0); set("boxHi", c.half.x, c.half.y, c.half.z, 0)
                set("boxCfg", c.keepInside ? 1 : 2, cos(c.yaw), sin(c.yaw), top)
                co.boxNode.isHidden = false; co.boxNode.simdPosition = ctr; co.boxNode.simdScale = c.half * 2; co.boxNode.simdEulerAngles = SIMD3(0, c.yaw, 0)
            } else { set("boxCfg", 0, 1, 0, top); co.boxNode.isHidden = true }
            if let g = grade { set("grade", g.x, g.y, g.z, 1) } else { set("grade", 0, 1, 1, 0) }
        } else { co.boxNode.isHidden = true }
        co.onTap = onTap
        co.onStroke = onStroke
        let drawing = onStroke != nil
        if co.drawPan?.isEnabled != drawing { co.drawPan?.isEnabled = drawing; v.allowsCameraControl = !drawing }      // one finger draws; the camera waits
        // measurement overlay: white spheres at the points, yellow rods between them
        let ok = "\(lines.map { "\($0)" }.joined())|\(closed)|\(co.centre)"
        if ok != co.overlayKey {
            co.overlayKey = ok
            co.overlay.childNodes.forEach { $0.removeFromParentNode() }
            let pts = lines.map { $0 - co.centre }
            let size = max(0.004, (pts.count > 1 ? simd_length(pts[0] - pts[pts.count - 1]) : 0.5) * 0.008)
            for p in pts { let s = SCNSphere(radius: CGFloat(size * 1.8)); s.firstMaterial?.diffuse.contents = UIColor.white; s.firstMaterial?.lightingModel = .constant; let n = SCNNode(geometry: s); n.simdPosition = p; co.overlay.addChildNode(n) }
            var segs = Array(zip(pts, pts.dropFirst())); if closed && pts.count > 2 { segs.append((pts[pts.count - 1], pts[0])) }
            for (a, b) in segs {
                let l = simd_length(b - a); guard l > 1e-4 else { continue }
                let cy = SCNCylinder(radius: CGFloat(size), height: CGFloat(l)); cy.firstMaterial?.diffuse.contents = UIColor.systemYellow; cy.firstMaterial?.lightingModel = .constant
                let n = SCNNode(geometry: cy); n.simdPosition = (a + b) / 2
                n.simdOrientation = simd_quatf(from: SIMD3(0, 1, 0), to: simd_normalize(b - a))
                co.overlay.addChildNode(n)
                n.addChildNode(label(Prefs.length(l), color: .systemYellow, size: CGFloat(size * 9), y: Float(size * 3)))
            }
        }
        let pk = pins.map { "\($0.x),\($0.y),\($0.z),\($0.text)" }.joined() + "\(co.centre)"
        if pk != co.pinKey {
            co.pinKey = pk
            co.pinNode.childNodes.forEach { $0.removeFromParentNode() }
            for (i, p) in pins.enumerated() {
                let n = SCNNode(); n.simdPosition = p.p - co.centre
                let s = SCNSphere(radius: 0.012); s.firstMaterial?.diffuse.contents = UIColor.systemPink; s.firstMaterial?.lightingModel = .constant; n.addChildNode(SCNNode(geometry: s))
                n.addChildNode(label("\(i + 1). \(p.text)", color: .white, size: 0.04, y: 0.03))
                co.pinNode.addChildNode(n)
            }
        }
        if let (p, r) = marker { co.markerNode.isHidden = false; co.markerNode.simdPosition = p - co.centre; co.markerNode.simdScale = SIMD3(repeating: r) } else { co.markerNode.isHidden = true }
        if key != co.key {
            co.key = key
            let c = cloud
            let md = mode, sol = solid, gr = grid, wantSplats = splats && mesh == nil && cut == nil && grade == nil && SplatLook.available
            DispatchQueue.global(qos: .userInitiated).async {
                let g = c.geometry(maxPoints: 1_000_000, mode: md, size: ps)
                let bb = c.bounds, ctr = c.count > 0 ? bb.centre : .zero
                let step = max(1, c.count / 40_000)
                let smp = stride(from: 0, to: c.count, by: step).map { c.p[$0] - ctr }
                DispatchQueue.main.async {
                    guard co.key == key else { return }
                    co.node.geometry = g?.0; co.centre = ctr; co.sample = smp; co.bounds = bb; co.plainMat = nil
                    co.splatGrow = Float(max(1, c.count / 1_000_000)).squareRoot()
                    if wantSplats, let geo = co.node.geometry, let vv = co.view, let sm = SplatLook.material(radius: Float(gr) * 1.6 * co.splatGrow, pxScale: SplatLook.pxScale(vv)) { co.plainMat = geo.materials.first; geo.materials = [sm] }
                    PointCloudView.style(co.node.geometry, solid: sol, ps: ps, grid: gr)
                }
            }
        }
        PointCloudView.style(co.node.geometry, solid: solid, ps: ps, grid: grid)
    }
    static func style(_ g: SCNGeometry?, solid: Bool, ps: CGFloat, grid: Double) {
        guard let e = g?.elements.first else { return }
        if solid { e.pointSize = CGFloat(max(0.003, grid * 2.6)); e.minimumPointScreenSpaceRadius = 1; e.maximumPointScreenSpaceRadius = 80 }
        else { e.pointSize = ps; e.minimumPointScreenSpaceRadius = 1; e.maximumPointScreenSpaceRadius = ps * 1.5 }
    }
}

/// Gemini's plan in words, each step can be switched off
struct PartPlanSheet: View {
    let plan: PartPlan
    let seg: Segments?
    let apply: (Set<Int>) -> Void
    let cancel: () -> Void
    @State private var off: Set<Int> = []
    var body: some View {
        NavigationStack {
            List {
                Section { Text(plan.summary) }
                Section("Gemini's plan - tap a step to switch it off") {
                    ForEach(Array(plan.steps.enumerated()), id: \.offset) { i, s in
                        Button { if off.contains(i) { off.remove(i) } else { off.insert(i) } } label: {
                            HStack {
                                Image(systemName: off.contains(i) ? "circle" : "checkmark.circle.fill").foregroundStyle(off.contains(i) ? Color.secondary : (s.op == "protect" ? Color.green : Color.accentColor))
                                Label(SceneAI.title(s, seg), systemImage: SceneAI.icon(s.op)).foregroundStyle(off.contains(i) ? Color.secondary : Color.primary)
                            }
                        }
                    }
                }
                Section { Text("Parts are measured on the scan itself, so each step hits exactly the part named. Protected parts survive every step. You can undo afterwards.").font(.caption).foregroundStyle(.secondary) }
            }
            .navigationTitle("Gemini's plan").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel", action: cancel) }
                ToolbarItem(placement: .confirmationAction) { Button("Apply") { apply(off) }.bold().disabled(off.count == plan.steps.count) }
            }
        }
    }
}

struct SmartPlanSheet: View {
    let plan: SmartPlan
    let apply: () -> Void
    let cancel: () -> Void
    static func icon(_ op: String) -> String {
        ["protect": "lock.shield", "remove": "trash", "keep_only": "scope", "doubles": "square.on.square", "floaters": "aqi.low", "floor": "square.bottomhalf.filled",
         "smooth": "drop", "reflection": "sparkle.magnifyingglass", "brightness": "sun.max", "auto_colour": "paintpalette", "level": "level"][op] ?? "wand.and.stars"
    }
    static func title(_ s: SmartPlan.Step) -> String {
        let base = ["protect": "Protect", "remove": "Remove", "keep_only": "Keep only", "doubles": "Remove doubles", "floaters": "Remove stray points",
                    "floor": "Remove the floor", "smooth": "Smooth", "reflection": "Remove reflection", "brightness": "Brightness", "auto_colour": "Balance colours", "level": "Level"][s.op] ?? s.op
        var t = base
        if let w = s.what, !w.isEmpty { t += ": " + w }
        if let st = s.strength { t += " (\(st))" }
        return t
    }
    var body: some View {
        NavigationStack {
            List {
                Section { Text(plan.summary) }
                Section("Gemini's plan") {
                    ForEach(Array(plan.steps.enumerated()), id: \.offset) { _, s in
                        Label(SmartPlanSheet.title(s), systemImage: SmartPlanSheet.icon(s.op)).foregroundStyle(s.op == "protect" ? Color.green : Color.primary)
                    }
                }
                Section { Text("Protected parts are never removed by any step. You can undo afterwards.").font(.caption).foregroundStyle(.secondary) }
            }
            .navigationTitle("Gemini's plan").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel", action: cancel) }
                ToolbarItem(placement: .confirmationAction) { Button("Apply", action: apply).bold() }
            }
        }
    }
}

/// builds a textured model from a saved scan's photos + LiDAR depth, on the phone
struct ModelSheet: View {
    let item: ScanItem
    @StateObject private var b = ModelBuilder()
    @State private var preview: URL?
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Image(systemName: b.result == nil ? "cube.transparent" : "checkmark.circle.fill").font(.system(size: 48)).foregroundStyle(b.result == nil ? Color.primary : Color.green)
                Text(b.status.isEmpty ? "Uses the scan's sharpest full-resolution photos and their LiDAR depth." : b.status).multilineTextAlignment(.center).font(.subheadline)
                if b.running { ProgressView(value: b.progress).frame(width: 240) }
                if let r = b.result { Button("View model") { preview = Library.modelURL(r.id) }.buttonStyle(.borderedProminent) }
                else if !b.running {
                    Button("Build") { if let cap = Library.captureURL(item) { b.buildFromCapture(cap, name: item.name + " model", masking: false) } }.buttonStyle(.borderedProminent)
                }
            }
            .padding()
            .navigationTitle("Textured model").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } } }
            .fullScreenCover(item: $preview) { ModelPreview(url: $0) }
        }
    }
}

struct CloudPlanSheet: View {
    let cloud: Cloud
    let name: String
    @Environment(\.dismiss) private var dismiss
    @State private var img: UIImage?
    @State private var done = false
    @State private var share: URL?
    var body: some View {
        NavigationStack {
            Group {
                if let i = img { ScrollView([.horizontal, .vertical]) { Image(uiImage: i).resizable().scaledToFit().frame(width: 380) } }
                else if done { Text("Not enough wall points for a plan. Scan the walls at chest height.").multilineTextAlignment(.center).padding() }
                else { ProgressView("Drawing the floor plan...") }
            }
            .navigationTitle("Floor plan").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
                if let i = img { ToolbarItem(placement: .confirmationAction) { Menu {
                    Button("Share image") { if let d = i.pngData() { let u = Export.tmp("\(name) plan.png"); try? d.write(to: u); share = u } }
                    Button("Share as PDF") { let u = Export.tmp("\(name) plan.pdf"); try? Export.pdf(i, title: "\(name) - floor plan").write(to: u); share = u }
                } label: { Image(systemName: "square.and.arrow.up") } } }
            }
            .sheet(item: $share) { ShareSheet(items: [$0]) }
            .task { let c = cloud; img = await Task.detached { CloudPlan.image(c) }.value; done = true }
        }
    }
}
