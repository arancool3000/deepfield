import SwiftUI
import UIKit
import SceneKit
import Metal
import os

/// One saved item: a point cloud (from a scan or a merge) or a 3D model (USDZ from Pro Object Scan / Photos to 3D).
struct Pin: Codable, Hashable { var x: Float; var y: Float; var z: Float; var text: String; var p: SIMD3<Float> { SIMD3(x, y, z) } }

struct ScanItem: Codable, Identifiable, Hashable {
    var id = UUID()
    var name: String
    var date = Date()
    var count = 0                    // points (cloud) or 0 (model)
    var kind = "cloud"               // "cloud" | "model"
    var capture: String?             // folder under Documents/captures with the keyframe photos (for AI + the Mac)
    var notes: String?
    var pins: [Pin]?                 // notes pinned to spots on the scan
    var polished: Bool?              // 5.0: the automatic after-save polish has run
    var deleted: Date?               // 5.7 (47): when it went to Recently deleted (only set on items in trash)
    var photoSet: String?            // 5.7 (51): a Pro / photo model's own photos (folder under Documents/objects), kept with it
    var isModel: Bool { kind == "model" }
    var isWalk: Bool { kind == "walk" }              // 5.6: a Street View walk (spheres), no points
}

@MainActor
final class Library: ObservableObject {
    static let shared = Library()
    @Published private(set) var items: [ScanItem] = []
    /// 5.7 (47): deleted scans, newest first, folders and photos untouched: a delete can be taken back for 30 days.
    /// Kept apart from `items` so everything that lists, merges or polishes scans never sees them
    @Published private(set) var trash: [ScanItem] = []
    /// 5.6: scans the finishing screen is working on. The after-save polish waits for it: both ran at once on the same
    /// scan, and the polish (started from the ORIGINAL) then overwrote the finished scan - clean-up and fill lost
    var finishing: Set<UUID> = []
    @Published var polishing: Set<UUID> = [] { didSet { Awake.set("polish", !polishing.isEmpty) } }          // 5.0: scans being polished in the background
    /// 5.7 (46): bumped on main by every update(), so the polish can tell a scan changed even while its points write
    /// still waits on io (points.bin's date only moves once that write lands)
    private(set) var edits: [UUID: Int] = [:]
    /// 5.7 (48): scans open in the editor. The editor holds its own copy of the cloud and saves it whole, so a polish
    /// written under it was replaced by the next edit (and never ran again). Such a polish waits and runs on close
    var editing: Set<UUID> = []
    var polishAfterEdit: Set<UUID> = []

    nonisolated static var root: URL {
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("library", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    nonisolated static func folder(_ id: UUID) -> URL {
        let d = root.appendingPathComponent(id.uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    nonisolated static func pointsURL(_ id: UUID) -> URL { folder(id).appendingPathComponent("points.bin") }
    nonisolated static func modelURL(_ id: UUID) -> URL { folder(id).appendingPathComponent("model.usdz") }
    nonisolated static func thumbURL(_ id: UUID) -> URL { folder(id).appendingPathComponent("thumb.jpg") }
    nonisolated static func walkURL(_ id: UUID) -> URL { folder(id).appendingPathComponent("walk", isDirectory: true) }
    nonisolated static func originalURL(_ id: UUID) -> URL { folder(id).appendingPathComponent("points.unpolished.bin") }
    nonisolated static func worldMapURL(_ id: UUID) -> URL { folder(id).appendingPathComponent("room.arworldmap") }
    nonisolated static func captureURL(_ item: ScanItem) -> URL? {
        guard let c = item.capture else { return nil }
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("captures").appendingPathComponent(c)
    }
    private var indexURL: URL { Library.root.appendingPathComponent("index.json") }
    private var trashURL: URL { Library.root.appendingPathComponent("trash.json") }

    /// 5.6: ONE queue writes the points files, in order: two edits in a row wrote the same temp file at the same time
    /// and the older one could land last
    nonisolated static let io = DispatchQueue(label: "library.io", qos: .userInitiated)

    init() {
        if let d = try? Data(contentsOf: indexURL), let v = try? JSONDecoder().decode([ScanItem].self, from: d) { items = v }
        if let d = try? Data(contentsOf: trashURL), let v = try? JSONDecoder().decode([ScanItem].self, from: d) {
            let live = Set(items.map { $0.id })
            trash = v.filter { !live.contains($0.id) }
        }
        // 5.7 (47): only scans the user deleted more than 30 days ago go for good (no date: kept)
        let expired = Date().addingTimeInterval(-30 * 86400)
        for it in trash where (it.deleted ?? Date()) < expired { purge(it) }
        // 5.7 (47): a trashed scan's photos count as used, so the sweep never takes them while it can still come back
        let used = Set((items + trash).compactMap { $0.capture } + (items + trash).compactMap { $0.photoSet })
        DispatchQueue.global(qos: .background).asyncAfter(deadline: .now() + 20) { Library.sweep(used: used) }
        // models saved before 5.6 get their picture once
        let bare = items.filter { $0.isModel && !FileManager.default.fileExists(atPath: Library.root.appendingPathComponent($0.id.uuidString).appendingPathComponent("thumb.jpg").path) }.map { $0.id }
        if !bare.isEmpty {
            Library.io.asyncAfter(deadline: .now() + 3) {
                for id in bare { autoreleasepool { Library.modelThumb(id) } }
                DispatchQueue.main.async { Library.shared.objectWillChange.send() }
            }
        }
    }

    /// 5.6: photo folders nothing uses any more (a deleted scan's photos and depth, a Pro object's photos after its model
    /// was built, an abandoned capture) - they stayed for ever, hundreds of MB each. Only folders untouched for 2 days.
    nonisolated static func sweep(used: Set<String>) {
        let fm = FileManager.default, docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0], old = Date().addingTimeInterval(-2 * 86400)
        var freed: Int64 = 0
        // 5.7 (51): a Pro object / Pro room model's photos stay as long as the model does (they were swept after 2 days, so
        // no model could ever be rebuilt - at full detail on the iMac, or with a better pipeline later)
        for (dir, keepUsed) in [("captures", true), ("objects", true)] {
            let base = docs.appendingPathComponent(dir)
            for name in (try? fm.contentsOfDirectory(atPath: base.path)) ?? [] {
                if keepUsed && used.contains(name) { continue }
                let u = base.appendingPathComponent(name)
                guard let m = (try? u.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate, m < old else { continue }
                freed += Storage.size(u); try? fm.removeItem(at: u)
            }
        }
        if freed > 0 { Log.add("swept \(freed / 1_000_000) MB of unused photos") }
    }
    private func persist() {
        if let d = try? JSONEncoder().encode(items) { try? d.write(to: indexURL, options: .atomic) }
    }
    private func persistTrash() {
        if let d = try? JSONEncoder().encode(trash) { try? d.write(to: trashURL, options: .atomic) }
    }

    /// writes the cloud to disk off the main thread, then lists it
    func add(_ cloud: Cloud, name: String, capture: String?, done: ((ScanItem) -> Void)? = nil) {
        var item = ScanItem(name: name, count: cloud.count, capture: capture)
        let id = item.id
        Library.io.async {
            try? cloud.write(to: Library.pointsURL(id))
            Library.writeThumb(cloud, id)
            DispatchQueue.main.async {
                item.count = cloud.count
                self.items.insert(item, at: 0); self.persist(); done?(item)
            }
        }
    }
    func addModel(from url: URL, name: String, photoSet: String? = nil) -> ScanItem? {
        var item = ScanItem(name: name, kind: "model"); item.photoSet = photoSet
        do {
            let dst = Library.modelURL(item.id)
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.copyItem(at: url, to: dst)
        } catch { return nil }
        items.insert(item, at: 0); persist()
        let id = item.id
        Library.io.async { Library.modelThumb(id); DispatchQueue.main.async { self.objectWillChange.send() } }
        return item
    }

    /// 5.6: a picture of a model for its library card (models showed only a cube icon)
    nonisolated static func modelThumb(_ id: UUID) {
        guard let scene = try? SCNScene(url: modelURL(id), options: nil), let dev = MTLCreateSystemDefaultDevice() else { return }
        let (lo, hi) = scene.rootNode.boundingBox
        let a = SIMD3<Float>(Float(lo.x), Float(lo.y), Float(lo.z)), b = SIMD3<Float>(Float(hi.x), Float(hi.y), Float(hi.z))
        let c = (a + b) / 2, r = max(0.01, simd_length(b - a) / 2)
        guard r.isFinite, r < 1000 else { return }
        let cam = SCNNode(); cam.camera = SCNCamera(); cam.camera?.fieldOfView = 40; cam.camera?.zNear = Double(r) * 0.01; cam.camera?.zFar = Double(r) * 20
        cam.simdPosition = c + SIMD3(r * 1.3, r * 0.9, r * 2.2); cam.simdLook(at: c)
        scene.rootNode.addChildNode(cam)
        let ren = SCNRenderer(device: dev, options: nil)
        ren.scene = scene; ren.pointOfView = cam; ren.autoenablesDefaultLighting = true
        scene.background.contents = UIColor(white: 0.1, alpha: 1)
        let img = ren.snapshot(atTime: 0, with: CGSize(width: 320, height: 320), antialiasingMode: .multisampling4X)
        try? img.jpegData(compressionQuality: 0.8)?.write(to: thumbURL(id))
        Thumbs.forget(id)
    }
    /// 5.6: a walk's spheres and their positions; the first sphere's middle is the card picture
    func addWalk(from dir: URL, name: String, spots: Int) -> ScanItem? {
        var item = ScanItem(name: name, kind: "walk"); item.count = spots
        do { try FileManager.default.moveItem(at: dir, to: Library.walkURL(item.id)) } catch { return nil }
        if let img = UIImage(contentsOfFile: Library.walkURL(item.id).appendingPathComponent("spot000.jpg").path)?.cgImage {
            let h = img.height / 2, crop = img.cropping(to: CGRect(x: img.width / 2 - h / 2, y: img.height / 2 - h / 2, width: h, height: h))
            if let c = crop { try? UIImage(cgImage: c).preparingThumbnail(of: CGSize(width: 320, height: 320))?.jpegData(compressionQuality: 0.8)?.write(to: Library.thumbURL(item.id)) }
        }
        items.insert(item, at: 0); persist()
        return item
    }

    /// replaces a cloud's points (after editing)
    func update(_ item: ScanItem, cloud: Cloud) {
        guard let i = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[i].count = cloud.count
        let id = item.id
        edits[id, default: 0] += 1
        Library.io.async {
            // 5.7 (46): deleted while this waited - pointsURL/thumbURL would recreate its folder as an orphan
            guard FileManager.default.fileExists(atPath: Library.root.appendingPathComponent(id.uuidString).path) else { return }
            let tmp = Library.pointsURL(id).appendingPathExtension(UUID().uuidString.prefix(8) + ".tmp")
            if (try? cloud.write(to: tmp)) != nil { _ = try? FileManager.default.replaceItemAt(Library.pointsURL(id), withItemAt: tmp) }
            try? FileManager.default.removeItem(at: tmp)          // a failed write or replace leaves nothing behind
            Library.writeThumb(cloud, id)
            DispatchQueue.main.async { self.objectWillChange.send() }
        }
        persist()
    }
    func rename(_ item: ScanItem, to name: String) {
        guard let i = items.firstIndex(where: { $0.id == item.id }), !name.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        items[i].name = name; persist()
    }
    func setNotes(_ item: ScanItem, _ s: String) {
        guard let i = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[i].notes = s; persist()
    }
    func setCapture(_ item: ScanItem, _ name: String) {
        guard let i = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[i].capture = name; persist()
    }
    func setPolished(_ item: ScanItem) {
        guard let i = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[i].polished = true; persist()
    }
    func setPins(_ item: ScanItem, _ pins: [Pin]) {
        guard let i = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[i].pins = pins; persist()
    }
    /// a scan from another app (PLY / XYZ) into the library
    func importFile(_ url: URL, done: @escaping (ScanItem?) -> Void) {
        let ok = url.startAccessingSecurityScopedResource()
        DispatchQueue.global(qos: .userInitiated).async {
            let c = Export.importCloud(url)
            if ok { url.stopAccessingSecurityScopedResource() }
            DispatchQueue.main.async {
                guard let cl = c else { done(nil); return }
                self.add(cl, name: url.deletingPathExtension().lastPathComponent, capture: nil) { done($0) }
            }
        }
    }
    /// 5.7 (47): a delete is a move to Recently deleted - nothing on disk is touched until purge()
    func delete(_ item: ScanItem) {
        guard let i = items.firstIndex(where: { $0.id == item.id }) else { return }
        var gone = items.remove(at: i)          // the stored copy (the caller's may be older: notes, pins, count)
        gone.deleted = Date()
        trash.removeAll { $0.id == gone.id }
        trash.insert(gone, at: 0)
        persist(); persistTrash()
    }
    func restore(_ item: ScanItem) {
        guard let i = trash.firstIndex(where: { $0.id == item.id }) else { return }
        var back = trash.remove(at: i)
        back.deleted = nil
        if !items.contains(where: { $0.id == back.id }) { items.insert(back, at: 0) }
        persist(); persistTrash()
    }
    /// gone for good: only from Recently deleted (Delete now, or 30 days after the delete)
    func purge(_ item: ScanItem) {
        guard trash.contains(where: { $0.id == item.id }) else { return }
        trash.removeAll { $0.id == item.id }; persistTrash()
        // 5.7 (47): restored or saved again under the same id meanwhile - its folder is live, keep it
        guard !items.contains(where: { $0.id == item.id }) else { return }
        try? FileManager.default.removeItem(at: Library.folder(item.id))
        // 5.6: its photos and depth go too, unless another scan (a merge) still uses them
        if let c = item.capture, !items.contains(where: { $0.capture == c }), !trash.contains(where: { $0.capture == c }), let u = Library.captureURL(item) {
            Library.io.async { try? FileManager.default.removeItem(at: u) }
        }
    }
    nonisolated static func load(_ item: ScanItem) -> Cloud? {
        guard let d = try? Data(contentsOf: pointsURL(item.id), options: .alwaysMapped) else { return nil }
        return Cloud(data: d)
    }

    /// a small front-on picture of the cloud: dots, far ones first
    nonisolated static func writeThumb(_ cl: Cloud, _ id: UUID) {
        guard cl.count > 0 else { return }
        let S: CGFloat = 240, b = cl.bounds
        let span = max(0.01, max(b.size.x, b.size.y))
        let pick = CloudOps.downsample(cl, to: 40000)
        let order = pick.p.indices.sorted { pick.p[$0].z < pick.p[$1].z }
        let r = UIGraphicsImageRenderer(size: CGSize(width: S, height: S))
        let img = r.image { ctx in
            UIColor(white: 0.08, alpha: 1).setFill(); ctx.fill(CGRect(x: 0, y: 0, width: S, height: S))
            for i in order {
                let q = pick.p[i], c = pick.c[i]
                let x = CGFloat((q.x - b.centre.x) / span) * S * 0.9 + S / 2
                let y = S / 2 - CGFloat((q.y - b.centre.y) / span) * S * 0.9
                UIColor(red: CGFloat(c.x) / 255, green: CGFloat(c.y) / 255, blue: CGFloat(c.z) / 255, alpha: 1).setFill()
                ctx.fill(CGRect(x: x, y: y, width: 1.6, height: 1.6))
            }
        }
        try? img.jpegData(compressionQuality: 0.8)?.write(to: thumbURL(id))
        Thumbs.forget(id)
    }
}

/// 5.6 library thumbnails, decoded once: the Home list read and decoded every one from disk on the main thread each
/// time anything in the library changed (a polish, a save), which made Home stutter
enum Thumbs {
    private static let cache = NSCache<NSString, UIImage>()
    private static var missing = Set<UUID>()
    private static let lock = NSLock()
    static func image(_ id: UUID) -> UIImage? {
        if let i = cache.object(forKey: id.uuidString as NSString) { return i }
        lock.lock(); let miss = missing.contains(id); lock.unlock()
        if miss { return nil }
        let u = Library.root.appendingPathComponent(id.uuidString).appendingPathComponent("thumb.jpg")
        guard let i = UIImage(contentsOfFile: u.path)?.preparingForDisplay() else { lock.lock(); missing.insert(id); lock.unlock(); return nil }
        cache.setObject(i, forKey: id.uuidString as NSString)
        return i
    }
    static func forget(_ id: UUID) {
        cache.removeObject(forKey: id.uuidString as NSString)
        lock.lock(); missing.remove(id); lock.unlock()
    }
}

/// 5.0: the clean-up other apps do after capture, done in the background after every save: drift fix (only if photos
/// of the same places then agree better) and measured doubles. The original is kept so it can be restored.
enum Polish {
    @MainActor static func start(_ it: ScanItem) {
        let lib = Library.shared
        guard !it.isModel, !it.isWalk, it.capture != nil, it.polished != true, !lib.polishing.contains(it.id), !lib.finishing.contains(it.id) else { return }
        if lib.editing.contains(it.id) { lib.polishAfterEdit.insert(it.id); return }        // 5.7 (48): runs when the editor closes
        lib.polishing.insert(it.id)
        let gen0 = lib.edits[it.id, default: 0]
        Task.detached(priority: .utility) {
            var note = "nothing to change", deferred = false
            Library.io.sync {}                       // 5.7 (48): start from the file once queued saves (the editor's) have landed
            if let c0 = Library.load(it), c0.count > 2000, c0.count <= 9_000_000, os_proc_available_memory() > UInt64(c0.count) * 140 {
                var c = c0, changed = false
                let (d, m1) = Drift.fix(c, item: it) { _, _ in }
                if !m1.hasPrefix("No drift") && d.count > 0 { c = d; changed = true }
                let (e, m2) = Doubles.fix(c, item: it) { _ in }
                if e.count != c.count { c = e; changed = true }
                Log.add("polish \(it.name): \(m1) | \(m2)")
                // never over a scan that changed while this ran (an edit, a finishing): its work would be lost
                // 5.7 (46): checked and written in one main-actor step against the edit counter (not points.bin's date,
                // which lags a queued edit), and not at all once the scan is deleted
                let final = c, go = changed
                let (applied, open) = await MainActor.run { () -> (Bool, Bool) in
                    let lib = Library.shared
                    // 5.7 (48): an open editor would save its pre-polish copy over this; try again once it closes
                    if go && lib.editing.contains(it.id) { return (false, true) }
                    guard go, lib.items.contains(where: { $0.id == it.id }), !lib.finishing.contains(it.id), lib.edits[it.id, default: 0] == gen0 else { return (false, false) }
                    // the original is copied on io, so it is the file as it stands once earlier queued writes have landed
                    let dir = Library.root.appendingPathComponent(it.id.uuidString)
                    Library.io.async { try? FileManager.default.copyItem(at: dir.appendingPathComponent("points.bin"), to: dir.appendingPathComponent("points.unpolished.bin")) }
                    lib.update(it, cloud: final)
                    return (true, false)
                }
                deferred = open
                if changed && !applied && !open { Log.add("polish \(it.name): skipped, the scan changed meanwhile") }
                if applied { note = "\(c0.count) -> \(c.count) points" }
            }
            let n = note, later = deferred
            await MainActor.run {
                lib.polishing.remove(it.id)
                if later { lib.polishAfterEdit.insert(it.id); Log.add("polish \(it.name): waits for the editor to close") }
                else { lib.setPolished(it); Log.add("polish done: \(n)") }
            }
        }
    }
    /// 5.7 (48): the editor closed (its save is already queued): a polish that waited for it now runs on the saved scan
    @MainActor static func editorClosed(_ id: UUID) {
        let lib = Library.shared
        lib.editing.remove(id)
        guard lib.polishAfterEdit.remove(id) != nil, let now = lib.items.first(where: { $0.id == id }) else { return }
        start(now)
    }
}
