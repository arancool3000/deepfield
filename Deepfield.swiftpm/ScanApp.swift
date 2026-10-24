import SwiftUI
import Combine
import ARKit
import SceneKit
import CoreImage
import ImageIO
import AudioToolbox
import simd
import AVFoundation
import os

/* Deepfield (was KindleHub Scan) 4.6: everything on the phone. Home + library, SUPERPRO and ULTRA modes, Gemini Live voice guide you can
   talk to (it can work the torch, pause, save, change mode), AR checkpoints and labelled rescan marks, full-resolution
   keyframe stills, auto torch, Gemini-led cleaning that compares the photos with the scan, on-device textured models.
   History: 3.3 (more points: every depth pixel, medium confidence counts, blurry frames keep geometry; 3.2 = 3.1 + thin cables: 2 mm grid, 1-hit points beside a same-colour confident point) - a point-cloud scanner: LiDAR + photos, refined on a Mac ("Studio").

   3.1 (after 3.0's mesh): the output is a coloured POINT CLOUD again (v2.1's fusion and loose limits), plus:
   - Anchored points. ARKit's scene reconstruction runs HIDDEN (no mesh is drawn or exported) only so ARKit keeps
     mesh anchors; the app also adds a plain world anchor every ANCHOR_EVERY metres. Every fused voxel lives in the
     LOCAL frame of the anchor nearest to it, and is drawn / exported at anchor.transform * local. When ARKit
     re-aligns its anchors (loop closure), the points move with them, so a surface seen twice lands on itself.
   - Export merge: all voxels are re-binned on a MERGE_VOXEL world grid (hit-weighted mean position, sharpness-
     weighted mean colour); a merged point needs >= MIN_HITS hits in total, OR (thin cables) one hit
     with sharp-photo colour that agrees with a confident point within 6 mm. Points come from LiDAR depth only;
     the hidden mesh never drops or moves a point except through its anchor pose. Duplicates from two anchors collapse.
   - Colour only from sharp frames (>= COLOUR_SHARP x the session median), weighted by how sharp the photo was;
     the colour pixel is the camera pixel at the same instant as the LiDAR sample, off depth edges, so colour
     cannot come from something in front of the surface.
   - Pausing never loses anything: the cloud is our own copy. An anchor ARKit removes keeps its last pose here
     (3.0 deleted a surface whenever ARKit dropped / replaced its mesh anchor, and ignored the replacement while
     paused, so a paused scan emptied itself within seconds).

   On the phone:
   - Live fusion: the smoothed LiDAR depth (sceneDepth if smoothed is missing), high-confidence pixels only, nearer than
     MAX_DEPTH, never a pixel on a depth edge (the "flying pixels" that smear shelf edges) and never a person (person
     segmentation, where the phone has it). Every sample is averaged into its 4 mm voxel (running mean of position and
     colour, with hit counts), and only voxels seen at least MIN_HITS times are shown or exported.
   - Only perfect frames are kept. A frame is checked BEFORE it is committed to the cloud (staged): while the phone moves
     fast (and for 0.3 s after), while tracking is limited, or when its photo is blurry (variance of the Laplacian below
     0.55 x this session's median) the frame adds nothing and is not saved as a keyframe; the area it was looking at is
     marked "needs rescan" (red) until a good frame covers it. Moving fast also buzzes and beeps (at most every 0.8 s).
   - Keyframes for the Mac: every ~8 cm or ~8 degrees of steady motion: photo (JPEG q 0.9, 1920x1440 as captured),
     depth (Float32 metres), confidence (UInt8), person mask (UInt8 0/255) and pose + intrinsics JSON, in
     Documents/captures/<date>/frames/. Studio on the Mac refines the poses and fuses the HD result.
   - Smart check: an on-device HEURISTIC quality model (hand-written rules, NOT a neural network): few hits, low LiDAR
     confidence, blurry photos, rough or doubled surfaces (plane-fit thickness of 2 cm cells), holes (empty cells with
     scanned cells on both sides along two axes) and auto-removed areas. Weak areas are tinted red/amber in the live
     view, shown on a mini map, listed with an instruction and pointed at by an arrow. Nothing leaves the phone.
   - Export capture: the capture folder + cloud.ply + quality.json, zipped and shared. Export .ply: just the cloud. */

var MAX_DEPTH: Float = 2.5              // metres
let MIN_DEPTH: Float = 0.1
let VOXEL: Float = 0.002                // 2 mm averaging grid (thin cables survive)
var MAX_VOXELS = 60_000_000            // an upper bound only; the real limit is the memory iOS lets the app use
var MIN_HITS: UInt16 = 2                // shown / exported only after this many high-confidence samples
var STRIDE = 1                          // every depth pixel (256x192 depth map) - 3.3: for 2M+ points
var EDGE_REL: Float = 0.10              // |d - neighbour| / d above this = a depth edge, skipped
var FAST_MOVE: Float = 1.3             // m/s (per mode)
var FAST_TURN: Float = 160 * Float.pi / 180   // rad/s (per mode)
let STEADY_AFTER: TimeInterval = 0.3    // resume this long after the last fast frame
let ALERT_EVERY: TimeInterval = 0.8
var BLUR_REL: Float = 0.2              // blurry = sharpness below this x the session median
var KF_MOVE: Float = 0.08               // a keyframe every 8 cm ...
var KF_TURN: Float = 8 * Float.pi / 180      // ... or 8 degrees
var MAX_KEYFRAMES = 300
var PHOTO_LONG: CGFloat = 2560          // ARKit's native frame is 1920x1440; only a bigger one is scaled down
let CELL: Float = 0.02                  // Smart check cell size
let MAP_CELL: Float = 0.05              // mini map cell size
let DARK_LUX: CGFloat = 250             // ARKit ambientIntensity (lumen); about 1000 is a well-lit room
var MERGE_VOXEL: Float = 0.002         // export / viewer merge grid (world)
let ANCHOR_EVERY: Float = 0.5           // a world anchor every 0.5 m of phone travel
let ANCHOR_NEAR = 48                    // nearest anchors considered per frame
let THIN_NEAR: Float = 0.006            // a 1-hit point is kept if a confident point is this close ...
let THIN_COLOUR: Float = 60             // ... and their photo colours agree (RGB distance)
var ULTRA_RESOLVED: ScanKind? = nil     // what ULTRA decided the last scan was
var SWITCHED_KIND: ScanKind? = nil      // 5.7 (48): the mode the voice guide switched a SUPERPRO scan to (nil = as started); finishing should use it
var JPEG_Q: CGFloat = 0.8               // photo compression (0.8 is about 30% smaller than 0.9 and looks the same)
var USE_CAMERA = true                  // false = LiDAR only: no camera colour, blur check, light warnings or people finding
var MIN_CONF = UInt8(ARConfidenceLevel.medium.rawValue)   // LiDAR mode: high confidence only
var VIDEO_PREF = 0                     // 0 full-resolution stills, 1 plain video frames, 2 ultra-wide (if ARKit offers it)
var FUSE_GAP: TimeInterval = 0.03      // seconds between fusions (0 = every frame)
var PREVIEW_GAP: TimeInterval = 1.2
var PREVIEW_SOFT = false               // 5.0: Gaussian / Aran scans draw the live preview as dense soft dots (a splat-like surface)
var ALERTS_ON = true                   // haptics + beeps (Settings)
let COLOUR_SHARP: Float = 0.6           // colour only from frames at least this x the median sharpness

@main
struct ScanApp: App {
    @Environment(\.scenePhase) private var phase
    init() { CrashWatch.launch() }
    var body: some Scene {
        WindowGroup { HomeView().preferredColorScheme(.dark) }
            .onChange(of: phase) { _, p in if p == .background { CrashWatch.background() } else if p == .active { CrashWatch.active() } }
    }
}

// ---------- the live screen ----------
struct ContentView: View {
    var kind: ScanKind? = nil
    var continuing: ScanItem? = nil                 // carrying on a saved scan (Continue scanning)
    var onDone: (ScanItem?) -> Void = { _ in }
    @StateObject private var scanner = Scanner()
    @StateObject private var guide = LiveGuide()
    @StateObject private var pilot = UltraPilot()
    @StateObject private var torch = Torch()
    @State private var saving = false
    @State private var askExit = false
    @State private var askClear = false
    @State private var shareURL: URL?
    @State private var showShare = false
    @State private var showViewer = false
    @State private var reviewing = false
    @State private var left = false              // 5.7 (48): the screen was left (Discard / X / saved): a late save must not run
    private let guideTimer = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()

    private var title: String {
        kind.map { $0.auto ? $0.title : ($0.splats ? $0.title + " scan" : ($0.quick ? $0.title + ($0 == .blast && scanner.ultraWide ? " · ultra-wide" : "") : "SUPERPRO " + $0.title)) } ?? "Scan"
    }
    private func chip(_ icon: String, _ text: String) -> some View {
        Label(text, systemImage: icon).font(.caption.monospacedDigit().weight(.semibold)).lineLimit(1)
            .padding(.horizontal, 10).padding(.vertical, 6).background(.ultraThinMaterial, in: Capsule())
    }
    @AppStorage("scanDetails") private var details = false
    private func roundButton(_ icon: String, on: Bool = false, tint: Color = .white, _ go: @escaping () -> Void) -> some View {
        Button(action: go) {
            Image(systemName: icon).font(.title3.weight(.semibold)).foregroundStyle(on ? tint : .white)
                .frame(width: 46, height: 46).background(.ultraThinMaterial, in: Circle())
        }
    }

    var body: some View {
        ZStack {
            ARViewContainer(scanner: scanner).ignoresSafeArea()
            // 5.6 like Polycam: blue = not scanned yet; it clears as you scan
            if scanner.count > 0 || scanner.running { CoverageOverlay(mask: scanner.coverMask).ignoresSafeArea().allowsHitTesting(false) }
            VStack(spacing: 10) {
                // top: close, what this is + area, torch and voice on the side
                HStack(alignment: .top, spacing: 10) {
                    roundButton("xmark") {
                        // 5.7 (48): not while a save runs (Discard deleted its photos, then the save opened the scan anyway);
                        // with nothing to save, the photos and depth already written go too (they were left orphaned on disk)
                        guard !saving else { return }
                        scanner.pause(); if scanner.count > 0 { askExit = true } else { scanner.clear(); leave(nil) }
                    }
                    VStack(alignment: .leading, spacing: 1) {
                        Text(title).font(.subheadline.bold())
                        Text(scanner.count == 0 ? "Point at a corner and tap record" : String(format: "%.1f m² scanned", scanner.areaM2)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 12).padding(.vertical, 7).background(.ultraThinMaterial, in: Capsule())
                    Spacer(minLength: 0)
                    VStack(spacing: 10) {
                        roundButton(torch.on ? "flashlight.on.fill" : "flashlight.off.fill", on: torch.on, tint: .yellow) { torch.toggle() }
                        roundButton(guide.on ? "waveform.circle.fill" : "waveform.circle", on: guide.on, tint: .green) { guide.toggle() }
                    }
                }
                if scanner.warn { Label(scanner.status, systemImage: "exclamationmark.triangle.fill").font(.subheadline.bold()).foregroundStyle(.white).padding(10).background(Color.orange.opacity(0.9), in: Capsule()) }
                if details {
                    HStack(spacing: 6) {
                        chip("circle.grid.3x3.fill", scanner.count >= 1_000_000 ? String(format: "%.1fM", Double(scanner.count) / 1e6) : scanner.count.formatted())
                        if MAX_KEYFRAMES > 0 { chip("photo", "\(scanner.keyframes)") }
                        chip("ruler", Prefs.length(scanner.medianDepth))
                        SpeedBar(speed: scanner.shownSpeed)
                        Spacer(minLength: 0)
                    }
                }
                if torch.on {
                    HStack { Image(systemName: "sun.min"); Slider(value: $torch.level, in: 0.05...1) { _ in torch.apply() }; Image(systemName: "sun.max") }
                        .padding(.horizontal, 12).padding(.vertical, 4).background(.ultraThinMaterial, in: Capsule()).frame(maxWidth: 280)
                }
                if !scanner.spacingNote.isEmpty { Label(scanner.spacingNote, systemImage: "memorychip").font(.caption.bold()).padding(.horizontal, 12).padding(.vertical, 6).background(.ultraThinMaterial, in: Capsule()) }
                if kind?.auto == true && details { Text(pilot.note).font(.caption.bold()).padding(.horizontal, 12).padding(.vertical, 6).background(.ultraThinMaterial, in: Capsule()) }
                if !scanner.checkpointText.isEmpty { Label(scanner.checkpointText, systemImage: "mappin.and.ellipse").font(.caption.bold()).padding(.horizontal, 12).padding(.vertical, 6).background(.ultraThinMaterial, in: Capsule()) }
                if scanner.count == 0 {
                    VStack(alignment: .leading, spacing: 6) {
                        if let k = kind { Text(k.hint).font(.footnote) }
                        Label("Blue means not scanned yet - keep going until it is gone", systemImage: "square.fill").font(.footnote.bold()).foregroundStyle(Color(red: 0.45, green: 0.6, blue: 1))
                        Label("Walk through the space slowly, phone in both hands", systemImage: "figure.walk").font(.footnote)
                    }
                    .padding(12).frame(maxWidth: .infinity, alignment: .leading).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                }
                if guide.on || !guide.problem.isEmpty {
                    VStack(alignment: .leading, spacing: 3) {
                        if !guide.heard.isEmpty && guide.on { Text("You: " + guide.heard).font(.caption).foregroundStyle(.secondary) }
                        Text(guide.problem.isEmpty ? guide.line : guide.problem).font(.subheadline.weight(.medium))
                            .foregroundStyle(guide.problem.isEmpty ? Color.primary : Color.orange)
                    }
                    .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                }
                if scanner.smartOn && !scanner.clusters.isEmpty { rescanList }
                Spacer()
                HStack(alignment: .bottom) {
                    if details { MiniMapView(map: scanner.map) }
                    Spacer()
                    if scanner.smartOn && !scanner.targetText.isEmpty { arrowBox }
                }
                if !scanner.running && scanner.count > 0 {
                    Text(scanner.savedCount > 0 ? "Paused - \(scanner.savedCount.formatted()) points" : "Paused").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                // bottom, like Polycam: more on the left, record/pause in the middle, Done on the right
                HStack(alignment: .center) {
                    Menu {
                        Toggle("Show details", isOn: $details)
                        Toggle("Smart check", isOn: Binding(get: { scanner.smartOn }, set: { scanner.smartUserOff = !$0; scanner.smartOn = $0 }))
                        Button("Export capture (photos + depth)") { scanner.exportCapture { url in shareURL = url; showShare = url != nil } }
                        Button("Export .ply") { scanner.export { url in shareURL = url; showShare = url != nil } }
                        Button("Start again", role: .destructive) { if scanner.count > 0 && !saving { askClear = true } }.disabled(saving)   // 5.7 (48)
                    } label: { Image(systemName: "ellipsis").font(.title3.weight(.semibold)).foregroundStyle(.white).frame(width: 52, height: 52).background(.ultraThinMaterial, in: Circle()) }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    Button { scanner.toggle() } label: {
                        ZStack {
                            Circle().stroke(.white, lineWidth: 4).frame(width: 84, height: 84)
                            if scanner.running { HStack(spacing: 8) { Capsule().fill(.white).frame(width: 8, height: 30); Capsule().fill(.white).frame(width: 8, height: 30) } }
                            else { Circle().fill(Color.red).frame(width: 66, height: 66) }
                        }
                    }
                    .accessibilityLabel(scanner.running ? "Pause" : "Record")
                    Button { scanner.pause(); reviewing = true } label: {
                        Label(saving ? "Saving" : "Done", systemImage: saving ? "hourglass" : "checkmark").font(.headline)
                            .foregroundStyle(scanner.count == 0 || saving ? Color.white.opacity(0.5) : .white)
                            .padding(.horizontal, 18).padding(.vertical, 13).background(.ultraThinMaterial, in: Capsule())
                            .overlay(Capsule().stroke(.white.opacity(0.35)))
                    }.disabled(scanner.count == 0 || saving)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                }
                .padding(.bottom, 8)
            }
            .padding()
        }
        .confirmationDialog("Clear this scan?", isPresented: $askClear, titleVisibility: .visible) {
            Button("Clear everything", role: .destructive) { if !saving { scanner.clear() } }
        }
        .confirmationDialog("Leave this scan?", isPresented: $askExit, titleVisibility: .visible) {
            Button("Save and leave") { save() }
            Button("Discard", role: .destructive) { guard !saving else { return }; scanner.clear(); leave(nil) }
            Button("Keep scanning", role: .cancel) {}
        }
        .onReceive(guideTimer) { _ in scanner.updateGuide(); pilot.tick(scanner, torch: torch, guide: guide, ultra: kind?.auto == true); if kind == .omni { ScanKind.omniPhotos() } }
        .sheet(isPresented: $showShare) { if let u = shareURL { ShareSheet(items: [u]) } }
        .fullScreenCover(isPresented: $showViewer) { ScanViewer(scanner: scanner) }
        // 5.6 (38) like Polycam: Done shows the scan and where every photo was taken, then Extend or Process
        .fullScreenCover(isPresented: $reviewing) {
            ScanReview(scanner: scanner, photos: kind.map { !$0.splats && $0 != .aran } ?? true, extending: continuing != nil) { choice in
                reviewing = false
                guard let c = choice else { return }                 // Extend: back to scanning, everything kept
                PROCESS_CHOICE = c
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { save() }
            }
        }
        .onAppear {
            Prefs.applyToScanner(); kind?.apply(); Prefs.applyPhotoPrefs(kind); Prefs.applyReach(kind); SWITCHED_KIND = nil
            if let it = continuing {
                if let d = try? Data(contentsOf: Library.worldMapURL(it.id)), let map = try? NSKeyedUnarchiver.unarchivedObject(ofClass: ARWorldMap.self, from: d) { scanner.initialMap = map }
                DispatchQueue.global(qos: .userInitiated).async { if let old = Library.load(it) { DispatchQueue.main.async { scanner.showOld(old) } } }
            }
            scanner.reconfigure(people: kind != .person && USE_CAMERA)       // the camera set-up of THIS mode (the session started before it was applied)
            guide.scanner = scanner; guide.kind = kind
            if let k = kind, !k.quick, !scanner.smartUserOff { scanner.smartOn = true }
            if kind == .ultra { MAX_KEYFRAMES = 120 }
            if kind == .omni { ScanKind.omniPhotos() }
            guide.onTool = { name, args in
                switch name {
                case "set_torch":
                    let on = args["on"] as? Bool ?? true
                    if let l = args["level"] as? Double { torch.level = max(0.1, min(1, l)) }
                    if on { torch.on = true; torch.apply() } else { torch.off() }
                    return on ? "torch on" : "torch off"
                case "pause_scan": scanner.pause(); return "paused"
                case "resume_scan": if !scanner.running { scanner.toggle() }; return "scanning"
                case "save_scan": if scanner.count > 0 { save(); return "saving" } else { return "nothing scanned yet" }
                case "smart_check": scanner.smartOn = args["on"] as? Bool ?? true; scanner.smartUserOff = !scanner.smartOn; return scanner.smartOn ? "Smart check on" : "Smart check off"
                case "set_mode":
                    if let m = args["mode"] as? String, let k = ScanKind(rawValue: m) {
                        if kind?.auto == true { pilot.set(k, scanner, guide: guide, why: "you asked") }
                        else {
                            // 5.7 (48): from the settings baseline, like UltraPilot.set (the old mode's EDGE_REL, 4K video and
                            // photo size carried over); the shown count follows a changed MIN_HITS
                            Prefs.applyToScanner(); k.apply(); Prefs.applyPhotoPrefs(k); Prefs.applyReach(k); SWITCHED_KIND = k
                            scanner.setPeopleMasking(k != .person); scanner.recountGood()
                        }
                        return "\(k.title) mode"
                    }
                    return "unknown mode"
                default: return "not available"
                }
            }
        }
        .onAppear { PROCESS_CHOICE = .standard }
        .onDisappear { torch.off(); guide.stop() }
    }

    /// 5.7 (48): every way out: the camera stops and the voice guide lets go of this screen (its tool closure kept the
    /// guide, scanner and ARSession alive, running behind Home)
    private func leave(_ it: ScanItem?) {
        left = true; torch.off(); guide.stop(); guide.onTool = nil; scanner.close(); onDone(it)
    }

    /// stops scanning, saves the cloud (and its photos folder) to the library, and opens it in the editor
    private func save() {
        guard !saving, !left, scanner.count > 0 else { return }
        Log.add("save mode=\(kind?.rawValue ?? "standard") points=\(scanner.count) photos=\(scanner.keyframes)")
        scanner.pause(); saving = true
        let cap = scanner.captureName
        DispatchQueue.global(qos: .userInitiated).async {
            let cl = scanner.finishedCloud()
            DispatchQueue.main.async {
                if kind?.auto == true { ULTRA_RESOLVED = pilot.mode }
                if let it = continuing { carryOn(it, cl, cap); return }
                let name = pilot.subject.isEmpty ? "Scan " + Date().formatted(date: .abbreviated, time: .shortened) : pilot.subject.capitalized
                Library.shared.add(cl, name: name, capture: cap) { it in
                    scanner.saveWorldMap(to: Library.worldMapURL(it.id)) { saving = false; leave(it); Polish.start(it) }
                }
            }
        }
    }

    /// Continue scanning: the new points join the saved scan. With the room map, ARKit already put them in the same
    /// place; without one (older scans) they are lined up automatically first. Doubles are dropped, photos are kept together.
    private func carryOn(_ it: ScanItem, _ new: Cloud, _ cap: String?) {
        let lined = scanner.initialMap != nil
        DispatchQueue.global(qos: .userInitiated).async {
            guard let old = Library.load(it) else { DispatchQueue.main.async { saving = false; leave(it) }; return }
            var add = new, moved = matrix_identity_float4x4
            if !lined { let (T, _) = CloudOps.align2(new, to: old) { _ in }; moved = T; for i in 0..<add.count { add.p[i] = CloudOps.apply(T, add.p[i]) } }
            let (merged, n) = CloudOps.addNew(add.sane(), to: old, dup: 0.004)
            let final = CloudOps.dedupe(merged, grid: Float(max(0.002, Prefs.mergeGridMM / 1000)))
            // the new photos join the scan's photo folder (numbered after the old ones)
            var newCapture: String? = nil
            if let c = cap {
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("captures")
                let src = docs.appendingPathComponent(c).appendingPathComponent("frames")
                if let oldCap = Library.captureURL(it) {
                    let dst = oldCap.appendingPathComponent("frames"), stamp = String(Int(Date().timeIntervalSince1970) % 100000)
                    try? FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
                    var movedNames = Set<String>()
                    for f in (try? FileManager.default.contentsOfDirectory(atPath: src.path)) ?? [] { if (try? FileManager.default.moveItem(at: src.appendingPathComponent(f), to: dst.appendingPathComponent(stamp + f))) != nil { movedNames.insert(stamp + f) } }
                    // 5.6: lined up by matching (no room map): the photos move with their points - they kept their old
                    // poses, so textures landed in the wrong place and Doubles could take real points for copies
                    // the depth-only frames come along too (they were left behind in the new folder)
                    let dsrc = docs.appendingPathComponent(c).appendingPathComponent("depth"), ddst = oldCap.appendingPathComponent("depth")
                    try? FileManager.default.createDirectory(at: ddst, withIntermediateDirectories: true)
                    for f in (try? FileManager.default.contentsOfDirectory(atPath: dsrc.path)) ?? [] { if (try? FileManager.default.moveItem(at: dsrc.appendingPathComponent(f), to: ddst.appendingPathComponent(stamp + f))) != nil { movedNames.insert(stamp + f) } }
                    if !lined { Scanner.movePoses(dst, only: movedNames, by: moved); Scanner.movePoses(ddst, only: movedNames, by: moved) }
                } else {
                    newCapture = c
                    if !lined {
                        Scanner.movePoses(docs.appendingPathComponent(c).appendingPathComponent("frames"), only: nil, by: moved)
                        Scanner.movePoses(docs.appendingPathComponent(c).appendingPathComponent("depth"), only: nil, by: moved)
                    }
                }
            }
            DispatchQueue.main.async {
                Library.shared.update(it, cloud: final)
                if let nc = newCapture { Library.shared.setCapture(it, nc) }
                Log.add("continue scan: +\(n) points (\(lined ? "room map" : "auto-aligned"))")
                // 5.7 (48): only a room-map session is in the stored cloud's frame. Auto-aligned points were moved into the
                // old frame, but this session's map is still in its own, so the next Continue would relocalize into it,
                // skip the alignment and merge doubled, shifted points. Without a map it lines up again instead.
                if lined { scanner.saveWorldMap(to: Library.worldMapURL(it.id)) { saving = false; leave(it) } }
                else { saving = false; leave(it) }
            }
        }
    }

    private var torchControl: some View {
        VStack(spacing: 8) {
            Button { torch.toggle() } label: {
                Image(systemName: torch.on ? "flashlight.on.fill" : "flashlight.off.fill").font(.title2)
                    .frame(width: 50, height: 50)
                    .foregroundStyle(torch.on ? Color.yellow : Color.white)
            }
            .background(.ultraThinMaterial, in: Circle())
            Button { guide.toggle() } label: {
                Image(systemName: guide.on ? "waveform.circle.fill" : "waveform.circle").font(.title2)
                    .frame(width: 50, height: 50).foregroundStyle(guide.on ? Color.green : Color.white)
            }
            .background(.ultraThinMaterial, in: Circle())
            if torch.on {
                Slider(value: $torch.level, in: 0.05...1) { _ in torch.apply() }
                    .frame(width: 120).rotationEffect(.degrees(-90)).frame(width: 50, height: 120)
            }
        }
    }

    private var rescanList: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Rescan these areas").font(.subheadline.bold())
                Spacer()
                Button("Skip") { scanner.skipWeak() }.font(.caption.bold()).buttonStyle(.bordered)
            }
            ForEach(scanner.clusters.prefix(3)) { c in
                Text("\(c.id + 1). \(c.hint)").font(.caption).fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var arrowBox: some View {
        VStack(spacing: 6) {
            Image(systemName: scanner.arrowOnScreen ? "scope" : "arrow.up")
                .font(.system(size: 56, weight: .heavy))
                .foregroundStyle(scanner.targetRed ? Color.red : Color.orange)
                .rotationEffect(.radians(scanner.arrowOnScreen ? 0 : scanner.arrowAngle))
                .frame(width: 80, height: 80)
            Text(scanner.targetText).font(.caption.bold()).multilineTextAlignment(.center).frame(maxWidth: 150)
        }
        .padding(10).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }
}

/// what the mini map draws (built on the scan queue, drawn on main)
struct MiniMap {
    var good: [SIMD2<Int32>] = []
    var weak: [SIMD2<Int32>] = []
    var red: [SIMD2<Int32>] = []
    var phone = SIMD2<Float>(0, 0)     // world x, z (metres)
    var heading: Float = 0             // radians, 0 = facing world -z (map up), clockwise
    var cps: [SIMD2<Float>] = []       // checkpoints still to visit (world x, z)
    var cpsDone: [SIMD2<Float>] = []
}

/// a top-down (bird's-eye) map: grey = covered well, amber = weak, red = auto-removed / needs rescan, triangle = you
struct MiniMapView: View {
    let map: MiniMap
    var body: some View {
        Canvas { ctx, size in
            let all = map.good + map.weak + map.red
            var lo = map.phone, hi = map.phone
            for c in all {
                let p = SIMD2<Float>(Float(c.x), Float(c.y)) * MAP_CELL
                lo = simd_min(lo, p); hi = simd_max(hi, p + MAP_CELL)
            }
            let span = max(0.5, max(hi.x - lo.x, hi.y - lo.y)) * 1.1
            let mid = (lo + hi) / 2
            let s = Float(min(size.width, size.height)) / span
            func pt(_ w: SIMD2<Float>) -> CGPoint {
                CGPoint(x: CGFloat((w.x - mid.x) * s) + size.width / 2, y: CGFloat((w.y - mid.y) * s) + size.height / 2)
            }
            let cw = CGFloat(MAP_CELL * s)
            func cellsPath(_ cells: [SIMD2<Int32>]) -> Path {
                var path = Path()
                for c in cells {
                    let o = pt(SIMD2<Float>(Float(c.x), Float(c.y)) * MAP_CELL)
                    path.addRect(CGRect(x: o.x, y: o.y, width: max(1, cw), height: max(1, cw)))
                }
                return path
            }
            ctx.fill(cellsPath(map.good), with: .color(Color(white: 0.75)))
            ctx.fill(cellsPath(map.weak), with: .color(Color.orange))
            ctx.fill(cellsPath(map.red), with: .color(Color.red))
            for c in map.cpsDone { let o = pt(c); ctx.fill(Path(ellipseIn: CGRect(x: o.x - 3, y: o.y - 3, width: 6, height: 6)), with: .color(.green)) }
            for c in map.cps { let o = pt(c); ctx.stroke(Path(ellipseIn: CGRect(x: o.x - 4, y: o.y - 4, width: 8, height: 8)), with: .color(.cyan), lineWidth: 2) }
            let c = pt(map.phone), h = CGFloat(map.heading), r: CGFloat = 9
            func corner(_ a: CGFloat, _ len: CGFloat) -> CGPoint { CGPoint(x: c.x + sin(h + a) * len, y: c.y - cos(h + a) * len) }
            var tri = Path()
            tri.move(to: corner(0, r)); tri.addLine(to: corner(2.5, r * 0.8)); tri.addLine(to: corner(-2.5, r * 0.8)); tri.closeSubpath()
            ctx.fill(tri, with: .color(.cyan))
            ctx.stroke(tri, with: .color(.black), lineWidth: 1)
        }
        .frame(width: 140, height: 140)
        .background(Color.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 16))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.white.opacity(0.3), lineWidth: 1))
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController { UIActivityViewController(activityItems: items, applicationActivities: nil) }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

struct ARViewContainer: UIViewRepresentable {
    let scanner: Scanner
    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView(frame: .zero)
        v.automaticallyUpdatesLighting = false
        v.antialiasingMode = .none                 // 5.0: the camera picture and 3 px dots gain nothing from it (GPU and battery)
        v.preferredFramesPerSecond = 30
        v.scene.rootNode.addChildNode(scanner.previewNode)
        v.scene.rootNode.addChildNode(scanner.markerNode)
        v.scene.rootNode.addChildNode(scanner.checkpoints.node)
        v.scene.rootNode.addChildNode(scanner.guideArrow.node)
        v.scene.rootNode.addChildNode(scanner.oldNode)
        v.scene.rootNode.addChildNode(scanner.meshOverlay.node)
        scanner.arView = v
        scanner.attach(v.session)
        return v
    }
    func updateUIView(_ v: ARSCNView, context: Context) {}
    // 5.7 (48): the scanner is the coordinator so the view's removal can stop its camera session
    func makeCoordinator() -> Scanner { scanner }
    static func dismantleUIView(_ v: ARSCNView, coordinator: Scanner) { coordinator.close(); v.session.pause() }
}

// ---------- data ----------
/// one 4 mm voxel in the local frame of one anchor: running means (Welford) of the samples that landed in it
/// 5.0 lean: 36 bytes a point (was 64). Position and its weight stay full precision; the colour (0-255, Float16 steps
/// of 1/8 at most), its weight, the variance and the sharpness are half precision - far finer than they need. The rest of
/// the app still reads and writes `p` and `c` as before.
struct Vox {
    private var px: Float, py: Float, pz: Float
    var pw: Float = 0            // 4.5: position weight so far (near and high-confidence readings count more)
    var a: Int32                 // anchor index (0 = the fixed world origin)
    private var cr: Float16, cg: Float16, cb: Float16
    private var cw16: Float16, m216: Float16, sharp16: Float16
    var hits: UInt16             // high-confidence samples
    var seen: UInt16             // medium + high samples (confidence share = hits / seen)
    init(p: SIMD3<Float>, c: SIMD3<Float>, cw: Float, a: Int32, m2: Float, sharp: Float, hits: UInt16, seen: UInt16, pw: Float = 0) {
        px = p.x; py = p.y; pz = p.z; self.pw = pw; self.a = a
        cr = Float16(c.x); cg = Float16(c.y); cb = Float16(c.z)
        cw16 = Float16(min(cw, 60000)); m216 = Float16(min(m2 * 1e6, 60000)); sharp16 = Float16(min(sharp, 60000))
        self.hits = hits; self.seen = seen
    }
    /// mean position (LOCAL to anchor `a`, metres)
    var p: SIMD3<Float> { get { SIMD3(px, py, pz) } set { px = newValue.x; py = newValue.y; pz = newValue.z } }
    /// mean colour 0...255 (weighted by photo sharpness)
    var c: SIMD3<Float> { get { SIMD3(Float(cr), Float(cg), Float(cb)) } set { cr = Float16(newValue.x); cg = Float16(newValue.y); cb = Float16(newValue.z) } }
    /// colour weight so far (0 = no sharp photo yet)
    var cw: Float { get { Float(cw16) } set { cw16 = Float16(min(newValue, 60000)) } }
    /// sum of squared deviations of position (running variance = m2 / (hits - 1))
    var m2: Float { get { Float(m216) / 1e6 } set { m216 = Float16(min(newValue * 1e6, 60000)) } }      // stored in mm2 (metres2 is too small for half precision)
    /// mean sharpness of the photos it was seen in
    var sharp: Float { get { Float(sharp16) } set { sharp16 = Float16(min(newValue, 60000)) } }
}

/// 5.0 lean: the voxel lookup as one flat open-addressing table (16 bytes a slot, ~23 a point) instead of a Dictionary
/// (~34 a point). Same use: look up, insert, clear.
struct FlatIndex {
    private var ka: [Int32] = [], kk: [Int64] = [], val: [Int32] = []
    private var cap = 0
    private(set) var count = 0
    init() {}
    /// any table size (not only powers of two): slot = (hash * size) >> 64, so the table is never more than ~1.5x the points
    @inline(__always) private func slot(_ a: Int32, _ k: Int64) -> Int {
        var h = UInt64(bitPattern: k) &+ UInt64(UInt32(bitPattern: a)) &* 0x9E3779B97F4A7C15
        h ^= h >> 30; h = h &* 0xBF58476D1CE4E5B9; h ^= h >> 27; h = h &* 0x94D049BB133111EB; h ^= h >> 31
        return Int(h.multipliedFullWidth(by: UInt64(cap)).high)
    }
    private mutating func grow(to n: Int) {
        let newCap = max(1024, n * 10 / 7 + 1)
        let oa = ka, ok = kk, ov = val
        ka = []; kk = []; val = []                                       // let the old arrays go before the new ones exist where possible
        ka = [Int32](repeating: 0, count: newCap); kk = [Int64](repeating: 0, count: newCap); val = [Int32](repeating: -1, count: newCap)
        cap = newCap; count = 0
        for i in 0..<ov.count where ov[i] >= 0 { insert(oa[i], ok[i], ov[i]) }
    }
    private mutating func insert(_ a: Int32, _ k: Int64, _ v: Int32) {
        var s = slot(a, k)
        while val[s] >= 0 { if ka[s] == a && kk[s] == k { val[s] = v; return }; s += 1; if s == cap { s = 0 } }
        ka[s] = a; kk[s] = k; val[s] = v; count += 1
    }
    mutating func reserveCapacity(_ n: Int) { if n * 10 / 7 >= cap { grow(to: n) } }
    subscript(key: VKey) -> Int32? {
        get {
            guard cap > 0 else { return nil }
            var s = slot(key.a, key.k)
            while val[s] >= 0 { if ka[s] == key.a && kk[s] == key.k { return val[s] }; s += 1; if s == cap { s = 0 } }
            return nil
        }
        set {
            guard let v = newValue else { return }                  // removal is never needed: the table is rebuilt instead
            if (count + 1) * 10 > cap * 7 { grow(to: max(1024, count + count / 3)) }      // grow by a third, not double
            insert(key.a, key.k, v)
        }
    }
    mutating func removeAll(keepingCapacity keep: Bool = false) {
        if keep && cap > 0 { for i in 0..<cap { val[i] = -1 }; count = 0 } else { self = FlatIndex() }
    }
}

/// a merged world point (viewer, export, Smart check)
struct Pt {
    var p: SIMD3<Float>
    var c: SIMD3<Float>
    var sharp: Float
    var hits: Int
    var seen: Int
}

struct VKey: Hashable { let a: Int32; let k: Int64 }

@inline(__always) func flawOf(hits: Int, seen: Int, sharp: Float, blur: Float) -> Flaw {
    if hits < 4 { return .few }
    if 2 * hits < seen { return .lowConf }
    if sharp < blur { return .blurry }
    return .ok
}
@inline(__always) func xform(_ T: simd_float4x4, _ p: SIMD3<Float>) -> SIMD3<Float> {
    let q = T * SIMD4<Float>(p.x, p.y, p.z, 1); return SIMD3<Float>(q.x, q.y, q.z)
}

/// why an area is weak (Smart check)
enum Flaw: Int {
    case ok = 0, few, lowConf, blurry, noisy, hole, removed
    var label: String {
        switch self {
        case .ok: return "ok"
        case .few: return "few hits"
        case .lowConf: return "low LiDAR confidence"
        case .blurry: return "blurry photos"
        case .noisy: return "rough or doubled surface"
        case .hole: return "hole"
        case .removed: return "auto-removed (blur or fast motion)"
        }
    }
    var key: String {
        switch self {
        case .ok: return "ok"
        case .few: return "few_hits"
        case .lowConf: return "few_hits"           // too few HIGH-confidence hits
        case .blurry: return "blur"
        case .noisy: return "variance"
        case .hole: return "hole"
        case .removed: return "removed"
        }
    }
    var red: Bool { self == .few || self == .lowConf || self == .hole || self == .removed }
    var verb: String {
        switch self {
        case .ok: return "Look at"
        case .few: return "Move closer and sweep over"
        case .lowConf: return "Move closer and face straight on to"
        case .blurry: return "Move slower over"
        case .noisy: return "Rescan slowly"
        case .hole: return "Fill the gap at"
        case .removed: return "Slowly rescan"
        }
    }
}

struct WeakCluster: Identifiable {
    let id: Int
    let centre: SIMD3<Float>
    let size: SIMD3<Float>
    let reason: Flaw
    let cells: Int
    let part: String             // top / middle / bottom of the scan
    var hint: String
}

@inline(__always) func voxKey(_ p: SIMD3<Float>, _ g: Float = VOXEL) -> Int64 {
    (ik64((p.x / g)) & 0x1FFFFF) | ((ik64((p.y / g)) & 0x1FFFFF) << 21) | ((ik64((p.z / g)) & 0x1FFFFF) << 42)
}
@inline(__always) func cellKey(_ p: SIMD3<Float>) -> SIMD3<Int32> {
    SIMD3<Int32>(ik((p.x / CELL)), ik((p.y / CELL)), ik((p.z / CELL)))
}
@inline(__always) func position(_ T: simd_float4x4) -> SIMD3<Float> { SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z) }
func rotation(_ T: simd_float4x4) -> simd_quatf {
    simd_quatf(simd_float3x3(columns: (SIMD3<Float>(T.columns.0.x, T.columns.0.y, T.columns.0.z),
                                       SIMD3<Float>(T.columns.1.x, T.columns.1.y, T.columns.1.z),
                                       SIMD3<Float>(T.columns.2.x, T.columns.2.y, T.columns.2.z))))
}
/// rotation angle (radians, 0...pi) between two poses
func turnAngle(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
    let q = rotation(a).inverse * rotation(b)
    var ang = q.angle
    if ang > Float.pi { ang = 2 * Float.pi - ang }
    return abs(ang)
}

/// sharpness of a luma plane: variance of the Laplacian on an 8x-downsampled copy (higher = sharper)
func lumaSharpness(_ yBase: UnsafeMutableRawPointer, _ yRow: Int, _ w: Int, _ h: Int) -> Float {
    let s = 8, gw = w / s, gh = h / s
    guard gw > 2, gh > 2 else { return 0 }
    var g = [Float](repeating: 0, count: gw * gh)
    for y in 0..<gh {
        let row = yBase.advanced(by: y * s * yRow).assumingMemoryBound(to: UInt8.self)
        for x in 0..<gw { g[y * gw + x] = Float(row[x * s]) }
    }
    var sum: Double = 0, sum2: Double = 0, n: Double = 0
    for y in 1..<(gh - 1) {
        for x in 1..<(gw - 1) {
            let i = y * gw + x
            let l = Double(4 * g[i] - g[i - 1] - g[i + 1] - g[i - gw] - g[i + gw])
            sum += l; sum2 += l * l; n += 1
        }
    }
    let mean = sum / n
    return Float(max(0, sum2 / n - mean * mean))
}

/// copy a CVPixelBuffer's single plane, tightly packed (no row padding)
func copyPlane(_ pb: CVPixelBuffer, bytesPerPixel: Int) -> Data? {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), row = CVPixelBufferGetBytesPerRow(pb)
    let rowBytes = w * bytesPerPixel
    guard rowBytes <= row else { return nil }
    var d = Data(capacity: rowBytes * h)
    for y in 0..<h { d.append(base.advanced(by: y * row).assumingMemoryBound(to: UInt8.self), count: rowBytes) }
    return d
}

/// smallest eigenvalue of a symmetric 3x3 matrix (closed form)
func minEigenSym3(_ a00: Double, _ a01: Double, _ a02: Double, _ a11: Double, _ a12: Double, _ a22: Double) -> Double {
    let p1 = a01 * a01 + a02 * a02 + a12 * a12
    if p1 < 1e-30 { return min(a00, min(a11, a22)) }
    let q = (a00 + a11 + a22) / 3
    let p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + 2 * p1
    let p = (p2 / 6).squareRoot()
    if p < 1e-30 { return q }
    let b00 = (a00 - q) / p, b11 = (a11 - q) / p, b22 = (a22 - q) / p, b01 = a01 / p, b02 = a02 / p, b12 = a12 / p
    let det = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)
    let r = max(-1, min(1, det / 2))
    let phi = acos(r) / 3
    return q + 2 * p * cos(phi + 2 * Double.pi / 3)
}

// ---------- the scanner ----------
@inline(__always) func mapKey(_ p: SIMD3<Float>) -> SIMD2<Int32> { SIMD2<Int32>(ik((p.x / MAP_CELL)), ik((p.z / MAP_CELL))) }

/// sharpness of a whole camera image (locks it)
func frameSharpness(_ img: CVPixelBuffer) -> Float {
    CVPixelBufferLockBaseAddress(img, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(img, .readOnly) }
    guard let yb = CVPixelBufferGetBaseAddressOfPlane(img, 0) else { return 0 }
    return lumaSharpness(yb, CVPixelBufferGetBytesPerRowOfPlane(img, 0), CVPixelBufferGetWidthOfPlane(img, 0), CVPixelBufferGetHeightOfPlane(img, 0))
}

/// the person mask resampled to w x h (nearest), 0 or 255; nil if there is no segmentation
func personMask(_ seg: CVPixelBuffer?, _ w: Int, _ h: Int) -> [UInt8]? {
    guard let seg = seg else { return nil }
    CVPixelBufferLockBaseAddress(seg, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(seg, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(seg) else { return nil }
    let sw = CVPixelBufferGetWidth(seg), sh = CVPixelBufferGetHeight(seg), row = CVPixelBufferGetBytesPerRow(seg)
    guard sw > 0, sh > 0 else { return nil }
    var out = [UInt8](repeating: 0, count: w * h)
    for v in 0..<h {
        let sv = min(sh - 1, v * sh / h)
        let r = base.advanced(by: sv * row).assumingMemoryBound(to: UInt8.self)
        for u in 0..<w where r[min(sw - 1, u * sw / w)] != 0 { out[v * w + u] = 255 }
    }
    return out
}

final class Scanner: NSObject, ObservableObject, ARSessionDelegate {
    /// 5.6: rewrites the poses of the frames named in `only` (all if nil) - a continued scan lined up by matching
    static func movePoses(_ dir: URL, only: Set<String>?, by M: simd_float4x4) {
        guard M != matrix_identity_float4x4 else { return }
        for f in (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [] where f.hasSuffix(".json") && (only?.contains(f) ?? true) {
            let u = dir.appendingPathComponent(f)
            guard var j = (try? JSONSerialization.jsonObject(with: Data(contentsOf: u))) as? [String: Any], let t = j["transform"] as? [Double], t.count == 16 else { continue }
            let a = t.map { Float($0) }
            let T = M * simd_float4x4(columns: (SIMD4(a[0], a[1], a[2], a[3]), SIMD4(a[4], a[5], a[6], a[7]), SIMD4(a[8], a[9], a[10], a[11]), SIMD4(a[12], a[13], a[14], a[15])))
            var tm: [Double] = []; for c in 0..<4 { for r in 0..<4 { tm.append(Double(T[c][r])) } }
            j["transform"] = tm
            if let d = try? JSONSerialization.data(withJSONObject: j) { try? d.write(to: u, options: .atomic) }
        }
    }
    @Published var count = 0
    var jpegRequest: ((Data) -> Void)?       // the voice guide asks for one small camera frame at a time
    var ambient: CGFloat = 1000              // ARKit light estimate (auto torch)
    var lastCompact: TimeInterval = 0
    // 4.6 even spacing: at most `densityK` stored points per 6 mm cell of space. Full detail (9 = a 2 mm grid on a
    // surface) while memory is plentiful; as it fills, new points are spaced further apart so the whole room fits.
    var cellCount: [Int64: UInt8] = [:]
    // 5.6 coverage, like Polycam: 3 cm cells the LiDAR has hit (work queue). The live view tints blue whatever is
    // not covered yet, so where to scan next is obvious
    private var cover: [Int64: UInt8] = [:], coverTick = 0, coveredCells = 0
    @Published var areaM2: Float = 0                    // surface scanned so far (3 cm cells)
    static let maskCols = 24, maskRows = 44
    @Published var coverMask: [Bool] = []               // screen cells (row-major, maskCols x maskRows): true = scanned
    private var lastCover: TimeInterval = 0
    var viewSize = CGSize(width: 390, height: 844), viewOrientation = UIInterfaceOrientation.portrait
    /// which parts of the screen show surfaces that are scanned (5 times a second, from the live LiDAR depth)
    private func updateCoverage(_ frame: ARFrame) {
        guard frame.timestamp - lastCover > 0.2, let dep = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }
        lastCover = frame.timestamp
        if let v = arView { viewSize = v.bounds.size; viewOrientation = v.window?.windowScene?.interfaceOrientation ?? .portrait }
        let dm = dep.depthMap, dw = CVPixelBufferGetWidth(dm), dh = CVPixelBufferGetHeight(dm)
        let img = frame.capturedImage, iw = Float(CVPixelBufferGetWidth(img)), ih = Float(CVPixelBufferGetHeight(img))
        let K = frame.camera.intrinsics, T = frame.camera.transform
        let tr = frame.displayTransform(for: viewOrientation, viewportSize: viewSize).inverted()
        let C = Scanner.maskCols, R = Scanner.maskRows
        var pts = [SIMD3<Float>?](repeating: nil, count: C * R)
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(dm) {
            let row = CVPixelBufferGetBytesPerRow(dm)
            for r in 0..<R { for c in 0..<C {
                let ip = CGPoint(x: (CGFloat(c) + 0.5) / CGFloat(C), y: (CGFloat(r) + 0.5) / CGFloat(R)).applying(tr)
                guard ip.x >= 0, ip.y >= 0, ip.x < 1, ip.y < 1 else { continue }
                let z = base.advanced(by: Int(ip.y * CGFloat(dh)) * row).assumingMemoryBound(to: Float32.self)[Int(ip.x * CGFloat(dw))]
                guard z > 0.1, z < 5.5 else { continue }
                let u = Float(ip.x) * iw, v = Float(ip.y) * ih
                let cp = SIMD4<Float>((u - K[2][0]) / K[0][0] * z, -(v - K[2][1]) / K[1][1] * z, -z, 1)
                let w = T * cp; pts[r * C + c] = SIMD3(w.x, w.y, w.z)
            } }
        }
        CVPixelBufferUnlockBaseAddress(dm, .readOnly)
        work.async {
            let m = pts.map { p -> Bool in guard let p else { return false }; return (self.cover[voxKey(p, 0.03)] ?? 0) >= 3 }
            let a = Float(self.coveredCells) * 0.0009
            DispatchQueue.main.async { self.coverMask = m; if abs(a - self.areaM2) > 0.05 { self.areaM2 = a } }
        }
    }
    var densityK: UInt8 = 9
    var startAvail = 0
    var lastCountShown: TimeInterval = 0
    var dynGap: TimeInterval = 0               // 5.0: pause after the last fusion pass, from how long it took
    var compactFloor = 0                     // 5.0: no compaction below this size (the last one freed almost nothing)
    var lowStreak = 0                        // 5.0: consecutive integrations with memory under the reserve
    @Published var spacingNote = ""
    var yieldRatio: Float = 0.5              // share of the LiDAR readings kept (self-tuning)
    private var lastTuneT: TimeInterval = 0
    var medianDepth: Float = 1               // how far the surfaces in view are (ULTRA tuning, guide)
    let checkpoints = Checkpoints()
    let guideArrow = GuideArrow()
    var initialMap: ARWorldMap?              // continuing a saved scan: ARKit lines up with the room as it was
    let oldNode = SCNNode()                  // the saved scan, faint, so you can see where to carry on
    func saveWorldMap(to url: URL, done: @escaping () -> Void) {
        guard let s = session else { done(); return }
        s.getCurrentWorldMap { map, _ in
            if let m = map, let d = try? NSKeyedArchiver.archivedData(withRootObject: m, requiringSecureCoding: true) { try? d.write(to: url) }
            DispatchQueue.main.async { done() }
        }
    }
    func showOld(_ cl: Cloud) {
        let s = CloudOps.downsample(cl, to: 200_000)
        let verts = s.p.map { SCNVector3($0.x, $0.y, $0.z) }
        var cols: [Float] = []; cols.reserveCapacity(s.count * 3)
        for c in s.c { cols += [Float(c.x) / 255 * 0.6 + 0.2, Float(c.y) / 255 * 0.6 + 0.25, Float(c.z) / 255 * 0.6 + 0.35] }
        oldNode.geometry = pointGeometry(verts, cols, size: 2)
        oldNode.opacity = 0.55
    }
    var ultraWide = false                   // PowerBlast: running on the ultra-wide camera
    var hiRes = false                        // keyframe photos are full-resolution stills (iOS 16+)
    @Published var checkpointText = ""
    @Published var shownSpeed: Float = 0        // the speed meter (the per-frame speed, published 4 times a second)
    @Published var savedCount = 0            // the real number of points a save will keep (worked out on pause)
    @Published var keyframes = 0
    var photoPoses: [simd_float4x4] = []      // 5.6 (38) where each photo was taken (the review shows them)
    @Published var status = "Point at the setup, then tap Scan"
    @Published var warn = false
    @Published var running = false { didSet { Awake.set(Awake.key(self, "scan"), running) } }
    deinit { Awake.set(Awake.key(self, "scan"), false) }
    @Published var clusters: [WeakCluster] = []
    @Published var arrowAngle: Double = 0
    @Published var arrowOnScreen = false
    @Published var targetText = ""
    @Published var targetRed = true
    @Published var map = MiniMap()
    var smartUserOff = false                    // 5.6: the user switched Smart check off - nothing turns it back on
    /// 5.6: weak spots the scanner cannot improve (glass, black or shiny surfaces, too far) stop being asked for: a cell
    /// still weak after 3 checks while the camera was looking at it (about 25 s), or 2 minutes after it was first
    /// flagged, or skipped by the user, is settled for the rest of the scan
    private var weakSince: [SIMD3<Int32>: TimeInterval] = [:], weakSeen: [SIMD3<Int32>: Int] = [:]
    private var settled = Set<SIMD3<Int32>>(), lastWeak = Set<SIMD3<Int32>>()
    var lookT = matrix_identity_float4x4                // the camera, for "was it looking at that spot"
    func skipWeak() { work.async { self.settled.formUnion(self.lastWeak); DispatchQueue.main.async { self.clusters = []; self.targetText = ""; self.rebuildMarkers() } } }
    @Published var smartOn = false {
        didSet {
            let on = smartOn
            work.async { self.tint = on; self.updatePreview() }
            if on { analyse() } else { clusters = []; targetText = ""; rebuildMarkers() }
        }
    }
    let previewNode = SCNNode()
    let markerNode = SCNNode()
    let meshOverlay = MeshOverlay()
    weak var arView: ARSCNView?

    private var session: ARSession?
    private var personOn = false
    private let work = DispatchQueue(label: "scan.work", qos: .userInitiated)
    private let io = DispatchQueue(label: "scan.io", qos: .utility)
    private let ciContext = CIContext(options: nil)

    // main-thread state
    private var busy = false
    private var lastIntegrate: TimeInterval = 0
    private var lastAnalysis: TimeInterval = 0
    private var lastAlert: TimeInterval = 0
    private var lastFast: TimeInterval = -10
    private var prevT = matrix_identity_float4x4
    private var prevTime: TimeInterval = 0
    private var speed: Float = 0
    private var turn: Float = 0
    private var lastKF: simd_float4x4?
    private var lastDepthKF: simd_float4x4?          // 5.0: depth-only frames, for complete model surfaces
    private var depthFrames = 0
    private var ioBusy = false
    private var captureDir: URL?
    private var sharpHist: [Float] = []
    private var mapBusy = false
    private var lastAnchorPos: SIMD3<Float>?

    // work-queue state
    private var index = FlatIndex()
    private var vox: [Vox] = []
    private var anchorT: [simd_float4x4] = [matrix_identity_float4x4]   // index 0 = the world origin, never moves
    private var anchorIdx: [UUID: Int32] = [:]
    fileprivate var anchorGen = 0                 // 5.7 (47): bumped whenever an anchor pose changes; a cached merge is only reused at the same poses
    private var undoStack: [[Vox]] = []           // 5.0: each entry is only the voxels one removal took out
    private var lastPreview: TimeInterval = 0
    private var blurThresh: Float = 0
    private var tint = true
    private var noisyCells = Set<SIMD3<Int32>>()
    private var removedCells = Set<SIMD3<Int32>>()      // auto-removed views, cleared when a good frame covers them
    private var mapGood = Set<SIMD2<Int32>>()
    private var mapWeak = Set<SIMD2<Int32>>()
    private var bottomY: Float = 0, topY: Float = 0

    func attach(_ s: ARSession) {
        session = s; s.delegate = self
        run(withPeople: true, reset: false)
    }

    private func run(withPeople: Bool, reset: Bool) {
        guard let s = session else { return }
        let c = ARWorldTrackingConfiguration()
        var sem: ARConfiguration.FrameSemantics = []
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) { sem.insert(.sceneDepth) }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) { sem.insert(.smoothedSceneDepth) }
        if sem.isEmpty { setStatus("This iPhone has no LiDAR scanner", warn: true) }
        // 5.6 (38) like Polycam: ARKit's scene mesh, drawn as a faint white wireframe over what has been scanned (the
        // blue tint covers the rest). Display only: points never ride on mesh anchors, and it can be switched off
        if Prefs.scanMesh, ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) { c.sceneReconstruction = .mesh }
        personOn = false
        if withPeople {
            var withP = sem; withP.insert(.personSegmentationWithDepth)
            if ARWorldTrackingConfiguration.supportsFrameSemantics(withP) { sem = withP; personOn = true }
        }
        c.frameSemantics = sem
        c.isLightEstimationEnabled = true
        if let m = initialMap { c.initialWorldMap = m }
        hiRes = false
        // 5.0: photo-rich modes stream 4K video and keep frames from it. Full-resolution stills made iOS play the shutter
        // sound for every photo (it cannot be silenced) and paused the camera each time.
        if VIDEO_PREF == 0, #available(iOS 16.0, *), let f = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution { c.videoFormat = f }
        if VIDEO_PREF == 2, #available(iOS 16.0, *),
           let f = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: { $0.captureDeviceType == .builtInUltraWideCamera }) { c.videoFormat = f; ultraWide = true }
        if reset { s.run(c, options: [.resetTracking, .removeExistingAnchors, .resetSceneReconstruction]) } else { s.run(c) }
    }

    /// if the phone refuses people segmentation together with depth, run again without it
    func session(_ session: ARSession, didFailWithError error: Error) {
        if personOn { run(withPeople: false, reset: false) } else { setStatus("Camera error: \(error.localizedDescription)", warn: true) }
    }

    private func setStatus(_ s: String, warn w: Bool = false) {
        if status != s { status = s }
        if warn != w { warn = w }
    }

    @Published var full = false                 // 5.0: memory is used up - scanning cannot resume until the scan is saved or cleared
    func toggle() {
        if full && !running {
            setStatus("Memory is full: tap Save to keep this scan, or the bin to start again", warn: true)
            return
        }
        running.toggle()
        // 5.7 (48): vox is work-queue state; reading it here raced integrate()'s appends (a freed buffer, or a
        // copy-on-write of the whole voxel array), so its count is read on the work queue
        let line = "\(running ? "scan" : "pause") points=\(count) photos=\(keyframes) mode-depth=\(MAX_DEPTH)"
        work.async { Log.add(line + " voxels=\(self.vox.count)") }
        setStatus(running ? "Scanning - move slowly, 30-60 cm away" : "Paused - every point is kept")
        if !running && smartOn { analyse() }
        if !running && count > 0 { savedCount = 0; measureSaved() }
    }
    func pause() { if running { toggle() } }
    /// 5.7 (48): the scan screen is gone: the camera stops (the session ran on behind Home, and Discard's clear() even
    /// restarted it); with no session, a later run() does nothing
    func close() {
        if running { running = false }
        session?.pause(); session = nil
    }
    /// merging copies of one surface (from different anchors) is what a save does; count it for real
    /// 5.7 (39) a thinned look at the scan for the review screen: straight from the voxels, no merged copy (the full
    /// merge it used was a second whole scan in memory just before saving)
    func previewPoints(max m: Int) -> ([SIMD3<Float>], [SIMD3<UInt8>]) {
        work.sync {
            var g = 0; for v in vox where v.hits >= MIN_HITS { g += 1 }
            let step = Swift.max(1, g / m)
            var p: [SIMD3<Float>] = [], c: [SIMD3<UInt8>] = []; p.reserveCapacity(g / step + 1); c.reserveCapacity(g / step + 1)
            var k = 0
            for v in vox where v.hits >= MIN_HITS { if k % step == 0 { p.append(world(v)); c.append(SIMD3(u8(v.c.x), u8(v.c.y), u8(v.c.z))) }; k += 1 }
            return (p, c)
        }
    }
    func measureSaved() { work.async { let n = self.snapshotLocked().0.count; DispatchQueue.main.async { self.savedCount = n } } }

    func clear() {
        running = false; full = false
        let old = captureDir
        lastAnchorPos = nil
        captureDir = nil; keyframes = 0; photoPoses = []; lastKF = nil; lastDepthKF = nil; depthFrames = 0; clusters = []; targetText = ""; sharpHist.removeAll(); map = MiniMap(); checkpoints.clear(); checkpointText = ""
        rebuildMarkers()
        io.async { if let o = old { try? FileManager.default.removeItem(at: o) } }
        work.async {
            self.index.removeAll(); self.vox.removeAll(); self.goodN = 0; self.goodAt = MIN_HITS; self.cellCount.removeAll(); self.densityK = 9; self.startAvail = 0; self.lowStreak = 0; self.compactFloor = 0; self.undoStack.removeAll(); self.noisyCells.removeAll(); self.settled.removeAll(); self.weakSince.removeAll(); self.weakSeen.removeAll(); self.cover.removeAll(); self.coveredCells = 0
            self.removedCells.removeAll(); self.mapGood.removeAll(); self.mapWeak.removeAll()
            snapCache = nil                                  // 5.7 (46): the global merged copy outlived Discard and ate memory
            self.anchorT = [matrix_identity_float4x4]; self.anchorIdx.removeAll(); self.anchorGen &+= 1
            DispatchQueue.main.async {
                self.run(withPeople: self.personOn, reset: true); self.count = 0; self.previewNode.geometry = nil; self.setStatus("Point at the setup, then tap Scan") }
        }
    }

    private func alertFast(_ t: TimeInterval) {
        guard ALERTS_ON, t - lastAlert >= ALERT_EVERY else { return }
        lastAlert = t
        UINotificationFeedbackGenerator().notificationOccurred(.warning)
        AudioServicesPlaySystemSound(SystemSoundID(1057))
    }

    private func limitedText(_ r: ARCamera.TrackingState.Reason) -> String {
        switch r {
        case .excessiveMotion: return "Paused - moving too much"
        case .insufficientFeatures: return "Paused - not enough detail in view (add light or texture)"
        case .initializing: return "Starting - move the phone gently"
        case .relocalizing: return initialMap != nil ? "Point at part of the scan you already have - it will line up" : "Paused - relocalizing, point back at where you were"
        @unknown default: return "Paused - tracking is limited"
        }
    }

    // ---- anchors (mesh + our world anchors): only their poses are kept, on the work queue ----
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) { trackAnchors(anchors) }
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) { trackAnchors(anchors) }
    /// a removed anchor keeps its last pose here: points are never dropped because ARKit dropped an anchor
    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) { meshOverlay.remove(anchors) }
    private func trackAnchors(_ anchors: [ARAnchor]) {
        meshOverlay.update(anchors.compactMap { $0 as? ARMeshAnchor })
        let poses = anchors.filter { !($0 is ARMeshAnchor) }.map { ($0.identifier, $0.transform) }
        work.async {
            for (id, T) in poses {
                if let i = self.anchorIdx[id] { if self.anchorT[Int(i)] != T { self.anchorT[Int(i)] = T; self.anchorGen &+= 1 } }
                else { self.anchorIdx[id] = Int32(self.anchorT.count); self.anchorT.append(T); self.anchorGen &+= 1 }
            }
        }
    }
    @inline(__always) fileprivate func world(_ v: Vox) -> SIMD3<Float> { v.a == 0 ? v.p : xform(anchorT[Int(v.a)], v.p) }

    // ---- every camera frame (main thread) ----
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if let req = jpegRequest {
            jpegRequest = nil
            let pb = frame.capturedImage
            DispatchQueue.global(qos: .utility).async { if let d = smallJPEG(pb) { req(d) } }
        }
        let t = frame.timestamp, T = frame.camera.transform
        if prevTime > 0, t > prevTime {
            let dt = Float(t - prevTime)
            speed = speed * 0.5 + 0.5 * simd_length(position(T) - position(prevT)) / dt
            turn = turn * 0.5 + 0.5 * turnAngle(prevT, T) / dt
        }
        prevT = T; prevTime = t
        if running || count > 0 { updateCoverage(frame) }
        let fastNow = speed > FAST_MOVE || turn > FAST_TURN
        if fastNow { lastFast = t }
        ambient = frame.lightEstimate?.ambientIntensity ?? ambient
        if checkpoints.placed {
            let ph = position(frame.camera.transform)
            if checkpoints.update(phone: ph) { UINotificationFeedbackGenerator().notificationOccurred(.success) }
            let txt: String
            if let (i, p) = checkpoints.next(from: ph) { txt = "Checkpoint \(i + 1): \(Int((simd_length(p - ph) * 100).rounded())) cm away - \(checkpoints.remaining) left" }
            else { txt = "All checkpoints done" }
            if checkpointText != txt { checkpointText = txt }
        }
        guard running else { return }

        // auto-correction: nothing is taken while tracking is not normal
        switch frame.camera.trackingState {
        case .normal: break
        case .notAvailable: setStatus("Paused - tracking not available", warn: true); return
        case .limited(let r): setStatus(limitedText(r), warn: true); return
        }
        if lastAnchorPos == nil || simd_length(position(T) - lastAnchorPos!) >= ANCHOR_EVERY {
            lastAnchorPos = position(T)
            session.add(anchor: ARAnchor(name: "kh", transform: T))
        }
        guard let depth = frame.smoothedSceneDepth ?? frame.sceneDepth, let conf = depth.confidenceMap else { return }
        let K = frame.camera.intrinsics, img = frame.capturedImage
        let iw = CVPixelBufferGetWidth(img), ih = CVPixelBufferGetHeight(img)

        // too fast (or within 0.3 s of it): alert, take nothing, mark what was in view as needing a rescan
        if t - lastFast < STEADY_AFTER {
            if fastNow { alertFast(t) }
            setStatus("Slow down", warn: true)
            if !busy && t - lastIntegrate > 0.2 { lastIntegrate = t; markRemoved(depth.depthMap, conf, T, K, iw, ih) }
            return
        }
        let dark = USE_CAMERA && (frame.lightEstimate?.ambientIntensity ?? 1000) < DARK_LUX
        if dark { setStatus("Too dark for good photos - add light", warn: true) }
        else if MAX_KEYFRAMES == 0 { setStatus("Scanning - LiDAR only") }
        else { setStatus(keyframes >= MAX_KEYFRAMES ? "Scanning - photos done, LiDAR carries on" : "Scanning - move slowly, 30-60 cm away") }
        // self-tuning: if almost every reading is being thrown away, relax one step at a time and say so
        if running && t - lastTuneT > 3 && count < 50_000 && yieldRatio < 0.04 {
            lastTuneT = t
            if MIN_CONF > UInt8(ARConfidenceLevel.medium.rawValue) { MIN_CONF = UInt8(ARConfidenceLevel.medium.rawValue); setStatus("Few high-confidence readings here - using medium ones too", warn: true); Log.add("tune: MIN_CONF medium") }
            else if EDGE_REL < 0.16 { EDGE_REL += 0.03; Log.add("tune: EDGE_REL \(EDGE_REL)") }
            else if MIN_HITS > 1 { MIN_HITS -= 1; Log.add("tune: MIN_HITS \(MIN_HITS)") }
        }

        // 5.0: an even, cool pace. The gap follows how long the last pass took (a heavy pass at the start of a scan is
        // followed by a pause, so the load is steady instead of full power at first and idle later), and grows when the
        // phone is warm or in Low Power Mode.
        let therm = ProcessInfo.processInfo.thermalState
        let calm: Double = (therm == .serious || therm == .critical ? 3.0 : (therm == .fair ? 1.5 : 1.0)) * (ProcessInfo.processInfo.isLowPowerModeEnabled ? 1.5 : 1.0)
        let integrateDue = !busy && t - lastIntegrate > max(FUSE_GAP, dynGap) * calm
        var kfDue = !ioBusy && keyframes < MAX_KEYFRAMES
        if kfDue, let L = lastKF, simd_length(position(T) - position(L)) < KF_MOVE, turnAngle(L, T) < KF_TURN { kfDue = false }
        if integrateDue || kfDue {
            // the blur check (staged: decided before anything is committed)
            let fs: Float = USE_CAMERA ? frameSharpness(img) : 1
            sharpHist.append(fs); if sharpHist.count > 120 { sharpHist.removeFirst() }
            let sorted = sharpHist.sorted()
            let median = sorted[sorted.count / 2]
            let thresh = sharpHist.count >= 10 ? BLUR_REL * median : 0
            if USE_CAMERA && fs < thresh {
                // 3.3: a blurry frame still adds LiDAR geometry (colour is skipped by COLOUR_SHARP), it just is no keyframe
                setStatus("Blurry - hold steadier", warn: true)
                kfDue = false
            }
            let seg = personOn ? frame.segmentationBuffer : nil
            if kfDue { saveKeyframe(img, depth.depthMap, conf, seg, T, K, t, fs); lastDepthKF = T }
            else if USE_CAMERA && MAX_KEYFRAMES > 0 && depthFrames < 800 && !ioBusy,
                    lastDepthKF.map({ simd_length(position(T) - position($0)) >= 0.15 || turnAngle($0, T) >= 10 * .pi / 180 }) ?? true {
                lastDepthKF = T; saveDepthFrame(depth.depthMap, conf, T, K, iw, ih, t)
            }
            if integrateDue {
                busy = true; lastIntegrate = t
                let dmap = depth.depthMap
                let med: Float = sharpHist.count >= 10 ? median : 0
                work.async {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    self.integrate(dmap: dmap, conf: conf, seg: seg, image: img, T: T, K: K, fs: fs, blurT: thresh,
                                   median: med)
                    self.dynGap = min(0.25, (CFAbsoluteTimeGetCurrent() - t0) * (MAX_DEPTH > 5.5 ? 0.6 : 1.2))
                    let n = self.goodCount
                    // 4.6: the limit is the memory iOS gives the app, not a fixed count. Keep enough back for the save
                    // (it needs ~45 bytes per point on top of the scan). Clear single sightings at most every 8 s.
                    let avail = os_proc_available_memory()
                    // 5.0: the reserve is what the SAVE really needs (~60 bytes per voxel for the merged copy and its file)
                    // plus 350 MB for the system and a burst of full-resolution photos. It used to be a fixed 1.3 GB floor,
                    // which declared a 600k-point PowerBlast scan "full" the moment 7 photos were in flight.
                    let reserve = self.vox.count * 48 + 350_000_000
                    // even spacing as memory fills: 9 -> 5 -> 3 -> 1 points per 6 mm cell for NEW areas. The baseline is
                    // the MOST memory seen free, so a momentary dip (photo capture) never lowers it.
                    self.startAvail = max(self.startAvail, avail)
                    let frac = Double(avail - reserve) / Double(max(1, self.startAvail - reserve))
                    let k: UInt8 = frac > 0.5 ? 9 : (frac > 0.3 ? 5 : (frac > 0.15 ? 3 : 1))
                    if k != self.densityK {
                        self.densityK = k
                        let note = k == 9 ? "" : "Memory filling up: new points are spaced out (\(k == 5 ? "~3" : (k == 3 ? "~4" : "~6")) mm) so the whole room fits"
                        Log.add("density per 6 mm cell -> \(k), free \(avail / 1_000_000) MB, reserve \(reserve / 1_000_000) MB")
                        DispatchQueue.main.async { self.spacingNote = note }
                    }
                    if (avail < reserve + 300_000_000 || self.vox.count >= MAX_VOXELS * 95 / 100) && t - self.lastCompact > 8 {
                        self.lastCompact = t; self.compact()
                    }
                    // "full" needs the memory to stay short for 4 checks in a row (several seconds) AFTER a compaction has
                    // had its chance - a single low reading while photos are being written is not a full scan
                    let lowNow = os_proc_available_memory() < reserve
                    self.lowStreak = lowNow ? self.lowStreak + 1 : 0
                    let full = (self.lowStreak >= 4 && t - self.lastCompact > 3) || self.vox.count >= MAX_VOXELS * 95 / 100
                    if t - self.lastPreview > PREVIEW_GAP { self.lastPreview = t; self.updatePreview() }
                    // 5.0: the point count reaches the screen 4 times a second, not every frame (each update redrew the scan screen)
                    let showCount = t - self.lastCountShown > 0.25 || full
                    if showCount { self.lastCountShown = t }
                    DispatchQueue.main.async {
                        self.busy = false
                        if showCount && self.count != n { self.count = n }
                        if full && self.running {
                            self.running = false; self.full = true
                            self.setStatus("Memory full at \(n.formatted()) points (\(os_proc_available_memory() / 1_000_000) MB free) - tap Save", warn: true)
                            Log.add("full at \(n) points, voxels \(self.vox.count), free \(os_proc_available_memory() / 1_000_000) MB")
                            self.measureSaved()
                        }
                    }
                }
            }
        }
        // 4.1: was every 2 s over the whole cloud, which slowed scanning
        if smartOn && t - lastAnalysis > 8.0 { lastAnalysis = t; analyse() }
    }

    fileprivate var goodN = 0                 // voxels with MIN_HITS or more, kept up to date as they cross it
    private var goodAt: UInt16 = MIN_HITS      // 5.7 (48): the MIN_HITS goodN was counted at (work queue)
    private var goodCount: Int { goodN }
    /// 5.7 (48): goodN only counts voxels as they reach MIN_HITS exactly, so a changed MIN_HITS (the low-yield tune,
    /// a mode switch back to the baseline) left it wrong for the rest of the scan: count again (work queue only)
    @discardableResult private func syncGoodLocked() -> Bool {
        let mh = MIN_HITS
        guard goodAt != mh else { return false }
        goodAt = mh
        mapGood.removeAll(keepingCapacity: true)
        var g = 0; for x in vox where x.hits >= mh { g += 1; mapGood.insert(mapKey(world(x))) }
        goodN = g
        return true
    }
    /// 5.7 (48): brings the shown count up to date after MIN_HITS changed while paused
    func recountGood() {
        work.async { if self.syncGoodLocked() { let n = self.goodN; DispatchQueue.main.async { self.count = n } } }
    }

    /// a rejected frame: remember the 2 cm cells it looked at as "needs rescan"
    private func markRemoved(_ dmap: CVPixelBuffer, _ conf: CVPixelBuffer, _ T: simd_float4x4, _ K: simd_float3x3, _ iw: Int, _ ih: Int) {
        work.async {
            CVPixelBufferLockBaseAddress(dmap, .readOnly); CVPixelBufferLockBaseAddress(conf, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(dmap, .readOnly); CVPixelBufferUnlockBaseAddress(conf, .readOnly) }
            guard let dBase = CVPixelBufferGetBaseAddress(dmap), let cBase = CVPixelBufferGetBaseAddress(conf) else { return }
            let dw = CVPixelBufferGetWidth(dmap), dh = CVPixelBufferGetHeight(dmap)
            let dRow = CVPixelBufferGetBytesPerRow(dmap), cRow = CVPixelBufferGetBytesPerRow(conf)
            let sx = Float(dw) / Float(iw), sy = Float(dh) / Float(ih)
            let fx = K[0][0] * sx, fy = K[1][1] * sy, cx = K[2][0] * sx, cy = K[2][1] * sy
            for v in stride(from: 0, to: dh, by: 8) {
                let dr = dBase.advanced(by: v * dRow).assumingMemoryBound(to: Float32.self)
                let cr = cBase.advanced(by: v * cRow).assumingMemoryBound(to: UInt8.self)
                for u in stride(from: 0, to: dw, by: 8) {
                    let d = dr[u]
                    guard cr[u] >= UInt8(ARConfidenceLevel.medium.rawValue), d > MIN_DEPTH, d < MAX_DEPTH else { continue }
                    let pw = T * SIMD4<Float>((Float(u) - cx) * d / fx, -(Float(v) - cy) * d / fy, -d, 1)
                    self.removedCells.insert(cellKey(SIMD3<Float>(pw.x, pw.y, pw.z)))
                }
            }
        }
    }

    private func integrate(dmap: CVPixelBuffer, conf: CVPixelBuffer, seg: CVPixelBuffer?, image: CVPixelBuffer,
                           T: simd_float4x4, K: simd_float3x3, fs: Float, blurT: Float, median: Float) {
        blurThresh = blurT
        snapCache = nil                     // 5.7 (47): re-sighted voxels move and recolour without changing vox.count or goodN
        syncGoodLocked()                    // 5.7 (48): MIN_HITS may have changed since the last pass
        let minHits = goodAt
        // colour from this frame only if it is sharp; sharper photos weigh more (median frame = 1, capped at 4)
        let rel: Float = median > 0 ? fs / median : 1
        let colW: Float = median > 0 && rel < COLOUR_SHARP ? 0 : min(4, rel * rel)
        // the anchors near the phone, with their inverses for this frame
        let camPos = position(T)
        var cand: [(Int32, SIMD3<Float>, simd_float4x4)] = []
        if anchorT.count > 1 {
            var near: [(Float, Int)] = []
            near.reserveCapacity(anchorT.count)
            for i in 1..<anchorT.count { near.append((simd_length(position(anchorT[i]) - camPos), i)) }
            if near.count > ANCHOR_NEAR { near.sort { $0.0 < $1.0 }; near.removeLast(near.count - ANCHOR_NEAR) }
            for (_, i) in near { cand.append((Int32(i), position(anchorT[i]), anchorT[i].inverse)) }
        }
        let dw = CVPixelBufferGetWidth(dmap), dh = CVPixelBufferGetHeight(dmap)
        let person = personMask(seg, dw, dh)
        CVPixelBufferLockBaseAddress(dmap, .readOnly); CVPixelBufferLockBaseAddress(conf, .readOnly)
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dmap, .readOnly); CVPixelBufferUnlockBaseAddress(conf, .readOnly); CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        let iw = CVPixelBufferGetWidthOfPlane(image, 0), ih = CVPixelBufferGetHeightOfPlane(image, 0)
        guard let dBase = CVPixelBufferGetBaseAddress(dmap), let cBase = CVPixelBufferGetBaseAddress(conf),
              let yBase = CVPixelBufferGetBaseAddressOfPlane(image, 0), let uvBase = CVPixelBufferGetBaseAddressOfPlane(image, 1) else { return }
        let dRow = CVPixelBufferGetBytesPerRow(dmap), cRow = CVPixelBufferGetBytesPerRow(conf)
        let yRow = CVPixelBufferGetBytesPerRowOfPlane(image, 0), uvRow = CVPixelBufferGetBytesPerRowOfPlane(image, 1)

        // the camera intrinsics are for the full image; scale them to the depth map
        let sx = Float(dw) / Float(iw), sy = Float(dh) / Float(ih)
        let fx = K[0][0] * sx, fy = K[1][1] * sy, cx = K[2][0] * sx, cy = K[2][1] * sy
        let high = UInt8(ARConfidenceLevel.high.rawValue), medium = UInt8(ARConfidenceLevel.medium.rawValue)
        let clearRemoved = !removedCells.isEmpty
        func depthAt(_ u: Int, _ v: Int) -> Float {
            dBase.advanced(by: v * dRow).assumingMemoryBound(to: Float32.self)[u]
        }
        // 4.5 fast path. Phase A, on every core: depth -> world point, colour, and the nearest anchor (from a shortlist
        // of 4 per 16x16 patch instead of all 48). Phase B, one thread: the voxel store updates.
        struct Sample { var key: VKey; var lp: SIMD3<Float>; var p: SIMD3<Float>; var col: SIMD3<Float>
                        var alt1: VKey; var lp1: SIMD3<Float>; var alt2: VKey; var lp2: SIMD3<Float>; var w: Float }
        let B = 16, bw = (dw + B - 1) / B, bh = (dh + B - 1) / B
        var shortlist = [[(Int32, SIMD3<Float>, simd_float4x4)]](repeating: [], count: bw * bh)
        if !cand.isEmpty {
            for by in 0..<bh { for bx in 0..<bw {
                let u = min(dw - 1, bx * B + B / 2), v = min(dh - 1, by * B + B / 2)
                var d = depthAt(u, v); if !(d > MIN_DEPTH && d < 8) { d = 1.5 }
                let pc = T * SIMD4<Float>((Float(u) - cx) * d / fx, -(Float(v) - cy) * d / fy, -d, 1)
                let c3 = SIMD3<Float>(pc.x, pc.y, pc.z)
                shortlist[by * bw + bx] = Array(cand.sorted { simd_length_squared($0.1 - c3) < simd_length_squared($1.1 - c3) }.prefix(4))
            } }
        }
        let rows = Array(stride(from: 0, to: dh, by: STRIDE))
        let chunks = 8, per = (rows.count + chunks - 1) / chunks
        var parts = [[Sample]](repeating: [], count: chunks), depthParts = [[Float]](repeating: [], count: chunks)
        var triedParts = [Int](repeating: 0, count: chunks)
        let pm = person, ed = EDGE_REL, maxD = MAX_DEPTH, minConf = max(medium, MIN_CONF), camOn = USE_CAMERA
        parts.withUnsafeMutableBufferPointer { pb in depthParts.withUnsafeMutableBufferPointer { db in triedParts.withUnsafeMutableBufferPointer { tb in
            let pp = pb.baseAddress!, dp = db.baseAddress!, tp = tb.baseAddress!
            DispatchQueue.concurrentPerform(iterations: chunks) { ci in
                var out: [Sample] = []; out.reserveCapacity(dw * per / STRIDE + 1)
                var ds: [Float] = []
                var tried = 0
                let r0 = ci * per, r1 = min(rows.count, r0 + per)
                if r0 < r1 { for ri in r0..<r1 {
                    let v = rows[ri]
                    let cRowPtr = cBase.advanced(by: v * cRow).assumingMemoryBound(to: UInt8.self)
                    for u in stride(from: 0, to: dw, by: STRIDE) {
                        let cf = cRowPtr[u]
                        guard cf >= minConf else { tried += 1; continue }
                        if let m = pm, m[v * dw + u] != 0 { continue }
                        let d = depthAt(u, v)
                        if (u &* 7 &+ v &* 3) % 97 == 0, d > MIN_DEPTH, d < 9 { ds.append(d) }
                        guard d > MIN_DEPTH, d < maxD else { continue }
                        let lim = ed * d       // depth-edge rejection (NaN or 0 counts as an edge)
                        if u > 0, !(abs(d - depthAt(u - 1, v)) <= lim) { continue }
                        if u + 1 < dw, !(abs(d - depthAt(u + 1, v)) <= lim) { continue }
                        if v > 0, !(abs(d - depthAt(u, v - 1)) <= lim) { continue }
                        if v + 1 < dh, !(abs(d - depthAt(u, v + 1)) <= lim) { continue }
                        let pw4 = T * SIMD4<Float>((Float(u) - cx) * d / fx, -(Float(v) - cy) * d / fy, -d, 1)
                        let p = SIMD3<Float>(pw4.x, pw4.y, pw4.z)
                        let ix = min(iw - 1, Int(Float(u) / sx)), iy = min(ih - 1, Int(Float(v) / sy))
                        let Y = Float(yBase.advanced(by: iy * yRow + ix).assumingMemoryBound(to: UInt8.self).pointee)
                        let uvp = uvBase.advanced(by: (iy / 2) * uvRow + (ix / 2) * 2).assumingMemoryBound(to: UInt8.self)
                        let Cb = Float(uvp[0]) - 128, Cr = Float(uvp[1]) - 128
                        var col = SIMD3<Float>(max(0, min(255, Y + 1.402 * Cr)), max(0, min(255, Y - 0.344136 * Cb - 0.714136 * Cr)), max(0, min(255, Y + 1.772 * Cb)))
                        if !camOn { col = heightColour(p.y) }       // LiDAR only: coloured by height, not by the camera
                        // the nearest anchor holds the sample; the next two are where an existing copy of this surface may live
                        var r0: (Int32, SIMD3<Float>, Float) = (0, p, .greatestFiniteMagnitude), r1 = r0, r2 = r0
                        for c in shortlist[(v / B) * bw + u / B] {
                            let d2 = simd_length_squared(c.1 - p)
                            if d2 < r0.2 { r2 = r1; r1 = r0; r0 = (c.0, xform(c.2, p), d2) }
                            else if d2 < r1.2 { r2 = r1; r1 = (c.0, xform(c.2, p), d2) }
                            else if d2 < r2.2 { r2 = (c.0, xform(c.2, p), d2) }
                        }
                        // LiDAR error grows with distance squared; high confidence counts more than medium
                        let w = (cf == high ? 1.0 : 0.45) / max(0.09, d * d)
                        tried += 1
                        out.append(Sample(key: VKey(a: r0.0, k: voxKey(r0.1)), lp: r0.1, p: p, col: col,
                                          alt1: VKey(a: r1.0, k: voxKey(r1.1)), lp1: r1.1, alt2: VKey(a: r2.0, k: voxKey(r2.1)), lp2: r2.1, w: w))
                    }
                } }
                pp[ci] = out; dp[ci] = ds; tp[ci] = tried
            }
        } } }
        // how much of what the LiDAR offered was kept (the self-tuning reads it)
        let kept = parts.reduce(0) { $0 + $1.count }, offered = max(1, triedParts.reduce(0, +))
        yieldRatio = yieldRatio * 0.85 + Float(kept) / Float(offered) * 0.15
        let allDepth = depthParts.flatMap { $0 }
        if allDepth.count > 10 { let sd = allDepth.sorted(); medianDepth = sd[sd.count / 2] }
        _ = high
        for part in parts { for smp in part {
            coverTick &+= 1
            if coverTick & 3 == 0 { let ck = voxKey(smp.p, 0.03); let cv = cover[ck] ?? 0; if cv < 255 { cover[ck] = cv + 1; if cv == 2 { coveredCells += 1 } } }
            var lp = smp.lp
            let p = smp.p, col = smp.col
            var i: Int
            if let found = index[smp.key] { i = Int(found) }
            else if smp.alt1.a != smp.key.a, let f1 = index[smp.alt1] { i = Int(f1); lp = smp.lp1 }        // 4.5: same surface, older anchor
            else if smp.alt2.a != smp.key.a, let f2 = index[smp.alt2] { i = Int(f2); lp = smp.lp2 }
            else {
                if vox.count >= MAX_VOXELS { continue }
                let wk = voxKey(p, 0.006)
                let cn = cellCount[wk] ?? 0
                if cn >= densityK { continue }                 // this patch of space already has its share of points
                cellCount[wk] = cn + 1
                i = vox.count
                index[smp.key] = Int32(i)
                vox.append(Vox(p: lp, c: col, cw: 0, a: smp.key.a, m2: 0, sharp: fs, hits: 0, seen: 0, pw: 0))
            }
            var x = vox[i]
            if x.seen < UInt16.max { x.seen += 1 }
            // a sample far outside what this voxel has seen so far (3 sigma + 2 mm) is a stray reading, not surface
            if x.hits >= 3 {
                let varc = x.m2 / max(1e-6, x.pw)
                if simd_length_squared(lp - x.p) > 9 * varc + 4e-6 { vox[i] = x; continue }
            }
            if x.hits < UInt16.max {
                x.hits += 1
                let n = Float(x.hits)
                x.pw += smp.w
                let delta = lp - x.p
                x.p += delta * (smp.w / x.pw)
                x.m2 += smp.w * simd_dot(delta, lp - x.p)
                if colW > 0 { x.cw += colW; x.c += (col - x.c) * (colW / x.cw) }
                else if x.cw == 0 { x.c += (col - x.c) / n }
                x.sharp += (fs - x.sharp) / n
                if x.hits == minHits { mapGood.insert(mapKey(p)); goodN += 1 }
                if clearRemoved { removedCells.remove(cellKey(p)) }
            }
            vox[i] = x
        } }
    }

    // ---- keyframes for the Mac (decided on main, written on the io queue) ----
    private func ensureCaptureDir() -> URL? {
        if let d = captureDir { return d }
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd-HHmmss"
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        let d = docs.appendingPathComponent("captures", isDirectory: true).appendingPathComponent(f.string(from: Date()), isDirectory: true)
        do { try FileManager.default.createDirectory(at: d.appendingPathComponent("frames", isDirectory: true), withIntermediateDirectories: true) }
        catch { return nil }
        captureDir = d
        return d
    }

    private func saveKeyframe(_ img: CVPixelBuffer, _ dmap: CVPixelBuffer, _ conf: CVPixelBuffer, _ seg: CVPixelBuffer?,
                              _ T: simd_float4x4, _ K: simd_float3x3, _ ts: TimeInterval, _ sharp: Float) {
        guard let dir = ensureCaptureDir(),
              let dData = copyPlane(dmap, bytesPerPixel: 4), let cData = copyPlane(conf, bytesPerPixel: 1) else { return }
        let dw = CVPixelBufferGetWidth(dmap), dh = CVPixelBufferGetHeight(dmap)
        let pData = personMask(seg, dw, dh).map { Data($0) }
        lastKF = T; photoPoses.append(T)
        let idx = keyframes
        keyframes += 1
        ioBusy = true
        // 4.5: a full-resolution still from the camera (4K or more) instead of the 1920x1440 video frame; the video frame is
        // the fallback. Its own pose and intrinsics go with it.
        if hiRes, #available(iOS 16.0, *), let s = session {
            s.captureHighResolutionFrame { hr, _ in
                let im = hr?.capturedImage ?? img, TT = hr?.camera.transform ?? T, KK = hr?.camera.intrinsics ?? K
                self.io.async {
                    self.writeKeyframe(dir: dir, idx: idx, image: im, depth: dData, conf: cData, person: pData, dw: dw, dh: dh, T: TT, K: KK, ts: ts, sharp: sharp)
                    DispatchQueue.main.async { self.ioBusy = false }
                }
            }
            return
        }
        io.async {
            self.writeKeyframe(dir: dir, idx: idx, image: img, depth: dData, conf: cData, person: pData, dw: dw, dh: dh, T: T, K: K, ts: ts, sharp: sharp)
            DispatchQueue.main.async { self.ioBusy = false }
        }
    }

    /// 5.0: a depth-only frame (no photo): the LiDAR depth at half precision, its confidence and the pose. The 3D model
    /// takes its shape from these too, so surfaces between photos are not holes; ~150 KB each, at most 800 a scan.
    private func saveDepthFrame(_ dmap: CVPixelBuffer, _ conf: CVPixelBuffer, _ T: simd_float4x4, _ K: simd_float3x3, _ iw: Int, _ ih: Int, _ ts: TimeInterval) {
        guard let dir = ensureCaptureDir(), let dData = copyPlane(dmap, bytesPerPixel: 4), let cData = copyPlane(conf, bytesPerPixel: 1) else { return }
        let dw = CVPixelBufferGetWidth(dmap), dh = CVPixelBufferGetHeight(dmap)
        let idx = depthFrames; depthFrames += 1
        io.async {
            let folder = dir.appendingPathComponent("depth", isDirectory: true)
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let name = String(format: "%06d", idx)
            var half = [Float16](repeating: 0, count: dw * dh)
            dData.withUnsafeBytes { b in for i in 0..<half.count { half[i] = Float16(b.loadUnaligned(fromByteOffset: i * 4, as: Float.self)) } }
            var tm: [Double] = []; for c in 0..<4 { for r in 0..<4 { tm.append(Double(T[c][r])) } }
            var km: [Double] = []; for c in 0..<3 { for r in 0..<3 { km.append(Double(K[c][r])) } }
            let meta: [String: Any] = ["transform": tm, "intrinsics": km, "imageWidth": iw, "imageHeight": ih, "depthWidth": dw, "depthHeight": dh, "timestamp": ts, "depthOnly": true]
            do {
                try half.withUnsafeBufferPointer { try Data(buffer: $0).write(to: folder.appendingPathComponent(name + ".depth16")) }
                try cData.write(to: folder.appendingPathComponent(name + ".conf"))
                if let j = try? JSONSerialization.data(withJSONObject: meta) { try j.write(to: folder.appendingPathComponent(name + ".json")) }
            } catch {}
        }
    }

    private func writeKeyframe(dir: URL, idx: Int, image: CVPixelBuffer, depth: Data, conf: Data, person: Data?, dw: Int, dh: Int,
                               T: simd_float4x4, K: simd_float3x3, ts: TimeInterval, sharp: Float) {
        let name = String(format: "%06d", idx)
        let frames = dir.appendingPathComponent("frames", isDirectory: true)
        let iw = CGFloat(CVPixelBufferGetWidth(image)), ih = CGFloat(CVPixelBufferGetHeight(image))
        var s: CGFloat = 1
        var ci = CIImage(cvPixelBuffer: image)
        if max(iw, ih) > PHOTO_LONG {
            s = PHOTO_LONG / max(iw, ih)
            ci = ci.transformed(by: CGAffineTransform(scaleX: s, y: s))
        }
        let ow = Int((iw * s).rounded()), oh = Int((ih * s).rounded())
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let q = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let jpg = ciContext.jpegRepresentation(of: ci, colorSpace: cs, options: [q: JPEG_Q]) else { return }
        do {
            try jpg.write(to: frames.appendingPathComponent(name + ".jpg"))
            try depth.write(to: frames.appendingPathComponent(name + ".depth"))
            try conf.write(to: frames.appendingPathComponent(name + ".conf"))
            if let p = person { try p.write(to: frames.appendingPathComponent(name + ".person")) }
        } catch { return }

        var tm: [Double] = []
        for c in 0..<4 { for r in 0..<4 { tm.append(Double(T[c][r])) } }      // column-major (columns concatenated)
        var km: [Double] = []
        for c in 0..<3 { for r in 0..<3 { km.append(Double(K[c][r])) } }      // column-major
        let fs = Double(s)
        km[0] *= fs; km[4] *= fs; km[6] *= fs; km[7] *= fs                       // only if the photo was scaled down
        let meta: [String: Any] = [
            "transform": tm, "intrinsics": km,
            "imageWidth": ow, "imageHeight": oh, "depthWidth": dw, "depthHeight": dh,
            "timestamp": ts, "sharpness": Double(sharp)
        ]
        if let j = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) {
            try? j.write(to: frames.appendingPathComponent(name + ".json"))
        }
    }

    // ---- live preview (work queue) ----
    private func flaw(_ v: Vox) -> Flaw { flawOf(hits: Int(v.hits), seen: Int(v.seen), sharp: v.sharp, blur: blurThresh) }
    private func flaw(_ v: Pt) -> Flaw { flawOf(hits: v.hits, seen: v.seen, sharp: v.sharp, blur: blurThresh) }

    /// every voxel re-binned on the MERGE_VOXEL world grid at the anchors' CURRENT poses; work queue only.
    /// Hit-weighted position, colour-weight-weighted colour; a merged point needs >= MIN_HITS hits in total.
    /// every voxel re-binned on the MERGE_VOXEL world grid at the anchors' CURRENT poses; work queue only.
    /// 4.5: sorted keys instead of dictionaries - a third of the memory, so a 10M-point scan can be paused and saved.
    fileprivate func merged() -> (p: [SIMD3<Float>], c: [SIMD3<UInt8>]) {
        struct KI { var k: Int64; var i: Int32 }
        var ks: [KI] = []; ks.reserveCapacity(vox.count)
        for i in 0..<vox.count where vox[i].hits >= 1 { ks.append(KI(k: voxKey(world(vox[i]), MERGE_VOXEL), i: Int32(i))) }
        ks.sort { $0.k < $1.k }
        var outP: [SIMD3<Float>] = [], outC: [SIMD3<UInt8>] = []
        outP.reserveCapacity(ks.count / 2); outC.reserveCapacity(ks.count / 2)
        var thin: [(SIMD3<Float>, SIMD3<Float>)] = []                  // single sightings with sharp colour: wire candidates
        @inline(__always) func ck(_ p: SIMD3<Float>) -> SIMD3<Int32> {
            SIMD3<Int32>(ik((p.x / THIN_NEAR)), ik((p.y / THIN_NEAR)), ik((p.z / THIN_NEAR)))
        }
        var j = 0
        while j < ks.count {
            let k = ks[j].k
            var ps = SIMD3<Float>.zero, cs = SIMD3<Float>.zero, cw: Float = 0, h = 0
            while j < ks.count && ks[j].k == k {
                let v = vox[Int(ks[j].i)], hh = Float(v.hits)
                ps += world(v) * hh; h += Int(v.hits)
                let w: Float = v.cw > 0 ? v.cw : 0.01 * hh
                cs += v.c * w; cw += w
                j += 1
            }
            let p = ps / Float(h), c = cs / max(1e-6, cw)
            if h >= Int(MIN_HITS) {
                outP.append(p); outC.append(SIMD3(u8(c.x), u8(c.y), u8(c.z)))
            } else if cw >= 0.25 { thin.append((p, c)) }
        }
        ks = []                                                          // 5.0: freed before the next step
        // thin things (cables): a single sighting stays if a confident point of the same colour is close by.
        // 5.0: the confident cells are gathered only around the thin candidates (was every cell of the scan: ~240 MB)
        guard !thin.isEmpty else { return (outP, outC) }
        var want = Set<SIMD3<Int32>>()
        for (p, _) in thin { let k = ck(p); for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 { want.insert(k &+ SIMD3(dx, dy, dz)) } } } }
        var conf: [SIMD3<Int32>: (SIMD3<Float>, SIMD3<Float>, Float)] = [:]   // 6 mm cell -> (position sum, colour sum, n)
        for i in 0..<outP.count {
            let key = ck(outP[i]); guard want.contains(key) else { continue }
            let c = SIMD3<Float>(Float(outC[i].x), Float(outC[i].y), Float(outC[i].z))
            let o = conf[key] ?? (.zero, .zero, 0); conf[key] = (o.0 + outP[i], o.1 + c, o.2 + 1)
        }
        for (p, c) in thin {
            let k = ck(p)
            search: for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 {
                guard let o = conf[k &+ SIMD3(dx, dy, dz)] else { continue }
                if simd_length(o.0 / o.2 - p) <= THIN_NEAR * 1.5 && simd_length(o.1 / o.2 - c) <= THIN_COLOUR {
                    outP.append(p); outC.append(SIMD3(u8(c.x), u8(c.y), u8(c.z))); break search
                }
            } } }
        }
        return (outP, outC)
    }

    /// up to 200k of the good voxels as dots in their colours; Smart check tints weak ones red / amber
    fileprivate func updatePreview() {
        let good = goodCount
        guard good > 0 else { DispatchQueue.main.async { self.previewNode.geometry = nil }; return }
        let step = max(1, vox.count / 220_000)
        var verts: [SCNVector3] = []; var cols: [Float] = []
        verts.reserveCapacity(vox.count / step + 1); cols.reserveCapacity(3 * (vox.count / step + 1))
        for vi in stride(from: 0, to: vox.count, by: step) {
            let v = vox[vi]
            guard v.hits >= MIN_HITS else { continue }
            let w = world(v)
            verts.append(SCNVector3(w.x, w.y, w.z))
            var f = Flaw.ok
            if tint { f = flaw(v); if f == .ok && noisyCells.contains(cellKey(w)) { f = .noisy } }
            if f == .ok { cols += [v.c.x / 255, v.c.y / 255, v.c.z / 255] }
            else if f.red { cols += [1, 0.15, 0.1] } else { cols += [1, 0.7, 0] }
        }
        guard !verts.isEmpty else { DispatchQueue.main.async { self.previewNode.geometry = nil }; return }
        let geo = pointGeometry(verts, cols, size: 3)
        let soft = PREVIEW_SOFT, step2 = step
        DispatchQueue.main.async {
            // 5.0: Gaussian / Aran scans: the live view is drawn as real Gaussian splats, sized to cover the gaps
            if soft, let v = self.arView, let m = SplatLook.material(radius: max(0.004, Float(MERGE_VOXEL) * 1.6 * Float(step2).squareRoot()), pxScale: SplatLook.pxScale(v)) { geo.materials = [m] }
            self.previewNode.geometry = geo
        }
    }

    // ---- Smart check (heuristic quality model) ----
    func analyse() {
        work.async {
            let found = self.runAnalysis()
            if self.tint { self.updatePreview() }
            DispatchQueue.main.async {
                guard self.smartOn else { return }
                self.clusters = found
                self.rebuildMarkers()
                self.updateGuide()
            }
        }
    }

    private struct CellAcc {
        var n = 0
        var w = [Int](repeating: 0, count: 4)       // per Flaw .few, .lowConf, .blurry (index = rawValue)
        var sx = 0.0, sy = 0.0, sz = 0.0, sxx = 0.0, sxy = 0.0, sxz = 0.0, syy = 0.0, syz = 0.0, szz = 0.0
    }

    /// work queue only
    private func runAnalysis() -> [WeakCluster] {
        var cells: [SIMD3<Int32>: CellAcc] = [:]
        var lo: Float = .greatestFiniteMagnitude, hi: Float = -.greatestFiniteMagnitude
        // 4.1: a sample of the raw voxels (at most 250k), not a full merge of millions of points every pass
        let step = max(1, vox.count / 250_000)
        for i in stride(from: 0, to: vox.count, by: step) {
            let vx = vox[i]
            guard vx.hits >= 1 else { continue }
            let v = (p: world(vx), f: flaw(vx))
            let k = cellKey(v.p)
            var a = cells[k] ?? CellAcc()
            a.n += step
            let f = v.f
            if f != .ok { a.w[f.rawValue] += step }
            let x = Double(v.p.x), y = Double(v.p.y), z = Double(v.p.z)
            a.sx += x; a.sy += y; a.sz += z
            a.sxx += x * x; a.sxy += x * y; a.sxz += x * z; a.syy += y * y; a.syz += y * z; a.szz += z * z
            cells[k] = a
            lo = min(lo, v.p.y); hi = max(hi, v.p.y)
        }
        if lo > hi { lo = 0; hi = 0 }
        bottomY = lo; topY = hi
        var weak: [SIMD3<Int32>: Flaw] = [:]
        var noisy = Set<SIMD3<Int32>>()
        for (k, a) in cells {
            if a.n >= 3 {
                let few = a.w[Flaw.few.rawValue], lc = a.w[Flaw.lowConf.rawValue], bl = a.w[Flaw.blurry.rawValue]
                if 2 * (few + lc + bl) >= a.n {
                    weak[k] = few >= lc && few >= bl ? .few : (lc >= bl ? .lowConf : .blurry)
                    continue
                }
            }
            if a.n >= 8 {
                // thickness of the surface in this 2 cm cell: sqrt of the smallest covariance eigenvalue
                let n = Double(a.n), mx = a.sx / n, my = a.sy / n, mz = a.sz / n
                let lmin = minEigenSym3(a.sxx / n - mx * mx, a.sxy / n - mx * my, a.sxz / n - mx * mz,
                                        a.syy / n - my * my, a.syz / n - my * mz, a.szz / n - mz * mz)
                if lmin > 0.004 * 0.004 { weak[k] = .noisy; noisy.insert(k) }
            }
        }
        // holes: an empty cell with scanned cells on both sides along at least two axes
        let axes = [SIMD3<Int32>(1, 0, 0), SIMD3<Int32>(0, 1, 0), SIMD3<Int32>(0, 0, 1)]
        var tried = Set<SIMD3<Int32>>()
        for k in cells.keys {
            for e in axes {
                for cand in [k &+ e, k &- e] where cells[cand] == nil && !tried.contains(cand) {
                    tried.insert(cand)
                    var both = 0
                    for a in axes where cells[cand &+ a] != nil && cells[cand &- a] != nil { both += 1 }
                    if both >= 2 { weak[cand] = .hole }
                }
            }
        }
        // auto-removed views that no good frame has covered since
        for k in removedCells where weak[k] == nil || weak[k] == .few { weak[k] = .removed }
        // settle what cannot be improved (see weakSince)
        let now = CACurrentMediaTime(), camT = lookT, camPos = position(camT)
        let fwd = -SIMD3<Float>(camT.columns.2.x, camT.columns.2.y, camT.columns.2.z)
        for k in Array(weak.keys) {
            if settled.contains(k) { weak[k] = nil; continue }
            let first = weakSince[k] ?? now; weakSince[k] = first
            let c = (SIMD3<Float>(Float(k.x), Float(k.y), Float(k.z)) + 0.5) * CELL, d = c - camPos, dist = simd_length(d)
            if dist < 2.5 && dist > 0.05 && simd_dot(d / dist, fwd) > 0.75 { weakSeen[k, default: 0] += 1 }
            if (weakSeen[k] ?? 0) >= 3 || now - first > 120 { settled.insert(k); weak[k] = nil }
        }
        for k in Array(weakSince.keys) where weak[k] == nil && !settled.contains(k) { weakSince[k] = nil; weakSeen[k] = nil }      // fixed by rescanning
        lastWeak = Set(weak.keys)
        noisyCells = noisy
        var mw = Set<SIMD2<Int32>>()
        for (k, f) in weak where f != .removed {
            mw.insert(mapKey((SIMD3<Float>(Float(k.x), Float(k.y), Float(k.z)) + 0.5) * CELL))
        }
        mapWeak = mw

        // group weak cells into clusters (26-neighbourhood)
        var seen = Set<SIMD3<Int32>>()
        var out: [WeakCluster] = []
        for (start, _) in weak where !seen.contains(start) {
            var stack = [start]; seen.insert(start)
            var members: [SIMD3<Int32>] = []
            var votes = [Int](repeating: 0, count: 7)
            while let c = stack.popLast() {
                members.append(c)
                if let f = weak[c] { votes[f.rawValue] += 1 }
                for dx in -1...1 { for dy in -1...1 { for dz in -1...1 {
                    let nb = c &+ SIMD3<Int32>(Int32(dx), Int32(dy), Int32(dz))
                    if weak[nb] != nil && !seen.contains(nb) { seen.insert(nb); stack.append(nb) }
                } } }
            }
            var best = 1
            for r in 1..<7 where votes[r] > votes[best] { best = r }
            let reason = Flaw(rawValue: best) ?? .few
            var mn = members[0], mx = members[0]
            var sum = SIMD3<Float>(repeating: 0)
            for m in members {
                mn = SIMD3<Int32>(min(mn.x, m.x), min(mn.y, m.y), min(mn.z, m.z))
                mx = SIMD3<Int32>(max(mx.x, m.x), max(mx.y, m.y), max(mx.z, m.z))
                sum += (SIMD3<Float>(Float(m.x), Float(m.y), Float(m.z)) + 0.5) * CELL
            }
            let ext = mx &- mn &+ SIMD3<Int32>(1, 1, 1)
            let wideDims = (ext.x >= 2 ? 1 : 0) + (ext.y >= 2 ? 1 : 0) + (ext.z >= 2 ? 1 : 0)
            let keep: Bool
            switch reason {
            case .hole, .removed: keep = members.count >= 2
            case .noisy: keep = members.count >= 4 && wideDims >= 2      // a patch, not the line of an ordinary edge
            default: keep = members.count >= 6
            }
            guard keep else { continue }
            let centre = sum / Float(members.count)
            let span = max(0.01, topY - bottomY)
            let rel = (centre.y - bottomY) / span
            let part = rel > 0.66 ? "top" : (rel < 0.33 ? "bottom" : "middle")
            out.append(WeakCluster(id: 0, centre: centre, size: SIMD3<Float>(Float(ext.x), Float(ext.y), Float(ext.z)) * CELL,
                                   reason: reason, cells: members.count, part: part,
                                   hint: "\(reason.verb) the \(part) of the scan (\(reason.label))"))
        }
        // removed areas first (they are what the reader is asked to redo), then the biggest
        out.sort { a, b in
            if (a.reason == .removed) != (b.reason == .removed) { return a.reason == .removed }
            return a.cells > b.cells
        }
        return out.prefix(6).enumerated().map { (i, c) in
            WeakCluster(id: i, centre: c.centre, size: c.size, reason: c.reason, cells: c.cells, part: c.part, hint: c.hint)
        }
    }

    /// red / amber wire spheres over the weak areas (main thread)
    private func rebuildMarkers() {
        for c in markerNode.childNodes { c.removeFromParentNode() }
        guard smartOn else { return }
        for c in clusters {
            let r = CGFloat(max(0.03, min(0.25, simd_length(c.size) / 2)))
            let s = SCNSphere(radius: r); s.segmentCount = 12
            let m = SCNMaterial()
            m.diffuse.contents = (c.reason.red ? UIColor.red : UIColor.orange).withAlphaComponent(0.6)
            m.lightingModel = .constant; m.fillMode = .lines; m.isDoubleSided = true
            s.materials = [m]
            let node = SCNNode(geometry: s)
            node.position = SCNVector3(c.centre.x, c.centre.y, c.centre.z)
            node.runAction(.repeatForever(.sequence([.fadeOpacity(to: 0.35, duration: 0.6), .fadeOpacity(to: 1, duration: 0.6)])))
            node.addChildNode(label("\(c.id + 1)  Rescan: \(c.reason.label)", color: c.reason.red ? .systemRed : .systemOrange, size: 0.035, y: Float(r) + 0.04))
            markerNode.addChildNode(node)
        }
    }

    /// 4 times a second (main thread): the mini map, the instructions and the arrow, relative to where the phone is now
    func updateGuide() {
        guard let cam = session?.currentFrame?.camera else { return }
        let camT = cam.transform, camPos = position(camT)
        lookT = camT
        let fwd = -SIMD3<Float>(camT.columns.2.x, camT.columns.2.y, camT.columns.2.z)
        let heading = atan2(fwd.x, -fwd.z)
        let sp = running ? speed : 0
        if abs(sp - shownSpeed) > 0.02 { shownSpeed = sp }
        // the 3D arrow: the next checkpoint first, then the nearest weak spot
        let arrowTarget: SIMD3<Float>? = running ? (checkpoints.next(from: camPos)?.1 ?? (smartOn ? clusters.min(by: { simd_length($0.centre - camPos) < simd_length($1.centre - camPos) })?.centre : nil)) : nil
        guideArrow.update(cam: camT, target: arrowTarget)
        if !mapBusy {
            mapBusy = true
            work.async {
                var m = MiniMap()
                m.good = Array(self.mapGood.subtracting(self.mapWeak))
                m.weak = Array(self.mapWeak)
                var red = Set<SIMD2<Int32>>()
                for k in self.removedCells { red.insert(mapKey((SIMD3<Float>(Float(k.x), Float(k.y), Float(k.z)) + 0.5) * CELL)) }
                m.red = Array(red)
                DispatchQueue.main.async {
                    m.phone = SIMD2<Float>(camPos.x, camPos.z); m.heading = heading
                    for (i, p) in self.checkpoints.points.enumerated() { (self.checkpoints.done[i] ? { m.cpsDone.append(SIMD2(p.x, p.z)) } : { m.cps.append(SIMD2(p.x, p.z)) })() }
                    self.map = m
                    self.mapBusy = false
                }
            }
        }

        guard smartOn, !clusters.isEmpty, let v = arView else { if !targetText.isEmpty { targetText = "" }; return }
        let b = v.bounds
        guard b.width > 0 else { return }
        var changed = clusters
        var nearest = -1
        var nearestDist: Float = .greatestFiniteMagnitude
        for i in changed.indices {
            let c = changed[i]
            let pr = v.projectPoint(SCNVector3(c.centre.x, c.centre.y, c.centre.z))
            let behind = pr.z > 1 || pr.z < 0
            var dx = CGFloat(pr.x) - b.midX, dy = CGFloat(pr.y) - b.midY
            if behind { dx = -dx; dy = -dy }
            let on = !behind && b.contains(CGPoint(x: CGFloat(pr.x), y: CGFloat(pr.y)))
            let dist = simd_length(c.centre - camPos)
            let dir: String
            if behind { dir = "behind you" }
            else if on { dir = "ahead" }
            else if abs(dx) > abs(dy) { dir = dx < 0 ? "left" : "right" }
            else { dir = dy < 0 ? "up" : "down" }
            var verb = c.reason.verb
            if c.reason == .few && dist < 0.6 { verb = "Sweep slowly over" }
            changed[i].hint = "\(verb) the \(c.part) of the scan, \(dir) \(Int((dist * 100).rounded())) cm (\(c.reason.label))"
            if dist < nearestDist {
                nearestDist = dist; nearest = i
                let ang = Double(atan2(dx, -dy))
                if abs(ang - arrowAngle) > 0.05 { arrowAngle = ang }
                if arrowOnScreen != on { arrowOnScreen = on }
                let tt = "Rescan: \(Int((dist * 100).rounded())) cm \(on ? "ahead (red/amber marks)" : dir)"
                if targetText != tt { targetText = tt }
            }
        }
        if nearest >= 0 { let r = changed[nearest].reason.red; if targetRed != r { targetRed = r } }
        if changed.map({ $0.hint }) != clusters.map({ $0.hint }) { clusters = changed }
    }

    // ---- export ----
    private func plyData() -> (Data, Int) {
        let (mp, mc) = snapshotLocked()
        let n = mp.count
        var data = Data("ply\nformat binary_little_endian 1.0\ncomment KindleHub Scan 3.3, LiDAR fused, anchor-corrected, merged on a 2 mm grid (>= \(MIN_HITS) hits, or 1 hit beside a confident point of the same colour), metres, ARKit world (y up)\nelement vertex \(n)\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".utf8)
        data.reserveCapacity(data.count + n * 15)
        for i in 0..<n {
            var q = mp[i]
            withUnsafeBytes(of: &q) { data.append(contentsOf: $0.prefix(12)) }
            data.append(contentsOf: [mc[i].x, mc[i].y, mc[i].z])
        }
        return (data, n)
    }

    private static func stamp() -> String { let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd-HHmm"; return f.string(from: Date()) }

    func export(_ done: @escaping (URL?) -> Void) {
        running = false; setStatus("Saving...")
        work.async {
            let (data, n) = self.plyData()
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("KindleHub-scan-\(Scanner.stamp()).ply")
            do { try data.write(to: url); DispatchQueue.main.async { self.setStatus("Saved \(n.formatted()) points"); done(url) } }
            catch { DispatchQueue.main.async { self.setStatus("Could not save: \(error.localizedDescription)", warn: true); done(nil) } }
        }
    }

    /// the capture folder (frames/) + cloud.ply + quality.json, zipped for AirDrop to the Mac
    func exportCapture(_ done: @escaping (URL?) -> Void) {
        running = false; setStatus("Packing the capture...")
        guard let dir = ensureCaptureDir() else { setStatus("Could not make the capture folder", warn: true); done(nil); return }
        let nkf = keyframes
        work.async {
            let (ply, n) = self.plyData()
            let found = self.runAnalysis()
            self.io.sync {}                              // wait for keyframes still being written
            var stdSum: Double = 0, stdN = 0
            for v in self.vox where v.hits > 2 { stdSum += Double((v.m2 / max(1e-6, v.pw)).squareRoot()); stdN += 1 }
            let q: [String: Any] = [
                "clusters": found.map { c -> [String: Any] in
                    ["center": [Double(c.centre.x), Double(c.centre.y), Double(c.centre.z)],
                     "size": Double(max(c.size.x, max(c.size.y, c.size.z))),
                     "reason": c.reason.key]
                },
                "summary": [
                    "model": "Smart check: a heuristic quality model (rules), not a neural network",
                    "points": n, "keyframes": nkf, "voxel_size_m": Double(VOXEL), "merge_voxel_m": Double(MERGE_VOXEL),
                    "anchors": self.anchorT.count - 1, "cell_size_m": Double(CELL),
                    "mean_voxel_std_m": stdN > 0 ? stdSum / Double(stdN) : 0,
                    "removed_cells": self.removedCells.count
                ] as [String: Any]
            ]
            do {
                try ply.write(to: dir.appendingPathComponent("cloud.ply"))
                try JSONSerialization.data(withJSONObject: q, options: [.prettyPrinted, .sortedKeys]).write(to: dir.appendingPathComponent("quality.json"))
            } catch {
                DispatchQueue.main.async { self.setStatus("Could not save: \(error.localizedDescription)", warn: true); done(nil) }
                return
            }
            var err: NSError?
            var out: URL?
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent("KindleHub-capture-\(Scanner.stamp()).zip")
            NSFileCoordinator().coordinate(readingItemAt: dir, options: [.forUploading], error: &err) { zip in
                try? FileManager.default.removeItem(at: dest)
                do { try FileManager.default.copyItem(at: zip, to: dest); out = dest } catch { out = nil }
            }
            let result = out
            let why = err?.localizedDescription ?? "unknown"
            DispatchQueue.main.async {
                if result != nil { self.setStatus("Packed \(nkf) photos + \(n.formatted()) points") }
                else { self.setStatus("Could not zip: \(why)", warn: true) }
                done(result)
            }
        }
    }
}

/// 5.0: SceneKit reads vertex colours as LINEAR light; scan colours are sRGB (like the photos), so without this every
/// scan showed washed out (mid-grey 50% drew at 78%). One table lookup per channel.
let SRGB_TO_LINEAR: [Float] = (0..<1024).map { i in let c = Float(i) / 1023; return c <= 0.04045 ? c / 12.92 : powf((c + 0.055) / 1.055, 2.4) }
@inline(__always) func linear(_ c: Float) -> Float { SRGB_TO_LINEAR[Int(max(0, min(1, c.isFinite ? c : 0)) * 1023)] }

func pointGeometry(_ verts: [SCNVector3], _ colsIn: [Float], size: CGFloat) -> SCNGeometry {
    let vs = SCNGeometrySource(vertices: verts)
    let cols = colsIn.map(linear)
    let cd = Data(bytes: cols, count: cols.count * MemoryLayout<Float>.size)
    let cs = SCNGeometrySource(data: cd, semantic: .color, vectorCount: verts.count, usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
    let el = SCNGeometryElement(data: nil, primitiveType: .point, primitiveCount: verts.count, bytesPerIndex: 4)
    el.pointSize = size; el.minimumPointScreenSpaceRadius = 1; el.maximumPointScreenSpaceRadius = size + 2
    let geo = SCNGeometry(sources: [vs, cs], elements: [el])
    let m = SCNMaterial(); m.lightingModel = .constant; geo.materials = [m]
    return geo
}

// ---------- View scan: the cloud in 3D with editing tools ----------
struct ScanViewer: View {
    @ObservedObject var scanner: Scanner
    @Environment(\.dismiss) private var dismiss
    @State private var pointSize: Double = 3
    @State private var lowCut: Double = 0        // 0..1 of the cloud's height range
    @State private var highCut: Double = 1
    @State private var radius: Double = 1        // 0..1 of the cloud's widest reach from its centre
    @State private var busy = false
    @State private var stamp = 0                 // bumps to rebuild the 3D view

    var body: some View {
        ZStack(alignment: .bottom) {
            CloudView(scanner: scanner, pointSize: pointSize, lowCut: lowCut, highCut: highCut, radius: radius, stamp: stamp)
                .ignoresSafeArea()
            VStack(spacing: 10) {
                HStack {
                    Text("\(scanner.count.formatted()) points").font(.headline.monospacedDigit())
                    Spacer()
                    Button("Done") { dismiss() }.buttonStyle(.borderedProminent)
                }
                slider("Point size", $pointSize, 1...8)
                slider("Cut floor", $lowCut, 0...0.95)
                slider("Cut top", $highCut, 0.05...1)
                slider("Keep within", $radius, 0.05...1)
                HStack(spacing: 10) {
                    Button("Clean noise") { run { scanner.cleanNoise() } }
                    Button("Apply cuts") { let l = lowCut, h = highCut, r = radius; run { scanner.applyCuts(low: l, high: h, radius: r) }; lowCut = 0; highCut = 1; radius = 1 }
                    Button("Undo") { run { scanner.undo() } }.disabled(!scanner.canUndo)
                }
                .buttonStyle(.bordered).disabled(busy)
                Text("Drag to turn, pinch to zoom, two fingers to move.").font(.caption).foregroundStyle(.secondary)
            }
            .padding(16).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22)).padding()
        }
        .background(Color.black)
    }
    private func slider(_ name: String, _ v: Binding<Double>, _ r: ClosedRange<Double>) -> some View {
        HStack { Text(name).font(.subheadline).frame(width: 96, alignment: .leading); Slider(value: v, in: r) }
    }
    private func run(_ f: @escaping () -> Void) {
        busy = true
        DispatchQueue.global(qos: .userInitiated).async { f(); DispatchQueue.main.async { busy = false; stamp += 1 } }
    }
}

struct CloudView: UIViewRepresentable {
    let scanner: Scanner
    let pointSize: Double, lowCut: Double, highCut: Double, radius: Double, stamp: Int
    final class Coord {
        var key = ""; var node = SCNNode()
        // 5.7 (48): one build at a time. A slider drag started a full-cloud build per tick (~100 MB each on a big
        // scan, several running at once) and memory ran out; now only the newest settings are built when one finishes
        var building = false, want: (key: String, lo: Double, hi: Double, ra: Double)?, ps: CGFloat = 3
        func build(_ sc: Scanner) {
            guard !building, let w = want else { return }
            building = true; want = nil
            DispatchQueue.global(qos: .userInitiated).async {
                let g = sc.geometry(low: w.lo, high: w.hi, radius: w.ra, maxPoints: 1_000_000)
                DispatchQueue.main.async {
                    self.building = false
                    if self.key == w.key {
                        self.node.geometry = g
                        g?.elements.first?.pointSize = self.ps
                        g?.elements.first?.maximumPointScreenSpaceRadius = self.ps * 1.5
                    }
                    self.build(sc)
                }
            }
        }
    }
    func makeCoordinator() -> Coord { Coord() }
    func makeUIView(context: Context) -> SCNView {
        let v = SCNView(frame: .zero); v.backgroundColor = .black; v.allowsCameraControl = true; v.antialiasingMode = .multisampling4X
        let scene = SCNScene(); scene.rootNode.addChildNode(context.coordinator.node); v.scene = scene
        let cam = SCNNode(); cam.camera = SCNCamera(); cam.camera?.zNear = 0.01; cam.position = SCNVector3(0, 0, 2.2)
        scene.rootNode.addChildNode(cam); v.pointOfView = cam
        return v
    }
    func updateUIView(_ v: SCNView, context: Context) {
        let key = "\(stamp)|\(lowCut)|\(highCut)|\(radius)"
        let coord = context.coordinator
        coord.ps = CGFloat(pointSize)
        if key != coord.key {
            coord.key = key
            // 3.4: built OFF the main thread (merging millions of points on main froze the app and iOS killed it),
            // and at most 1M points drawn; the export still has every point
            coord.want = (key, lowCut, highCut, radius)
            coord.build(scanner)
        }
        context.coordinator.node.geometry?.elements.first?.pointSize = CGFloat(pointSize)
        context.coordinator.node.geometry?.elements.first?.maximumPointScreenSpaceRadius = CGFloat(pointSize * 1.5)
    }
}

/// the viewer's last merged cloud (the merge is the slow part; slider moves reuse it)
/// 5.7 (47): keyed on everything merged() reads that can change: voxel count, good count, anchor poses, MIN_HITS, MERGE_VOXEL
fileprivate var snapCache: (n: Int, g: Int, a: Int, mh: UInt16, mv: Float, s: ([SIMD3<Float>], [SIMD3<UInt8>]))?

/// editing works on the voxels themselves, so the averages and Smart check statistics survive a cut
extension Scanner {
    struct Bounds { let lo: SIMD3<Float>; let hi: SIMD3<Float>; let centre: SIMD3<Float>; let reach: Float }

    var canUndo: Bool { work.sync { !undoStack.isEmpty } }

    /// people are removed from scans by default; a Person scan needs them kept
    func setPeopleMasking(_ on: Bool) { run(withPeople: on, reset: false) }
    /// re-applies the camera set-up after a mode changed it (video format, people)
    func reconfigure(people: Bool) { run(withPeople: people, reset: true) }

    /// checkpoints around what has been scanned so far (object / person: two rings at the current distance)
    func placeCheckpoints(_ kind: ScanKind) {
        work.async {
            var sum = SIMD3<Float>.zero, n: Float = 0
            let step = max(1, self.vox.count / 60000)
            for i in stride(from: 0, to: self.vox.count, by: step) where self.vox[i].hits >= MIN_HITS { sum += self.world(self.vox[i]); n += 1 }
            guard n > 100 else { return }
            let c = sum / n, r = max(0.45, min(1.2, self.medianDepth))
            DispatchQueue.main.async {
                let rings: [(dy: Float, count: Int)] = kind == .person ? [(0.25, 8), (-0.35, 8)] : [(0.05, 8), (0.4, 6)]
                self.checkpoints.place(centre: c, radius: r, rings: rings)
            }
        }
    }

    /// the capture folder's name under Documents/captures (keyframe photos), if any were saved
    var captureName: String? { captureDir?.lastPathComponent }

    /// the finished cloud for the library (call off the main thread)
    func finishedCloud() -> Cloud {
        // 5.0: a 12.6M-point save crashed on memory. Everything only scanning needs (the voxel index, density cells,
        // map sets, undo copies of the whole scan, the cached merge) is freed first: ~600 MB to 2 GB on a big scan.
        let t0 = CFAbsoluteTimeGetCurrent()
        let (p, c, reused): ([SIMD3<Float>], [SIMD3<UInt8>], Bool) = work.sync {
            // 5.7 (47): the merge made when the scan was paused is reused if nothing it depends on changed since;
            // it is taken out of the cache first so only one copy stays alive
            var cached: ([SIMD3<Float>], [SIMD3<UInt8>])?
            if snapCacheValid() { cached = snapCache!.s }
            index = FlatIndex(); cellCount = [:]; mapGood = []; mapWeak = []; undoStack = []; noisyCells = []; snapCache = nil
            // 5.8 (53): the voxels go too once the merge is taken (13.6M voxels are ~490 MB, kept alive through the save,
            // polish and photo surface - a save crashed); the scan screen closes after saving
            if let m = cached { vox = []; return (m.0, m.1, true) }
            let m = merged(); vox = []
            return (m.p, m.c, false)
        }
        Log.add("save merge \(reused ? "reused from pause" : "recomputed") points=\(p.count) in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - t0)) s")
        var o = Cloud(); o.p = p; o.c = c; return o
    }

    /// the merged cloud, cached until the voxels change (work queue only)
    func snapshotLocked() -> ([SIMD3<Float>], [SIMD3<UInt8>]) {
        if snapCacheValid(), let c = snapCache { return c.s }
        let m = merged()
        snapCache = (vox.count, goodN, anchorGen, MIN_HITS, MERGE_VOXEL, (m.p, m.c))
        return (m.p, m.c)
    }
    /// 5.7 (47): the cached merge still matches the voxels, the anchor poses and the merge settings (work queue only)
    private func snapCacheValid() -> Bool {
        guard let c = snapCache else { return false }
        return c.n == vox.count && c.g == goodN && c.a == anchorGen && c.mh == MIN_HITS && c.mv == MERGE_VOXEL
    }
    /// good (shown) points and colours
    var snapshot: ([SIMD3<Float>], [SIMD3<UInt8>]) {
        work.sync { snapshotLocked() }
    }

    private static func bounds(_ pts: [SIMD3<Float>]) -> Bounds {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for p in pts { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let c = (lo + hi) / 2
        var reach: Float = 0.001
        for p in pts { reach = max(reach, simd_length(SIMD2<Float>(p.x - c.x, p.z - c.z))) }
        return Bounds(lo: lo, hi: hi, centre: c, reach: reach)
    }

    private static func passes(_ p: SIMD3<Float>, _ b: Bounds, low: Double, high: Double, radius: Double) -> Bool {
        let h0 = b.lo.y + Float(low) * (b.hi.y - b.lo.y), h1 = b.lo.y + Float(high) * (b.hi.y - b.lo.y), r = Float(radius) * b.reach
        return p.y >= h0 && p.y <= h1 && simd_length(SIMD2<Float>(p.x - b.centre.x, p.z - b.centre.z)) <= r
    }

    /// the cloud centred on its middle, for the viewer
    func geometry(low: Double, high: Double, radius: Double, maxPoints: Int) -> SCNGeometry? {
        let (all, cols) = snapshot
        guard !all.isEmpty else { return nil }
        let b = Scanner.bounds(all)
        let keep = all.map { Scanner.passes($0, b, low: low, high: high, radius: radius) }
        let n = keep.filter { $0 }.count
        guard n > 0 else { return nil }
        let step = max(1, n / maxPoints)
        var verts: [SCNVector3] = []; var cf: [Float] = []
        verts.reserveCapacity(n / step + 1); cf.reserveCapacity(3 * (n / step + 1))
        var k = 0
        for i in 0..<all.count where keep[i] {
            if k % step == 0 {
                let p = all[i] - b.centre, q = cols[i]
                verts.append(SCNVector3(p.x, p.y, p.z)); cf += [Float(q.x) / 255, Float(q.y) / 255, Float(q.z) / 255]
            }
            k += 1
        }
        return pointGeometry(verts, cf, size: 3)
    }

    /// keep only voxels that pass `test`; work queue only
    private func filterLocked(_ test: (Vox) -> Bool) {
        // 5.0: undo keeps only what this removal takes out (was two whole copies of the scan - up to 1.6 GB)
        var kept: [Vox] = [], gone: [Vox] = []
        kept.reserveCapacity(vox.count)
        for v in vox { if test(v) { kept.append(v) } else { gone.append(v) } }
        undoStack.append(gone); if undoStack.count > 3 { undoStack.removeFirst() }
        setLocked(kept)
    }
    /// frees room: voxels with a single sighting go; everything seen twice or more stays (work queue only)
    func compact() {
        let before = vox.count
        // 5.0: a compaction that freed almost nothing is not repeated until the scan has grown 10% (it rebuilt a
        // 13M-entry index every 8 s for minutes, freeing nothing, making scanning lag)
        if before < compactFloor { return }
        undoStack = []                                                   // whole-scan copies go first
        vox.removeAll { !($0.hits >= 2 || $0.seen >= 3) }
        if vox.count > before * 9 / 10 { vox.removeAll { $0.hits < 2 } }
        Log.add("compact voxels=\(before) -> \(vox.count)")
        compactFloor = vox.count > before * 95 / 100 ? vox.count * 11 / 10 : 0
        reindex()
    }
    private func setLocked(_ v: [Vox]) { vox = v; reindex() }
    private func reindex() {
        snapCache = nil
        cellCount.removeAll(keepingCapacity: true)
        for x in vox { let k = voxKey(world(x), 0.006); cellCount[k] = (cellCount[k] ?? 0) &+ 1 }
        index.removeAll(keepingCapacity: true); index.reserveCapacity(vox.count + vox.count / 5)
        mapGood.removeAll(keepingCapacity: true)
        for (i, x) in vox.enumerated() {
            index[VKey(a: x.a, k: voxKey(x.p))] = Int32(i)
            if x.hits >= MIN_HITS { mapGood.insert(mapKey(world(x))) }
        }
        var g = 0; for x in vox where x.hits >= MIN_HITS { g += 1 }; goodN = g; goodAt = MIN_HITS
        let n = goodCount
        DispatchQueue.main.async { self.count = n; self.objectWillChange.send() }
        updatePreview()
    }

    func applyCuts(low: Double, high: Double, radius: Double) {
        let (all, _) = snapshot
        guard !all.isEmpty else { return }
        let b = Scanner.bounds(all)
        work.sync { filterLocked { Scanner.passes(self.world($0), b, low: low, high: high, radius: radius) } }
    }

    /// drop points with fewer than 4 shown neighbours in their 2 cm cell (stray specks)
    func cleanNoise() {
        work.sync {
            var cnt: [SIMD3<Int32>: Int32] = [:]
            for v in vox where v.hits >= MIN_HITS { cnt[cellKey(world(v)), default: 0] += 1 }
            filterLocked { (cnt[cellKey(self.world($0))] ?? 0) >= 4 }
        }
    }

    func undo() { work.sync { if let gone = undoStack.popLast() { vox += gone; reindex() } } }
}

/// the phone's torch while scanning (ARKit owns the camera; the torch is set on the same device)
final class Torch: ObservableObject {
    @Published var on = false
    @Published var level: Double = 0.6
    private var device: AVCaptureDevice? { AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) }
    func toggle() { if on { off() } else { on = true; apply() } }
    func apply() {
        guard let d = device, d.hasTorch else { on = false; return }
        do {
            try d.lockForConfiguration()
            if on { try d.setTorchModeOn(level: Float(max(0.05, min(1, level)))) } else { d.torchMode = .off }
            d.unlockForConfiguration()
        } catch { on = false }
    }
    func off() { guard on else { return }; on = false; apply() }
}

/// green when the pace is right for this mode, amber when it is quick, red when points will be dropped
struct SpeedBar: View {
    let speed: Float
    var body: some View {
        let limit = FAST_MOVE, f = CGFloat(min(1, speed / limit))
        let col: Color = speed < limit * 0.5 ? .green : (speed < limit * 0.85 ? .yellow : .red)
        HStack(spacing: 6) {
            Image(systemName: "speedometer").font(.caption)
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.25)).frame(width: 46, height: 6)
                Capsule().fill(col).frame(width: max(4, 46 * f), height: 6)
            }
        }
        .padding(.horizontal, 10).padding(.vertical, 8).background(.ultraThinMaterial, in: Capsule())
    }
}

/// LiDAR-only colouring: a survey-style ramp by height (ARKit's origin is where the phone started, about chest height)
@inline(__always) func heightColour(_ y: Float) -> SIMD3<Float> {
    let t = max(0, min(1, (y + 1.6) / 3.2))
    let stops: [SIMD3<Float>] = [SIMD3(40, 70, 210), SIMD3(20, 190, 220), SIMD3(70, 210, 90), SIMD3(240, 210, 40), SIMD3(235, 70, 50)]
    let f = t * Float(stops.count - 1), i = min(stops.count - 2, Int(f)), k = f - Float(i)
    return stops[i] * (1 - k) + stops[i + 1] * k
}


/// 5.6 the blue "not scanned yet" tint over the live camera (Polycam's clearest idea): soft-edged, gone where the
/// LiDAR has covered the surface
struct CoverageOverlay: View {
    let mask: [Bool]
    var body: some View {
        Canvas { ctx, size in
            let C = Scanner.maskCols, R = Scanner.maskRows
            guard mask.count == C * R else { return }
            let w = size.width / CGFloat(C), h = size.height / CGFloat(R)
            var p = Path()
            for r in 0..<R { for c in 0..<C where !mask[r * C + c] { p.addRect(CGRect(x: CGFloat(c) * w - 1, y: CGFloat(r) * h - 1, width: w + 2, height: h + 2)) } }
            ctx.fill(p, with: .color(Color(red: 0.25, green: 0.35, blue: 1).opacity(0.42)))
        }
        .blur(radius: 10)
        .animation(.easeOut(duration: 0.25), value: mask)
    }
}


/// 5.6 (38) the white wireframe over scanned surfaces (Polycam's), from ARKit's scene mesh. Each mesh piece is rebuilt
/// at most twice a second, off the main thread; SceneKit draws it
final class MeshOverlay {
    let node = SCNNode()
    private var parts: [UUID: SCNNode] = [:], built: [UUID: TimeInterval] = [:], busy = Set<UUID>()
    private let q = DispatchQueue(label: "mesh.overlay", qos: .utility)
    private lazy var material: SCNMaterial = {
        let m = SCNMaterial()
        m.diffuse.contents = UIColor.white; m.lightingModel = .constant; m.fillMode = .lines
        m.isDoubleSided = true; m.writesToDepthBuffer = false; m.transparency = 0.35
        return m
    }()

    /// ARSession delegate thread
    func update(_ anchors: [ARMeshAnchor]) {
        guard !anchors.isEmpty else { return }
        let now = CACurrentMediaTime()
        q.async {
            for a in anchors where !self.busy.contains(a.identifier) && now - (self.built[a.identifier] ?? 0) > 0.5 {
                self.built[a.identifier] = now; self.busy.insert(a.identifier)
                let g = MeshOverlay.geometry(a.geometry), id = a.identifier, T = a.transform
                g?.materials = [self.material]
                DispatchQueue.main.async {
                    let n = self.parts[id] ?? { let n = SCNNode(); self.node.addChildNode(n); self.parts[id] = n; return n }()
                    n.geometry = g; n.simdTransform = T
                    self.q.async { self.busy.remove(id) }
                }
            }
        }
    }
    func remove(_ anchors: [ARAnchor]) {
        let ids = anchors.map(\.identifier)
        DispatchQueue.main.async { for id in ids { self.parts.removeValue(forKey: id)?.removeFromParentNode() } }
        q.async { for id in ids { self.built[id] = nil } }
    }

    private static func geometry(_ m: ARMeshGeometry) -> SCNGeometry? {
        let v = m.vertices, f = m.faces
        guard v.count > 0, f.count > 0, f.indexCountPerPrimitive == 3 else { return nil }
        let vd = Data(bytes: v.buffer.contents().advanced(by: v.offset), count: v.stride * v.count)
        let src = SCNGeometrySource(data: vd, semantic: .vertex, vectorCount: v.count, usesFloatComponents: true,
                                    componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: v.stride)
        let fd = Data(bytes: f.buffer.contents(), count: f.count * 3 * f.bytesPerIndex)
        let el = SCNGeometryElement(data: fd, primitiveType: .triangles, primitiveCount: f.count, bytesPerIndex: f.bytesPerIndex)
        return SCNGeometry(sources: [src], elements: [el])
    }
}
