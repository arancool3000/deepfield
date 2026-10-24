import SwiftUI
import simd

extension CloudOps {
    /// the floor height: the biggest level plane, else the lowest 3% of points
    static func floorHeight(_ cl: Cloud) -> Float {
        if let (n, d) = floorPlane(cl), n.y > 0.95 { return -d / n.y }
        let ys = sample(cl.p, grid: 0.05, max: 50000).map { $0.y }.sorted()
        return ys.isEmpty ? 0 : ys[ys.count * 3 / 100]
    }

    /// 4.5 merge alignment: floors first (height), then a top-down map of everything above the floor is matched at
    /// every 5 degrees and every position (floor plans slid over each other), then the best five are refined by ICP.
    static func align2(_ src: Cloud, to ref: Cloud, progress: (Float) -> Void) -> (simd_float4x4, Float) {
        let fr = floorHeight(ref), fs = floorHeight(src), dy = fr - fs
        let cell: Float = 0.25
        func cells(_ pts: [SIMD3<Float>], above f: Float) -> [SIMD2<Float>] {
            var seen = Set<SIMD2<Int32>>(), out: [SIMD2<Float>] = []
            for p in pts where p.y > f + 0.12 {
                let k = SIMD2<Int32>(ik((p.x / cell)), ik((p.z / cell)))
                if seen.insert(k).inserted { out.append((SIMD2<Float>(Float(k.x), Float(k.y)) + 0.5) * cell) }
            }
            return out
        }
        let rp = sample(ref.p, grid: 0.05, max: 300_000), sp = sample(src.p, grid: 0.05, max: 300_000)
        let rc = cells(rp, above: fr)
        var sc = cells(sp, above: fs)
        guard rc.count > 10, sc.count > 10 else { return align(src, to: ref, progress: progress) }
        if sc.count > 900 { let st = sc.count / 900; sc = stride(from: 0, to: sc.count, by: st).map { sc[$0] } }
        let refSet = Set(rc.map { SIMD2<Int32>(ik(($0.x / cell)), ik(($0.y / cell))) })
        let rmid = rc.reduce(SIMD2<Float>.zero, +) / Float(rc.count), smid = sc.reduce(SIMD2<Float>.zero, +) / Float(sc.count)
        var cands: [(Float, Float, Float, Int)] = []             // yaw, tx, tz, score
        let span: Float = 3.5
        for yi in 0..<72 {
            let th = Float(yi) * 5 * .pi / 180, c = cos(th), s = sin(th)
            let rot = sc.map { q -> SIMD2<Float> in let d = q - smid; return SIMD2(c * d.x + s * d.y, -s * d.x + c * d.y) }
            var sx = -span
            while sx <= span {
                var sz = -span
                while sz <= span {
                    let off = rmid + SIMD2(sx, sz)
                    var score = 0
                    for q in rot { let w = q + off; if refSet.contains(SIMD2<Int32>(ik((w.x / cell)), ik((w.y / cell)))) { score += 1 } }
                    if cands.count < 5 || score > cands[cands.count - 1].3 {
                        cands.append((th, off.x, off.y, score)); cands.sort { $0.3 > $1.3 }; if cands.count > 5 { cands.removeLast() }
                    }
                    sz += cell
                }
                sx += cell
            }
            progress(Float(yi + 1) / 72 * 0.5)
        }
        // each candidate as a 3D transform (turn about y around the source middle, then move), refined by ICP
        let r1 = sample(ref.p, grid: 0.025, max: 60000), g1 = GridIndex(r1, cell: 0.15)
        var best = (matrix_identity_float4x4, Float(-1))
        for (ci, cnd) in cands.enumerated() {
            let R = simd_float3x3(simd_quatf(angle: cnd.0, axis: SIMD3(0, 1, 0)))      // same turn as the 2-D search: x' = c x + s z
            let sm3 = SIMD3<Float>(smid.x, fs, smid.y)
            var T = m44(R); T.columns.3 = SIMD4(SIMD3<Float>(cnd.1, fs + dy, cnd.2) - R * sm3, 1)
            let s1 = sample(src.p, grid: 0.025, max: 8000)
            var fit: Float = 0
            for md: Float in [0.15, 0.08, 0.04] { (T, fit) = icp(s1, r1, g1, start: T, maxDist: md, iterations: 15) }
            if fit > best.1 { best = (T, fit) }
            progress(0.5 + Float(ci + 1) / Float(cands.count) * 0.3)
        }
        var (T, fit) = best
        for (i, v) in [Float(0.008), 0.003].enumerated() {
            let r = sample(ref.p, grid: v, max: 400_000), s = sample(src.p, grid: v, max: 30000)
            (T, fit) = icp(s, r, GridIndex(r, cell: v * 3), start: T, maxDist: v * 3, iterations: 25)
            progress(0.8 + Float(i + 1) * 0.1)
        }
        // the old turn-only search as a second opinion
        let old = align(src, to: ref) { _ in }
        return old.1 > fit + 0.05 ? old : (T, fit)
    }
}

/// a merge step waiting for the user: both scans overlaid, nudges, Refine, Accept / Skip
final class ReviewJob: Identifiable {
    let id = UUID()
    let name: String
    let base: Cloud, src: Cloud
    var T: simd_float4x4
    var fit: Float
    let done: (simd_float4x4?) -> Void
    init(name: String, base: Cloud, src: Cloud, T: simd_float4x4, fit: Float, done: @escaping (simd_float4x4?) -> Void) {
        self.name = name; self.base = CloudOps.downsample(base, to: 350_000); self.src = CloudOps.downsample(src, to: 350_000); self.T = T; self.fit = fit; self.done = done
    }
}

struct AlignReview: View {
    let job: ReviewJob
    @Environment(\.dismiss) private var dismiss
    @State private var cur = matrix_identity_float4x4          // the working transform of the new scan
    @State private var started = false
    @State private var show = 0                                // 0 both, 1 base only, 2 new only
    @State private var mode = 1                                // 0 nudge, 1 matching points
    @State private var fine = false                            // 5.0: manual tools hidden until asked for
    @State private var guess = 0                               // which automatic guess is shown (turns of 90 degrees)
    @State private var fitNow: Float = 0
    @State private var stepMM = 10.0
    @State private var stepDeg = 1.0
    @State private var pairsBase: [SIMD3<Float>] = []
    @State private var pairsNew: [SIMD3<Float>] = []
    @State private var stamp = 0
    @State private var preview = Cloud()
    @State private var fitText = ""
    @State private var refining = false
    @State private var answered = false
    @State private var buildGen = 0                            // 5.7 (46): only the newest overlay build may land

    private var expectingBase: Bool { pairsBase.count == pairsNew.count }
    private var shownNow: Int { fine && mode == 1 ? (expectingBase ? 1 : 2) : show }
    private var verdict: (String, Color) {
        if fitNow >= 0.55 { return ("Great match - \(Int(fitNow * 100))% of the new scan sits on the base", .green) }
        if fitNow >= 0.35 { return ("Probably right - \(Int(fitNow * 100))% overlap. Have a look.", .yellow) }
        return ("Doubtful - only \(Int(fitNow * 100))% overlap. Try another way.", .orange)
    }

    private func rebuild() {
        buildGen += 1
        let T = cur, b = job.base, s = job.src, sh = shownNow, g = buildGen
        Task.detached {
            var o = Cloud()
            if sh != 2 { o.p = b.p; o.c = b.c.map { k in SIMD3(UInt8(min(255, Int(k.x) / 2 + 20)), UInt8(min(255, Int(k.y) / 2 + 60)), UInt8(min(255, Int(k.z) / 2 + 120))) } }      // base: blue
            if sh != 1 { for i in 0..<s.count { o.p.append(CloudOps.apply(T, s.p[i])); let k = s.c[i]; o.c.append(SIMD3(UInt8(min(255, Int(k.x) / 2 + 128)), UInt8(min(255, Int(k.y) / 2 + 70)), UInt8(Int(k.z) / 3))) } }  // new: orange
            await MainActor.run { guard g == buildGen else { return }; preview = o; stamp += 1 }      // 5.7 (46): a slower, older build is dropped
        }
    }
    private var pins: [Pin] {
        pairsBase.enumerated().map { Pin(x: $1.x, y: $1.y, z: $1.z, text: "\($0 + 1) blue") } +
        pairsNew.enumerated().map { Pin(x: $1.x, y: $1.y, z: $1.z, text: "\($0 + 1) orange") }
    }
    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                PointCloudView(cloud: preview, stamp: stamp, pointSize: 2, background: "black", mode: "photo",
                               onTap: fine && mode == 1 ? { p in tap(p) } : nil, pins: fine ? pins : [])
                    .frame(maxHeight: .infinity)
                Text("Blue is the base, orange is \(job.name).").font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    Circle().fill(verdict.1).frame(width: 10, height: 10)
                    Text(refining ? "Snapping into place..." : verdict.0).font(.subheadline.weight(.medium)).lineLimit(2)
                }.padding(.horizontal)
                HStack(spacing: 10) {
                    Button { nextGuess() } label: { Label("Try another way", systemImage: "arrow.triangle.2.circlepath").frame(maxWidth: .infinity) }
                        .buttonStyle(.bordered).controlSize(.large).disabled(refining)
                    Button { answer(cur) } label: { Label("Looks right", systemImage: "checkmark").frame(maxWidth: .infinity) }
                        .buttonStyle(.borderedProminent).controlSize(.large).disabled(refining)
                }.padding(.horizontal)
                DisclosureGroup("Fine-tune by hand", isExpanded: $fine) {
                    VStack(spacing: 6) {
                        Picker("", selection: $mode) { Text("Match points").tag(1); Text("Nudge").tag(0) }.pickerStyle(.segmented)
                            .onChange(of: mode) { _, _ in rebuild() }
                        if mode == 1 { pairsPanel } else { nudgePanel }
                        HStack {
                            Button(refining ? "Snapping..." : "Snap into place") { refine() }.buttonStyle(.bordered).disabled(refining)
                            Button("Start over") { cur = job.T; guess = 0; fitNow = job.fit; pairsBase = []; pairsNew = []; rebuild() }.buttonStyle(.bordered)
                                .disabled(refining)      // 5.7 (46): the refine result would overwrite it
                        }
                    }
                }
                .onChange(of: fine) { _, _ in rebuild() }
                .padding(.horizontal).padding(.bottom, 6)
            }
            .navigationTitle("Line up the scans").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Skip this scan") { answer(nil) } }
            }
            .onAppear { if !started { started = true; cur = job.T; fitNow = job.fit; fitText = ""; rebuild() } }
        }
        .interactiveDismissDisabled()
    }

    /// 5.0: the next automatic guess - the new scan turned another 90 degrees about its middle, then snapped (ICP)
    private func nextGuess() {
        guess = (guess + 1) % 4
        let c = CloudOps.apply(job.T, job.src.bounds.centre)
        let R = m44(simd_float3x3(simd_quatf(angle: Float(guess) * .pi / 2, axis: SIMD3(0, 1, 0))))
        var a = matrix_identity_float4x4; a.columns.3 = SIMD4(-c, 1)
        var b = matrix_identity_float4x4; b.columns.3 = SIMD4(c, 1)
        cur = b * R * a * job.T
        rebuild(); refine()
    }

    // ---- matching points ----
    private var pairsPanel: some View {
        VStack(spacing: 6) {
            Text(expectingBase ? "Tap point \(pairsBase.count + 1) on the BLUE scan (a corner, an edge, a switch...)" : "Now tap the SAME spot on the ORANGE scan")
                .font(.subheadline.weight(.medium)).multilineTextAlignment(.center).padding(.horizontal)
            HStack {
                Button("Undo") { if pairsNew.count == pairsBase.count, !pairsNew.isEmpty { pairsNew.removeLast() } else if !pairsBase.isEmpty { pairsBase.removeLast() }; rebuild() }
                    .disabled(pairsBase.isEmpty)
                Button("Clear") { pairsBase = []; pairsNew = []; rebuild() }.disabled(pairsBase.isEmpty)
                Button("Align with \(pairsNew.count) pair\(pairsNew.count == 1 ? "" : "s")") { solvePairs() }
                    .buttonStyle(.borderedProminent).disabled(pairsNew.isEmpty || !expectingBase || refining)
            }.buttonStyle(.bordered).font(.caption)
            Text("1 pair moves it, 2 pairs also turn it, 3 or more fit it exactly.").font(.caption2).foregroundStyle(.secondary)
        }
    }
    private func tap(_ p: SIMD3<Float>) {
        if expectingBase { pairsBase.append(p) } else { pairsNew.append(p) }
        rebuild()
    }
    /// the transform taking the orange picks onto the blue ones, applied on top of the current one
    private func solvePairs() {
        let a = pairsNew, b = pairsBase, n = min(a.count, b.count)
        guard n > 0 else { return }
        var fix = matrix_identity_float4x4
        if n == 1 {
            fix.columns.3 = SIMD4(b[0] - a[0], 1)
        } else if n == 2 {
            // turn about up (scans are gravity-aligned), then move: the direction between the two picks must agree
            let da = SIMD2<Float>(a[1].x - a[0].x, a[1].z - a[0].z), db = SIMD2<Float>(b[1].x - b[0].x, b[1].z - b[0].z)
            // 5.7 (48): this turn takes a direction at atan2(x, z) to atan2(x, z) + angle, so orange onto blue is blue - orange
            // (it was the other way round, doubling the offset)
            let ang = atan2(db.x, db.y) - atan2(da.x, da.y)
            let R = simd_float3x3(simd_quatf(angle: ang, axis: SIMD3(0, 1, 0)))
            let ca = (a[0] + a[1]) / 2, cb = (b[0] + b[1]) / 2
            fix = m44(R); fix.columns.3 = SIMD4(cb - R * ca, 1)
        } else {
            fix = CloudOps.rigid(Array(a.prefix(n)), Array(b.prefix(n)))
        }
        cur = fix * cur
        var err: Float = 0
        for i in 0..<n { err += simd_length(CloudOps.apply(fix, a[i]) - b[i]) }
        fitText = "matched \(n) pair\(n == 1 ? "" : "s"), average error \(Prefs.length(err / Float(n)))"
        pairsBase = []; pairsNew = []
        rebuild()
    }

    // ---- nudges ----
    private var nudgePanel: some View {
        VStack(spacing: 6) {
            Picker("", selection: $show) { Text("Both").tag(0); Text("Blue only").tag(1); Text("Orange only").tag(2) }.pickerStyle(.segmented).padding(.horizontal)
                .onChange(of: show) { _, _ in rebuild() }
            HStack {
                Picker("Move", selection: $stepMM) { Text("1 mm").tag(1.0); Text("1 cm").tag(10.0); Text("5 cm").tag(50.0) }.pickerStyle(.segmented)
                Picker("Turn", selection: $stepDeg) { Text("0.2°").tag(0.2); Text("1°").tag(1.0); Text("5°").tag(5.0) }.pickerStyle(.segmented)
            }.padding(.horizontal)
            Grid(horizontalSpacing: 6, verticalSpacing: 6) {
                GridRow { nb("←", move: SIMD3(-1, 0, 0)); nb("→", move: SIMD3(1, 0, 0)); nb("Up", move: SIMD3(0, 1, 0)); nb("Down", move: SIMD3(0, -1, 0)) }
                GridRow { nb("Back", move: SIMD3(0, 0, -1)); nb("Front", move: SIMD3(0, 0, 1)); rb("Turn ↺", axis: SIMD3(0, 1, 0), 1); rb("Turn ↻", axis: SIMD3(0, 1, 0), -1) }
                GridRow { rb("Tilt fwd", axis: SIMD3(1, 0, 0), 1); rb("Tilt back", axis: SIMD3(1, 0, 0), -1); rb("Roll ←", axis: SIMD3(0, 0, 1), 1); rb("Roll →", axis: SIMD3(0, 0, 1), -1) }
            }.font(.caption).buttonStyle(.bordered).padding(.horizontal)
            .disabled(refining)                                    // 5.7 (46): nudges made while refining were lost
        }
    }
    private func nb(_ t: String, move d: SIMD3<Float>) -> some View {
        Button(t) { cur.columns.3 += SIMD4(d * Float(stepMM / 1000), 0); rebuild() }.frame(maxWidth: .infinity)
    }
    private func rb(_ t: String, axis: SIMD3<Float>, _ sign: Float) -> some View {
        Button(t) {
            let c = CloudOps.apply(cur, job.src.bounds.centre)
            let R = m44(simd_float3x3(simd_quatf(angle: sign * Float(stepDeg) * .pi / 180, axis: axis)))
            var a = matrix_identity_float4x4; a.columns.3 = SIMD4(-c, 1)
            var b = matrix_identity_float4x4; b.columns.3 = SIMD4(c, 1)
            cur = b * R * a * cur; rebuild()
        }.frame(maxWidth: .infinity)
    }

    private func refine() {
        refining = true
        let start = cur, s = job.src, b = job.base
        Task.detached {
            let r = CloudOps.sample(b.p, grid: 0.006, max: 300_000), sp = CloudOps.sample(s.p, grid: 0.006, max: 25000)
            var (T, fit) = CloudOps.icp(sp, r, GridIndex(r, cell: 0.06), start: start, maxDist: 0.06, iterations: 25)
            (T, fit) = CloudOps.icp(sp, r, GridIndex(r, cell: 0.02), start: T, maxDist: 0.015, iterations: 25)
            await MainActor.run { cur = T; fitNow = fit; fitText = "\(Int(fit * 100))% overlap after refining"; refining = false; rebuild() }
        }
    }
    private func answer(_ T: simd_float4x4?) { guard !answered else { return }; answered = true; job.done(T); dismiss() }
}
