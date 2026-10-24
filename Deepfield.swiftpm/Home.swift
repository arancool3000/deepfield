import SwiftUI
import simd

struct HomeView: View {
    @ObservedObject private var lib = Library.shared
    @State private var showScan = false
    @State private var scanKind: ScanKind?
    @State private var pickKind = false
    @State private var finishing: ScanItem?
    @State private var finishKind: ScanKind = .object
    @State private var pro: ProPick?          // 5.5: ONE value says which (a flag + a second @State opened Pro room as Pro object)
    @State private var showPlan = false
    @State private var showPhoto3D = false
    @State private var showWalk = false
    @State private var walking: ScanItem?
    @State private var splatItem: ScanItem?
    @State private var continueItem: ScanItem?
    @State private var splatDetail = 1
    @State private var aranItem: ScanItem?
    @State private var importing = false
    @State private var importNote = ""
    @State private var showPhotos = false
    @State private var showPhotoCap = false
    @State private var update: Updates.Info?
    @State private var showMerge = false
    @State private var showSettings = false
    @AppStorage("seenWhatsNew") private var seenWhatsNew = ""
    @State private var editing: ScanItem?
    @State private var preview: URL?
    @State private var renaming: ScanItem?
    @State private var newName = ""
    @State private var share: URL?
    @State private var search = ""
    @State private var purgeTarget: ScanItem?          // 5.7 (47): Delete now asks first
    @AppStorage("librarySort") private var sort = 0
    private var shown: [ScanItem] {
        var v = lib.items
        if !search.isEmpty { v = v.filter { $0.name.localizedCaseInsensitiveContains(search) || ($0.notes ?? "").localizedCaseInsensitiveContains(search) } }
        switch sort { case 1: v.sort { $0.name.localizedCompare($1.name) == .orderedAscending }; case 2: v.sort { $0.count > $1.count }; default: v.sort { $0.date > $1.date } }
        return v
    }
    private func section(_ t: String) -> some View {
        Text(t)
            .font(.title3.weight(.semibold))
            .tracking(-0.3)
            .foregroundStyle(.white)
            .accessibilityAddTraits(.isHeader)
    }

    private func quick(
        _ title: String,
        _ sub: String,
        _ icon: String,
        _ tint: Color,
        wide: Bool = false,
        _ go: @escaping () -> Void
    ) -> some View {
        Button(action: go) {
            VStack(alignment: .leading, spacing: 16) {
                Image(systemName: icon)
                    .font(.system(size: 21, weight: .regular))
                    .foregroundStyle(tint)
                    .frame(height: 26)

                VStack(alignment: .leading, spacing: 5) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                        .fixedSize(horizontal: false, vertical: true)

                    Text(sub)
                        .font(.caption)
                        .foregroundStyle(Color.white.opacity(0.48))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, minHeight: 84, alignment: .topLeading)
            .padding(12)
            .background(
                Color(white: 0.085),
                in: RoundedRectangle(cornerRadius: 18, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(.white.opacity(0.055), lineWidth: 1)
            }
            .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
    }

    private func featured(
        _ title: String,
        _ subtitle: String,
        _ icon: String,
        _ go: @escaping () -> Void
    ) -> some View {
        Button(action: go) {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    Image(systemName: icon)
                        .font(.system(size: 22, weight: .medium))
                        .foregroundStyle(Color(red: 0.69, green: 0.77, blue: 0.91))

                    Spacer(minLength: 4)

                    Image(systemName: "arrow.up.right")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.3))
                }

                VStack(alignment: .leading, spacing: 7) {
                    Text(title)
                        .font(.system(.subheadline, design: .rounded, weight: .bold))
                        .tracking(0.3)
                        .foregroundStyle(.white)

                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.55))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, minHeight: 108, alignment: .topLeading)
            .padding(18)
            .background(
                Color(white: 0.105),
                in: RoundedRectangle(cornerRadius: 20, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(.white.opacity(0.08), lineWidth: 1)
            }
            .contentShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
    }

    @State private var showCrashReport = CrashWatch.crashedLastTime

    private let cols3 = Array(repeating: GridItem(.flexible(), spacing: 10, alignment: .top), count: 3)
    private let cols = [
        GridItem(.flexible(), spacing: 12, alignment: .top),
        GridItem(.flexible(), spacing: 12, alignment: .top)
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 32) {
                    if let u = update {
                        UpdateBanner(info: u)
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Button {
                            scanKind = .omni; showScan = true
                        } label: {
                            VStack(alignment: .leading, spacing: 26) {
                                HStack(alignment: .center) {
                                    Label("BEST QUALITY", systemImage: "circle.hexagongrid.fill")
                                        .font(.system(size: 10, weight: .semibold))
                                        .tracking(1.5)
                                        .foregroundStyle(
                                            Color(red: 0.72, green: 0.82, blue: 1)
                                        )

                                    Spacer()

                                    Image(systemName: "viewfinder")
                                        .font(.system(size: 27, weight: .ultraLight))
                                        .foregroundStyle(.white.opacity(0.45))
                                        .accessibilityHidden(true)
                                }

                                VStack(alignment: .leading, spacing: 10) {
                                    Text("OMNI")
                                        .font(.system(size: 44, weight: .semibold))
                                        .tracking(-1.8)
                                        .foregroundStyle(.white)

                                    Text("Capture every dimension.")
                                        .font(.title3.weight(.medium))
                                        .tracking(-0.4)
                                        .foregroundStyle(.white.opacity(0.9))

                                    Text("LiDAR, 4K photos and depth.\nFused into one detailed 3D model.")
                                        .font(.subheadline)
                                        .foregroundStyle(.white.opacity(0.55))
                                        .lineSpacing(3)
                                        .fixedSize(horizontal: false, vertical: true)
                                }

                                HStack {
                                    Text("Start OMNI scan")
                                        .font(.subheadline.weight(.semibold))

                                    Spacer()

                                    Image(systemName: "arrow.right")
                                        .font(.system(size: 15, weight: .semibold))
                                }
                                .foregroundStyle(Color(white: 0.08))
                                .padding(.horizontal, 18)
                                .padding(.vertical, 16)
                                .background(
                                    Color(red: 0.79, green: 0.86, blue: 1),
                                    in: RoundedRectangle(cornerRadius: 14, style: .continuous)
                                )
                            }
                            .padding(22)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background {
                                RoundedRectangle(cornerRadius: 26, style: .continuous)
                                    .fill(
                                        LinearGradient(
                                            colors: [
                                                Color(red: 0.14, green: 0.18, blue: 0.25),
                                                Color(red: 0.075, green: 0.09, blue: 0.12)
                                            ],
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        )
                                    )
                            }
                            .overlay {
                                RoundedRectangle(cornerRadius: 26, style: .continuous)
                                    .strokeBorder(
                                        LinearGradient(
                                            colors: [.white.opacity(0.18), .white.opacity(0.04)],
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        ),
                                        lineWidth: 1
                                    )
                            }
                            .shadow(color: .black.opacity(0.22), radius: 16, y: 8)
                            .contentShape(
                                RoundedRectangle(cornerRadius: 26, style: .continuous)
                            )
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Start OMNI scan")
                        .accessibilityHint("Best quality. Fuses LiDAR, 4K photos and depth into a detailed 3D model.")

                        LazyVGrid(columns: cols, spacing: 12) {
                            featured("ULTRA", "Automatic capture.\nGuided from start to finish.", "bolt.fill") {
                                scanKind = .ultra; showScan = true
                            }

                            featured("SUPERPRO", "Tailored capture.\nChoose your subject.", "sparkles") {
                                pickKind = true
                            }
                            .confirmationDialog(
                                "What are you scanning?",
                                isPresented: $pickKind,
                                titleVisibility: .visible
                            ) {
                                ForEach(ScanKind.picks) { k in
                                    Button(k.title) { scanKind = k; showScan = true }
                                }
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 14) {
                        section("Scan modes")

                        LazyVGrid(columns: cols3, spacing: 10) {
                            quick("Standard", "LiDAR + photos", "viewfinder", .blue.opacity(0.85)) {
                                scanKind = nil; showScan = true
                            }
                            quick("PowerBlast", "Big rooms, fast", "bolt.horizontal.fill", .red.opacity(0.85)) {
                                scanKind = .blast; showScan = true
                            }
                            quick("Lite", "Fast, tiny files", "leaf", .mint.opacity(0.85)) {
                                scanKind = .lite; showScan = true
                            }
                            quick("LiDAR", "Pure geometry", "dot.radiowaves.left.and.right", .cyan.opacity(0.85)) {
                                scanKind = .lidar; showScan = true
                            }
                            quick("Point cloud", "Dense colour", "circle.grid.3x3.fill", .teal.opacity(0.85)) {
                                scanKind = .cloud; showScan = true
                            }
                            quick("Floor plan", "Walls, doors, sizes", "square.split.bottomrightquarter", .indigo.opacity(0.85)) {
                                showPlan = true
                            }
                            quick("3D photo", "One shot, real depth", "camera.aperture", .pink.opacity(0.85)) {
                                showPhoto3D = true
                            }
                            quick("Gaussian", "Scan, then splats", "sparkles", .purple.opacity(0.85)) {
                                scanKind = .gaussian; showScan = true
                            }
                            quick("Aran", "Sharpest splats", "wand.and.stars", .orange.opacity(0.85)) {
                                scanKind = .aran; showScan = true
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 14) {
                        section("Create & tools")

                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(alignment: .top, spacing: 12) {
                                quick("Photo capture", "Shoot a 3D model", "camera.on.rectangle", .white.opacity(0.65)) {
                                    showPhotoCap = true
                                }
                                .frame(width: 164)

                                quick("Pro object", "Textured model", "cube.transparent", .white.opacity(0.65)) {
                                    pro = ProPick(id: false)
                                }
                                .frame(width: 164)

                                quick("Pro room", "Apple room capture", "house.and.flag", .white.opacity(0.65)) {
                                    pro = ProPick(id: true)
                                }
                                .frame(width: 164)

                                quick("Walk", "Street View of your place", "figure.walk.circle", .white.opacity(0.65)) {
                                    showWalk = true
                                }
                                .frame(width: 164)

                                quick("Photos to 3D", "From your photos", "photo.stack", .white.opacity(0.65)) {
                                    showPhotos = true
                                }
                                .frame(width: 164)

                                quick("Merge", "Combine scans", "square.stack.3d.up", .white.opacity(0.65)) {
                                    showMerge = true
                                }
                                .frame(width: 164)
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 18) {
                        Rectangle()
                            .fill(.white.opacity(0.08))
                            .frame(height: 1)
                            .padding(.bottom, 6)

                        HStack(spacing: 10) {
                            Text("Library")
                                .font(.title2.weight(.semibold))
                                .tracking(-0.5)
                                .foregroundStyle(.white)
                                .accessibilityAddTraits(.isHeader)

                            Text("\(shown.count)")
                                .font(.caption.weight(.medium).monospacedDigit())
                                .foregroundStyle(.white.opacity(0.55))
                                .padding(.horizontal, 9)
                                .padding(.vertical, 5)
                                .background(.white.opacity(0.07), in: Capsule())

                            Spacer()

                            Menu {
                                Picker("Sort", selection: $sort) {
                                    Text("Newest").tag(0)
                                    Text("Name").tag(1)
                                    Text("Most points").tag(2)
                                }
                            } label: {
                                Label("Sort", systemImage: "arrow.up.arrow.down")
                                    .font(.caption.weight(.medium))
                                    .foregroundStyle(.white.opacity(0.75))
                                    .padding(.horizontal, 13)
                                    .frame(minHeight: 44)
                                    .background(
                                        .white.opacity(0.05),
                                        in: RoundedRectangle(cornerRadius: 12)
                                    )
                            }
                        }

                        if lib.items.isEmpty {
                            VStack(spacing: 12) {
                                Image(systemName: "square.stack.3d.up")
                                    .font(.system(size: 30, weight: .light))
                                    .foregroundStyle(.white.opacity(0.35))
                                    .padding(.bottom, 4)

                                Text("A new dimension starts here.")
                                    .font(.subheadline.weight(.medium))
                                    .foregroundStyle(.white.opacity(0.8))

                                Text("Your scans will appear here.")
                                    .font(.caption)
                                    .foregroundStyle(.white.opacity(0.45))
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.horizontal, 20)
                            .padding(.vertical, 44)
                            .background(
                                Color(white: 0.065),
                                in: RoundedRectangle(cornerRadius: 22, style: .continuous)
                            )
                        } else {
                            if shown.isEmpty {
                                Text("No scans match \"\(search)\".")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 32)
                            }

                            LazyVGrid(columns: cols, spacing: 14) {
                                ForEach(shown) { it in card(it) }
                            }
                        }

                        // 5.7 (47): a delete can be taken back - deleted scans wait here, collapsed, for 30 days
                        if !lib.trash.isEmpty { recentlyDeleted }
                    }
                }
                .padding(16)
                .preferredColorScheme(.dark)
            }
            .navigationTitle("Deepfield")
            .searchable(text: $search, prompt: "Search scans")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) { Button { importing = true } label: { Image(systemName: "square.and.arrow.down") } }
                ToolbarItem(placement: .topBarTrailing) { Button { showSettings = true } label: { Image(systemName: "gearshape") } }
            }
            .fileImporter(isPresented: $importing, allowedContentTypes: [.data, .plainText, .commaSeparatedText], allowsMultipleSelection: true) { r in
                guard case .success(let urls) = r else { return }
                importNote = "Importing \(urls.count) file\(urls.count == 1 ? "" : "s")..."
                for u in urls { lib.importFile(u) { it in importNote = it == nil ? "Could not read \(u.lastPathComponent) (PLY or XYZ files)" : "Imported \(it!.name)" } }
            }
            .overlay(alignment: .bottom) {
                if !importNote.isEmpty { Text(importNote).font(.footnote).padding(10).background(.ultraThinMaterial, in: Capsule()).padding(.bottom, 20).onTapGesture { importNote = "" } }
            }
            .fullScreenCover(isPresented: $showScan) {
                ContentView(kind: scanKind) { item in
                    showScan = false
                    guard let i = item else { return }
                    // 5.6 (38): the review's choice decides the finishing. Fast opens the scan as it is; Default does what
                    // the mode always did; Dense and iMac always run the finishing (a plain scan finishes as a room: the
                    // gentlest clean-up)
                    let k = SWITCHED_KIND ?? scanKind, choice = PROCESS_CHOICE      // 5.7 (48): a mode switched by voice finishes as that mode
                    let finish: ScanKind? = choice == .fast ? nil : (k.map { $0 == .aran || $0.splats || ($0.quick && choice == .standard) ? nil : $0 } ?? (choice == .standard ? nil : .room))
                    if let f = finish { finishKind = f; Library.shared.finishing.insert(i.id) }      // before the polish starts
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                        if choice != .fast, let k, k == .aran { aranItem = i } else if choice != .fast, let k, k.splats { splatItem = i; splatDetail = 1 }
                        else if finish != nil { finishing = i } else { editing = i }
                    }
                }
            }
            .fullScreenCover(item: $pro) { p in ProScanView(area: p.id) }
            .fullScreenCover(isPresented: $showPhotoCap) { PhotoCaptureView() }
            .fullScreenCover(isPresented: $showPlan) { FloorPlanScanView() }
            .fullScreenCover(isPresented: $showPhoto3D) { Photo3DScreen() }
            .fullScreenCover(isPresented: $showWalk) { WalkCaptureScreen() }
            .fullScreenCover(item: $walking) { WalkViewer(item: $0) }
            .fullScreenCover(item: $continueItem) { it in ContentView(kind: nil, continuing: it) { done in continueItem = nil; if let d = done { DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { editing = d } } } }
            .fullScreenCover(item: $splatItem) { SplatSheet(item: $0, startDetail: splatDetail, autoStart: true) }
            .fullScreenCover(item: $aranItem) { AranView(item: $0) }
            .fullScreenCover(item: $editing) { it in      // 5.7 (48): a background polish waits while the scan is open here
                EditorView(item: it).onAppear { Library.shared.editing.insert(it.id) }.onDisappear { Polish.editorClosed(it.id) }
            }
            .sheet(isPresented: $showPhotos) { PhotosTo3DView().interactiveDismissDisabled() }   // 5.7 (48): its Close button already waits for the build; a swipe-down left it running unseen, and a second could start
            .sheet(isPresented: $showMerge) { MergeView() }
            .sheet(isPresented: $showSettings) { SettingsView() }
            .task { update = await Updates.check() }
            .sheet(isPresented: Binding(get: { seenWhatsNew != WhatsNew5.version }, set: { if !$0 { seenWhatsNew = WhatsNew5.version } })) { WhatsNew5 { seenWhatsNew = WhatsNew5.version } }
            .sheet(isPresented: $showCrashReport) { BugReportView(crashed: true) }
            .sheet(item: $finishing) { SuperProFinish(item: $0, kind: finishKind, resolved: finishKind.auto ? ULTRA_RESOLVED : nil) }
            .fullScreenCover(item: $preview) { ModelPreview(url: $0) }
            .sheet(item: $share) { ShareSheet(items: [$0]) }
            .alert("Rename", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
                TextField("Name", text: $newName)
                Button("Save") { if let r = renaming { lib.rename(r, to: newName) } }
                Button("Cancel", role: .cancel) {}
            }
        }
    }

    private func action(_ title: String, _ sub: String, _ icon: String, _ tint: Color, _ go: @escaping () -> Void) -> some View {
        Button(action: go) {
            VStack(alignment: .leading, spacing: 10) {
                Image(systemName: icon).font(.title2).foregroundStyle(tint)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline).foregroundStyle(.primary)
                    Text(sub).font(.caption).foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, minHeight: 96, alignment: .leading)
            .padding(14)
            .background(Color(white: 0.12), in: RoundedRectangle(cornerRadius: 18))
        }
        .buttonStyle(.plain)
    }

    private func card(_ it: ScanItem) -> some View {
        Button { open(it) } label: {
            VStack(alignment: .leading, spacing: 6) {
                ZStack {
                    Color(white: 0.1)
                    if let img = Thumbs.image(it.id) { Image(uiImage: img).resizable().scaledToFill() }
                    else { Image(systemName: it.isWalk ? "figure.walk.circle" : (it.isModel ? "cube.fill" : "circle.grid.3x3.fill")).font(.largeTitle).foregroundStyle(.secondary) }
                }
                .frame(height: 140).clipShape(RoundedRectangle(cornerRadius: 14))
                Text(it.name).font(.subheadline.bold()).lineLimit(1).foregroundStyle(.primary)
                if lib.polishing.contains(it.id) { HStack(spacing: 4) { ProgressView().controlSize(.mini); Text("Polishing...").font(.caption) }.foregroundStyle(.secondary) }
                else { Text(it.isWalk ? "Walk, \(it.count) spot\(it.count == 1 ? "" : "s")" : (it.isModel ? "3D model" : "\(it.count.formatted()) points")).font(.caption).foregroundStyle(.secondary) }
            }
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button { open(it) } label: { Label("Open", systemImage: "eye") }
            if !it.isModel && !it.isWalk { Button { continueItem = it } label: { Label("Continue scanning", systemImage: "plus.viewfinder") } }
            Button { newName = it.name; renaming = it } label: { Label("Rename", systemImage: "pencil") }
            Button { exportItem(it) } label: { Label("Share", systemImage: "square.and.arrow.up") }
            Button(role: .destructive) { lib.delete(it) } label: { Label("Delete", systemImage: "trash") }
        }
    }

    private var recentlyDeleted: some View {
        DisclosureGroup("Recently deleted (\(lib.trash.count))") {
            VStack(alignment: .leading, spacing: 10) {
                Text("Kept for 30 days, then removed.").font(.caption).foregroundStyle(.secondary)
                ForEach(lib.trash) { it in
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(it.name).font(.subheadline).lineLimit(1).foregroundStyle(.white)
                            if let d = it.deleted { Text("deleted \(d.formatted(.relative(presentation: .named)))").font(.caption).foregroundStyle(.secondary) }
                        }
                        Spacer()
                        Button("Restore") { lib.restore(it) }.font(.caption.weight(.medium)).buttonStyle(.bordered)
                        Button("Delete now", role: .destructive) { purgeTarget = it }.font(.caption.weight(.medium)).buttonStyle(.bordered)
                    }
                }
            }
            .padding(.top, 8)
        }
        .font(.subheadline.weight(.medium))
        .foregroundStyle(.white)
        .tint(.white)
        .padding(14)
        .background(Color(white: 0.065), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .confirmationDialog("Delete for good?", isPresented: Binding(get: { purgeTarget != nil }, set: { if !$0 { purgeTarget = nil } }), titleVisibility: .visible, presenting: purgeTarget) { t in
            Button("Delete \"\(t.name)\"", role: .destructive) { lib.purge(t); purgeTarget = nil }
            Button("Cancel", role: .cancel) { purgeTarget = nil }
        } message: { _ in
            Text("Its points and photos are removed from this iPhone. This can't be undone.")
        }
    }

    private func open(_ it: ScanItem) {
        if it.isWalk { walking = it } else if it.isModel { preview = Library.modelURL(it.id) } else { editing = it }
    }
    private func exportItem(_ it: ScanItem) {
        if it.isModel { share = Library.modelURL(it.id); return }
        if it.isWalk { share = Library.walkURL(it.id).appendingPathComponent("spot000.jpg"); return }
        DispatchQueue.global(qos: .userInitiated).async {
            guard let c = Library.load(it) else { return }
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(it.name.replacingOccurrences(of: "/", with: "-")).ply")
            try? c.ply().write(to: url)
            DispatchQueue.main.async { share = url }
        }
    }
}

// ---------- merging saved scans ----------
struct MergeView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var lib = Library.shared
    @State private var picked: [UUID] = []
    @State private var running = false
    @State private var progress: Float = 0
    @State private var log: [String] = []
    @State private var done: ScanItem?
    @State private var review: ReviewJob?
    @AppStorage("mergeCheckEach") private var checkEach = false

    var clouds: [ScanItem] { lib.items.filter { !$0.isModel && !$0.isWalk } }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(clouds) { it in
                        Button { toggle(it.id) } label: {
                            HStack {
                                Image(systemName: picked.contains(it.id) ? "checkmark.circle.fill" : "circle").foregroundStyle(picked.contains(it.id) ? Color.accentColor : .secondary)
                                VStack(alignment: .leading) { Text(it.name); Text("\(it.count.formatted()) points").font(.caption).foregroundStyle(.secondary) }
                                Spacer()
                                if let i = picked.firstIndex(of: it.id) { Text(i == 0 ? "Base" : "#\(i + 1)").font(.caption.bold()).foregroundStyle(.secondary) }
                            }
                        }.foregroundStyle(.primary)
                    }
                } header: { Text("Pick 2 or more") } footer: {
                    Text("The first one you pick is the base. Scans must overlap and nothing should have moved between them. Overlapping parts are kept once, so there are no doubles.")
                }
                Section { Toggle("Let me check every line-up", isOn: $checkEach) } footer: { Text("Off: clear matches merge by themselves and you are only asked when the app is unsure.") }
                if running || !log.isEmpty {
                    Section("Progress") {
                        if running { ProgressView(value: progress) }
                        ForEach(Array(log.enumerated()), id: \.offset) { Text($0.element).font(.footnote) }
                    }
                }
            }
            .navigationTitle("Merge scans").navigationBarTitleDisplayMode(.inline)
            .interactiveDismissDisabled(running)       // 5.7 (46): a swipe-down left the merge running unseen, and a second could start
            .fullScreenCover(item: $review) { AlignReview(job: $0) }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() }.disabled(running) }
                ToolbarItem(placement: .confirmationAction) {
                    if done != nil { Button("Done") { dismiss() }.bold() }
                    else { Button("Merge") { merge() }.bold().disabled(picked.count < 2 || running) }
                }
            }
        }
    }

    private func toggle(_ id: UUID) {
        guard !running else { return }
        if let i = picked.firstIndex(of: id) { picked.remove(at: i) } else { picked.append(id) }
    }

    private func merge() {
        let items = picked.compactMap { id in lib.items.first { $0.id == id } }
        running = true; log = []; progress = 0
        let dup = Float(max(0.003, Prefs.mergeGridMM / 1000 * 2.5))
        Task {
            guard var base = await Task.detached(operation: { Library.load(items[0]) }).value else { log = ["Could not open \(items[0].name)"]; running = false; return }
            let n = items.count - 1
            var parts: [(URL, simd_float4x4)] = Library.captureURL(items[0]).map { [($0, matrix_identity_float4x4)] } ?? []
            var used = [items[0].name]          // 5.7 (46): only the scans that really merged count, and name the result
            for (k, it) in items.dropFirst().enumerated() {
                guard let src = await Task.detached(operation: { Library.load(it) }).value else { log.append("Could not open \(it.name)"); continue }
                log.append("Lining up \(it.name)...")
                let b0 = base
                let (T, fit) = await Task.detached { CloudOps.align2(src, to: b0) { f in Task { @MainActor in progress = (Float(k) + f * 0.9) / Float(n) } } }.value
                // 5.0: a clear match merges straight away; only an unsure one asks you to look
                let chosen: simd_float4x4?
                if fit >= 0.55 && !checkEach {
                    chosen = T; log.append("\(it.name): lined up automatically (\(Int(fit * 100))% overlap)")
                } else {
                    chosen = await withCheckedContinuation { cont in
                        review = ReviewJob(name: it.name, base: b0, src: src, T: T, fit: fit) { cont.resume(returning: $0) }
                    }
                    review = nil
                }
                guard let TT = chosen else { log.append("\(it.name): skipped"); continue }
                let grid = Float(max(0.002, Prefs.mergeGridMM / 1000))
                let (merged, added) = await Task.detached { () -> (Cloud, Int) in
                    var s2 = src; for i in 0..<s2.count { s2.p[i] = CloudOps.apply(TT, s2.p[i]) }
                    let r = CloudOps.addNew(s2.sane(), to: b0, dup: dup)
                    return (CloudOps.dedupe(r.0, grid: grid), r.1)         // copies in the same tiny cell become one
                }.value
                base = merged; used.append(it.name)
                if let c = Library.captureURL(it) { parts.append((c, TT)) }
                Log.add("merge \(it.name): +\(added) points, total \(merged.count)")
                log.append("\(it.name): merged, \(added.formatted()) new points")
            }
            guard used.count >= 2 else { log.append("Nothing merged, so nothing was saved"); running = false; return }
            let name = "Merged " + used.prefix(3).joined(separator: " + ")
            // every scan's photos, moved by its line-up (the first scan's alone before 5.6)
            // 5.7 (46): a base with no photos still takes the one other scan's, moved by its line-up
            let pp = parts
            let cap: String? = pp.count >= 2 || (items[0].capture == nil && !pp.isEmpty) ? await Task.detached { CaptureMerge.combine(pp) }.value ?? items[0].capture : items[0].capture
            lib.add(base, name: name, capture: cap) { it in done = it; progress = 1; running = false; log.append("Saved \"\(it.name)\" (\(it.count.formatted()) points)") }
        }
    }
}

/// shown once after updating to 5.0 - only what this build really does
struct WhatsNew5: View {
    static let version = "5.8"                // shown once per version; bump with every release
    let done: () -> Void
    private let items: [(String, String, String)] = [
        ("brain", "Pick your AI", "The AI tools now work with Gemini, ChatGPT, Claude, Grok, DeepSeek or OpenRouter - your own key, kept in the Keychain. Settings > AI. The voice guide and AI painter still use Gemini."),
        ("circle.grid.3x3.fill", "Quick fill that fills", "Quick fill makes room first, so it fills fully instead of sometimes adding nothing."),
        ("photo.on.rectangle", "Pro photos kept", "Pro object and Pro room photos stay as long as their model, so it can be rebuilt later."),
        ("bolt.shield", "48 crash and data-loss fixes", "A deep search found and fixed 12 crashes, 10 ways to lose data and 26 other bugs - including Quick fill on big scans, which now fills only what memory allows instead of crashing."),
        ("cube.fill", "Photoreal without holes", "Where the photo surface has a gap, your scan now shows through instead of black."),
        ("figure.walk.circle", "Walk panoramas fixed", "Every photo was being placed upside down in its own spot - the cause of flipped pictures and furniture in the wrong place. Fixed, and the LiDAR now puts near things where they really are."),
        ("trash", "Recently deleted", "Deleting a scan is no longer for ever: it waits 30 days in Recently deleted on the home screen, where you can restore it."),
        ("arrow.uturn.backward.circle", "Undo history", "Many more undo steps, named - hold the undo button to jump back to before any edit."),
        ("paintpalette", "True photo colours", "Splats, AI fill and 3D photos were reading every photo upside down for colour. Fixed."),
        ("sparkles", "Splats trained on your photos", "Gaussian splats are now trained on the phone against your scan's photos, like Scaniverse: sharp, no confetti. Train more if you want it finer."),
        ("cube.transparent", "Pro object, calmer", "No more forced extra laps: a short ring never blocks, photos come quicker, and no more doubled objects. Finish from 40 photos."),
        ("camera.aperture", "3D photos without holes", "Built from the photo you took, whole and sharp, with true depth. The guided frames colour what is behind edges. No more black holes."),
        ("checklist", "Review before processing", "Done now shows your scan and a small camera where every photo was taken. Extend to carry on scanning, or Process: Fast, Default, Dense or On the iMac, each with a time estimate that learns from your phone."),
        ("square.grid.3x3", "White mesh while scanning", "Like Polycam: scanned surfaces get a faint white mesh, and blue still shows what is left. Settings can switch it off."),
        ("circle.hexagongrid.circle.fill", "OMNI photogrammetry keeps going", "Closing the Finishing screen no longer drops the photogrammetry model halfway, and it gets the AI name too."),
        ("figure.walk.circle", "Walk", "Street View and a 3D dollhouse of your own place: turn round on each spot, walk a few steps, again. Then walk through it - moves fly through the 3D model - see it as a cut-away dollhouse or a floor plan, measure anything, or move the phone to look round."),
        ("circle.grid.3x3.fill", "Quick fill, every point", "Quick fill now fills around every point of the scan - edges, curves and fine detail too - about 4x the points."),
        ("desktopcomputer", "iMac processing", "On home Wi-Fi your iMac builds Pro object, Pro room, Photos to 3D and OMNI models at full detail - about 4x the phone's, with an 8K texture. Away from home they build on the phone. Settings > iMac processing."),
        ("camera.aperture", "3D photo, guided", "Tap, then follow the cross: left, right, up, down. The 3D photo is only what the camera really saw in colour - nothing invented."),
        ("arkit", "Real AR", "Models are placed in your room through the camera: tap a floor or table, drag, pinch and twist. Real things in front hide them."),
        ("house.and.flag", "Pro room is a room capture", "It opened Pro object before. Now it is Apple's room capture: no box, walk round and build."),
        ("wand.and.rays", "AI fill that fits", "Only real holes are filled - ringed by real wall, floor or ceiling - in the surface's own colour and grain, never a patchwork."),
        ("arrow.down.circle", "Updates that install", "Every build is its own file, so SideStore never refuses an update again."),
        ("camera.filters", "Photoreal", "Your scans shown with your real photos projected onto the 3D shape as you move - from where you stood it looks like the photo. The default look in the editor, models, Aran and 3D photos."),
        ("circle.hexagongrid.circle.fill", "OMNI scan", "Above ULTRA: dense 4K photos and depth frames, then every method in turn - drift, doubles, Quick fill and the finest photoreal model."),
        ("cube.transparent", "Pro object, guided", "Three guided rings - level, high, low - then flip it for the bottom. The floor is cut away automatically."),
        ("cube", "Models you can work with", "Place in AR, turn, sit on the floor, set the real height, cut away the bottom, save, share as USDZ, OBJ, STL or PLY."),
        ("wand.and.stars", "Aran rebuilt", "The finest photo model Deepfield makes, viewed photoreal - no more acid rain."),
        ("circle.grid.3x3.fill", "Quick fill", "About 3x the points on your surfaces, instantly and offline."),
        ("camera.on.rectangle", "Photo capture", "Manual, automatic or timed photos at the quality you choose, then a 3D model on the phone."),
        ("memorychip", "Big scans that save", "Memory full only when it really is, and a 12M-point scan no longer crashes while saving."),
        ("square.on.square", "Doubles, measured", "A reference model is built from your photos and drift-corrected; whatever your scan has that it does not, in space the camera saw through, goes."),
        ("arrow.triangle.merge", "Drift fix that helps", "Point-to-plane matching between photos and revisits. It only changes your scan if photos of the same place agree better afterwards."),
        ("number.circle", "Gemini edits by parts", "Your scan is split into numbered, measured parts. Gemini names exact parts - remove, recolour, flatten, fill, protect - and you switch steps on or off."),
        ("scissors", "Cut you can see", "A live box on the scan: what will go turns dark red while you drag. Turn the box to match the walls."),
        ("cube.fill", "Solid view is solid", "A real surface built from your points, lit like a model."),
        ("paintpalette", "True colours", "Scans no longer look washed out. Colour sliders preview live."),
        ("eye", "See your scan", "Hide every tool with the eye button. Landscape puts the tools at the side."),
        ("rectangle.on.rectangle.angled", "Merging that lines up by itself", "Clear matches merge on their own; when unsure: Looks right, or Try another way."),
        ("speaker.slash", "Quieter, smoother scanning", "No shutter sound, less lag, a voice guide that stays off when you turn it off, and a progress bar for point generation."),
    ]
    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(items, id: \.1) { i in
                        HStack(alignment: .top, spacing: 14) {
                            Image(systemName: i.0).font(.title2).foregroundStyle(.tint).frame(width: 34)
                            VStack(alignment: .leading, spacing: 3) { Text(i.1).font(.headline); Text(i.2).font(.subheadline).foregroundStyle(.secondary) }
                        }.padding(.vertical, 4)
                    }
                } header: { Text("Deepfield \(WhatsNew5.version)").font(.largeTitle.bold()).foregroundStyle(.primary).textCase(nil).padding(.bottom, 6) }
            }
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Continue") { done() }.bold() } }
        }
    }
}

/// which Pro capture to open: id true = room (area mode), false = object
struct ProPick: Identifiable { let id: Bool }
