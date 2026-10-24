import SwiftUI
import UIKit

/// 5.0 updates over home Wi-Fi: every build made on the iMac is published there (http://Home-Computer.local:8765, on the
/// home network only - nothing public). On launch the app asks the iMac for the newest build; a newer one is offered
/// with one tap, installed by SideStore on the phone. Away from home the check simply finds nothing.
enum Updates {
    static let host = "http://Home-Computer.local:8765"
    struct Info: Decodable, Equatable { let version: String; let build: Int; let notes: String?; let url: String? }   // url: this build's own file (5.5 (21)+)
    /// 5.6: the REAL version lives in DFVersion / DFBuild. The bundle's own version is pinned at 5.5 (17) in every build:
    /// SideStore holds a stale listing of 5.5 (17), never refreshes it, and refuses any install that does not match it
    static var currentBuild: Int { Int(Bundle.main.infoDictionary?["DFBuild"] as? String ?? Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "0") ?? 0 }
    static var currentVersion: String { Bundle.main.infoDictionary?["DFVersion"] as? String ?? Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "" }
    static func check() async -> Info? {
        guard let url = URL(string: host + "/version.json") else { return nil }
        var req = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 4)
        req.httpMethod = "GET"
        guard let (d, r) = try? await URLSession.shared.data(for: req), (r as? HTTPURLResponse)?.statusCode == 200,
              let i = try? JSONDecoder().decode(Info.self, from: d), i.build > currentBuild else { return nil }
        return i
    }
    /// SideStore downloads, signs and installs the new build on the phone
    static func sideStoreURL(_ info: Info?) -> URL? {
        guard let ipa = (info?.url ?? host + "/Deepfield.ipa").addingPercentEncoding(withAllowedCharacters: .alphanumerics) else { return nil }
        return URL(string: "sidestore://install?url=" + ipa)
    }
}

/// the banner on Home when a newer build is on the iMac
struct UpdateBanner: View {
    let info: Updates.Info
    @State private var noSideStore = false
    var body: some View {
        Button {
            if let u = Updates.sideStoreURL(info), UIApplication.shared.canOpenURL(u) { UIApplication.shared.open(u) }
            else if let u = Updates.sideStoreURL(info) { UIApplication.shared.open(u) { ok in if !ok { noSideStore = true } } }
        } label: {
            HStack(spacing: 12) {
                Image(systemName: "arrow.down.circle.fill").font(.title2)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Deepfield \(info.version) (\(info.build)) is ready").font(.subheadline.bold())
                    Text("First refresh SideStore (Sources, pull down), then tap Update.").font(.caption).lineLimit(2).opacity(0.85)
                }
                Spacer(minLength: 0)
                Text("Update").font(.subheadline.bold()).padding(.horizontal, 12).padding(.vertical, 6).background(.white.opacity(0.25), in: Capsule())
            }
            .foregroundStyle(.white).padding(14).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.blue.gradient, in: RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
        .alert("SideStore is not installed", isPresented: $noSideStore) {
            Button("OK", role: .cancel) {}
        } message: { Text("Install the update from your Mac with iloader instead: the file is in Downloads on your MacBook and at \(Updates.host)/Deepfield.ipa.") }
    }
}
