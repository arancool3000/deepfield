// swift-tools-version: 5.9

// KindleHub Scan: a LiDAR point-cloud scanner for iPhone Pro. Open this folder (KindleHubScan.swiftpm) in Xcode 15 or
// later and press Run with your iPhone selected - the same steps as Vision (README.md one level up).

import PackageDescription
import AppleProductTypes

let package = Package(
    name: "KindleHubScan",
    platforms: [
        .iOS("17.0")
    ],
    products: [
        .iOSApplication(
            name: "Deepfield",
            targets: ["AppModule"],
            bundleIdentifier: "com.arankeyhan.kindlehubscan",
            teamIdentifier: "",
            displayVersion: "5.8",
            bundleVersion: "53",
            appIcon: .asset("AppIcon"),
            accentColor: .asset("AccentColor"),
            supportedDeviceFamilies: [
                .phone
            ],
            supportedInterfaceOrientations: [
                .portrait,
                .landscapeRight,
                .landscapeLeft
            ],
            capabilities: [
                .camera(purposeString: "Deepfield uses the camera and the LiDAR scanner to make a 3D point cloud of what you point it at."),
                .microphone(purposeString: "The voice guide listens so you can talk to it while you scan."),
                .localNetwork(purposeString: "Deepfield checks your iMac on your home Wi-Fi for new versions of the app.")
            ],
            appCategory: .utilities
        )
    ],
    targets: [
        .executableTarget(
            name: "AppModule",
            path: "."
        )
    ]
)
