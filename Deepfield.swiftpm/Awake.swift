import UIKit

/// 5.6 keeps the screen on while anything long is running (a scan, a capture, a model build, a polish). Nothing did
/// before: an auto-lock in the middle of a 3-minute build suspended the app and the build died with it.
enum Awake {
    private static var holds = Set<String>()
    static func set(_ key: String, _ on: Bool) {
        let go = {
            if on { holds.insert(key) } else { holds.remove(key) }
            let want = !holds.isEmpty
            if UIApplication.shared.isIdleTimerDisabled != want { UIApplication.shared.isIdleTimerDisabled = want }
        }
        if Thread.isMainThread { go() } else { DispatchQueue.main.async(execute: go) }
    }
    static func key(_ o: AnyObject, _ tag: String) -> String { tag + "-\(ObjectIdentifier(o).hashValue)" }
}
