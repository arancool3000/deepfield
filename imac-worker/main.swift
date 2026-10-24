// Deepfield iMac worker: Apple photogrammetry on the M1 at FULL detail (the phone can only make "reduced").
// usage: dfphoto <images folder> <out.usdz> <detail: medium|full|raw> <mask 0|1>
// A folder from a LiDAR scan (photo + .depth + .conf + .json per frame) is built from samples WITH the LiDAR depth, so
// the model has its real size; a plain folder of photos is read by Apple directly.
import Foundation
import RealityKit
import CoreImage
import CoreVideo

let a = CommandLine.arguments
guard a.count >= 5 else { print("E usage"); exit(2) }
let input = URL(fileURLWithPath: a[1]), out = URL(fileURLWithPath: a[2])
let detail: PhotogrammetrySession.Request.Detail = a[3] == "raw" ? .raw : (a[3] == "medium" ? .medium : .full)
var cfg = PhotogrammetrySession.Configuration()
cfg.featureSensitivity = .high
cfg.sampleOrdering = .unordered
cfg.isObjectMaskingEnabled = a[4] == "1"
setvbuf(stdout, nil, _IOLBF, 0)
guard PhotogrammetrySession.isSupported else { print("E this Mac cannot run photogrammetry"); exit(3) }

let ci = CIContext()
let fm = FileManager.default
let names = ((try? fm.contentsOfDirectory(atPath: input.path)) ?? []).sorted()
let jpgs = names.filter { $0.lowercased().hasSuffix(".jpg") && names.contains(String($0.dropLast(4)) + ".depth") && names.contains(String($0.dropLast(4)) + ".json") }

func sample(_ id: Int, _ jpg: String) -> PhotogrammetrySample? {
    let base = input.appendingPathComponent(String(jpg.dropLast(4)))
    guard let img = CIImage(contentsOf: input.appendingPathComponent(jpg)),
          let j = (try? JSONSerialization.jsonObject(with: Data(contentsOf: base.appendingPathExtension("json")))) as? [String: Any],
          let dw = j["depthWidth"] as? Int, let dh = j["depthHeight"] as? Int,
          let d = try? Data(contentsOf: base.appendingPathExtension("depth")), d.count >= dw * dh * 4 else { return nil }
    let w = Int(img.extent.width), h = Int(img.extent.height)
    let attrs = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary
    var pb: CVPixelBuffer?
    guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess, let ib = pb else { return nil }
    ci.render(img, to: ib)
    var s = PhotogrammetrySample(id: id, image: ib)
    let conf = try? Data(contentsOf: base.appendingPathExtension("conf"))
    var dpb: CVPixelBuffer?
    if CVPixelBufferCreate(nil, dw, dh, kCVPixelFormatType_DepthFloat32, attrs, &dpb) == kCVReturnSuccess, let dm = dpb {
        CVPixelBufferLockBaseAddress(dm, [])
        if let p = CVPixelBufferGetBaseAddress(dm) {
            let row = CVPixelBufferGetBytesPerRow(dm)
            d.withUnsafeBytes { src in
                for y in 0..<dh {
                    let dst = p.advanced(by: y * row).assumingMemoryBound(to: Float32.self)
                    for x in 0..<dw {
                        var v = src.loadUnaligned(fromByteOffset: (y * dw + x) * 4, as: Float32.self)
                        if let c = conf, c.count == dw * dh, c[y * dw + x] == 0 { v = 0 }
                        dst[x] = v
                    }
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(dm, [])
        s.depthDataMap = dm
    }
    return s
}

let session: PhotogrammetrySession
do {
    if jpgs.count >= 10 {
        print("I \(jpgs.count) photos with LiDAR depth")
        session = try PhotogrammetrySession(input: jpgs.enumerated().lazy.compactMap { sample($0.offset, $0.element) }, configuration: cfg)
    } else {
        session = try PhotogrammetrySession(input: input, configuration: cfg)
    }
} catch { print("E \(error)"); exit(4) }
Task {
    do {
        try session.process(requests: [.modelFile(url: out, detail: detail)])
        var ok = false
        for try await o in session.outputs {
            switch o {
            case .requestProgress(_, let f): print("P \(f)")
            case .requestComplete: ok = true
            case .requestError(_, let e): print("E \(e)")
            case .processingComplete: exit(ok ? 0 : 5)
            case .processingCancelled: print("E cancelled"); exit(6)
            case .invalidSample(let id, let why): print("W sample \(id): \(why)")
            case .skippedSample(let id): print("W skipped \(id)")
            default: break
            }
        }
    } catch { print("E \(error)"); exit(7) }
}
dispatchMain()
