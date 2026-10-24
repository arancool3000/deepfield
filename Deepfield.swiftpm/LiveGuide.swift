import Foundation
import AVFoundation
import CoreImage
import UIKit

/// Spoken scanning guide on Gemini Live (gemini-3.8-live over a WebSocket). It sees a small camera frame about once a
/// second, hears the scan's state AND you (the microphone, with echo cancelling), answers out loud, and can act:
/// torch, pause, resume, save, Smart check, scan mode.
@MainActor
final class LiveGuide: ObservableObject {
    @Published var on = false
    @Published var line = ""              // what it last said
    @Published var heard = ""             // what it last heard you say
    @Published var problem = ""
    @Published var listening = true       // microphone on while the guide is on

    static let model = "models/gemini-3.8-live"
    private var ws: URLSessionWebSocketTask?
    private var ready = false
    private var timer: Timer?
    private var tick = 0
    private var lastAsk: TimeInterval = 0, lastWeak = "", wasWarn = false
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let outFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24000, channels: 1, interleaved: false)!
    private let micFmt = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!
    private var converter: AVAudioConverter?
    private var pendingLine = "", pendingHeard = ""
    weak var scanner: Scanner?
    var kind: ScanKind?
    var subject = ""                      // what ULTRA's AI thinks is being scanned
    /// the app's handler for the guide's actions; returns a short result the model is told
    var onTool: ((String, [String: Any]) -> String)?

    var userOff = false                      // 5.0: the user switched it off - nothing turns it back on by itself
    func toggle() { if on { userOff = true; stop() } else { userOff = false; start() } }

    func start() {
        guard !on else { return }
        // 5.8: the voice guide is Gemini Live - it needs the user's own Gemini key whichever AI the tools use
        guard !Gemini.key.isEmpty else { problem = "The voice guide uses Gemini Live: add a Gemini key in Settings > AI (free at aistudio.google.com/apikey)."; return }
        on = true; problem = ""; line = "Connecting the voice guide..."; heard = ""
        AVAudioApplication.requestRecordPermission { ok in Task { @MainActor in self.startAudio(mic: ok) } }
        let url = URL(string: "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=" + Gemini.key)!
        let t = URLSession.shared.webSocketTask(with: url)
        t.maximumMessageSize = 16 << 20
        ws = t; ready = false
        t.resume()
        let what = kind.map { "The scan mode is \($0.title)." } ?? ""
        let tools: [[String: Any]] = [
            ["name": "set_torch", "description": "Turn the phone's torch (flashlight) on or off, optionally at a brightness 0.1-1",
             "parameters": ["type": "OBJECT", "properties": ["on": ["type": "BOOLEAN"], "level": ["type": "NUMBER"]], "required": ["on"]]],
            ["name": "pause_scan", "description": "Pause scanning (every point is kept)"],
            ["name": "resume_scan", "description": "Start or resume scanning"],
            ["name": "save_scan", "description": "Finish and save the scan to the library"],
            ["name": "smart_check", "description": "Turn the Smart check (weak-area finder) on or off",
             "parameters": ["type": "OBJECT", "properties": ["on": ["type": "BOOLEAN"]], "required": ["on"]]],
            ["name": "set_mode", "description": "Tune the scanner for what is being scanned",
             "parameters": ["type": "OBJECT", "properties": ["mode": ["type": "STRING", "enum": ["room", "area", "object", "surface", "person"]]], "required": ["mode"]]],
        ]
        send(["setup": [
            "model": LiveGuide.model,
            "generationConfig": ["responseModalities": ["AUDIO"]],
            "outputAudioTranscription": [String: Any](),
            "inputAudioTranscription": [String: Any](),
            "tools": [["functionDeclarations": tools]],
            "systemInstruction": ["parts": [["text": """
            You are the voice guide inside Deepfield, a 3D scanning app (iPhone LiDAR + photos). You see the camera about \
            once a second and get status messages from the app. \(what) The user can talk to you: answer their questions briefly \
            and use your tools when they ask for something (torch, pause, resume, save, Smart check, mode). When a status message \
            asks "What next?", reply in ONE short sentence (max 12 words) with the single most useful next move. Say what is being \
            scanned only ONCE in the whole session, the first time (specifically, e.g. "a robot built from perforated plastic"). The move: (an area not yet covered, a weak spot the app reports, thin or shiny parts needing a slower closer \
            pass, slowing down, more light, the right distance). If the user asks "what now?" or similar, give concrete step-by-step directions (where to stand, which side, how close). Never repeat advice. Never chat, praise or narrate. Calm and brief.
            """]]]
        ]])
        receive()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in Task { @MainActor in self?.pulse() } }
    }

    private func startAudio(mic: Bool) {
        guard on else { return }
        let ses = AVAudioSession.sharedInstance()
        do {
            try ses.setCategory(mic ? .playAndRecord : .playback, mode: mic ? .voiceChat : .spokenAudio,
                                options: mic ? [.defaultToSpeaker, .allowBluetooth, .mixWithOthers] : [.mixWithOthers, .duckOthers])
            try ses.setActive(true)
            if player.engine == nil { engine.attach(player); engine.connect(player, to: engine.mainMixerNode, format: outFmt) }
            if mic {
                let input = engine.inputNode
                try? input.setVoiceProcessingEnabled(true)       // echo cancelling: it must not hear itself
                let inFmt = input.outputFormat(forBus: 0)
                guard let conv = AVAudioConverter(from: inFmt, to: micFmt) else { throw NSError(domain: "mic", code: 1) }
                converter = conv
                let mf = micFmt
                input.removeTap(onBus: 0)
                input.installTap(onBus: 0, bufferSize: 2048, format: inFmt) { [weak self] buf, _ in
                    guard let d = LiveGuide.convert(buf, conv, mf) else { return }
                    Task { @MainActor in self?.micData(d) }
                }
            } else { listening = false }
            if !engine.isRunning { try engine.start() }
            player.play()
        } catch { problem = "Audio: \(error.localizedDescription)" }
    }

    /// microphone audio -> 16 kHz 16-bit mono (converted on the audio thread, before the buffer is reused)
    nonisolated static func convert(_ buf: AVAudioPCMBuffer, _ conv: AVAudioConverter, _ fmt: AVAudioFormat) -> Data? {
        let ratio = fmt.sampleRate / buf.format.sampleRate
        guard let out = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(Double(buf.frameLength) * ratio) + 32) else { return nil }
        var fed = false
        var err: NSError?
        conv.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return buf
        }
        guard out.frameLength > 0, let p = out.int16ChannelData else { return nil }
        return Data(bytes: p[0], count: Int(out.frameLength) * 2)
    }
    private func micData(_ d: Data) {
        guard on, ready, listening else { return }
        send(["realtimeInput": ["audio": ["mimeType": "audio/pcm;rate=16000", "data": d.base64EncodedString()]]])
    }

    func stop() {
        on = false; ready = false
        timer?.invalidate(); timer = nil
        ws?.cancel(with: .normalClosure, reason: nil); ws = nil
        engine.inputNode.removeTap(onBus: 0)
        player.stop(); engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        line = ""; heard = ""
    }

    /// tells the guide something (ULTRA uses this when it re-tunes)
    func say(_ text: String) { if ready { send(["realtimeInput": ["text": text]]) } }

    /// once a second: a frame; every 10 s while scanning: the scan state and a request for the next tip
    private func pulse() {
        guard on, ready, let s = scanner else { return }
        tick += 1
        if !s.running && tick % 4 != 0 { return }
        // 5.7 (48): a frame request still waiting (ULTRA's one-time subject question) is left alone - it would be lost
        if s.jpegRequest == nil { s.requestJPEG { [weak self] d in
            Task { @MainActor in self?.send(["realtimeInput": ["video": ["mimeType": "image/jpeg", "data": d.base64EncodedString()]]]) }
        } }
        // 4.5: it speaks only when something changed (a new weak spot, a new warning) or after 60 quiet seconds
        let now = Date().timeIntervalSince1970
        let weakIDs = s.clusters.prefix(3).map { "\($0.id)-\($0.reason.rawValue)" }.joined(separator: ",")
        let newWarn = s.warn && !wasWarn; wasWarn = s.warn
        let changed = newWarn || (!weakIDs.isEmpty && weakIDs != lastWeak) || now - lastAsk > 60
        if s.running && changed && now - lastAsk > (newWarn ? 8 : 25) {
            lastAsk = now; lastWeak = weakIDs
            let weak = s.clusters.prefix(3).map { $0.hint }.joined(separator: "; ")
            let arrow = s.targetText.isEmpty ? "" : "The app's arrow points to: \(s.targetText)."
            let msg = "Status: \(s.count) points, \(s.keyframes) photos, about \(String(format: "%.1f", s.medianDepth)) m from the surface. \(subject.isEmpty ? "" : "Subject: \(subject).") \(s.warn ? "Warning: \(s.status)." : "") \(weak.isEmpty ? "" : "Weak spots: \(weak).") \(arrow) What next?"
            send(["realtimeInput": ["text": msg]])
        }
    }

    private func send(_ obj: [String: Any]) {
        guard let ws, let d = try? JSONSerialization.data(withJSONObject: obj), let s = String(data: d, encoding: .utf8) else { return }
        ws.send(.string(s)) { [weak self] e in
            // 5.7 (46): only the live socket may fail the guide (a cancelled one must not stop a restarted session)
            if let e { Task { @MainActor in if let self, self.ws === ws { self.fail(e.localizedDescription) } } }
        }
    }

    private func receive() {
        guard let sock = ws else { return }
        sock.receive { [weak self] r in
            Task { @MainActor in
                guard let self, self.on, self.ws === sock else { return }      // 5.7 (46): ignore a stopped session's socket
                switch r {
                case .failure(let e): self.fail(e.localizedDescription); return
                case .success(let m):
                    var data: Data?
                    switch m { case .data(let d): data = d; case .string(let s): data = s.data(using: .utf8); @unknown default: break }
                    if let d = data { self.handle(d) }
                }
                self.receive()
            }
        }
    }

    private func handle(_ d: Data) {
        guard let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return }
        if j["setupComplete"] != nil { ready = true; line = "Voice guide on - talk to me any time"; lastAsk = Date().timeIntervalSince1970; return }
        if let e = j["error"] as? [String: Any] { fail(e["message"] as? String ?? "error"); return }
        if let tc = j["toolCall"] as? [String: Any], let calls = tc["functionCalls"] as? [[String: Any]] {
            var out: [[String: Any]] = []
            for c in calls {
                let name = c["name"] as? String ?? "", args = c["args"] as? [String: Any] ?? [:]
                let result = onTool?(name, args) ?? "not available"
                out.append(["id": c["id"] ?? "", "name": name, "response": ["result": result]])
            }
            send(["toolResponse": ["functionResponses": out]])
            return
        }
        guard let sc = j["serverContent"] as? [String: Any] else { return }
        if sc["interrupted"] != nil, engine.isRunning { player.stop(); player.play() }      // you talked over it: stop speaking
        if let mt = sc["modelTurn"] as? [String: Any], let parts = mt["parts"] as? [[String: Any]] {
            for p in parts {
                if let inl = p["inlineData"] as? [String: Any], let b = inl["data"] as? String, let pcm = Data(base64Encoded: b) { play(pcm) }
            }
        }
        if let tr = sc["inputTranscription"] as? [String: Any], let t = tr["text"] as? String { pendingHeard += t; heard = pendingHeard.trimmingCharacters(in: .whitespaces) }
        if let tr = sc["outputTranscription"] as? [String: Any], let t = tr["text"] as? String { pendingLine += t }
        if sc["turnComplete"] != nil {
            if !pendingLine.isEmpty { line = pendingLine.trimmingCharacters(in: .whitespaces) }
            pendingLine = ""; pendingHeard = ""
        }
    }

    /// 16-bit little-endian mono PCM at 24 kHz -> the player
    private func play(_ pcm: Data) {
        let n = pcm.count / 2
        guard n > 0, engine.isRunning, let buf = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: AVAudioFrameCount(n)) else { return }
        buf.frameLength = AVAudioFrameCount(n)
        let out = buf.floatChannelData![0]
        pcm.withUnsafeBytes { raw in
            let s = raw.bindMemory(to: Int16.self)
            for i in 0..<n { out[i] = Float(Int16(littleEndian: s[i])) / 32768 }
        }
        player.scheduleBuffer(buf, completionHandler: nil)
        if !player.isPlaying { player.play() }
    }

    private func fail(_ why: String) {
        guard on else { return }
        problem = "Voice guide stopped: \(why)"
        stop()
    }
}

extension Scanner {
    /// a small portrait JPEG of the next camera frame (the buffer is let go straight after encoding)
    /// 5.7 (48): a request already waiting is kept - both get the same frame (it used to be overwritten)
    func requestJPEG(_ done: @escaping (Data) -> Void) {
        if let prev = jpegRequest { jpegRequest = { d in prev(d); done(d) } } else { jpegRequest = done }
    }
}

let sharedCI = CIContext(options: [.useSoftwareRenderer: false])
func smallJPEG(_ pb: CVPixelBuffer, long: CGFloat = 512) -> Data? {
    var im = CIImage(cvPixelBuffer: pb).oriented(.right)
    let s = long / max(im.extent.width, im.extent.height)
    im = im.transformed(by: CGAffineTransform(scaleX: s, y: s))
    return sharedCI.jpegRepresentation(of: im, colorSpace: CGColorSpaceCreateDeviceRGB(), options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.6])
}
