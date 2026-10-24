import Foundation
import Security

/// 5.8 the AI behind every AI tool: Gemini, ChatGPT (OpenAI), Claude (Anthropic), Grok (xAI), DeepSeek or OpenRouter.
/// Each provider uses the user's own key, kept in the iOS Keychain - nothing is built in.
enum AIProvider: String, CaseIterable, Identifiable {
    case gemini, openai, claude, grok, deepseek, openrouter
    var id: String { rawValue }
    var title: String { ["gemini": "Gemini", "openai": "ChatGPT (OpenAI)", "claude": "Claude (Anthropic)", "grok": "Grok (xAI)", "deepseek": "DeepSeek", "openrouter": "OpenRouter"][rawValue]! }
    /// where to get a key
    var keyURL: String { ["gemini": "aistudio.google.com/apikey", "openai": "platform.openai.com/api-keys", "claude": "platform.claude.com/settings/keys", "grok": "console.x.ai", "deepseek": "platform.deepseek.com/api_keys", "openrouter": "openrouter.ai/keys"][rawValue]! }
    /// OpenAI-style chat completions endpoint (all but Gemini and Claude)
    var base: String { ["openai": "https://api.openai.com/v1", "grok": "https://api.x.ai/v1", "deepseek": "https://api.deepseek.com/v1", "openrouter": "https://openrouter.ai/api/v1"][rawValue] ?? "" }
    /// a sensible model when none is picked (Load models in Settings lists what the key can use)
    var defaultModel: String { ["gemini": "", "openai": "", "claude": "claude-opus-5-5", "grok": "", "deepseek": "deepseek-chat", "openrouter": "openrouter/auto"][rawValue]! }
    var seesImages: Bool { self != .deepseek }

    static var current: AIProvider { AIProvider(rawValue: UserDefaults.standard.string(forKey: "ai.provider") ?? "") ?? .gemini }
    var model: String {
        get { let m = UserDefaults.standard.string(forKey: "ai.model." + rawValue) ?? ""; return m.isEmpty ? defaultModel : m }
        nonmutating set { UserDefaults.standard.set(newValue, forKey: "ai.model." + rawValue) }
    }
    var key: String {
        get { Keys.get("ai-key-" + rawValue) ?? "" }
        nonmutating set { let v = newValue.trimmingCharacters(in: .whitespacesAndNewlines); if v.isEmpty { Keys.delete("ai-key-" + rawValue) } else { Keys.set("ai-key-" + rawValue, v) } }
    }
}

enum AIProviders {
    struct Failure: LocalizedError { let errorDescription: String? }

    /// one request to the chosen provider (not Gemini - Gemini.ask handles that): text plus optional JPEGs, text back
    static func ask(_ p: AIProvider, _ prompt: String, images: [Data], json: Bool) async throws -> String {
        let k = p.key
        guard !k.isEmpty else { throw Failure(errorDescription: "Add your \(p.title) key in Settings > AI (get one at \(p.keyURL)).") }
        let model = p.model
        guard !model.isEmpty else { throw Failure(errorDescription: "Pick a \(p.title) model in Settings > AI (Load models).") }
        let text = json ? prompt + "\n\nReply with JSON only." : prompt
        var req: URLRequest
        if p == .claude {
            // Messages API over HTTPS (no Swift SDK); Opus 5.5 thinks adaptively by default, refusals fall back server-side
            req = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
            req.setValue(k, forHTTPHeaderField: "x-api-key")
            req.setValue("2026-10-21", forHTTPHeaderField: "anthropic-version")
            req.setValue("server-side-fallback-2026-07-01", forHTTPHeaderField: "anthropic-beta")
            var content: [[String: Any]] = images.map { ["type": "image", "source": ["type": "base64", "media_type": "image/jpeg", "data": $0.base64EncodedString()]] }
            content.append(["type": "text", "text": text])
            req.httpBody = try JSONSerialization.data(withJSONObject: ["model": model, "max_tokens": 16000, "fallbacks": "default",
                                                                       "messages": [["role": "user", "content": content]]])
        } else {
            req = URLRequest(url: URL(string: p.base + "/chat/completions")!)
            req.setValue("Bearer " + k, forHTTPHeaderField: "Authorization")
            if p == .openrouter { req.setValue("Deepfield", forHTTPHeaderField: "X-Title") }
            var content: [[String: Any]] = [["type": "text", "text": text]]
            if p.seesImages { content += images.map { ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64," + $0.base64EncodedString()]] } }
            var body: [String: Any] = ["model": model, "messages": [["role": "user", "content": p.seesImages ? content as Any : text as Any]]]
            if json && p != .openrouter { body["response_format"] = ["type": "json_object"] }
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        req.httpMethod = "POST"; req.timeoutInterval = 150
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        let j = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        guard code == 200, let j else {
            let msg = (j?["error"] as? [String: Any])?["message"] as? String ?? String(data: data.prefix(200), encoding: .utf8) ?? ""
            if code == 401 || code == 403 { throw Failure(errorDescription: "\(p.title) refused the key: \(msg)") }
            if code == 429 { throw Failure(errorDescription: "\(p.title): rate limit or credit used up. \(msg)") }
            throw Failure(errorDescription: "\(p.title) did not answer (HTTP \(code)). \(msg)")
        }
        var out = ""
        if p == .claude {
            if j["stop_reason"] as? String == "refusal" { throw Failure(errorDescription: "Claude declined this request.") }
            out = ((j["content"] as? [[String: Any]]) ?? []).filter { $0["type"] as? String == "text" }.compactMap { $0["text"] as? String }.joined()
        } else {
            out = ((j["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String ?? ""
        }
        guard !out.isEmpty else { throw Failure(errorDescription: "\(p.title) sent an empty answer.") }
        return json ? stripFences(out) : out
    }

    /// ```json ... ``` around a JSON answer is removed
    static func stripFences(_ s: String) -> String {
        var t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.hasPrefix("```") { t = String(t.drop(while: { $0 != "\n" }).dropFirst()); if let r = t.range(of: "```", options: .backwards) { t = String(t[..<r.lowerBound]) } }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// the models this key can use (for the Settings picker)
    static func models(_ p: AIProvider) async throws -> [String] {
        let k = p.key
        guard !k.isEmpty else { throw Failure(errorDescription: "Add the key first.") }
        var req: URLRequest
        switch p {
        case .gemini: req = URLRequest(url: URL(string: "https://generativelanguage.googleapis.com/v1beta/models")!); req.setValue(k, forHTTPHeaderField: "x-goog-api-key")
        case .claude: req = URLRequest(url: URL(string: "https://api.anthropic.com/v1/models")!); req.setValue(k, forHTTPHeaderField: "x-api-key"); req.setValue("2026-10-21", forHTTPHeaderField: "anthropic-version")
        default: req = URLRequest(url: URL(string: p.base + "/models")!); req.setValue("Bearer " + k, forHTTPHeaderField: "Authorization")
        }
        req.timeoutInterval = 30
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200, let j = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw Failure(errorDescription: "\(p.title) refused the key.") }
        if p == .gemini {
            return ((j["models"] as? [[String: Any]]) ?? []).compactMap { m in
                guard ((m["supportedGenerationMethods"] as? [String]) ?? []).contains("generateContent") else { return nil }
                return (m["name"] as? String).map { $0.hasPrefix("models/") ? String($0.dropFirst(7)) : $0 }
            }
        }
        return ((j["data"] as? [[String: Any]]) ?? []).compactMap { $0["id"] as? String }.sorted()
    }
}

/// a small Keychain wrapper (one generic password per account name)
enum Keys {
    private static func q(_ a: String) -> [String: Any] { [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "KindleHubScan", kSecAttrAccount as String: a] }
    static func get(_ a: String) -> String? {
        var x = q(a); x[kSecReturnData as String] = true; x[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: AnyObject?
        guard SecItemCopyMatching(x as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        return String(data: d, encoding: .utf8)
    }
    static func set(_ a: String, _ v: String) {
        let d = Data(v.utf8)
        if SecItemUpdate(q(a) as CFDictionary, [kSecValueData as String: d] as CFDictionary) == errSecItemNotFound {
            var x = q(a); x[kSecValueData as String] = d; SecItemAdd(x as CFDictionary, nil)
        }
    }
    static func delete(_ a: String) { SecItemDelete(q(a) as CFDictionary) }
}
