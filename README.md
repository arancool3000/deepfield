# Deepfield

An iPhone LiDAR 3D scanner (SwiftUI, ARKit, SceneKit, RealityKit, Metal; iOS 17+, LiDAR iPhone or iPad Pro).

- LiDAR point-cloud scanning with live coverage, a review step and many scan modes (room, object, OMNI, ULTRA, Gaussian...)
- On-device Gaussian splat training on the GPU, photoreal models, Apple Object Capture, 3D photos, Walk tours
- An editor with cut, measure, fill, drift and doubles fixes, and AI tools
- Export to PLY, OBJ, STL, USDZ, GLB, LAS, XYZ, DXF and 3D Gaussian Splatting files

## AI
The AI tools use **your own key**, chosen in Settings > AI: Gemini, ChatGPT (OpenAI), Claude (Anthropic), Grok (xAI),
DeepSeek or OpenRouter. Keys are stored in the iOS Keychain. No key is built in. The live voice guide and the AI
image painter use Gemini.

## Building
Open `Deepfield.swiftpm` in Swift Playgrounds or Xcode 16+. `imac-worker/` is an optional Mac helper that builds
full-detail photogrammetry models on your home network.
