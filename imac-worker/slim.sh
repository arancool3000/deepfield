#!/bin/bash
# Deepfield worker: makes Apple's full-detail model phone-sized. The 8K colour stays (JPEG), the surface-detail normal map
# goes to 4K, ambient occlusion and roughness to 1K, and the 100 MB displacement map (nothing on the phone uses it) goes.
# usage: slim.sh in.usdz out.usdz
set -e
IN="$1"; OUT="$2"; W=$(mktemp -d)
cd "$W"; unzip -q "$IN"
U=$(ls *.usdc | head -1); usdcat "$U" -o m.usda; rm -f "$U"
sed -i '' '/unconnected:displacement/d' m.usda
for f in 0/*.png; do
  case "$f" in
    *_tex0.png)  sips -s format jpeg -s formatOptions 90 "$f" --out "${f%.png}.jpg" >/dev/null ;;
    *_norm0.png) sips -Z 4096 -s format jpeg -s formatOptions 92 "$f" --out "${f%.png}.jpg" >/dev/null ;;
    *)           sips -Z 1024 -s format jpeg -s formatOptions 85 "$f" --out "${f%.png}.jpg" >/dev/null ;;
  esac
  b=$(basename "$f"); sed -i '' "s#${b}#${b%.png}.jpg#g" m.usda; rm -f "$f"
done
rm -f 0/*.exr
usdcat m.usda -o model.usdc
rm -f "$OUT"; usdzip "$OUT" model.usdc 0/*.jpg >/dev/null
cd /; rm -rf "$W"
