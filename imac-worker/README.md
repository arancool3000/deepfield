# Deepfield iMac worker

Builds Apple photogrammetry models at FULL detail on the iMac's M1 for the phone (an iPhone only makes "reduced").
Home Wi-Fi only, port 8766 (`http://Home-Computer.local:8766`).

- `main.swift` -> `dfphoto`: `swiftc -O main.swift -o dfphoto` (with `DEVELOPER_DIR` set to the Xcode in ~/Downloads).
- `slim.sh`: makes the 225 MB full-detail USDZ phone-sized (~12 MB): 8K colour JPEG, 4K normals, no displacement map.
- `server.py`: POST /job (zip of photos, ?detail=&mask=&name=), GET /job/<id>, GET /job/<id>/model.usdz, GET /ping.
  Runs as the LaunchAgent `com.arankeyhan.deepfield-worker`. One job at a time; finished jobs are deleted after a day.

Measured: 60 photos at full detail in 172 s.
