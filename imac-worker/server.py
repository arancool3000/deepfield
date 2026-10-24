#!/usr/bin/env python3
"""Deepfield iMac worker (home Wi-Fi only, port 8766).
The phone uploads a zip of photos; the M1 runs Apple photogrammetry at FULL detail (the phone can only make 'reduced'),
slims the result to a phone-sized USDZ and the phone downloads it. One job at a time; jobs are deleted after a day."""
import json, os, shutil, socket, subprocess, threading, time, uuid, zipfile, queue, re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HOME = os.path.expanduser("~/deepfield-worker")
JOBS = os.path.join(HOME, "jobs")
os.makedirs(JOBS, exist_ok=True)
IMG = re.compile(r"\.(heic|jpg|jpeg|png)$", re.I)
SIDE = re.compile(r"\.(depth|conf|json)$", re.I)       # a LiDAR frame's depth, confidence and pose: names must stay paired
jobs = {}
q = queue.Queue()
lock = threading.Lock()

def save(j):
    with open(os.path.join(JOBS, j["id"], "job.json"), "w") as f: json.dump(j, f)

def run():
    while True:
        jid = q.get()
        j = jobs[jid]; d = os.path.join(JOBS, jid)
        try:
            j["state"] = "running"; j["stage"] = "Unpacking the photos"; save(j)
            inp = os.path.join(d, "in"); os.makedirs(inp, exist_ok=True)
            with zipfile.ZipFile(os.path.join(d, "upload.zip")) as z:
                n = 0
                for info in z.infolist():
                    name = os.path.basename(info.filename)
                    if info.is_dir() or name.startswith(".") or not (IMG.search(name) or SIDE.search(name)): continue
                    target = os.path.join(inp, name)
                    if os.path.exists(target): target = os.path.join(inp, f"{n:05d}_{name}")
                    with z.open(info) as src, open(target, "wb") as dst: shutil.copyfileobj(src, dst)
                    if IMG.search(name): n += 1
            os.remove(os.path.join(d, "upload.zip"))
            j["photos"] = n
            if n < 10: raise RuntimeError(f"only {n} photos arrived")
            j["stage"] = f"Building from {n} photos on the iMac"; save(j)
            raw = os.path.join(d, "raw.usdz")
            p = subprocess.Popen(["/usr/bin/caffeinate", "-i", os.path.join(HOME, "dfphoto"), inp, raw, j["detail"], "1" if j["mask"] else "0"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            err = ""
            for line in p.stdout:
                if line.startswith("P "):
                    try: j["progress"] = round(float(line[2:]) * 0.97, 3)
                    except ValueError: pass
                elif line.startswith("E "): err = line[2:].strip()
            p.wait()
            if p.returncode != 0 or not os.path.exists(raw): raise RuntimeError(err or f"photogrammetry stopped ({p.returncode})")
            j["stage"] = "Making it phone-sized"; save(j)
            subprocess.run([os.path.join(HOME, "slim.sh"), raw, os.path.join(d, "model.usdz")], check=True, capture_output=True)
            os.remove(raw); shutil.rmtree(inp, ignore_errors=True)
            j["state"] = "done"; j["progress"] = 1; j["stage"] = "Done"; j["size"] = os.path.getsize(os.path.join(d, "model.usdz"))
        except Exception as e:
            j["state"] = "failed"; j["error"] = str(e)
        j["finished"] = time.time(); save(j)

def sweep():
    while True:
        for jid in list(jobs):
            j = jobs[jid]
            if j.get("finished") and time.time() - j["finished"] > 86400:
                shutil.rmtree(os.path.join(JOBS, jid), ignore_errors=True); jobs.pop(jid, None)
        time.sleep(3600)

class H(BaseHTTPRequestHandler):
    def send(self, code, obj=None, body=None, ctype="application/json"):
        b = body if body is not None else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
    def do_GET(self):
        u = urlparse(self.path); parts = [x for x in u.path.split("/") if x]
        if parts == ["ping"]:
            busy = sum(1 for j in list(jobs.values()) if j["state"] in ("queued", "running"))
            return self.send(200, {"ok": True, "worker": 1, "machine": "iMac M1", "busy": busy})
        j = jobs.get(parts[1]) if len(parts) >= 2 and parts[0] == "job" else None
        if j is not None:
            if len(parts) == 2:
                ahead = sum(1 for o in list(jobs.values()) if o["state"] == "queued" and o["created"] < j["created"])
                return self.send(200, dict(j, ahead=ahead))
            if parts[2] == "model.usdz" and j["state"] == "done":
                path = os.path.join(JOBS, j["id"], "model.usdz"); size = os.path.getsize(path)
                self.send_response(200); self.send_header("Content-Type", "model/vnd.usdz+zip"); self.send_header("Content-Length", str(size)); self.end_headers()
                with open(path, "rb") as f: shutil.copyfileobj(f, self.wfile)
                return
        self.send(404, {"error": "not found"})
    def do_POST(self):
        u = urlparse(self.path); qs = parse_qs(u.query)
        if u.path != "/job": return self.send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > 4_000_000_000: return self.send(413, {"error": "size"})
        jid = uuid.uuid4().hex[:12]; d = os.path.join(JOBS, jid); os.makedirs(d)
        with open(os.path.join(d, "upload.zip"), "wb") as f:
            left = n
            while left > 0:
                b = self.rfile.read(min(1 << 20, left))
                if not b: break
                f.write(b); left -= len(b)
        if left > 0: shutil.rmtree(d, ignore_errors=True); return self.send(400, {"error": "upload cut off"})
        detail = (qs.get("detail") or ["full"])[0]
        if detail not in ("medium", "full", "raw"): detail = "full"
        j = {"id": jid, "state": "queued", "progress": 0, "stage": "Waiting its turn", "detail": detail, "mask": (qs.get("mask") or ["0"])[0] == "1",
             "name": (qs.get("name") or ["Model"])[0][:80], "created": time.time()}
        jobs[jid] = j; save(j); q.put(jid)
        self.send(200, {"id": jid})

class S6(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0); super().server_bind()

# jobs from before a restart: finished ones stay downloadable, unfinished ones are reported as stopped
for jid in os.listdir(JOBS):
    try:
        with open(os.path.join(JOBS, jid, "job.json")) as f: j = json.load(f)
    except Exception: shutil.rmtree(os.path.join(JOBS, jid), ignore_errors=True); continue
    if j["state"] in ("queued", "running"): j.update(state="failed", error="the iMac restarted during the build", finished=time.time())
    jobs[jid] = j
threading.Thread(target=run, daemon=True).start()
threading.Thread(target=sweep, daemon=True).start()
S6(("::", 8766), H).serve_forever()
