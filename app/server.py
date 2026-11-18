"""Backend deteksi APD + zona dari file video CCTV.

Jalankan:  py -3 server.py   lalu buka http://localhost:8000

Pipeline per area (satu thread per video):
  baca frame -> YOLOv8n track (orang + kendaraan) -> model APD -> cocokkan APD ke orang
  -> aturan (APD wajib, zona terlarang, jarak kendaraan, posisi rebah) -> event + foto bukti
  -> frame beranotasi untuk stream MJPEG

Batasan yang harus diingat:
- Helm: model khusus keremberke/yolov8s-hard-hat-detection (mAP@0.5 = 0.83).
- Model APD publik lain (mAP@0.5 = 0.27): masker, sarung tangan, kacamata, sepatu. Helm dari model
  ini tidak dipakai karena di uji video tidak pernah mendeteksi helm sama sekali.
  Earplug, apron, jas lab, jaket, kedok las, jas hujan, seragam = "tidak terdeteksi" (abu-abu).
- Forklift tidak ada di COCO; dipakai kelas car/truck/motorcycle sebagai proksi.
- Jarak kendaraan dan zona dalam piksel/rasio frame, belum dikalibrasi ke meter.
- Akurasi sangat bergantung sudut kamera, resolusi orang (min ~120 px tinggi) dan pencahayaan.
"""
import json
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
VIDEOS = BASE / "videos"
SNAPS = BASE / "events"
DASHBOARD = BASE.parent / "dashboard.html"
VIDEOS.mkdir(exist_ok=True)
SNAPS.mkdir(exist_ok=True)

# ---------- Konfigurasi (tunable per lokasi) ----------
CFG = {
    "person_conf": 0.35,
    "vehicle_conf": 0.70,      # proksi forklift (COCO) sering salah: palet kardus terbaca "truk" s/d 0.64
    "ppe_conf": 0.30,
    "helmet_conf": 0.30,
    "helmet_colors_allowed": ["putih", "kuning", "merah"],  # warna helm yang boleh; lainnya = pelanggaran
    "helmet_color_min_votes": 5,     # minimal pembacaan warna dalam window_s sebelum dinilai
    "helmet_color_majority": 0.60,   # warna terlarang harus >= 60% pembacaan (cegah salah baca sesaat)
    "helmet_color_min_px": 20,       # helm lebih sempit dari ini (orang jauh) tidak dibaca warnanya
    "ppe_mode": "frame",       # "frame" = seluruh frame sekaligus; "crop" = per orang (potong + perbesar).
                               # Uji label manual: crop menemukan helm lebih banyak (86% vs 80%) tapi lebih
                               # sering mengira topi/helm motor sebagai helm (32% vs 18%) -> default frame.
    "crop_size": 320,          # sisi panjang potongan orang sebelum masuk model APD (mode crop)
    "max_crops": 24,           # batas jumlah orang per frame yang dicek APD-nya (mode crop)
    "imgsz": 640,              # model APD
    "person_imgsz": 960,       # orang jauh lebih terdeteksi di 960; di GPU hanya +0,4 ms vs 640 (di CPU ±2× lebih lambat)
    "max_width": 960,          # frame diperkecil ke lebar ini sebelum inferensi
    "min_head_px": 12,         # taksiran tinggi kepala minimal (px) agar APD dinilai; lihat head_px()
    "window_s": 4.0,           # detik terakhir yang dipakai untuk voting APD (berbasis waktu, bukan jumlah
                               # frame: di GPU 30 fps, 15 frame hanya 0,5 s -> pekerja membungkuk sebentar = alarm)
    "min_neg": 3,              # minimal deteksi "no_X" dalam window_s untuk pelanggaran (dan lebih banyak dari positif)
    "absent_as_missing": ["helm"],  # item yang dianggap tidak dipakai bila tak pernah terlihat
    "absent_after_s": 4.0,     # ... setelah orang terlacak selama ini
    "event_cooldown_s": 30,    # jeda event sama untuk orang yang sama
    "vehicle_near_ratio": 0.18,  # jarak horizontal kaki orang ke kendaraan < rasio lebar frame -> peringatan
    "vehicle_depth_ratio": 0.08,  # ... dan selisih tinggi titik pijak (kaki vs dasar kendaraan) < rasio tinggi frame
    "on_vehicle_overlap": 0.30,   # orang yang >30% kotaknya di dalam kotak kendaraan = pengendara/operator
    "vehicle_area_cooldown_s": 30,  # maksimal 1 alarm kendaraan per area per jeda ini (ID orang sering berganti)
    "fall_ratio": 1.25,        # lebar/tinggi kotak orang di atas ini = rebah
    "fall_s": 1.5,
}

# kelas model APD -> id APD dashboard (positif, negatif)
PPE_MAP = {  # helmet/no_helmet sengaja tidak dipetakan: ditangani HELMET_MAP
    "mask": ("masker", True), "no_mask": ("masker", False),
    "glove": ("sarung", True), "no_glove": ("sarung", False),
    "goggles": ("kacamata", True), "no_goggles": ("kacamata", False),
    "shoes": ("sepatu", True), "no_shoes": ("sepatu", False),
}
HELMET_MAP = {"Hardhat": ("helm", True), "NO-Hardhat": ("helm", False)}
# APD dashboard -> item yang bisa dicek model
ALIAS = {"helm": "helm", "masker": "masker", "sarung": "sarung", "latex": "sarung",
         "kacamata": "kacamata", "kedok": None, "sepatu": "sepatu", "boot": "sepatu"}
LABEL = {"helm": "Helm", "masker": "Masker", "sarung": "Sarung tangan", "kacamata": "Kacamata", "sepatu": "Sepatu"}
VEHICLE = {2: "mobil", 3: "motor", 7: "truk"}

# ---------- State global ----------
lock = threading.Lock()
rules = {}       # area_id -> {"req": [...], "zone": [x,y,w,h] (0..1) | None, "name": str, "vehicle_alarm": bool}
jobs = {}        # area_id -> Job
events = []      # semua event (dict) berurutan


def _last_event_id():
    """Nomor event terakhir di log/foto, supaya restart server tidak menimpa foto bukti lama."""
    last = 0
    try:
        with open(BASE / "events.jsonl", encoding="utf-8") as f:
            last = max((json.loads(l)["id"] for l in f if l.strip()), default=0)
    except FileNotFoundError:
        pass
    snaps = [int(f.name[:6]) for f in SNAPS.glob("*.jpg") if f.name[:6].isdigit()]
    return max([last] + snaps)


next_event_id = _last_event_id() + 1

_models_lock = threading.Lock()


def load_models():
    from ultralytics import YOLO
    with _models_lock:
        return (YOLO(str(MODELS / "yolov8n.pt")), YOLO(str(MODELS / "ppe_best.pt")),
                YOLO(str(MODELS / "helmet_best.pt")))


def add_event(area, kind, sev, msg, frame=None, track=None):
    global next_event_id
    with lock:
        eid = next_event_id
        next_event_id += 1
    snap = None
    if frame is not None:
        snap = f"{eid:06d}_a{area}.jpg"
        cv2.imwrite(str(SNAPS / snap), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    ev = {"id": eid, "t": time.time(), "area": area, "type": kind, "sev": sev, "msg": msg,
          "track": track, "snap": snap, "source": "AI"}
    with lock:
        events.append(ev)
        if len(events) > 2000:
            del events[:500]
    with open(BASE / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _ppe_runs(checkable, ppe, helmet):
    """(model, peta kelas, conf) — tiap model hanya dijalankan bila item-nya diwajibkan di area ini."""
    runs = []
    if "helm" in checkable:
        runs.append((helmet, HELMET_MAP, CFG["helmet_conf"]))
    if any(i != "helm" for i in checkable):
        runs.append((ppe, PPE_MAP, CFG["ppe_conf"]))
    return runs


def head_px(box):
    """Taksiran tinggi kepala (px) dari kotak orang.

    Orang berdiri penuh (kotak ramping): kepala ±1/7 tinggi kotak. Orang yang badan bawahnya
    tertutup meja/mesin (kotak pendek-lebar): kepala ±1/3 lebar kotak. Ambil yang terbesar,
    supaya orang setengah badan dengan kepala jelas tetap dinilai APD-nya.
    """
    x1, y1, x2, y2 = box
    return max((y2 - y1) / 7, (x2 - x1) / 3)


def helmet_color(frame, box):
    """Warna dominan helm dari piksel tengah-atas kotak helm (HSV), atau None bila tidak yakin.

    Area baca 8-50% tinggi & 20-80% lebar kotak, supaya wajah/latar tidak ikut terbaca.
    Batas kuning dimulai di hue 14: di uji video malam, helm kuning di bawah lampu kerja
    terbaca hue 14-16, sedangkan helm kuning siang/stock >= 19; helm oranye asli ±10-12.
    Hitam/abu gelap tidak dinilai (bayangan, helm membelakangi cahaya).
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < CFG["helmet_color_min_px"] or h < 6:
        return None
    roi = frame[max(0, y1 + int(.08 * h)):y1 + int(.50 * h), max(0, x1 + int(.20 * w)):x2 - int(.20 * w)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    votes = {
        "putih": (sat < 55) & (val > 120),
        "kuning": (hue >= 14) & (hue < 35) & (sat >= 80) & (val >= 90),
        "oranye": (hue >= 7) & (hue < 14) & (sat >= 100) & (val >= 90),
        "merah": ((hue < 7) | (hue >= 165)) & (sat >= 90) & (val >= 60),
        "hijau": (hue >= 35) & (hue < 85) & (sat >= 70) & (val >= 60),
        "biru": (hue >= 90) & (hue < 130) & (sat >= 70) & (val >= 50),
    }
    counts = {k: int(v.sum()) for k, v in votes.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] >= 0.25 * len(hsv) else None


def detect_ppe(frame, persons, checkable, ppe, helmet, mode=None):
    """Kembalikan ({track_id: {item: True/False}}, [((item, positif), xyxy, conf), ...] untuk digambar).

    mode "frame": model APD melihat seluruh frame sekaligus.
    mode "crop" : tiap orang dipotong (sedikit diperlebar) dan diperbesar ke crop_size, lalu dicek
                  sendiri-sendiri, sehingga helm/masker orang jauh jauh lebih besar di mata model.
    Helm (Hardhat/NO-Hardhat) hanya dihitung bila berada di area kepala (±1/3 atas kotak orang) dan
    diberikan ke orang yang kepalanya paling dekat, supaya helm orang di belakang tidak terhitung
    milik orang di depannya. Orang yang kepalanya terpotong tepi atas frame tidak dinilai helmnya.
    Item lain harus berada di dalam kotak orangnya.
    """
    mode = mode or CFG["ppe_mode"]
    H, W = frame.shape[:2]
    targets = [(tid, box) for tid, box, _ in persons
               if tid is not None and head_px(box) >= CFG["min_head_px"]]
    runs = _ppe_runs(checkable, ppe, helmet)
    obs, dets = {}, []
    if not targets or not runs:
        return obs, dets

    cands = []  # (item_pos, xyxy, conf, tid sumber crop / None)
    if mode == "frame":
        for model, cmap, conf in runs:
            for b in model.predict(frame, conf=conf, imgsz=CFG["imgsz"], verbose=False)[0].boxes:
                name = model.names[int(b.cls[0])]
                if name in cmap and cmap[name][0] in checkable:
                    cands.append((cmap[name], [int(v) for v in b.xyxy[0].tolist()], float(b.conf[0]), None))
    else:
        crops, meta = [], []
        for tid, (x1, y1, x2, y2) in sorted(targets, key=lambda t: -(t[1][3] - t[1][1]))[:CFG["max_crops"]]:
            pw, ph = x2 - x1, y2 - y1
            cx1, cy1 = max(0, int(x1 - 0.15 * pw)), max(0, int(y1 - 0.12 * ph))
            cx2, cy2 = min(W, int(x2 + 0.15 * pw)), min(H, int(y2 + 0.05 * ph))
            crop = frame[cy1:cy2, cx1:cx2]
            scale = CFG["crop_size"] / max(crop.shape[:2])
            crops.append(cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR))
            meta.append((tid, cx1, cy1, scale))
        for model, cmap, conf in runs:
            for res, (tid, ox, oy, scale) in zip(
                    model.predict(crops, conf=conf, imgsz=CFG["crop_size"], verbose=False), meta):
                for b in res.boxes:
                    name = model.names[int(b.cls[0])]
                    if name in cmap and cmap[name][0] in checkable:
                        a1, b1, a2, b2 = [v / scale for v in b.xyxy[0].tolist()]
                        cands.append((cmap[name], [int(a1 + ox), int(b1 + oy), int(a2 + ox), int(b2 + oy)],
                                      float(b.conf[0]), tid))

    def iou(p, q):
        ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
        iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
        inter = ix * iy
        return inter / max((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter, 1)

    # helm yang sama bisa muncul di beberapa crop yang bertumpuk: simpan yang confidence-nya tertinggi
    kept = []
    for c in sorted(cands, key=lambda c: -c[2]):
        if c[0][0] == "helm" and any(k[0][0] == "helm" and iou(k[1], c[1]) > 0.5 for k in kept):
            continue
        kept.append(c)

    boxes = dict(targets)
    for item_pos, (a1, b1, a2, b2), conf, src in kept:
        item = item_pos[0]
        cx, cy = (a1 + a2) / 2, (b1 + b2) / 2
        owner = None
        if item == "helm":
            # tahap 1: area kepala (±1/3 atas kotak); tahap 2 (bila tak ada): di mana saja dalam kotak,
            # untuk orang membungkuk yang kepalanya di samping/tengah kotak
            for head_only in (True, False):
                best = None
                for tid, (x1, y1, x2, y2) in targets:
                    pw, ph = x2 - x1, y2 - y1
                    if y1 <= 0.03 * H:  # kepala terpotong tepi atas frame
                        continue
                    top, bottom = (y1 - 0.10 * ph, y1 + 0.35 * ph) if head_only else (y1 - 0.10 * ph, y2)
                    if not (x1 - 0.05 * pw <= cx <= x2 + 0.05 * pw and top <= cy <= bottom):
                        continue
                    d = (((cx - (x1 + x2) / 2) / pw) ** 2 + ((cy - (y1 + 0.12 * ph)) / ph) ** 2) ** 0.5
                    if best is None or d < best[0]:
                        best = (d, tid)
                if best:
                    owner = best[1]
                    break
        else:
            m = 0.10 if src is None else 0.05
            for tid in ([src] if src is not None else boxes):
                x1, y1, x2, y2 = boxes[tid]
                mx, my = (x2 - x1) * m, (y2 - y1) * m
                if x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my:
                    owner = tid
                    break
        if owner is None:
            continue
        o = obs.setdefault(owner, {})
        o[item] = o.get(item, False) or item_pos[1]  # positif menang
        if item == "helm" and item_pos[1] and "helm_color" not in o:
            color = helmet_color(frame, [a1, b1, a2, b2])
            if color:
                o["helm_color"] = color
        dets.append((item_pos, [a1, b1, a2, b2], conf))
    return obs, dets

class Track:
    def __init__(self, now):
        self.first = now
        self.hist = deque(maxlen=2000)  # (waktu, {item: True/False}) per frame; dipangkas ke window_s saat dipakai
        self.last_event = {}                     # jenis -> waktu
        self.lying_since = None
        self.missing = []

    def cooldown(self, key, now):
        if now - self.last_event.get(key, -1e9) < CFG["event_cooldown_s"]:
            return False
        self.last_event[key] = now
        return True


class Job(threading.Thread):
    def __init__(self, area, path):
        super().__init__(daemon=True)
        self.area, self.path = area, path
        self.stop_flag = False
        self.jpeg = None
        self.stat = {"state": "memuat model", "fps": 0.0, "persons": 0, "compliant": 0,
                     "file": Path(path).name, "missing_now": [], "error": None}
        self.tracks = {}
        self.last_vehicle_alarm = -1e9

    def required(self):
        req = rules.get(self.area, {}).get("req", [])
        checkable = sorted({ALIAS[k] for k in req if ALIAS.get(k)})
        unchecked = [k for k in req if not ALIAS.get(k)]
        return checkable, unchecked

    def run(self):
        try:
            self._run()
        except Exception as e:  # tampilkan ke dashboard, jangan diam-diam mati
            self.stat.update(state="error", error=repr(e))

    def _run(self):
        coco, ppe, helmet = load_models()
        import torch  # ultralytics otomatis memakai GPU CUDA bila tersedia
        self.stat["device"] = "GPU" if torch.cuda.is_available() else "CPU"
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError("Video tidak bisa dibuka (codec?)")
        vfps = cap.get(cv2.CAP_PROP_FPS) or 25
        self.stat["state"] = "berjalan"
        t0, played, ema = time.time(), 0, None
        while not self.stop_flag:
            # lompati frame agar mengikuti waktu nyata walau CPU lambat
            target = int((time.time() - t0) * vfps)
            while played < target - 1:
                if not cap.grab():
                    break
                played += 1
            ok, frame = cap.read()
            played += 1
            if not ok:  # putar ulang seperti CCTV
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                t0, played = time.time(), 0
                for t in self.tracks.values():
                    t.hist.clear()
                continue
            h, w = frame.shape[:2]
            if w > CFG["max_width"]:
                frame = cv2.resize(frame, (CFG["max_width"], int(h * CFG["max_width"] / w)))
            ts = time.time()
            self.process(frame, coco, ppe, helmet)
            dt = time.time() - ts
            ema = dt if ema is None else ema * 0.9 + dt * 0.1
            self.stat["fps"] = round(1 / ema, 1) if ema else 0
        cap.release()
        self.stat["state"] = "berhenti"

    def process(self, frame, coco, ppe, helmet):
        now = time.time()
        H, W = frame.shape[:2]
        checkable, unchecked = self.required()
        zone = rules.get(self.area, {}).get("zone")

        r = coco.track(frame, persist=True, classes=[0, 2, 3, 7],
                       conf=min(CFG["person_conf"], CFG["vehicle_conf"]),
                       imgsz=CFG["person_imgsz"], tracker="bytetrack.yaml", verbose=False)[0]
        persons, vehicles = [], []
        for b in r.boxes:
            cls = int(b.cls[0])
            if float(b.conf[0]) < (CFG["person_conf"] if cls == 0 else CFG["vehicle_conf"]):
                continue
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            tid = int(b.id[0]) if b.id is not None else None
            (persons if cls == 0 else vehicles).append((tid, xyxy, cls))

        obs_by_tid, ppe_dets = detect_ppe(frame, persons, checkable, ppe, helmet)

        ann = frame.copy()
        if zone:
            zx, zy, zw, zh = int(zone[0] * W), int(zone[1] * H), int(zone[2] * W), int(zone[3] * H)
            ov = ann.copy()
            cv2.rectangle(ov, (zx, zy), (zx + zw, zy + zh), (60, 60, 230), -1)
            ann = cv2.addWeighted(ov, 0.18, ann, 0.82, 0)
            cv2.rectangle(ann, (zx, zy), (zx + zw, zy + zh), (60, 60, 230), 2)
            cv2.putText(ann, "ZONA TERLARANG", (zx + 6, zy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 255), 2)
        for _, (x1, y1, x2, y2), cls in vehicles:
            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(ann, f"kendaraan ({VEHICLE[cls]})", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        for (item, pos), (x1, y1, x2, y2), _ in ppe_dets:
            cv2.rectangle(ann, (x1, y1), (x2, y2), (80, 200, 80) if pos else (60, 60, 255), 1)

        n_ok, missing_now, seen = 0, set(), set()
        for tid, (x1, y1, x2, y2), _ in persons:
            if tid is None:
                continue
            seen.add(tid)
            tr = self.tracks.setdefault(tid, Track(now))
            ph, pw = y2 - y1, x2 - x1
            problems = []
            pending = False  # True = bukti APD belum cukup untuk dinilai
            too_small = bool(checkable) and head_px((x1, y1, x2, y2)) < CFG["min_head_px"]  # kepala terlalu kecil

            def shot():  # foto bukti: frame + orang yang bersangkutan ditandai merah
                s = ann.copy()
                cv2.rectangle(s, (x1, y1), (x2, y2), (0, 0, 255), 3)
                return s

            # --- APD: pengamatan frame ini (dari detect_ppe), lalu voting beberapa frame ---
            if checkable and not too_small:
                tr.hist.append((now, obs_by_tid.get(tid, {})))
                while tr.hist and now - tr.hist[0][0] > CFG["window_s"]:
                    tr.hist.popleft()
                recent = [o for _, o in tr.hist]
                # kepala terpotong tepi atas frame -> "helm tak terlihat" bukan bukti tidak memakai helm
                head_visible = y1 > 0.03 * H
                missing = []
                for item in checkable:
                    pos = sum(1 for o in recent if o.get(item) is True)
                    neg = sum(1 for o in recent if o.get(item) is False)
                    if neg >= CFG["min_neg"] and neg > pos:
                        missing.append(item)
                    elif (item in CFG["absent_as_missing"] and pos == 0 and head_visible
                          and now - tr.first >= CFG["absent_after_s"] and len(recent) >= 3):
                        missing.append(item)
                tr.missing = missing
                # --- warna helm: mayoritas pembacaan dalam window_s ---
                colors = [o["helm_color"] for o in recent if o.get("helm_color")]
                tr.helm_color = None
                if "helm" in checkable and "helm" not in missing and len(colors) >= CFG["helmet_color_min_votes"]:
                    top, n_top = Counter(colors).most_common(1)[0]
                    tr.helm_color = top
                    if top not in CFG["helmet_colors_allowed"] and n_top / len(colors) >= CFG["helmet_color_majority"]:
                        problems.append(f"Helm {top}")
                        missing_now.add("helm")
                        if tr.cooldown("warna:" + top, now):
                            add_event(self.area, "APD", "med",
                                      f"Orang #{tid} memakai helm {top} (wajib: {'/'.join(CFG['helmet_colors_allowed'])})",
                                      shot(), tid)
                pending = not missing and (len(recent) < 3 or now - tr.first < CFG["absent_after_s"])
                if missing:
                    problems.append("-" + ",".join(LABEL[m] for m in missing))
                    missing_now.update(missing)
                    key = "APD:" + ",".join(missing)
                    if tr.cooldown(key, now):
                        add_event(self.area, "APD", "med",
                                  f"Orang #{tid} tanpa {', '.join(LABEL[m] for m in missing)}", shot(), tid)

            foot = ((x1 + x2) / 2, y2)
            # --- Zona terlarang (titik kaki di dalam zona) ---
            if zone:
                zx, zy, zw, zh = zone[0] * W, zone[1] * H, zone[2] * W, zone[3] * H
                if zx <= foot[0] <= zx + zw and zy <= foot[1] <= zy + zh:
                    problems.append("ZONA")
                    if tr.cooldown("zona", now):
                        add_event(self.area, "Intrusi zona", "high", f"Orang #{tid} masuk zona terlarang", shot(), tid)
            # pengendara/operator (kotaknya menempel di kendaraan) tidak dinilai jarak & posisi rebah
            def overlap(v):
                ix = max(0, min(x2, v[2]) - max(x1, v[0]))
                iy = max(0, min(y2, v[3]) - max(y1, v[1]))
                return ix * iy / max(pw * ph, 1)
            on_vehicle = any(overlap(v) > CFG["on_vehicle_overlap"] for _, v, _ in vehicles)
            # --- Kendaraan terlalu dekat: jarak di titik pijak, hanya bila kedalamannya mirip ---
            # opsional per area (checklist di dashboard); default mati
            vehicle_alarm = rules.get(self.area, {}).get("vehicle_alarm", False)
            for _, (v1, w1, v2, w2), cls in ([] if on_vehicle or not vehicle_alarm else vehicles):
                gap_x = max(0, v1 - foot[0], foot[0] - v2)
                gap_y = abs(w2 - foot[1])
                if gap_y < CFG["vehicle_depth_ratio"] * H and gap_x < CFG["vehicle_near_ratio"] * W:
                    problems.append("DEKAT KENDARAAN")
                    if (now - self.last_vehicle_alarm >= CFG["vehicle_area_cooldown_s"]
                            and tr.cooldown("kendaraan", now)):
                        self.last_vehicle_alarm = now
                        add_event(self.area, "Forklift", "high",
                                  f"Orang #{tid} dekat {VEHICLE[cls]} (jarak {gap_x / W:.0%} lebar frame)", shot(), tid)
                    break
            # --- Rebah / kemungkinan terjatuh ---
            # orang yang terpotong tepi frame (atau lengan terentang dekat kamera) bukan rebah
            ex, ey = 0.03 * W, 0.03 * H
            at_edge = x1 <= ex or y1 <= ey or x2 >= W - ex or y2 >= H - ey
            if not at_edge and not on_vehicle and pw / max(ph, 1) > CFG["fall_ratio"]:
                tr.lying_since = tr.lying_since or now
                if now - tr.lying_since >= CFG["fall_s"]:
                    problems.append("JATUH?")
                    if tr.cooldown("jatuh", now):
                        add_event(self.area, "Perilaku tidak aman", "high",
                                  f"Orang #{tid} kemungkinan terjatuh (posisi rebah)", shot(), tid)
            else:
                tr.lying_since = None

            if not problems and not pending and not too_small:
                n_ok += 1
            if problems:
                col, txt = (0, 0, 255), " ".join(problems)
            elif too_small:
                col, txt = (150, 150, 150), "terlalu kecil"
            elif pending:
                col, txt = (0, 200, 230), "cek APD..."
            else:
                col, txt = (60, 200, 60), "OK" + (f" ({tr.helm_color})" if getattr(tr, "helm_color", None) else "")
            cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
            label = f"#{tid} {txt}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(ann, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
            cv2.putText(ann, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for tid in [t for t in self.tracks if t not in seen and now - self.tracks[t].first > 120]:
            del self.tracks[tid]

        hud = f"CAM-{self.area:02d}  {rules.get(self.area, {}).get('name', '')}  {time.strftime('%H:%M:%S')}  AI {self.stat['fps']} fps ({self.stat.get('device', '?')})"
        cv2.rectangle(ann, (0, 0), (W, 24), (0, 0, 0), -1)
        cv2.putText(ann, hud, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if unchecked:
            cv2.putText(ann, "Tidak bisa dicek model: " + ", ".join(unchecked), (8, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        self.jpeg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        self.stat.update(persons=len([p for p in persons if p[0] is not None]), compliant=n_ok,
                         missing_now=sorted(missing_now), unchecked=unchecked)


# ---------- API ----------
app = FastAPI(title="Surveillance APD")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def index():
    return FileResponse(DASHBOARD)


@app.get("/api/health")
def health():
    ok = all((MODELS / f).exists() for f in ("ppe_best.pt", "yolov8n.pt", "helmet_best.pt"))
    return {"ok": True, "models": ok, "config": CFG}


@app.post("/api/rules")
def set_rules(body: dict):
    # {"<area_id>": {"req": [...], "zone": [x,y,w,h] | null, "name": "...", "vehicle_alarm": bool}}
    with lock:
        for k, v in body.items():
            cur = rules.setdefault(int(k), {})
            cur.update({kk: vv for kk, vv in v.items() if kk in ("req", "zone", "name", "vehicle_alarm")})
    return {"ok": True}


@app.post("/api/upload")
async def upload(area: int = Form(...), file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}:
        raise HTTPException(400, "Format video tidak didukung")
    dest = VIDEOS / f"area{area:02d}{suffix}"
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    stop(area)
    job = Job(area, str(dest))
    job.stat["file"] = file.filename
    jobs[area] = job
    job.start()
    return {"ok": True, "file": file.filename}


@app.post("/api/stop/{area}")
def stop(area: int):
    job = jobs.pop(area, None)
    if job:
        job.stop_flag = True
    return {"ok": True}


@app.get("/api/status")
def status():
    return {str(a): j.stat for a, j in jobs.items()}


@app.get("/api/events")
def get_events(since: int = 0):
    with lock:
        return [e for e in events if e["id"] > since][-200:]


@app.get("/api/snap/{name}")
def snap(name: str):
    p = (SNAPS / name).resolve()
    if p.parent != SNAPS.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/stream/{area}")
def stream(area: int):
    def gen():
        last = None
        while area in jobs:
            buf = jobs[area].jpeg
            if buf is not None and buf is not last:
                last = buf
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
            time.sleep(0.04)
    if area not in jobs:
        return JSONResponse({"error": "tidak ada video berjalan"}, status_code=404)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
