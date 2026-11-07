"""
Лабораторна робота №6. Робота з відкритим програмним кодом.
Виконала: студентка групи ФБ-31 Голомовза Дар'я

Стаття:  Z. Ge, S. Liu, F. Wang, Z. Li, J. Sun. "YOLOX: Exceeding YOLO Series in 2021",
         arXiv:2107.08430, 2021.  https://arxiv.org/abs/2107.08430
Код:     https://github.com/Megvii-BaseDetection/YOLOX  (офіційні ONNX-моделі з Releases 0.1.1rc0)

Експерименти:
  1) демо всіх чотирьох моделей (Nano, Tiny, S, M) на тестових фото;
  2) швидкість інференсу на CPU (ONNX Runtime);
  3) точність на COCO128 (mAP@0.5 і mAP@0.5:0.95, власна реалізація COCO-метрики);
  4) спроба покращення: test-time augmentation (горизонтальне віддзеркалення) + об'єднання з NMS;
  5) спроба покращення: NMS окремо для кожного класу (class-aware) замість class-agnostic з офіційного демо.
"""
import argparse, glob, json, os, time
import cv2, numpy as np, onnxruntime as ort
from run_official_demo import install_torch_free_yolox

ap = argparse.ArgumentParser()
ap.add_argument("--yolox-dir", default="../YOLOX")
ap.add_argument("--coco128", default="data/coco128")
ap.add_argument("--images", nargs="*", default=None)
args = ap.parse_args()
install_torch_free_yolox(os.path.abspath(args.yolox_dir))
from yolox.data.data_augment import preproc            # офіційна передобробка (letterbox + паддінг 114)
from yolox.utils import demo_postprocess, multiclass_nms, vis
from yolox.data.datasets import COCO_CLASSES

MODELS = {"YOLOX-Nano": ("models/yolox_nano.onnx", 416), "YOLOX-Tiny": ("models/yolox_tiny.onnx", 416),
          "YOLOX-S": ("models/yolox_s.onnx", 640), "YOLOX-M": ("models/yolox_m.onnx", 640)}
os.makedirs("outputs", exist_ok=True)


class Detector:
    def __init__(self, path, size):
        so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count()
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.size = (size, size); self.inp = self.s.get_inputs()[0].name

    def raw(self, img):
        """Повертає боксі xyxy (у пікселях оригіналу) і матрицю скорів (N, 80) — до NMS."""
        x, r = preproc(img, self.size)
        out = self.s.run(None, {self.inp: x[None]})[0]
        p = demo_postprocess(out, self.size)[0]
        b = p[:, :4]; sc = p[:, 4:5] * p[:, 5:]            # obj * cls, як в офіційному демо
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / r
        return xyxy, sc

    def detect(self, img, score_thr=0.001, nms=0.65, tta=False, class_agnostic=True):
        xyxy, sc = self.raw(img)
        if tta:                                              # горизонтальне віддзеркалення
            w = img.shape[1]
            fb, fs = self.raw(img[:, ::-1].copy())
            fb = fb.copy(); fb[:, [0, 2]] = w - fb[:, [2, 0]]
            xyxy, sc = np.vstack([xyxy, fb]), np.vstack([sc, fs])
        d = multiclass_nms(xyxy, sc, nms_thr=nms, score_thr=score_thr, class_agnostic=class_agnostic)
        return np.zeros((0, 6)) if d is None else d          # x1,y1,x2,y2,score,cls


# ---------- 1. Демо на фото ----------
imgs = args.images or [os.path.join(args.yolox_dir, "assets/dog.jpg"), "data/bus.jpg", "data/kitchen.jpg"]
dets = {m: Detector(*cfg) for m, cfg in MODELS.items()}
rows = []
for ip in imgs:
    img = cv2.imread(ip); tiles = []
    for m, d in dets.items():
        r = d.detect(img, score_thr=0.3, nms=0.45)
        v = vis(img.copy(), r[:, :4], r[:, 4], r[:, 5], conf=0.3, class_names=COCO_CLASSES)
        cv2.putText(v, m, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)
        cv2.putText(v, m, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
        tiles.append(cv2.resize(v, (640, int(640 * img.shape[0] / img.shape[1]))))
        rows.append(dict(image=os.path.basename(ip), model=m, n=int(len(r)),
                         objects=sorted(f"{COCO_CLASSES[int(c)]} {s:.2f}" for c, s in zip(r[:, 5], r[:, 4]))))
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(f"outputs/demo_{os.path.splitext(os.path.basename(ip))[0]}.jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(rows, open("outputs/demo_detections.json", "w"), ensure_ascii=False, indent=1)

# ---------- 2. Швидкість на CPU ----------
img = cv2.imread(imgs[0]); speed = {}
for m, d in dets.items():
    x, _ = preproc(img, d.size)
    for _ in range(3): d.s.run(None, {d.inp: x[None]})
    t = []
    for _ in range(20):
        t0 = time.perf_counter(); d.s.run(None, {d.inp: x[None]}); t.append((time.perf_counter() - t0) * 1000)
    speed[m] = dict(ms_median=float(np.median(t)), fps=float(1000 / np.median(t)),
                    size_mb=os.path.getsize(MODELS[m][0]) / 2**20)

# ---------- 3–4. COCO128: mAP ----------
def load_gt(img_path, shape):
    lp = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(lp): return np.zeros((0, 5))
    a = np.loadtxt(lp, ndmin=2); h, w = shape[:2]
    c, x, y, bw, bh = a.T
    return np.stack([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h, c], 1)

def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)

def coco_map(all_dets, all_gts):
    """COCO-подібний mAP: 10 порогів IoU 0.5:0.95, 101-точкова інтерполяція, усереднення по класах."""
    thr = np.linspace(.5, .95, 10); aps = np.full((10, 80), np.nan)
    for c in range(80):
        n_gt = sum(int((g[:, 4] == c).sum()) for g in all_gts)
        if n_gt == 0: continue
        recs = []
        for i, (d, g) in enumerate(zip(all_dets, all_gts)):
            dc, gc = d[d[:, 5] == c], g[g[:, 4] == c]
            for s in dc[:, 4]: recs.append((s, i))
        if not recs: aps[:, c] = 0; continue
        order = sorted(range(len(recs)), key=lambda k: -recs[k][0])
        for t_i, t in enumerate(thr):
            used = {i: np.zeros(int((g[:, 4] == c).sum()), bool) for i, g in enumerate(all_gts)}
            per_img = {i: (d[d[:, 5] == c][np.argsort(-d[d[:, 5] == c][:, 4])], g[g[:, 4] == c]) for i, (d, g) in enumerate(zip(all_dets, all_gts))}
            ptr = {i: 0 for i in per_img}; tp = []
            for k in order:
                s, i = recs[k]; dc, gc = per_img[i]; j = ptr[i]; ptr[i] += 1
                if len(gc) == 0: tp.append(0); continue
                ov = iou(dc[j:j+1, :4], gc[:, :4])[0]; ov[used[i]] = -1
                b = ov.argmax()
                if ov[b] >= t: used[i][b] = True; tp.append(1)
                else: tp.append(0)
            tp = np.array(tp); ctp = np.cumsum(tp); rec = ctp / n_gt; prec = ctp / np.arange(1, len(tp) + 1)
            prec = np.maximum.accumulate(prec[::-1])[::-1]
            rs = np.linspace(0, 1, 101); q = np.searchsorted(rec, rs, side="left")
            aps[t_i, c] = np.mean([prec[x] if x < len(prec) else 0 for x in q])
    return float(np.nanmean(aps)), float(np.nanmean(aps[0])), int(np.sum(~np.isnan(aps[0])))

paths = sorted(glob.glob(os.path.join(args.coco128, "images/train2017/*.jpg")))
gts, cache = [], {}
for p in paths:
    im = cv2.imread(p); cache[p] = im; gts.append(load_gt(p, im.shape))
evals = {}
for m, d in dets.items():
    for tta in (False, True):
        t0 = time.perf_counter()
        ds = [d.detect(cache[p], score_thr=0.001, nms=0.65, tta=tta) for p in paths]
        el = time.perf_counter() - t0
        m95, m50, ncls = coco_map(ds, gts)
        evals[f"{m}{' + TTA' if tta else ''}"] = dict(mAP50_95=m95, mAP50=m50, n_classes=ncls, sec_per_img=el / len(paths))
        print(m, "TTA" if tta else "", f"mAP50-95={m95:.3f} mAP50={m50:.3f} {el/len(paths)*1000:.0f} ms/img", flush=True)

for m, d in dets.items():
    ds = [d.detect(cache[p], score_thr=0.001, nms=0.65, class_agnostic=False) for p in paths]
    m95, m50, ncls = coco_map(ds, gts)
    evals[f"{m} + class-aware NMS"] = dict(mAP50_95=m95, mAP50=m50, n_classes=ncls)
    print(m, "class-aware NMS", f"mAP50-95={m95:.3f} mAP50={m50:.3f}", flush=True)

json.dump(dict(speed=speed, coco128=evals, n_images=len(paths), n_gt=int(sum(len(g) for g in gts)),
               cpu_threads=os.cpu_count(), ort=ort.__version__), open("outputs/results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(speed, indent=1))
