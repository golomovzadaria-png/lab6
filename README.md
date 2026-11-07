# Лабораторна робота №6. Робота з відкритим програмним кодом

**Виконала:** студентка групи ФБ-31 Голомовза Дар'я

**Стаття:** Z. Ge, S. Liu, F. Wang, Z. Li, J. Sun. *YOLOX: Exceeding YOLO Series in 2021.* arXiv:2107.08430, 2021 — https://arxiv.org/abs/2107.08430

**Оригінальний код:** https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0)

## Про статтю

YOLOX — anchor-free детектор об'єктів сімейства YOLO від Megvii. Автори беруть YOLOv3 (Darknet53) і послідовно додають:

| Крок | AP на COCO val, % |
| --- | --- |
| YOLOv3 baseline | 38.5 |
| + decoupled head (окремі гілки класифікації та регресії) | 39.6 |
| + сильні аугментації (Mosaic, MixUp) | 42.0 |
| + anchor-free (1 прогноз на позицію замість 3) | 42.9 |
| + multi positives (центральна область 3×3) | 45.0 |
| + SimOTA (динамічне призначення позитивних прикладів) | **47.3** |

YOLOX-L досягає 50.0 % AP при 68.9 FPS на Tesla V100 (+1.8 AP до YOLOv5-L), а YOLOX-Nano з 0.91 млн параметрів — 25.3 % AP (+1.8 AP до NanoDet). Окремою моделлю YOLOX-L команда посіла 1-ше місце в Streaming Perception Challenge (CVPR 2021).

## Що в репозиторії

| Файл | Що робить |
| --- | --- |
| `run_official_demo.py` | запускає **офіційне** демо `demo/ONNXRuntime/onnx_inference.py` без змін і без встановлення PyTorch |
| `experiments.py` | порівняння моделей Nano/Tiny/S/M на фото, швидкість на CPU, mAP на COCO128, спроби покращення (TTA, class-aware NMS) |
| `download.sh` | завантажує репозиторій YOLOX, ONNX-моделі та тестові дані |
| `outputs/` | результати: зображення з детекціями, `results.json`, `demo_detections.json`, `log.txt` |

## Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./download.sh

# 1. Офіційне демо YOLOX-S на assets/dog.jpg
python run_official_demo.py --yolox-dir ../YOLOX -- -m models/yolox_s.onnx \
    -i ../YOLOX/assets/dog.jpg -o outputs/official -s 0.3 --input_shape 640,640

# 2. Усі експерименти (~4 хв на 2 ядрах CPU)
python experiments.py --yolox-dir ../YOLOX
```

## Результати

**Офіційне демо (YOLOX-S):** bicycle 95.5 %, dog 91.3 %, truck 61.2 %, potted plant 43.9 % — `outputs/official/dog.jpg`.

**COCO128** (128 зображень, 929 об'єктів, 71 клас), ONNX Runtime 1.29, CPU 2 потоки:

| Модель | Вхід | mAP@0.5:0.95 | mAP@0.5 | CPU, мс | Розмір, МБ |
| --- | --- | --- | --- | --- | --- |
| YOLOX-Nano | 416 | 0.322 | 0.489 | 10.3 | 3.5 |
| YOLOX-Tiny | 416 | 0.430 | 0.612 | 39.8 | 19.3 |
| YOLOX-S | 640 | 0.508 | 0.714 | 120.4 | 34.2 |
| YOLOX-M | 640 | 0.570 | 0.749 | 337.0 | 96.6 |

**Спроби покращення** (mAP@0.5:0.95 на COCO128):

| Модель | Офіційне демо | + TTA (flip) | + class-aware NMS |
| --- | --- | --- | --- |
| YOLOX-Nano | 0.322 | 0.336 (+0.014) | 0.334 (+0.012) |
| YOLOX-Tiny | 0.430 | 0.434 (+0.005) | 0.441 (+0.012) |
| YOLOX-S | 0.508 | 0.513 (+0.004) | 0.525 (+0.016) |
| YOLOX-M | 0.570 | 0.587 (+0.016) | 0.592 (+0.022) |

TTA подвоює час інференсу; class-aware NMS (NMS окремо для кожного класу замість class-agnostic в офіційному демо) не змінює швидкість моделі.

> COCO128 — це 128 зображень з COCO **train2017**, на якому моделі навчалися, тому абсолютні значення mAP завищені порівняно з COCO val. Вони придатні для порівняння моделей і методів між собою, а не з таблицями статті.
