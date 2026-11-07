#!/usr/bin/env bash
# Завантажує офіційний код YOLOX, ONNX-моделі з GitHub Releases і тестові дані.
set -e
[ -d ../YOLOX ] || git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git ../YOLOX
mkdir -p models data
R=https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0
for m in yolox_nano yolox_tiny yolox_s yolox_m; do
  [ -f models/$m.onnx ] || curl -L -o models/$m.onnx $R/$m.onnx
done
A=https://github.com/ultralytics/assets/releases/download/v0.0.0
[ -f data/bus.jpg ] || curl -L -o data/bus.jpg $A/bus.jpg
if [ ! -d data/coco128 ]; then
  curl -L -o data/coco128.zip $A/coco128.zip && unzip -q data/coco128.zip -d data && rm data/coco128.zip
fi
cp data/coco128/images/train2017/000000000164.jpg data/kitchen.jpg
echo "Готово."
