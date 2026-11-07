"""
Лабораторна робота №6, ФБ-31 Голомовза Дар'я.

Запуск ОФІЦІЙНОГО демо YOLOX (demo/ONNXRuntime/onnx_inference.py) без встановлення PyTorch.

Офіційний скрипт імпортує кілька допоміжних функцій із пакета `yolox`
(preproc, COCO_CLASSES, multiclass_nms, demo_postprocess, vis, mkdir).
Ці функції написані на чистому NumPy/OpenCV, але `import yolox` тягне за собою torch.
Тому тут ми завантажуємо лише потрібні файли репозиторію напряму і реєструємо їх
як модулі `yolox.*`, а потім виконуємо офіційний скрипт без жодних змін.

Використання:
    python run_official_demo.py --yolox-dir ../YOLOX -- -m models/yolox_s.onnx \
        -i ../YOLOX/assets/dog.jpg -o outputs -s 0.3 --input_shape 640,640
"""
import argparse, importlib.util, os, runpy, sys, types


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_torch_free_yolox(root):
    pkg = lambda n: sys.modules.setdefault(n, types.ModuleType(n))
    yolox, utils, data, datasets = pkg("yolox"), pkg("yolox.utils"), pkg("yolox.data"), pkg("yolox.data.datasets")
    yolox.utils, yolox.data, data.datasets = utils, data, datasets

    demo_utils = load_file("yolox.utils.demo_utils", os.path.join(root, "yolox/utils/demo_utils.py"))
    visualize = load_file("yolox.utils.visualize", os.path.join(root, "yolox/utils/visualize.py"))
    for f in ("mkdir", "multiclass_nms", "demo_postprocess"):
        setattr(utils, f, getattr(demo_utils, f))
    utils.vis = visualize.vis
    # data_augment.py імпортує xyxy2cxcywh з yolox.utils (використовується лише в навчанні)
    utils.xyxy2cxcywh = lambda b: b
    load_file("yolox.data.data_augment", os.path.join(root, "yolox/data/data_augment.py"))
    classes = load_file("yolox.data.datasets.coco_classes", os.path.join(root, "yolox/data/datasets/coco_classes.py"))
    datasets.COCO_CLASSES = classes.COCO_CLASSES


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolox-dir", default="../YOLOX", help="шлях до клонованого репозиторію YOLOX")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    root = os.path.abspath(args.yolox_dir)
    install_torch_free_yolox(root)
    script = os.path.join(root, "demo/ONNXRuntime/onnx_inference.py")
    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")
