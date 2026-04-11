"""
PCB 資料集準備腳本
支援下載 DeepPCB 資料集並轉換為 YOLOv8 格式
"""

import os
import sys
import shutil
import random
import argparse
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


# DeepPCB 類別對應
DEEPPCB_CLASSES = {
    "missing_hole":    0,
    "mouse_bite":      1,
    "open_circuit":    2,
    "short":           3,
    "spur":            4,
    "spurious_copper": 5,
}


def voc_to_yolo(
    xml_path: str,
    image_w: int,
    image_h: int,
    class_map: Dict[str, int]
) -> List[str]:
    """
    將 Pascal VOC XML 格式標注轉換為 YOLO 格式

    Args:
        xml_path:  XML 標注檔案路徑
        image_w:   影像寬度
        image_h:   影像高度
        class_map: 類別名稱 → 類別 ID 映射

    Returns:
        YOLO 格式的標注行列表
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None:
            continue
        class_name = name_elem.text.strip().lower().replace(" ", "_")

        if class_name not in class_map:
            print(f"  [警告] 未知類別: {class_name}，跳過")
            continue

        class_id = class_map[class_name]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # 轉換為 YOLO 格式 (中心點 + 寬高，正規化)
        x_center = (xmin + xmax) / 2 / image_w
        y_center = (ymin + ymax) / 2 / image_h
        width    = (xmax - xmin) / image_w
        height   = (ymax - ymin) / image_h

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    return yolo_lines


def split_dataset(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    將資料集分割為 train/val/test

    Args:
        image_dir:   原始影像目錄
        label_dir:   原始標注目錄 (YOLO 格式 .txt)
        output_dir:  輸出目錄
        train_ratio: 訓練集比例
        val_ratio:   驗證集比例
        test_ratio:  測試集比例
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "三個比例之和必須為 1"

    image_path = Path(image_dir)
    label_path = Path(label_dir)
    output_path = Path(output_dir)

    # 建立輸出目錄
    for split in ["train", "val", "test"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # 收集影像
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = []
    for ext in extensions:
        images.extend(image_path.glob(f"*{ext}"))
        images.extend(image_path.glob(f"*{ext.upper()}"))

    # 只保留有對應標注的影像
    valid_images = []
    for img in images:
        label_file = label_path / f"{img.stem}.txt"
        if label_file.exists():
            valid_images.append(img)

    print(f"找到 {len(valid_images)} 張有效影像（含標注）")

    # 隨機分割
    random.seed(seed)
    random.shuffle(valid_images)

    n = len(valid_images)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": valid_images[:n_train],
        "val":   valid_images[n_train:n_train + n_val],
        "test":  valid_images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        for img_file in split_images:
            # 複製影像
            dst_img = output_path / split_name / "images" / img_file.name
            shutil.copy2(str(img_file), str(dst_img))

            # 複製標注
            src_label = label_path / f"{img_file.stem}.txt"
            dst_label = output_path / split_name / "labels" / f"{img_file.stem}.txt"
            shutil.copy2(str(src_label), str(dst_label))

        print(f"  {split_name}: {len(split_images)} 張")

    print(f"\n資料集分割完成，儲存於: {output_path}")
    return splits


def convert_voc_to_yolo_batch(
    image_dir: str,
    annotation_dir: str,
    output_image_dir: str,
    output_label_dir: str,
    class_map: Dict[str, int] = DEEPPCB_CLASSES,
    default_size: Tuple[int, int] = (600, 600),
):
    """批次將 VOC XML 標注轉換為 YOLO 格式"""
    image_path = Path(image_dir)
    annot_path = Path(annotation_dir)
    out_img    = Path(output_image_dir)
    out_lbl    = Path(output_label_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    xml_files = list(annot_path.glob("*.xml"))
    print(f"找到 {len(xml_files)} 個 XML 標注檔，開始轉換...")

    converted = 0
    for xml_file in xml_files:
        # 找對應影像
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = image_path / f"{xml_file.stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            print(f"  [跳過] 找不到對應影像: {xml_file.stem}")
            continue

        try:
            import cv2
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
            else:
                w, h = default_size
        except Exception:
            w, h = default_size

        # 轉換標注
        yolo_lines = voc_to_yolo(str(xml_file), w, h, class_map)

        if not yolo_lines:
            continue

        # 儲存 YOLO 標注
        label_file = out_lbl / f"{xml_file.stem}.txt"
        with open(label_file, 'w') as f:
            f.write("\n".join(yolo_lines))

        # 複製影像
        shutil.copy2(str(img_file), str(out_img / img_file.name))
        converted += 1

    print(f"轉換完成: {converted}/{len(xml_files)} 個檔案")


def create_dataset_yaml(
    data_dir: str,
    num_classes: int,
    class_names: List[str],
    output_path: str = "configs/pcb_dataset.yaml"
):
    """生成資料集 YAML 設定檔"""
    yaml_content = f"""# PCB 瑕疵偵測資料集設定
path: {os.path.abspath(data_dir)}

train: train/images
val: val/images
test: test/images

nc: {num_classes}

names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"資料集 YAML 設定已生成: {output_path}")


def show_dataset_instructions():
    """顯示資料集準備說明"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          PCB 瑕疵偵測 - 資料集準備說明                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  【方法一】使用 DeepPCB 公開資料集 (推薦)                       ║
║  ─────────────────────────────────────                       ║
║  1. 前往 GitHub 下載:                                         ║
║     https://github.com/tangsanli5201/DeepPCB                 ║
║                                                              ║
║  2. 解壓後目錄結構:                                            ║
║     DeepPCB/                                                 ║
║     ├── images/          (原始影像)                           ║
║     └── annotations/     (XML 標注)                          ║
║                                                              ║
║  3. 執行轉換:                                                 ║
║     python src/prepare_dataset.py --mode convert             ║
║       --image-dir DeepPCB/images                             ║
║       --annot-dir DeepPCB/annotations                        ║
║                                                              ║
║  【方法二】使用 Roboflow 資料集                                  ║
║  ─────────────────────────────────────                       ║
║  1. 前往 https://universe.roboflow.com                        ║
║  2. 搜尋 "PCB defect"                                        ║
║  3. 選擇 YOLOv8 格式下載                                      ║
║  4. 直接放入 data/ 目錄使用                                    ║
║                                                              ║
║  【方法三】自行標注                                             ║
║  ─────────────────────────────────────                       ║
║  1. 安裝標注工具: pip install labelImg                         ║
║  2. 執行: labelImg                                           ║
║  3. 設定輸出格式為 YOLO                                        ║
║  4. 標注完成後執行資料集分割:                                    ║
║     python src/prepare_dataset.py --mode split               ║
║       --image-dir data/raw/images                            ║
║       --label-dir data/raw/labels                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="PCB 資料集準備工具")
    parser.add_argument("--mode", type=str,
                        choices=["info", "convert", "split", "all"],
                        default="info",
                        help="操作模式: info/convert/split/all")
    parser.add_argument("--image-dir", type=str, default="",
                        help="影像目錄")
    parser.add_argument("--annot-dir", type=str, default="",
                        help="XML 標注目錄")
    parser.add_argument("--label-dir", type=str, default="",
                        help="YOLO 標注目錄")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="輸出目錄")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    args = parser.parse_args()

    if args.mode == "info":
        show_dataset_instructions()

    elif args.mode == "convert":
        if not args.image_dir or not args.annot_dir:
            print("請提供 --image-dir 和 --annot-dir 參數")
            sys.exit(1)
        out_images = os.path.join(args.output_dir, "raw", "images")
        out_labels = os.path.join(args.output_dir, "raw", "labels")
        convert_voc_to_yolo_batch(
            args.image_dir, args.annot_dir,
            out_images, out_labels,
        )
        print("\n下一步：執行 split 分割資料集")
        print(f"  python src/prepare_dataset.py --mode split "
              f"--image-dir {out_images} --label-dir {out_labels}")

    elif args.mode == "split":
        if not args.image_dir or not args.label_dir:
            print("請提供 --image-dir 和 --label-dir 參數")
            sys.exit(1)
        split_dataset(
            args.image_dir, args.label_dir,
            args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        create_dataset_yaml(
            args.output_dir,
            num_classes=len(DEEPPCB_CLASSES),
            class_names=list(DEEPPCB_CLASSES.keys()),
        )
        print("\n下一步：開始訓練")
        print("  python src/train.py --model n --epochs 100")

    elif args.mode == "all":
        show_dataset_instructions()


if __name__ == "__main__":
    main()
