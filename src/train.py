"""
PCB 瑕疵偵測 - YOLOv8 模型訓練腳本
使用 Ultralytics YOLOv8 + PyTorch 進行訓練
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO


def check_environment():
    """檢查訓練環境"""
    print("=" * 60)
    print("  PCB 瑕疵偵測 - 環境檢查")
    print("=" * 60)

    # Python 版本
    print(f"Python 版本: {sys.version}")

    # PyTorch 與 CUDA
    print(f"PyTorch 版本: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")

    if cuda_available:
        print(f"GPU 數量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  警告: 未偵測到 GPU，將使用 CPU 訓練 (速度較慢)")

    print("=" * 60)
    return cuda_available


def prepare_dataset(data_yaml: str) -> bool:
    """驗證資料集路徑與結構"""
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    base_path = Path(data_config.get('path', '.'))
    train_path = base_path / data_config.get('train', 'train/images')
    val_path = base_path / data_config.get('val', 'val/images')

    print("\n資料集結構驗證:")
    print(f"  訓練集: {train_path}")
    print(f"  驗證集: {val_path}")

    issues = []
    if not train_path.exists():
        issues.append(f"找不到訓練集目錄: {train_path}")
    else:
        train_images = list(train_path.glob("*.jpg")) + \
                       list(train_path.glob("*.png")) + \
                       list(train_path.glob("*.jpeg"))
        print(f"  訓練圖片數量: {len(train_images)}")

    if not val_path.exists():
        issues.append(f"找不到驗證集目錄: {val_path}")
    else:
        val_images = list(val_path.glob("*.jpg")) + \
                     list(val_path.glob("*.png")) + \
                     list(val_path.glob("*.jpeg"))
        print(f"  驗證圖片數量: {len(val_images)}")

    if issues:
        for issue in issues:
            print(f"  [錯誤] {issue}")
        return False

    print("  資料集結構驗證通過！")
    return True


def train(
    data_yaml: str = "configs/pcb_dataset.yaml",
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    device: str = "auto",
    pretrained: bool = True,
    resume: bool = False,
    resume_path: str = None,
    project: str = "./results",
    name: str = None,
):
    """
    執行 YOLOv8 訓練

    Args:
        data_yaml:    資料集設定檔路徑
        model_size:   模型大小 n/s/m/l/x
        epochs:       訓練回合數
        batch_size:   批次大小
        image_size:   輸入影像尺寸
        device:       訓練裝置 ('auto', 'cpu', '0', '0,1')
        pretrained:   是否使用預訓練權重
        resume:       是否從上次中斷點繼續訓練
        resume_path:  繼續訓練的權重路徑
        project:      結果儲存目錄
        name:         實驗名稱
    """
    # ── 環境檢查 ──
    cuda_available = check_environment()

    if device == "auto":
        device = "0" if cuda_available else "cpu"

    # ── 資料集驗證 ──
    if not prepare_dataset(data_yaml):
        print("\n[錯誤] 請先準備好資料集再執行訓練")
        print("請參考 README.md 了解如何下載和準備 PCB 資料集")
        sys.exit(1)

    # ── 實驗名稱 ──
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"pcb_yolov8{model_size}_{timestamp}"

    # ── 載入模型 ──
    if resume and resume_path:
        print(f"\n從斷點繼續訓練: {resume_path}")
        model = YOLO(resume_path)
    elif pretrained:
        model_name = f"yolov8{model_size}.pt"
        print(f"\n使用預訓練模型: {model_name}")
        model = YOLO(model_name)
    else:
        model_yaml = f"yolov8{model_size}.yaml"
        print(f"\n從頭訓練: {model_yaml}")
        model = YOLO(model_yaml)

    # ── 開始訓練 ──
    print(f"\n開始訓練...")
    print(f"  資料集: {data_yaml}")
    print(f"  模型: YOLOv8{model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch: {batch_size}")
    print(f"  Image Size: {image_size}")
    print(f"  Device: {device}")
    print(f"  儲存位置: {project}/{name}")
    print("-" * 60)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=project,
        name=name,
        resume=resume,
        pretrained=pretrained,
        # 優化器
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # 資料增強
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.3,
        mosaic=1.0,
        mixup=0.1,
        # 訓練技巧
        patience=50,
        save_period=10,
        val=True,
        plots=True,
        # 多任務
        workers=4,
        seed=42,
    )

    # ── 訓練完成 ──
    best_model_path = Path(project) / name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  訓練完成！")
    print(f"  最佳模型: {best_model_path}")
    print("=" * 60)

    # ── 驗證最佳模型 ──
    if best_model_path.exists():
        print("\n執行最終驗證...")
        best_model = YOLO(str(best_model_path))
        metrics = best_model.val(data=data_yaml, imgsz=image_size, device=device)
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision:{metrics.box.mp:.4f}")
        print(f"  Recall:   {metrics.box.mr:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PCB 瑕疵偵測 - YOLOv8 訓練腳本"
    )
    parser.add_argument("--data", type=str,
                        default="configs/pcb_dataset.yaml",
                        help="資料集 YAML 設定檔路徑")
    parser.add_argument("--model", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 模型大小 (n=最快, x=最準)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="訓練回合數")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="輸入影像尺寸")
    parser.add_argument("--device", type=str, default="auto",
                        help="訓練裝置 (auto/cpu/0/0,1)")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="不使用預訓練權重 (從頭訓練)")
    parser.add_argument("--resume", action="store_true",
                        help="從上次中斷點繼續訓練")
    parser.add_argument("--resume-path", type=str, default=None,
                        help="繼續訓練的權重檔案路徑")
    parser.add_argument("--project", type=str, default="./results",
                        help="結果儲存目錄")
    parser.add_argument("--name", type=str, default=None,
                        help="實驗名稱")

    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
        pretrained=not args.no_pretrain,
        resume=args.resume,
        resume_path=args.resume_path,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
