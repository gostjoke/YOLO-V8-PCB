"""
PCB 影像前處理模組
使用 OpenCV 進行影像增強、去噪、對比度調整等操作
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PCBPreprocessor:
    """PCB 影像前處理器"""

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            target_size: 目標影像尺寸 (width, height)
        """
        self.target_size = target_size

    def load_image(self, image_path: str) -> np.ndarray:
        """載入影像"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"無法讀取影像: {image_path}")
        return img

    def resize(self, image: np.ndarray) -> np.ndarray:
        """調整影像尺寸"""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

    def denoise(self, image: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """
        影像去噪
        Args:
            method: 去噪方法 - 'gaussian', 'bilateral', 'nlm'
        """
        if method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "nlm":
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return image

    def enhance_contrast(self, image: np.ndarray, method: str = "clahe") -> np.ndarray:
        """
        對比度增強
        Args:
            method: 增強方法 - 'clahe', 'histogram', 'gamma'
        """
        if method == "clahe":
            # 轉換到 LAB 色彩空間
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        elif method == "histogram":
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y = cv2.equalizeHist(y)
            enhanced = cv2.merge([y, cr, cb])
            return cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)

        elif method == "gamma":
            gamma = 1.5
            lookup_table = np.array([
                ((i / 255.0) ** (1.0 / gamma)) * 255
                for i in range(256)
            ]).astype("uint8")
            return cv2.LUT(image, lookup_table)

        return image

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """影像銳化 - 增強 PCB 邊緣細節"""
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        return cv2.filter2D(image, -1, kernel)

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Canny 邊緣偵測 (用於視覺化)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """正規化到 [0, 1]"""
        return image.astype(np.float32) / 255.0

    def full_pipeline(self, image_path: str, save_path: Optional[str] = None) -> np.ndarray:
        """
        完整前處理流水線
        1. 載入 → 2. 去噪 → 3. 對比度增強 → 4. 銳化 → 5. 調整尺寸
        """
        img = self.load_image(image_path)
        img = self.denoise(img, method="bilateral")
        img = self.enhance_contrast(img, method="clahe")
        img = self.sharpen(img)
        img = self.resize(img)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)

        return img

    def batch_process(self, input_dir: str, output_dir: str,
                      extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]):
        """批次處理資料夾中的所有影像"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        print(f"找到 {len(image_files)} 張影像，開始批次處理...")

        for i, img_file in enumerate(image_files):
            try:
                save_path = output_path / img_file.name
                self.full_pipeline(str(img_file), str(save_path))
                print(f"[{i+1}/{len(image_files)}] 處理完成: {img_file.name}")
            except Exception as e:
                print(f"[錯誤] 處理 {img_file.name} 時發生錯誤: {e}")

        print("批次處理完成！")


class AugmentationPipeline:
    """資料增強流水線 - 用於訓練資料擴充"""

    def __init__(self, image_size: int = 640):
        self.image_size = image_size

        # 訓練用增強
        self.train_transform = A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size,
                                scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))

        # 驗證/推論用 (只做 Resize)
        self.val_transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))

    def apply_train(self, image: np.ndarray, bboxes: list, labels: list):
        """應用訓練增強"""
        result = self.train_transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels
        )
        return result['image'], result['bboxes'], result['class_labels']

    def apply_val(self, image: np.ndarray, bboxes: list, labels: list):
        """應用驗證增強"""
        result = self.val_transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels
        )
        return result['image'], result['bboxes'], result['class_labels']


def visualize_preprocessing(image_path: str, save_path: Optional[str] = None):
    """
    視覺化前處理各階段的效果
    產生比較圖: 原始 vs 去噪 vs 增強對比 vs 銳化
    """
    preprocessor = PCBPreprocessor()
    original = preprocessor.load_image(image_path)

    denoised = preprocessor.denoise(original, method="bilateral")
    enhanced = preprocessor.enhance_contrast(denoised, method="clahe")
    sharpened = preprocessor.sharpen(enhanced)
    edges = preprocessor.detect_edges(enhanced)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 縮放所有影像到相同大小
    h, w = 320, 320
    imgs = [
        cv2.resize(original, (w, h)),
        cv2.resize(denoised, (w, h)),
        cv2.resize(enhanced, (w, h)),
        cv2.resize(sharpened, (w, h)),
    ]

    labels = ["原始影像", "雙邊去噪", "CLAHE 增強", "銳化"]
    labeled_imgs = []
    for img, label in zip(imgs, labels):
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        labeled_imgs.append(img)

    top_row = np.hstack(labeled_imgs[:2])
    bottom_row = np.hstack(labeled_imgs[2:])
    comparison = np.vstack([top_row, bottom_row])

    if save_path:
        cv2.imwrite(save_path, comparison)
        print(f"比較圖已儲存: {save_path}")

    return comparison


if __name__ == "__main__":
    # 測試範例
    preprocessor = PCBPreprocessor(target_size=(640, 640))
    print("PCBPreprocessor 初始化成功")
    print(f"目標尺寸: {preprocessor.target_size}")

    aug = AugmentationPipeline(image_size=640)
    print("AugmentationPipeline 初始化成功")
