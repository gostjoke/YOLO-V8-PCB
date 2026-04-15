"""
PCB 傳統電腦視覺分析模組
------------------------------------------------------
提供不依賴深度學習的輔助分析工具，可與 YOLOv8 互補使用：

1. ReferenceDiffAnalyzer  - 黃金樣本對比法（Golden Sample Comparison）
                            以良品影像為基準，透過影像對齊與差異比對找出瑕疵候選區
2. SolderPasteAnalyzer    - 錫膏連通區域分析（Connected Component Analysis）
                            以 HSV/灰階門檻分割錫膏區域，計算面積、覆蓋率、體積指標
3. BlobDefectAnalyzer     - 以形態學運算 + 輪廓分析偵測細小缺陷（毛刺/破洞）

這些方法可作為：
  • 無標注資料情況下的快速檢查
  • 驗證 YOLO 模型漏檢的第二層防線
  • 錫膏印刷體積估算（SPI 應用）
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────
# 資料類別
# ──────────────────────────────────────────────────────
@dataclass
class DefectCandidate:
    """傳統 CV 法偵測出的瑕疵候選區"""
    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2)
    area: int                             # 像素面積
    centroid: Tuple[int, int]             # 質心
    method: str = "unknown"               # 偵測方法名稱
    score: float = 0.0                    # 相似度/異常分數
    extra: Dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────
# 1. 黃金樣本對比法
# ──────────────────────────────────────────────────────
class ReferenceDiffAnalyzer:
    """以良品（Golden Sample）為基準比對測試影像的差異區域。

    流程:
        1. 特徵點匹配（ORB / SIFT）對齊兩張影像
        2. 計算絕對差值 → 二值化 → 形態學去噪
        3. 連通區域分析 → 輸出瑕疵候選 bbox
    """

    def __init__(
        self,
        min_area: int = 30,
        diff_threshold: int = 40,
        use_sift: bool = False,
    ) -> None:
        self.min_area = min_area
        self.diff_threshold = diff_threshold
        self.detector = cv2.SIFT_create() if use_sift else cv2.ORB_create(5000)
        self.matcher = cv2.BFMatcher(
            cv2.NORM_L2 if use_sift else cv2.NORM_HAMMING,
            crossCheck=True,
        )

    def align(
        self, reference: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """將 target 對齊到 reference 的座標系"""
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.detector.detectAndCompute(ref_gray, None)
        kp2, des2 = self.detector.detectAndCompute(tgt_gray, None)
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            # 特徵點不足 → 回退到 resize 對齊
            aligned = cv2.resize(target, (reference.shape[1], reference.shape[0]))
            return aligned, None

        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        if len(matches) < 10:
            aligned = cv2.resize(target, (reference.shape[1], reference.shape[0]))
            return aligned, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            aligned = cv2.resize(target, (reference.shape[1], reference.shape[0]))
            return aligned, None

        h, w = reference.shape[:2]
        aligned = cv2.warpPerspective(target, H, (w, h))
        return aligned, H

    def analyze(
        self, reference: np.ndarray, target: np.ndarray
    ) -> Tuple[List[DefectCandidate], np.ndarray, np.ndarray]:
        """比對差異並回傳候選瑕疵、差異熱圖、標註影像"""
        aligned, _ = self.align(reference, target)

        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        # 高斯平滑以抑制雜訊
        ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        tgt_gray = cv2.GaussianBlur(tgt_gray, (5, 5), 0)

        diff = cv2.absdiff(ref_gray, tgt_gray)
        _, mask = cv2.threshold(
            diff, self.diff_threshold, 255, cv2.THRESH_BINARY
        )
        # 形態學：開運算去雜點 + 閉運算連通瑕疵
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 連通區域分析
        num, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates: List[DefectCandidate] = []
        annotated = aligned.copy()

        for i in range(1, num):  # 跳過背景 (label 0)
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            cx, cy = centroids[i]
            score = float(diff[y:y+h, x:x+w].mean())
            candidates.append(DefectCandidate(
                bbox=(x, y, x + w, y + h),
                area=int(area),
                centroid=(int(cx), int(cy)),
                method="reference_diff",
                score=round(score, 2),
            ))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, f"diff:{score:.0f}",
                        (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 差異熱圖 (便於視覺化)
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        return candidates, heatmap, annotated


# ──────────────────────────────────────────────────────
# 2. 錫膏連通區域分析
# ──────────────────────────────────────────────────────
class SolderPasteAnalyzer:
    """錫膏印刷檢測 (SPI) 輔助分析。

    依色彩門檻（HSV）或灰階門檻分割錫膏區域，計算：
        • 錫膏塊數量
        • 平均面積
        • 覆蓋率 (coverage)
        • 推估體積 (以灰階亮度估算相對厚度)
    """

    def __init__(
        self,
        hsv_lower: Tuple[int, int, int] = (0, 0, 150),
        hsv_upper: Tuple[int, int, int] = (180, 80, 255),
        min_area: int = 15,
    ) -> None:
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area

    def segment(self, image: np.ndarray) -> np.ndarray:
        """分割錫膏亮區並回傳二值遮罩"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def analyze(self, image: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """回傳統計字典與標註影像"""
        mask = self.segment(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        annotated = image.copy()
        pads: List[Dict] = []
        total_area = 0
        total_brightness = 0.0

        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            roi = gray[y:y+h, x:x+w]
            brightness = float(roi.mean())
            # 相對體積 = 面積 × 平均亮度 (作為厚度 proxy)
            volume_proxy = area * brightness
            total_area += area
            total_brightness += brightness
            pads.append({
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
                "brightness": round(brightness, 2),
                "volume_proxy": round(volume_proxy, 2),
            })
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)

        img_area = image.shape[0] * image.shape[1]
        stats_out = {
            "pad_count": len(pads),
            "total_area": int(total_area),
            "coverage": round(total_area / img_area, 4),
            "avg_brightness": round(
                total_brightness / max(len(pads), 1), 2
            ),
            "pads": pads,
        }
        cv2.putText(
            annotated,
            f"Pads: {len(pads)}  Coverage: {stats_out['coverage']*100:.2f}%",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        return stats_out, annotated


# ──────────────────────────────────────────────────────
# 3. 形態學 + 輪廓分析偵測器
# ──────────────────────────────────────────────────────
class BlobDefectAnalyzer:
    """以自適應門檻 + 形態學找出異常小區塊 (毛刺、破洞、異物)"""

    def __init__(self, min_area: int = 10, max_area: int = 5000) -> None:
        self.min_area = min_area
        self.max_area = max_area

    def analyze(
        self, image: np.ndarray
    ) -> Tuple[List[DefectCandidate], np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25, C=7,
        )
        # 取出非常小的異常點
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: List[DefectCandidate] = []
        annotated = image.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2
            # 用周長/面積比當作異常分數 (毛刺通常周長/面積較大)
            perimeter = cv2.arcLength(cnt, True)
            score = perimeter / (area + 1e-5)
            candidates.append(DefectCandidate(
                bbox=(x, y, x + w, y + h),
                area=int(area),
                centroid=(cx, cy),
                method="blob",
                score=round(float(score), 3),
            ))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 1)
        return candidates, annotated


# ──────────────────────────────────────────────────────
# CLI 測試
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="傳統 CV PCB 分析器測試")
    parser.add_argument("--mode", choices=["diff", "solder", "blob"],
                        default="blob")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--reference", type=str, default=None,
                        help="(diff 模式) 黃金樣本影像路徑")
    parser.add_argument("--output", type=str, default="./analyzed.jpg")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"無法讀取 {args.image}")

    if args.mode == "diff":
        assert args.reference, "--reference 必填"
        ref = cv2.imread(args.reference)
        analyzer = ReferenceDiffAnalyzer()
        cands, heat, annotated = analyzer.analyze(ref, img)
        print(f"偵測到 {len(cands)} 個差異區域")
    elif args.mode == "solder":
        analyzer = SolderPasteAnalyzer()
        stats, annotated = analyzer.analyze(img)
        print(f"錫膏塊數: {stats['pad_count']}, 覆蓋率: {stats['coverage']:.2%}")
    else:
        analyzer = BlobDefectAnalyzer()
        cands, annotated = analyzer.analyze(img)
        print(f"偵測到 {len(cands)} 個異常區塊")

    cv2.imwrite(args.output, annotated)
    print(f"結果已儲存: {args.output}")
