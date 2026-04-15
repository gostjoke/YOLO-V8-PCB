"""
PCB 瑕疵偵測 - 推論模組
支援單張影像、批次影像、資料夾批次偵測
"""

import cv2
import csv
import numpy as np
import os
import json
import time
import argparse
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Generator
from datetime import datetime

import torch
from ultralytics import YOLO


# 瑕疵類別顏色對應 (BGR 格式)
DEFECT_COLORS = {
    "missing_hole":    (0,   0,   255),  # 紅
    "mouse_bite":      (0,   165, 255),  # 橘
    "open_circuit":    (0,   255, 255),  # 黃
    "short":           (0,   255, 0),    # 綠
    "spur":            (255, 0,   0),    # 藍
    "spurious_copper": (255, 0,   255),  # 紫
    # 錫膏類型
    "insufficient_solder": (0, 0, 255),
    "excess_solder":       (0, 165, 255),
    "solder_bridge":       (0, 255, 0),
    "missing_solder":      (255, 0, 0),
    "cold_solder":         (255, 255, 0),
    "offset_solder":       (255, 0, 255),
}

DEFAULT_COLOR = (128, 128, 128)


class PCBDefectDetector:
    """PCB 瑕疵偵測器"""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        image_size: int = 640,
    ):
        """
        Args:
            model_path:    訓練好的 .pt 模型路徑
            confidence:    信心度門檻 (0~1)
            iou_threshold: NMS IoU 門檻
            device:        推論裝置 ('auto', 'cpu', '0')
            image_size:    推論影像尺寸
        """
        if device == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.image_size = image_size

        print(f"載入模型: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"偵測類別: {list(self.class_names.values())}")
        print(f"推論裝置: {device}")

    def predict_image(
        self,
        image: Union[str, np.ndarray],
        return_annotated: bool = True,
    ) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        對單張影像進行瑕疵偵測

        Args:
            image:            影像路徑或 numpy array
            return_annotated: 是否返回標註後的影像

        Returns:
            detections: 偵測結果列表
            annotated:  標註後影像 (BGR numpy array) 或 None
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"無法讀取影像: {image}")
        else:
            img = image.copy()

        # YOLOv8 推論
        results = self.model.predict(
            source=img,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections = []
        annotated = img.copy() if return_annotated else None

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # 取得邊界框
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                detection = {
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox":       [x1, y1, x2, y2],
                    "center":     [(x1 + x2) // 2, (y1 + y2) // 2],
                    "area":       (x2 - x1) * (y2 - y1),
                }
                detections.append(detection)

                # 繪製標註
                if return_annotated:
                    color = DEFECT_COLORS.get(cls_name, DEFAULT_COLOR)
                    annotated = self._draw_detection(
                        annotated, x1, y1, x2, y2,
                        cls_name, conf, color
                    )

        # 繪製統計資訊
        if return_annotated and detections:
            annotated = self._draw_summary(annotated, detections)

        return detections, annotated

    def _draw_detection(
        self,
        image: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        class_name: str, confidence: float,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """繪製單個偵測框"""
        # 邊框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 標籤背景
        label = f"{class_name}: {confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y = max(y1 - 10, text_h + 5)
        cv2.rectangle(
            image,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y + baseline),
            color, -1
        )

        # 標籤文字
        cv2.putText(
            image, label,
            (x1, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2
        )
        return image

    def _draw_summary(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """在影像左上角繪製統計摘要"""
        # 統計各類別數量
        class_counts = {}
        for det in detections:
            name = det["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

        # 背景矩形
        margin = 10
        line_h = 28
        total_lines = len(class_counts) + 1
        rect_h = total_lines * line_h + margin * 2
        rect_w = 280

        overlay = image.copy()
        cv2.rectangle(overlay, (margin, margin),
                      (margin + rect_w, margin + rect_h),
                      (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        # 標題
        cv2.putText(image, f"瑕疵數量: {len(detections)}",
                    (margin * 2, margin * 2 + line_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 各類別
        for i, (name, count) in enumerate(class_counts.items()):
            color = DEFECT_COLORS.get(name, DEFAULT_COLOR)
            text = f"  {name}: {count}"
            y = margin * 2 + (i + 2) * line_h - 5
            cv2.putText(image, text,
                        (margin * 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return image

    def predict_folder(
        self,
        input_dir: str,
        output_dir: str,
        save_json: bool = True,
        extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]
    ) -> List[Dict]:
        """
        批次偵測資料夾中的所有影像

        Returns:
            所有影像的偵測結果列表
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 收集所有影像
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        print(f"\n找到 {len(image_files)} 張影像，開始批次偵測...")
        all_results = []

        for i, img_file in enumerate(image_files):
            try:
                detections, annotated = self.predict_image(str(img_file))

                # 儲存標註影像
                if annotated is not None:
                    save_path = output_path / f"detected_{img_file.name}"
                    cv2.imwrite(str(save_path), annotated)

                result = {
                    "image": img_file.name,
                    "defect_count": len(detections),
                    "detections": detections,
                    "status": "PASS" if len(detections) == 0 else "FAIL"
                }
                all_results.append(result)

                status_icon = "✓" if result["status"] == "PASS" else "✗"
                print(f"[{i+1}/{len(image_files)}] {status_icon} {img_file.name}"
                      f" - 瑕疵: {len(detections)}")

            except Exception as e:
                print(f"[錯誤] {img_file.name}: {e}")

        # 儲存 JSON 報告
        if save_json:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = output_path / f"detection_report_{timestamp}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n偵測報告已儲存: {report_path}")

        # 統計摘要
        total = len(all_results)
        passed = sum(1 for r in all_results if r["status"] == "PASS")
        failed = total - passed
        print(f"\n{'='*40}")
        print(f"  批次偵測完成")
        print(f"  總計: {total}  通過: {passed}  失敗: {failed}")
        print(f"  良品率: {passed/total*100:.1f}%" if total > 0 else "")
        print(f"{'='*40}")

        return all_results

    # ────────────────────────────────────────────────
    # 影片 / 攝影機串流推論
    # ────────────────────────────────────────────────
    def predict_video(
        self,
        source: Union[str, int],
        output_path: Optional[str] = None,
        show: bool = False,
        frame_skip: int = 1,
    ) -> Dict:
        """對影片檔或攝影機 (0) 進行即時偵測。

        Args:
            source:      影片檔路徑或 int (webcam index)
            output_path: 輸出 .mp4 路徑（None 則不存檔）
            show:        是否以視窗播放（無頭環境應設為 False）
            frame_skip:  每幾幀推論一次（提升速度）

        Returns:
            統計字典（總幀數、總瑕疵數、每類別累計、平均 FPS）
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片來源: {source}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

        total_frames = 0
        total_defects = 0
        class_totals: Dict[str, int] = {}
        last_annotated = None
        t_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                total_frames += 1
                if (total_frames - 1) % frame_skip != 0 and last_annotated is not None:
                    annotated = last_annotated
                else:
                    detections, annotated = self.predict_image(frame)
                    last_annotated = annotated
                    total_defects += len(detections)
                    for d in detections:
                        class_totals[d["class_name"]] = \
                            class_totals.get(d["class_name"], 0) + 1

                if writer is not None:
                    writer.write(annotated)
                if show:
                    cv2.imshow("PCB Defect - Video", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        elapsed = max(time.time() - t_start, 1e-6)
        return {
            "total_frames": total_frames,
            "total_defects": total_defects,
            "class_totals": class_totals,
            "avg_fps": round(total_frames / elapsed, 2),
            "output_path": output_path,
        }

    # ────────────────────────────────────────────────
    # 報告輸出 (CSV)
    # ────────────────────────────────────────────────
    @staticmethod
    def export_csv_report(results: List[Dict], csv_path: str) -> str:
        """將 predict_folder 的結果輸出為 CSV 報表 (每列 = 一個瑕疵)"""
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image", "status", "defect_index", "class_name",
                "confidence", "x1", "y1", "x2", "y2", "area",
            ])
            for r in results:
                if not r["detections"]:
                    writer.writerow([r["image"], r["status"], "", "", "",
                                     "", "", "", "", ""])
                    continue
                for i, d in enumerate(r["detections"], 1):
                    x1, y1, x2, y2 = d["bbox"]
                    writer.writerow([
                        r["image"], r["status"], i, d["class_name"],
                        d["confidence"], x1, y1, x2, y2, d["area"],
                    ])
        print(f"CSV 報告已儲存: {csv_path}")
        return csv_path

    # ────────────────────────────────────────────────
    # 模型匯出 (ONNX / TorchScript / OpenVINO)
    # ────────────────────────────────────────────────
    def export_model(
        self,
        format: str = "onnx",
        image_size: int = 640,
        half: bool = False,
        dynamic: bool = True,
        simplify: bool = True,
    ) -> str:
        """匯出為部署格式。支援 onnx / torchscript / openvino / engine (TensorRT) 等。"""
        print(f"\n匯出模型為 {format.upper()} 格式...")
        export_path = self.model.export(
            format=format,
            imgsz=image_size,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
        )
        print(f"匯出完成: {export_path}")
        return str(export_path)

    def evaluate_model(
        self,
        data_yaml: str,
        image_size: int = 640,
        batch_size: int = 32,
    ) -> Dict:
        """評估模型效能指標"""
        print("\n執行模型評估...")
        metrics = self.model.val(
            data=data_yaml,
            imgsz=image_size,
            batch=batch_size,
            device=self.device,
            verbose=True,
        )
        results = {
            "mAP50":     round(float(metrics.box.map50), 4),
            "mAP50-95":  round(float(metrics.box.map), 4),
            "Precision": round(float(metrics.box.mp), 4),
            "Recall":    round(float(metrics.box.mr), 4),
        }
        print("\n模型評估結果:")
        for k, v in results.items():
            print(f"  {k}: {v}")
        return results


def main():
    parser = argparse.ArgumentParser(description="PCB 瑕疵偵測 - 推論腳本")
    parser.add_argument("--model", type=str, required=True,
                        help="模型權重路徑 (.pt)")
    parser.add_argument("--source", type=str, default=None,
                        help="輸入影像路徑或資料夾 (若用 --video/--export 可省略)")
    parser.add_argument("--output", type=str, default="./results/inference",
                        help="輸出資料夾")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="信心度門檻")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 門檻")
    parser.add_argument("--device", type=str, default="auto",
                        help="推論裝置")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="影像尺寸")
    parser.add_argument("--no-save-json", action="store_true",
                        help="不儲存 JSON 報告")
    parser.add_argument("--save-csv", action="store_true",
                        help="批次偵測時同時輸出 CSV 報告")
    parser.add_argument("--video", type=str, default=None,
                        help="影片檔路徑 (或數字 0 開啟 webcam)")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="影片推論每 N 幀跑一次，提升 FPS")
    parser.add_argument("--export", type=str, default=None,
                        choices=["onnx", "torchscript", "openvino",
                                 "engine", "coreml"],
                        help="匯出模型部署格式")

    args = parser.parse_args()

    detector = PCBDefectDetector(
        model_path=args.model,
        confidence=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        image_size=args.imgsz,
    )

    # ── 模型匯出 ──
    if args.export:
        detector.export_model(format=args.export, image_size=args.imgsz)
        return

    # ── 影片 / 攝影機 ──
    if args.video is not None:
        src: Union[str, int] = (
            int(args.video) if args.video.isdigit() else args.video
        )
        out_vid = str(Path(args.output) / "detected_video.mp4")
        stats = detector.predict_video(
            source=src,
            output_path=out_vid,
            frame_skip=args.frame_skip,
        )
        print("\n影片推論統計:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args.source is None:
        print("錯誤: 請提供 --source (或使用 --video / --export)")
        return
    source_path = Path(args.source)

    if source_path.is_dir():
        # 批次偵測
        results = detector.predict_folder(
            input_dir=str(source_path),
            output_dir=args.output,
            save_json=not args.no_save_json,
        )
        if args.save_csv:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            detector.export_csv_report(
                results, str(Path(args.output) / f"report_{ts}.csv")
            )
    elif source_path.is_file():
        # 單張偵測
        detections, annotated = detector.predict_image(str(source_path))
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"detected_{source_path.name}"
        cv2.imwrite(str(save_path), annotated)
        print(f"\n偵測到 {len(detections)} 個瑕疵")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.4f}")
        print(f"標註影像已儲存: {save_path}")
    else:
        print(f"錯誤: 找不到來源 {args.source}")


if __name__ == "__main__":
    main()
