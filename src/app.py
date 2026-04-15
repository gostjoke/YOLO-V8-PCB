"""
PCB 瑕疵偵測系統 - Gradio GUI 介面
啟動方式: python src/app.py
"""

import cv2
import csv
import numpy as np
import gradio as gr
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import torch

# 嘗試載入推論 / 分析模組
try:
    from inference import PCBDefectDetector
    from preprocess import PCBPreprocessor, visualize_preprocessing
    from analyzer import (
        ReferenceDiffAnalyzer, SolderPasteAnalyzer, BlobDefectAnalyzer,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from inference import PCBDefectDetector
    from preprocess import PCBPreprocessor, visualize_preprocessing
    from analyzer import (
        ReferenceDiffAnalyzer, SolderPasteAnalyzer, BlobDefectAnalyzer,
    )

# ── 全域變數 ──────────────────────────────────────────
detector: Optional[PCBDefectDetector] = None
current_model_path: str = ""


# ── 輔助函式 ─────────────────────────────────────────
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """PIL → OpenCV BGR"""
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """OpenCV BGR → PIL"""
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def format_detections_table(detections: list) -> str:
    """將偵測結果格式化為 Markdown 表格"""
    if not detections:
        return "✅ **未偵測到瑕疵，PCB 品質良好**"

    rows = ["| # | 瑕疵類型 | 信心度 | 位置 (x1,y1,x2,y2) | 面積 |",
            "|---|---------|--------|---------------------|------|"]
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["bbox"]
        rows.append(
            f"| {i} | {det['class_name']} | "
            f"{det['confidence']:.2%} | "
            f"({x1},{y1},{x2},{y2}) | "
            f"{det['area']} px² |"
        )

    # 彙總
    class_counts = {}
    for det in detections:
        name = det["class_name"]
        class_counts[name] = class_counts.get(name, 0) + 1

    summary_parts = [f"**{name}** × {cnt}" for name, cnt in class_counts.items()]
    summary = "⚠️ 偵測到瑕疵: " + "、".join(summary_parts)

    return f"{summary}\n\n" + "\n".join(rows)


# ── 模型管理 ─────────────────────────────────────────
def load_model(model_path: str, conf: float, iou: float) -> str:
    """載入模型"""
    global detector, current_model_path

    if not model_path:
        return "❌ 請先指定模型路徑"

    if not Path(model_path).exists():
        return f"❌ 找不到模型檔案: {model_path}"

    try:
        device = "0" if torch.cuda.is_available() else "cpu"
        detector = PCBDefectDetector(
            model_path=model_path,
            confidence=conf,
            iou_threshold=iou,
            device=device,
        )
        current_model_path = model_path
        class_names = list(detector.class_names.values())
        gpu_info = f"GPU ({torch.cuda.get_device_name(0)})" \
            if torch.cuda.is_available() else "CPU"
        return (f"✅ 模型載入成功！\n"
                f"📁 路徑: {model_path}\n"
                f"🖥️ 裝置: {gpu_info}\n"
                f"🏷️ 偵測類別: {', '.join(class_names)}")
    except Exception as e:
        return f"❌ 載入失敗: {str(e)}"


# ── 單張影像偵測 ──────────────────────────────────────
def detect_single(
    image: Optional[Image.Image],
    conf: float,
    iou: float,
) -> Tuple[Optional[Image.Image], str]:
    """偵測單張 PCB 影像"""
    if image is None:
        return None, "⚠️ 請上傳一張 PCB 影像"

    if detector is None:
        return image, "⚠️ 請先在「模型設定」頁面載入模型"

    # 更新信心度閾值
    detector.confidence = conf
    detector.iou_threshold = iou

    try:
        cv2_img = pil_to_cv2(image)
        detections, annotated = detector.predict_image(cv2_img)
        result_pil = cv2_to_pil(annotated)
        table = format_detections_table(detections)
        return result_pil, table
    except Exception as e:
        return image, f"❌ 偵測失敗: {str(e)}"


# ── 影片偵測 ─────────────────────────────────────────
def detect_video(
    video_path: Optional[str],
    conf: float,
    iou: float,
    frame_skip: int,
) -> Tuple[Optional[str], str]:
    """對影片進行瑕疵偵測並回傳標註後影片路徑"""
    if not video_path:
        return None, "⚠️ 請上傳一段影片"
    if detector is None:
        return None, "⚠️ 請先在「模型設定」頁面載入模型"
    detector.confidence = conf
    detector.iou_threshold = iou

    out_dir = Path("./results/video_inference")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"detected_{ts}.mp4"

    try:
        stats = detector.predict_video(
            source=video_path,
            output_path=str(out_path),
            show=False,
            frame_skip=max(1, int(frame_skip)),
        )
        cls_lines = "\n".join(
            f"  - {n}: {c}" for n, c in stats["class_totals"].items()
        ) or "  (無偵測到瑕疵)"
        msg = (
            f"✅ 影片偵測完成\n\n"
            f"📹 總幀數: {stats['total_frames']}\n"
            f"⚠️ 瑕疵累計: {stats['total_defects']}\n"
            f"🚀 平均 FPS: {stats['avg_fps']}\n"
            f"📊 類別統計:\n{cls_lines}\n\n"
            f"📁 輸出: {out_path}"
        )
        return str(out_path), msg
    except Exception as e:
        return None, f"❌ 影片偵測失敗: {str(e)}"


# ── 前處理預覽 ────────────────────────────────────────
def preview_preprocess(
    image: Optional[Image.Image],
    denoise_method: str,
    contrast_method: str,
    apply_sharpen: bool,
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """套用選定的 OpenCV 前處理並回傳比較圖"""
    if image is None:
        return None, None
    pre = PCBPreprocessor()
    cv_img = pil_to_cv2(image)
    processed = pre.denoise(cv_img, method=denoise_method)
    processed = pre.enhance_contrast(processed, method=contrast_method)
    if apply_sharpen:
        processed = pre.sharpen(processed)

    # 比較圖 (原始 | 前處理 | Canny)
    edges = pre.detect_edges(processed)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    h = 360
    ratio = h / cv_img.shape[0]
    w = int(cv_img.shape[1] * ratio)
    imgs = [cv2.resize(x, (w, h)) for x in (cv_img, processed, edges_bgr)]
    for img, label in zip(imgs, ["Original", "Processed", "Edges"]):
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    comparison = np.hstack(imgs)
    return cv2_to_pil(processed), cv2_to_pil(comparison)


# ── 傳統 CV 分析器 ────────────────────────────────────
def analyze_reference_diff(
    reference: Optional[Image.Image],
    target: Optional[Image.Image],
    diff_threshold: int,
    min_area: int,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """黃金樣本對比"""
    if reference is None or target is None:
        return None, None, "⚠️ 請同時上傳良品與測試影像"
    try:
        analyzer = ReferenceDiffAnalyzer(
            min_area=int(min_area),
            diff_threshold=int(diff_threshold),
        )
        ref_cv = pil_to_cv2(reference)
        tgt_cv = pil_to_cv2(target)
        cands, heatmap, annotated = analyzer.analyze(ref_cv, tgt_cv)
        lines = [
            f"🔎 **對比分析完成** — 偵測到 **{len(cands)}** 個差異區域",
            "",
            "| # | BBox | 面積 | 差異分數 |",
            "|---|------|------|---------|",
        ]
        for i, c in enumerate(cands[:30], 1):
            lines.append(
                f"| {i} | {c.bbox} | {c.area} | {c.score} |"
            )
        if len(cands) > 30:
            lines.append(f"| ... | *尚有 {len(cands)-30} 筆未顯示* | | |")
        return cv2_to_pil(annotated), cv2_to_pil(heatmap), "\n".join(lines)
    except Exception as e:
        return None, None, f"❌ 分析失敗: {str(e)}"


def analyze_solder(
    image: Optional[Image.Image],
    v_min: int,
    min_area: int,
) -> Tuple[Optional[Image.Image], str]:
    """錫膏分析"""
    if image is None:
        return None, "⚠️ 請上傳影像"
    try:
        analyzer = SolderPasteAnalyzer(
            hsv_lower=(0, 0, int(v_min)),
            hsv_upper=(180, 80, 255),
            min_area=int(min_area),
        )
        cv_img = pil_to_cv2(image)
        stats, annotated = analyzer.analyze(cv_img)
        msg = (
            f"### 🧪 錫膏印刷分析結果\n"
            f"- 錫膏塊數量: **{stats['pad_count']}**\n"
            f"- 總面積: {stats['total_area']} px²\n"
            f"- 覆蓋率: **{stats['coverage']*100:.2f}%**\n"
            f"- 平均亮度 (厚度 proxy): {stats['avg_brightness']}\n"
        )
        return cv2_to_pil(annotated), msg
    except Exception as e:
        return None, f"❌ 分析失敗: {str(e)}"


def analyze_blob(
    image: Optional[Image.Image],
    min_area: int,
    max_area: int,
) -> Tuple[Optional[Image.Image], str]:
    """形態學小斑點分析"""
    if image is None:
        return None, "⚠️ 請上傳影像"
    try:
        analyzer = BlobDefectAnalyzer(
            min_area=int(min_area), max_area=int(max_area)
        )
        cv_img = pil_to_cv2(image)
        cands, annotated = analyzer.analyze(cv_img)
        return cv2_to_pil(annotated), (
            f"🔹 偵測到 **{len(cands)}** 個異常區塊 "
            f"(面積範圍 {min_area}~{max_area} px²)"
        )
    except Exception as e:
        return None, f"❌ 分析失敗: {str(e)}"


# ── 模型匯出 ─────────────────────────────────────────
def export_model_ui(format: str, image_size: int) -> str:
    if detector is None:
        return "⚠️ 請先載入模型"
    try:
        path = detector.export_model(
            format=format, image_size=int(image_size)
        )
        return f"✅ 匯出成功\n格式: {format}\n路徑: {path}"
    except Exception as e:
        return f"❌ 匯出失敗: {str(e)}"


# ── 批次資料夾偵測 ────────────────────────────────────
def detect_folder(
    input_dir: str,
    output_dir: str,
    conf: float,
    iou: float,
) -> str:
    """批次偵測資料夾中的 PCB 影像"""
    if not input_dir:
        return "⚠️ 請輸入輸入資料夾路徑"

    if detector is None:
        return "⚠️ 請先在「模型設定」頁面載入模型"

    if not Path(input_dir).exists():
        return f"❌ 找不到資料夾: {input_dir}"

    detector.confidence = conf
    detector.iou_threshold = iou

    output_dir = output_dir or "./results/batch_inference"

    try:
        results = detector.predict_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            save_json=True,
        )
        # 同步輸出 CSV 報告
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = str(Path(output_dir) / f"report_{ts}.csv")
        try:
            detector.export_csv_report(results, csv_path)
        except Exception:
            csv_path = None

        total = len(results)
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = total - passed

        # 找出最多瑕疵的影像
        worst = sorted(results, key=lambda x: x["defect_count"], reverse=True)[:3]
        worst_info = "\n".join([
            f"  - {r['image']}: {r['defect_count']} 個瑕疵"
            for r in worst if r["defect_count"] > 0
        ])

        return (
            f"✅ 批次偵測完成\n\n"
            f"📊 **統計摘要**\n"
            f"  總計: {total} 張\n"
            f"  通過: {passed} 張  ✅\n"
            f"  失敗: {failed} 張  ❌\n"
            f"  良品率: {passed/total*100:.1f}%\n\n"
            f"📁 **結果儲存位置**\n"
            f"  {output_dir}\n"
            f"  CSV 報告: {csv_path or '(匯出失敗)'}\n\n"
            f"⚠️ **瑕疵最多的影像**\n"
            f"{worst_info if worst_info else '  無瑕疵'}"
        )
    except Exception as e:
        return f"❌ 批次偵測失敗: {str(e)}"


# ── 建立 Gradio UI ────────────────────────────────────
def create_ui():
    """建立 Gradio 介面"""

    with gr.Blocks(
        title="PCB 瑕疵偵測系統",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .title-text { text-align: center; }
        .status-box { font-family: monospace; }
        """
    ) as demo:

        # ── 標題 ──
        gr.Markdown(
            """
            # 🔬 PCB 瑕疵偵測系統
            ### 基於 YOLOv8 + OpenCV + PyTorch
            """,
            elem_classes="title-text"
        )

        with gr.Tabs():

            # ══════════════════════════════════════════
            # Tab 1: 模型設定
            # ══════════════════════════════════════════
            with gr.Tab("⚙️ 模型設定"):
                gr.Markdown("### 載入訓練好的 YOLOv8 模型")

                with gr.Row():
                    with gr.Column(scale=2):
                        model_path_input = gr.Textbox(
                            label="模型路徑 (.pt 檔案)",
                            placeholder="例: ./results/pcb_yolov8n/weights/best.pt",
                            info="輸入訓練完成的 YOLOv8 權重檔案路徑"
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("**推論參數**")
                        conf_slider = gr.Slider(
                            minimum=0.1, maximum=0.9,
                            value=0.25, step=0.05,
                            label="信心度門檻 (Confidence)"
                        )
                        iou_slider = gr.Slider(
                            minimum=0.1, maximum=0.9,
                            value=0.45, step=0.05,
                            label="NMS IoU 門檻"
                        )

                load_btn = gr.Button("🚀 載入模型", variant="primary", size="lg")
                model_status = gr.Textbox(
                    label="模型狀態",
                    value="尚未載入模型",
                    interactive=False,
                    lines=5,
                    elem_classes="status-box"
                )

                load_btn.click(
                    fn=load_model,
                    inputs=[model_path_input, conf_slider, iou_slider],
                    outputs=[model_status]
                )

                gr.Markdown("""
                ---
                **💡 提示：**
                - 若無自訓練模型，可使用 `python src/train.py` 先進行訓練
                - 訓練完成後，最佳模型位於 `results/<實驗名稱>/weights/best.pt`
                - GPU 可大幅提升推論速度，建議使用 NVIDIA GPU
                """)

            # ══════════════════════════════════════════
            # Tab 2: 單張影像偵測
            # ══════════════════════════════════════════
            with gr.Tab("🔍 單張影像偵測"):
                gr.Markdown("### 上傳 PCB 影像進行即時瑕疵偵測")

                with gr.Row():
                    conf_single = gr.Slider(
                        minimum=0.1, maximum=0.9,
                        value=0.25, step=0.05,
                        label="信心度門檻"
                    )
                    iou_single = gr.Slider(
                        minimum=0.1, maximum=0.9,
                        value=0.45, step=0.05,
                        label="IoU 門檻"
                    )

                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            label="輸入影像",
                            type="pil",
                            height=400,
                        )
                        detect_btn = gr.Button(
                            "🔎 開始偵測", variant="primary", size="lg"
                        )

                    with gr.Column():
                        output_image = gr.Image(
                            label="偵測結果",
                            type="pil",
                            height=400,
                        )

                result_table = gr.Markdown(
                    "尚未執行偵測",
                    label="偵測詳情"
                )

                detect_btn.click(
                    fn=detect_single,
                    inputs=[input_image, conf_single, iou_single],
                    outputs=[output_image, result_table]
                )

                gr.Markdown("""
                ---
                **📋 瑕疵類別說明：**
                | 類別 | 說明 |
                |------|------|
                | missing_hole | 缺孔 - PCB 上應有的孔洞缺失 |
                | mouse_bite | 鼠咬缺口 - 邊緣缺口瑕疵 |
                | open_circuit | 開路 - 導線斷路 |
                | short | 短路 - 導線短接 |
                | spur | 毛刺 - 多餘的細小導線 |
                | spurious_copper | 多餘銅箔 - 不應存在的銅區域 |
                """)

            # ══════════════════════════════════════════
            # Tab 3: 批次資料夾偵測
            # ══════════════════════════════════════════
            with gr.Tab("📁 批次資料夾偵測"):
                gr.Markdown("### 對整個資料夾的 PCB 影像進行批次瑕疵偵測")

                with gr.Row():
                    input_dir_box = gr.Textbox(
                        label="輸入資料夾路徑",
                        placeholder="例: ./data/test/images",
                    )
                    output_dir_box = gr.Textbox(
                        label="輸出資料夾路徑",
                        placeholder="例: ./results/batch_inference",
                        value="./results/batch_inference"
                    )

                with gr.Row():
                    conf_batch = gr.Slider(
                        minimum=0.1, maximum=0.9,
                        value=0.25, step=0.05,
                        label="信心度門檻"
                    )
                    iou_batch = gr.Slider(
                        minimum=0.1, maximum=0.9,
                        value=0.45, step=0.05,
                        label="IoU 門檻"
                    )

                batch_btn = gr.Button(
                    "🚀 開始批次偵測", variant="primary", size="lg"
                )
                batch_result = gr.Textbox(
                    label="批次偵測結果",
                    lines=15,
                    interactive=False,
                    elem_classes="status-box"
                )

                batch_btn.click(
                    fn=detect_folder,
                    inputs=[input_dir_box, output_dir_box,
                            conf_batch, iou_batch],
                    outputs=[batch_result]
                )

            # ══════════════════════════════════════════
            # Tab: 影片 / 攝影機偵測
            # ══════════════════════════════════════════
            with gr.Tab("🎥 影片偵測"):
                gr.Markdown("### 上傳 PCB 產線影片進行連續瑕疵偵測")
                with gr.Row():
                    video_input = gr.Video(label="輸入影片", height=360)
                    video_output = gr.Video(label="偵測結果", height=360)
                with gr.Row():
                    conf_vid = gr.Slider(0.1, 0.9, 0.25, 0.05,
                                         label="信心度門檻")
                    iou_vid = gr.Slider(0.1, 0.9, 0.45, 0.05,
                                        label="IoU 門檻")
                    skip_vid = gr.Slider(1, 10, 1, 1,
                                         label="Frame Skip (每 N 幀推論)")
                video_btn = gr.Button("▶️ 開始偵測", variant="primary",
                                      size="lg")
                video_status = gr.Textbox(label="執行狀態", lines=10,
                                          interactive=False,
                                          elem_classes="status-box")
                video_btn.click(
                    fn=detect_video,
                    inputs=[video_input, conf_vid, iou_vid, skip_vid],
                    outputs=[video_output, video_status],
                )

            # ══════════════════════════════════════════
            # Tab: 前處理預覽
            # ══════════════════════════════════════════
            with gr.Tab("🎨 前處理預覽"):
                gr.Markdown(
                    "### 以 OpenCV 檢視去噪 / 對比 / 銳化對影像的影響"
                )
                with gr.Row():
                    pre_in = gr.Image(label="輸入", type="pil", height=360)
                    pre_out = gr.Image(label="處理後", type="pil",
                                       height=360)
                pre_cmp = gr.Image(label="比較圖 (Original | Processed | Edges)",
                                   type="pil")
                with gr.Row():
                    denoise_sel = gr.Dropdown(
                        ["gaussian", "bilateral", "nlm"],
                        value="bilateral", label="去噪方法",
                    )
                    contrast_sel = gr.Dropdown(
                        ["clahe", "histogram", "gamma"],
                        value="clahe", label="對比增強",
                    )
                    sharpen_chk = gr.Checkbox(value=True, label="套用銳化")
                pre_btn = gr.Button("🔧 執行前處理", variant="primary")
                pre_btn.click(
                    fn=preview_preprocess,
                    inputs=[pre_in, denoise_sel, contrast_sel, sharpen_chk],
                    outputs=[pre_out, pre_cmp],
                )

            # ══════════════════════════════════════════
            # Tab: 傳統 CV 分析器
            # ══════════════════════════════════════════
            with gr.Tab("🧪 傳統 CV 分析"):
                gr.Markdown(
                    "### 以經典影像處理方法輔助偵測 (可與 YOLO 互補)"
                )
                with gr.Tabs():
                    # 黃金樣本比對
                    with gr.Tab("Ⓐ 黃金樣本比對"):
                        with gr.Row():
                            ref_img = gr.Image(label="良品 (Reference)",
                                               type="pil", height=280)
                            tgt_img = gr.Image(label="測試影像 (Target)",
                                               type="pil", height=280)
                        with gr.Row():
                            diff_out = gr.Image(label="差異標註",
                                                type="pil", height=280)
                            heat_out = gr.Image(label="差異熱圖",
                                                type="pil", height=280)
                        with gr.Row():
                            diff_th = gr.Slider(10, 120, 40, 5,
                                                label="差異門檻")
                            diff_min = gr.Slider(5, 500, 30, 5,
                                                 label="最小面積")
                        diff_btn = gr.Button("🔍 開始比對",
                                             variant="primary")
                        diff_table = gr.Markdown("尚未分析")
                        diff_btn.click(
                            fn=analyze_reference_diff,
                            inputs=[ref_img, tgt_img, diff_th, diff_min],
                            outputs=[diff_out, heat_out, diff_table],
                        )

                    # 錫膏分析
                    with gr.Tab("Ⓑ 錫膏分析 (SPI)"):
                        with gr.Row():
                            sp_in = gr.Image(label="錫膏影像",
                                             type="pil", height=320)
                            sp_out = gr.Image(label="分析結果",
                                              type="pil", height=320)
                        with gr.Row():
                            sp_v = gr.Slider(50, 250, 150, 5,
                                             label="亮度門檻 (V_min)")
                            sp_area = gr.Slider(5, 300, 15, 5,
                                                label="最小塊面積")
                        sp_btn = gr.Button("🧪 分析錫膏",
                                           variant="primary")
                        sp_stats = gr.Markdown("尚未分析")
                        sp_btn.click(
                            fn=analyze_solder,
                            inputs=[sp_in, sp_v, sp_area],
                            outputs=[sp_out, sp_stats],
                        )

                    # Blob 分析
                    with gr.Tab("Ⓒ 形態學 Blob 偵測"):
                        with gr.Row():
                            blob_in = gr.Image(label="輸入",
                                               type="pil", height=320)
                            blob_out = gr.Image(label="結果",
                                                type="pil", height=320)
                        with gr.Row():
                            blob_min = gr.Slider(1, 200, 10, 1,
                                                 label="最小面積")
                            blob_max = gr.Slider(100, 10000, 5000, 100,
                                                 label="最大面積")
                        blob_btn = gr.Button("🔎 Blob 分析",
                                             variant="primary")
                        blob_stats = gr.Markdown("尚未分析")
                        blob_btn.click(
                            fn=analyze_blob,
                            inputs=[blob_in, blob_min, blob_max],
                            outputs=[blob_out, blob_stats],
                        )

            # ══════════════════════════════════════════
            # Tab: 模型匯出
            # ══════════════════════════════════════════
            with gr.Tab("📦 模型匯出"):
                gr.Markdown(
                    "### 將訓練好的模型匯出為部署格式\n"
                    "- **ONNX** → 跨平台通用格式 (CPU/GPU/邊緣裝置)\n"
                    "- **TorchScript** → PyTorch 原生部署\n"
                    "- **OpenVINO** → Intel 裝置加速\n"
                    "- **TensorRT (engine)** → NVIDIA GPU 極速推論\n"
                )
                with gr.Row():
                    exp_format = gr.Dropdown(
                        ["onnx", "torchscript", "openvino",
                         "engine", "coreml"],
                        value="onnx", label="匯出格式",
                    )
                    exp_size = gr.Slider(320, 1280, 640, 32,
                                         label="影像尺寸")
                exp_btn = gr.Button("🚀 開始匯出",
                                    variant="primary", size="lg")
                exp_status = gr.Textbox(label="匯出狀態", lines=4,
                                        interactive=False,
                                        elem_classes="status-box")
                exp_btn.click(
                    fn=export_model_ui,
                    inputs=[exp_format, exp_size],
                    outputs=[exp_status],
                )

            # ══════════════════════════════════════════
            # Tab 4: 使用說明
            # ══════════════════════════════════════════
            with gr.Tab("📖 使用說明"):
                gr.Markdown("""
                ## PCB 瑕疵偵測系統使用說明

                ### 🚀 快速開始

                **步驟一：準備資料集**
                ```bash
                # 下載 DeepPCB 公開資料集 (推薦)
                python src/prepare_dataset.py --dataset deeppcb
                ```

                **步驟二：訓練模型**
                ```bash
                # 使用預設設定訓練 YOLOv8n (最快)
                python src/train.py --model n --epochs 100

                # 使用較大模型獲得更高精度
                python src/train.py --model s --epochs 150 --batch 8
                ```

                **步驟三：在此介面中載入模型並進行偵測**

                ---

                ### 📊 模型大小選擇指南

                | 模型 | 參數量 | 速度 | 精度 | 建議場景 |
                |------|--------|------|------|---------|
                | YOLOv8n | 3.2M | 最快 | 普通 | 即時生產線 |
                | YOLOv8s | 11.2M | 快 | 良好 | 一般應用 |
                | YOLOv8m | 25.9M | 中 | 好 | 高精度需求 |
                | YOLOv8l | 43.7M | 慢 | 很好 | 離線分析 |
                | YOLOv8x | 68.2M | 最慢 | 最佳 | 研究/分析 |

                ---

                ### 📁 公開 PCB 資料集資源

                1. **DeepPCB** (推薦)
                   - GitHub: `tangsanli5201/DeepPCB`
                   - 6 類瑕疵: 缺孔/鼠咬/開路/短路/毛刺/多餘銅箔
                   - 共 1500+ 張影像

                2. **PCB Defect Dataset (Kaggle)**
                   - https://www.kaggle.com/datasets/akhatova/pcb-defects

                3. **Roboflow PCB Universe**
                   - https://universe.roboflow.com/search?q=pcb+defect

                ---

                ### 🏷️ 自訂資料集標注

                使用 **LabelImg** 或 **Roboflow** 進行標注：
                ```bash
                # 安裝 LabelImg
                pip install labelImg
                labelImg
                ```
                - 選擇 YOLO 格式輸出
                - 標注後的標籤放入 `data/train/labels/`
                - 對應影像放入 `data/train/images/`
                """)

    return demo


# ── 主程式 ───────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCB 瑕疵偵測 - Gradio GUI")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="伺服器 Host")
    parser.add_argument("--port", type=int, default=7860,
                        help="伺服器 Port")
    parser.add_argument("--share", action="store_true",
                        help="建立公開分享連結")
    parser.add_argument("--model", type=str, default=None,
                        help="預先載入的模型路徑")

    args = parser.parse_args()

    # 若指定了模型，預先載入
    if args.model:
        result = load_model(args.model, 0.25, 0.45)
        print(result)

    demo = create_ui()
    print(f"\n🚀 PCB 瑕疵偵測系統啟動中...")
    print(f"   本地訪問: http://localhost:{args.port}")
    print(f"   按 Ctrl+C 停止服務\n")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )
