"""
PCB 瑕疵偵測系統 - Gradio GUI 介面
啟動方式: python src/app.py
"""

import cv2
import numpy as np
import gradio as gr
import json
import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import torch

# 嘗試載入推論模組
try:
    from inference import PCBDefectDetector
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from inference import PCBDefectDetector

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
            f"  {output_dir}\n\n"
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
