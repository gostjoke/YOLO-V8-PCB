# 🔬 PCB 瑕疵偵測系統

> 基於 **Python + OpenCV + PyTorch + YOLOv8** 的印刷電路板（PCB）瑕疵與錫膏自動化品質檢測平台

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-yellow)](https://gradio.app/)

---

## 📌 專案簡介

本系統利用深度學習技術對 PCB 影像進行自動化瑕疵偵測，能識別製造過程中常見的 **6 種瑕疵類型**，以及錫膏印刷（SPI, Solder Paste Inspection）問題。

核心技術組合：
- **YOLOv8**：高效能 Anchor-Free 目標偵測模型，支援從 Nano 到 Extra-Large 等五種規格
- **OpenCV**：影像前處理（去噪、對比增強、銳化、邊緣偵測）
- **PyTorch**：深度學習框架，支援 GPU 加速訓練與推論
- **Gradio**：快速建立互動式網頁 GUI 介面，無需前端開發

---

## 🗂️ 專案結構

```
Yolov PCB/
├── configs/
│   ├── pcb_dataset.yaml      # 資料集設定（路徑、類別定義）
│   └── train_config.yaml     # 訓練超參數（學習率、增強策略等）
├── data/
│   ├── train/
│   │   ├── images/           # 訓練集影像
│   │   └── labels/           # 訓練集標注（YOLO .txt 格式）
│   ├── val/
│   │   ├── images/           # 驗證集影像
│   │   └── labels/           # 驗證集標注
│   └── test/
│       └── images/           # 測試集影像（無標注）
├── models/                   # 自訂模型架構設定（可選）
├── notebooks/                # Jupyter 分析與實驗筆記本
├── results/                  # 訓練結果、推論輸出（自動生成）
│   └── <實驗名稱>/
│       ├── weights/
│       │   ├── best.pt       # 最佳驗證效能的模型
│       │   └── last.pt       # 最後一個 epoch 的模型
│       └── ...               # 訓練曲線、混淆矩陣等圖表
├── src/
│   ├── app.py                # Gradio GUI 主程式入口
│   ├── inference.py          # 推論模組（PCBDefectDetector 類別）
│   ├── preprocess.py         # OpenCV 影像前處理工具
│   └── train.py              # YOLOv8 模型訓練腳本
└── requirements.txt          # Python 套件依賴清單
```

---

## 🏷️ 支援的瑕疵類別

### PCB 瑕疵偵測（標準 6 類）

| ID | 類別名稱 | 中文說明 | 常見成因 |
|----|---------|---------|---------|
| 0 | `missing_hole` | 缺孔 | 鑽孔設備故障、PCB 對位偏移 |
| 1 | `mouse_bite` | 鼠咬缺口 | 板邊裁切不良、機械撞擊 |
| 2 | `open_circuit` | 開路 | 蝕刻過度、導線斷裂 |
| 3 | `short` | 短路 | 蝕刻不足、殘留銅導致意外連接 |
| 4 | `spur` | 毛刺 | 蝕刻不均勻、導線邊緣殘留 |
| 5 | `spurious_copper` | 多餘銅箔 | 光阻顯影不完全 |

### 錫膏偵測擴充類別（SPI，可選設定）

修改 `configs/pcb_dataset.yaml` 中 `names` 區塊即可切換至錫膏模式：

| 類別名稱 | 說明 |
|---------|------|
| `insufficient_solder` | 錫量不足，可能導致虛焊 |
| `excess_solder` | 錫量過多，可能造成橋接 |
| `solder_bridge` | 錫橋，相鄰焊墊短路 |
| `missing_solder` | 焊墊完全缺錫 |
| `cold_solder` | 冷焊，焊錫結晶化品質不佳 |
| `offset_solder` | 錫膏偏移超出容許範圍 |

---

## ⚙️ 環境需求

| 項目 | 需求 |
|------|------|
| Python | >= 3.10（建議 3.10 / 3.11，避免 3.12+ 相容性問題） |
| GPU（建議） | NVIDIA GPU，VRAM >= 4GB，CUDA 11.8 / 12.x |
| RAM | >= 8GB（訓練建議 16GB+） |
| 作業系統 | Windows 10/11、Ubuntu 20.04+、macOS 12+ |

---

## 🚀 快速開始

### 步驟一：安裝 uv 與依賴套件

```bash
# 安裝 uv（若尚未安裝）
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 使用 uv 同步專案（自動建立虛擬環境 + 安裝所有依賴）
uv sync

# 若有 NVIDIA GPU，可指定 PyTorch CUDA 版本:
uv sync --extra-index-url https://download.pytorch.org/whl/cu121

# 安裝開發工具（ruff, pytest 等）
uv sync --dev

# 也可用 uv run 直接執行腳本（自動啟用虛擬環境）
uv run python src/train.py --model n --epochs 100
```

> **傳統 pip 方式（替代方案）：**
> ```bash
> python -m venv venv
> source venv/bin/activate   # Linux / macOS
> venv\Scripts\activate      # Windows
> pip install -r requirements.txt
> ```

### 步驟二：準備資料集

#### 方法 A：DeepPCB 公開資料集（推薦）

1. 前往 [DeepPCB GitHub](https://github.com/tangsanli5201/DeepPCB) 下載資料集
2. 轉換為 YOLO 格式並分割資料集：

```bash
# 轉換 VOC XML 標注 → YOLO txt 格式
uv run python src/prepare_dataset.py --mode convert \
  --image-dir /path/to/DeepPCB/images \
  --annot-dir /path/to/DeepPCB/annotations \
  --output-dir ./data/raw

# 按 8:1:1 自動分割 train / val / test
uv run python src/prepare_dataset.py --mode split \
  --image-dir ./data/raw/images \
  --label-dir ./data/raw/labels \
  --output-dir ./data
```

#### 方法 B：Roboflow 線上資料集

```bash
uv add roboflow
```

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("pcb-defect")
dataset = project.version(1).download("yolov8")
```

#### 方法 C：自行拍攝並標注

1. 安裝標注工具：
   ```bash
   # Python 3.12+ 需要加入 setuptools（避免 distutils 模組缺失錯誤）
   uv tool install labelImg --with setuptools --reinstall

   # 若已正常安裝則直接執行
   labelImg
   ```
   > **⚠️ 注意（Python 3.12+）**：Python 3.12 移除了 `distutils` 模組，直接 `uv tool install labelImg` 會出現 `ModuleNotFoundError: No module named 'distutils'`。請務必加上 `--with setuptools` 旗標。若仍有問題，建議改用 Python 3.11 執行。

2. 在 LabelImg 中選擇 **YOLO 格式**，對 PCB 瑕疵框選邊界框並儲存。
3. 將影像與標注分別放入 `data/train/images/` 和 `data/train/labels/`。

### 步驟三：確認資料集設定

編輯 `configs/pcb_dataset.yaml`，確認路徑正確：

```yaml
path: ./data        # 資料集根目錄
train: train/images
val: val/images
test: test/images

nc: 6               # 類別數
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
```

### 步驟四：訓練模型

```bash
# 使用 YOLOv8n（最快，適合即時應用）
uv run python src/train.py --model n --epochs 100

# 使用 YOLOv8s（平衡速度與精度）
uv run python src/train.py --model s --epochs 150 --batch 8

# 使用 YOLOv8m（較高精度）
uv run python src/train.py --model m --epochs 200 --batch 4

# 指定 GPU 與 batch size
uv run python src/train.py --model n --epochs 100 --device 0 --batch 16

# 從中斷點繼續訓練
uv run python src/train.py --resume --resume-path ./results/pcb_defect_v1/weights/last.pt
```

訓練完成後，最佳模型自動儲存於：
```
results/<實驗名稱>/weights/best.pt
```

### 步驟五：啟動 GUI 偵測介面

```bash
# 基本啟動
uv run python src/app.py

# 預先指定模型路徑
uv run python src/app.py --model ./results/pcb_defect_v1/weights/best.pt

# 建立 Gradio 公開分享連結（適合遠端展示）
uv run python src/app.py --share

# 自訂伺服器 IP 與 Port
uv run python src/app.py --host 0.0.0.0 --port 8080
```

開啟瀏覽器訪問：**`http://localhost:7860`**

### 步驟六：命令列批次推論（選用）

```bash
# 單張影像偵測
uv run python src/inference.py \
  --model ./results/pcb_defect_v1/weights/best.pt \
  --source ./data/test/images/sample.jpg \
  --output ./results/inference \
  --conf 0.25

# 批次資料夾偵測
uv run python src/inference.py \
  --model ./results/pcb_defect_v1/weights/best.pt \
  --source ./data/test/images/ \
  --output ./results/inference \
  --conf 0.3 \
  --save-json
```

---

## 🖥️ GUI 介面功能說明

系統提供 4 個功能頁籤：

### ⚙️ 模型設定
- 輸入 `.pt` 模型權重路徑並載入
- 調整**信心度門檻**（Confidence Threshold）：過濾低可信度偵測結果
- 調整 **NMS IoU 門檻**：控制重疊框的合併方式
- 顯示模型類別清單、GPU/CPU 裝置資訊

### 🔍 單張影像偵測
- 上傳任意 PCB 影像（JPG / PNG / BMP）
- 輸出標注後影像（含彩色 Bounding Box 與類別標籤）
- 顯示偵測詳情表格（瑕疵類型、信心度、座標位置、面積）

### 📁 批次資料夾偵測
- 輸入資料夾路徑，一次偵測所有影像
- 統計輸出：總數、良品數、不良品數、**良品率（%）**
- 列出瑕疵數量最多的前三張影像
- 結果自動儲存為標注影像與 JSON 報告

### 📖 使用說明
- 系統內建的完整操作指引、資料集說明、模型選擇建議

---

## 📊 模型選擇指南

| 模型 | 參數量 | mAP50（參考） | 推論速度（GPU） | 建議場景 |
|------|--------|--------------|---------------|---------|
| YOLOv8n | 3.2M | ~0.92 | ~5ms/張 | 即時生產線、邊緣裝置 |
| YOLOv8s | 11.2M | ~0.95 | ~8ms/張 | 一般工業應用 |
| YOLOv8m | 25.9M | ~0.96 | ~12ms/張 | 高精度需求 |
| YOLOv8l | 43.7M | ~0.97 | ~20ms/張 | 離線批次分析 |
| YOLOv8x | 68.2M | ~0.98 | ~30ms/張 | 研究與精度優先 |

> 📌 以上 mAP 數值為基於 DeepPCB 資料集的參考估計，實際效能依資料量與訓練設定而異。

---

## 🔧 OpenCV 影像前處理功能

`src/preprocess.py` 提供以下前處理方法：

| 功能 | 方法 | 說明 |
|------|------|------|
| 去噪 | 高斯濾波、雙邊濾波、NLM | 移除拍攝雜訊，保留邊緣 |
| 對比增強 | CLAHE、直方圖均衡、Gamma 校正 | 改善低對比度影像 |
| 銳化 | 拉普拉斯銳化核 | 強化瑕疵邊緣細節 |
| 邊緣偵測 | Canny Edge Detection | 提取結構特徵 |
| 資料增強 | 翻轉、旋轉、縮放、色彩抖動、高斯噪聲 | 訓練集擴充用 |

---

## 🛠️ 關鍵訓練超參數

`configs/train_config.yaml` 中的重要設定說明：

```yaml
# 基本設定
model: yolov8n.pt    # 選擇模型規格（n/s/m/l/x）
epochs: 100          # 訓練回合數（建議 100~200）
batch: 16            # 批次大小（8GB VRAM → 16，4GB → 8）
imgsz: 640           # 輸入尺寸（PCB 高解析建議 640 或 1280）
optimizer: AdamW     # 優化器（AdamW 收斂穩定）
lr0: 0.001           # 初始學習率
patience: 50         # Early Stopping（50 回合無改善即停止）

# 資料增強策略
mosaic: 1.0          # Mosaic 增強（4 張拼接，提升小目標偵測）
mixup: 0.1           # MixUp 增強（改善類別邊界）
copy_paste: 0.1      # Copy-Paste 增強（瑕疵數量平衡）
flipud: 0.3          # 垂直翻轉
fliplr: 0.5          # 水平翻轉
degrees: 10.0        # 旋轉角度範圍（±10°）
```

---

## 💡 訓練技巧建議

1. **資料量**：每個瑕疵類別至少 100 張影像，500 張以上效果更佳
2. **標注品質**：精確的邊界框比大量粗糙標注更重要，邊界框應緊貼瑕疵
3. **類別不平衡**：若各類別數量差異大，可使用 Copy-Paste 增強或調整損失權重
4. **批次大小**：在 GPU VRAM 允許範圍內盡量加大，有助訓練穩定性
5. **學習率暖身**：預設 `warmup_epochs: 3`，避免初期梯度爆炸
6. **遷移學習**：使用 COCO 預訓練權重（預設行為）可大幅縮短訓練時間至 1/3 以下
7. **驗證集品質**：驗證集應包含各類別且與訓練集分布相似，但不重疊

---

## 📦 公開資料集資源

| 資料集 | 影像數 | 類別 | 下載方式 |
|-------|--------|------|---------|
| [DeepPCB](https://github.com/tangsanli5201/DeepPCB) | 1,500+ | 6 種瑕疵 | GitHub 直接下載 |
| [PCB Defect (Kaggle)](https://www.kaggle.com/datasets/akhatova/pcb-defects) | 1,386 | 6 種瑕疵 | Kaggle API |
| [Roboflow PCB Universe](https://universe.roboflow.com/search?q=pcb+defect) | 多種 | 多種 | Roboflow API |

---

## 📐 技術架構流程

```
PCB 影像輸入
     │
     ▼
┌─────────────────────┐
│  OpenCV 前處理        │  去噪 / 對比增強 / 銳化
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  YOLOv8 推論          │  Backbone (CSPDarknet)
│  (PyTorch)           │  Neck (PANet)
│                      │  Head (Anchor-Free 偵測)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  NMS 後處理           │  去除重疊預測框
└─────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  輸出結果                       │
│  • 標注影像（Bounding Box）       │
│  • 瑕疵類別 & 信心度              │
│  • JSON 報告                    │
│  • 良品 / 不良品判定              │
└──────────────────────────────┘
```

---

## 🤝 相關資源

| 資源 | 連結 |
|------|------|
| YOLOv8 官方文件 | https://docs.ultralytics.com |
| PyTorch 安裝指引 | https://pytorch.org/get-started/locally/ |
| DeepPCB 資料集 | https://github.com/tangsanli5201/DeepPCB |
| Roboflow 標注平台 | https://roboflow.com |
| LabelImg 標注工具 | https://github.com/heartexlabs/labelImg |
| OpenCV 文件 | https://docs.opencv.org |
| Gradio 文件 | https://www.gradio.app/docs |

---

## 📄 授權聲明

本專案僅供學術研究與工業開發使用。使用公開資料集時，請遵守各資料集原始授權條款。

---

*本系統由 Python + OpenCV + PyTorch + YOLOv8 + Gradio 技術組合打造*
