# 📊 論文復現：結合深層學習與大數據分析增強智慧城市基礎設施與人力資源管理系統 (2026)

本專案旨在完整復現 2026 年發表於 Ain Shams Engineering Journal 的研究論文，專注於 HRMS 人力資源管理系統中的員工離職預測任務。

## 📁 專案結構 (Project Structure)

```
.
├── 📄 1_environment_and_EDA.ipynb      # 主程序 (資料預處理、模型建構、訓練與評估)
├── 📄 usco_optimizer.py                # 智慧型單一候選解優化器 (USCO 核心算法)
├── 📄 README.md                        # 本說明文件
└── 📁 archive/                         # 資料集存放目錄
    └── 📄 WA_Fn-UseC_-HR-Employee-Attrition.csv  # IBM HR 原始資料集
```

## 🛠️ 演算法核心架構 (Algorithm Core)

### 1. 表格轉圖像技術 (Tabular-to-Spatial Transformation)

依據論文 Section 3.4.1，傳統的表格資料（Tabular Data）無法直接送入卷積神經網絡。本專案採用創新的表格轉圖像技術，將 HR 員工數據轉換為 2D 圖像矩陣，實現自動特徵提取。

### 2. 智慧單一候選解優化器 (USCO Algorithm)

實現了論文提出的 USCO 演算法，用於超參數自動優化和模型訓練。

## ⚙️ 環境配置 (Environment Setup)

無需修改任何一行代碼，即可自動在本地 CPU（測試、排錯）與 Google Colab T4 GPU（正式大批量訓練）之間自動切換，百分之百不報錯。

### 運行步驟

1. **雲端運行** (推薦)：直接在 Google Colab 中打開 `1_environment_and_EDA.ipynb`
2. **本地運行**：安裝依賴後在 Jupyter 中執行

### 調試模式

將 `QUICK_DEBUG_MODE` 設為 `True`。此時調參疊代次數會強制降為 3 次、訓練輪次降為 1 輪，以防本機 CPU 跑太久而過載。

## 📈 評估指標對照 (Section 4.3 & 5.2)

本專案使用的評估指標包括：
- **準確率 (Accuracy)**
- **精確率 (Precision)**
- **召回率 (Recall)**
- **F1-Score**
- **AUC-ROC**

## 📚 引用 (Citation)

```bibtex
@article{zhang2026integrating,
  title={Integrating deep learning and big data analytics for enhanced smart city infrastructure and human resource management systems},
  author={Zhang, Jianjun and Zhang, Ao and Liu, Xingying and Luo, Pan},
  journal={Ain Shams Engineering Journal},
  volume={17},
  pages={103962},
  year={2026},
  publisher={Elsevier}
}
```

---

**最後更新**: 2026 年 6 月
