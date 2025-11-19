## Folder Naming Convention | 資料夾命名規則

Simulation folders are named using the following pattern:

`Nanofocusing_TXX_FRYY_YZ_FOMN`

對應的命名邏輯為：

- `TXX`  → **薄膜厚度 (Thickness)**  
  - 例如 `T10` = 10 nm、`T20` = 20 nm、`T30` = 30 nm

- `FRYY` → **Filter radius 濾波半徑 (FR, filter radius)**  
  - 例如 `FR10` = 10 nm filter radius  
  - `FR20` = 20 nm filter radius  

- `YF / YT` → **Yee grid 設定**  
  - `YT` = Yee grid = **True**  
  - `YF` = Yee grid = **False**  

- `FOMN` → **使用的 FOM 編號 (Figure of Merit index)**  
  - `FOM1`、`FOM2`、`FOM3` 分別代表不同的目標函數定義  
    （例如：傳輸/面積比、近場場強增強、或其他組合；可在後續小節詳細說明）

### Example

- `Nanofocusing_T10_FR10_YF_FOM1`  
  → 10 nm 厚度、filter radius = 10 nm、Yee grid = False、使用 FOM1。

- `Nanofocusing_T20_FR20_YT_FOM3`  
  → 20 nm 厚度、filter radius = 20 nm、Yee grid = True、使用 FOM3。

## 📘 Instruction Files | 說明文件位置

專案中包含多個教學與流程說明檔，方便理解整個 Nanofocusing 優化與驗證方法。

### 1. Adjoint Method 說明（Adjoint Optimization Guide）
路徑：Instruction_Nanofocusing_T20_FR20_YT_FOM3/Instruction_Adjoint.ipynb

內容包含：
- Adjoint Method 的原理與流程  
- MaterialGrid 與 sensitivities 的更新方式  
- Beta-scaling、filter + projection 的作用  
- Meep Adjoint 實作註解（含設計區、激發源、FOM 設計）

---

### 2. 模擬驗證說明（Pre-Simulation Documentation）
路徑：Instruction_Nanofocusing_T20_FR20_YT_FOM3/final_data/Instruction(DFT_GAS)_PreSim.ipynb

內容包含：
- 空場 (empty-field) 模擬方法  
- DFT_GAS 設定與使用  
- 參考光源正規化方式  
- 用來比對優化後結構之完整驗證流程

---

### 3. 後處理與畫圖說明（Post Processing & Visualization）
路徑：nstruction_Nanofocusing_T20_FR20_YT_FOM3/final_data/PostSim_analyze.ipynb

內容包含：
- 讀取 Ex/Ey/Ez DFT fields  
- 計算 |E|²、場強增強倍率  
- XZ/XY/YZ 截面畫圖方法  
- 模式體積 (mode volume) 與 localization length 計算  
- 能量穿透率 (Transmission) 與反射率 (Reflection) 分析

---

這些說明檔提供完整的模擬框架，包括：

- Meep 模擬設定  
- Adjoint 優化流程  
- 空場正規化  
- 後處理分析與圖示流程  

讓使用者能完整重現 nanofocusing 的模擬與優化結果。


## 📁 Dataset Structure 說明

每一個 Nanofocusing 資料夾都包含以下三個主要資料夾：

- `final_data/`
- `s_change/`
- `v_data/`
- 以及一個主程式（Python .py 檔），負責執行整個優化流程。

以下為各資料夾的詳細功能說明：

---

## 🔵 final_data — 最終優化結果與驗證檔案

`final_data/` 包含整個拓樸優化結束後的最終結果，包括：

### **1. 最終設計參數（用來繪製最終結構）**
以下 `.npy` 檔記錄了優化後的設計分佈與中間變數，可用於重建最終結構：

- `Post_beta_scale_array.npy`  
  儲存各階段 beta 值的紀錄（beta-scaling 參數軌跡）

- `Post_cur_beta_array.npy`  
  儲存目前使用的 beta 值

- `Post_eta_i_array.npy`  
  儲存 filter 後投影前的連續參數（η_i）

- `Post_x_array.npy`  
  儲存最終投影後的 0/1 結構（最終可製造拓樸）

### **2. 優化過程紀錄**

- `Post_evaluation_history.npy`  
  儲存每一輪的 FOM（Figure of Merit）變化，用來觀察收斂情形。

### **3. 驗證模擬相關檔案（DFT_GAS.py 跑出的結果）**

此區域包含所有 **空場/結構場驗證模擬** 所需的資訊：

- `DFT_GAS.py`  
  用於對優化後結構進行頻域分析與驗證（Empty-field + Structure-field）。

- `FOM.npy`  
  儲存最終 FOM 計算結果。

- `FOM_ET_empty.npy`  
  空場（Empty-field）加權 E/T 計算結果。

- `DFT_mode_volume/`  
  包含模式體積計算需要的能量密度分佈。

- `TRAN/`  
  包含傳輸分析的資料（Tx, Ty, intensity maps 等）。

- `Instruction(DFT_GAS)_PreSim.ipynb`  
  模擬驗證（Pre-simulation）教學與空場正規化說明。

- `PostSim_analyze.ipynb`  
  所有後處理與繪圖流程（Ex/Ey/Ez、|E|²、聚焦截面、模式體積等）。

---

## 🔵 s_change — 每一步優化的結構變化紀錄

`s_change/` 資料夾儲存：

- 每一次 adjoint iteration 更新後的 **設計區圖形（PNG）**

檔名形式例如：
s_change000.png
s_change001.png
s_change002.png
...

這些圖片可用於動畫化（visual tracking）以觀察拓樸如何在優化過程中逐步演化。

---

## 🔵 v_data — 每階段結果與視覺化資料

`v_data/` 主要包含：

- 每一階段 beta-scaling 的參數（例如 `Post_beta_scale_arrayXXX.npy`）
- 用於生成動畫、圖表的視覺化資料
- `Animate_Structure.ipynb` 用於產生完整的優化過程動畫

也可能包含：

- `PICTURE/` 內含各階段結構 png
- 每一次 beta-scaling 完成後的中繼值（000、001、002...）

此資料夾可用於繪製：

- 優化過程動畫  
- 不同 beta 階段的結構變化  
- 條件收斂圖、E-field 計算比較

---

## 🔵 主程式 .py（優化程式入口）

所有資料夾中都包含一個主程式（例如 `MIN_17_12_3Dim_Ag_nearfocusing.py`），負責：

- 設定模擬區域、網格、材料
- 定義 FOM（1/2/3）
- 設定設計變數（MaterialGrid）
- 執行 forward + adjoint 計算
- 執行 filter + tanh projection
- 使用 NLopt 計算設計更新
- 將結果輸出到上述三個資料夾

整個專案的核心流程由此 py 檔啟動。






