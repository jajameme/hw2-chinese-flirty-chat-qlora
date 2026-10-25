# 作業二實作計畫：土味情話多輪對話專家（QLoRA）

Oct 25, 2026 · @Ling hsuan

## 目標、模型與硬體限制

沿用老師 `cases/llm_finetune` 的 4-bit QLoRA 流程，只換掉資料來源與 system prompt，訓練出能接梗至少 2 輪的「土味情話」專家；先用 Qwen2.5-0.5B-Instruct 跑通流程，品質不夠再換 1.5B。

| 模型 | 在 6 GB 上 | 用途 |
| --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | 輕鬆，可不開 gradient checkpointing | 第一輪：驗證語料與流程 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 需開 gradient checkpointing、batch 1–2、長度 512 | 第二輪：提升幽默感與連貫性 |

**GTX 1660 Ti 限制**

- 6 GB VRAM：小 batch + gradient accumulation，必要時開 gradient checkpointing。
- 不支援 BF16：4-bit compute dtype 與 Trainer 混合精度都用 FP16（老師範例針對 T4 本來就是這樣設定）。
- bitsandbytes 4-bit NF4 支援 compute capability 7.5，可做 QLoRA。
- Qwen 偏向簡體輸出：system prompt 明寫繁體中文，輸出再用 OpenCC `s2twp` 轉換。

**假設**：Windows 10/11 + PowerShell；bitsandbytes 在 Windows 有問題時改用 WSL2。

## 環境：.venv

在 repo 資料夾內建立 `.venv`，先裝 GPU 版 PyTorch，再裝老師 `requirements-colab.txt` 的套件版本。

1. 更新 NVIDIA 驅動（`nvidia-smi` 的 CUDA Version ≥ 12.8），安裝 Python 3.12。
2. 建立並啟用：

   ```powershell
   cd hw2_flirty_chat
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1     # Linux/WSL: source .venv/bin/activate
   python -m pip install -U pip
   ```
3. 安裝 PyTorch（必須帶 `--index-url`）：

   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
4. 安裝訓練套件：

   ```powershell
   pip install transformers==5.17.0 trl==1.13.0 peft==0.21.0 bitsandbytes==0.50.2 datasets==5.0.1 accelerate==1.15.0 matplotlib OpenCC==1.4.2
   ```
5. `check_gpu.py`：確認 `torch.cuda.is_available()` 為 True、compute capability 為 (7, 5)，並用 `BitsAndBytesConfig(load_in_4bit=True)` 載入 0.5B 模型試跑一次，確認 bitsandbytes 正常。
6. 若 PEFT 匯入時報 torchao 版本錯誤：`pip uninstall -y torchao`（老師 README 的作法，本專案不使用 torchao）。
7. 完成後 `pip freeze > requirements.txt`；`.gitignore` 加入 `.venv/`。

## 語料建置

語料品質決定成果，目標是 300–500 段、每段 2–5 輪的多輪對話，由 AI 分批產生再人工篩選，最後轉成老師範例吃的 `{"messages": [...]}` JSONL。

1. 定 15–20 個情境：搭訕、告白、約會、道歉、吃醋、早安晚安、下雨天、節日、考試、職場、被拒絕後救援、遠距離等。
2. 用 ChatGPT / Claude / Gemini / Grok 分批產生：每次指定一個情境與輸出 JSON 格式，要求「使用者追問、專家接梗」的結構（模仿作業範例的錄取通知梗），每次 20–30 段。
3. 人工篩選：刪重複梗、冒犯或物化對方的內容、與情境無關的回答。
4. `build_corpus.py`（取代 `download_chat_data.py`）
   - 合併 `raw/*.json`。
   - 驗證 role 依序為 system → user → assistant 交替，最後一則是 assistant。
   - 以 user 第一句去重；全文用 OpenCC `s2twp` 轉繁體。
   - 每段開頭插入固定 system prompt，輸出 `data/chat.jsonl`。
5. 直接沿用 `prepare_chat_data.py`：它把每個 assistant 回覆展開成一筆 prompt/completion（`MAX_CONTEXT_MESSAGES=6`），切成 train 90% / valid 10%。
6. 在 README 記錄：情境清單、產生工具、原始筆數、篩選後筆數、展開後訓練樣本數。

**system prompt（訓練與推論必須完全相同）**

```text
你是一個土味情話高手，會用親切、溫和、帶點幽默的口吻，以繁體中文回答使用者，並且能接住使用者的追問繼續接梗。
```

**一筆語料範例**

```json
{"messages": [
  {"role": "system", "content": "你是一個土味情話高手……"},
  {"role": "user", "content": "教我如何透過長相來認識女性朋友。"},
  {"role": "assistant", "content": "你就跟她說「你長得很像我一個朋友」。"},
  {"role": "user", "content": "長得像你朋友？"},
  {"role": "assistant", "content": "沒錯，跟她說「你長得像我下一任女朋友」。"}
]}
```

## 檔案結構與訓練參數

程式幾乎全部沿用老師範例，主要改 `train_qlora.py` 上方的常數，把 T4 16 GB 的設定縮到 6 GB。

```text
hw2_flirty_chat/
├── .venv/                  # 不提交
├── check_gpu.py
├── raw/                    # AI 產生、人工篩選後的原始 JSON
├── build_corpus.py         # → data/chat.jsonl
├── prepare_chat_data.py    # → data/sft/train.jsonl、valid.jsonl（沿用）
├── train_qlora.py          # → outputs/flirty_lora_05b/（改參數）
├── plot_loss.py            # → images/loss_curve.png（沿用）
├── infer_lora_chat.py      # 多輪互動（改 system prompt）
├── compare_base.py         # 同題比較原始模型 vs 微調後
├── images/
├── requirements.txt
└── README.md
```

| 參數 | 原範例（T4 16 GB） | 1660 Ti 6 GB | 理由 |
| --- | --- | --- | --- |
| `MODEL_ID` | 0.5B | 0.5B → 1.5B | 先跑通再求品質 |
| `DATA_DIR` | `data/sft` | 不變 | 由 prepare\_chat\_data.py 產生 |
| `OUTPUT_DIR` | `outputs/qwen25_05b_lora` | `outputs/flirty_lora_05b` / `_15b` | 區分版本 |
| `MAX_LENGTH` | 768 | 512 | 對話短，省 VRAM |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | 8 | 4（0.5B）/ 1–2（1.5B） | 6 GB |
| `PER_DEVICE_EVAL_BATCH_SIZE` | 16 | 4 | 6 GB |
| `GRADIENT_ACCUMULATION_STEPS` | 16 | 4（0.5B）/ 8（1.5B） | 等效 batch 約 16；資料少，步數不能太少 |
| `GRADIENT_CHECKPOINTING` | False | 1.5B 時 True | 用時間換記憶體 |
| `EPOCHS` | 5 | 3–5，看 eval loss | 小資料易過擬合 |
| `CHECK_STEPS` | 50 | 10–20 | 總步數只有數十到上百 |
| `LORA_R` / `LORA_ALPHA` | 8 / 16 | 8/16，不夠再試 16/32 | 學語氣需要多一點容量 |
| 4-bit compute dtype / AMP | FP16 | 不變 | 1660 Ti 同樣無 BF16 |
| `OPTIM` | paged\_adamw\_8bit | 不變 | 省 optimizer 記憶體 |
| `COMPLETION_ONLY_LOSS` / `EOS_TOKEN` | True / Qwen 的 im\_end | 不變 | 只學 assistant 回覆；同為 Qwen 模板 |

執行順序：

```powershell
python check_gpu.py
python build_corpus.py
python prepare_chat_data.py
python train_qlora.py
python plot_loss.py
python infer_lora_chat.py
```

## 推論展示、驗收與繳交

展示的重點是「能延續上一輪的梗」，所以至少錄一段 2 輪以上（建議照作業範例做到 5 輪）的對話，並和原始模型對比。

**推論**

1. `infer_lora_chat.py`：改 `ADAPTER_DIR`、`SYSTEM_PROMPT`（與訓練一致）；`TEMPERATURE` 0.7–0.9、`TOP_P` 0.9，保留 `MAX_HISTORY_TURNS` 對話歷史；輸出經 OpenCC 轉繁體。
2. `compare_base.py`：同一組 5 個開場問題，分別問原始模型與微調模型，輸出對照表放進 README。
3. 錄對話截圖或影片；`plot_loss.py` 的 loss 曲線放進 README。
4. LoRA adapter 通常只有數十 MB，可直接 push；超過 100 MB 才需要 Git LFS。

**常見問題**

| 症狀 | 原因 | 處理 |
| --- | --- | --- |
| `CUDA out of memory` | batch 或長度太大 | batch 減半、累積加倍、`MAX_LENGTH` 384、開 gradient checkpointing |
| bitsandbytes 匯入失敗 | Windows 環境不相容 | 升級 bitsandbytes；仍不行改 WSL2 |
| torchao 版本錯誤 | 環境有舊版 torchao | `pip uninstall -y torchao` |
| loss 變 `nan` | FP16 溢位 | 降學習率到 5e-5；確認 LoRA 參數維持 FP32 |
| 回答忘記上一輪 | 推論沒帶歷史，或語料多為單輪 | 檢查 history 組裝；補更多 3 輪以上語料 |
| 一直重複同一句梗 | 語料重複或過擬合 | 去重、減 epoch、提高 temperature |
| 回答簡體字 | Qwen 預設偏簡體 | system prompt 指定繁體 + OpenCC `s2twp` |

**GitHub repo**

- 建在你自己的 GitHub 帳號下，獨立一個 repo（建議名稱 `hw2_flirty_chat`），設為 **Public**，老師可直接開啟連結。
- 不 fork 老師的 `telunyang/python_nlp`，只複製 `cases/llm_finetune` 需要改寫的檔案，README 註明改寫出處。
- GitHub 帳號與最終 repo 名稱：實作建立 repo 時再向你確認。

**繳交檢查**

- [ ] `.gitignore` 排除 `.venv/`、`checkpoint-*`、`__pycache__/`
- [ ] 語料檔與 README 的筆數統計一致
- [ ] 至少 2 輪的對話展示（截圖或影片）與 loss 曲線
- [ ] README：基礎模型、語料來源與產生方式、資料筆數、訓練參數、套件版本、執行方法、成果
- [ ] 照 README 在新的 `.venv` 從頭跑一次確認可重現
- [ ] repo 連結寄給老師，信件標題含班別與姓名

**參考來源**：[telunyang/python\_nlp](https://github.com/telunyang/python_nlp)（`cases/llm_finetune`）、[PyTorch Get Started](https://pytorch.org/get-started/locally/)
