import sys
import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import requests
import tkinter as tk
from tkinter import ttk, scrolledtext

# ================= 參數設定 =================
WHISPER_MODEL = "tiny"  # 可選擇 "tiny", "base", "small", "medium", "large-v2", "turbo" 等模型
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"  # 請確認您的 Ollama 已經 pull 了這個模型，或改為您本機有的模型
SAMPLE_RATE = 16000
BLOCK_DURATION = 2
# ============================================

# 建立通訊佇列
audio_queue = queue.Queue()
text_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """將麥克風音訊放入佇列"""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def whisper_worker():
    """負責執行 Faster-Whisper 的背景執行緒"""
    print(f"[*] 載入 Whisper ({WHISPER_MODEL})...")
    model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="default")
    print("[*] Whisper 準備就緒")
    
    while True:
        chunk = audio_queue.get()
        if chunk is None: break  # 毒丸機制 (Poison Pill) 結束執行緒
        audio_data = chunk.flatten()

        # 進行語音辨識，啟用 VAD 過濾底噪
        segments, _ = model.transcribe(
            audio_data,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        for segment in segments:
            text = segment.text.strip()
            if text:
                text_queue.put(text)

def ollama_worker(app_gui):
    """負責呼叫 Ollama API 進行翻譯的背景執行緒"""
    while True:
        original_text = text_queue.get()
        if original_text is None: break
        
        mode = app_gui.get_translation_mode()
        prompt = build_prompt(original_text, mode)

        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=10
            )
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                # 透過 Tkinter 的 after 方法安全地更新主執行緒的 GUI
                app_gui.root.after(0, app_gui.update_text, original_text, result)
            else:
                print(f"[!] Ollama API 錯誤: {response.status_code}")
        except Exception as e:
            print(f"[!] 無法連線至 Ollama: {e}")

def build_prompt(text, mode):
    """根據選擇的模式構建 Ollama 的 System Prompt"""
    base_instruction = "You are a professional realtime translator. Output ONLY the translation or the original text as instructed, without any conversational filler or quotes.\n\n"
    
    if mode == "1":
        rule = "Target Audience: Taiwanese. Rule: If the text is English, translate it to Traditional Chinese (zh-TW). If the text is already Chinese, output it exactly as is."
    elif mode == "2":
        rule = "Target Audience: Foreigners. Rule: If the text is Chinese, translate it to English. If the text is already English, output it exactly as is."
    elif mode == "3":
        rule = "Target Audience: Everyone. Rule: If the text is Chinese, translate it to English. If the text is English, translate it to Traditional Chinese (zh-TW). Output ONLY the translated text."
    else:
        rule = "Output the text exactly as is."

    return f"{base_instruction}{rule}\n\nText to process: {text}"

class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("即時 AI 語音翻譯字幕 (Whisper + Ollama)")
        self.root.geometry("900x650")
        self.root.configure(bg="#222222")
        
        self.font_large = ("Helvetica", 24, "bold")
        self.font_medium = ("Helvetica", 16)

        # 頂部控制面板
        control_frame = tk.Frame(root, bg="#333333")
        control_frame.pack(pady=10, fill=tk.X, padx=20)

        tk.Label(control_frame, text=" 翻譯模式：", font=self.font_medium, bg="#333333", fg="white").pack(side=tk.LEFT, pady=10)

        self.mode_var = tk.StringVar(value="1")
        modes = [
            ("1. 給台灣人看 (英翻中，中保留)", "1"),
            ("2. 給外國人看 (中翻英，英保留)", "2"),
            ("3. 給所有人看 (中翻英，英翻中)", "3")
        ]
        
        self.mode_combo = ttk.Combobox(control_frame, textvariable=self.mode_var, values=[m[0] for m in modes], state="readonly", font=self.font_medium, width=35)
        self.mode_combo.pack(side=tk.LEFT, padx=10)
        self.mode_combo.current(0)

        # 字幕顯示區塊
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg="black", fg="white")
        self.text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # --- 視覺層級標籤設定 ---
        self.text_area.tag_config("label_asr", foreground="#888888", font=self.font_medium)  # 灰色標籤
        self.text_area.tag_config("text_asr", foreground="#AAAAAA", font=self.font_medium)    # 淺灰原文
        self.text_area.tag_config("label_ai", foreground="#00FF00", font=self.font_medium)   # 亮綠色 AI 標籤
        self.text_area.tag_config("text_ai", foreground="#FFFFFF", font=self.font_large)     # 白色大字體翻譯
        self.text_area.tag_config("text_skip", foreground="#666666", font=self.font_medium)  # 暗灰色跳過提示
        
        self.text_area.insert(tk.END, "等待語音輸入中...\n\n", "label_asr")

    def get_translation_mode(self):
        return self.mode_var.get().split(".")[0]

    def update_text(self, original, translated):
        """將原文與翻譯結果分層、分色顯示"""
        # 1. 顯示 Whisper 辨識的原文
        self.text_area.insert(tk.END, "[語音] ", "label_asr")
        self.text_area.insert(tk.END, f"{original}\n", "text_asr")

        # 2. 顯示 Ollama 的處理結果
        # 判斷是否無須翻譯 (結果相同，或是空字串)
        if original.strip() == translated.strip() or not translated.strip():
            self.text_area.insert(tk.END, " ↳ [AI] ", "label_asr")
            self.text_area.insert(tk.END, "(符合目標語言，維持原文)\n\n", "text_skip")
        else:
            self.text_area.insert(tk.END, " ↳ [翻譯] ", "label_ai")
            self.text_area.insert(tk.END, f"{translated}\n\n", "text_ai")

        # 自動捲動到最底
        self.text_area.see(tk.END)

def main():
    root = tk.Tk()
    app = TranslatorGUI(root)

    # 啟動背景執行緒
    threading.Thread(target=whisper_worker, daemon=True).start()
    threading.Thread(target=ollama_worker, args=(app,), daemon=True).start()

    # 啟動麥克風收音
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        blocksize=SAMPLE_RATE * BLOCK_DURATION,
        callback=audio_callback
    )
    
    with stream:
        root.mainloop()
        
    # 關閉視窗後的清理
    audio_queue.put(None)
    text_queue.put(None)
    print("\n[*] 系統已關閉。")

if __name__ == "__main__":
    main()