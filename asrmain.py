import sys
import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ================= 參數設定 =================
MODEL_SIZE = "tiny"        # 預設使用 tiny 模型
LANGUAGE = "zh"            # 預設語言：中文
SAMPLE_RATE = 16000        # Whisper 要求的取樣率 (16kHz)
BLOCK_DURATION = 2         # 每次截取的音訊長度 (預設 2 秒)
# ============================================

# 建立音訊緩衝佇列
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Sounddevice 錄音回呼函式，將音訊資料不間斷地放入佇列"""
    if status:
        print(status, file=sys.stderr)
    # indata 是包含音訊特徵的 NumPy 陣列
    audio_queue.put(indata.copy())

def main():
    print(f"[*] 正在載入 Whisper ({MODEL_SIZE}) 模型...")
    
    # 初始化 Faster Whisper 模型
    # device="auto" 會自動偵測並優先調用 NVIDIA GPU (CUDA)，若無則退回 CPU
    # compute_type="default" 確保在不同硬體上的推論相容性
    model = WhisperModel(MODEL_SIZE, device="auto", compute_type="default")
    
    print("[*] 模型載入完成！開始即時收音辨識... (按 Ctrl+C 停止)\n")

    # 設定音訊輸入串流
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,                     # 強制單聲道
        dtype=np.float32,               # Whisper 接受 float32 格式
        blocksize=SAMPLE_RATE * BLOCK_DURATION,
        callback=audio_callback
    )

    with stream:
        try:
            while True:
                # 從佇列中取得音訊區塊 (會阻塞等待直到有新資料)
                chunk = audio_queue.get()
                audio_data = chunk.flatten()

                # 將音訊交給模型進行推論
                # 啟用 vad_filter=True 是一種最佳實踐，可過濾掉環境底噪，降低運算負載
                segments, info = model.transcribe(
                    audio_data,
                    beam_size=5,
                    language=LANGUAGE,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                # 輸出辨識結果
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        # 印出時間軸與辨識文字
                        print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {text}")
                        
        except KeyboardInterrupt:
            print("\n[*] 結束即時語音辨識。")

if __name__ == "__main__":
    main()