"""
Lip-to-Chinese: record video + audio, generate multiple Chinese text candidates
via Whisper, select one and copy to clipboard.
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import tempfile

import numpy as np
import cv2
import sounddevice as sd
import soundfile as sf
from PIL import Image, ImageTk


# ── Face / mouth detection ────────────────────────────────────────────────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

SAMPLE_RATE = 16000
MAX_RECORD_SEC = 30
NUM_CANDIDATES = 5          # max candidates to show
WHISPER_MODEL_SIZE = "small"  # tiny / base / small / medium / large


def detect_face_mouth(frame_bgr):
    """Draw face bbox and estimated mouth region; return annotated frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        mx = x + w // 4
        my = y + int(h * 0.62)
        mw = w // 2
        mh = int(h * 0.32)
        cv2.rectangle(frame_bgr, (mx, my), (mx + mw, my + mh), (0, 165, 255), 2)
        cv2.putText(frame_bgr, "mouth", (mx, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    return frame_bgr, len(faces) > 0


# ── Whisper multi-candidate generation ───────────────────────────────────────

def get_chinese_candidates(audio_np, model, n=NUM_CANDIDATES):
    """
    Run Whisper multiple times with increasing temperatures to get diverse
    Chinese text candidates. Returns a deduplicated list of up to n strings.
    audio_np: float32 mono array at 16000 Hz
    """
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    candidates = []
    seen = set()

    for temp in temperatures:
        try:
            result = model.transcribe(
                audio_np,
                language="zh",
                temperature=temp,
                fp16=False,
                verbose=False,
            )
            text = result["text"].strip()
            for ch in ["…", "♪", "♫", "【", "】"]:
                text = text.replace(ch, "")
            text = text.strip()
            if text and text not in seen:
                candidates.append(text)
                seen.add(text)
        except Exception:
            pass
        if len(candidates) >= n:
            break

    return candidates if candidates else ["(No result — please try again)"]


# ── Main GUI application ──────────────────────────────────────────────────────

class LipToChineseApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Lip → Chinese Text Candidates")
        self.root.resizable(False, False)

        # 状态
        self.recording = False
        self.audio_chunks = []
        self.audio_stream = None
        self.record_start = 0.0
        self.candidates = []
        self.selected_var = tk.IntVar(value=0)
        self.whisper_model = None
        self.model_loading = False

        # webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self._build_ui()
        self._update_camera()
        self._load_whisper_async()

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.configure(bg="#1e1e2e")

        # ── Left: camera preview ──────────────────────────────────────────
        left = tk.Frame(self.root, bg="#1e1e2e")
        left.pack(side=tk.LEFT, padx=12, pady=12)

        self.cam_label = tk.Label(left, bg="black", width=480, height=360)
        self.cam_label.pack()

        self.face_status = tk.Label(left, text="● Detecting...", bg="#1e1e2e",
                                     fg="#888", font=("Arial", 10))
        self.face_status.pack(pady=4)

        # ── Right: control panel ──────────────────────────────────────────
        right = tk.Frame(self.root, bg="#1e1e2e", width=340)
        right.pack(side=tk.RIGHT, padx=12, pady=12, fill=tk.BOTH, expand=True)
        right.pack_propagate(False)

        tk.Label(right, text="Lip → Chinese Candidates", bg="#1e1e2e",
                 fg="white", font=("Arial", 15, "bold")).pack(pady=(0, 8))

        self.status_var = tk.StringVar(value="Loading speech model...")
        self.status_lbl = tk.Label(right, textvariable=self.status_var,
                                    bg="#1e1e2e", fg="#aaa", font=("Arial", 10),
                                    wraplength=320)
        self.status_lbl.pack()

        self.rec_btn = tk.Button(
            right, text="● Start Recording", font=("Arial", 14, "bold"),
            bg="#e05c5c", fg="white", activebackground="#c04444",
            relief=tk.FLAT, padx=16, pady=8, state=tk.DISABLED,
            command=self._toggle_recording,
        )
        self.rec_btn.pack(pady=12, fill=tk.X)

        self.time_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.time_var, bg="#1e1e2e",
                 fg="#e05c5c", font=("Arial", 11, "bold")).pack()

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=10)

        tk.Label(right, text="Candidates (click to select):", bg="#1e1e2e",
                 fg="#ccc", font=("Arial", 11, "bold")).pack(anchor=tk.W)

        self.cand_frame = tk.Frame(right, bg="#2a2a3e")
        self.cand_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        self.progress = ttk.Progressbar(right, mode="indeterminate")

        self.confirm_btn = tk.Button(
            right, text="✓ Confirm & Copy to Clipboard",
            font=("Arial", 11), bg="#5c9ede", fg="white",
            activebackground="#3a7ebf", relief=tk.FLAT,
            padx=8, pady=6, state=tk.DISABLED,
            command=self._confirm,
        )
        self.confirm_btn.pack(pady=8, fill=tk.X)

        tk.Label(right, text="Selected:", bg="#1e1e2e",
                 fg="#aaa", font=("Arial", 10)).pack(anchor=tk.W)
        self.result_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self.result_var, bg="#1e1e2e",
                 fg="#7ecb7e", font=("Arial", 13, "bold"),
                 wraplength=320, justify=tk.LEFT).pack(anchor=tk.W, pady=4)

    # ── 摄像头更新 ────────────────────────────────────────────────────────────

    def _update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame, has_face = detect_face_mouth(frame)

            if self.recording:
                r = int(time.time() * 3) % 2
                cv2.circle(frame, (16, 16), 9, (0, 0, 255) if r else (60, 60, 60), -1)
                cv2.putText(frame, "REC", (30, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            self.face_status.configure(
                text="● Face detected" if has_face else "○ No face detected",
                fg="#5adf5a" if has_face else "#df5a5a",
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((480, 360))
            photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=photo)
            self.cam_label.image = photo

        self.root.after(33, self._update_camera)

    # ── Whisper 异步加载 ──────────────────────────────────────────────────────

    def _load_whisper_async(self):
        self.model_loading = True

        def _load():
            import whisper
            self.status_var.set(f"Loading Whisper {WHISPER_MODEL_SIZE} model (first run downloads it)...")
            self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            self.root.after(0, self._on_model_ready)

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_ready(self):
        self.model_loading = False
        self.status_var.set("Ready — face the camera, speak Chinese, then record")
        self.rec_btn.configure(state=tk.NORMAL)

    # ── 录制控制 ──────────────────────────────────────────────────────────────

    def _toggle_recording(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self.recording = True
        self.audio_chunks = []
        self.record_start = time.time()

        self.rec_btn.configure(text="■ Stop Recording", bg="#888")
        self.confirm_btn.configure(state=tk.DISABLED)
        self.status_var.set("● Recording… click again to stop")
        self._clear_candidates()

        self.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._audio_cb,
        )
        self.audio_stream.start()
        self._tick_time()

    def _audio_cb(self, indata, frames, time_info, status):
        self.audio_chunks.append(indata.copy())

    def _tick_time(self):
        if not self.recording:
            return
        elapsed = time.time() - self.record_start
        self.time_var.set(f"⏱ {elapsed:.1f}s / {MAX_RECORD_SEC}s")
        if elapsed >= MAX_RECORD_SEC:
            self._stop_recording()
        else:
            self.root.after(100, self._tick_time)

    def _stop_recording(self):
        self.recording = False
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

        self.time_var.set("")
        self.rec_btn.configure(state=tk.DISABLED, text="● Start Recording", bg="#e05c5c")
        self.status_var.set("Recognising, please wait…")
        self.progress.pack(fill=tk.X, pady=4)
        self.progress.start(12)

        audio_np = np.concatenate(self.audio_chunks, axis=0).flatten()
        threading.Thread(target=self._run_recognition,
                         args=(audio_np,), daemon=True).start()

    # ── Whisper 推理 ──────────────────────────────────────────────────────────

    def _run_recognition(self, audio_np):
        try:
            candidates = get_chinese_candidates(audio_np, self.whisper_model)
        except Exception as e:
            candidates = [f"(Error: {e})"]
        self.root.after(0, lambda: self._show_candidates(candidates))

    def _show_candidates(self, candidates):
        self.progress.stop()
        self.progress.pack_forget()

        self.candidates = candidates
        self._clear_candidates()

        self.selected_var.set(0)
        for i, text in enumerate(candidates):
            row = tk.Frame(self.cand_frame, bg="#2a2a3e")
            row.pack(fill=tk.X, padx=6, pady=3)

            rb = tk.Radiobutton(
                row, variable=self.selected_var, value=i,
                bg="#2a2a3e", activebackground="#3a3a5e",
                selectcolor="#3a3a5e", fg="white",
            )
            rb.pack(side=tk.LEFT)

            lbl = tk.Label(
                row, text=f"{i+1}. {text}",
                bg="#2a2a3e", fg="#eee",
                font=("Arial", 12), wraplength=270,
                justify=tk.LEFT, anchor=tk.W, cursor="hand2",
            )
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            # 点击文字也能选中
            lbl.bind("<Button-1>", lambda e, idx=i: self.selected_var.set(idx))

        self.confirm_btn.configure(state=tk.NORMAL)
        self.rec_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Done — {len(candidates)} candidate(s). Select one and confirm.")

    def _clear_candidates(self):
        for w in self.cand_frame.winfo_children():
            w.destroy()
        self.result_var.set("—")
        self.confirm_btn.configure(state=tk.DISABLED)

    # ── 确认选择 ──────────────────────────────────────────────────────────────

    def _confirm(self):
        idx = self.selected_var.get()
        if 0 <= idx < len(self.candidates):
            text = self.candidates[idx]
            self.result_var.set(text)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set("✓ Selected and copied to clipboard")

    # ── 运行 & 关闭 ───────────────────────────────────────────────────────────

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.recording = False
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    app = LipToChineseApp()
    app.run()
