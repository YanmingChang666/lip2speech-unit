"""
Real-time lip-to-speech using webcam.
Pipeline: webcam -> OpenCV face detect -> mouth ROI -> lip2speech model -> vocoder -> audio
"""

import sys
import os

# model.py uses `DBG = len(sys.argv)==1` to skip imports; ensure it's False
_ADDED_DUMMY_ARG = False
if len(sys.argv) == 1:
    sys.argv.append('_run_')
    _ADDED_DUMMY_ARG = True

import time
import threading
import queue
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import sounddevice as sd
from argparse import Namespace

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{ROOT}/fairseq')
sys.path.insert(0, ROOT)

# ── config ────────────────────────────────────────────────────────────────────
LIP2SPEECH_CKPT = f'{ROOT}/checkpoints/lip2speech_lrs3_avhubert_multi.pt'
AVHUBERT_CKPT   = f'{ROOT}/checkpoints/large_vox_iter5.pt'
VOCODER_CKPT    = f'{ROOT}/checkpoints/vocoder_lrs3_aug_multi.pt'
VOCODER_CFG     = f'{ROOT}/checkpoints/config.json'
SPK_EMB_DIR     = f'{ROOT}/datasets/lrs3/spk_emb/test'

CHUNK_FRAMES = 50        # frames per inference chunk (2s at 25fps)
WEBCAM_FPS   = 30        # webcam capture rate
MODEL_FPS    = 25        # model expects 25fps mouth video
CROP_SIZE    = 96        # mouth ROI size before center-crop
MODEL_CROP   = 88        # center-crop size fed to model
IMAGE_MEAN   = 0.421
IMAGE_STD    = 0.165
SAMPLE_RATE  = 16000
# ──────────────────────────────────────────────────────────────────────────────


# ── Face / mouth detection (OpenCV Haar cascade) ──────────────────────────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def extract_mouth_roi(frame_bgr, crop=CROP_SIZE):
    """
    Detect face with Haar cascade, estimate mouth position from bbox.
    Returns (crop x crop) grayscale uint8 array, or None if no face found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None

    # Largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    # Mouth center ≈ 78% down the face, horizontally centered
    cx = x + w // 2
    cy = y + int(h * 0.78)
    half = int(h * 0.22)   # crop half-size relative to face height

    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    img_h, img_w = frame_bgr.shape[:2]
    pad = max(0, -x1, -y1, x2 - img_w, y2 - img_h)
    if pad > 0:
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x1 += pad; y1 += pad; x2 += pad; y2 += pad

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.resize(roi, (crop, crop))


def center_crop(frames, size=MODEL_CROP):
    """frames: (T, H, W) -> (T, size, size)"""
    h, w = frames.shape[1], frames.shape[2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return frames[:, y0:y0+size, x0:x0+size]


def preprocess_frames(frames_gray):
    """
    frames_gray: list of (H, W) uint8 arrays
    Returns: float32 tensor (1, 1, T, 88, 88) ready for model
    """
    arr = np.stack(frames_gray, axis=0).astype(np.float32)  # (T, H, W)
    arr = arr / 255.0
    arr = center_crop(arr, MODEL_CROP)
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    # (T, H, W) -> (T, H, W, 1) -> (1, 1, T, H, W)
    arr = arr[:, :, :, np.newaxis]
    tensor = torch.from_numpy(arr).float()
    tensor = tensor.unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous()
    return tensor


# ── Model loading ─────────────────────────────────────────────────────────────

def load_speaker_embedding(spk_emb_dir):
    """Average all available speaker embeddings."""
    embs = []
    for root, _, files in os.walk(spk_emb_dir):
        for f in files:
            if f.endswith('.npy'):
                embs.append(np.load(os.path.join(root, f)))
    if not embs:
        raise FileNotFoundError(f"No .npy speaker embeddings found in {spk_emb_dir}")
    emb = np.stack(embs).mean(axis=0)
    print(f"Loaded speaker embedding from {len(embs)} files, shape={emb.shape}")
    return emb


def load_lip2speech_model(ckpt_path, avhubert_path, user_dir):
    from fairseq import checkpoint_utils, tasks, utils
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
    from omegaconf import OmegaConf

    utils.import_user_module(Namespace(user_dir=user_dir))

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [ckpt_path],
        arg_overrides={"w2v_path": avhubert_path}
    )
    model = models[0].eval().cuda()
    saved_cfg.task.modalities = ['video']
    task = tasks.setup_task(saved_cfg.task)

    from multi_target_lip2speech.sequence_generator import MultiTargetSequenceGenerator
    from fairseq.dataclass.configs import GenerationConfig
    gen_cfg = GenerationConfig()
    generator = task.build_generator(models, gen_cfg)
    return model, models, generator, task


def load_vocoder(ckpt_path, cfg_path):
    sys.path.insert(0, f'{ROOT}/multi_input_vocoder')
    sys.path.insert(0, f'{ROOT}/speech-resynthesis')
    from models_multi_input import MelCodeGenerator
    sys.path.pop(0)
    sys.path.pop(0)

    with open(cfg_path) as f:
        h = json.load(f)

    class AttrDict(dict):
        def __getattr__(self, key): return self[key]
        def get(self, key, default=None): return super().get(key, default)

    h = AttrDict(h)
    vocoder = MelCodeGenerator(h).cuda()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    vocoder.load_state_dict(ckpt['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    print("Vocoder loaded.")
    return vocoder, h


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_lip2speech(frames_gray, models, generator, task, spk_emb_np):
    """
    frames_gray: list of (H, W) uint8 grayscale mouth ROIs (already at CROP_SIZE)
    Returns: (units_str, mel_np) or None on failure
    """
    if len(frames_gray) < 5:
        return None

    video_tensor = preprocess_frames(frames_gray).cuda()  # (1,1,T,88,88)
    T = video_tensor.shape[2]

    padding_mask = torch.zeros(1, T, dtype=torch.bool).cuda()
    spk_emb = torch.from_numpy(spk_emb_np).float().unsqueeze(0).cuda()

    # Dummy target for mel truncation (model uses it only for length)
    target = torch.zeros(1, T, dtype=torch.long).cuda()
    target_lengths = torch.LongTensor([T]).cuda()

    sample = {
        "id": torch.LongTensor([0]),
        "net_input": {
            "source": {"audio": None, "video": video_tensor},
            "padding_mask": padding_mask,
            "spk_emb": spk_emb,
        },
        "target": target,
        "target_lengths": target_lengths,
        "ntokens": T,
        "utt_id": ["realtime"],
    }

    try:
        hypos, sample = task.inference_step(generator, models, sample)
    except Exception as e:
        print(f"[lip2speech error] {e}")
        return None

    if not hypos or "mels" not in sample:
        return None

    dictionary = task.target_dictionary
    symbols_ignore = {dictionary.pad(), dictionary.eos(), dictionary.bos()}
    # dictionary.string() already returns "14 131 42 ..." for unit tokens
    units_str = dictionary.string(hypos[0][0]["tokens"].int().cpu(),
                                  extra_symbols_to_ignore=symbols_ignore).strip()

    mel_np = sample["mels"][0]  # (T_mel, 80)
    return units_str, mel_np


@torch.no_grad()
def run_vocoder(units_str, mel_np, vocoder_model, spk_emb_np):
    """
    units_str: space-separated integer string, e.g. "14 14 131 ..."
    mel_np: (T_mel, 80) numpy array
    Returns: (T_audio,) int16 numpy audio array
    """
    units = [int(v) for v in units_str.split() if v.lstrip('-').isdigit()]
    units = [u for u in units if 0 <= u < 200]   # valid unit range for KM200
    if not units:
        return None

    code = torch.LongTensor(units).unsqueeze(0).cuda()       # (1, T_code)
    mel  = torch.from_numpy(mel_np.T).float().unsqueeze(0).cuda()  # (1, 80, T_mel)
    spkr = torch.from_numpy(spk_emb_np).float().unsqueeze(0).cuda()  # (1, 256)

    audio = vocoder_model(code=code, mel=mel, spkr=spkr)
    if isinstance(audio, tuple):
        audio = audio[0]

    audio = audio.squeeze().cpu().numpy()
    audio = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
    return audio


# ── Main real-time loop ───────────────────────────────────────────────────────

def run_inference_async(chunk, models, generator, task, spk_emb,
                        vocoder_model, audio_queue):
    """Run lip2speech + vocoder in a background thread."""
    t0 = time.time()
    result = run_lip2speech(chunk, models, generator, task, spk_emb)
    if result is None:
        print("[Inference] No output.")
        return
    units_str, mel_np = result
    audio = run_vocoder(units_str, mel_np, vocoder_model, spk_emb)
    if audio is not None and len(audio) > 0:
        elapsed = time.time() - t0
        print(f"[Inference] {len(chunk)} frames → {len(units_str.split())} units "
              f"| audio={len(audio)/SAMPLE_RATE:.2f}s | latency={elapsed:.2f}s")
        audio_queue.put(audio)
    else:
        print("[Inference] Vocoder produced no audio.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['auto', 'record'], default='record',
                        help='auto: infer every 2s continuously; record: press SPACE to record')
    # Strip the dummy arg added at top if needed
    parse_args = [a for a in sys.argv[1:] if a != '_run_']
    args = parser.parse_args(parse_args)

    print("Loading models (this may take ~30s)...")
    spk_emb = load_speaker_embedding(SPK_EMB_DIR)

    user_dir = f'{ROOT}/multi_target_lip2speech'
    model, models, generator, task = load_lip2speech_model(
        LIP2SPEECH_CKPT, AVHUBERT_CKPT, user_dir
    )
    print("Lip2Speech model loaded.")

    vocoder_model, vocoder_cfg = load_vocoder(VOCODER_CKPT, VOCODER_CFG)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    cap.set(cv2.CAP_PROP_FPS, WEBCAM_FPS)

    sample_every = max(1, round(WEBCAM_FPS / MODEL_FPS))

    if args.mode == 'record':
        print("\n[Record mode]  SPACE = start/stop recording  |  Q = quit\n")
    else:
        print(f"\n[Auto mode]  Infer every {CHUNK_FRAMES/MODEL_FPS:.1f}s  |  Q = quit\n")

    audio_queue = queue.Queue()

    def audio_player():
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            sd.play(audio.astype(np.float32) / 32768.0, SAMPLE_RATE)
            sd.wait()

    player_thread = threading.Thread(target=audio_player, daemon=True)
    player_thread.start()

    frame_buf = []
    webcam_frame_cnt = 0
    recording = False          # record-mode state
    infer_thread = None        # background inference thread

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        webcam_frame_cnt += 1

        # ── Key handling (check every frame for responsiveness) ──────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and args.mode == 'record':
            if not recording:
                # Start recording
                recording = True
                frame_buf.clear()
                print("[●] Recording started...")
            else:
                # Stop and infer
                recording = False
                chunk = list(frame_buf)
                frame_buf.clear()
                if len(chunk) < 5:
                    print("[!] Too short, skip.")
                else:
                    print(f"[■] Stopped. {len(chunk)} frames ({len(chunk)/MODEL_FPS:.1f}s). Running inference...")
                    if infer_thread and infer_thread.is_alive():
                        print("[!] Previous inference still running, waiting...")
                        infer_thread.join()
                    infer_thread = threading.Thread(
                        target=run_inference_async,
                        args=(chunk, models, generator, task, spk_emb,
                              vocoder_model, audio_queue),
                        daemon=True
                    )
                    infer_thread.start()

        # ── Frame sampling ────────────────────────────────────────────────────
        if webcam_frame_cnt % sample_every != 0:
            cv2.imshow('Lip2Speech', frame)
            continue

        roi = extract_mouth_roi(frame)

        # ── Buffer frames ─────────────────────────────────────────────────────
        should_buffer = (args.mode == 'auto') or (args.mode == 'record' and recording)
        if roi is not None and should_buffer:
            frame_buf.append(roi)

        # ── Build display frame ───────────────────────────────────────────────
        disp = frame.copy()
        face_ok = roi is not None
        face_color = (0, 255, 0) if face_ok else (0, 0, 255)
        cv2.putText(disp, "Face OK" if face_ok else "No face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

        if args.mode == 'record':
            if recording:
                # Blinking red REC indicator
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(disp, (disp.shape[1] - 30, 25), 12, (0, 0, 255), -1)
                cv2.putText(disp, f"REC  {len(frame_buf)/MODEL_FPS:.1f}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif infer_thread and infer_thread.is_alive():
                cv2.putText(disp, "Processing...", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            else:
                cv2.putText(disp, "SPACE to record", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        else:
            cv2.putText(disp, f"Buffer: {len(frame_buf)}/{CHUNK_FRAMES}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Mouth ROI inset (top-right)
        if roi is not None:
            roi_rgb = cv2.cvtColor(cv2.resize(roi, (96, 96)), cv2.COLOR_GRAY2BGR)
            disp[10:106, -106:-10] = roi_rgb

        cv2.imshow('Lip2Speech', disp)

        # ── Auto-mode: infer when buffer full ─────────────────────────────────
        if args.mode == 'auto' and len(frame_buf) >= CHUNK_FRAMES:
            chunk = list(frame_buf)
            frame_buf.clear()
            if infer_thread is None or not infer_thread.is_alive():
                infer_thread = threading.Thread(
                    target=run_inference_async,
                    args=(chunk, models, generator, task, spk_emb,
                          vocoder_model, audio_queue),
                    daemon=True
                )
                infer_thread.start()

    # Cleanup
    if infer_thread and infer_thread.is_alive():
        infer_thread.join(timeout=30)
    audio_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    player_thread.join(timeout=3)
    print("Done.")


if __name__ == '__main__':
    main()
