"""
Shared foundation for the pronunciation-teaching / lip-mapping system.

Used by:
  pronunciation_curriculum.py  - curriculum data + TTS cache generation
  record_pronunciations.py     - teach -> countdown -> record 3 takes per item
  train_pronunciation_model.py - AV-HuBERT features + classifier head training
  realtime_pronunciation.py    - webcam lip movement -> pronunciation -> TTS audio

Data layout (fixed spec, all paths relative to repo root):
  datasets/pronunciation/tts/{item_id}.wav          16 kHz mono teaching audio
  datasets/pronunciation/recordings/{item_id}/take{k}.npy   (T,96,96) uint8 mouth ROI @25fps
  datasets/pronunciation/recordings/{item_id}/take{k}.json  per-take metadata
  datasets/pronunciation/recordings/_silence/take{k}.npy    idle-face calibration clips
  datasets/pronunciation/features/{item_id}/take{k}_aug{j}.npy  (T,1024) float32 AV-HuBERT features
  datasets/pronunciation/session.json               recording progress + calibration
  checkpoints/pronunciation_head.pt                 trained classifier:
      {"state_dict", "label_map": [item_id,...], "config": {...},
       "feat_mean": (1024,), "feat_std": (1024,)}
"""

import sys
import os

# avhubert modules use `DBG = len(sys.argv)==1` to switch to debug-imports;
# make sure it is False before any avhubert import (same hack as realtime_lip2speech.py)
if len(sys.argv) == 1:
    sys.argv.append('_run_')

import json
import glob
import time

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{ROOT}/fairseq')
sys.path.insert(0, ROOT)

# ── config ────────────────────────────────────────────────────────────────────
AVHUBERT_CKPT = f'{ROOT}/checkpoints/large_vox_iter5.pt'
HEAD_CKPT     = f'{ROOT}/checkpoints/pronunciation_head.pt'

PRON_DIR     = f'{ROOT}/datasets/pronunciation'
TTS_DIR      = f'{PRON_DIR}/tts'
REC_DIR      = f'{PRON_DIR}/recordings'
FEAT_DIR     = f'{PRON_DIR}/features'
SESSION_FILE = f'{PRON_DIR}/session.json'

SILENCE_ID = '_silence'     # extra class: idle face / not speaking

WEBCAM_FPS  = 30            # webcam capture rate
MODEL_FPS   = 25            # model expects 25 fps mouth video
CROP_SIZE   = 96            # mouth ROI size stored on disk
MODEL_CROP  = 88            # crop fed to AV-HuBERT
IMAGE_MEAN  = 0.421
IMAGE_STD   = 0.165
SAMPLE_RATE = 16000

RECORD_SECONDS = 2.0        # length of one take
TAKES_PER_ITEM = 3
# ──────────────────────────────────────────────────────────────────────────────


def ensure_dirs():
    for d in (PRON_DIR, TTS_DIR, REC_DIR, FEAT_DIR):
        os.makedirs(d, exist_ok=True)


# ── Face / mouth detection (OpenCV Haar cascade, same as realtime_lip2speech) ─
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

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    # Mouth center ~78% down the face, horizontally centered
    cx = x + w // 2
    cy = y + int(h * 0.78)
    half = int(h * 0.22)

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


# ── Quality metrics ───────────────────────────────────────────────────────────

def movement_energy(frames_gray):
    """Mean absolute inter-frame difference. frames_gray: (T,H,W) uint8 or list."""
    arr = np.asarray(frames_gray, dtype=np.float32)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(arr, axis=0))))


def brightness_stats(frame_bgr):
    """Return (mean, std) of grayscale brightness, 0-255."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


class FrameSampler:
    """
    Decimate a webcam stream to MODEL_FPS by fractional-stride accumulation.
    (An integer stride cannot express 30->25: round(30/25)=1 keeps every frame;
    this keeps exactly 25 of every 30.) Call take() once per webcam frame.
    """

    def __init__(self, src_fps, dst_fps=MODEL_FPS):
        src = float(src_fps) if src_fps and 5.0 <= src_fps <= 120.0 else float(WEBCAM_FPS)
        self.ratio = min(1.0, float(dst_fps) / src)
        self.acc = 0.0

    def take(self):
        self.acc += self.ratio
        if self.acc >= 1.0:
            self.acc -= 1.0
            return True
        return False


# ── CJK text overlay (cv2.putText cannot render Chinese) ─────────────────────

_FONT_CANDIDATES = [
    os.environ.get('PRON_CJK_FONT', ''),
    # Linux
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    # Windows
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/simsun.ttc',
    # macOS
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
]

_cjk_font_path = None
_font_cache = {}


def find_cjk_font():
    """Locate a CJK-capable TTF/TTC font. Returns path or None."""
    global _cjk_font_path
    if _cjk_font_path is not None:
        return _cjk_font_path or None
    for p in _FONT_CANDIDATES:
        if p and os.path.isfile(p):
            _cjk_font_path = p
            return p
    for pattern in ('/usr/share/fonts/**/*CJK*', '/usr/share/fonts/**/*cjk*'):
        hits = [h for h in glob.glob(pattern, recursive=True)
                if os.path.isfile(h)
                and h.lower().endswith(('.ttf', '.ttc', '.otf'))]
        if hits:
            _cjk_font_path = hits[0]
            return hits[0]
    _cjk_font_path = ''       # searched, none found
    return None


_text_bitmap_cache = {}          # (font_path, px, text) -> float32 alpha (H,W)
_TEXT_CACHE_MAX = 512


def _text_alpha_bitmap(font_path, px, text):
    """Render text once to a small grayscale alpha bitmap (cached)."""
    key = (font_path, px, text)
    hit = _text_bitmap_cache.get(key)
    if hit is not None:
        return hit
    from PIL import Image, ImageDraw, ImageFont
    fkey = (font_path, px)
    if fkey not in _font_cache:
        _font_cache[fkey] = ImageFont.truetype(font_path, px)
    font = _font_cache[fkey]
    ascent, descent = font.getmetrics()
    th = ascent + descent
    try:
        tw = int(font.getlength(text))            # PIL >= 8.0
    except AttributeError:
        tw = font.getsize(text)[0]
    canvas = Image.new('L', (max(tw, 1), max(th, 1)), 0)
    ImageDraw.Draw(canvas).text((0, 0), text, font=font, fill=255)
    alpha = np.asarray(canvas, dtype=np.float32) / 255.0
    if len(_text_bitmap_cache) >= _TEXT_CACHE_MAX:
        _text_bitmap_cache.clear()
    _text_bitmap_cache[key] = alpha
    return alpha


def put_text(img_bgr, text, org, px=28, color_bgr=(255, 255, 255)):
    """
    Draw text (CJK-capable) onto a BGR image at org=(x, y-top).
    Renders each distinct (text, px) to a small cached bitmap and alpha-blends
    it with numpy - no full-frame PIL round-trip, cheap enough for 30 fps loops.
    Falls back to cv2.putText (ASCII only) when no CJK font / PIL available.
    Returns the modified image.
    """
    font_path = find_cjk_font()
    if font_path and text:
        try:
            alpha = _text_alpha_bitmap(font_path, px, text)
            th, tw = alpha.shape
            ih, iw = img_bgr.shape[:2]
            x, y = int(org[0]), int(org[1])
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + tw, iw), min(y + th, ih)
            if x2 > x1 and y2 > y1:
                a = alpha[y1 - y:y2 - y, x1 - x:x2 - x, np.newaxis]
                region = img_bgr[y1:y2, x1:x2].astype(np.float32)
                color = np.asarray(color_bgr, dtype=np.float32)
                img_bgr[y1:y2, x1:x2] = \
                    (region * (1.0 - a) + color * a).astype(np.uint8)
            return img_bgr
        except Exception:
            pass
    ascii_text = text.encode('ascii', 'ignore').decode()
    cv2.putText(img_bgr, ascii_text, (org[0], org[1] + px),
                cv2.FONT_HERSHEY_SIMPLEX, px / 36.0, color_bgr, 2)
    return img_bgr


# ── Audio helpers ─────────────────────────────────────────────────────────────

def make_beep(freq=880.0, dur=0.12, sr=SAMPLE_RATE, volume=0.4):
    """Sine beep with 5 ms fade in/out. Returns float32 array."""
    t = np.arange(int(dur * sr)) / sr
    beep = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade = int(0.005 * sr)
    if fade > 0 and len(beep) > 2 * fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        beep[:fade] *= ramp
        beep[-fade:] *= ramp[::-1]
    return beep


def play_array(audio_f32, sr=SAMPLE_RATE, blocking=True):
    """Play a float32 numpy array through the default output device."""
    import sounddevice as sd
    sd.play(audio_f32, sr)
    if blocking:
        sd.wait()


def load_wav(path, target_sr=SAMPLE_RATE):
    """Load a wav as mono float32 at target_sr."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def play_wav(path, blocking=True):
    play_array(load_wav(path), SAMPLE_RATE, blocking)


# ── AV-HuBERT frozen feature extractor ────────────────────────────────────────

def preprocess_frames(frames_gray, crop_offset=None):
    """
    frames_gray: list/array of (96,96) uint8 mouth ROIs.
    crop_offset: (y0, x0) top-left of the 88x88 crop; None = center crop.
    Returns float32 torch tensor (1, 1, T, 88, 88), normalized.
    """
    import torch
    arr = np.asarray(frames_gray, dtype=np.float32) / 255.0      # (T,96,96)
    h, w = arr.shape[1], arr.shape[2]
    if crop_offset is None:
        y0 = (h - MODEL_CROP) // 2
        x0 = (w - MODEL_CROP) // 2
    else:
        y0 = int(np.clip(crop_offset[0], 0, h - MODEL_CROP))
        x0 = int(np.clip(crop_offset[1], 0, w - MODEL_CROP))
    arr = arr[:, y0:y0 + MODEL_CROP, x0:x0 + MODEL_CROP]
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    tensor = torch.from_numpy(arr[:, :, :, np.newaxis]).float()  # (T,88,88,1)
    return tensor.unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous()


def load_avhubert(ckpt_path=AVHUBERT_CKPT, device=None):
    """
    Load the pretrained AV-HuBERT model as a frozen feature extractor.
    Returns (model, device). Feature dim = model.encoder_embed_dim (1024 for Large).
    """
    import torch
    from argparse import Namespace
    from fairseq import checkpoint_utils, utils as fairseq_utils

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fairseq_utils.import_user_module(Namespace(user_dir=f'{ROOT}/avhubert'))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    if hasattr(model, 'w2v_model'):                # fine-tuned wrapper -> unwrap
        model = model.w2v_model
    if hasattr(model, 'remove_pretraining_modules'):
        model.remove_pretraining_modules()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"AV-HuBERT loaded on {device} "
          f"(embed_dim={getattr(model, 'encoder_embed_dim', '?')})")
    return model, device


def extract_avhubert_features(model, frames_gray, device, crop_offset=None):
    """
    frames_gray: (T,96,96) uint8 mouth ROIs (video-only, audio=None).
    Returns (T', D) float32 numpy features from the frozen encoder.
    """
    import torch
    video = preprocess_frames(frames_gray, crop_offset).to(device)   # (1,1,T,88,88)
    with torch.no_grad():
        feats, _ = model.extract_finetune(
            source={'audio': None, 'video': video},
            padding_mask=None,
            mask=False,
        )
    return feats.squeeze(0).float().cpu().numpy()


# ── Classifier head (shared by trainer and realtime app) ─────────────────────

try:
    import torch as _torch
    import torch.nn as _nn

    class PronunciationHead(_nn.Module):
        """
        AV-HuBERT features (B,T,D) -> pronunciation class logits.
        LayerNorm -> BiGRU -> masked mean+max pool -> MLP.
        """

        def __init__(self, input_dim=1024, hidden=256, num_classes=51, dropout=0.3):
            super().__init__()
            self.norm = _nn.LayerNorm(input_dim)
            self.gru = _nn.GRU(input_dim, hidden, batch_first=True,
                               bidirectional=True)
            self.fc = _nn.Sequential(
                _nn.Linear(hidden * 4, 256),
                _nn.ReLU(),
                _nn.Dropout(dropout),
                _nn.Linear(256, num_classes),
            )

        def forward(self, x, lengths=None):
            """x: (B,T,D) float. lengths: (B,) long or None (= full length)."""
            x = self.norm(x)
            B, T, _ = x.shape
            if lengths is None:
                lengths = _torch.full((B,), T, dtype=_torch.long, device=x.device)
            if bool((lengths < T).any()):
                # pack so the backward GRU direction never consumes pad frames
                # (training batches are padded; realtime inference never is)
                packed = _nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                out, _ = self.gru(packed)
                out, _ = _nn.utils.rnn.pad_packed_sequence(
                    out, batch_first=True, total_length=T)
            else:
                out, _ = self.gru(x)                    # (B,T,2*hidden)
            mask = (_torch.arange(T, device=x.device)[None, :]
                    < lengths[:, None])                 # (B,T) bool
            maskf = mask.unsqueeze(-1).float()
            mean = (out * maskf).sum(1) / maskf.sum(1).clamp(min=1.0)
            neg_inf = out.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            mx = neg_inf.max(dim=1).values
            return self.fc(_torch.cat([mean, mx], dim=-1))   # (B,num_classes)

except ImportError:                                   # torch-less machine: recorder still works
    PronunciationHead = None


def save_head_checkpoint(path, model, label_map, config, feat_mean, feat_std):
    import torch
    torch.save({
        'state_dict': model.state_dict(),
        'label_map': list(label_map),
        'config': dict(config),
        'feat_mean': np.asarray(feat_mean, dtype=np.float32),
        'feat_std': np.asarray(feat_std, dtype=np.float32),
    }, path)


def load_head_checkpoint(path=HEAD_CKPT, device='cpu'):
    """Returns (model, label_map, feat_mean, feat_std)."""
    import torch
    try:                          # self-produced checkpoint holds numpy arrays;
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:             # torch < 1.13 has no weights_only kwarg
        ckpt = torch.load(path, map_location='cpu')
    cfg = ckpt['config']
    model = PronunciationHead(**cfg)
    model.load_state_dict(ckpt['state_dict'])
    model.eval().to(device)
    return model, ckpt['label_map'], ckpt['feat_mean'], ckpt['feat_std']


# ── Session (recording progress + calibration) ────────────────────────────────

def load_session():
    if os.path.isfile(SESSION_FILE):
        with open(SESSION_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {'completed': {}, 'calibration': None}


def save_session(session):
    ensure_dirs()
    tmp = SESSION_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SESSION_FILE)


def take_paths(item_id, take):
    """Returns (npy_path, json_path) for a recording take (1-based)."""
    d = f'{REC_DIR}/{item_id}'
    return f'{d}/take{take}.npy', f'{d}/take{take}.json'


def save_take(item_id, take, frames_gray, meta):
    npy_path, json_path = take_paths(item_id, take)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, np.asarray(frames_gray, dtype=np.uint8))
    meta = dict(meta)
    meta.update({'item_id': item_id, 'take': take,
                 'fps': MODEL_FPS, 'n_frames': len(frames_gray),
                 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
