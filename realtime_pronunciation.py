"""
Real-time personalized lip reading: webcam lip movement -> pronunciation -> TTS.

Pipeline:
  webcam 30fps -> sample to 25fps -> mouth ROI -> movement-energy segmentation
  -> frozen AV-HuBERT features -> PronunciationHead (trained on YOUR lips)
  -> big hanzi + pinyin overlay + teaching TTS audio playback

Segmentation: a small state machine on the movement energy of the last 5
sampled ROI frames. Rising energy for 3 consecutive frames starts a segment
(with a 5-frame pre-roll); low energy for 8 consecutive frames (or hitting
--max-frames) ends it. Segments shorter than --min-frames are discarded.

Run AFTER train_pronunciation_model.py (needs checkpoints/pronunciation_head.pt).

Usage:
  python realtime_pronunciation.py                    # defaults
  python realtime_pronunciation.py --debug            # + top-3 probability bars
  python realtime_pronunciation.py --threshold-scale 1.3   # less sensitive start
Keys: q = quit
"""

import sys
import os

# Import the shared foundation FIRST: it patches sys.argv for the avhubert
# DBG hack and inserts fairseq/ + repo root into sys.path.
from pronunciation_common import (
    HEAD_CKPT, SILENCE_ID, WEBCAM_FPS, MODEL_FPS, SAMPLE_RATE,
    extract_mouth_roi, movement_energy, FrameSampler, put_text,
    play_array, load_wav,
    load_avhubert, extract_avhubert_features, load_head_checkpoint,
    load_session,
)
from pronunciation_curriculum import item_by_id, tts_path

import time
import threading
import queue
import argparse
import warnings
from collections import deque
warnings.filterwarnings('ignore')

import numpy as np
import torch
import cv2

# ── config ────────────────────────────────────────────────────────────────────
ENERGY_WINDOW   = 5         # sampled ROI frames per movement-energy estimate
PREROLL_FRAMES  = 5         # frames kept before the detected segment start
RISE_FRAMES     = 2         # consecutive frames above thr to start a segment
FALL_FRAMES     = 8         # consecutive frames below thr*FALL_RATIO to end one
FALL_RATIO      = 0.7       # end-of-segment threshold = thr * FALL_RATIO
REALTIME_SCALE  = 1.8       # thr = idle_energy * this. Gentler than the
                            # recorder quality gate (2.5): starting a segment
                            # needs the bar crossed on consecutive LIVE frames,
                            # not once per whole take
MOVEMENT_FLOOR  = 0.4       # thr never below this
DEFAULT_THRESHOLD = 1.0     # only used when session.json has no calibration

HIGHLIGHT_S     = 1.5       # bright hanzi+pinyin display time after a hit
UNKNOWN_SHOW_S  = 1.0       # gray "?" display time after a rejected segment
COOLDOWN_S      = 0.5       # ignore new segment starts after a playback trigger
STD_FLOOR       = 1e-5      # added to feat_std at normalization time

WINDOW_NAME     = 'Pronunciation Lip Reading'
# ──────────────────────────────────────────────────────────────────────────────

# recording states
IDLE, RECORDING = 'idle', 'recording'


# ── loading helpers ───────────────────────────────────────────────────────────

def resolve_device(choice):
    if choice == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return choice


def movement_threshold_from_session(scale):
    """
    Segmentation threshold derived from the calibrated IDLE energy:
        thr = max(idle_energy * REALTIME_SCALE, MOVEMENT_FLOOR) * scale
    Deliberately NOT the recorder's stored movement_threshold (a quality gate
    checked once per take against the PEAK energy) - the live start trigger
    must be crossed on consecutive frames, so it needs a lower bar.
    """
    calib = load_session().get('calibration')
    idle = None
    if isinstance(calib, dict):
        idle = calib.get('idle_energy')
        if idle is None and calib.get('movement_threshold') is not None:
            idle = float(calib['movement_threshold']) / 2.5   # legacy session
    try:
        thr = max(float(idle) * REALTIME_SCALE, MOVEMENT_FLOOR) * scale
        print(f"Movement threshold: {thr:.2f} "
              f"(idle {float(idle):.2f} x {REALTIME_SCALE:g} x scale {scale:g})")
        return thr
    except (TypeError, ValueError):
        thr = DEFAULT_THRESHOLD * scale
        print(f"[!] No movement calibration in session.json - using default "
              f"threshold {thr:.2f}.\n"
              f"    Run record_pronunciations.py first for a personalized value.")
        return thr


def build_display_info(label_map):
    """{item_id: (hanzi, pinyin)} for every non-silence class."""
    info = {}
    for cid in label_map:
        if cid == SILENCE_ID:
            continue
        it = item_by_id(cid)
        if it is None:
            print(f"[!] '{cid}' not found in the curriculum - showing raw id.")
            info[cid] = (cid, cid)
        else:
            info[cid] = (it['hanzi'], it['pinyin'])
    return info


def preload_tts(label_map):
    """Preload {item_id: float32 wav} for every non-silence class."""
    cache, missing = {}, []
    for cid in label_map:
        if cid == SILENCE_ID:
            continue
        wav = tts_path(cid)
        if os.path.isfile(wav):
            cache[cid] = load_wav(wav)
        else:
            missing.append(cid)
    if missing:
        print(f"[!] No TTS audio for {len(missing)} item(s): {' '.join(missing)}\n"
              f"    These stay display-only. Generate audio with: "
              f"python pronunciation_curriculum.py --generate-tts")
    print(f"TTS cache: {len(cache)} wavs preloaded.")
    return cache


# ── background classifier ─────────────────────────────────────────────────────

def classify_async(segment, avhubert, device, head, feat_mean, feat_std,
                   label_map, display_info, tts_cache, audio_queue,
                   shared, lock, conf):
    """
    Runs in a background thread (one at a time). AV-HuBERT features -> head
    -> softmax top-3 -> publish into `shared` under `lock`; confident non-silence
    predictions also queue their TTS wav and arm the segmentation cooldown.
    """
    t0 = time.time()
    try:
        feats = extract_avhubert_features(avhubert, segment, device)
        x = (feats - feat_mean) / (feat_std + STD_FLOOR)
        xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        lengths = torch.tensor([xt.shape[1]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = head(xt, lengths=lengths)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    except Exception as e:
        print(f"[classify error] {e}")
        return

    order = np.argsort(probs)[::-1][:3]
    top3 = [(label_map[i], float(probs[i])) for i in order]
    latency_ms = (time.time() - t0) * 1000.0
    label, prob = top3[0]
    top3_str = ' '.join(f'{l}:{p:.2f}' for l, p in top3)
    now = time.time()

    if label == SILENCE_ID or prob < conf:
        reason = 'silence' if label == SILENCE_ID else f'p<{conf:g}'
        print(f"[pred] ? ({reason}) top3={top3_str} latency={latency_ms:.0f} ms")
        with lock:
            shared['unknown_until'] = now + UNKNOWN_SHOW_S
            shared['top3'] = top3
        return

    hanzi, pinyin = display_info.get(label, (label, label))
    print(f"[pred] {label} {hanzi} ({pinyin}) p={prob:.2f} "
          f"top3={top3_str} latency={latency_ms:.0f} ms")

    audio = tts_cache.get(label)
    if audio is not None:
        audio_queue.put(audio)
    with lock:
        shared['last_pred'] = {'label': label, 'hanzi': hanzi, 'pinyin': pinyin,
                               'prob': prob, 'time': now}
        shared['top3'] = top3
        shared['cooldown_until'] = now + COOLDOWN_S   # also for display-only items


# ── audio player thread (same pattern as realtime_lip2speech.py) ──────────────

def audio_player(audio_queue):
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        play_array(audio, SAMPLE_RATE, blocking=True)


# ── main real-time loop ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Real-time lip reading: webcam lip movement -> pronunciation '
                    '-> hanzi/pinyin overlay + TTS playback.')
    parser.add_argument('--camera', type=int, default=0,
                        help='webcam index for cv2.VideoCapture')
    parser.add_argument('--conf', type=float, default=0.55,
                        help='min top-1 probability to accept a prediction')
    parser.add_argument('--max-frames', type=int, default=60,
                        help='hard cap on segment length (25fps frames)')
    parser.add_argument('--min-frames', type=int, default=12,
                        help='segments shorter than this are discarded')
    parser.add_argument('--threshold-scale', type=float, default=1.0,
                        help='multiplier on the calibrated movement threshold')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'],
                        default='auto')
    parser.add_argument('--debug', action='store_true',
                        help='render top-3 labels + probability bars')
    args = parser.parse_args([a for a in sys.argv[1:] if a != '_run_'])

    # ── load models + data ────────────────────────────────────────────────────
    if not os.path.isfile(HEAD_CKPT):
        sys.exit(f"ERROR: classifier checkpoint not found: {HEAD_CKPT}\n"
                 "Run train_pronunciation_model.py first.")

    device = resolve_device(args.device)
    print("Loading models (this may take ~30s)...")
    head, label_map, feat_mean, feat_std = load_head_checkpoint(HEAD_CKPT, device)
    feat_mean = np.asarray(feat_mean, dtype=np.float32)
    feat_std = np.asarray(feat_std, dtype=np.float32)
    print(f"Classifier head loaded: {len(label_map)} classes "
          f"({len(label_map) - 1} items + silence).")

    avhubert, device = load_avhubert(device=device)
    display_info = build_display_info(label_map)
    tts_cache = preload_tts(label_map)
    thr = movement_threshold_from_session(args.threshold_scale)

    # ── webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {args.camera}")
    cap.set(cv2.CAP_PROP_FPS, WEBCAM_FPS)
    sampler = FrameSampler(cap.get(cv2.CAP_PROP_FPS))   # true 30->25 decimation
    # must match the recorder's sampling so segments look like training clips

    print(f"\nReady. conf>={args.conf:g}, segment {args.min_frames}-"
          f"{args.max_frames} frames, device={device}.")
    print("Speak one curriculum syllable at a time.  Q = quit\n")

    # ── shared classifier state (lock-protected) ──────────────────────────────
    shared = {'last_pred': None, 'unknown_until': 0.0, 'top3': None,
              'cooldown_until': 0.0}
    lock = threading.Lock()

    audio_queue = queue.Queue()
    player_thread = threading.Thread(target=audio_player, args=(audio_queue,),
                                     daemon=True)
    player_thread.start()

    # ── segmentation state ────────────────────────────────────────────────────
    state = IDLE
    roi_window = deque(maxlen=ENERGY_WINDOW)   # recent ROIs for movement energy
    preroll = deque(maxlen=PREROLL_FRAMES)     # frames kept before segment start
    segment = []
    rise_cnt, fall_cnt = 0, 0
    last_energy = 0.0
    last_roi = None
    webcam_frame_cnt = 0
    infer_thread = None
    near_miss = 0            # sampled IDLE frames with movement just below thr
    hint_printed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        webcam_frame_cnt += 1

        # ── key handling (every frame, for responsiveness) ────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

        # one shared-state snapshot per frame
        with lock:
            snap = dict(shared)

        # ── sampled-frame processing: ROI + segmentation state machine ────────
        if sampler.take():
            roi = extract_mouth_roi(frame)
            last_roi = roi

            if roi is not None:
                roi_window.append(roi)
                last_energy = (movement_energy(list(roi_window))
                               if len(roi_window) >= 2 else 0.0)
            else:
                # lost face: drop stale frames so the return jump is not
                # counted as movement
                roi_window.clear()
                last_energy = 0.0

            if state == IDLE:
                if roi is not None:
                    preroll.append(roi)
                    in_cooldown = time.time() < snap['cooldown_until']
                    if last_energy > thr and not in_cooldown:
                        rise_cnt += 1
                    else:
                        rise_cnt = 0
                    # movement that stays just under the bar -> show a hint
                    if thr * 0.5 < last_energy <= thr:
                        near_miss = min(near_miss + 1, 200)
                        if near_miss == 25 and not hint_printed:
                            hint_printed = True
                            print("[hint] Lips are moving but stay below the "
                                  "segmentation threshold - try: "
                                  "python realtime_pronunciation.py "
                                  "--threshold-scale 0.7")
                    elif last_energy > thr:
                        near_miss = 0
                    if rise_cnt >= RISE_FRAMES:
                        state = RECORDING
                        segment = list(preroll)
                        rise_cnt, fall_cnt = 0, 0
                        near_miss = 0
                else:
                    preroll.clear()
                    rise_cnt = 0

            elif state == RECORDING:
                if roi is not None:
                    segment.append(roi)
                    if last_energy < thr * FALL_RATIO:
                        fall_cnt += 1
                    else:
                        fall_cnt = 0
                else:
                    fall_cnt += 1          # lost face counts as quiet

                if fall_cnt >= FALL_FRAMES or len(segment) >= args.max_frames:
                    finished, segment = segment, []
                    state = IDLE
                    preroll.clear()
                    rise_cnt, fall_cnt = 0, 0
                    if len(finished) < args.min_frames:
                        print(f"[seg] too short ({len(finished)} < "
                              f"{args.min_frames} frames) - discarded")
                    elif infer_thread is not None and infer_thread.is_alive():
                        print(f"[!] classifier busy - segment dropped "
                              f"({len(finished)} frames)")
                    else:
                        infer_thread = threading.Thread(
                            target=classify_async,
                            args=(finished, avhubert, device, head,
                                  feat_mean, feat_std, label_map, display_info,
                                  tts_cache, audio_queue, shared, lock,
                                  args.conf),
                            daemon=True,
                        )
                        infer_thread.start()

        # ── build display frame (every webcam frame) ──────────────────────────
        disp = frame.copy()
        h, w = disp.shape[:2]
        now = time.time()
        classifying = infer_thread is not None and infer_thread.is_alive()

        # face indicator (top-left)
        face_ok = last_roi is not None
        put_text(disp, '有人脸 Face OK' if face_ok else '未检测到人脸 No face',
                 (10, 10), px=22,
                 color_bgr=(0, 255, 0) if face_ok else (0, 0, 255))

        # mouth ROI inset (top-right)
        if last_roi is not None:
            roi_rgb = cv2.cvtColor(cv2.resize(last_roi, (96, 96)),
                                   cv2.COLOR_GRAY2BGR)
            disp[10:106, -106:-10] = roi_rgb

        # blinking REC dot while capturing (left of the ROI inset)
        if state == RECORDING and int(now * 2) % 2 == 0:
            cv2.circle(disp, (w - 130, 25), 10, (0, 0, 255), -1)

        # prediction overlay (center-top): gray "?" for rejected segments,
        # else the last prediction - bright for HIGHLIGHT_S, then dimmed
        if now < snap['unknown_until']:
            px = 64
            put_text(disp, '?', ((w - px // 2) // 2, 16), px=px,
                     color_bgr=(160, 160, 160))
        elif snap['last_pred'] is not None:
            pred = snap['last_pred']
            hot = (now - pred['time']) < HIGHLIGHT_S
            hz_px, py_px = 72, 34
            hz_color = (0, 215, 255) if hot else (120, 120, 120)
            py_color = (255, 255, 255) if hot else (120, 120, 120)
            hz, py = pred['hanzi'], pred['pinyin']
            put_text(disp, hz, ((w - len(hz) * hz_px) // 2, 12),
                     px=hz_px, color_bgr=hz_color)
            put_text(disp, py, (int((w - len(py) * py_px * 0.55) // 2),
                                12 + hz_px + 8), px=py_px, color_bgr=py_color)

        # state text (bottom-left)
        if classifying:
            state_txt, state_color = '识别中 classifying', (0, 165, 255)
        elif state == RECORDING:
            state_txt = f'捕捉中 capturing {len(segment) / MODEL_FPS:.1f}s'
            state_color = (0, 0, 255)
        else:
            state_txt, state_color = '等待说话 waiting', (200, 200, 200)
        put_text(disp, state_txt, (10, h - 40), px=26, color_bgr=state_color)

        # movement meter (always on): green while above the trigger threshold
        put_text(disp, f'动作 E {last_energy:.2f} / 阈值 thr {thr:.2f}',
                 (10, h - 72), px=20,
                 color_bgr=(0, 255, 0) if last_energy > thr else (180, 180, 180))
        if near_miss >= 25 and state == IDLE:
            put_text(disp, '嘴唇在动但低于阈值 movement below threshold - '
                           'try --threshold-scale 0.7',
                     (10, h - 100), px=20, color_bgr=(0, 165, 255))

        # debug: top-3 probability bars
        if args.debug:
            if snap['top3'] is not None:
                y0 = h - 172
                for i, (lbl, p) in enumerate(snap['top3']):
                    y = y0 + i * 30
                    cv2.rectangle(disp, (10, y + 3), (10 + int(200 * p), y + 25),
                                  (60, 160, 60), -1)
                    hz, _ = display_info.get(lbl, ('', lbl))
                    put_text(disp, f'{lbl} {hz} {p:.2f}', (16, y), px=20,
                             color_bgr=(255, 255, 255))

        cv2.imshow(WINDOW_NAME, disp)

    # ── cleanup ───────────────────────────────────────────────────────────────
    if infer_thread is not None and infer_thread.is_alive():
        infer_thread.join(timeout=30)
    audio_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    player_thread.join(timeout=3)
    print("Done.")


if __name__ == '__main__':
    main()
