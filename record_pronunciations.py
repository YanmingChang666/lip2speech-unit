"""
Guided recording app for the Mandarin pronunciation-teaching + lip-reading suite.

Teaches each curriculum item via cached TTS, then records the reader's mouth ROI
3 times per item (countdown + quality gates), building the personal dataset that
train_pronunciation_model.py consumes.

Flow:
  startup checks -> idle-face calibration (sets the lip-movement threshold) ->
  per item: TEACH (hanzi / pinyin / hint, TTS plays twice) -> SPACE ->
            3-2-1 countdown -> record take (REC dot, progress bar, face status) ->
            quality gates (face coverage, lip movement) -> save + session update ->
  END screen with next-step hint.

Data written (authoritative spec: pronunciation_common.py docstring):
  datasets/pronunciation/recordings/{item_id}/take{k}.npy|.json
  datasets/pronunciation/recordings/_silence/take{k}.npy        (calibration clips)
  datasets/pronunciation/session.json                           (progress + calibration)

Usage:
  python record_pronunciations.py                      # all 50 core items
  python record_pronunciations.py --extended           # + 8 extended items
  python record_pronunciations.py --redo ba            # re-record one item
  python record_pronunciations.py --recalibrate        # redo idle-face calibration
  python record_pronunciations.py --camera 0 --takes 3 --seconds 2.0

Keys: SPACE=record | A=replay TTS | S=skip | R=retry | P=pause | Q=quit (saved)
"""

import sys
import os

# IMPORTANT: import pronunciation_common before anything avhubert/fairseq-related
# (it patches sys.argv for the avhubert DBG hack and sets sys.path). This recorder
# never imports torch/avhubert itself - common's heavy helpers stay uncalled.
from pronunciation_common import (
    REC_DIR, FEAT_DIR, SILENCE_ID, WEBCAM_FPS, MODEL_FPS, SAMPLE_RATE,
    RECORD_SECONDS, TAKES_PER_ITEM,
    ensure_dirs, extract_mouth_roi, movement_energy, brightness_stats,
    FrameSampler, put_text, make_beep, play_array, load_wav,
    load_session, save_session, save_take,
)
from pronunciation_curriculum import get_items, item_by_id, tts_path, check_tts_cache

import time
import shutil
import argparse

import numpy as np
import cv2

# ── config ────────────────────────────────────────────────────────────────────
WINDOW            = 'Record Pronunciations'
COUNTDOWN_SECS    = 3           # 3-2-1 before every take
TTS_GAP_SEC       = 0.4         # gap between the two teach playbacks
FACE_COVERAGE_MIN = 0.9         # take needs >= 90% face-found frames
CALIB_FACE_MIN    = 0.5         # calibration take is redone below this coverage
MOVEMENT_SCALE    = 2.5         # movement_threshold = idle_energy * scale ...
MOVEMENT_FLOOR    = 1.5         # ... but never below this floor
BRIGHT_RANGE      = (60, 200)   # acceptable mean grayscale brightness
FAIL_RETRY_LIMIT  = 3           # auto-retries before the R/A/S choice screen
FAIL_MSG_SEC      = 1.5
SAVED_FLASH_SEC   = 0.8
NEXT_TAKE_SEC     = 1.0
TOPBAR_H          = 34

WHITE  = (255, 255, 255)
GRAY   = (185, 185, 185)
GREEN  = (90, 220, 90)
RED    = (60, 60, 255)
YELLOW = (0, 215, 255)
ORANGE = (0, 165, 255)

BEEP_DIGIT = make_beep(880)
BEEP_GO    = make_beep(1320, 0.25)
BEEP_SAVED = make_beep(1320, 0.12)

KEY_SPACE = 32
KEY_NONE  = 255
# ──────────────────────────────────────────────────────────────────────────────


# ── drawing helpers (all CJK text goes through pronunciation_common.put_text) ─

def text_w(text, px):
    """Rough pixel width: CJK glyphs ~ px wide, Latin/pinyin ~ 0.55*px."""
    return int(sum(px if ord(c) > 0x2E7F else px * 0.55 for c in text))


def put_center(img, text, y, px=28, color=WHITE):
    x = max(0, (img.shape[1] - text_w(text, px)) // 2)
    put_text(img, text, (x, y), px, color)


def draw_lines_center(disp, lines):
    """lines: list of (text, px, color) drawn as a vertically centered block."""
    if not lines:
        return
    total = sum(px + 14 for _, px, _ in lines) - 14
    y = disp.shape[0] // 2 - total // 2
    for text, px, color in lines:
        put_center(disp, text, y, px, color)
        y += px + 14


def draw_top_bar(disp, item_no, n_items, take_no, takes, done, total):
    cv2.rectangle(disp, (0, 0), (disp.shape[1], TOPBAR_H), (32, 32, 32), -1)
    item_s = f'{item_no}/{n_items}' if item_no else '-'
    take_s = f'{take_no}/{takes}' if take_no else '-'
    pct = int(round(100.0 * done / max(1, total)))
    put_text(disp, f'项目 Item {item_s} · 第 {take_s} 次 take {take_s} · 总进度 {pct}%',
             (10, 6), px=20, color_bgr=WHITE)


def draw_roi_inset(disp, roi):
    """Mouth-ROI preview top-right, same placement as realtime_lip2speech.py."""
    if roi is None or disp.shape[1] < 300 or disp.shape[0] < 120:
        return
    inset = cv2.cvtColor(cv2.resize(roi, (96, 96)), cv2.COLOR_GRAY2BGR)
    disp[10:106, -106:-10] = inset


def draw_record_progress(disp, frac):
    h, w = disp.shape[:2]
    x0, x1, y0, y1 = 20, w - 20, h - 42, h - 26
    cv2.rectangle(disp, (x0, y0), (x1, y1), (90, 90, 90), 2)
    fill = int((x1 - x0 - 4) * min(1.0, max(0.0, frac)))
    if fill > 0:
        cv2.rectangle(disp, (x0 + 2, y0 + 2), (x0 + 2 + fill, y1 - 2), (0, 200, 255), -1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Guided lip-shape recording for the pronunciation curriculum.')
    parser.add_argument('--camera', type=int, default=0, help='webcam index')
    parser.add_argument('--takes', type=int, default=TAKES_PER_ITEM,
                        help='takes to record per item')
    parser.add_argument('--seconds', type=float, default=RECORD_SECONDS,
                        help='length of one take in seconds')
    parser.add_argument('--extended', action='store_true',
                        help='include the 8 extended curriculum items')
    parser.add_argument('--redo', metavar='ITEM_ID',
                        help='delete + re-record one item, e.g. --redo ba')
    parser.add_argument('--recalibrate', action='store_true',
                        help='redo the idle-face calibration')
    args = parser.parse_args([a for a in sys.argv[1:] if a != '_run_'])

    ensure_dirs()

    # ── curriculum + session ─────────────────────────────────────────────────
    items = get_items(args.extended)
    if args.redo:
        it = item_by_id(args.redo, extended=True)
        if it is None:
            sys.exit(f"Unknown item id: {args.redo!r} (see pronunciation_curriculum.py)")
        items = [it]

    session = load_session()
    session.setdefault('completed', {})
    session.setdefault('calibration', None)

    # ── TTS cache check + preload (instant playback later) ───────────────────
    missing = check_tts_cache(items)
    if missing:
        print(f"{len(missing)} item(s) missing TTS audio: {', '.join(missing)}")
        print("Fix with:  python pronunciation_curriculum.py --generate-tts"
              + (' --extended' if args.extended else ''))
        sys.exit(1)

    print(f"Preloading {len(items)} TTS clips...")
    gap = np.zeros(int(TTS_GAP_SEC * SAMPLE_RATE), dtype=np.float32)
    tts, tts_twice = {}, {}
    for it in items:
        wav = load_wav(tts_path(it['id']))
        tts[it['id']] = wav
        tts_twice[it['id']] = np.concatenate([wav, gap, wav])

    # ── webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open webcam index {args.camera}. "
                 f"Check the --camera index, that no other app is using the camera, "
                 f"and camera permissions.")
    cap.set(cv2.CAP_PROP_FPS, WEBCAM_FPS)
    sampler = FrameSampler(cap.get(cv2.CAP_PROP_FPS))   # true 30->25 decimation
    n_target = max(2, int(round(args.seconds * MODEL_FPS)))

    # --redo deletes data, so do it only after the webcam is confirmed working;
    # also purge the item's derived AV-HuBERT feature cache, otherwise the
    # trainer would silently keep using features of the deleted takes.
    if args.redo:
        shutil.rmtree(f'{REC_DIR}/{args.redo}', ignore_errors=True)
        shutil.rmtree(f'{FEAT_DIR}/{args.redo}', ignore_errors=True)
        session['completed'].pop(args.redo, None)
        save_session(session)
        print(f"Cleared previous takes + cached features of '{args.redo}' - "
              "re-recording it now.")

    calib = session['calibration']
    need_calib = args.recalibrate or calib is None
    movement_threshold = (calib or {}).get('movement_threshold', MOVEMENT_FLOOR)

    # ── state machine context ─────────────────────────────────────────────────
    state       = ''        # calib_intro/countdown/record/teach/message/choice/pause/end
    prev_state  = ''        # resume target for pause
    item_idx    = -1
    take_k      = 0
    pending     = []        # take numbers still to record for the current item
    fail_count  = 0
    cd_t0       = 0.0       # countdown start time
    cd_last     = None      # last beeped digit
    cd_mode     = 'item'    # 'item' | 'calib'
    cd_next     = None      # callable fired at GO
    frames      = []        # ROI frames of the take in progress
    misses      = 0         # sampled frames with no face during the take
    rec_mode    = 'item'    # 'item' | 'calib'
    msg_lines   = []        # [(text, px, color), ...]
    msg_until   = 0.0
    msg_next    = None      # callable fired when the message expires
    choice_lines = []       # fail reason shown on the R/A/S screen
    last_frames = None      # last failed attempt (for A=accept)
    last_e      = 0.0
    last_cov    = 0.0
    calib_take  = 1
    calib_takes = args.takes
    calib_fail  = 0         # consecutive failed calibration clips
    calib_E     = []        # movement energy per calibration take
    calib_B     = []        # frame brightness samples during calibration
    last_roi    = None
    face_ok     = False

    def is_complete(it):
        return len(session['completed'].get(it['id'], [])) >= args.takes

    def progress():
        total = len(items) * args.takes
        done = sum(min(len(session['completed'].get(it['id'], [])), args.takes)
                   for it in items)
        return done, total

    def show_message(lines, secs, next_fn):
        nonlocal state, msg_lines, msg_until, msg_next
        msg_lines, msg_until, msg_next = lines, time.time() + secs, next_fn
        state = 'message'

    def start_countdown(mode, next_fn):
        nonlocal state, cd_t0, cd_last, cd_mode, cd_next
        cd_t0, cd_last, cd_mode, cd_next = time.time(), None, mode, next_fn
        state = 'countdown'

    def start_record(mode):
        nonlocal state, frames, misses, rec_mode
        frames, misses, rec_mode = [], 0, mode
        state = 'record'

    def begin_item_take():
        start_countdown('item', lambda: start_record('item'))

    def enter_teach():
        nonlocal state, pending, take_k, fail_count
        it = items[item_idx]
        comp = session['completed'].get(it['id'], [])
        pending = [k for k in range(1, args.takes + 1) if k not in comp]
        take_k = pending[0] if pending else args.takes
        fail_count = 0
        play_array(tts_twice[it['id']], SAMPLE_RATE, blocking=False)   # play twice
        state = 'teach'

    def next_item():
        nonlocal item_idx, state
        item_idx += 1
        while item_idx < len(items) and is_complete(items[item_idx]):
            item_idx += 1
        if item_idx >= len(items):
            state = 'end'
        else:
            enter_teach()

    def goto_calib_intro():
        nonlocal state
        state = 'calib_intro'

    def start_calibration():
        nonlocal calib_take, calib_E, calib_B, calib_fail
        calib_take, calib_E, calib_B, calib_fail = 1, [], [], 0
        # old idle-clip features would otherwise be reused by the trainer
        shutil.rmtree(f'{FEAT_DIR}/{SILENCE_ID}', ignore_errors=True)
        start_countdown('calib', lambda: start_record('calib'))

    def finish_calib_take():
        nonlocal calib_take, movement_threshold, calib_fail
        if len(frames) < CALIB_FACE_MIN * n_target:
            calib_fail += 1
            if calib_fail >= FAIL_RETRY_LIMIT:   # escape hatch: back to intro
                calib_fail = 0
                show_message([('多次未检测到面部 face not found repeatedly', 30, RED),
                              ('请调整位置和光线后重试', 26, ORANGE),
                              ('adjust position/lighting, then SPACE to retry',
                               22, ORANGE)],
                             2.5, goto_calib_intro)
                return
            show_message([('未检测到面部，重试此段', 32, RED),
                          ('face not found - retrying this clip', 26, RED)],
                         FAIL_MSG_SEC, lambda: start_record('calib'))
            return
        calib_fail = 0
        save_take(SILENCE_ID, calib_take, frames, {'purpose': 'calibration'})
        calib_E.append(movement_energy(frames))
        if calib_take < calib_takes:
            calib_take += 1
            show_message([(f'保持静止 keep still - 第 {calib_take}/{calib_takes} 次', 30, WHITE)],
                         0.8, lambda: start_record('calib'))
            return
        idle = float(np.mean(calib_E))
        movement_threshold = max(idle * MOVEMENT_SCALE, MOVEMENT_FLOOR)
        bright = float(np.mean(calib_B)) if calib_B else 0.0
        session['calibration'] = {'idle_energy': idle,
                                  'movement_threshold': movement_threshold,
                                  'brightness': bright}
        save_session(session)
        print(f"Calibration done: idle_energy={idle:.2f} "
              f"movement_threshold={movement_threshold:.2f} brightness={bright:.0f}")
        if not (BRIGHT_RANGE[0] <= bright <= BRIGHT_RANGE[1]):
            show_message([('光线不佳 poor lighting', 40, ORANGE),
                          (f'画面亮度 {bright:.0f}（建议 {BRIGHT_RANGE[0]}-{BRIGHT_RANGE[1]}）'
                           f' brightness out of range', 24, ORANGE)],
                         FAIL_MSG_SEC, next_item)
        else:
            show_message([('校准完成 calibration done', 36, GREEN)], 1.0, next_item)

    def after_save():
        nonlocal pending, take_k, fail_count
        it = items[item_idx]
        comp = session['completed'].get(it['id'], [])
        pending = [k for k in range(1, args.takes + 1) if k not in comp]
        if pending:
            take_k, fail_count = pending[0], 0
            show_message([('准备下一次 get ready for next take', 32, WHITE)],
                         NEXT_TAKE_SEC, begin_item_take)
        else:
            next_item()

    def do_save_take(take_frames, e, cov):
        it = items[item_idx]
        save_take(it['id'], take_k, take_frames,
                  {'quality': {'movement': round(e, 3), 'face_cov': round(cov, 3)},
                   'hanzi': it['hanzi'], 'pinyin': it['pinyin']})
        comp = session['completed'].setdefault(it['id'], [])
        if take_k not in comp:
            comp.append(take_k)
            comp.sort()
        save_session(session)                     # persist immediately: crash-safe
        play_array(BEEP_SAVED, SAMPLE_RATE, blocking=False)
        show_message([('✓ 已保存 saved', 44, GREEN)], SAVED_FLASH_SEC, after_save)

    def finish_item_take():
        nonlocal state, fail_count, last_frames, last_e, last_cov, choice_lines
        # true face coverage: found frames over all sampled frames (a full-length
        # take with many misses must NOT read as 100%)
        cov = len(frames) / max(1, len(frames) + misses)
        e = movement_energy(frames)
        if len(frames) < n_target or cov < FACE_COVERAGE_MIN:
            reason = [('面部丢失过多 face lost too often', 32, RED),
                      (f'采到 {len(frames)}/{n_target} 帧，检出率 {cov * 100:.0f}% '
                       f'({len(frames)}/{n_target} frames, '
                       f'{cov * 100:.0f}% face found)', 24, RED)]
        elif e < movement_threshold:
            reason = [('嘴唇几乎没动，请出声读', 32, RED),
                      ('lips barely moved - please speak the syllable aloud', 24, RED)]
        else:
            do_save_take(list(frames), e, cov)
            return
        fail_count += 1
        last_frames, last_e, last_cov = list(frames), e, cov
        if fail_count >= FAIL_RETRY_LIMIT:
            choice_lines = reason
            state = 'choice'
        else:
            show_message(reason, FAIL_MSG_SEC, begin_item_take)

    # ── initial state ─────────────────────────────────────────────────────────
    if need_calib:
        state = 'calib_intro'
        print("Calibration first: keep your face still and silent.")
    else:
        print(f"Using saved calibration: movement_threshold={movement_threshold:.2f}")
        next_item()

    done0, total0 = progress()
    print(f"{len(items)} item(s), {args.takes} takes each - "
          f"{done0}/{total0} takes already saved.")
    print("Keys: SPACE=record | A=replay | S=skip | R=retry | P=pause | Q=quit\n")

    # ── one loop over cap.read(): preview never freezes, keys every frame ─────
    webcam_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam frame grab failed - exiting (progress is saved).")
            break
        webcam_cnt += 1
        now = time.time()

        # ── key handling (checked EVERY webcam frame) ────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break                                  # session already saved per take
        if state == 'pause':
            if key != KEY_NONE:
                state = prev_state                 # any key resumes
        elif key in (ord('p'), ord('P')) and state in ('calib_intro', 'teach',
                                                       'choice', 'end'):
            prev_state, state = state, 'pause'
        elif state == 'calib_intro' and key == KEY_SPACE:
            start_calibration()
        elif state == 'teach':
            if key == KEY_SPACE:
                if pending:
                    begin_item_take()
                else:
                    next_item()
            elif key in (ord('a'), ord('A')):
                play_array(tts[items[item_idx]['id']], SAMPLE_RATE, blocking=False)
            elif key in (ord('s'), ord('S')):
                next_item()
        elif state == 'choice':
            if key in (ord('r'), ord('R')):
                fail_count = 0
                begin_item_take()
            elif key in (ord('a'), ord('A')) and last_frames:
                do_save_take(last_frames, last_e, last_cov)
            elif key in (ord('s'), ord('S')):
                next_item()

        # ── frame sampling: 30 fps webcam -> 25 fps model rate ───────────────
        if sampler.take():
            last_roi = extract_mouth_roi(frame)
            face_ok = last_roi is not None
            if state == 'record':
                if last_roi is not None:
                    frames.append(last_roi)
                else:
                    misses += 1                    # no-face frame: count, don't append
                if rec_mode == 'calib':
                    calib_B.append(brightness_stats(frame)[0])

        # ── timed state updates ───────────────────────────────────────────────
        if state == 'countdown':
            elapsed = now - cd_t0
            if elapsed >= COUNTDOWN_SECS:
                play_array(BEEP_GO, SAMPLE_RATE, blocking=False)
                cd_next()
            else:
                digit = COUNTDOWN_SECS - int(elapsed)
                if digit != cd_last:
                    cd_last = digit
                    play_array(BEEP_DIGIT, SAMPLE_RATE, blocking=False)
        elif state == 'record':
            if len(frames) >= n_target or (len(frames) + misses) >= int(n_target * 1.5):
                if rec_mode == 'calib':
                    finish_calib_take()
                else:
                    finish_item_take()
        elif state == 'message' and now >= msg_until:
            fn, msg_next = msg_next, None
            if fn:
                fn()

        # ── draw ──────────────────────────────────────────────────────────────
        disp = frame.copy()
        h, w = disp.shape[:2]
        it = items[item_idx] if 0 <= item_idx < len(items) else None

        if state == 'calib_intro':
            put_center(disp, '校准 Calibration', h // 2 - 120, 48, WHITE)
            put_center(disp, '保持面部静止，不要说话', h // 2 - 44, 30, WHITE)
            put_center(disp, 'Keep your face still, do not speak', h // 2 - 4, 26, WHITE)
            put_center(disp, f'将录制 {calib_takes} 段 x {args.seconds:.1f}s 静止片段',
                       h // 2 + 40, 22, GRAY)
            put_center(disp, '空格开始 SPACE to start', h // 2 + 84, 28, YELLOW)

        elif state == 'countdown':
            digit = max(1, COUNTDOWN_SECS - int(now - cd_t0))
            if cd_mode == 'item' and it is not None:
                put_center(disp, it['hanzi'], TOPBAR_H + 12, 56, WHITE)
                put_center(disp, it['pinyin'], TOPBAR_H + 76, 34, YELLOW)
                put_center(disp, f'第 {take_k}/{args.takes} 次 take {take_k}/{args.takes}',
                           h - 74, 26, WHITE)
            else:
                put_center(disp, f'校准 calibration {calib_take}/{calib_takes}',
                           TOPBAR_H + 24, 30, WHITE)
                put_center(disp, '保持静止 keep still', h - 74, 26, WHITE)
            put_center(disp, str(digit), h // 2 - 55, 120, YELLOW)

        elif state == 'teach' and it is not None:
            put_center(disp, it['hanzi'], TOPBAR_H + 16, 90, WHITE)
            put_center(disp, it['pinyin'], TOPBAR_H + 118, 50, YELLOW)
            y = TOPBAR_H + 182
            if it['hint']:
                put_center(disp, it['hint'], y, 28, WHITE)
                y += 38
            put_center(disp, f"口型 viseme: {it['viseme']}", y, 24, GRAY)
            put_center(disp, '空格=开始录制 SPACE=record | A=再听 replay', h - 78, 22, GRAY)
            put_center(disp, 'S=跳过 skip | Q=退出保存 quit', h - 48, 22, GRAY)

        elif state == 'record':
            if rec_mode == 'item' and it is not None:
                put_center(disp, it['hanzi'], TOPBAR_H + 10, 48, WHITE)
                put_center(disp, it['pinyin'], TOPBAR_H + 64, 28, YELLOW)
                label = f'REC 第 {take_k}/{args.takes} 次 take {take_k}/{args.takes}'
            else:
                label = f'REC 校准 calibration {calib_take}/{calib_takes}'
            if int(now * 2) % 2 == 0:                          # blinking REC dot
                cv2.circle(disp, (24, TOPBAR_H + 28), 10, RED, -1)
            put_text(disp, label, (42, TOPBAR_H + 14), 24, RED)
            put_text(disp, '面部正常 Face OK' if face_ok else '未检测到面部 No face',
                     (20, h - 80), 24, GREEN if face_ok else RED)
            draw_record_progress(disp, len(frames) / n_target)

        elif state == 'message':
            draw_lines_center(disp, msg_lines)

        elif state == 'choice':
            draw_lines_center(disp, list(choice_lines) + [
                (f'已连续失败 {fail_count} 次 failed {fail_count} times', 24, ORANGE),
                ('R=重试 retry | A=仍然保存 accept | S=跳过此项 skip item', 24, YELLOW),
            ])

        elif state == 'pause':
            disp[:] = disp // 2
            put_center(disp, '已暂停 Paused', h // 2 - 60, 48, WHITE)
            put_center(disp, '按任意键继续 press any key to resume', h // 2 + 20, 26, GRAY)

        elif state == 'end':
            n_done = sum(1 for x in items if is_complete(x))
            put_center(disp, f'全部完成 All done! {n_done}/{len(items)} items',
                       h // 2 - 90, 44, GREEN)
            put_center(disp, '下一步 next step:', h // 2 - 10, 28, WHITE)
            put_center(disp, 'python train_pronunciation_model.py', h // 2 + 30, 26, YELLOW)
            put_center(disp, 'Q=退出 quit', h - 60, 24, GRAY)

        # persistent top bar + mouth-ROI inset on every screen
        done, total = progress()
        draw_top_bar(disp, item_idx + 1 if it is not None else None, len(items),
                     take_k if it is not None else None, args.takes, done, total)
        draw_roi_inset(disp, last_roi)
        cv2.imshow(WINDOW, disp)

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    done, total = progress()
    n_done = sum(1 for x in items if is_complete(x))
    print(f"Session saved: {done}/{total} takes ({n_done}/{len(items)} items complete).")
    if n_done < len(items):
        print("Run record_pronunciations.py again to continue where you left off.")
    else:
        print("Next step: python train_pronunciation_model.py")


if __name__ == '__main__':
    main()
