"""
Offline trainer for the personalized pronunciation lip-reading head.

Pipeline:
  1. scan    datasets/pronunciation/recordings/{item_id}/take{k}.npy  (+ _silence)
  2. augment every TRAIN take (speed / crop offset / brightness / flip / noise)
  3. extract frozen AV-HuBERT features, cached at
             datasets/pronunciation/features/{item_id}/take{k}_aug{j}.npy
  4. train   PronunciationHead (LayerNorm -> BiGRU -> masked pool -> MLP)
  5. report  best val top-1/top-3, per-class results, top confusion pairs
  6. save    checkpoints/pronunciation_head.pt  (see pronunciation_common docstring)

Split rule: per class, take --val-take (or the highest existing take) is held out
for validation and gets ONLY the un-augmented aug0 features; every other take
contributes aug0 plus --augs-per-take augmented variants to training.

Usage:
  python train_pronunciation_model.py                  # defaults: 60 epochs, 8 augs/take
  python train_pronunciation_model.py --force-extract  # rebuild the feature cache
  python train_pronunciation_model.py --extended       # include the 8 extended items

Run AFTER record_pronunciations.py (needs >= 2 takes per item + silence calibration).
Next step: python realtime_pronunciation.py
"""

import sys
import os

# Import the shared foundation FIRST: it patches sys.argv for the avhubert
# DBG hack and inserts fairseq/ + repo root into sys.path.
from pronunciation_common import (
    REC_DIR, FEAT_DIR, HEAD_CKPT, SILENCE_ID,
    PronunciationHead, load_avhubert, extract_avhubert_features,
    save_head_checkpoint, ensure_dirs,
)
from pronunciation_curriculum import get_items

import re
import glob
import time
import zlib
import argparse
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn

# ── config ────────────────────────────────────────────────────────────────────
HIDDEN          = 256       # PronunciationHead GRU hidden size
DROPOUT         = 0.3
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
PATIENCE        = 12        # early-stop patience (epochs without val improvement)
MIN_TAKES       = 2         # curriculum classes need >= 2 takes to be included

MIN_FRAMES      = 10        # temporal resampling never goes below this
SPEED_RANGE     = (0.8, 1.25)
ALPHA_RANGE     = (0.8, 1.2)     # contrast
BETA_RANGE      = (-20.0, 20.0)  # brightness
NOISE_SIGMA_MAX = 6.0
FLIP_PROB       = 0.5
MAX_CROP_OFF    = 8         # 96 - 88: valid top-left offsets are 0..8 inclusive

PROGRESS_EVERY  = 50        # extraction progress print interval (clips)
STD_FLOOR       = 1e-5      # feat_std floor to avoid divide-by-zero
# ──────────────────────────────────────────────────────────────────────────────


# ── scan recordings ───────────────────────────────────────────────────────────

def list_takes(item_id):
    """Sorted list of take numbers with a take{k}.npy under REC_DIR/{item_id}/."""
    takes = []
    for p in glob.glob(f'{REC_DIR}/{item_id}/take*.npy'):
        m = re.fullmatch(r'take(\d+)\.npy', os.path.basename(p))
        if m:
            takes.append(int(m.group(1)))
    return sorted(takes)


def scan_recordings(items):
    """
    Print a table of available takes per class and build the label map.
    Returns (label_map, takes_by_class).
      label_map      : [SILENCE_ID] + item ids (curriculum order) with >= MIN_TAKES takes
      takes_by_class : {class_id: [take numbers]} for every class in label_map
    """
    print(f"\nScanning recordings under {REC_DIR}\n")
    print(f"  {'item':10s} {'pinyin':8s} {'takes':>5s}   status")
    print(f"  {'-' * 44}")

    # Silence class first: required by the realtime not-speaking gate.
    silence_takes = list_takes(SILENCE_ID)
    if not silence_takes:
        sys.exit(
            "ERROR: no silence recordings found "
            f"({REC_DIR}/{SILENCE_ID}/take*.npy).\n"
            "The realtime gate requires a silence (idle face) class.\n"
            "Run the recorder's calibration first: python record_pronunciations.py"
        )
    status = 'ok' if len(silence_takes) >= MIN_TAKES else \
        'WARNING: only 1 take (kept - required class, no val row)'
    print(f"  {SILENCE_ID:10s} {'--':8s} {len(silence_takes):5d}   {status}")

    label_map = [SILENCE_ID]
    takes_by_class = {SILENCE_ID: silence_takes}
    excluded = []

    for it in items:
        takes = list_takes(it['id'])
        if len(takes) >= MIN_TAKES:
            status = 'ok'
            label_map.append(it['id'])
            takes_by_class[it['id']] = takes
        elif len(takes) == 1:
            status = 'EXCLUDED: only 1 take (need >= 2)'
            excluded.append(it['id'])
        else:
            status = 'EXCLUDED: not recorded'
            excluded.append(it['id'])
        print(f"  {it['id']:10s} {it['pinyin']:8s} {len(takes):5d}   {status}")

    print(f"\n  {len(label_map)} classes included "
          f"({len(label_map) - 1} items + silence), {len(excluded)} excluded.")
    if excluded:
        print(f"  Excluded items need more takes: {' '.join(excluded)}")
        print("  Record them with: python record_pronunciations.py")
    if len(label_map) < 3:
        sys.exit("ERROR: fewer than 2 usable pronunciation classes - "
                 "record more items before training.")
    return label_map, takes_by_class


def choose_val_takes(label_map, takes_by_class, requested_val_take):
    """
    Per class: hold out --val-take if it exists, else the highest take.
    Classes with a single take get no val row (all data goes to training).
    Returns {class_id: val_take_number or None}.
    """
    val_takes = {}
    for cid in label_map:
        takes = takes_by_class[cid]
        if len(takes) < 2:
            val_takes[cid] = None
            print(f"  [!] {cid}: single take -> all to train, no validation row")
        elif requested_val_take in takes:
            val_takes[cid] = requested_val_take
        else:
            val_takes[cid] = max(takes)
    return val_takes


# ── augmentation ──────────────────────────────────────────────────────────────

def make_augmented_clip(clip_u8, item_id, take, aug_j):
    """
    clip_u8: (T,96,96) uint8 mouth ROI clip.
    Returns (frames_u8, crop_offset) where crop_offset is (y0,x0) for the 88x88
    crop done inside extract_avhubert_features (None = center crop).
    aug_j == 0 is ALWAYS the untouched original with center crop.
    Deterministic per (item_id, take, aug_j).
    """
    if aug_j == 0:
        return clip_u8, None

    # crc32, NOT hash(): Python's str hash is salted per process, which would
    # silently change the augmentations (and defeat the cache) on every run
    rs = np.random.RandomState(zlib.crc32(f'{item_id}/{take}/{aug_j}'.encode()))
    arr = clip_u8.astype(np.float32)

    # 1) temporal speed change: linear frame-index resampling, keep >= MIN_FRAMES
    speed = rs.uniform(*SPEED_RANGE)
    t = arr.shape[0]
    new_t = max(MIN_FRAMES, int(round(t / speed)))
    idx = np.linspace(0.0, t - 1.0, new_t)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, t - 1)
    w = (idx - lo).astype(np.float32)[:, None, None]
    arr = arr[lo] * (1.0 - w) + arr[hi] * w

    # 2) random 88x88 crop offset - NOT applied here; passed to the extractor
    crop_offset = (int(rs.randint(0, MAX_CROP_OFF + 1)),
                   int(rs.randint(0, MAX_CROP_OFF + 1)))

    # 3) brightness / contrast
    alpha = rs.uniform(*ALPHA_RANGE)
    beta = rs.uniform(*BETA_RANGE)
    arr = np.clip(arr * alpha + beta, 0.0, 255.0)

    # 4) horizontal flip
    if rs.uniform() < FLIP_PROB:
        arr = arr[:, :, ::-1]

    # 5) gaussian noise
    sigma = rs.uniform(0.0, NOISE_SIGMA_MAX)
    if sigma > 0.0:
        arr = arr + rs.normal(0.0, sigma, size=arr.shape)

    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr), crop_offset


# ── feature extraction (cached) ───────────────────────────────────────────────

def feat_path(item_id, take, aug_j):
    return f'{FEAT_DIR}/{item_id}/take{take}_aug{aug_j}.npy'


def rec_path(item_id, take):
    return f'{REC_DIR}/{item_id}/take{take}.npy'


def remove_stale_val_augs(label_map, val_takes):
    """Val takes must only ever have aug0 features on disk (leakage guard)."""
    removed = 0
    for cid in label_map:
        vt = val_takes[cid]
        if vt is None:
            continue
        for p in glob.glob(f'{FEAT_DIR}/{cid}/take{vt}_aug*.npy'):
            m = re.fullmatch(r'take\d+_aug(\d+)\.npy', os.path.basename(p))
            if m and int(m.group(1)) >= 1:
                os.remove(p)
                removed += 1
    if removed:
        print(f"  Removed {removed} stale val-take augmentation feature files "
              "(leakage guard).")


def aug_range_for(take, val_take, augs_per_take):
    """Val take -> only aug0. Train take -> aug0 + augs_per_take variants."""
    if take == val_take:
        return range(1)
    return range(augs_per_take + 1)


def cache_valid(item_id, take, aug_j):
    """
    A cached feature file counts only if it is newer than its source recording -
    re-recorded takes (--redo / --recalibrate) must invalidate their features.
    """
    fp = feat_path(item_id, take, aug_j)
    if not os.path.isfile(fp):
        return False
    try:
        return os.path.getmtime(fp) >= os.path.getmtime(rec_path(item_id, take))
    except OSError:
        return False


def extract_all_features(label_map, takes_by_class, val_takes, args):
    """
    Ensure every needed feature file exists (reuse cache unless --force-extract).
    Loads AV-HuBERT lazily - skipped entirely when everything is cached.
    """
    # Plan the work first so a fully-cached run never loads the 3 GB model.
    jobs = []                                   # (item_id, take, aug_j)
    for cid in label_map:
        for take in takes_by_class[cid]:
            for j in aug_range_for(take, val_takes[cid], args.augs_per_take):
                jobs.append((cid, take, j))
    todo = [j for j in jobs
            if args.force_extract or not cache_valid(*j)]

    print(f"\nFeature extraction: {len(jobs)} clips total, "
          f"{len(jobs) - len(todo)} cached, {len(todo)} to extract.")
    if not todo:
        return

    model, device = load_avhubert(
        device=None if args.device == 'auto' else args.device)

    processed, extracted, failed = 0, 0, 0
    t0 = time.time()
    clip_cache_key, clip = None, None           # avoid reloading a take 9 times
    for cid, take, j in todo:
        processed += 1
        try:
            if clip_cache_key != (cid, take):
                clip = np.load(rec_path(cid, take))
                clip_cache_key = (cid, take)
            frames, crop_offset = make_augmented_clip(clip, cid, take, j)
            feats = extract_avhubert_features(model, frames, device,
                                              crop_offset=crop_offset)
            os.makedirs(os.path.dirname(feat_path(cid, take, j)), exist_ok=True)
            np.save(feat_path(cid, take, j), feats.astype(np.float32))
            extracted += 1
        except Exception as e:                  # never crash a long run mid-way
            failed += 1
            print(f"  [!] extraction failed for {cid} take{take} aug{j}: "
                  f"{e} - skipped")
        if processed % PROGRESS_EVERY == 0:
            rate = processed / max(time.time() - t0, 1e-6)
            print(f"  ... {processed}/{len(todo)} clips "
                  f"({extracted} ok, {failed} failed, {rate:.1f} clips/s)")

    print(f"  Extraction done in {time.time() - t0:.0f}s: "
          f"{extracted} new, {failed} failed.")


# ── dataset build + normalization ─────────────────────────────────────────────

def build_dataset(label_map, takes_by_class, val_takes, augs_per_take):
    """
    Load cached features into memory.
    Returns (train_feats, train_labels, val_feats, val_labels) with features
    as float32 (T,D) arrays and labels as label_map indices.
    """
    train_feats, train_labels = [], []
    val_feats, val_labels = [], []
    missing = 0
    for ci, cid in enumerate(label_map):
        vt = val_takes[cid]
        for take in takes_by_class[cid]:
            for j in aug_range_for(take, vt, augs_per_take):
                p = feat_path(cid, take, j)
                if not os.path.isfile(p):
                    missing += 1
                    continue
                f = np.load(p).astype(np.float32)
                if f.ndim != 2 or f.shape[0] < 1:
                    missing += 1
                    continue
                if take == vt:
                    val_feats.append(f)
                    val_labels.append(ci)
                else:
                    train_feats.append(f)
                    train_labels.append(ci)
    if missing:
        print(f"  [!] {missing} feature files missing/invalid "
              "(failed extractions) - skipped")
    if not train_feats:
        sys.exit("ERROR: no training features available - "
                 "check the extraction errors above.")
    print(f"  Dataset: {len(train_feats)} train clips, "
          f"{len(val_feats)} val clips, dim={train_feats[0].shape[1]}")
    return train_feats, train_labels, val_feats, val_labels


def compute_norm_stats(train_feats):
    """Per-dim mean/std (float32) over TRAIN features only, frame-wise."""
    d = train_feats[0].shape[1]
    s = np.zeros(d, dtype=np.float64)
    ss = np.zeros(d, dtype=np.float64)
    n = 0
    for f in train_feats:
        f64 = f.astype(np.float64)
        s += f64.sum(axis=0)
        ss += (f64 ** 2).sum(axis=0)
        n += f.shape[0]
    mean = s / n
    var = np.maximum(ss / n - mean ** 2, 0.0)
    std = np.maximum(np.sqrt(var), STD_FLOOR)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_inplace(feats, mean, std):
    for i in range(len(feats)):
        feats[i] = ((feats[i] - mean) / std).astype(np.float32)


# ── training ──────────────────────────────────────────────────────────────────

def make_batch(feats, labels, idxs, device):
    """Pad a list of (T,D) clips to the batch max T. Returns (x, y, lengths)."""
    lens = [feats[i].shape[0] for i in idxs]
    t_max = max(lens)
    d = feats[idxs[0]].shape[1]
    x = torch.zeros(len(idxs), t_max, d, dtype=torch.float32)
    for bi, i in enumerate(idxs):
        x[bi, :lens[bi]] = torch.from_numpy(feats[i])
    y = torch.tensor([labels[i] for i in idxs], dtype=torch.long)
    lengths = torch.tensor(lens, dtype=torch.long)
    return x.to(device), y.to(device), lengths.to(device)


@torch.no_grad()
def evaluate(model, feats, labels, batch_size, device):
    """Returns (top1_preds (N,), top3_hit (N,) 0/1) over the given split."""
    model.eval()
    preds, top3 = [], []
    for i0 in range(0, len(feats), batch_size):
        idxs = list(range(i0, min(i0 + batch_size, len(feats))))
        x, _, lengths = make_batch(feats, labels, idxs, device)
        logits = model(x, lengths)
        k = min(3, logits.shape[1])
        topk = logits.topk(k, dim=1).indices.cpu().numpy()
        preds.extend(int(t[0]) for t in topk)
        top3.extend(int(labels[i] in topk[bi]) for bi, i in enumerate(idxs))
    return np.asarray(preds), np.asarray(top3)


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_head(train_feats, train_labels, val_feats, val_labels,
               num_classes, args, device):
    """Returns (config, best_state_dict, best_val_top1 or None)."""
    d = train_feats[0].shape[1]
    config = {'input_dim': d, 'hidden': HIDDEN,
              'num_classes': num_classes, 'dropout': DROPOUT}
    model = PronunciationHead(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    have_val = len(val_feats) > 0
    if not have_val:
        print("  [!] no validation clips (all classes single-take): "
              "training full schedule, keeping the final model.")

    print(f"\nTraining PronunciationHead on {device} "
          f"(dim={d}, classes={num_classes}, {len(train_feats)} clips)\n")

    best_acc, best_state, bad_epochs = -1.0, None, 0
    n = len(train_feats)
    for epoch in range(1, args.epochs + 1):
        model.train()
        lr_now = optimizer.param_groups[0]['lr']
        perm = np.random.permutation(n)
        loss_sum, seen = 0.0, 0
        for i0 in range(0, n, args.batch_size):
            idxs = perm[i0:i0 + args.batch_size].tolist()
            x, y, lengths = make_batch(train_feats, train_labels, idxs, device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(idxs)
            seen += len(idxs)
        scheduler.step()

        line = (f"  epoch {epoch:3d}/{args.epochs}  lr {lr_now:.2e}  "
                f"loss {loss_sum / max(seen, 1):.4f}")
        if have_val:
            preds, _ = evaluate(model, val_feats, val_labels,
                                args.batch_size, device)
            acc = float((preds == np.asarray(val_labels)).mean())
            line += f"  val_top1 {acc * 100:5.1f}%"
            if acc > best_acc:
                best_acc, best_state, bad_epochs = acc, cpu_state_dict(model), 0
                line += "  *best*"
            else:
                bad_epochs += 1
            print(line)
            if bad_epochs >= PATIENCE:
                print(f"  Early stop: no val improvement for {PATIENCE} epochs.")
                break
        else:
            print(line)

    if best_state is None:                      # no-val path: keep final weights
        best_state = cpu_state_dict(model)
        return config, best_state, None
    return config, best_state, best_acc


# ── report ────────────────────────────────────────────────────────────────────

def class_display(cid, items_by_id):
    if cid == SILENCE_ID:
        return f'{SILENCE_ID} (idle face)'
    it = items_by_id.get(cid)
    return f"{cid} {it['pinyin']} {it['hanzi']}" if it else cid


def print_report(model, label_map, items_by_id, val_feats, val_labels,
                 batch_size, device):
    """Best-model validation report: top-1/3, per-class, confusion pairs."""
    print(f"\n{'=' * 62}\nValidation report (best model)\n{'=' * 62}")
    if not val_feats:
        print("  No validation clips - record a 2nd take per item to enable "
              "validation metrics.")
        return

    preds, top3 = evaluate(model, val_feats, val_labels, batch_size, device)
    truth = np.asarray(val_labels)
    print(f"  val top-1 accuracy: {float((preds == truth).mean()) * 100:.1f}%   "
          f"top-3 accuracy: {float(top3.mean()) * 100:.1f}%   "
          f"({len(truth)} clips)")

    print("\n  Per-class validation:")
    for ci, cid in enumerate(label_map):
        rows = np.where(truth == ci)[0]
        if len(rows) == 0:
            continue
        ok = int((preds[rows] == ci).sum())
        mark = 'ok ' if ok == len(rows) else 'BAD'
        wrong = ''
        if ok < len(rows):
            got = [class_display(label_map[p], items_by_id)
                   for p in preds[rows] if p != ci]
            wrong = f"  -> predicted: {', '.join(got)}"
        print(f"    [{mark}] {class_display(cid, items_by_id):24s} "
              f"{ok}/{len(rows)}{wrong}")

    pairs = Counter((int(t), int(p)) for t, p in zip(truth, preds) if t != p)
    if pairs:
        print("\n  Top confusion pairs (true -> predicted) - these lip shapes "
              "look alike on your mouth:")
        for (t, p), c in pairs.most_common(10):
            print(f"    {class_display(label_map[t], items_by_id):24s} -> "
                  f"{class_display(label_map[p], items_by_id):24s} x{c}")
        print("\n  Hint: confused items often improve after re-recording with "
              "clearer, more exaggerated\n  lip shapes: "
              "python record_pronunciations.py --redo <item_id>")
    else:
        print("\n  No confusions - every validation clip was classified "
              "correctly.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Augment recordings, extract AV-HuBERT features and train '
                    'the pronunciation classifier head.')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--augs-per-take', type=int, default=8,
                        help='augmented variants per TRAIN take (plus aug0)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'],
                        default='auto')
    parser.add_argument('--extended', action='store_true',
                        help='include the 8 extended curriculum items')
    parser.add_argument('--force-extract', action='store_true',
                        help='ignore the feature cache and re-extract everything')
    parser.add_argument('--val-take', type=int, default=3,
                        help='take held out for validation (fallback: highest)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args([a for a in sys.argv[1:] if a != '_run_'])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ensure_dirs()
    t_start = time.time()

    # 1) scan recordings, decide classes + validation takes
    items = get_items(extended=args.extended)
    items_by_id = {it['id']: it for it in items}
    label_map, takes_by_class = scan_recordings(items)
    val_takes = choose_val_takes(label_map, takes_by_class, args.val_take)
    remove_stale_val_augs(label_map, val_takes)

    # 2) augmentation + AV-HuBERT feature extraction (cached)
    extract_all_features(label_map, takes_by_class, val_takes, args)

    # 3) load features, normalize with TRAIN-only statistics
    train_feats, train_labels, val_feats, val_labels = build_dataset(
        label_map, takes_by_class, val_takes, args.augs_per_take)
    feat_mean, feat_std = compute_norm_stats(train_feats)
    normalize_inplace(train_feats, feat_mean, feat_std)
    normalize_inplace(val_feats, feat_mean, feat_std)

    # 4) train the classifier head
    if args.device == 'auto':
        head_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        head_device = args.device
    config, best_state, best_acc = train_head(
        train_feats, train_labels, val_feats, val_labels,
        len(label_map), args, head_device)

    # 5) report with the best weights
    best_model = PronunciationHead(**config)
    best_model.load_state_dict(best_state)
    best_model.eval().to(head_device)
    if best_acc is not None:
        print(f"\nBest val top-1 during training: {best_acc * 100:.1f}%")
    print_report(best_model, label_map, items_by_id, val_feats, val_labels,
                 args.batch_size, head_device)

    # 6) save checkpoint
    best_model.cpu()
    os.makedirs(os.path.dirname(HEAD_CKPT), exist_ok=True)
    save_head_checkpoint(HEAD_CKPT, best_model, label_map, config,
                         feat_mean, feat_std)
    print(f"\nSaved checkpoint: {HEAD_CKPT} "
          f"({len(label_map)} classes, dim={config['input_dim']})")
    print(f"Total time: {time.time() - t_start:.0f}s")
    print("\nNext step - realtime lip reading:\n"
          "  python realtime_pronunciation.py")


if __name__ == '__main__':
    main()
