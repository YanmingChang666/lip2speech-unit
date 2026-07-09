"""
Mandarin pronunciation curriculum for lip-shape recording.

50 core items designed for maximum lip-shape (viseme) coverage:
  - all 21 initials  (b p m f d t n l g k h j q x zh ch sh r z c s)
  - all 6 simple finals (a o e i u v) + major compound/nasal finals
    (ai ei ao ou an en ang eng ong er ia ie iao iou uo uei uan ve van)
Each item is a common one-syllable word so TTS pronounces it naturally.

Usage:
  python pronunciation_curriculum.py                  # list items + TTS cache status
  python pronunciation_curriculum.py --generate-tts   # build TTS wav cache (needs edge-tts + network once)
  python pronunciation_curriculum.py --play ba        # test one item's audio
  python pronunciation_curriculum.py --extended       # include 8 extra items in the above

TTS cache: datasets/pronunciation/tts/{item_id}.wav  (16 kHz mono)
"""

import os
import sys
import argparse

from pronunciation_common import (
    TTS_DIR, SAMPLE_RATE, ensure_dirs, play_wav,
)

# ── Curriculum ────────────────────────────────────────────────────────────────
# fields: id (ASCII pinyin, v = u-umlaut), pinyin (tone-marked, for display),
#         hanzi (spoken by TTS), group, viseme (lip-shape category), hint
#
# Group A - simple/basic finals, zero initial (12): every base mouth shape
# Group B - all 21 initials, finals chosen to maximize shape contrast (21)
# Group C - compound / glide / nasal finals (17): shape-transition coverage

ITEMS = [
    # ── A: simple finals ──────────────────────────────────────────────────────
    dict(id='a',    pinyin='ā',    hanzi='啊', group='simple-final', viseme='open',
         hint='张大嘴 open wide'),
    dict(id='o',    pinyin='ó',    hanzi='哦', group='simple-final', viseme='round',
         hint='嘴唇收圆 round lips'),
    dict(id='e',    pinyin='é',    hanzi='鹅', group='simple-final', viseme='mid',
         hint='嘴角放松微开 relaxed half-open'),
    dict(id='yi',   pinyin='yī',   hanzi='一', group='simple-final', viseme='spread',
         hint='嘴角向两边展开 spread lips'),
    dict(id='wu',   pinyin='wǔ',   hanzi='五', group='simple-final', viseme='round-tight',
         hint='嘴唇拢圆突出 tight rounded lips'),
    dict(id='yu',   pinyin='yú',   hanzi='鱼', group='simple-final', viseme='round-front',
         hint='圆唇前突 rounded protruded'),
    dict(id='ai',   pinyin='ài',   hanzi='爱', group='simple-final', viseme='open>spread', hint=''),
    dict(id='ao',   pinyin='ào',   hanzi='奥', group='simple-final', viseme='open>round', hint=''),
    dict(id='ou',   pinyin='ōu',   hanzi='欧', group='simple-final', viseme='mid>round', hint=''),
    dict(id='an',   pinyin='ān',   hanzi='安', group='simple-final', viseme='open-nasal', hint=''),
    dict(id='ang',  pinyin='áng',  hanzi='昂', group='simple-final', viseme='open-nasal', hint=''),
    dict(id='er',   pinyin='èr',   hanzi='二', group='simple-final', viseme='mid-retroflex', hint=''),

    # ── B: all 21 initials ────────────────────────────────────────────────────
    dict(id='ba',   pinyin='bā',   hanzi='八', group='initial', viseme='bilabial+open',
         hint='双唇闭合再张开 lips close then open'),
    dict(id='po',   pinyin='pō',   hanzi='坡', group='initial', viseme='bilabial+round', hint=''),
    dict(id='mi',   pinyin='mǐ',   hanzi='米', group='initial', viseme='bilabial+spread', hint=''),
    dict(id='fu',   pinyin='fú',   hanzi='福', group='initial', viseme='labiodental',
         hint='上齿咬下唇 teeth on lower lip'),
    dict(id='da',   pinyin='dà',   hanzi='大', group='initial', viseme='alveolar+open', hint=''),
    dict(id='ti',   pinyin='tí',   hanzi='提', group='initial', viseme='alveolar+spread', hint=''),
    dict(id='nv',   pinyin='nǚ',   hanzi='女', group='initial', viseme='alveolar+round-front', hint=''),
    dict(id='le',   pinyin='lè',   hanzi='乐', group='initial', viseme='alveolar+mid', hint=''),
    dict(id='ge',   pinyin='gē',   hanzi='哥', group='initial', viseme='velar+mid', hint=''),
    dict(id='ku',   pinyin='kū',   hanzi='哭', group='initial', viseme='velar+round', hint=''),
    dict(id='he',   pinyin='hē',   hanzi='喝', group='initial', viseme='velar+mid', hint=''),
    dict(id='ji',   pinyin='jī',   hanzi='鸡', group='initial', viseme='palatal+spread', hint=''),
    dict(id='qu',   pinyin='qù',   hanzi='去', group='initial', viseme='palatal+round-front', hint=''),
    dict(id='xie',  pinyin='xiè',  hanzi='谢', group='initial', viseme='palatal+glide', hint=''),
    dict(id='zhu',  pinyin='zhū',  hanzi='猪', group='initial', viseme='retroflex+round', hint=''),
    dict(id='chi',  pinyin='chī',  hanzi='吃', group='initial', viseme='retroflex+spread', hint=''),
    dict(id='shu',  pinyin='shū',  hanzi='书', group='initial', viseme='retroflex+round', hint=''),
    dict(id='ri',   pinyin='rì',   hanzi='日', group='initial', viseme='retroflex+spread', hint=''),
    dict(id='zi',   pinyin='zì',   hanzi='字', group='initial', viseme='dental+spread', hint=''),
    dict(id='ca',   pinyin='cā',   hanzi='擦', group='initial', viseme='dental+open', hint=''),
    dict(id='si',   pinyin='sì',   hanzi='四', group='initial', viseme='dental+spread', hint=''),

    # ── C: compound / glide / nasal finals ────────────────────────────────────
    dict(id='bai',  pinyin='bái',  hanzi='白', group='compound-final', viseme='bilabial+open>spread', hint=''),
    dict(id='mei',  pinyin='měi',  hanzi='美', group='compound-final', viseme='bilabial+mid>spread', hint=''),
    dict(id='dao',  pinyin='dào',  hanzi='道', group='compound-final', viseme='open>round', hint=''),
    dict(id='tou',  pinyin='tóu',  hanzi='头', group='compound-final', viseme='mid>round', hint=''),
    dict(id='lan',  pinyin='lán',  hanzi='蓝', group='compound-final', viseme='open-nasal', hint=''),
    dict(id='gen',  pinyin='gēn',  hanzi='根', group='compound-final', viseme='mid-nasal', hint=''),
    dict(id='mang', pinyin='máng', hanzi='忙', group='compound-final', viseme='bilabial+open-nasal', hint=''),
    dict(id='feng', pinyin='fēng', hanzi='风', group='compound-final', viseme='labiodental+mid-nasal', hint=''),
    dict(id='hong', pinyin='hóng', hanzi='红', group='compound-final', viseme='round-nasal', hint=''),
    dict(id='jia',  pinyin='jiā',  hanzi='家', group='compound-final', viseme='spread>open', hint=''),
    dict(id='yao',  pinyin='yào',  hanzi='要', group='compound-final', viseme='spread>open>round', hint=''),
    dict(id='niu',  pinyin='niú',  hanzi='牛', group='compound-final', viseme='spread>round', hint=''),
    dict(id='duo',  pinyin='duō',  hanzi='多', group='compound-final', viseme='round>mid', hint=''),
    dict(id='shui', pinyin='shuǐ', hanzi='水', group='compound-final', viseme='round>spread', hint=''),
    dict(id='chuan', pinyin='chuán', hanzi='船', group='compound-final', viseme='round>open-nasal', hint=''),
    dict(id='yue',  pinyin='yuè',  hanzi='月', group='compound-final', viseme='round-front>mid', hint=''),
    dict(id='yuan', pinyin='yuán', hanzi='圆', group='compound-final', viseme='round-front>open-nasal', hint=''),
]

# Optional extras for near-exhaustive final coverage (--extended)
EXTENDED_ITEMS = [
    dict(id='yin',  pinyin='yīn',  hanzi='音', group='extended', viseme='spread-nasal', hint=''),
    dict(id='ying', pinyin='yīng', hanzi='鹰', group='extended', viseme='spread-nasal', hint=''),
    dict(id='yang', pinyin='yáng', hanzi='羊', group='extended', viseme='spread>open-nasal', hint=''),
    dict(id='yong', pinyin='yòng', hanzi='用', group='extended', viseme='spread>round-nasal', hint=''),
    dict(id='wa',   pinyin='wā',   hanzi='蛙', group='extended', viseme='round>open', hint=''),
    dict(id='wai',  pinyin='wài',  hanzi='外', group='extended', viseme='round>open>spread', hint=''),
    dict(id='wang', pinyin='wáng', hanzi='王', group='extended', viseme='round>open-nasal', hint=''),
    dict(id='yun',  pinyin='yún',  hanzi='云', group='extended', viseme='round-front-nasal', hint=''),
]

assert len(ITEMS) == 50, f"curriculum should have 50 core items, got {len(ITEMS)}"
assert len({it['id'] for it in ITEMS + EXTENDED_ITEMS}) == len(ITEMS) + len(EXTENDED_ITEMS), \
    "duplicate item ids"


def get_items(extended=False):
    return ITEMS + (EXTENDED_ITEMS if extended else [])


def item_by_id(item_id, extended=True):
    for it in get_items(extended):
        if it['id'] == item_id:
            return it
    return None


def tts_path(item_id):
    return f'{TTS_DIR}/{item_id}.wav'


# ── TTS generation (edge-tts -> 16 kHz wav cache) ─────────────────────────────

def _mp3_to_wav(mp3_path, wav_path, sr=SAMPLE_RATE):
    """Convert mp3 to mono 16k wav. Tries ffmpeg CLI first, then librosa."""
    import subprocess
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-i', mp3_path,
             '-ac', '1', '-ar', str(sr), wav_path],
            check=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    import librosa
    import soundfile as sf
    audio, _ = librosa.load(mp3_path, sr=sr, mono=True)
    sf.write(wav_path, audio, sr)


def generate_tts(items, voice='zh-CN-XiaoxiaoNeural', rate='-20%', overwrite=False):
    """
    Generate teaching audio for every item via edge-tts (Microsoft neural TTS).
    Slightly slowed (rate=-20%) so learners hear the syllable clearly.
    Requires network on first run; results are cached as wav.
    """
    try:
        import edge_tts
    except ImportError:
        sys.exit("edge-tts not installed. Run: pip install edge-tts")
    import asyncio
    import tempfile

    ensure_dirs()

    async def gen_one(item):
        wav = tts_path(item['id'])
        if os.path.isfile(wav) and not overwrite:
            return 'cached'
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            mp3 = f.name
        part = wav + '.part.wav'      # atomic: a failed run never looks 'cached'
        try:
            com = edge_tts.Communicate(item['hanzi'], voice, rate=rate)
            await com.save(mp3)
            _mp3_to_wav(mp3, part)
            os.replace(part, wav)
        finally:
            for p in (mp3, part):
                if os.path.isfile(p):
                    os.remove(p)
        return 'generated'

    async def gen_all():
        done = 0
        for item in items:
            status = await gen_one(item)
            done += 1
            print(f"[{done}/{len(items)}] {item['id']:6s} {item['hanzi']} ({item['pinyin']}) - {status}")

    asyncio.run(gen_all())
    print(f"\nTTS cache complete: {TTS_DIR}")


def check_tts_cache(items):
    """Returns list of item ids missing from the TTS cache."""
    return [it['id'] for it in items if not os.path.isfile(tts_path(it['id']))]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--generate-tts', action='store_true',
                        help='generate the TTS wav cache via edge-tts')
    parser.add_argument('--overwrite', action='store_true',
                        help='regenerate existing TTS files too')
    parser.add_argument('--voice', default='zh-CN-XiaoxiaoNeural')
    parser.add_argument('--rate', default='-20%',
                        help="TTS speed adjustment; pass with '=' for negative "
                             "values, e.g. --rate=-15%%")
    parser.add_argument('--extended', action='store_true',
                        help='include the 8 extended items')
    parser.add_argument('--play', metavar='ITEM_ID',
                        help='play one cached item, e.g. --play ba')
    args = parser.parse_args([a for a in sys.argv[1:] if a != '_run_'])

    items = get_items(args.extended)

    if args.play:
        item = item_by_id(args.play)
        if item is None:
            sys.exit(f"Unknown item id: {args.play}")
        wav = tts_path(item['id'])
        if not os.path.isfile(wav):
            sys.exit(f"No TTS cache for '{item['id']}'. Run --generate-tts first.")
        print(f"Playing {item['id']} {item['hanzi']} ({item['pinyin']})")
        play_wav(wav)
        return

    if args.generate_tts:
        generate_tts(items, voice=args.voice, rate=args.rate,
                     overwrite=args.overwrite)
        return

    missing = check_tts_cache(items)
    print(f"Curriculum: {len(items)} items "
          f"({len(ITEMS)} core{' + 8 extended' if args.extended else ''})\n")
    for it in items:
        mark = ' ' if it['id'] not in missing else '!'
        hint = f"  [{it['hint']}]" if it['hint'] else ''
        print(f" {mark} {it['id']:6s} {it['pinyin']:6s} {it['hanzi']}  "
              f"{it['group']:15s} {it['viseme']}{hint}")
    if missing:
        print(f"\n{len(missing)} items missing TTS audio (marked '!').")
        print("Run: python pronunciation_curriculum.py --generate-tts")
    else:
        print("\nTTS cache complete.")


if __name__ == '__main__':
    main()
