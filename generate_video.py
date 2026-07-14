#!/usr/bin/env python3
"""
Mikasan Narrator - long-form cartoon generator
-----------------------------------------------
Turns a plain-text script into a video narrated by a stable, natural free TTS voice
(edge-tts), with the fixed Mikasan character (assets/character/character.png), a
solid tan background, and typewriter-style accumulating captions -- all free, no API
keys.

HOW IT WORKS:
  1. Narration is synthesized with edge-tts (Microsoft's free neural TTS) -- switched
     from cloned-voice XTTS-v2, which was cracking/unstable on longer lines.
  2. The character is a FIXED reference image (assets/character/character.png), held
     static and placed on alternating left/right sides per scene.
  3. The background is a solid flat color (BACKGROUND_COLOR below), matching the
     reference art style.
  4. Captions build up word by word (typewriter/karaoke style) and ACCUMULATE across
     multiple lines until the caption block is full, then clear and restart.
     Rendered on the right side, lower on the frame (comic speech-bubble style), in a
     soft rounded font (Baloo 2) with content words highlighted in yellow.

NOTE ON CHARACTER VARIETY: right now there's only one pose (character.png), so the
character is fully static. If you add more pose images (e.g.
assets/character/pose_gesture.png for a hand-raised pose), it's a small change to
swap between them periodically.

Usage:
    python generate_video.py --script scripts/example_script.txt --out output/final_video.mp4
"""

import argparse
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

CANVAS_W, CANVAS_H = 1920, 1080
FPS = 25
DEFAULT_VOICE_NAME = "en-US-AriaNeural"  # stable, natural free neural voice (edge-tts)
CHARACTER_PATH = Path(__file__).parent / "assets" / "character" / "character.png"

# Sampled directly from Saba's reference art
BACKGROUND_COLOR = (246, 225, 200)

# Captions were lagging behind the voice -- pull each word's reveal time earlier and
# compress the pacing a bit so text keeps up with (or slightly leads) speech.
CAPTION_LEAD_TIME = 0.35
CAPTION_SPEED_FACTOR = 0.78

# Right side, lower than a top banner -- comic speech-bubble placement.
CAPTION_TOP_Y = 300
CAPTION_RIGHT_MARGIN = 90
CAPTION_BOX_WIDTH = 760
CAPTION_MAX_LINES = 4
CAPTION_FONT_SIZE = 50
CAPTION_FONT_WEIGHT = 600  # Baloo 2 variable-weight axis value (semi-bold)
CAPTION_FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/baloo2/Baloo2%5Bwght%5D.ttf"
CAPTION_FONT_FALLBACKS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
# Small words rendered in plain white; everything else highlighted yellow,
# matching the reference caption mockup (HELLO plain, EXCITED/LEARN/EXPLORE yellow).
CAPTION_STOPWORDS = {
    "a", "an", "the", "is", "it", "to", "of", "in", "on", "at", "for", "and", "but",
    "or", "so", "i", "i'm", "im", "you", "your", "we", "us", "me", "my", "with",
    "let's", "lets", "that", "this", "be", "was", "were", "are", "am", "just",
}


def parse_script(path: str):
    raw = Path(path).read_text(encoding="utf-8")
    blocks = raw.split("---")

    settings = {"voice_name": DEFAULT_VOICE_NAME, "music": None}
    header = blocks[0].strip()
    for line in header.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("VOICE:"):
            val = line.split(":", 1)[1].strip()
            settings["voice_name"] = val or DEFAULT_VOICE_NAME
        elif line.upper().startswith("MUSIC:"):
            val = line.split(":", 1)[1].strip()
            settings["music"] = val or None

    scenes = []
    for block in blocks[1:]:
        block = block.strip()
        if not block:
            continue
        lines = [l for l in block.splitlines() if l.strip()]
        narration_lines = []
        for line in lines:
            if line.upper().startswith("IMAGE:"):
                continue
            narration_lines.append(line.strip())
        narration = " ".join(narration_lines).strip()
        if not narration:
            continue
        scenes.append({"narration": narration})

    if not scenes:
        raise ValueError("No scenes found. Separate scenes with '---' lines.")
    return settings, scenes


class Narrator:
    """Free, stable neural TTS via edge-tts (Microsoft). Not a cloned voice --
    switched from XTTS-v2 cloning, which was cracking on longer narration lines."""

    def __init__(self, voice_name: str = DEFAULT_VOICE_NAME):
        self.voice_name = voice_name

    def synth(self, text: str, out_wav: Path):
        import edge_tts

        async def _run():
            communicate = edge_tts.Communicate(text, self.voice_name)
            await communicate.save(str(out_wav))

        asyncio.run(_run())


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}\n{result.stderr[-3000:]}")
    return result


def get_duration(path: Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(result.stdout.strip())


def load_character() -> Image.Image:
    if not CHARACTER_PATH.exists():
        raise FileNotFoundError(
            f"Character art not found at {CHARACTER_PATH}. Upload character.png to "
            "assets/character/ (via GitHub's Upload files UI) before running this."
        )
    return Image.open(CHARACTER_PATH).convert("RGBA")


def load_caption_font(workdir: Path, size: int = CAPTION_FONT_SIZE):
    """Downloads (once) and loads Baloo 2, a soft rounded font, set to a semi-bold
    weight. Falls back to a system bold font if the download fails."""
    font_path = workdir / "Baloo2.ttf"
    try:
        if not font_path.exists():
            resp = requests.get(CAPTION_FONT_URL, timeout=30)
            resp.raise_for_status()
            font_path.write_bytes(resp.content)
        font = ImageFont.truetype(str(font_path), size)
        try:
            font.set_variation_by_axes([CAPTION_FONT_WEIGHT])
        except Exception:  # noqa: BLE001
            pass
        return font
    except Exception as e:  # noqa: BLE001
        print(f"  [font] Baloo 2 download/load failed ({e}), falling back to a system font.")
        for path in CAPTION_FONT_FALLBACKS:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()


def is_highlighted(word: str) -> bool:
    stripped = word.strip(".,!?;:\"'").lower()
    return stripped not in CAPTION_STOPWORDS and len(stripped) > 0


def build_caption_layout(narration: str, duration: float, font):
    words = narration.split()
    if not words:
        return []

    per_word = (duration / len(words)) * CAPTION_SPEED_FACTOR

    measure_img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    measure_draw = ImageDraw.Draw(measure_img)
    space_w = measure_draw.textbbox((0, 0), "  ", font=font, stroke_width=5)[2]

    pages = []
    current_page_words = []
    line_idx = 0
    cursor_x = 0
    t = 0.0

    def start_new_page():
        nonlocal current_page_words, line_idx, cursor_x
        if current_page_words:
            pages.append({"words": current_page_words, "page_start": current_page_words[0]["start"]})
        current_page_words = []
        line_idx = 0
        cursor_x = 0

    for w in words:
        disp = w.upper()
        word_w = measure_draw.textbbox((0, 0), disp, font=font, stroke_width=5)[2]
        if cursor_x > 0 and cursor_x + word_w > CAPTION_BOX_WIDTH:
            line_idx += 1
            cursor_x = 0
            if line_idx >= CAPTION_MAX_LINES:
                start_new_page()

        start_time = max(0.0, t - CAPTION_LEAD_TIME)
        current_page_words.append({"word": w, "start": start_time, "line": line_idx, "x": cursor_x})
        cursor_x += word_w + space_w
        t += per_word

    if current_page_words:
        pages.append({"words": current_page_words, "page_start": current_page_words[0]["start"]})

    return pages


def active_page_words_at(pages, t: float):
    active = None
    for page in pages:
        if page["page_start"] <= t:
            active = page
        else:
            break
    if active is None:
        return None
    return [w for w in active["words"] if w["start"] <= t]


def render_caption_overlay(words: list, canvas_w: int, canvas_h: int, font) -> Image.Image:
    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    if not words:
        return overlay
    draw = ImageDraw.Draw(overlay)

    line_height = int(CAPTION_FONT_SIZE * 1.35)
    box_left = canvas_w - CAPTION_RIGHT_MARGIN - CAPTION_BOX_WIDTH
    max_line = max(w["line"] for w in words)
    box_height = (max_line + 1) * line_height

    pad = 22
    bar_box = (box_left - pad, CAPTION_TOP_Y - pad, box_left + CAPTION_BOX_WIDTH + pad,
               CAPTION_TOP_Y + box_height + pad)
    draw.rounded_rectangle(bar_box, radius=22, fill=(15, 15, 20, 185))

    for w in words:
        color = (255, 214, 10, 255) if is_highlighted(w["word"]) else (255, 255, 255, 255)
        x = box_left + w["x"]
        y = CAPTION_TOP_Y + w["line"] * line_height
        draw.text((x, y), w["word"].upper(), font=font, fill=color,
                   stroke_width=5, stroke_fill=(20, 20, 25, 255))

    return overlay


def render_scene(char_img: Image.Image, pages, side: str, duration: float,
                  audio_path: Path, out_path: Path, font, char_scale=0.85, fps=FPS):
    char_h = int(CANVAS_H * char_scale)
    char_w = int(char_h * char_img.width / char_img.height)
    margin = int(CANVAS_W * 0.05)
    anchor_x = margin if side == "left" else CANVAS_W - char_w - margin
    anchor_y = CANVAS_H - char_h + 20

    resized_char = char_img.resize((char_w, char_h), Image.LANCZOS)
    total_frames = max(int(duration * fps), fps)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(fps),
        "-i", "-", "-i", str(audio_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(out_path), "-loglevel", "error",
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    base_frame = Image.new("RGBA", (CANVAS_W, CANVAS_H), (*BACKGROUND_COLOR, 255))
    base_frame.alpha_composite(resized_char, (anchor_x, anchor_y))

    last_words_key = None
    caption_overlay = None

    for i in range(total_frames):
        t = i / fps

        revealed = active_page_words_at(pages, t)
        revealed_key = tuple((w["word"], w["line"]) for w in revealed) if revealed else None
        if revealed_key != last_words_key:
            caption_overlay = render_caption_overlay(revealed or [], CANVAS_W, CANVAS_H, font)
            last_words_key = revealed_key

        frame = base_frame if not revealed else base_frame.copy()
        if revealed and caption_overlay is not None:
            frame = base_frame.copy()
            frame.alpha_composite(caption_overlay, (0, 0))

        proc.stdin.write(frame.convert("RGB").tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame encoding failed for {out_path}")


def concat_clips(clip_paths, out_path: Path, workdir: Path):
    list_file = workdir / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in clip_paths), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy",
         str(out_path), "-loglevel", "error"])


def add_background_music(video_path: Path, music_path: Path, out_path: Path, volume=0.12):
    run([
        "ffmpeg", "-y", "-i", str(video_path), "-stream_loop", "-1", "-i", str(music_path),
        "-filter_complex",
        f"[1:a]volume={volume}[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-shortest", str(out_path), "-loglevel", "error",
    ])


def main():
    ap = argparse.ArgumentParser(description="Generate a Mikasan-narrated cartoon with a static character, solid background, and typewriter captions.")
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", default="output/final_video.mp4")
    ap.add_argument("--voice-name", default=None, help="edge-tts voice name, e.g. en-US-AriaNeural")
    ap.add_argument("--music", default=None)
    ap.add_argument("--character-scale", type=float, default=0.85)
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    settings, scenes = parse_script(args.script)
    voice_name = args.voice_name or settings["voice_name"]
    music = args.music or settings["music"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = out_path.parent / "_work"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    print(f"Loaded {len(scenes)} scene(s). Voice: {voice_name}. Music: {music or 'none'}")
    char_img = load_character()
    font = load_caption_font(workdir)
    narrator = Narrator(voice_name)

    clip_paths = []
    for i, scene in enumerate(scenes):
        print(f"\n[{i+1}/{len(scenes)}] {scene['narration'][:70]}...")
        audio_wav = workdir / f"scene_{i:03}.wav"
        clip_path = workdir / f"scene_{i:03}.mp4"
        side = "left" if i % 2 == 0 else "right"

        print("  -> synthesizing narration (edge-tts)")
        narrator.synth(scene["narration"], audio_wav)
        duration = get_duration(audio_wav) + 0.3

        pages = build_caption_layout(scene["narration"], duration, font)

        print(f"  -> rendering scene ({side} side, {duration:.1f}s, {int(duration*FPS)} frames)")
        render_scene(char_img, pages, side, duration, audio_wav, clip_path, font,
                     char_scale=args.character_scale)
        clip_paths.append(clip_path)

    print("\nConcatenating all scenes...")
    no_music_path = workdir / "combined_no_music.mp4"
    concat_clips(clip_paths, no_music_path, workdir)

    if music and Path(music).exists():
        print("Mixing background music...")
        add_background_music(no_music_path, Path(music), out_path)
    else:
        if music:
            print(f"  (music file '{music}' not found, skipping)")
        shutil.copy(no_music_path, out_path)

    if not args.keep_temp:
        shutil.rmtree(workdir, ignore_errors=True)

    total_duration = get_duration(out_path)
    print(f"\nDone! Video saved to: {out_path} ({total_duration:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
