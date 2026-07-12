#!/usr/bin/env python3
"""
Mikasan Narrator - long-form cartoon generator
-----------------------------------------------
Turns a plain-text script into a video narrated in Saba's own cloned voice, with the
fixed Mikasan character (assets/character/character.png), a solid tan background, and
karaoke-style animated captions synced to the narration -- all free, no API keys.

HOW IT WORKS:
  1. The character is a FIXED reference image (assets/character/character.png) -- not
     regenerated per scene. It's placed on alternating left/right sides per scene,
     with subtle sway/breathing motion so it doesn't feel like a static cutout.
  2. The background is a solid flat color (BACKGROUND_COLOR below), matching the
     reference art style -- not an AI-generated scene image.
  3. Captions are built from the narration text, timed proportionally across the
     scene's audio duration, and revealed progressively WORD BY WORD as if being
     typed/spoken (karaoke style), rendered near the top of the frame in bold
     uppercase with a black stroke outline. Content words are highlighted in yellow,
     matching the reference caption mockup.

Usage:
    python generate_video.py --script scripts/example_script.txt --out output/final_video.mp4
"""

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CANVAS_W, CANVAS_H = 1920, 1080
FPS = 25
DEFAULT_VOICE_SAMPLE = "voice_samples/Bee.m4a"
CHARACTER_PATH = Path(__file__).parent / "assets" / "character" / "character.png"

# Sampled directly from Saba's reference art
BACKGROUND_COLOR = (246, 225, 200)

CAPTION_WORDS_PER_LINE = 6
CAPTION_FONT_CANDIDATES = [
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

    settings = {"voice_sample": DEFAULT_VOICE_SAMPLE, "music": None}
    header = blocks[0].strip()
    for line in header.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("VOICE_SAMPLE:"):
            val = line.split(":", 1)[1].strip()
            settings["voice_sample"] = val or DEFAULT_VOICE_SAMPLE
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
                continue  # image prompts are no longer used (fixed background)
            narration_lines.append(line.strip())
        narration = " ".join(narration_lines).strip()
        if not narration:
            continue
        scenes.append({"narration": narration})

    if not scenes:
        raise ValueError("No scenes found. Separate scenes with '---' lines.")
    return settings, scenes


class ClonedVoice:
    def __init__(self, speaker_wav: Path, language: str = "en"):
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS
        print("  Loading XTTS-v2 (first run downloads ~1.9GB model)...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.speaker_wav = str(speaker_wav)
        self.language = language

    def synth(self, text: str, out_wav: Path):
        self.tts.tts_to_file(
            text=text, speaker_wav=self.speaker_wav, language=self.language, file_path=str(out_wav),
        )


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


def build_caption_lines(narration: str, duration: float):
    """Splits narration into on-screen lines, each with per-word start times (relative
    to the scene start) so words can be revealed progressively as if being spoken."""
    words = narration.split()
    if not words:
        return []
    per_word = duration / len(words)

    lines = []
    t = 0.0
    for i in range(0, len(words), CAPTION_WORDS_PER_LINE):
        chunk_words = words[i:i + CAPTION_WORDS_PER_LINE]
        word_entries = []
        wt = t
        for w in chunk_words:
            word_entries.append({"word": w, "start": wt})
            wt += per_word
        line_end = wt
        lines.append({"start": t, "end": line_end, "words": word_entries})
        t = line_end
    return lines


def revealed_text_at(lines, t: float):
    for line in lines:
        if line["start"] <= t < line["end"]:
            revealed = [w["word"] for w in line["words"] if w["start"] <= t]
            return revealed
    return None


def load_caption_font(size: int):
    for path in CAPTION_FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def is_highlighted(word: str) -> bool:
    stripped = word.strip(".,!?;:\"'").lower()
    return stripped not in CAPTION_STOPWORDS and len(stripped) > 0


def render_caption_overlay(words: list, canvas_w: int, font, top_y: int) -> Image.Image:
    """Renders the revealed words as a single centered line, uppercase, with
    per-word yellow/white coloring and a bold black stroke outline (video-game
    caption style), on a semi-transparent rounded bar."""
    if not words:
        return Image.new("RGBA", (canvas_w, top_y + 40), (0, 0, 0, 0))

    display_words = [w.upper() for w in words]
    spacer = " "
    full_text = spacer.join(display_words)

    measure_img = Image.new("RGBA", (canvas_w, 400), (0, 0, 0, 0))
    measure_draw = ImageDraw.Draw(measure_img)
    bbox = measure_draw.textbbox((0, 0), full_text, font=font, stroke_width=6)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x, pad_y = 32, 18
    overlay = Image.new("RGBA", (canvas_w, top_y + text_h + pad_y * 3), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x = (canvas_w - text_w) // 2
    bar_box = (x - pad_x, top_y - pad_y, x + text_w + pad_x, top_y + text_h + pad_y)
    draw.rounded_rectangle(bar_box, radius=20, fill=(15, 15, 20, 190))

    cursor_x = x
    for original_word, disp_word in zip(words, display_words):
        color = (255, 214, 10, 255) if is_highlighted(original_word) else (255, 255, 255, 255)
        draw.text((cursor_x, top_y), disp_word, font=font, fill=color,
                   stroke_width=6, stroke_fill=(20, 20, 25, 255))
        word_bbox = draw.textbbox((cursor_x, top_y), disp_word + " ", font=font, stroke_width=6)
        cursor_x = word_bbox[2]

    return overlay


def render_scene(char_img: Image.Image, caption_lines, side: str, duration: float,
                  audio_path: Path, out_path: Path, char_scale=0.85, fps=FPS):
    char_h = int(CANVAS_H * char_scale)
    char_w = int(char_h * char_img.width / char_img.height)
    margin = int(CANVAS_W * 0.05)
    anchor_x = margin if side == "left" else CANVAS_W - char_w - margin
    anchor_y = CANVAS_H - char_h + 20

    resized_char = char_img.resize((char_w, char_h), Image.LANCZOS)
    total_frames = max(int(duration * fps), fps)

    font = load_caption_font(52)
    caption_top_y = 60

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(fps),
        "-i", "-", "-i", str(audio_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(out_path), "-loglevel", "error",
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    bg = Image.new("RGBA", (CANVAS_W, CANVAS_H), (*BACKGROUND_COLOR, 255))
    sway_phase = random.uniform(0, math.pi * 2)
    breathe_phase = random.uniform(0, math.pi * 2)
    last_revealed_key = None
    caption_overlay = None

    for i in range(total_frames):
        t = i / fps

        bob = int(5 * math.sin(2 * math.pi * t / 2.6))
        sway = int(4 * math.sin(2 * math.pi * t / 3.4 + sway_phase))
        rotation = 1.4 * math.sin(2 * math.pi * t / 4.1 + sway_phase)
        breathe_scale = 1.0 + 0.012 * math.sin(2 * math.pi * t / 3.0 + breathe_phase)

        transformed = resized_char.rotate(rotation, resample=Image.BICUBIC, expand=False)
        if breathe_scale != 1.0:
            new_w = int(transformed.width * breathe_scale)
            new_h = int(transformed.height * breathe_scale)
            transformed = transformed.resize((new_w, new_h), Image.LANCZOS)

        paste_x = anchor_x + sway - (transformed.width - char_w) // 2
        paste_y = anchor_y + bob - (transformed.height - char_h) // 2

        frame = bg.copy()
        frame.alpha_composite(transformed, (paste_x, paste_y))

        revealed = revealed_text_at(caption_lines, t)
        revealed_key = tuple(revealed) if revealed else None
        if revealed_key != last_revealed_key:
            caption_overlay = render_caption_overlay(revealed or [], CANVAS_W, font, caption_top_y)
            last_revealed_key = revealed_key
        if caption_overlay is not None and revealed:
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
    ap = argparse.ArgumentParser(description="Generate a Mikasan-narrated cartoon with a fixed character, solid background, and karaoke captions.")
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", default="output/final_video.mp4")
    ap.add_argument("--voice-sample", default=None)
    ap.add_argument("--language", default="en")
    ap.add_argument("--music", default=None)
    ap.add_argument("--character-scale", type=float, default=0.85)
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    settings, scenes = parse_script(args.script)
    voice_sample = args.voice_sample or settings["voice_sample"]
    music = args.music or settings["music"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = out_path.parent / "_work"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    print(f"Loaded {len(scenes)} scene(s). Voice sample: {voice_sample}. Music: {music or 'none'}")
    char_img = load_character()
    voice = ClonedVoice(Path(voice_sample), language=args.language)

    clip_paths = []
    for i, scene in enumerate(scenes):
        print(f"\n[{i+1}/{len(scenes)}] {scene['narration'][:70]}...")
        audio_wav = workdir / f"scene_{i:03}.wav"
        clip_path = workdir / f"scene_{i:03}.mp4"
        side = "left" if i % 2 == 0 else "right"

        print("  -> synthesizing narration (cloned voice, XTTS-v2)")
        voice.synth(scene["narration"], audio_wav)
        duration = get_duration(audio_wav) + 0.3

        caption_lines = build_caption_lines(scene["narration"], duration)

        print(f"  -> rendering scene ({side} side, {duration:.1f}s, {int(duration*FPS)} frames)")
        render_scene(char_img, caption_lines, side, duration, audio_wav, clip_path,
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
