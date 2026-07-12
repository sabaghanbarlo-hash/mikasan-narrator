#!/usr/bin/env python3
"""
Mikasan Narrator - long-form cartoon generator
-----------------------------------------------
Turns a plain-text script into a video narrated in Saba's own cloned voice, with an
AI-generated Mikasan chibi character (positioned left/right, alternating per scene),
video-game-style captions synced to the narration, and vivid cartoon backgrounds --
all free, no API keys.

HOW IT WORKS:
  1. Each scene's narration is scanned for simple emotion cues (happy/thinking/
     surprised/neutral) to pick an expression for the character pose.
  2. Pollinations.ai generates ONE character pose for that scene (hand-drawn digital
     illustration style), background removal (rembg) isolates it.
  3. The character is placed on alternating left/right sides per scene, with subtle
     sway/breathing motion so it doesn't feel like a static cutout.
  4. Captions are generated from the narration text, timed proportionally across the
     scene's audio duration, and rendered as a bold bar near the TOP of the frame
     (video-game dialogue style) rather than traditional bottom subtitles.
  5. Backgrounds use a stronger cartoon-styled prompt plus a real color/contrast
     boost in post-processing for a more vivid, less flat look.
  Note: no lip-sync or blinking in this version -- removed by request for simplicity.

Usage:
    python generate_video.py --script scripts/example_script.txt --out output/final_video.mp4
"""

import argparse
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

CANVAS_W, CANVAS_H = 1920, 1080
FPS = 25
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
ANIME_STYLE_SUFFIX = (", vivid cartoon illustration, bold clean outlines, saturated colors, "
                      "dynamic cartoon lighting, exaggerated cartoon style, hand-drawn animation "
                      "background art, no photorealism, no muted colors")
DEFAULT_VOICE_SAMPLE = "voice_samples/Bee.m4a"

CHARACTER_IDENTITY = (
    "a single solo chibi anime girl, one character only, one pose only, no other people, "
    "Mikasa-inspired, short black bob hair, red scarf, big expressive dark eyes, "
    "simple black long-sleeve top, full body, standing, front facing, centered in frame, "
    "plain flat pastel background, hand-drawn digital illustration style, webtoon art, "
    "bold clean linework, cel shaded, vivid saturated colors, isolated single figure"
)

EXPRESSION_RULES = [
    (r"\b(happy|excited|great|love|joy|smile|amazing|wonderful|fun)\b", "bright happy smile, cheerful expression"),
    (r"\b(think|wonder|maybe|consider|hmm|question|why|how|curious)\b", "thoughtful pensive expression, hand near chin"),
    (r"\b(wow|shock|surprised|suddenly|what\?|can't believe|whoa)\b", "surprised expression, wide eyes, eyebrows raised"),
    (r"\b(sad|hard|difficult|tired|hurt|lonely|cry|lost)\b", "soft melancholic expression, gentle downward gaze"),
]
DEFAULT_EXPRESSION = "calm gentle expression, soft smile"

MAX_ACCEPTABLE_ASPECT_RATIO = 0.85  # catches multi-character grids slipping past the prompt
MAX_POSE_GENERATION_ATTEMPTS = 4

CAPTION_WORDS_PER_CHUNK = 6
CAPTION_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


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
        image_prompt = None
        narration_lines = []
        for line in lines:
            if line.upper().startswith("IMAGE:") and image_prompt is None:
                image_prompt = line.split(":", 1)[1].strip()
            else:
                narration_lines.append(line.strip())
        narration = " ".join(narration_lines).strip()
        if not narration:
            continue
        if not image_prompt:
            image_prompt = narration[:200]
        scenes.append({"image_prompt": image_prompt, "narration": narration})

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


def fetch_pollinations_image(prompt: str, out_path: Path, width, height, model="flux-anime", retries=4, seed=None):
    seed = seed if seed is not None else random.randint(1, 999_999)
    url = (f"{POLLINATIONS_BASE}/{quote(prompt)}?width={width}&height={height}"
           f"&seed={seed}&nologo=true&model={model}")
    fallback_url = (f"{POLLINATIONS_BASE}/{quote(prompt)}?width={width}&height={height}"
                     f"&seed={seed}&nologo=true")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            try_url = url if attempt <= retries - 1 else fallback_url
            resp = requests.get(try_url, timeout=120)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            if out_path.stat().st_size > 5000:
                return
            last_err = "image response too small"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        print(f"  [image] attempt {attempt} failed ({last_err}), retrying...")
        time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to fetch image for prompt {prompt!r}: {last_err}")


def pick_expression(narration: str) -> str:
    lowered = narration.lower()
    for pattern, expression in EXPRESSION_RULES:
        if re.search(pattern, lowered):
            return expression
    return DEFAULT_EXPRESSION


def autocrop_to_content(img: Image.Image) -> Image.Image:
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img


def generate_character_pose(scene: dict, workdir: Path, index: int) -> Image.Image:
    """Generates one context-appropriate character pose, background-removed and
    cropped to content. Retries if the result looks like a multi-character grid."""
    from rembg import remove

    expression = pick_expression(scene["narration"])
    prompt = f"{CHARACTER_IDENTITY}, {expression}"

    last_cutout = None
    for attempt in range(1, MAX_POSE_GENERATION_ATTEMPTS + 1):
        raw_path = workdir / f"char_{index:03}_raw_{attempt}.png"
        fetch_pollinations_image(prompt, raw_path, 768, 1024, model="flux-anime")

        raw_img = Image.open(raw_path).convert("RGBA")
        cutout = remove(raw_img)
        cutout = autocrop_to_content(cutout)
        last_cutout = cutout

        aspect_ratio = cutout.width / cutout.height
        if aspect_ratio <= MAX_ACCEPTABLE_ASPECT_RATIO:
            return cutout
        print(f"  [character] attempt {attempt} looked like multiple characters "
              f"(aspect ratio {aspect_ratio:.2f}), regenerating...")

    print("  [character] all attempts looked wide; using the last one anyway.")
    return last_cutout


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


def build_caption_chunks(narration: str, duration: float):
    """Splits narration into small on-screen chunks, timed proportionally to each
    chunk's word count across the scene's audio duration (even-pace approximation --
    there's no forced word-alignment step in this simplified pipeline)."""
    words = narration.split()
    if not words:
        return []
    per_word = duration / len(words)
    chunks = []
    t = 0.0
    for i in range(0, len(words), CAPTION_WORDS_PER_CHUNK):
        chunk_words = words[i:i + CAPTION_WORDS_PER_CHUNK]
        chunk_dur = per_word * len(chunk_words)
        chunks.append({"start": t, "end": t + chunk_dur, "text": " ".join(chunk_words)})
        t += chunk_dur
    return chunks


def caption_at(chunks, t: float):
    for c in chunks:
        if c["start"] <= t < c["end"]:
            return c["text"]
    return ""


def load_caption_font(size: int):
    for path in CAPTION_FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_caption(draw: ImageDraw.ImageDraw, text: str, canvas_w: int, font, top_y: int):
    if not text:
        return
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=6)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas_w - text_w) // 2

    pad_x, pad_y = 28, 16
    bar_box = (x - pad_x, top_y - pad_y, x + text_w + pad_x, top_y + text_h + pad_y)
    overlay = Image.new("RGBA", (canvas_w, top_y + text_h + pad_y * 3), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(bar_box, radius=18, fill=(15, 15, 20, 190))
    overlay_draw.text((x, top_y), text, font=font, fill=(255, 255, 255, 255),
                       stroke_width=6, stroke_fill=(20, 20, 25, 255))
    return overlay


def prepare_background(image_path: Path, width, height) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    src_ratio = img.width / img.height
    dst_ratio = width / height
    if src_ratio > dst_ratio:
        new_h = height
        new_w = int(height * src_ratio)
    else:
        new_w = width
        new_h = int(width / src_ratio)
    img = img.resize((new_w, new_h))
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    # Real color/contrast boost so backgrounds read as vivid cartoon art, not flat AI-image output
    img = ImageEnhance.Color(img).enhance(1.35)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img


def render_scene(background: Image.Image, char_img: Image.Image, caption_chunks, side: str,
                  duration: float, audio_path: Path, out_path: Path, char_scale=0.85, fps=FPS):
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
    bg_rgba = background.convert("RGBA")

    sway_phase = random.uniform(0, math.pi * 2)
    breathe_phase = random.uniform(0, math.pi * 2)
    last_caption_text = None
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

        frame = bg_rgba.copy()
        frame.alpha_composite(transformed, (paste_x, paste_y))

        caption_text = caption_at(caption_chunks, t)
        if caption_text != last_caption_text:
            caption_draw_target = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(caption_draw_target)
            caption_overlay = draw_caption(d, caption_text, CANVAS_W, font, caption_top_y)
            last_caption_text = caption_text
        if caption_overlay is not None and caption_text:
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
    ap = argparse.ArgumentParser(description="Generate a Mikasan-narrated cartoon with left/right character placement and top captions.")
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
    voice = ClonedVoice(Path(voice_sample), language=args.language)

    clip_paths = []
    for i, scene in enumerate(scenes):
        print(f"\n[{i+1}/{len(scenes)}] {scene['narration'][:70]}...")
        audio_wav = workdir / f"scene_{i:03}.wav"
        bg_image_path = workdir / f"scene_{i:03}_bg.jpg"
        clip_path = workdir / f"scene_{i:03}.mp4"
        side = "left" if i % 2 == 0 else "right"

        print("  -> synthesizing narration (cloned voice, XTTS-v2)")
        voice.synth(scene["narration"], audio_wav)
        duration = get_duration(audio_wav) + 0.3

        caption_chunks = build_caption_chunks(scene["narration"], duration)

        print("  -> generating context-driven character pose")
        char_img = generate_character_pose(scene, workdir, i)

        print(f"  -> generating background: {scene['image_prompt'][:70]}...")
        fetch_pollinations_image(f"{scene['image_prompt']}{ANIME_STYLE_SUFFIX}", bg_image_path, CANVAS_W, CANVAS_H)
        background = prepare_background(bg_image_path, CANVAS_W, CANVAS_H)

        print(f"  -> rendering scene ({side} side, {duration:.1f}s, {int(duration*FPS)} frames)")
        render_scene(background, char_img, caption_chunks, side, duration, audio_wav,
                     clip_path, char_scale=args.character_scale)
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
