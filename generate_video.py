#!/usr/bin/env python3
"""
Mikasan Narrator - long-form cartoon generator
-----------------------------------------------
Turns a plain-text script into a video narrated in Saba's own cloned voice, with an
AI-generated Mikasan chibi character whose pose/expression adapts per scene based on
context, plus lip-sync and blinking -- all free, no API keys.

HOW THE CHARACTER WORKS (no hand-drawn art required):
  1. Each scene's narration is scanned for simple emotion cues (happy/thinking/
     surprised/neutral) to pick an expression prompt.
  2. Pollinations.ai generates ONE character pose image for that scene, styled as
     hand-drawn digital illustration (not a photoreal AI-image look).
  3. Background removal (rembg) isolates the character onto a transparent PNG.
  4. A real anime-face detector (lbpcascade_animeface, OpenCV) locates the actual
     face in the generated image. Mouth/eye states for lip-sync and blinking are
     drawn relative to THAT DETECTED FACE (not guessed fixed positions), directly
     onto the same image so they stay aligned.
  5. Subtle sway/breathing motion is added at render time so the character doesn't
     look like a stiff static cutout.
  Note: arm/gesture animation is not supported in this AI-generated mode (no reliable
  way to isolate and reposition a limb from a flat generated image).

Usage:
    python generate_video.py --script scripts/example_script.txt --out output/final_video.mp4
"""

import argparse
import json
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

import numpy as np
import requests
from PIL import Image, ImageDraw

CANVAS_W, CANVAS_H = 1920, 1080
FPS = 25
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
ANIME_STYLE_SUFFIX = (", hand-drawn digital illustration, webtoon style, digital painting, "
                      "clean linework, cel shaded, vibrant anime color palette, no photorealism")
DEFAULT_VOICE_SAMPLE = "voice_samples/Bee.m4a"
FACE_CASCADE_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"

# Fixed identity so the character reads as "the same character" across scenes even
# though each pose is a fresh generation. Keep this description consistent.
# IMPORTANT: avoid words like "character sheet" / "reference sheet" / "turnaround" --
# image models read those as a request for a multi-pose grid, producing several
# copies of the character in one image instead of a single pose.
CHARACTER_IDENTITY = (
    "a single solo chibi anime girl, one character only, one pose only, no other people, "
    "Mikasa-inspired, short black bob hair, red scarf, big expressive dark eyes, "
    "simple black long-sleeve top, full body, standing, front facing, centered in frame, "
    "plain flat pastel background, hand-drawn digital illustration style, webtoon art, "
    "clean linework, cel shaded, isolated single figure"
)

EXPRESSION_RULES = [
    (r"\b(happy|excited|great|love|joy|smile|amazing|wonderful|fun)\b", "bright happy smile, cheerful expression"),
    (r"\b(think|wonder|maybe|consider|hmm|question|why|how|curious)\b", "thoughtful pensive expression, hand near chin"),
    (r"\b(wow|shock|surprised|suddenly|what\?|can't believe|whoa)\b", "surprised expression, wide eyes, eyebrows raised"),
    (r"\b(sad|hard|difficult|tired|hurt|lonely|cry|lost)\b", "soft melancholic expression, gentle downward gaze"),
]
DEFAULT_EXPRESSION = "calm gentle expression, soft smile"

# Fallback proportions (fractions of the whole cropped body) ONLY used if face
# detection fails on a given image -- real face detection is tried first.
FALLBACK_EYE_L_REGION = (0.34, 0.12, 0.46, 0.20)
FALLBACK_EYE_R_REGION = (0.54, 0.12, 0.66, 0.20)
FALLBACK_MOUTH_REGION = (0.42, 0.22, 0.58, 0.28)

# Where eyes/mouth sit as fractions WITHIN a detected face box (standard proportions
# for a front-facing anime face).
FACE_EYE_L_REGION = (0.18, 0.38, 0.42, 0.52)
FACE_EYE_R_REGION = (0.58, 0.38, 0.82, 0.52)
FACE_MOUTH_REGION = (0.35, 0.68, 0.65, 0.82)

MAX_ACCEPTABLE_ASPECT_RATIO = 0.85  # width / height; catches multi-character grids
MAX_POSE_GENERATION_ATTEMPTS = 4

VISEME_MAP = {
    "A": "closed", "B": "closed", "G": "closed", "X": "closed",
    "C": "mid", "E": "mid", "F": "mid", "H": "mid",
    "D": "wide",
}

_face_cascade = None


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


def get_face_cascade(workdir: Path):
    """Downloads (once) and loads the anime face detector cascade."""
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    import cv2

    cascade_path = workdir / "lbpcascade_animeface.xml"
    if not cascade_path.exists():
        resp = requests.get(FACE_CASCADE_URL, timeout=60)
        resp.raise_for_status()
        cascade_path.write_bytes(resp.content)
    _face_cascade = cv2.CascadeClassifier(str(cascade_path))
    return _face_cascade


def detect_face_box(img: Image.Image, workdir: Path):
    """Returns (left, top, right, bottom) pixel box of the largest detected anime
    face, or None if no face was found."""
    import cv2

    cascade = get_face_cascade(workdir)
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
    if len(faces) == 0:
        return None
    # pick the largest detected face box
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return (int(fx), int(fy), int(fx + fw), int(fy + fh))


def generate_character_pose(scene: dict, workdir: Path, index: int):
    """Generates one context-appropriate character pose. Returns
    (background-removed RGBA image cropped to content, face_box or None).

    Retries if the result looks like a multi-character grid (too wide relative to
    height for a single standing chibi figure) rather than one pose.
    """
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
            face_box = detect_face_box(cutout, workdir)
            if face_box is None:
                print(f"  [character] attempt {attempt}: no face detected, using fallback face position.")
            return cutout, face_box
        print(f"  [character] attempt {attempt} looked like multiple characters "
              f"(aspect ratio {aspect_ratio:.2f}), regenerating...")

    print("  [character] all attempts looked wide; using the last one anyway.")
    return last_cutout, detect_face_box(last_cutout, workdir)


def region_in_image_fractions(face_box, region_within_face, img_size):
    """Converts a region defined as fractions WITHIN a face box into fractions of
    the whole image, so draw_eye_state/draw_mouth_state (which work in whole-image
    fractions) can be reused either way."""
    fx, fy, fx2, fy2 = face_box
    fw, fh = fx2 - fx, fy2 - fy
    l, t, r, b = region_within_face
    iw, ih = img_size
    return (
        (fx + l * fw) / iw, (fy + t * fh) / ih,
        (fx + r * fw) / iw, (fy + b * fh) / ih,
    )


def draw_eye_state(img: Image.Image, region_frac, closed: bool):
    w, h = img.size
    l, t, r, b = region_frac
    box = (int(l * w), int(t * h), int(r * w), int(b * h))
    if closed:
        draw = ImageDraw.Draw(img)
        cy = (box[1] + box[3]) // 2
        draw.line([(box[0], cy), (box[2], cy)], fill=(40, 30, 30, 255), width=max(2, int(h * 0.006)))


def draw_mouth_state(img: Image.Image, region_frac, state: str):
    w, h = img.size
    l, t, r, b = region_frac
    cx = int((l + r) / 2 * w)
    cy = int((t + b) / 2 * h)
    half_w = int((r - l) * w / 2)
    full_half_h = int((b - t) * h / 2)
    scale = {"closed": 0.15, "mid": 0.55, "wide": 1.0}[state]
    half_h = max(1, int(full_half_h * scale))
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        fill=(90, 40, 45, 255),
    )


def build_character_states(base_pose: Image.Image, face_box):
    if face_box is not None:
        eye_l = region_in_image_fractions(face_box, FACE_EYE_L_REGION, base_pose.size)
        eye_r = region_in_image_fractions(face_box, FACE_EYE_R_REGION, base_pose.size)
        mouth = region_in_image_fractions(face_box, FACE_MOUTH_REGION, base_pose.size)
    else:
        eye_l, eye_r, mouth = FALLBACK_EYE_L_REGION, FALLBACK_EYE_R_REGION, FALLBACK_MOUTH_REGION

    frames = {}
    for eyes_state in ("open", "closed"):
        for mouth_state in ("closed", "mid", "wide"):
            frame = base_pose.copy()
            if eyes_state == "closed":
                draw_eye_state(frame, eye_l, True)
                draw_eye_state(frame, eye_r, True)
            draw_mouth_state(frame, mouth, mouth_state)
            frames[(eyes_state, mouth_state)] = frame
    return frames


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


def run_rhubarb(wav_path: Path, out_json: Path, rhubarb_bin="rhubarb"):
    try:
        run([rhubarb_bin, "-f", "json", "-o", str(out_json), "--recognizer", "phonetic", str(wav_path)])
        data = json.loads(out_json.read_text(encoding="utf-8"))
        return data.get("mouthCues", [])
    except Exception as e:  # noqa: BLE001
        print(f"  [lipsync] Rhubarb unavailable/failed ({e}); falling back to a simple approximation.")
        return None


def approximate_mouth_cues(wav_path: Path, duration: float):
    import wave
    import audioop

    cues = []
    try:
        with wave.open(str(wav_path), "rb") as wf:
            framerate = wf.getframerate()
            chunk_ms = 90
            chunk_frames = int(framerate * chunk_ms / 1000)
            t = 0.0
            while True:
                frames = wf.readframes(chunk_frames)
                if not frames:
                    break
                rms = audioop.rms(frames, wf.getsampwidth())
                if rms < 200:
                    letter = "X"
                elif rms < 800:
                    letter = "B"
                elif rms < 2000:
                    letter = "C"
                else:
                    letter = "D"
                cues.append({"start": t, "end": t + chunk_ms / 1000, "value": letter})
                t += chunk_ms / 1000
    except Exception as e:  # noqa: BLE001
        print(f"  [lipsync] fallback also failed ({e}); using static mid-mouth for whole clip.")
        cues = [{"start": 0, "end": duration, "value": "C"}]
    return cues


def mouth_state_at(cues, t: float) -> str:
    for cue in cues:
        if cue["start"] <= t < cue["end"]:
            return VISEME_MAP.get(cue["value"], "closed")
    return "closed"


def make_blink_schedule(duration: float, min_gap=2.5, max_gap=5.5, blink_len=0.14):
    schedule = []
    t = random.uniform(1.0, min_gap)
    while t < duration:
        schedule.append((t, t + blink_len))
        t += random.uniform(min_gap, max_gap)
    return schedule


def is_blinking(schedule, t: float) -> bool:
    return any(start <= t < end for start, end in schedule)


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
    return img.crop((left, top, left + width, top + height))


def render_scene(background: Image.Image, char_frames: dict, mouth_cues, blink_schedule,
                  duration: float, audio_path: Path, out_path: Path, char_scale=0.85, fps=FPS):
    sample_img = next(iter(char_frames.values()))
    char_h = int(CANVAS_H * char_scale)
    char_w = int(char_h * sample_img.width / sample_img.height)
    anchor_x = (CANVAS_W - char_w) // 2
    anchor_y = CANVAS_H - char_h + 20

    resized_cache = {key: img.resize((char_w, char_h), Image.LANCZOS) for key, img in char_frames.items()}
    total_frames = max(int(duration * fps), fps)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(fps),
        "-i", "-", "-i", str(audio_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(out_path), "-loglevel", "error",
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    bg_rgba = background.convert("RGBA")

    # Slight per-scene randomization so sway doesn't look identical every scene
    sway_phase = random.uniform(0, math.pi * 2)
    breathe_phase = random.uniform(0, math.pi * 2)

    for i in range(total_frames):
        t = i / fps
        mouth_state = mouth_state_at(mouth_cues, t) if mouth_cues else "closed"
        eyes_state = "closed" if is_blinking(blink_schedule, t) else "open"
        char_img = resized_cache[(eyes_state, mouth_state)]

        # Natural motion: vertical bob (existing), horizontal sway, slight rotation,
        # and a gentle breathing scale pulse -- keeps a static cutout from feeling stiff.
        bob = int(5 * math.sin(2 * math.pi * t / 2.6))
        sway = int(4 * math.sin(2 * math.pi * t / 3.4 + sway_phase))
        rotation = 1.4 * math.sin(2 * math.pi * t / 4.1 + sway_phase)
        breathe_scale = 1.0 + 0.012 * math.sin(2 * math.pi * t / 3.0 + breathe_phase)

        transformed = char_img.rotate(rotation, resample=Image.BICUBIC, expand=False)
        if breathe_scale != 1.0:
            new_w = int(transformed.width * breathe_scale)
            new_h = int(transformed.height * breathe_scale)
            transformed = transformed.resize((new_w, new_h), Image.LANCZOS)

        paste_x = anchor_x + sway - (transformed.width - char_w) // 2
        paste_y = anchor_y + bob - (transformed.height - char_h) // 2

        frame = bg_rgba.copy()
        frame.alpha_composite(transformed, (paste_x, paste_y))
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
    ap = argparse.ArgumentParser(description="Generate a Mikasan-narrated cartoon with an AI-generated, context-adaptive character.")
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", default="output/final_video.mp4")
    ap.add_argument("--voice-sample", default=None)
    ap.add_argument("--language", default="en")
    ap.add_argument("--music", default=None)
    ap.add_argument("--character-scale", type=float, default=0.85)
    ap.add_argument("--rhubarb-bin", default="rhubarb")
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
        rhubarb_json = workdir / f"scene_{i:03}_mouth.json"
        clip_path = workdir / f"scene_{i:03}.mp4"

        print("  -> synthesizing narration (cloned voice, XTTS-v2)")
        voice.synth(scene["narration"], audio_wav)
        duration = get_duration(audio_wav) + 0.3

        print("  -> computing lip-sync (Rhubarb)")
        mouth_cues = run_rhubarb(audio_wav, rhubarb_json, args.rhubarb_bin)
        if mouth_cues is None:
            mouth_cues = approximate_mouth_cues(audio_wav, duration)
        blink_schedule = make_blink_schedule(duration)

        print("  -> generating context-driven character pose (with face detection)")
        base_pose, face_box = generate_character_pose(scene, workdir, i)
        char_frames = build_character_states(base_pose, face_box)

        print(f"  -> generating background: {scene['image_prompt'][:70]}...")
        fetch_pollinations_image(f"{scene['image_prompt']}{ANIME_STYLE_SUFFIX}", bg_image_path, CANVAS_W, CANVAS_H)
        background = prepare_background(bg_image_path, CANVAS_W, CANVAS_H)

        print(f"  -> rendering animated scene ({duration:.1f}s, {int(duration*FPS)} frames)")
        render_scene(background, char_frames, mouth_cues, blink_schedule, duration, audio_wav,
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
