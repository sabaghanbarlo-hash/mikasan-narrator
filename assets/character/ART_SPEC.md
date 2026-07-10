# Mikasan Character Art Spec

This is the exact file list the render pipeline (`generate_video.py`) expects in `assets/character/`.
Draw/commission to these specs and the pipeline will work with zero code changes.

## Canvas
- All files are transparent-background PNGs (RGBA).
- Pick one canvas size for `base.png` and stick to it, e.g. **900x1200** (portrait chibi). Put the real number in `rig.json`.
- `base.png` = the full character MINUS eyes, mouth, and right arm (those are drawn separately and pasted on top each frame). Leave those regions either blank/transparent or with a neutral placeholder that reads fine when covered.

## Files needed
| File | What it is |
|---|---|
| `base.png` | Full body, no eyes/mouth/right-arm detail (canvas-sized) |
| `eye_l_open.png` | Left eye, open, cropped tight to just the eye shape |
| `eye_r_open.png` | Right eye, open, cropped tight |
| `eye_closed.png` | A single closed-eye/eyelid shape (used for both eyes) |
| `mouth_closed.png` | Mouth shape: closed |
| `mouth_mid.png` | Mouth shape: half-open (mid-syllable) |
| `mouth_wide.png` | Mouth shape: wide open (vowel sounds) |
| `arm_idle.png` | Right arm, resting position |
| `arm_raised.png` | Right arm, raised/gesturing position |

Each of these (except base.png) is a small cropped sprite — not full-canvas — sized to just the element itself.

## rig.json (positions)
Create `assets/character/rig.json` with the pixel coordinates (top-left corner, in base.png's coordinate space) where each sprite should be pasted:

```json
{
  "canvas_size": [900, 1200],
  "left_eye_pos": [0, 0],
  "right_eye_pos": [0, 0],
  "mouth_pos": [0, 0],
  "right_arm_idle_pos": [0, 0],
  "right_arm_raised_pos": [0, 0]
}
```
Fill in the real [x, y] values once you know where those features land on your base.png canvas — easiest way is to open base.png in any image editor with a grid/ruler and read off the top-left corner of where each sprite should sit.

## Style direction
- Chibi proportions, anime cel-shaded style
- Mikasa-inspired color palette (dark hair, red scarf as a signature accessory) — original design, not a 1:1 copy of the AoT character
- Should read clearly at both full-canvas size and scaled down to ~85% of a 1080p frame height (how it'll actually be composited)

## Once art is ready
Drop all files into `assets/character/` (via GitHub's Upload files UI, same as the voice sample) and ping me — I'll wire it in and run a test render.
