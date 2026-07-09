# Mikasan Narrator

Free, automated pipeline for a Mikasa-inspired chibi anime narrator character, extending the free-ai-video-generator project.

## Goal
- Chibi anime narrator (Mikasa-inspired), rigged with lip sync, blinking, body bob, gestures
- Script -> long-form video pipeline (Pollinations.ai backgrounds + FFmpeg compositing)
- Voice cloned from Saba's own voice (free/open-source: XTTS-v2 or RVC)
- Approval gate -> fully automatic YouTube upload via GitHub Actions

## Status
- [x] Repo scaffolded
- [ ] Voice sample uploaded (drop into `voice_samples/`)
- [ ] XTTS-v2 voice clone proof-of-concept
- [ ] Character rig
- [ ] Script -> video pipeline
- [ ] Approval + auto-upload workflow

## Voice sample
Upload a clean 20-60s voice sample (mp3/wav/m4a) to `voice_samples/`.
Use GitHub's web UI (Add file -> Upload files) for binary audio -- avoids corruption issues we hit before with PNGs.
