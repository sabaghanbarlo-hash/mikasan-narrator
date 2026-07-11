# Character art is no longer needed here.

The pipeline now generates the Mikasan character automatically per scene:
Pollinations.ai produces a context-appropriate pose (happy/thoughtful/surprised/neutral,
detected from the scene's narration), background removal isolates it, and lip-sync/blink
states are drawn directly onto that same image so they stay aligned.

See CHARACTER_IDENTITY and EXPRESSION_RULES at the top of `generate_video.py` if you want
to adjust the character's fixed look or how expressions are detected.

Note: arm/gesture animation isn't supported in this mode (no reliable way to isolate and
reposition a limb from a flat AI-generated image). If that's wanted later, hand-drawn
layered rig art is the way to get it back.
