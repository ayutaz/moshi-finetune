"""Shadow `silentcipher` so Irodori-TTS generates unwatermarked audio.

Put this directory first on PYTHONPATH and `import silentcipher` raises ImportError, which
`irodori_tts/watermark.py` already handles: it logs that audio will not be watermarked and
carries on. Nothing is uninstalled, so the venv is unchanged and other work that wants the
watermark still gets it.

M3 needs this because the watermark is an injected signal that would sit in every synthetic
turn of both training datasets while V-real's channel A - real corpus audio - carries none.
That is a systematic difference between the two datasets on top of the one difference the
comparison is designed to isolate, and a voice model trained on it could learn to reproduce
it. m3/DATASET_SPEC.md fixed this decision before any turn was rendered.

Verified after the fact by grepping the generation log for `silentcipher_watermark`: the
speaker-B probe of 2026-08-21 was written before this existed and carries the marker on all
eleven clips, so those files are watermarked and the M2 audio is too.
"""

raise ImportError("silentcipher is disabled for M3 generation; see this file's docstring")
