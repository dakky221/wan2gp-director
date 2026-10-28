# Changelog

## 8.0

- Replace unlimited scene extensions with normally 3–4 clip scenes, a hard maximum of four, explicit closing prompts and fresh cuts at boundaries.
- Reserve a concluding scene rather than asking the last clip alone to resolve an unfinished film.
- Keep per-clip visual feedback; review four earlier frames plus the original three ending frames.
- Distinguish sampled evidence and uncertainty; missing late-frame motion is not proof an entire action failed.
- Persist character status, prop changes and recent facts; prevent confirmed eliminations from resetting to active or appearing as named spectators in H3 prompts.
- Keep the full Simple-mode cast roster rather than inheriting only the opening scene's cast.
- Follow each assigned scene development, with visual adaptation, instead of indefinitely replacing it with the previous review's retry request.
- Repair near-duplicate prompts before GPU rendering; cut away after two positively stalled clips.
- Keep working context compact while retaining full per-clip logs on disk.
- Use outputs_story_loop_h3_v8 for new jobs. Keep existing locations when resuming older jobs.
- Retain acceleration selection, Simple/Expert setup, optional first image, HEVC default, exact-frame continuation, frame-limit safeguards, audio-sync assembly, startup-noise filtering and named final exports.

Validation: adversarial simulated 60-clip run, saved v7 repetition cases, interruption/resume, startup/LoRA checks, and real HEVC/32kHz AAC media fixtures with exact PCM counts and flash/click alignment. Wan2GP v18.84 is the previously tested integration baseline; a full GPU-generated v8 episode remains to be tested.
