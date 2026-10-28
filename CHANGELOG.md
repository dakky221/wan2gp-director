# Changelog

## 8.4

- Use the permanent script filename `ollama_mcp_wan_story_loop_h3_directed.py`. Keep the release in `__version__`, the startup/CLI version display, new-job metadata and this changelog.
- Require exact double-quoted dialogue within `<d>` tags, tied to a named speaker. Request one speaker at a time; replace vague/overlapping speech beats with sequential quoted lines or silent reactions.
- Require established names in shot/action descriptions rather than collective substitutes such as “the warriors” or “the pair.”
- Share the wording rules across opening and continuation prompt builders, and reinforce them in scene/brief planning. No new semantic validation or GPU/media changes.
- Retain the configurable LoRA strength, no-LoRA option and prior review recovery/directing fixes.

Validation: syntax, startup/version metadata, acceleration/strength and resume tests, review recovery and the simulated 60-clip directing run. Effects on rendered dialogue/identity require a generation test.

## 8.3

- Ask for an external LoRA's strength after its step count (default 1.0).
- Add `--lora-strength`, supporting finite nonnegative scalar values such as 0.5 and 0.7; the interactive question also accepts decimal commas.
- Pass the chosen scalar through the existing MCP `loras_multipliers` field and persist it in the job's generation configuration.
- Resume restores the saved strength and accepts older 1.0 jobs; acceleration overrides remain prohibited on resume.
- PDD and no-LoRA modes skip the strength question and reject an explicit external strength flag.
- Keep the v8.1 pacing fix, v8.2 no-LoRA option and all directing/media behavior unchanged.

Validation: menu/CLI strengths, 0.5/0.7 MCP payloads, invalid input, startup/save/resume, older jobs, PDD/no-LoRA regressions, and reactive scene-limit/review checks. GPU quality differences have not been assessed here.

## 8.2

- Add menu option 2, No acceleration LoRA, using the plain FL2VA Pruned 20B preset with a configurable step count (default 20).
- Add `--lora none --steps 20`. No-LoRA mode explicitly sends empty `activated_loras` and `loras_multipliers` fields to MCP.
- Keep PDD 8-Step as the default and preserve the custom adapter path with its 6-step default. Local filenames now start at menu option 3.
- Persist no-LoRA settings using the existing generation configuration; resume restores them without needing a local adapter file.
- Retain v8.1 review recovery and all existing directing, continuation, HEVC and audio assembly behavior.

Validation: interactive/CLI selection, invalid steps, switching away from a selected adapter, exact MCP payload, startup/resume, existing PDD/custom adapter tests, and the reactive directing/review regressions. GPU quality comparison remains to be performed.

## 8.1

- Normalize visual-review pacing labels before applying the scene policy. A premature finish ends the current scene while preserving the remaining film budget.
- Accept common equivalents of new_scene/continue; unknown or missing labels defer to the hard scene schedule. At zero remaining clips, close the film.
- Apply the same normalization to cached reviews on resume; retain original labels in diagnostics.
- Keep existing observation/continuity validation, saved clips, hard scene limits, finale reserve, media encoding and audio synchronization unchanged.
- Regression checks recreate a clip-21 review requesting finish with 9 clips remaining, accepting it without repeated corrections or regeneration.

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
