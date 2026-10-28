# Wan2GP H3 Directed Story Loop

**Version 8.0 · Wan2GP v18.84 integration baseline**

Turn an idea into a multi-scene film using MiniMax H3 and a vision-capable LLM. After each generated clip, the LLM reviews seven sampled frames, records visible consequences and helps direct the next clip. Scenes normally use 3–4 clips, with limits enforced by the script and clean cuts by default. Prompts adapt to the rendered footage rather than being written for the whole episode in advance.

The script uses Wan2GP's MCP interface and launches its own headless Wan2GP process. It does not require a browser UI.

## Features

- **Simple mode:** enter an idea and an approximate runtime. The LLM fills in the premise, genre, visual style, character descriptions, voices and directing notes.
- **Expert mode:** supply those details yourself.
- **Bounded reactive directing:** the LLM chooses 3 or 4 clips per scene, and the script enforces the limit. Scenes may finish earlier; budget remainders may be shorter. The director cannot extend a scene indefinitely.
- **A reserved ending:** the final 4 clips of runs with at least 8 clips are reserved for the finale; smaller films use a smaller ending budget.
- **Persistent continuity:** saved character status, prop changes and recent observed/uncertain events guide subsequent prompts. Confirmed eliminated characters cannot be reassigned as active cast.
- **Repetition checks:** near-duplicate prompts are repaired before GPU rendering; two reviews positively identifying no progress trigger a fresh scene cut. Incomplete visual coverage alone does not trigger that rule.
- **Selected transitions:** fresh scene boundaries use clean cuts by default, or motivated fades through black. Exact-frame continuation applies inside scenes; scene boundaries release the old starting image. Location changes are optional when a new objective makes sense in the same place.
- **Visual review and continuation:** reviews receive four earlier samples at about 8%, 30%, 55%, 78%, plus three ending frames at 90%, 95% and the last frame. Continuation prompt writing still receives the three ending frames, and H3 receives only the exact last decoded PNG.
- **Acceleration selector:** PDD 8-Step by default, or a local LoRA with a chosen step count.
- **HEVC output by default**, with an H.264 override.
- **Sample-aligned audio assembly:** per-clip PCM audio is aligned to the video frame count; AAC is encoded once per assembled movie to avoid accumulating encoder delay at joins.
- **Resume and rebuild:** reuse saved clips after an interruption, or rebuild the edit without generating videos again.
- **Organized output:** raw clips stay inside each job; completed movies appear in the output root with an LLM-generated filename.

## How v8 directs a scene

1. The scene planner chooses a permitted clip count and a distinct dramatic objective, then writes one development per clip. It receives the full requested cast roster, pending characters, the continuity ledger and recent scene outcomes.
2. Each H3 prompt is written immediately before its clip. Visual feedback can adapt its assigned development, but cannot replace it with unlimited attempts at the previous beat.
3. The final clip of each scene receives an explicit scene-closing instruction before rendering. Afterwards, the script enforces a cut or motivated fade even if the reviewing LLM requests another continuation.
4. The next scene starts as a fresh shot, without the previous H3 starting frame. It must advance a different objective rather than rename and restart the old fight.
5. The final reserved scene is directed as the payoff and aftermath of the film.

The limits are code-enforced. Visual interpretation, story quality and similarity detection still depend on the model and sampled evidence. Seven samples are not the full video: an action missing from them must not automatically be treated as a failed render. The ledger only accepts state changes the reviewer marks observed, and preserves uncertainty separately; it cannot guarantee the reviewer judged the evidence correctly.

The default is a maximum of 4 clips per scene, not an unrestricted “long action scene” setting. User notes guide the content and transitions within those limits. The limits and finale budget are saved in each job's manifest so resume keeps the same policy. No extra startup questions were added.

## Requirements

- A working **Wan2GP installation with MiniMax H3 support**. **Wan2GP v18.84** is the integration baseline tested with v7. The v8 directing changes have been checked with simulated LLM/MCP runs and real FFmpeg media fixtures; a full GPU-generated v8 episode has not been tested here.
- The Python environment used by that Wan2GP installation. Windows/Conda commands are shown below.
- **FFmpeg and FFprobe on PATH**. FFmpeg must include `libx265` for the default HEVC output, or `libx264` for H.264.
- A **vision-capable LLM** through OpenRouter or Ollama. Images are sent throughout generation; a text-only model cannot perform the intended visual review.
- The usual GPU, RAM and model files needed to run MiniMax H3 in Wan2GP.

Wan2GP, model weights, LoRAs and FFmpeg are not included in this repository.

## Installation and script placement

1. Download or clone this repository.
2. Copy **`ollama_mcp_wan_story_loop_h3_directed_v8_0.py` into your Wan2GP folder, directly beside `wgp.py`**. Do not place it in a subfolder.
3. Copy `requirements-story-loop.txt` into that same folder. This filename deliberately differs from Wan2GP's own requirements file; do not replace Wan2GP's requirements or README.
4. Activate your existing Wan2GP environment and install the additional Python dependencies:

```bat
conda activate wan2gp_new
cd /d C:\AI\Wan2GP
python -m pip install -r requirements-story-loop.txt
```

Replace `wan2gp_new` and `C:\AI\Wan2GP` with your environment and installation path.

Check FFmpeg availability from the same terminal:

```bat
ffmpeg -version
ffprobe -version
```

The script defaults to `WAN2GP_DIR = SCRIPT_DIR`, so placing it beside `wgp.py` is important. Its bundled defaults refer to the `defaults` and `loras` folders in that installation.

## Configure the OpenRouter API key

OpenRouter is the default LLM backend. Create a file named **`.env` beside the script** containing:

```dotenv
OPENROUTER_API_KEY=sk-or-your-key-here
```

Use the included `.env.example` as a template. Replace the placeholder with your real key. Ensure the file is named `.env`, not `.env.txt`.

If a `.env` already exists in your Wan2GP folder, add the key to it rather than replacing the file. Existing environment variables take precedence over `.env` values.

**Do not commit your real `.env` or API key.** The supplied `.gitignore` excludes `.env` and allows the placeholder-only `.env.example`.

The default OpenRouter model is `google/gemma-4-31b-it`. Change it in Expert mode, with `--model`, or in the configuration section at the top of the script. Use a model that supports image inputs.

## Run

From your Wan2GP folder, with its environment active:

```bat
python ollama_mcp_wan_story_loop_h3_directed_v8_0.py
```

Startup asks for:

1. **Acceleration / LoRA** — press Enter for PDD 8-Step. A local LoRA selection also asks for steps.
2. **Simple or Expert mode.**
3. In **Simple** mode: your idea and runtime in minutes. It uses the configured provider/model without additional provider questions.
4. In **Expert** mode: backend, model, theme, genre, visual style, notes, clip count and optional first image, unless supplied as command-line arguments.

For example, choose Simple and enter `an archaeologist discovers a machine beneath an abandoned monastery`, then `3` minutes. The script expands the idea, chooses a movie filename and starts generating.

Each clip uses 362 frames at 24 FPS, approximately 15.08 seconds. Runtime requests are rounded to whole clips, and continuation joins remove one repeated frame. A three-minute request is approximately 12 clips; runtime is not an exact duration guarantee.

### Start directly in Simple mode

```bat
python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --lora pdd --mode simple --idea "An archaeologist discovers a machine beneath an abandoned monastery" --minutes 3
```

### Use Ollama instead of OpenRouter

```bat
python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --backend ollama --model gemma4:26b
```

Ollama must already be running with the chosen vision model installed. An OpenRouter API key is not required when using Ollama. The default Ollama endpoint is `http://127.0.0.1:11434/api/generate`; edit `OLLAMA_URL` in the script if your service uses another address.

Ollama and Wan2GP can compete for GPU/RAM. This reactive workflow does not automatically unload either model between calls.

### Start from an image

```bat
python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --start-image "C:\Pictures\opening.png"
```

The image is copied into the job's assets and anchors the first clip. In Simple mode, pass `--start-image` or configure `START_IMAGE` at the top of the script; Simple mode does not add an image-path question.

## Acceleration and local LoRAs

**Default:** `minimax_h3_fl2va_pruned_pdd`, 8 steps. Wan2GP's PDD preset includes its own accelerator; the script does not stack an external acceleration LoRA on top of it.

Other `.safetensors` files are listed from:

```text
YOUR_WAN2GP_FOLDER\loras\minimax_h3
```

Subfolders are included. Selecting a local LoRA switches to **`minimax_h3_fl2va_pruned`** (FL2VA Pruned 20B), applies that one adapter at strength **1.0**, and asks for the step count. The input defaults to **6 steps**; this is editable and is not a universal recommendation for every adapter. Choose a LoRA compatible with the pruned FL2VA base.

For example, if already installed in that folder:

```bat
python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --lora "TaoMate-H3-FL2VA-3Step-ComfyUI-BF16.safetensors" --steps 6
```

No LoRAs are downloaded by this script's selector. The selected adapter and steps are saved with the job and restored on resume.

## Output layout

New v8 jobs use **`outputs_story_loop_h3_v8`**. Resuming an older job keeps that job in its existing folder.

| Location | Contents |
| --- | --- |
| `outputs_story_loop_h3_v8/Moonlit Reckoning.mp4` | A completed film with a theme-based LLM filename |
| `outputs_story_loop_h3_v8/directed_v8_0_<timestamp>/raw/` | Original standalone segment videos |
| `JOB_FOLDER/frames/` and `JOB_FOLDER/sheets/` | Sampled ending frames, exact continuation anchors and contact sheets |
| `JOB_FOLDER/director/` | Saved director requests, replies and filename request |
| `JOB_FOLDER/story_manifest.json` | Saved settings, clip paths, scene decisions, continuity ledger and resume state |
| `JOB_FOLDER/sheets/*_overview.jpg` | Four earlier labelled frame samples for the visual review |
| `JOB_FOLDER/sheets/*_contact_sheet.jpg` | Three ending frames used for motion/continuation context |
| `JOB_FOLDER/segment_...` | Per-clip prompts and generation settings |
| `JOB_FOLDER/story_full.mp4` | An in-progress assembled checkpoint; moved to the output root once complete |

Assets and editing caches also remain inside the job folder. Movie filenames are sanitized for Windows; duplicate titles get numeric suffixes. The chosen export name is reused on resume/reassembly. If naming fails, the script uses a theme-based fallback instead of stopping the episode.

Newly generated raw clips go into the current job. Resuming an older job preserves its existing clip paths; it does not reorganize older raw files already saved elsewhere.

## Resume an interrupted run

Keep the complete job folder, its raw clips and referenced LoRA files. Resume using the **job folder**, not a movie filename:

```bat
python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --resume "C:\AI\Wan2GP\outputs_story_loop_h3_v8\directed_v8_0_YOUR_JOB"
```

Completed raw clips are reused. If the run stopped after saving a clip but before reviewing its ending, that review is retried before generating the next clip. Saved generation settings and the filename are retained. Directed v6/v7 jobs are also accepted, with v8 directing limits applied to future clips. An already overlong active scene is closed before planning another. Legacy text reviews do not automatically become confirmed eliminations. Existing raw files are kept in place. Starting a new v8 run gives the clearest comparison; resuming or reassembling cannot repair repetitive footage already rendered.

## Rebuild the edit only

```bat
python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --assemble "C:\AI\Wan2GP\outputs_story_loop_h3_v8\directed_v8_0_YOUR_JOB"
```

This does not start Wan2GP or request new video generations. It uses the saved clips and cached edits where compatible. Complete directed jobs are exported to the output root; incomplete edits remain inside the job folder. Older jobs without a stored title can use an offline theme-based filename.

## Codec and directing options

HEVC/x265 is the default for new generation, resume and assembly. The old `--hevc` flag is still accepted. To choose H.264/x264 instead:

```bat
python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --h264
```

Use **Extra notes** in Expert mode to guide pacing, dialogue, character appearance and transitions. For example: `An investigation told through distinct clues, decisions and consequences. Prefer cuts. Keep clothing and damage consistent. No background music.`

Theme, genre, style and notes accept `@file.txt` for longer input. Example:

```bat
python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --lora pdd --mode expert --theme "@idea.txt" --notes "@notes.txt"
```

View all command-line options:

```bat
python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --help
```

## Troubleshooting

- **Cannot find `wgp.py`:** place the script directly in your Wan2GP folder, or adjust `WAN2GP_DIR` in the script.
- **Missing H3 preset:** check the `defaults` folder in the Wan2GP installation. The default needs `minimax_h3_fl2va_pruned_pdd.json`; custom adapters use `minimax_h3_fl2va_pruned.json`.
- **Missing API key:** check `.env` placement and spelling, and restart the process after editing it. Ollama mode does not need this key.
- **FFmpeg/FFprobe not found, or unknown encoder:** ensure both executables are on PATH in the active terminal and that FFmpeg includes the selected encoder.
- **LoRA missing on resume:** restore it under the same relative path in `loras/minimax_h3`.
- **Startup banners on MCP stdout:** known banner/backend parsing tracebacks are hidden during initialization. Other errors remain visible. Set `HIDE_WANGP_STARTUP_NOISE = False` in the script to show those messages again.
- **Stopped after rendering a clip:** keep the job folder and use `--resume`. Raw clips are saved before the visual review. Ending frames are extracted by decoded video-frame index, avoiding the previous failure caused by audio/container duration extending beyond the last video frame.
- **Repeated prompt rejected:** the script asks the LLM to correct it before rendering. If it fails all three attempts, the job stops with existing clips saved; `--resume` retries. This prevents silent GPU time spent generating nearly identical prompts.
- **Assembly failed:** raw videos remain available; fix the local FFmpeg/file issue and retry `--assemble`.

## Related projects

- [Wan2GP](https://github.com/deepbeepmeep/Wan2GP)
- [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [OpenRouter](https://openrouter.ai/)
- [Ollama](https://ollama.com/)
