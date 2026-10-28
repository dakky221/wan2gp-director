"""MiniMax H3 Directed Story Loop v8.0 — bounded reactive directing.

Place this script beside wgp.py and run in the existing Wan2GP environment:
    python -m pip install requests pillow openai mcp python-dotenv
    python ollama_mcp_wan_story_loop_h3_directed_v8_0.py

Tested integration baseline: Wan2GP v18.84. FFmpeg/FFprobe must be on PATH.
Put OPENROUTER_API_KEY=sk-or-your-key in .env beside the script (never commit it).
OpenRouter or local Ollama must use a vision-capable model.

First select PDD 8-Step (default) or a compatible local acceleration LoRA from
loras/minimax_h3, then Simple/Expert. Simple asks only idea and runtime; the LLM
fills the creative details. --lora pdd, --lora NAME --steps 6, --mode simple,
--idea TEXT, --minutes N, --backend ollama and --model ID support CLI setup.
--start-image IMAGE optionally anchors the very first clip. Text inputs accept
@file.txt. Local Ollama shares memory with Wan2GP; no automatic unloading.

V8 keeps one-render/one-review feedback. Scene plans contain normally 3 or 4
clips, with a hard maximum of 4; budget remainders may be shorter. Their final
clip is explicitly directed to resolve the scene BEFORE the script enforces a
fresh editorial cut. Cuts are default; motivated fades through black remain.
The director cannot extend a scene or turn it into unlimited animation retries.
The final 4 clips (shorter for small runs) are reserved for a concluding scene.
No full episode of H3 prompts is generated in advance.

Reviews receive four earlier samples (8%, 30%, 55%, 78%) plus the original three
ending frames (90%, 95%, last). They must distinguish observed events from
uncertainty. Continuation prompts see the three ending frames, and H3 receives
only the exact last PNG. A saved ledger tracks character/prop changes and recent
facts; confirmed eliminations cannot reset to active. Near-duplicate prompts
are repaired before rendering; invalid replies stop after 3 attempts with clips
saved for --resume. This is assistance, not a guarantee of visual/story fidelity.

Each generation is standalone (362 frames at 24fps by default), never a growing
Wan2GP Continue Last Video. Existing frame-limit, exact-decoded-frame extraction,
known-startup-noise filter and audio-sync fixes are retained. HEVC/x265 is default;
--h264 selects H.264 and --hevc remains accepted. Video caches are silent; exact
PCM samples are joined before AAC is encoded once per assembled movie.

New output: outputs_story_loop_h3_v8/JOB/raw for native clips. Other intermediates
stay in JOB. Only the completed, safely named film is moved to the output root.
    python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --resume "RUN_FOLDER"
    python ollama_mcp_wan_story_loop_h3_directed_v8_0.py --assemble "RUN_FOLDER"
Resume accepts v6/v7/v8 jobs; legacy active scenes are closed at v8 limits rather
than extended, and existing raw files are kept in place. Old text reviews are
not treated as confirmed status updates. --assemble changes editing only and
cannot repair repetition already present in rendered footage.
"""

import argparse
from difflib import SequenceMatcher
import asyncio
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
import wave
from pathlib import Path
from contextlib import contextmanager

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# =========================
# CONFIG
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")  # Existing environment variables take precedence.

# Keep this script inside your Wan2GP folder, or replace SCRIPT_DIR with its path.
WAN2GP_DIR = SCRIPT_DIR
OUTPUT_DIR = WAN2GP_DIR / "outputs_story_loop_h3_v8"

LLM_BACKEND = "openrouter"  # "openrouter" or "ollama"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma4:26b"
OLLAMA_ENABLE_THINKING = False

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "google/gemma-4-31b-it"
OPENROUTER_ENABLE_REASONING = False
OPENROUTER_SITE_URL = "http://localhost"
OPENROUTER_APP_NAME = "Wan2GP H3 Story Loop"

# This selects the PDD preset, including its PDD heads module and built-in LoRA.
# Do NOT separately add the PDD LoRA or a different acceleration LoRA.
PDD_MODEL_TYPE = "minimax_h3_fl2va_pruned_pdd"
PRUNED_MODEL_TYPE = "minimax_h3_fl2va_pruned"
MODEL_TYPE = "minimax_h3_fl2va_pruned_pdd"
ACTIVATED_LORAS = []  # Empty preserves PDD's built-in acceleration by default.
LORAS_MULTIPLIERS = ""
CUSTOM_LORA_DEFAULT_STEPS = 6
RESOLUTION = "1280x720"
VIDEO_LENGTH = 362  # Frames per new generation window; about 15.1 s at 24 FPS.
FPS = 24
NUM_INFERENCE_STEPS = 8
SAMPLE_SOLVER = "euler"
GUIDANCE_PHASES = 1
GUIDANCE_SCALE = 1.0
FLOW_SHIFT = 12.0
SEED = -1

# The LLM chooses variable scene lengths; only the total clip count is fixed.
HIDE_WANGP_STARTUP_NOISE = True  # Hide known banner/backend stdout parsing tracebacks only.
DIRECTOR_ATTEMPTS = 3
FADE_SECONDS = 1.0  # Legacy --assemble fallback; new v6 boundaries store explicit fades.
EDIT_CRF = 18
HEVC_CRF = 24  # Used with --hevc; lower = higher quality / larger files.
EDIT_PRESET = "fast"  # CPU encoding; avoids competing with H3 for GPU memory.
RECENT_PROMPTS = 3
EDIT_PIPELINE_VERSION = 6  # Sample-accurate PCM audio cache; AAC only at final mux.

# Optional first image. Leave empty to generate the first clip from text.
# Example: START_IMAGE = r"C:\pictures\start.png"
START_IMAGE = ""

# Pin the historical MCP calls used below. For an old Wan2GP that rejects
# --mcp-api-version, set this to "". Keep the stdio transport.
MCP_API_VERSION = "1"
# Add any CLI options you normally need, e.g. ["--attention", "sage2"].
WAN2GP_EXTRA_ARGS = []

# Positions within the NEWEST clip, not within the entire accumulated video.
FRAME_PERCENTS = [0.90, 0.95, 1.0]
REVIEW_FRAME_PERCENTS = [0.08, 0.30, 0.55, 0.78, 0.90, 0.95, 1.0]
SCENE_MIN_CLIPS = 3
SCENE_MAX_CLIPS = 4  # Enforced; the director cannot extend a scene.
MAX_STALLED_CLIPS = 2
POLL_SECONDS = 5
JOB_TIMEOUT_SECONDS = 60 * 60


# =========================
# LLM HELPERS
# =========================

def img_to_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def image_to_data_url(path):
    path = Path(path)
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported LLM image format: {path.suffix}")
    return f"data:{mime};base64,{img_to_base64(path)}"


def ollama_generate(prompt, image_paths=None, temperature=0.7):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": OLLAMA_ENABLE_THINKING,
        "options": {"temperature": temperature},
    }
    if image_paths:
        payload["images"] = [img_to_base64(p) for p in image_paths]
    response = requests.post(OLLAMA_URL, json=payload, timeout=1800)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def openrouter_generate(prompt, image_paths=None, temperature=0.7):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Put OPENROUTER_API_KEY=sk-or-your-key in .env beside this script.")
    content = [{"type": "text", "text": prompt}]
    for path in image_paths or []:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(path)},
        })
    with OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        timeout=1800.0,
        default_headers={"HTTP-Referer": OPENROUTER_SITE_URL, "X-Title": OPENROUTER_APP_NAME},
    ) as client:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            extra_body={"reasoning": {"enabled": OPENROUTER_ENABLE_REASONING}},
        )
    return (response.choices[0].message.content or "").strip()


def llm_generate(prompt, image_paths=None, temperature=0.7):
    if LLM_BACKEND == "openrouter":
        reply = openrouter_generate(prompt, image_paths, temperature)
    elif LLM_BACKEND == "ollama":
        reply = ollama_generate(prompt, image_paths, temperature)
    else:
        raise ValueError(f"Unknown LLM backend: {LLM_BACKEND}")
    if not reply:
        raise RuntimeError("The LLM returned an empty reply.")
    return reply


# =========================
# H3 PROMPT BUILDERS
# =========================

I2V_ALIGNMENT = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)

H3_PROMPT_RULES = """
Write a concise English MiniMax H3 audiovisual prompt. Follow the user's idea.
Describe observable action, natural physical interaction, and clear progression.
Keep style, identities, anatomy, clothing, props, geography and lighting consistent
within a scene, unless a change is requested. Fit actions and any requested speech
into the duration. A director-requested scene change takes priority over keeping
the old location or continuing its motion.

EVERY clip's prompt must be self-contained. H3 cannot read previous prompts,
story history, the director plan or character cards. At the first appearance of
each recurring character in EVERY prompt, reintroduce their character type,
distinctive appearance, clothing and established speaker ID, such as "the human
diplomat in a high-collar white uniform (S1)". Preserve these identities and IDs
across clips. Use descriptive names in action sentences, not bare S1/S2 labels.
Include the actual place, important objects and state needed to understand this
clip. Never write "same as before" instead of a description.

Use exactly these three fields, in this order:
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...

Start Shot 1 without a timestamp. Describe style, composition, subjects, action,
reactions and camera movement in chronological order. Prefer continuous action.
Additional shots are allowed when useful to the idea; avoid unnecessary cuts.
Write each cut as [Shot 2] At 00:03.500, the camera cuts to...
Number shots sequentially; cut times increase and stay inside this clip's duration.
Do not timestamp ordinary actions. Describe camera type, speed and amplitude
only when meaningful; a fixed camera is fine.

Include dialogue, singing or narration only when requested. Keep exact words and
language. Use stable speaker IDs such as (S1) and <d>[English] Exact words.</d>.
For voiceover say "says in an off-screen voiceover" and keep visible lips closed.
For speech across cuts use <scenetrans> at both connections; use <cutoff> only
for an intentionally interrupted ending. Put exact visible text in double quotes.

overall_soundscape: a short paragraph of ambience, physical sounds and nonverbal
vocal sounds. Do not repeat dialogue or music. Use N/A for requested total silence.
non_diegetic_music: audience-only background score. Default to N/A unless the user
requests music; then describe instruments, tempo and dynamics. Music produced
inside the scene belongs in integrated_multimodal_description.

Fades are applied by the editor ONLY. Never describe fades, a black ending frame,
or a black opening frame in the generated footage. Keep action visible throughout.
No markdown, commentary, negative prompt, or unrelated events. Return the three
fields plus the first-frame instruction only when the task says a start frame exists.
""".strip()


def parse_scene_plan(reply, clip_count, character_bible, previous_location=""):
    raw = strip_code_fence(reply)
    try:
        # Accept an otherwise valid object if the model added a short preamble.
        start = raw.index("{")
        plan, end = json.JSONDecoder().raw_decode(raw[start:])
        if raw[start + end:].strip():
            raise ValueError("Extra text after director JSON")
    except (ValueError, TypeError) as exc:
        raise ValueError("Director did not return a valid JSON plan.") from exc
    if not isinstance(plan, dict):
        raise ValueError("Director plan must be an object.")
    for key in ("title", "location", "purpose"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"Director plan is missing {key}.")
        plan[key] = plan[key].strip()
    if previous_location and plan["location"].casefold() == previous_location.casefold():
        raise ValueError("The new scene needs a different location.")
    beats = plan.get("beats")
    if not isinstance(beats, list) or len(beats) != clip_count or not all(isinstance(b, str) and b.strip() for b in beats):
        raise ValueError(f"Director must provide exactly {clip_count} nonempty clip beats.")
    cast = plan.get("characters")
    if not isinstance(cast, list):
        raise ValueError("Director characters must be an array (possibly empty).")
    by_name = {c["name"].casefold(): c for c in character_bible}
    by_id = {c["speaker_id"]: c["name"].casefold() for c in character_bible if c["speaker_id"]}
    merged = [dict(c) for c in character_bible]
    canonical_cast = []
    seen = set()
    for character in cast:
        if not isinstance(character, dict):
            raise ValueError("Each character card must be an object.")
        if not all(isinstance(character.get(k), str) for k in ("name", "speaker_id", "description")):
            raise ValueError("Character cards need name, speaker_id and description strings.")
        card = {k: character[k].strip() for k in ("name", "speaker_id", "description")}
        name, speaker = card["name"].casefold(), card["speaker_id"]
        if not name or not card["description"] or name in seen:
            raise ValueError("Empty or duplicate character card.")
        if speaker and not re.fullmatch(r"S[1-9]\d*", speaker):
            raise ValueError(f"Invalid speaker ID: {speaker}")
        if name in by_name:
            # The director cannot silently change a recurring character's look or ID.
            card = dict(by_name[name])
        else:
            if speaker and speaker in by_id and by_id[speaker] != name:
                raise ValueError(f"Speaker ID {speaker} already belongs to another character.")
            merged.append(card)
            by_name[name] = card
            if speaker:
                by_id[speaker] = name
        seen.add(name)
        canonical_cast.append(card)
    plan["characters"] = canonical_cast
    plan["beats"] = [b.strip() for b in beats]
    return plan, merged


def read_director_json(reply):
    raw = strip_code_fence(reply)
    start = raw.find('{')
    if start < 0:
        raise ValueError('Director reply contains no JSON object.')
    obj, end = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(obj, dict) or raw[start + end:].strip():
        raise ValueError('Return one JSON object only.')
    return obj


def validate_transition(obj, allow_continue=True):
    if not isinstance(obj, dict):
        raise ValueError('transition must be an object with type and seconds.')
    allowed = {'cut', 'fade', 'continue'} if allow_continue else {'cut', 'fade'}
    if obj.get('type') not in allowed:
        raise ValueError(f'transition.type must be one of {sorted(allowed)}.')
    value = obj.get('seconds', 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('transition.seconds must be a finite number.')
    if obj['type'] == 'fade' and not 0 < value <= (VIDEO_LENGTH - 2) / FPS / 2:
        raise ValueError('Use a positive fade duration shorter than half a clip (seconds per side).')
    return {'type': obj['type'], 'seconds': float(value) if obj['type'] == 'fade' else 0.0}


def build_director_request(manifest, incoming):
    choices, phase = scene_options(manifest)
    remaining = manifest['total_segments'] - len(manifest['segments'])
    unseen = [n for n, s in manifest['story_state']['characters'].items() if not s.get('introduced')]
    handoff = manifest['segments'][-1].get('assessment', {}) if manifest['segments'] else {}
    return f'''You are the director of a film being generated ONE CLIP AT A TIME.
Plan ONLY the next scene. This is v8: scene lengths are ENFORCED, not extendable.
Clips remaining: {remaining}, each about {VIDEO_LENGTH / FPS:.2f} seconds.
Choose EXACTLY one of these scene lengths: {choices}. The script will cut after
that many clips. The last {manifest['directing_config']['finale_clips']} clips of
the film are reserved for its ending. Never spend them extending earlier action.
STORY PHASE: {phase}

Original user idea: {manifest.get('user_idea') or manifest['theme']}
Theme: {manifest['theme']}
Genre: {manifest['genre']}
Style: {manifest['style']}
User notes: {manifest['extra']}
Incoming edit: {json.dumps(incoming)}
Previous handoff: {json.dumps(handoff, ensure_ascii=False)}
Complete cast roster: {json.dumps(manifest['character_bible'], ensure_ascii=False)}
Requested characters not yet visually introduced: {json.dumps(unseen)}
Story record:\n{story_context(manifest)}

Give this scene a NEW dramatic objective and a visible CHANGE by its end.
Do not restart the previous scene's fight, grab, sword draw or failed animation
under a different title/location. If the last scene stalled, abandon its exact
choreography: use a fresh angle, a different activity or a later consequence.
Unresolved events stay uncertain. Do not declare a death or elimination merely
because a character is prone/off-screen. Use the ledger; eliminated characters
cannot return as active participants, and discarded props do not reset.
For a tournament, actually progress through encounters/participants; do not
silently shrink a requested roster to the first four characters. Introduce
unseen requested characters while there is room, and budget for a final outcome.
Use one clear development per clip. Beats must differ and advance causally:
establish/act, turn or consequence, then a readable local outcome. The final beat
MUST settle this scene, never introduce another attack or ask for an extension.
For the FINALE, conclude the whole film within these beats; no new subplot.
Do not reduce 15 seconds to a single repetitive head turn or preparation pose.

If images are attached they describe the preceding ending, not proof of its
entire clip. A fresh scene after a cut/fade gets NO old H3 starting image. Its
opening may establish a different time/place/viewpoint, causally connected to
what is actually known. Plain cuts are the default; location changes are useful
when motivated. Do not generate black frames; the editor alone applies fades.
Keep recognizable cast appearances/IDs; include only this scene's active cast.
Do not include eliminated people even as background spectators.

Return JSON only:
{{"title":"...","location":"...","purpose":"NEW change this scene makes",
"episode_goal":"broad goal","ending_goal":"definite intended payoff",
"opening_transition":{{"type":"cut","seconds":0}},
"characters":[{{"name":"...","speaker_id":"S1","description":"..."}}],
"beats":["one concrete development per clip"],
"next_scene_direction":"a different objective after this scene, empty in finale"}}
opening_transition applies only to the film's first clip; otherwise the incoming
edit is already selected. A fade needs a clear time-passage/ending motivation.
'''.strip()


def parse_flexible_scene(reply, manifest, incoming):
    obj = read_director_json(reply)
    choices, phase = scene_options(manifest)
    beats = obj.get('beats')
    if not isinstance(beats, list) or len(beats) not in choices:
        raise ValueError(f'v8 requires exactly one of these clip counts: {choices}; no extensions.')
    plan, bible = parse_scene_plan(json.dumps(obj), len(beats), manifest['character_bible'])
    for a, b in zip(plan['beats'], plan['beats'][1:]):
        if action_similarity(a, b) >= .88:
            raise ValueError('Scene beats repeat the same action. Give each clip a distinct development.')
    if manifest['scenes'] and action_similarity(plan['purpose'], manifest['scenes'][-1]['plan']['purpose']) >= .88:
        raise ValueError('This repeats the previous scene objective. Move to a new consequence or thread.')
    eliminated = {n.casefold() for n, s in manifest['story_state']['characters'].items() if s['status'] == 'eliminated'}
    if any(c['name'].casefold() in eliminated for c in plan['characters']):
        raise ValueError('An eliminated character cannot rejoin the active cast. Respect the saved ledger.')
    plan['incoming_transition'] = (validate_transition(obj.get('opening_transition'), allow_continue=False)
                                   if not manifest['segments'] else dict(incoming))
    # Even a legacy continuous boundary must stop carrying the old image forward.
    if plan['incoming_transition']['type'] == 'continue':
        plan['incoming_transition'] = {'type': 'cut', 'seconds': 0.0}
    plan['episode_goal'] = str(obj.get('episode_goal', manifest.get('episode_goal', manifest['theme'])))
    plan['ending_goal'] = str(obj.get('ending_goal', manifest.get('ending_goal', 'Resolve the story.')))
    plan['phase'] = phase
    plan['next_scene_direction'] = str(obj.get('next_scene_direction', 'Advance to a different dramatic objective.'))
    return plan, bible


def story_context(manifest):
    state = manifest.get('story_state', {'characters': {}, 'props': {}, 'facts': []})
    lines = [f"Story goal (intention): {manifest.get('episode_goal', manifest['theme'])}",
             f"Intended ending: {manifest.get('ending_goal', 'To be directed.')}",
             'PERSISTENT CONTINUITY LEDGER (observations, not permission to reset):\n' + json.dumps(state, ensure_ascii=False)]
    # Store the full logs on disk, but keep the model's working context bounded.
    for scene in manifest['scenes'][-24:]:
        plan = scene['plan']
        lines.append(f"Scene {scene['scene']}: {plan['title']} | {plan['location']} | "
                     f"Objective: {plan['purpose'][:200]} | {len(scene['clip_indices'])} clips. "
                     f"Outcome: {scene.get('outcome', 'Still in progress / not verified.')[:400]} "
                     f"Boundary: {scene.get('boundary_reason', '')[:160]}")
    for entry in manifest['segments'][-4:]:
        assessment = entry.get('assessment', {})
        if assessment:
            lines.append(f"Clip {entry['segment']} observed ending: {assessment['observed_state']}\n"
                         f"Sampled events: {json.dumps(assessment.get('events', []), ensure_ascii=False)}")
    for entry in manifest['segments'][-2:]:
        lines.append(f"Clip {entry['segment']} PROMPT INTENTION, not verified fact:\n{entry['prompt'][:1100]}")
    return '\n\n'.join(lines)


def build_direction_block(manifest, scene, position, continued):
    plan = scene['plan']
    index = len(manifest['segments']) + 1
    limit = scene['clip_limit']
    # The next review cannot replace this with another attempt at the old beat.
    beat = plan['beats'][position - 1]
    latest = manifest['segments'][-1].get('assessment', {}) if position > 1 else {}
    cast = '\n'.join(f"- {c['name']} ({c['speaker_id']}): {c['description']}" for c in plan['characters'])
    lines = [f"DIRECTOR v8: scene {scene['scene']}, clip {position}/{limit}; HARD limit, no extension.",
             f"Episode clip {index}/{manifest['total_segments']}. Story phase: {plan.get('phase', '')}",
             f"Title: {plan['title']}. Location: {plan['location']}. Objective: {plan['purpose']}",
             f"THIS clip's ASSIGNED development: {beat}",
             f"CAST CARDS (repeat identifying appearance):\n{cast}",
             'Use the last frame as the opening pose only; do not keep a completed action frozen for another clip.',
             'Adapt the assigned development to actual footage. Missing from late samples does NOT mean an action failed.',
             'Never retry the previous grab, draw, punch or preparation merely to obtain perfect choreography.',
             'Show a meaningful change within this clip. Use the saved prop/character status, not their default loadout.',
             'A dropped sword stays dropped. An eliminated person does not stand up or watch from the background.',
             'Keep footage visible through the ending; no generated fades or black frames.']
    if latest:
        lines.append('Latest visual advice (adaptation only; cannot replace the assigned development or extend the scene): ' + latest.get('next_beat', ''))
    if continued:
        lines.append('CONTINUATION: start at the exact last image with the actual subjects/props/positions. Move onward from it.')
    else:
        lines.append('FRESH SHOT after a real editorial cut: establish this scene. Old images are context only, never an H3 start reference.')
    if position == limit:
        lines.append('SCENE BREAK — MANDATORY IN THIS CLIP: finish the local action and show its consequence now. '
                     'The script WILL cut afterwards. No fresh attack, winding up, cliffhanger or additional retry. '
                     'If the planned choreography is infeasible, simplify and show a readable outcome or reaction; do not invent unverified past events.')
    if plan.get('phase', '').startswith('FINALE'):
        lines.append('FINALE: spend these reserved clips resolving the main thread, not introducing or prolonging another confrontation.')
    if index == manifest['total_segments']:
        lines.append('FINAL FILM CLIP: show the decisive outcome and/or aftermath, not a new exchange followed by everyone resetting. Finish dialogue before the end.')
    return '\n'.join(lines), beat


def build_assessment_request(manifest, entry):
    scene = next(s for s in manifest['scenes'] if s['scene'] == entry['scene'])
    remaining = manifest['total_segments'] - len(manifest['segments'])
    closing = len(scene['clip_indices']) >= scene['clip_limit']
    return f'''You are the vision director reviewing SAMPLED EVIDENCE from the latest clip.
Image 1 is a labelled 2x2 overview sampled near 8%, 30%, 55%, 78% of the clip.
Image 2 is three ending frames, TOP TO BOTTOM, near 90%, 95%, and the exact last frame.
Together they cover more of the clip, but they are NOT its complete video/audio.
Do not claim to hear sound. Stationary late frames do NOT prove earlier action
failed. An absent action is UNCERTAIN unless evidence establishes its outcome.
Never demand another attempt just because a punch/grab/drop is not in the ending.

Theme: {manifest['theme']}
User notes: {manifest['extra']}
Clips remaining: {remaining}. Scene clips used: {len(scene['clip_indices'])}/{scene['clip_limit']}.
MANDATORY SCENE BOUNDARY NOW: {closing}.
Current scene plan (intent): {json.dumps(scene['plan'], ensure_ascii=False)}
Story record:\n{story_context(manifest)}
Latest assigned action (not verified fact): {entry['director_beat']}

Describe the actual ending, then any important events VISIBLE in earlier samples.
progress is advanced, stalled or uncertain. Use stalled only for positive evidence
of no meaningful change; incomplete coverage alone is uncertain. Do not count a
single still pose as proof that the entire clip was static. Keep advice concise.
Advance to a consequence, another activity or a new objective; never endlessly
repair the same choreography. For continue, next_beat is advice for adapting the
NEXT assigned development, not permission to replace it with an old action.

Record persistent changes conservatively. Character status is active, defeated,
eliminated or unknown. Prone/off-screen is NOT proof of elimination. Eliminated
requires unambiguous visible elimination/death/destruction evidence, and cannot
be reset to active. observed state updates must cite visible evidence; uncertain
ones are recorded as uncertainty, never made into confirmed world state.
Props stay held/dropped/lost/destroyed until a visible change occurs. Name their
owner and the prop precisely. Empty arrays are fine; do not invent changes.

If this scene hits its limit, scene_action MUST be new_scene (finish if no clips
remain), and transition MUST be cut or a motivated fade. Otherwise you may end
early for a meaningful outcome or continue within the remaining scene slots.
Plain CUT is default. A fresh boundary MUST release the old starting frame.
If only one clip remains, direct a simple outcome/aftermath. No new threat.
Do not extend because a scene feels unresolved. It is permissible to leave an
uncertain detail unresolved and cut to another objective; don't falsely certify it.

Return JSON only:
{{"observed_state":"factual ending; mention uncertainty",
"visible_characters":["known names visibly identified in any sample"],
"events":[{{"fact":"important sampled event","certainty":"observed or uncertain"}}],
"progress":"advanced or stalled or uncertain",
"state_updates":{{"characters":[{{"name":"known name","status":"active or defeated or eliminated or unknown","certainty":"observed or uncertain","evidence":"specific visible evidence"}}],
"props":[{{"owner":"known name or unowned","prop":"specific item","state":"held or dropped or lost or destroyed or unknown","certainty":"observed or uncertain","evidence":"specific visible evidence"}}]}},
"scene_action":"continue or new_scene or finish",
"reason":"pacing reason, never another retry of the same beat",
"transition":{{"type":"continue or cut or fade","seconds":0,"motivation":"required for fade"}},
"next_beat":"adaptation advice, empty for finish",
"scene_outcome":"what has actually changed; say unresolved if uncertain",
"next_scene_goal":"different objective, not a replay of this action"}}
'''.strip()


def parse_assessment(reply, remaining, manifest=None):
    obj = read_director_json(reply)
    for key in ('observed_state', 'reason', 'scene_outcome'):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError(f'Assessment needs nonempty {key}.')
    if obj.get('scene_action') not in ({'continue', 'new_scene'} if remaining else {'finish', 'continue', 'new_scene'}):
        raise ValueError('Choose continue/new_scene while clips remain, finish when none remain.')
    # Parse a valid-but-wrong continuation even at the budget limit; the code
    # below will close it. Do not let a stubborn final review block assembly.
    obj['transition'] = validate_transition(obj.get('transition'), allow_continue=True)
    if obj['transition']['type'] == 'fade':
        if not str(read_director_json(reply)['transition'].get('motivation', '')).strip():
            obj['transition'] = {'type': 'cut', 'seconds': 0.0}
    if obj.get('progress') not in ('advanced', 'stalled', 'uncertain'):
        raise ValueError('progress must be advanced/stalled/uncertain, based on sampled evidence.')
    visible = obj.get('visible_characters')
    if not isinstance(visible, list) or not all(isinstance(n, str) for n in visible):
        raise ValueError('visible_characters must be an array of names.')
    known = {c['name'].casefold(): c['name'] for c in manifest['character_bible']} if manifest else {}
    if manifest and any(n.casefold() not in known for n in visible):
        raise ValueError('Use established names for visible_characters; omit unidentified extras.')
    obj['visible_characters'] = [known.get(n.casefold(), n) for n in visible]
    events = obj.get('events')
    if not isinstance(events, list) or len(events) > 6:
        raise ValueError('Return at most six concise sampled events.')
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get('fact'), str) or not event['fact'].strip() or event.get('certainty') not in ('observed', 'uncertain'):
            raise ValueError('Each event needs a fact and observed/uncertain certainty.')
    changes = obj.get('state_updates')
    if not isinstance(changes, dict) or not all(isinstance(changes.get(k), list) for k in ('characters', 'props')):
        raise ValueError('state_updates needs character and prop arrays.')
    for kind, updates in changes.items():
        if kind not in ('characters', 'props'):
            raise ValueError('Unknown state_updates category.')
        if len(updates) > 12:
            raise ValueError('Too many state changes; record only actual changes.')
        for change in updates:
            required = ('name', 'status', 'certainty', 'evidence') if kind == 'characters' else ('owner', 'prop', 'state', 'certainty', 'evidence')
            if not isinstance(change, dict) or not all(isinstance(change.get(k), str) and change[k].strip() for k in required):
                raise ValueError(f'State update requires nonempty strings: {required}.')
            if change['certainty'] not in ('observed', 'uncertain'):
                raise ValueError('State certainty must be observed/uncertain.')
            if kind == 'characters':
                if manifest and change['name'].casefold() not in known:
                    raise ValueError('State update names an unknown character.')
                change['name'] = known.get(change['name'].casefold(), change['name'])
                if change['status'] not in ('active', 'defeated', 'eliminated', 'unknown'):
                    raise ValueError('Invalid character status.')
            else:
                if change['state'] not in ('held', 'dropped', 'lost', 'destroyed', 'unknown'):
                    raise ValueError('Invalid prop state.')
                if manifest and change['owner'].casefold() not in known and change['owner'].casefold() != 'unowned':
                    raise ValueError('Prop owner must be an established character or unowned.')
                change['owner'] = known.get(change['owner'].casefold(), change['owner'])
    for key in ('next_beat', 'next_scene_goal'):
        if not isinstance(obj.get(key), str):
            raise ValueError(f'{key} must be a string.')
    return obj


async def director_json(run_dir, stage, request, images, validator):
    log_dir = run_dir / 'director'
    log_dir.mkdir(exist_ok=True)
    for attempt in range(1, DIRECTOR_ATTEMPTS + 1):
        # Timestamped logs preserve previous attempts across a resumed run.
        stamp = time.time_ns()
        (log_dir / f'{stage}_{stamp}_request.txt').write_text(request, encoding='utf-8')
        reply = await asyncio.to_thread(llm_generate, request, images, 0.55)
        (log_dir / f'{stage}_{stamp}_reply.txt').write_text(reply, encoding='utf-8')
        try:
            return validator(reply)
        except (ValueError, TypeError, KeyError) as exc:
            if attempt == DIRECTOR_ATTEMPTS:
                raise ValueError(f'{stage}: director failed validation: {exc}. Use --resume to retry from saved clips.') from exc
            print(f'Director correction: {exc}')
            request += f'\nYour last reply failed validation: {exc}. Return a complete corrected JSON. Previous reply:\n{reply}'


def build_first_prompt_request(theme, genre, style, extra, clip_seconds, has_start_image=False,
                               direction="", history="", is_new_scene=False):
    if has_start_image:
        frame_rules = (
            "An actual start image is attached. Treat it as the exact opening frame.\n"
            f"Begin the final prompt with this exact line:\n{I2V_ALIGNMENT}\n"
            "Then add a blank line and the three fields. Develop forward from the image; "
            "there is no final reference frame."
        )
    else:
        frame_rules = (
            "This is TEXT-TO-VIDEO: H3 receives no starting image. An attached old contact sheet, if any, is analytical context ONLY and must not dictate the new composition. Start directly with "
            "integrated_multimodal_description:. Do not mention Picture 1, Picture 2 "
            "or a reference-frame alignment."
        )
    return f"""{H3_PROMPT_RULES}

Create the OPENING clip of {'a NEW scene' if is_new_scene else 'the film'}.
Follow the director's assigned beat and story phase. Establish the place,
characters and stakes before escalation; do not default to an instant fight.
If the director says this is the finale or a one-clip scene, include its resolution.
Theme: {theme}
Genre: {genre}
Visual style: {style}
Extra notes: {extra}
This clip's duration: about {clip_seconds:.2f} seconds.

{frame_rules}

{direction}

Earlier story context (H3 will NOT receive this history):
{history or '(none)'}
{'Start a fresh shot in the director-selected place and time, which may be the same location. Use the director-selected cut/fade; the editor handles it. Do not continue the old shot or mention a nonexistent start image.' if is_new_scene else ''}

Return ONLY the finished H3 prompt.
""".strip()


def build_continuation_request(theme, genre, style, extra, clip_seconds, story_history_text, last_prompt,
                               direction=""):
    return f"""{H3_PROMPT_RULES}

Continue a multi-clip story. The attached image is a VERTICAL CONTACT SHEET of
three late frames from the previous clip, ordered TOP TO BOTTOM in time.
It is analytical evidence of motion, NOT a three-panel image for the video model.
The script extracts the previous clip's exact final frame and passes that PNG as
the next generation's starting image. Each generated video is a standalone clip;
the growing movie and its soundtrack are NOT passed back into H3.
Infer that state from the bottom panel and motion across all three panels.
Do not describe the contact sheet, panels, or three simultaneous subjects.

Theme: {theme}
Genre: {genre}
Visual style: {style}
Extra notes: {extra}
Duration of the NEW continuation: about {clip_seconds:.2f} seconds.

Story history (context only; do not recap):
{story_history_text}

Most recent prompt:
{last_prompt}

{direction}

Begin at the previous ending with the same subjects, positions, props, motion
direction, camera and environment. Continue the current movement immediately;
do not reset poses or replay completed actions. Actual visible state takes priority over an incompatible planned beat. Adapt that beat to what really rendered.
When the director requests SCENE BREAK, complete the motion and resolve the scene;
do not keep extending the fight or introduce another attack. Otherwise advance
the next ASSIGNED development; never use this as permission to repeat a previous beat. Every prompt must explicitly reintroduce its cast using the
character cards: H3 does not know who bare S1/S2 labels refer to from earlier clips.
Cuts may occur later when useful, but preserve the opening continuity.
Shot numbers restart at [Shot 1]; timestamps start at zero for the NEW clip only.
The actual final frame of the previous video is this clip's <Picture 1>, not the
whole contact sheet. There is no Picture 2 or required final reference image.

Return EXACTLY this wrapper (MOTION is analysis and will not go to Wan2GP):
MOTION: <one concise sentence describing the observed motion>
PROMPT:
{I2V_ALIGNMENT}

integrated_multimodal_description: [Shot 1] <the next action sequence>
overall_soundscape: <natural sounds>
non_diegetic_music: <N/A unless music was requested>
""".strip()


def strip_code_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text, count=1)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def parse_motion_and_prompt(text):
    text = strip_code_fence(text)
    motion_match = re.search(r"^MOTION:[ \t]*(.*)$", text, re.I | re.M)
    prompt_match = re.search(r"^PROMPT:[ \t]*", text, re.I | re.M)
    motion = motion_match.group(1).strip() if motion_match else ""
    prompt = text[prompt_match.end():].strip() if prompt_match else text
    return motion, strip_code_fence(prompt)


def prepare_h3_prompt(text, has_start_frame):
    """Validate the body and enforce the real task's first-frame instruction."""
    text = strip_code_fence(text)
    fields = ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"]
    matches = list(re.finditer(
        r"^(integrated_multimodal_description|overall_soundscape|non_diegetic_music):[ \t]*",
        text, re.M,
    ))
    if [m.group(1) for m in matches] != fields:
        raise ValueError("LLM output must contain each H3 field once, in the required order. See the saved reply.")
    body = text[matches[0].start():].strip()
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index < 2 else len(text)
        value = text[match.end():end].strip()
        if not value:
            raise ValueError(f"LLM returned an empty {match.group(1)} field.")
        if index == 0 and not value.startswith("[Shot 1]"):
            raise ValueError("The multimodal description must start with [Shot 1].")
    return f"{I2V_ALIGNMENT}\n\n{body}" if has_start_frame else body


# =========================
# FFMPEG / FRAME HELPERS
# =========================

def run_media_command(command):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed:\n{result.stderr[-3000:]}")
    return result.stdout


def probe_video(video_path):
    data = json.loads(run_media_command([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video_path),
    ]))
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"No video stream in {video_path}")
    duration_value = video.get("duration") or data.get("format", {}).get("duration")
    duration = float(duration_value)
    if duration <= 0:
        raise ValueError(f"No positive video duration in {video_path}")
    return {
        "duration": duration, "width": int(video["width"]), "height": int(video["height"]),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "frames": int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None,
    }


def validate_generated_clip(video_path, expected_frames=VIDEO_LENGTH, fps=FPS):
    """Reject an accumulated/truncated result instead of repeatedly appending its tail."""
    info = probe_video(video_path)
    measured_frames = round(info["duration"] * fps)
    if abs(measured_frames - expected_frames) > 2:
        raise RuntimeError(
            f"Expected a standalone {expected_frames / fps:.2f}s clip, but Wan2GP returned "
            f"{info['duration']:.2f}s ({measured_frames} frames): {video_path}. "
            "Stopping so this result is not repeatedly appended as new footage."
        )
    return info


def configure_edit_codec(settings, hevc=None):
    """CLI selects a codec; internal assembly calls without a choice retain it."""
    previous = settings.get("video_codec", "h264")
    selected = previous if hevc is None else ("hevc" if hevc else "h264")
    if selected != previous:
        settings["crf"] = HEVC_CRF if selected == "hevc" else EDIT_CRF
    settings["video_codec"] = selected
    if settings["video_codec"] not in ("h264", "hevc"):
        raise ValueError("Edit video_codec must be h264 or hevc.")
    return settings["video_codec"]


def video_encoder_args(settings):
    codec = settings.get("video_codec", "h264")
    if codec not in ("h264", "hevc"):
        raise ValueError(f"Unsupported edit video codec: {codec}")
    args = [
        "-c:v", "libx265" if codec == "hevc" else "libx264",
        "-preset", settings["preset"], "-crf", str(settings["crf"]),
    ]
    if codec == "hevc":
        # Keep CPU thread pools bounded even on hosts reporting hundreds of CPUs.
        pools = max(1, min(16, os.cpu_count() or 4))
        args += ["-tag:v", "hvc1", "-x265-params", f"log-level=error:pools={pools}"]
    return args


def edit_segment(video_path, output_path, continued, starts_scene, ends_scene, canvas, settings, standalone=False, fade_in=None, fade_out=None):
    """Normalize one clip (or a legacy accumulated tail); fade only scene boundaries."""
    info = probe_video(video_path)
    fps = settings["fps"]
    if standalone:
        validate_generated_clip(video_path, settings["video_length"], fps)
    # Legacy manifests may contain accumulated outputs; new runs start at zero.
    span = info["duration"] if standalone else min(info["duration"], settings["video_length"] / fps)
    # I2V repeats the previous ending as this clip's first frame. Keep it once.
    overlap = 1 / fps if continued and span > 1 / fps else 0.0
    start = (0.0 if standalone else max(0.0, info["duration"] - span)) + overlap
    frames = max(1, round((span - overlap) * fps))
    duration = frames / fps
    limit = max(0.0, (duration - 1 / fps) / 2)
    fade_in = min(float(settings.get("fade_seconds", 0) if starts_scene else 0) if fade_in is None else float(fade_in), limit)
    fade_out = min(float(settings.get("fade_seconds", 0) if ends_scene else 0) if fade_out is None else float(fade_out), limit)
    if min(fade_in, fade_out) < 0 or not all(math.isfinite(v) for v in (fade_in, fade_out)):
        raise ValueError("Invalid fade duration.")
    width, height = canvas
    video_filters = [
        "setpts=PTS-STARTPTS",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1", f"fps={fps}", "format=yuv420p",
    ]
    rate = 48000
    samples = round(frames * rate / fps)
    # Frame-count video time and sample-count audio time must agree exactly at
    # every boundary. Do not cache AAC here: each encoder adds delay/padding.
    audio_filters = [
        f"aresample={rate}:async=1:first_pts=0", "apad",
        f"atrim=end_sample={samples}", "asetpts=N/SR/TB",
    ]
    if fade_in > 0:
        video_filters.append(f"fade=t=in:st=0:d={fade_in:.6f}")
        audio_filters.append(f"afade=t=in:st=0:d={fade_in:.6f}")
    if fade_out > 0:
        fade_start = max(0.0, duration - 1 / fps - fade_out)
        video_filters.append(f"fade=t=out:st={fade_start:.6f}:d={fade_out:.6f}")
        audio_filters.append(f"afade=t=out:st={fade_start:.6f}:d={fade_out:.6f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".partial.mp4")
    audio_path = output_path.with_suffix(".wav")
    temporary_audio = output_path.with_name(output_path.stem + ".partial.wav")
    # Video and PCM are written separately so the video encoder cannot end the
    # audio early, and each clip has exactly the required number of audio samples.
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-ss", f"{start:.12f}", "-i", str(video_path)]
    command += [
        "-map", "0:v:0", "-an", "-vf", ",".join(video_filters),
        *video_encoder_args(settings),
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-frames:v", str(frames),
        "-map_metadata", "-1", "-movflags", "+faststart", str(temporary),
    ]
    run_media_command(command)
    if probe_video(temporary)["frames"] != frames:
        raise RuntimeError(f"Edited video frame count does not match the audio timeline: {temporary}")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if info["has_audio"]:
        command += ["-ss", f"{start:.12f}", "-i", str(video_path), "-map", "0:a:0"]
    else:
        command += ["-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=stereo", "-map", "0:a:0"]
    command += [
        "-vn", "-af", ",".join(audio_filters),
        "-c:a", "pcm_s16le", "-ar", str(rate), "-ac", "2",
        "-t", f"{samples / rate:.12f}", "-map_metadata", "-1", str(temporary_audio),
    ]
    run_media_command(command)
    with wave.open(str(temporary_audio), "rb") as pcm:
        if pcm.getnframes() != samples:
            raise RuntimeError(f"Audio sample count mismatch for {video_path}: {pcm.getnframes()} instead of {samples}")
    os.replace(temporary, output_path)
    os.replace(temporary_audio, audio_path)
    return {
        "source_start_seconds": start, "frames": frames, "duration_seconds": duration,
        "fade_in": fade_in, "fade_out": fade_out,
        "audio_samples": samples, "audio_path": str(audio_path),
        "pipeline_version": EDIT_PIPELINE_VERSION,
        "video_codec": settings.get("video_codec", "h264"),
        "crf": settings["crf"], "preset": settings["preset"],
    }


def join_pcm_audio(audio_paths, output_path):
    """Append decoded PCM samples exactly, with no encoder delay at clip joins."""
    total_samples = 0
    with wave.open(str(output_path), "wb") as combined:
        combined.setnchannels(2)
        combined.setsampwidth(2)
        combined.setframerate(48000)
        for path in audio_paths:
            with wave.open(str(path), "rb") as clip:
                if (clip.getnchannels(), clip.getsampwidth(), clip.getframerate(), clip.getcomptype()) != (2, 2, 48000, "NONE"):
                    raise ValueError(f"Unexpected PCM format in {path}")
                total_samples += clip.getnframes()
                while True:
                    block = clip.readframes(65536)
                    if not block:
                        break
                    combined.writeframesraw(block)
    return total_samples


def safe_story_filename(value, fallback='Untitled Story'):
    """Return one short Windows-safe MP4 basename, never a path."""
    name = str(value or '').strip().strip('"\'`')
    name = re.sub(r'\.(mp4|mkv|mov|avi|webm)$', '', name, flags=re.I)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip(' .')[:80].rstrip(' .')
    if not name:
        name = str(fallback or 'Untitled Story')
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip(' .')[:80].rstrip(' .') or 'Untitled Story'
    reserved = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)), *(f'lpt{i}' for i in range(1, 10))}
    if name.split('.')[0].casefold() in reserved:
        name = 'Story ' + name
    return name + '.mp4'


def set_fallback_story_name(manifest):
    """Offline fallback also keeps --assemble usable for older jobs without a title."""
    manifest.setdefault('story_title', manifest.get('theme') or 'Untitled Story')
    manifest['story_filename'] = safe_story_filename(
        manifest.get('story_filename') or manifest['story_title'])


def parse_story_name(reply):
    obj = read_director_json(reply)
    for key in ('title', 'filename'):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError(f'Story naming response needs nonempty {key}.')
    return {'title': obj['title'].strip(), 'filename': safe_story_filename(obj['filename'])}


async def ensure_story_name(run_dir, manifest):
    if manifest.get('story_filename'):
        set_fallback_story_name(manifest)
        return
    request = ('Choose a short memorable movie title and an MP4 filename from this theme. '
               'Do not write a script, shot list or video prompt. Preserve the premise and tone. '
               'Use a readable filename with about 2-7 words; no directory, date, forbidden Windows '
               'filename characters or reserved device names. Return JSON only: '
               '{"title":"Moonlit Reckoning","filename":"Moonlit Reckoning.mp4"}.\n' +
               json.dumps({'theme': manifest['theme'], 'genre': manifest.get('genre', ''),
                           'notes': manifest.get('extra', '')}, ensure_ascii=False))
    try:
        name = await director_json(run_dir, 'story_filename', request, None, parse_story_name)
        manifest.update(story_title=name['title'], story_filename=name['filename'])
    except Exception as exc:
        # Naming must never discard/stop a good episode because its optional call failed.
        print(f'Could not generate a movie filename; using a theme-based name: {exc}')
        set_fallback_story_name(manifest)
    write_json(run_dir / 'story_manifest.json', manifest)
    print(f"Story title: {manifest['story_title']} | final filename: {manifest['story_filename']}")


def story_is_complete(manifest):
    total = manifest.get('total_segments')
    scenes = manifest.get('scenes', [])
    # Old directed v4 manifests did not save the total. Explicit --assemble can
    # finish all their known scenes; v6/v7/v8 always have an exact saved clip budget.
    if total is None and scenes and all(s.get('completed') for s in scenes):
        total = max((i for s in scenes for i in s.get('clip_indices', [])), default=0)
    segments = manifest.get('segments', [])
    if not isinstance(total, int) or total < 1 or len(segments) != total:
        return False
    return (manifest.get('assembled_segments') == total and not manifest.get('assembly_error')
            and bool(segments[-1].get('ends_scene'))
            and bool(scenes) and all(s.get('completed') for s in scenes))


def reserve_story_destination(run_dir, manifest):
    root = run_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    if manifest.get('published_filename'):
        # A saved basename belongs to this job. Reassembly replaces that job's export.
        return root / safe_story_filename(manifest['published_filename'])
    set_fallback_story_name(manifest)
    base = Path(manifest['story_filename']).stem
    counter = 1
    while True:
        filename = base + ('' if counter == 1 else f'_{counter}') + '.mp4'
        destination = root / filename
        try:
            # Atomically reserve the name only after the complete edit is ready.
            # Another job choosing the same title must claim a different filename.
            with destination.open('xb'):
                pass
            break
        except FileExistsError:
            counter += 1
    manifest['published_filename'] = destination.name
    try:
        write_json(run_dir / 'story_manifest.json', manifest)
    except Exception:
        destination.unlink(missing_ok=True)  # Only our newly created reservation.
        manifest.pop('published_filename', None)
        raise
    return destination


def publish_completed_story(run_dir, manifest):
    current = Path(manifest.get('final_video_path') or run_dir / 'story_full.mp4')
    if not story_is_complete(manifest):
        return current
    if not current.is_file():
        # Recover a move that finished just before the last manifest write was interrupted.
        saved = manifest.get('published_filename')
        recovered = run_dir.parent / safe_story_filename(saved) if saved else None
        if recovered and recovered.is_file() and recovered.stat().st_size:
            manifest['final_video_path'] = str(recovered)
            write_json(run_dir / 'story_manifest.json', manifest)
            return recovered
        return current
    destination = reserve_story_destination(run_dir, manifest)
    if current.resolve() != destination.resolve():
        os.replace(current, destination)  # Same filesystem; no re-encoding or duplicate final movie.
    manifest['final_video_path'] = str(destination)
    write_json(run_dir / 'story_manifest.json', manifest)
    return destination


def assemble_story(run_dir, manifest):
    """Cache each new segment once, then join them without repeating accumulated footage."""
    segments = manifest["segments"]
    if not segments:
        raise ValueError("There are no generated segments to assemble yet.")
    settings = manifest["edit_settings"]
    video_codec = configure_edit_codec(settings)
    cache_dir = "edited_directed_v6_hevc" if video_codec == "hevc" else "edited_directed_v6_h264"
    canvas = settings.get("canvas")
    if canvas is None:
        first = probe_video(Path(segments[0]["video_path"]))
        canvas = [first["width"] // 2 * 2, first["height"] // 2 * 2]
        settings["canvas"] = canvas
    edited = []
    audio_paths = []
    total_frames = 0
    for entry in segments:
        # Bypass all v1/v2 AAC caches when repairing an existing run.
        output = run_dir / cache_dir / f"clip_{entry['segment']:04d}.mp4"
        audio_path = output.with_suffix(".wav")
        cached = entry.get("edit", {})
        cache_matches = (
            cached.get("pipeline_version") == EDIT_PIPELINE_VERSION
            and cached.get("video_codec", "h264") == video_codec
            and cached.get("crf", settings["crf"]) == settings["crf"]
            and cached.get("preset", settings["preset"]) == settings["preset"]
            and cached.get("requested_fade_in") == entry.get("fade_in")
            and cached.get("requested_fade_out") == entry.get("fade_out")
        )
        if not output.is_file() or not audio_path.is_file() or not cache_matches:
            entry["edit"] = edit_segment(
                Path(entry["video_path"]), output,
                entry["continues_previous"], entry["starts_scene"], entry["ends_scene"], canvas, settings,
                standalone=entry.get("standalone_clip", False),
                fade_in=entry.get("fade_in"), fade_out=entry.get("fade_out"),
            )
            entry["edit"]["requested_fade_in"] = entry.get("fade_in")
            entry["edit"]["requested_fade_out"] = entry.get("fade_out")
        entry["edited_path"] = str(output)
        entry["audio_path"] = str(audio_path)
        total_frames += entry["edit"]["frames"]
        edited.append(output)
        audio_paths.append(audio_path)
    # All names below are generated ASCII names; source paths never enter the concat syntax.
    concat_file = run_dir / "edit_concat.ffconcat"
    concat_file.write_text(
        "ffconcat version 1.0\n" + "".join(
            f"file '{cache_dir}/{p.name}'\nduration {entry['edit']['frames'] / settings['fps']:.12f}\n"
            for p, entry in zip(edited, segments)
        ), encoding="utf-8",
    )
    combined_pcm = run_dir / "story_audio.partial.wav"
    audio_samples = join_pcm_audio(audio_paths, combined_pcm)
    expected_samples = round(total_frames * 48000 / settings["fps"])
    if abs(audio_samples - expected_samples) > len(segments):
        raise RuntimeError("Combined audio length does not match the video frame timeline.")
    output = run_dir / "story_full.mp4"
    temporary = run_dir / "story_full.partial.mp4"
    run_media_command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "1", "-i", str(concat_file),
        "-i", str(combined_pcm),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        *(["-tag:v", "hvc1"] if video_codec == "hevc" else []),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-t", f"{total_frames / settings['fps']:.12f}",
        "-use_editlist", "1", "-movie_timescale", "48000",
        "-movflags", "+faststart", str(temporary),
    ])
    if probe_video(temporary)["frames"] != total_frames:
        raise RuntimeError("The assembled video has an unexpected frame count.")
    # Preserve a user's already-generated movie when upgrading its old edit.
    backup = run_dir / "story_full_before_v6_edit.mp4"
    if output.is_file() and manifest.get("edit_pipeline_version") != EDIT_PIPELINE_VERSION and not backup.exists():
        shutil.copy2(output, backup)
    os.replace(temporary, output)
    combined_pcm.unlink(missing_ok=True)
    manifest["final_video_path"] = str(output)
    manifest["assembled_segments"] = len(segments)
    manifest["edit_pipeline_version"] = EDIT_PIPELINE_VERSION
    manifest["assembled_video_frames"] = total_frames
    manifest["assembled_video_codec"] = video_codec
    manifest["assembled_audio_samples"] = audio_samples
    manifest.pop("assembly_error", None)
    write_json(run_dir / "story_manifest.json", manifest)
    return publish_completed_story(run_dir, manifest)


def ffprobe_duration(video_path):
    output = run_media_command([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ])
    duration = float(output.strip())
    if duration <= 0:
        raise ValueError(f"Video has no positive duration: {video_path}")
    return duration


def frame_sample_times(duration, newest_clip_seconds, percents):
    # Wan2GP can return an accumulated video. Always inspect only its latest tail.
    span = min(duration, newest_clip_seconds)
    start = max(0.0, duration - span)
    return [min(max(0.0, duration - 1.0 / FPS), start + span * p) for p in percents]


def video_frame_count(video_path, decode=False):
    """Count the VIDEO stream independently of a longer/padded audio track."""
    if not decode:
        count = probe_video(video_path)['frames']
        if count is not None and count > 0:
            return count
    data = json.loads(run_media_command([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(video_path),
    ]))
    streams = data.get('streams', [])
    value = streams[0].get('nb_read_frames') if streams else None
    if not str(value).isdigit() or int(value) < 1:
        raise RuntimeError(f'Could not count decoded video frames: {video_path}')
    return int(value)


def write_frame_indices(video_path, output_paths, indices):
    """Decode in presentation order once; select zero-based frames, never seek past EOF."""
    unique = sorted(set(indices))
    parent = Path(output_paths[0]).parent
    parent.mkdir(parents=True, exist_ok=True)
    # A private temporary directory prevents an old PNG from hiding a failed decode.
    with tempfile.TemporaryDirectory(prefix='decode_', dir=parent) as temporary:
        selected = 'select=' + '+'.join(fr'eq(n\,{i})' for i in unique)
        template = Path(temporary) / 'frame_%03d.png'
        run_media_command([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-i', str(video_path), '-map', '0:v:0', '-an', '-sn', '-dn',
            '-vf', selected, '-vsync', '0', '-frames:v', str(len(unique)),
            '-start_number', '1', str(template),
        ])
        decoded = {index: Path(temporary) / f'frame_{pos:03d}.png'
                   for pos, index in enumerate(unique, 1)}
        if any(not path.is_file() or not path.stat().st_size for path in decoded.values()):
            return False
        # Verify every result before replacing any previous output.
        for path in decoded.values():
            with Image.open(path) as image:
                image.verify()
        for index, destination in zip(indices, output_paths):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_name(destination.stem + '.partial.png')
            shutil.copyfile(decoded[index], pending)
            os.replace(pending, destination)
    return True


def extract_frame_samples(video_path, output_paths, percents, newest_clip_seconds=None):
    if not output_paths or len(output_paths) != len(percents):
        raise ValueError('Each requested sample needs an output path.')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in percents):
        raise ValueError('Sample positions must be between zero and one.')
    for attempt in range(2):
        # Count decoded frames rather than trusting container durations or header
        # counts. A second attempt also tolerates a transient incomplete decode.
        count = video_frame_count(video_path, decode=True)
        span = count if newest_clip_seconds is None else min(count, max(1, round(newest_clip_seconds * FPS)))
        start = count - span
        indices = [min(count - 1, start + round((span - 1) * p)) for p in percents]
        if write_frame_indices(video_path, output_paths, indices):
            return [Path(path) for path in output_paths]
    raise RuntimeError(f'Could not extract the requested decoded video frames from {video_path}. '
                       'The raw clip is saved; check that it decodes normally in FFmpeg.')


def extract_frames(video_path, out_dir, newest_clip_seconds):
    out_dir = Path(out_dir)
    percents = list(FRAME_PERCENTS)
    if percents:
        percents[-1] = 1.0  # The continuation anchor is always the actual last frame.
    paths = [out_dir / f'frame_{i}.png' for i in range(1, len(percents) + 1)]
    return extract_frame_samples(video_path, paths, percents, newest_clip_seconds)


def extract_final_frame(video_path, output_path):
    """Save the exact last decoded video frame even when audio/container duration is longer."""
    return extract_frame_samples(video_path, [Path(output_path)], [1.0])[0]


def build_vertical_contact_sheet(image_paths, out_path):
    images = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    width = max(img.width for img in images)
    resized = []
    for img in images:
        if img.width != width:
            img = img.resize((width, round(img.height * width / img.width)))
        resized.append(img)
    sheet = Image.new("RGB", (width, sum(img.height for img in resized)))
    y = 0
    for img in resized:
        sheet.paste(img, (0, y))
        y += img.height
    sheet.save(out_path, quality=92)
    return out_path


# =========================
# MCP / WANGP HELPERS
# =========================

def parse_tool_result(result):
    data = getattr(result, "structuredContent", None)
    if data is None:
        text_blocks = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        raw = "\n".join(text_blocks)
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            data = raw
    if getattr(result, "isError", False):
        raise RuntimeError(f"Wan2GP MCP error: {data}")
    return data


def get_latest_progress(job_data):
    for event in reversed(job_data.get("events", [])):
        if event.get("kind") == "progress":
            return event.get("data", {})
    return None


def print_clean_status(job_data):
    progress = get_latest_progress(job_data)
    if isinstance(progress, dict) and progress:
        print(f"{progress.get('status', '')} | {progress.get('progress', 0)}% | "
              f"step {progress.get('current_step', '?')}/{progress.get('total_steps', '?')}")
    else:
        print(f"done={job_data.get('done')} | events={len(job_data.get('events', []))}")


def build_generation_source(prompt, start_image=None):
    source = {
        "model_type": MODEL_TYPE,
        "prompt": prompt,
        "resolution": RESOLUTION,
        "video_length": VIDEO_LENGTH,
        "sliding_window_size": VIDEO_LENGTH,
        "sliding_window_overlap": 1,
        "sample_solver": SAMPLE_SOLVER,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_phases": GUIDANCE_PHASES,
        "guidance_scale": GUIDANCE_SCALE,
        "flow_shift": FLOW_SHIFT,
        "seed": SEED,
        "force_fps": str(FPS),
        "repeat_generation": 1,
        "image_mode": 0,
        # Keep the entire structured H3 prompt as ONE prompt, including blank lines.
        "multi_prompts_gen_type": "FG",
        "prompt_enhancer": "",  # Gemma already writes the H3 prompt.
        "video_prompt_type": "",
        "audio_prompt_type": "",  # No audio INPUT; H3 still generates native audio.
        # Never use L/V or video_source: those re-import the growing video/audio
        # and can hit the source-frame cap or repeatedly encode older audio.
        "image_prompt_type": "S" if start_image else "",
        "video_source": None,
    }
    if start_image:
        source["image_start"] = str(Path(start_image).resolve())
    # PDD keeps its built-in adapter. The plain pruned path receives ONLY the
    # selected external adapter, using Wan2GP's standard settings/MCP fields.
    if ACTIVATED_LORAS:
        source["activated_loras"] = list(ACTIVATED_LORAS)
        source["loras_multipliers"] = LORAS_MULTIPLIERS
    # Returned file paths are sufficient; do not request video tensors in memory.
    return source


async def wangp_generate(session, source):
    gen_data = parse_tool_result(await session.call_tool("wangp_generate", {
        "source": source, "wait": False,
    }))
    if not isinstance(gen_data, dict) or "job_id" not in gen_data:
        raise RuntimeError(f"Unexpected wangp_generate response: {gen_data}")
    job_id = gen_data["job_id"]
    print(f"\nJob started: {job_id}")
    started = time.monotonic()
    while True:
        job_data = parse_tool_result(await session.call_tool("wangp_get_job", {"job_id": job_id}))
        if not isinstance(job_data, dict):
            raise RuntimeError(f"Unexpected job response: {job_data}")
        print_clean_status(job_data)
        if job_data.get("done") is True:
            result = job_data.get("result")
            if not isinstance(result, dict) or result.get("success") is not True:
                raise RuntimeError("Generation failed:\n" + json.dumps(job_data, indent=2, ensure_ascii=False))
            files = result.get("generated_files", [])
            videos = [p for p in files if Path(p).suffix.lower() in (".mp4", ".mkv", ".mov", ".webm", ".avi")]
            if not videos:
                raise RuntimeError(f"Job succeeded but returned no video file: {files}")
            path = Path(videos[0])
            if not path.is_absolute():
                path = WAN2GP_DIR / path
            if not path.is_file():
                raise FileNotFoundError(f"Returned video is not available locally: {path}")
            return path
        if time.monotonic() - started > JOB_TIMEOUT_SECONDS:
            try:
                await session.call_tool("wangp_cancel_job", {"job_id": job_id})
            except Exception as exc:
                print(f"Could not request job cancellation: {exc}")
            raise TimeoutError("Wan2GP job timed out; cancellation was requested.")
        await asyncio.sleep(POLL_SECONDS)


# =========================
# STORY / STARTUP HELPERS
# =========================

def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def append_log(path, entry):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def preflight():
    validate_acceleration()
    if not (WAN2GP_DIR / "wgp.py").is_file():
        raise FileNotFoundError(f"Set WAN2GP_DIR to your Wan2GP folder: {WAN2GP_DIR}")
    if not (WAN2GP_DIR / "defaults" / f"{MODEL_TYPE}.json").is_file():
        raise FileNotFoundError(f"This Wan2GP installation does not contain {MODEL_TYPE}. Update Wan2GP first.")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is not on PATH. Run from your Wan2GP conda environment.")
    if LLM_BACKEND == "openrouter" and not OPENROUTER_API_KEY:
        raise RuntimeError(f"Set OPENROUTER_API_KEY in {SCRIPT_DIR / '.env'}")
    if VIDEO_LENGTH < 107 or VIDEO_LENGTH > 362 or (VIDEO_LENGTH - 5) % 17:
        raise ValueError("Use an H3 frame count from 107 to 362 inclusive (17k + 5).")
    if NUM_INFERENCE_STEPS != 8 and MODEL_TYPE.endswith("_pdd"):
        raise ValueError("This PDD 8-Step preset requires NUM_INFERENCE_STEPS = 8.")
    if START_IMAGE and not Path(START_IMAGE).is_file():
        raise FileNotFoundError(f"START_IMAGE does not exist: {START_IMAGE}")
    if FADE_SECONDS < 0:
        raise ValueError("FADE_SECONDS cannot be negative.")


async def capture_ending(run_dir, entry):
    frame_dir = run_dir / 'frames' / f"segment_{entry['segment']:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = [frame_dir / f'review_frame_{i + 1}.png' for i in range(len(REVIEW_FRAME_PERCENTS))]
    await asyncio.to_thread(extract_frame_samples, Path(entry['video_path']), paths,
                            REVIEW_FRAME_PERCENTS, VIDEO_LENGTH / FPS)
    anchor = frame_dir / 'actual_last_frame.png'
    await asyncio.to_thread(shutil.copyfile, paths[-1], anchor)
    paths[-1] = anchor
    sheet = run_dir / 'sheets' / f"segment_{entry['segment']:04d}_contact_sheet.jpg"
    sheet.parent.mkdir(exist_ok=True)
    await asyncio.to_thread(build_vertical_contact_sheet, paths[-3:], sheet)
    overview = sheet.with_name(f"segment_{entry['segment']:04d}_overview.jpg")
    stamps = [p * (VIDEO_LENGTH - 1) / FPS for p in REVIEW_FRAME_PERCENTS[:4]]
    await asyncio.to_thread(build_overview_contact_sheet, paths[:4], overview, stamps)
    entry['review_overview'] = str(overview)
    return anchor, sheet


async def review_last_clip(run_dir, manifest):
    entry = manifest['segments'][-1]
    anchor, sheet = await capture_ending(run_dir, entry)
    entry['ending_frame'], entry['contact_sheet'] = str(anchor), str(sheet)
    if not entry.get('assessment'):
        entry['assessment'] = await director_json(run_dir, f"clip_{entry['segment']:04d}_review",
            build_assessment_request(manifest, entry), [Path(entry['review_overview']), sheet],
            lambda reply: parse_assessment(reply, manifest['total_segments'] - len(manifest['segments']), manifest))
        print(f"Observed ending: {entry['assessment']['observed_state']}")
    apply_story_state(manifest, entry)
    assessment = enforce_scene_policy(manifest, entry)
    print(f"Director: {assessment['scene_action']} | {assessment['transition']['type']} | "
          f"{assessment.get('policy_reason') or assessment['reason']}")
    scene = next(s for s in manifest['scenes'] if s['scene'] == entry['scene'])
    entry['ends_scene'] = assessment['scene_action'] in ('new_scene', 'finish')
    transition = assessment['transition']
    entry['fade_out'] = transition['seconds'] if entry['ends_scene'] and transition['type'] == 'fade' else 0.0
    scene['completed'] = entry['ends_scene']
    if scene['completed']:
        scene['outgoing_transition'] = transition
    write_json(run_dir / 'story_manifest.json', manifest)
    return anchor, sheet


async def assemble_checkpoint(run_dir, manifest):
    try:
        print('\nAssembling with the director\'s selected cuts and fades...')
        output = await asyncio.to_thread(assemble_story, run_dir, manifest)
        print(f"Story assembled through clip {manifest['assembled_segments']}: {output}")
        return output
    except Exception as exc:
        manifest['assembly_error'] = str(exc)
        write_json(run_dir / 'story_manifest.json', manifest)
        print(f'Assembly failed; raw clips are saved and generation can continue: {exc}')
        print(f'Retry with: python {Path(__file__).name} --assemble "{run_dir}"')
        return None


async def generate_prompt(run_dir, manifest, scene, position, anchor, sheet):
    index = len(manifest['segments']) + 1
    continued = position > 1
    start_image = anchor if continued else (manifest.get('start_image') if index == 1 else None)
    if continued and not start_image:
        raise ValueError('Continuation requires the actual previous ending PNG.')
    direction, beat = build_direction_block(manifest, scene, position, continued)
    history = story_context(manifest)
    if continued:
        request = build_continuation_request(manifest['theme'], manifest['genre'], manifest['style'],
            manifest['extra'], VIDEO_LENGTH / FPS, history, manifest['segments'][-1]['prompt'], direction=direction)
        images = [sheet]
    else:
        request = build_first_prompt_request(manifest['theme'], manifest['genre'], manifest['style'],
            manifest['extra'], VIDEO_LENGTH / FPS, bool(start_image), direction=direction,
            history=history, is_new_scene=bool(manifest['segments']))
        images = [Path(start_image)] if start_image else ([sheet] if sheet else None)
        if sheet and not start_image:
            request += '\nAttached images show the PREVIOUS scene for context only. This is a fresh shot after a cut, without an old H3 starting image. Preserve known consequences; do not resume its exact fight.'
    for attempt in range(1, DIRECTOR_ATTEMPTS + 1):
        stamp = time.time_ns()
        (run_dir / f'segment_{index:04d}_prompt_request_{stamp}.txt').write_text(request, encoding='utf-8')
        reply = await asyncio.to_thread(llm_generate, request, images, .7)
        (run_dir / f'segment_{index:04d}_llm_reply_{stamp}.txt').write_text(reply, encoding='utf-8')
        try:
            motion, raw = parse_motion_and_prompt(reply) if continued else ('', reply)
            prompt = prepare_h3_prompt(raw, has_start_frame=bool(start_image))
            validate_prompt_progress(prompt, manifest, scene)
            return prompt, motion, start_image, continued, beat
        except (ValueError, TypeError) as exc:
            if attempt == DIRECTOR_ATTEMPTS:
                raise ValueError(f'Clip {index} prompt failed validation: {exc}. No GPU render was started; --resume retries from saved clips.') from exc
            request += f'\nYour reply failed validation: {exc}. Return a corrected prompt following THIS assigned development: {beat}. Previous rejected reply:\n{reply}'


async def render_reactive(run_dir, manifest, session):
    initialize_directing(manifest)
    manifest_path = run_dir / 'story_manifest.json'
    total = manifest['total_segments']
    anchor, sheet = None, None
    while True:
        if manifest['segments']:
            last = manifest['segments'][-1]
            had_review = bool(last.get('assessment'))
            anchor, sheet = await review_last_clip(run_dir, manifest)
            if last['ends_scene'] and (not had_review or manifest.get('assembled_segments') != len(manifest['segments'])):
                await assemble_checkpoint(run_dir, manifest)
        if len(manifest['segments']) >= total:
            break
        scene = manifest['scenes'][-1] if manifest['scenes'] and not manifest['scenes'][-1]['completed'] else None
        if scene is None:
            incoming = manifest['segments'][-1]['assessment']['transition'] if manifest['segments'] else {'type': 'cut', 'seconds': 0}
            scene_number = len(manifest['scenes']) + 1
            print(f'\n=== DIRECTOR: SCENE {scene_number} | {total - len(manifest["segments"])} clips remain ===')
            request = build_director_request(manifest, incoming)
            images = [sheet] if sheet else ([Path(manifest['start_image'])] if manifest.get('start_image') else None)
            if not manifest['segments'] and images:
                request += '\nThe attached image is the actual FIRST starting image, not a contact sheet. Respect its people, clothing, setting and opening composition.'
            plan, bible = await director_json(run_dir, f'scene_{scene_number:03d}_plan', request, images,
                                             lambda reply: parse_flexible_scene(reply, manifest, incoming))
            manifest['character_bible'] = bible
            initialize_directing(manifest)
            if not manifest['scenes']:
                manifest['episode_goal'], manifest['ending_goal'] = plan['episode_goal'], plan['ending_goal']
            scene = {'scene': scene_number, 'plan': plan, 'clip_limit': len(plan['beats']), 'clip_indices': [], 'completed': False}
            manifest['scenes'].append(scene)
            write_json(run_dir / f'scene_{scene_number:03d}_plan.json', plan)
            write_json(manifest_path, manifest)
            print(f"{plan['title']} | {plan['location']} | hard limit: {len(plan['beats'])} clips | incoming {plan['incoming_transition']['type']}")
        position = len(scene['clip_indices']) + 1
        index = len(manifest['segments']) + 1
        print(f'\n=== CLIP {index}/{total} | SCENE {scene["scene"]}, CLIP {position} ===')
        prompt, motion, start_image, continued, beat = await generate_prompt(
            run_dir, manifest, scene, position, anchor, sheet)
        source = build_generation_source(prompt, start_image)
        (run_dir / f'segment_{index:04d}_prompt.txt').write_text(prompt, encoding='utf-8')
        write_json(run_dir / f'segment_{index:04d}_settings.json', source)
        print(f'\nPrompt sent to Wan2GP:\n{prompt}')
        video_path = Path(await wangp_generate(session, source)).resolve()
        info = await asyncio.to_thread(validate_generated_clip, video_path, VIDEO_LENGTH, FPS)
        if any(Path(e['video_path']).resolve() == video_path for e in manifest['segments']):
            raise RuntimeError('Wan2GP returned an already used video path; stopping to avoid duplicate footage.')
        incoming = scene['plan']['incoming_transition']
        entry = {'segment': index, 'scene': scene['scene'], 'scene_position': position,
            'continues_previous': continued, 'starts_scene': position == 1, 'ends_scene': False,
            'standalone_clip': True, 'start_image': str(start_image) if start_image else None,
            'generated_seconds': info['duration'], 'director_beat': beat,
            'prompt': prompt, 'motion_analysis': motion, 'video_path': str(video_path),
            'fade_in': incoming['seconds'] if position == 1 and incoming['type'] == 'fade' else 0.0,
            'fade_out': 0.0}
        manifest['segments'].append(entry)
        scene['clip_indices'].append(index)
        manifest['latest_video_path'] = str(video_path)
        # Save BEFORE the vision review; resume can retry it without regenerating this clip.
        write_json(manifest_path, manifest)
        append_log(run_dir / 'story_log.jsonl', entry)
        print(f'Generated video: {video_path}')
    if (manifest.get('assembled_segments') != total or manifest.get('assembly_error')
            or not Path(manifest.get('final_video_path', '')).is_file()):
        await assemble_checkpoint(run_dir, manifest)
    if manifest.get('assembled_segments') == total and not manifest.get('assembly_error'):
        try:
            await asyncio.to_thread(publish_completed_story, run_dir, manifest)
        except Exception as exc:
            manifest['assembly_error'] = str(exc)
            write_json(manifest_path, manifest)
            print(f'Final movie export needs attention; the completed edit/raw clips are saved: {exc}')
    if manifest.get('assembled_segments') == total and not manifest.get('assembly_error'):
        print(f"\nAll done: {manifest['final_video_path']}")
    else:
        print('\nGeneration finished; raw clips are saved. Retry --assemble to repair the edit.')


def input_text(label, default=''):
    value = input(label).strip() or default
    if value.startswith('@'):
        return Path(value[1:].strip().strip('"')).expanduser().read_text(encoding='utf-8').strip()
    return value


def prepare_start_image(source, run_dir):
    if not source:
        return None
    path = Path(str(source).strip().strip('"')).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    destination = run_dir / 'assets' / 'first_frame.png'
    destination.parent.mkdir(exist_ok=True)
    with Image.open(path) as image:
        image.convert('RGB').save(destination)
    return str(destination)


def generation_config():
    names = ('MODEL_TYPE', 'RESOLUTION', 'VIDEO_LENGTH', 'FPS', 'NUM_INFERENCE_STEPS',
             'SAMPLE_SOLVER', 'GUIDANCE_PHASES', 'GUIDANCE_SCALE', 'FLOW_SHIFT', 'SEED',
             'ACTIVATED_LORAS', 'LORAS_MULTIPLIERS')
    return {name: globals()[name] for name in names}


def validate_resume(manifest):
    if manifest.get('script_version') not in (6, 7, 8):
        raise ValueError('Generation --resume is for directed v6/v7/v8 runs. Use --assemble for an older v4 run.')
    if not isinstance(manifest.get('total_segments'), int) or not 0 <= len(manifest['segments']) <= manifest['total_segments']:
        raise ValueError('Invalid saved clip budget.')
    config = manifest['generation_config']
    # Original v6/v6.1 runs had only built-in PDD and did not save these fields.
    legacy_keys = set(generation_config()) - {'ACTIVATED_LORAS', 'LORAS_MULTIPLIERS'}
    if set(config) == legacy_keys:
        config.update(ACTIVATED_LORAS=[], LORAS_MULTIPLIERS='')
    if set(config) != set(generation_config()):
        raise ValueError('Saved generation settings are incomplete.')
    globals().update(config)  # Continue with this run's settings, even if defaults changed.
    validate_acceleration()
    initialize_directing(manifest)
    seen = set()
    for i, entry in enumerate(manifest['segments'], 1):
        path = Path(entry['video_path']).resolve()
        if entry['segment'] != i or path in seen:
            raise ValueError('Saved clips must be a contiguous sequence with unique raw paths.')
        validate_generated_clip(path, VIDEO_LENGTH, FPS)
        seen.add(path)
    expected = [e['segment'] for scene in manifest['scenes'] for e in manifest['segments'] if e['scene'] == scene['scene']]
    actual = [i for scene in manifest['scenes'] for i in scene['clip_indices']]
    if actual != expected or sorted(actual) != list(range(1, len(manifest['segments']) + 1)):
        raise ValueError('Saved scene/clip assignment is inconsistent.')


def local_h3_loras():
    root = WAN2GP_DIR / 'loras' / 'minimax_h3'
    if not root.is_dir():
        return []
    return sorted((p.relative_to(root).as_posix() for p in root.rglob('*')
                   if p.is_file() and p.suffix.casefold() == '.safetensors'), key=str.casefold)


def resolve_h3_lora(value):
    root = (WAN2GP_DIR / 'loras' / 'minimax_h3').resolve()
    raw = str(value).strip().strip('"').replace('\\', '/')
    supplied = Path(raw).expanduser()
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'Choose a LoRA inside {root}.') from exc
    if not path.is_file() or path.suffix.casefold() != '.safetensors':
        raise FileNotFoundError(f'LoRA not found: {path}. Put the .safetensors file in {root}.')
    return relative.as_posix()


def configure_acceleration(lora=None, steps=None):
    """Default PDD or exactly one local adapter on the plain FL2VA pruned base."""
    global MODEL_TYPE, NUM_INFERENCE_STEPS, ACTIVATED_LORAS, LORAS_MULTIPLIERS
    if lora is None:
        choices = local_h3_loras()
        print('\n=== H3 acceleration ===')
        print('1. PDD 8-Step (current settings; default)')
        for index, filename in enumerate(choices, 2):
            print(f'{index}. {filename}')
        if not choices:
            print(f'No local LoRAs found in {WAN2GP_DIR / "loras" / "minimax_h3"}.')
        while True:
            answer = input('Acceleration / LoRA [1]: ').strip() or '1'
            if answer.casefold() in ('1', 'pdd', 'default'):
                lora = 'pdd'
                break
            if answer.isdigit() and 2 <= int(answer) <= len(choices) + 1:
                lora = choices[int(answer) - 2]
                break
            if not answer.isdigit():
                try:
                    lora = resolve_h3_lora(answer)
                    break
                except (ValueError, FileNotFoundError):
                    pass
            print('Choose one of the listed numbers or an existing local LoRA filename.')
    if str(lora).strip().casefold() in ('pdd', 'default'):
        if steps is not None and steps != 8:
            raise ValueError('PDD requires 8 steps. Select a local LoRA to use a different step count.')
        MODEL_TYPE, NUM_INFERENCE_STEPS = PDD_MODEL_TYPE, 8
        ACTIVATED_LORAS, LORAS_MULTIPLIERS = [], ''
    else:
        selected = resolve_h3_lora(lora)
        if steps is None:
            while True:
                answer = input(f'Inference steps for {selected} [{CUSTOM_LORA_DEFAULT_STEPS}]: ').strip() or str(CUSTOM_LORA_DEFAULT_STEPS)
                try:
                    steps = int(answer)
                    if steps > 0:
                        break
                except ValueError:
                    pass
                print('Enter a positive integer step count.')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError('Inference steps must be a positive integer.')
        MODEL_TYPE, NUM_INFERENCE_STEPS = PRUNED_MODEL_TYPE, steps
        ACTIVATED_LORAS, LORAS_MULTIPLIERS = [selected], '1.0'
    print(f'Model: {MODEL_TYPE} | {NUM_INFERENCE_STEPS} steps | LoRA: {", ".join(ACTIVATED_LORAS) or "built-in PDD"}\n')


def validate_acceleration():
    if isinstance(NUM_INFERENCE_STEPS, bool) or not isinstance(NUM_INFERENCE_STEPS, int) or NUM_INFERENCE_STEPS < 1:
        raise ValueError('Inference steps must be a positive integer.')
    if not isinstance(ACTIVATED_LORAS, list) or any(not isinstance(p, str) for p in ACTIVATED_LORAS):
        raise ValueError('Saved activated LoRAs must be a list of local filenames.')
    if not isinstance(LORAS_MULTIPLIERS, str):
        raise ValueError('LoRA multipliers must be a string.')
    if MODEL_TYPE.endswith('_pdd'):
        if NUM_INFERENCE_STEPS != 8 or ACTIVATED_LORAS:
            raise ValueError('PDD uses its built-in accelerator at 8 steps; do not stack a custom LoRA on it.')
    elif ACTIVATED_LORAS:
        if MODEL_TYPE != PRUNED_MODEL_TYPE or len(ACTIVATED_LORAS) != 1 or LORAS_MULTIPLIERS != '1.0':
            raise ValueError('Custom acceleration uses one LoRA at 1.0 on the plain FL2VA pruned model.')
        for value in ACTIVATED_LORAS:
            resolve_h3_lora(value)  # Also check again when resuming, before loading models.


def choose_input_mode(value=None):
    aliases = {'1': 'simple', 'simple': 'simple', 's': 'simple',
               '2': 'expert', 'expert': 'expert', 'e': 'expert'}
    if value is not None:
        return aliases[value.strip().lower()]
    while True:
        answer = input('Mode [1 = Simple / 2 = Expert] [1]: ').strip().lower() or '1'
        if answer in aliases:
            return aliases[answer]
        print('Choose 1 (Simple) or 2 (Expert).')


def clips_for_minutes(minutes):
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError('Episode length must be a positive, finite number of minutes.')
    seconds = minutes * 60
    if not math.isfinite(seconds) or seconds < VIDEO_LENGTH / FPS:
        raise ValueError(f'Choose at least {VIDEO_LENGTH / FPS:.2f} seconds ({VIDEO_LENGTH / FPS / 60:.3f} minutes).')
    # Runtime is approximate: clips use the same H3-compatible frame count as v6.
    # The editor also removes one repeated frame from each continuation join.
    return max(1, round(seconds * FPS / VIDEO_LENGTH))


def validate_creative_brief(reply):
    obj = read_director_json(reply)
    for key in ('theme', 'genre', 'visual_style', 'extra_notes'):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError(f'Creative brief needs a nonempty {key}.')
        obj[key] = obj[key].strip()
    characters = obj.get('characters')
    if not isinstance(characters, list):
        raise ValueError('characters must be an array; use an empty array if none are needed.')
    seen = set()
    for card in characters:
        if not isinstance(card, dict):
            raise ValueError('Each character must have name, description and voice strings.')
        for key in ('name', 'description', 'voice'):
            if not isinstance(card.get(key), str) or not card[key].strip():
                raise ValueError(f'Each character needs nonempty {key}. Use silent for a silent character voice.')
            card[key] = card[key].strip()
        if card['name'].casefold() in seen:
            raise ValueError('Character names must be unique.')
        seen.add(card['name'].casefold())
    return obj


async def expand_simple_idea(run_dir, idea, minutes, total, image=None):
    request = f'''You are a creative producer preparing a brief for a REACTIVE video director.
User idea: {idea}
Requested runtime: about {minutes:g} minutes, using {total} clips of about {VIDEO_LENGTH / FPS:.2f} seconds.

Turn this short idea into a clear, entertaining film premise. Choose the theme,
genre, visual style, extra directing notes and concise character references.
Preserve the user's named characters, sides, premise and explicit requests.
For familiar fictional characters, describe their recognizable appearance,
clothing, distinctive props/abilities and voice. The video model cannot reliably
infer their appearance from names alone. Choose one consistent interpretation.
Keep invented supporting cast small; do not remove explicitly requested people
just to simplify the story. Keep it filmable at this runtime. If an actual first
image is attached, its visible people, outfits, setting and style anchor the
opening; describe those accurately rather than contradicting the image.

The theme should give a premise, stakes and a goal suitable for a complete film.
Choose cinematic live-action unless the idea suggests another style. Extra notes
should guide tone, concise dialogue if useful, physical continuity, meaningful
progression, a real ending, and scene objectives that resolve in 3-4 clips.
Prefer plain cuts; use fades only where motivated. The script enforces scenes
of 3-4 clips (shorter when the budget requires), reserving its final clips for
a clear ending. Do not request longer scenes or make every scene the same duel.
Preserve the entire named roster when the runtime permits; use concise references.
A new scene must change the situation, not merely restart a failed action.
Do not force relocation at every scene. Default to no background music
unless the user requests music or a musical premise clearly requires it.
Every H3 clip must reintroduce visible/speaking characters with identifying details.
Damage, props and actions must remain consistent with actually rendered footage.

This is ONLY a creative brief, not a screenplay, fixed shot list, scene schedule
or pre-generated video prompts. The director will observe each actual clip before
writing the next. Leave it room to adapt. Do not introduce extra user questions.
Return JSON only:
{{"theme":"expanded premise","genre":"...","visual_style":"...",
"extra_notes":"concise useful directing preferences, not a rigid script",
"characters":[{{"name":"...","description":"type, build, face/hair, clothing, props and distinctive abilities",
"voice":"voice/accent, or silent"}}]}}
'''.strip()
    brief = await director_json(run_dir, 'simple_creative_brief', request,
                                [Path(image)] if image else None, validate_creative_brief)
    refs = '\n'.join(f"- {c['name']}: {c['description']}. Voice: {c['voice']}." for c in brief['characters'])
    notes = brief['extra_notes']
    if refs:
        notes += '\n\nCHARACTER REFERENCES (repeat identifying details in every relevant H3 prompt):\n' + refs
    notes += '\nKeep character identities, clothing, props and voice consistent; adapt each next action to the actual rendered ending frames.'
    write_json(run_dir / 'creative_brief.json', brief)
    print(f"\nCreative brief prepared:\nTheme: {brief['theme']}\nGenre: {brief['genre']}\nVisual style: {brief['visual_style']}\nExtra notes: {notes}\n")
    return brief, notes


class WanGPStartupNoiseFilter(logging.Filter):
    """Hide only known non-JSON startup banners reported by the MCP stdio reader."""
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled

    def filter(self, record):
        if not self.enabled or record.getMessage() != 'Failed to parse JSONRPC message from server':
            return True
        if not record.exc_info or record.exc_info[1] is None:
            return True
        # Pydantic provides the original input line, independently of the formatted
        # traceback's truncation/escaping, without importing Pydantic here.
        try:
            errors = record.exc_info[1].errors()
        except (AttributeError, TypeError, ValueError):
            return True
        if not isinstance(errors, list) or not errors:
            return True
        for error in errors:
            if not isinstance(error, dict) or error.get('type') != 'json_invalid':
                return True
            line = error.get('input')
            if not isinstance(line, str):
                return True
            line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line).strip()
            if not (line.startswith(('***', '[RAM]', '[VRAM]', '[Triton]', '[INT8]', '[FP8]'))):
                return True
        return False


@contextmanager
def quiet_wangp_startup():
    logger = logging.getLogger('mcp.client.stdio')
    noise_filter = WanGPStartupNoiseFilter(HIDE_WANGP_STARTUP_NOISE)
    logger.addFilter(noise_filter)
    try:
        yield noise_filter
    finally:
        logger.removeFilter(noise_filter)


def initialize_directing(manifest):
    """Persist policy and a small continuity ledger; do not guess old outcomes."""
    total = manifest['total_segments']
    manifest.setdefault('directing_config', {
        'scene_min': SCENE_MIN_CLIPS, 'scene_max': SCENE_MAX_CLIPS,
        'finale_clips': 4 if total >= 8 else (3 if total >= 5 else total),
        'stall_limit': MAX_STALLED_CLIPS})
    policy = manifest['directing_config']
    if not (1 <= policy['scene_min'] <= policy['scene_max'] <= 4
            and 1 <= policy['finale_clips'] <= min(4, total)
            and 1 <= policy['stall_limit'] <= 4):
        raise ValueError('Invalid saved v8 directing limits.')
    bible = manifest.setdefault('character_bible', [])
    known = {c['name'].casefold() for c in bible}
    used = {c['speaker_id'] for c in bible}
    # Keep the full requested roster, not just whoever appeared in scene one.
    for card in manifest.get('creative_brief', {}).get('characters', []):
        if card['name'].casefold() not in known:
            number = 1
            while f'S{number}' in used:
                number += 1
            speaker = f'S{number}'
            bible.append({'name': card['name'], 'speaker_id': speaker,
                          'description': card['description'] + '. Voice: ' + card.get('voice', 'unspecified')})
            known.add(card['name'].casefold())
            used.add(speaker)
    state = manifest.setdefault('story_state', {'characters': {}, 'props': {}, 'facts': []})
    for card in bible:
        state['characters'].setdefault(card['name'], {
            'status': 'unknown', 'introduced': False, 'evidence': 'Not yet visually confirmed.', 'clip': 0})
    # A v6/v7 run can continue, but historical prose is not proof of an elimination.
    if manifest.get('directing_version') != 8:
        for entry in manifest['segments']:
            if entry.get('assessment'):
                entry.setdefault('state_applied', True)
        manifest['directing_version'] = 8
    for scene in manifest['scenes']:
        scene.setdefault('clip_limit', min(len(scene['plan']['beats']), policy['scene_max']))

def scene_options(manifest):
    remaining = manifest['total_segments'] - len(manifest['segments'])
    policy = manifest['directing_config']
    reserve = policy['finale_clips']
    if remaining <= reserve:
        return [remaining], 'FINALE: settle the main conflict and show its lasting consequence or aftermath.'
    available = remaining - reserve
    low, high = min(policy['scene_min'], available), min(policy['scene_max'], available)
    choices = list(range(low, high + 1))
    # Prefer partitions of 3/4, avoiding a tiny leftover scene when possible.
    preferred = [n for n in choices if available - n == 0 or available - n in (3, 4) or available - n >= 6]
    if preferred:
        choices = preferred
    done = len(manifest['segments'])
    if not done:
        phase = 'OPENING: establish the premise, important people and stakes; do not begin an endless fight.'
    elif done / manifest['total_segments'] < .45:
        phase = 'DEVELOPMENT: change the situation through a decision, discovery, completed encounter or consequence.'
    else:
        phase = 'ESCALATION: pay off earlier setups, resolve secondary threads and move toward the reserved finale.'
    return choices, phase

def normalized_action(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()

def action_similarity(a, b):
    # Word comparison avoids difflib's character-frequency heuristic hiding
    # near-identical long prompts after punctuation/whitespace normalization.
    return SequenceMatcher(None, normalized_action(a).split(), normalized_action(b).split(), autojunk=False).ratio()

def apply_story_state(manifest, entry):
    if entry.get('state_applied'):
        return
    state, assessment = manifest['story_state'], entry['assessment']
    clip = entry['segment']
    for name in assessment.get('visible_characters', []):
        state['characters'][name]['introduced'] = True
    for change in assessment.get('state_updates', {}).get('characters', []):
        if change['certainty'] != 'observed' or change['status'] == 'unknown':
            continue
        prior = state['characters'][change['name']]
        if prior['status'] == 'eliminated' and change['status'] != 'eliminated':
            # Rendered contradictions are retained as warnings, never resurrection.
            assessment.setdefault('continuity_conflicts', []).append(f"{change['name']} was eliminated; ignore reset to {change['status']}.")
            continue
        prior.update(status=change['status'], evidence=change['evidence'][:500], clip=clip, introduced=True)
    for change in assessment.get('state_updates', {}).get('props', []):
        if change['certainty'] == 'observed' and change['state'] != 'unknown':
            key = change['owner'].casefold() + ':' + change['prop'].casefold()
            state['props'][key] = {**change, 'clip': clip}
    for event in assessment.get('events', []):
        state['facts'].append({**event, 'clip': clip})
    state['facts'] = state['facts'][-24:]
    entry['state_applied'] = True

def enforce_scene_policy(manifest, entry):
    scene = next(s for s in manifest['scenes'] if s['scene'] == entry['scene'])
    assessment = entry['assessment']
    remaining = manifest['total_segments'] - len(manifest['segments'])
    used = len(scene['clip_indices'])
    recent = [e for e in manifest['segments'] if e['scene'] == scene['scene']][-manifest['directing_config']['stall_limit']:]
    stalled = len(recent) >= manifest['directing_config']['stall_limit'] and all(
        e.get('assessment', {}).get('progress') == 'stalled' for e in recent)
    reason = ''
    if not remaining:
        reason = 'Film clip budget completed.'
    elif used >= scene['clip_limit']:
        reason = f"Hard scene limit reached ({scene['clip_limit']} clips)."
    elif remaining <= manifest['directing_config']['finale_clips'] and not scene['plan'].get('phase', '').startswith('FINALE'):
        reason = 'Reserved finale must begin now.'
    elif stalled:
        reason = 'Repeated positively identified lack of progress; cut away rather than retry choreography.'
    elif assessment.get('continuity_conflicts'):
        reason = 'The render contradicts a confirmed elimination; use a fresh shot without resurrecting the character.'
    if reason:
        assessment.setdefault('model_scene_action', assessment['scene_action'])
        assessment['scene_action'] = 'finish' if not remaining else 'new_scene'
        assessment['policy_reason'] = reason
    if assessment['scene_action'] in ('new_scene', 'finish'):
        # A genuine boundary always releases the old image. Default is a cut.
        if assessment['transition']['type'] not in ('cut', 'fade'):
            assessment['transition'] = {'type': 'cut', 'seconds': 0.0}
        assessment['next_beat'] = ''
        if reason and remaining:
            assessment['next_scene_goal'] = scene['plan'].get('next_scene_direction') or (
                'Advance another unresolved objective or introduce a pending participant. '
                'Do not replay this scene or retry its last choreography; preserve uncertainty.')
        scene['outcome'] = assessment.get('scene_outcome', assessment['observed_state'])
        scene['boundary_reason'] = reason or assessment['reason']
    else:
        assessment['transition'] = {'type': 'continue', 'seconds': 0.0}
    return assessment

def build_overview_contact_sheet(image_paths, out_path, seconds):
    panel_width, label_height = 512, 28
    panels = []
    for path, stamp in zip(image_paths, seconds):
        with Image.open(path) as image:
            frame = image.convert('RGB')
            frame.thumbnail((panel_width, 384))
            panel = Image.new('RGB', (panel_width, frame.height + label_height), '#111111')
            panel.paste(frame, ((panel_width - frame.width) // 2, label_height))
            ImageDraw.Draw(panel).text((8, 7), f'Observed sample near {stamp:.2f}s (not the complete clip)', fill='white')
            panels.append(panel)
    height = max(p.height for p in panels)
    sheet = Image.new('RGB', (panel_width * 2, height * 2), '#111111')
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2) * panel_width, (i // 2) * height))
    sheet.save(out_path, quality=92)
    return out_path

def validate_prompt_progress(prompt, manifest, scene):
    def description(text):
        return text.partition('integrated_multimodal_description:')[2].partition('overall_soundscape:')[0] or text
    for entry in manifest['segments'][-3:]:
        if entry['scene'] == scene['scene'] and action_similarity(description(prompt), description(entry['prompt'])) >= .90:
            raise ValueError('This prompt nearly repeats a recent clip. Advance the ASSIGNED beat; do not retry old choreography.')
    # Names are forbidden in active prompts once a confirmed elimination is saved.
    # Aftermath can show an unnamed fallen body; it must not recast the person.
    content = description(prompt).casefold()
    for name, state in manifest['story_state']['characters'].items():
        if state['status'] == 'eliminated' and re.search(r'(?<!\w)' + re.escape(name.casefold()) + r'(?!\w)', content):
            raise ValueError(f'{name} is eliminated. Remove them from the active prompt, including background spectators.')


async def main(cli):
    global LLM_BACKEND, OLLAMA_MODEL, OPENROUTER_MODEL, START_IMAGE
    print('=== MiniMax H3 Directed v8.0: render, observe, direct the next clip ===')
    if cli.resume:
        run_dir = cli.resume.expanduser().resolve()
        manifest = json.loads((run_dir / 'story_manifest.json').read_text(encoding='utf-8'))
        if any(getattr(cli, name) is not None for name in ('clips', 'minutes', 'idea', 'theme', 'genre', 'style', 'notes', 'start_image', 'lora', 'steps')):
            raise ValueError('--resume keeps the original story/image/budget. Use a new run to change them.')
        validate_resume(manifest)
        LLM_BACKEND = cli.backend or manifest['llm_backend']
        model = cli.model or manifest['llm_model']
        START_IMAGE = manifest.get('start_image') or ''
        previous_codec = manifest['edit_settings'].get('video_codec', 'h264')
        configure_edit_codec(manifest['edit_settings'], cli.hevc)
        if manifest['edit_settings']['video_codec'] != previous_codec:
            manifest.pop('assembled_segments', None)
    else:
        configure_acceleration(cli.lora, cli.steps)
        input_mode = choose_input_mode(cli.mode)
        minutes, idea = None, None
        if input_mode == 'simple':
            idea = cli.idea or cli.theme or input_text('Your idea (or @file.txt): ')
            if not idea.strip():
                raise ValueError('Enter an idea for the film.')
            if cli.clips is not None:
                total = cli.clips
                if total < 1:
                    raise ValueError('Choose at least one clip.')
                minutes = total * VIDEO_LENGTH / FPS / 60
            else:
                minutes = cli.minutes if cli.minutes is not None else float(
                    input_text('Episode length in minutes [3]: ', '3').replace(',', '.'))
                total = clips_for_minutes(minutes)
            # Simple has no provider/model or detailed setup questions.
            LLM_BACKEND = cli.backend or LLM_BACKEND
            model = cli.model or (OLLAMA_MODEL if LLM_BACKEND == 'ollama' else OPENROUTER_MODEL)
            theme, genre, style, extra = idea, '', '', ''
            image = cli.start_image or START_IMAGE
        else:
            LLM_BACKEND = cli.backend or input_text(f'LLM backend [openrouter/ollama] [{LLM_BACKEND}]: ', LLM_BACKEND).lower()
            default = OLLAMA_MODEL if LLM_BACKEND == 'ollama' else OPENROUTER_MODEL
            model = cli.model or input_text(f'Vision model [{default}]: ', default)
            theme = cli.theme if cli.theme is not None else (cli.idea or input_text('Theme (or @file.txt): ', 'a science-fiction adventure'))
            genre = cli.genre if cli.genre is not None else input_text('Genre [science-fiction adventure]: ', 'science-fiction adventure')
            style = cli.style if cli.style is not None else input_text('Visual style [cinematic, live-action]: ', 'cinematic, live-action')
            extra = cli.notes if cli.notes is not None else input_text('Extra notes / directing wishes (or @file.txt): ')
            total = clips_for_minutes(cli.minutes) if cli.minutes is not None else (
                cli.clips if cli.clips is not None else int(input_text('How many clips? [4]: ', '4')))
            if total < 1:
                raise ValueError('Choose at least one clip.')
            image = cli.start_image or input_text('First image path [blank = text-only opening]: ', START_IMAGE)
        # @files also work through command-line arguments.
        values = [theme, genre, style, extra]
        values = [Path(v[1:].strip().strip('"')).expanduser().read_text(encoding='utf-8').strip() if v.startswith('@') else v for v in values]
        theme, genre, style, extra = values
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_dir = OUTPUT_DIR / f"directed_v8_0_{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns() % 1000000:06d}"
        run_dir.mkdir()
        START_IMAGE = prepare_start_image(image, run_dir) or ''
        manifest = {'script_version': 8, 'script_release': '8.0', 'input_mode': input_mode,
            'user_idea': theme if input_mode == 'simple' else None, 'requested_minutes': minutes, 'theme': theme, 'genre': genre, 'style': style, 'extra': extra,
            'total_segments': total, 'model_type': MODEL_TYPE, 'resolution': RESOLUTION,
            'frames_per_clip': VIDEO_LENGTH, 'steps': NUM_INFERENCE_STEPS,
            'llm_backend': LLM_BACKEND, 'llm_model': model, 'start_image': START_IMAGE or None,
            'generation_config': generation_config(), 'clip_pipeline': 'standalone_i2v_last_frame',
            'character_bible': [], 'scenes': [], 'segments': [],
            'edit_settings': {'fps': FPS, 'video_length': VIDEO_LENGTH, 'fade_seconds': 0.0,
                              'crf': EDIT_CRF, 'preset': EDIT_PRESET}}
        configure_edit_codec(manifest['edit_settings'], cli.hevc)
    if LLM_BACKEND not in ('ollama', 'openrouter'):
        raise ValueError('Choose ollama or openrouter, with a vision-capable model.')
    if LLM_BACKEND == 'ollama':
        OLLAMA_MODEL = model
    else:
        OPENROUTER_MODEL = model
    manifest['llm_backend'], manifest['llm_model'] = LLM_BACKEND, model
    preflight()
    if not cli.resume and input_mode == 'simple':
        print(f'Using {LLM_BACKEND} / {model} to expand the idea. Runtime: about {minutes:g} minutes ({total} clips).')
        brief, notes = await expand_simple_idea(run_dir, theme, minutes, total, START_IMAGE)
        manifest.update(theme=brief['theme'], genre=brief['genre'], style=brief['visual_style'],
                        extra=notes, creative_brief=brief)
        # Explicit command-line preferences may refine the automatic choices.
        for arg, field in (('genre', 'genre'), ('style', 'style'), ('notes', 'extra')):
            value = getattr(cli, arg)
            if value is not None:
                if value.startswith('@'):
                    value = Path(value[1:].strip().strip('"')).expanduser().read_text(encoding='utf-8').strip()
                if field == 'extra':
                    manifest[field] += '\nUser override: ' + value
                else:
                    manifest[field] = value
    initialize_directing(manifest)
    await ensure_story_name(run_dir, manifest)
    raw_output_dir = run_dir / 'raw'
    raw_output_dir.mkdir(exist_ok=True)
    manifest['raw_output_dir'] = str(raw_output_dir)
    (run_dir / 'frames').mkdir(exist_ok=True)
    (run_dir / 'sheets').mkdir(exist_ok=True)
    write_json(run_dir / 'story_manifest.json', manifest)
    print(f"Run folder: {run_dir}\nTotal clip budget: {manifest['total_segments']} | output {manifest['edit_settings']['video_codec']}")
    args = ['wgp.py', '--mcp', '--mcp-transport', 'stdio']
    if MCP_API_VERSION:
        args += ['--mcp-api-version', MCP_API_VERSION]
    args += ['--mcp-allow-read-file-system', '--output-dir', str(raw_output_dir), *WAN2GP_EXTRA_ARGS]
    child_env = {k: v for k, v in os.environ.items() if k != 'OPENROUTER_API_KEY'}
    child_env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    server = StdioServerParameters(command=sys.executable, args=args, cwd=str(WAN2GP_DIR), env=child_env)
    with quiet_wangp_startup() as startup_noise:
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                startup_noise.enabled = False  # Keep all post-startup diagnostics visible.
                await render_reactive(run_dir, manifest, session)


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description='Directed v8.0: per-job segments and named final exports, selectable acceleration, Simple/Expert setup, reactive vision, enforced 3-4 clip scenes, persistent state, reserved finale, cuts by default.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--assemble', type=Path, metavar='RUN_FOLDER', help='Rebuild the edit only; supports older directed runs too.')
    modes.add_argument('--resume', type=Path, metavar='RUN_FOLDER', help='Resume an interrupted directed v6/v7/v8 generation.')
    parser.add_argument('--lora', help='pdd (default preset) or one local filename under loras/minimax_h3. Otherwise asked first.')
    parser.add_argument('--steps', type=int, help='Step count for the selected local LoRA; PDD requires 8.')
    codec = parser.add_mutually_exclusive_group()
    codec.add_argument('--hevc', action='store_true', help='HEVC/x265 output; enabled by default (CRF 24).')
    codec.add_argument('--h264', dest='hevc', action='store_false', help='Use H.264/x264 instead of the default HEVC.')
    parser.set_defaults(hevc=True)
    parser.add_argument('--start-image', type=Path, help='Optional first image; copied into the run folder.')
    parser.add_argument('--mode', choices=['simple', 'expert'], help='Skip the startup mode question.')
    parser.add_argument('--idea', help='Short idea for Simple mode; otherwise asked interactively.')
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--clips', type=int, help='Total clip budget; each clip is about 15.08 seconds.')
    duration.add_argument('--minutes', type=float, help='Approximate runtime in minutes, rounded to whole H3 clips.')
    parser.add_argument('--backend', choices=['openrouter', 'ollama'])
    parser.add_argument('--model', help='Vision-capable LLM model ID.')
    parser.add_argument('--theme')
    parser.add_argument('--genre')
    parser.add_argument('--style')
    parser.add_argument('--notes', help='Directing preferences and character descriptions; @file.txt is supported.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    cli = parse_cli_args()
    try:
        if cli.assemble:
            folder = cli.assemble.expanduser().resolve()
            manifest = json.loads((folder / 'story_manifest.json').read_text(encoding='utf-8'))
            configure_edit_codec(manifest['edit_settings'], cli.hevc)
            print(f'Assembled story: {assemble_story(folder, manifest)}')
        else:
            asyncio.run(main(cli))
    except KeyboardInterrupt:
        print('\nStopped. Raw clips are saved; use --resume "RUN_FOLDER" to continue a v6/v7/v8 run.')
