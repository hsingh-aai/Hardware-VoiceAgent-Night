---
name: build-voice-agent
description: Build a voice agent you can talk to that also takes notes you can ask about — AssemblyAI Voice Agent + Streaming STT + LLM Gateway behind a small Python relay, with a live web dashboard, persistent memory, tools (web search, volume, remember), a wake word, and optional hardware (reSpeaker XVF3800 USB mic array, Seeed XIAO ESP32-S3 + NS4168 amp, or a classic ESP32 in an old phone or toy). Use this whenever someone wants to build, rebuild or extend a talking agent, AI phone, talking toy (Furby), voice notetaker or "Voice Agents Night" / workshop project — from scratch or from this repo — or wire a mic and speaker to an ESP32 for one, or debug one that replies to itself, plays silence, ignores its personality, forgets things, wakes up at random, or drops sentences from its notes. Use it even if they don't mention AssemblyAI by name.
---

# Build a voice agent that also takes notes

What gets built: speak into a microphone, an agent answers out loud in character,
and every turn — yours and its — lands in speaker-labelled notes you can ask
questions of later. It remembers facts between calls and can look things up.

```
mic ──► relay.py (laptop) ──► Voice Agent API  (STT → LLM → TTS) ──► speaker
              │          └──► Streaming STT    (accurate, speaker-labelled notes)
              │                       notes ──► LLM Gateway  ("what did I say about…")
              └──► dashboard  http://localhost:8081
```

The relay is the design decision everything else hangs off. It holds the API key,
speaks TLS, and does the JSON framing; any device just moves raw audio over a
plain WebSocket. So the key never lives on hardware, and a microcontroller with
320 KB of RAM never has to run TLS.

## Start by settling two things

1. **Where are they building?** If the `Hardware-VoiceAgent-Night` repo is present
   (look for `relay/relay.py` and `workshop/`), extend it — the working relay, the
   dashboard and `tests/verify.py` already exist. If the folder is empty, build it
   phase by phase as below, using `workshop/phase-*.md` from the repo as the
   reference implementation if they can clone it.
2. **Which audio path?** This decides the hardware section, and nothing on the AI
   side changes between them:
   - **Laptop only** — its own mic and speakers. Fastest. Use headphones, or the
     agent hears itself (see Echo).
   - **reSpeaker XVF3800 over USB** — a 4-mic array with hardware echo
     cancellation and a speaker socket. Plug and play; best audio for the least work.
   - **A device on Wi-Fi** — XIAO ESP32-S3 (+ NS4168 amp), or a classic ESP32 in a
     phone or toy. Cordless, but the most work and the most ways to break.

For anything past "laptop only", read `references/hardware.md` before wiring —
it has the pin maps and the mistakes that cost a board.

## Ground rules, and why

- **The API key goes in `relay/.env`, gitignored, and nowhere else.** Never in
  firmware, never in a commit. `src/secrets.h` (Wi-Fi) is gitignored too. If a key
  ever appears in a chat or a commit, tell the user to rotate it.
- **Prove each step by observing it, not by compiling it.** A build that passes says
  nothing about whether the agent can hear. Every step below has a *done when* that
  you check by running the thing and talking to it.
- **You can test without a human speaking.** Synthesize the line, convert it, and
  stream it in as the microphone would:
  ```bash
  say -v Samantha -o q.aiff "Hey Furby, what's the weather today?"
  afconvert -f WAVE -d LEI16@24000 -c 1 q.aiff q.wav     # 24 kHz mono 16-bit
  ```
  Send the WAV in 50 ms chunks (`input.audio`), then ~1 s of zeros so turn detection
  fires. This is how every behaviour in this skill was verified.
- **Read a file before patching it, and check the patch landed.** A string
  replacement whose anchor no longer matches does nothing and reports nothing.

## Build order

Each phase works on its own and builds on the last. The API details for every
message named here are in `references/api-contracts.md` — read it before writing
the client code; most of its entries are bugs that produce no error.

**1 · Notetaker** — Streaming STT on the mic, printing finished turns.
*Done when:* two people speaking produce lines labelled `A:` and `B:`.

**2 · Ask your notes** — send the notes to LLM Gateway with a question.
*Done when:* it answers from the notes, says when it was said, and says so plainly
when the notes don't cover the question. Pick the model from `GET /v1/models`.

**3 · Voice agent** — `session.update` with a system prompt, greeting, voice; stream
mic in, play `reply.audio` out.
*Done when:* the greeting is audible and "what's your name?" gets an in-character
answer. If it's silent or generic, the contracts file names the cause.

**4 · Both at once** — one mic, two connections, fanned out in the relay. See *Echo*.
*Done when:* you talk to the agent and both your line and its reply appear in notes.

**5 · Memory** — `memory.md`, read fresh into every session's system prompt with
today's date in front. See *Memory*.
*Done when:* a brand-new session answers a question only the file could answer, and
corrects you when you state the fact wrongly.

**6 · Tools** — declare them in `session.tools`, answer `tool.call` with `tool.result`.
Useful ones: `remember` (append to memory.md), `search_web` (SafeSearch on if it
talks to children), `set_volume` / `adjust_volume`, `start_listening` (notes-only).
*Done when:* "turn it up" changes the level and the agent says so.

**7 · Wake word** — in notes-only mode the agent is deaf, so the notetaker listens
for its name to hand back. See *Wake word*.

**8 · Hardware** — `references/hardware.md`.

**9 · Verify the whole thing.** In the repo: start the relay (`./phone`), then
`~/esp-tools/bin/python tests/verify.py --quick`. It plays the part of a phone with
synthesized voices against the live APIs and restores notes, call history and
memory afterwards. Anything new you add should get a check there.

## The traps that fail silently

These are the ones worth knowing before you start, because none of them raises an
error — the system just behaves wrongly.

**Echo.** The mic hears the agent's own voice, transcribes it as the user, and the
agent replies to itself forever. Laptops cancel echo in video calls because mic and
speaker share one clock; here the reply crosses the network twice, so the relay uses
a **half-duplex gate** — ignore the mic while the agent speaks. Two details make or
break it:
- Time the gate by the *duration of audio queued*, not by `reply.done`: audio arrives
  faster than real time, so `reply.done` comes long before the speaker falls silent.
  `speak_until = max(speak_until, now) + len(pcm) / 2 / RATE`
- While gated, send the **notetaker silence, not nothing**. It closes a sentence by
  hearing silence after it; starve it and the sentence before the agent's reply is
  never written down — the agent heard and answered it, but it's missing from notes.

A device with its own speaker should gate on its *own* playback state (bytes in the
DAC/I2S buffer), not on the relay's estimate, which runs early by the network delay.
Hardware AEC (the reSpeaker) is the real fix and allows interrupting the agent.

**Memory needs an identity line and permission to disagree.** A fact stored as
"Sam's mother's birthday — 1 March" doesn't match "my mum's birthday" unless
the prompt says who is speaking. And without an explicit instruction to correct
contradictions, the agent agrees with whatever it's told — backwards for memory.
Keep `memory.md` as markdown (it goes into the prompt verbatim and people edit it by
hand) and gitignore it: it holds real personal facts. Ship `memory.example.md`.
```markdown
## Who you are talking to
- The person speaking is **Sam**. "my", "me" and "I" mean Sam.
## Dates
- Sam's mother's birthday — **1 March**
```
Append to the prompt: *"If someone states something that contradicts the list,
politely correct them and give what you have. Don't simply agree, and don't claim
not to know something written above."*

**A lookup tool can override the memory it should defer to.** If the agent has a
tool that searches past *conversations* (a `recall_notes`), a "when is my…"
question sends it there; the transcript has nothing; the empty result outranks the
memory in the prompt, the agent says it doesn't know — and then `remember`s the
wrong date the user offers next. A short direct question in testing won't show
this; a rambling real conversation will. Three fixes, all cheap:
- the lookup tool returns the memory too, labelled as the reliable source;
- its description says known facts are answered from memory, not looked up;
- `remember` refuses a value that contradicts memory, and asks instead.
Test it with a multi-turn conversation, not a single question.

**The wake word must be an address, not a mention.** With "hey" optional, someone in
the room saying "a Furby that someone bought" woke the agent. Accept "hey/ok/hi" plus the
name anywhere, or the name *followed by a comma, ! or ?* at the start of a
turn — the notetaker's formatter punctuates a direct address ("Furby, wake up") but
not a mention ("Furby workshop starts at six"). Check it against real streaming
transcripts of both kinds, not just strings typed by hand.
```python
_NAME = r"(?:furby|furbie|ferby|firby)"   # include near-misses STT produces
WAKE = re.compile(rf"\b(?:hey|ok|okay|hi|hello)[\s,]+{_NAME}\b|^\W*{_NAME}\s*[,!?]", re.I)
```
Prefer a two-syllable name; "Pip" came back from STT as "Phil".

**Short answers embellish.** A kids-style "one or two sentences" rule makes the model
fill gaps ("build your own little Furby friend" for a voice-agent workshop). If a fact
must be exact, say so in the prompt or relax the length rule for factual answers.

**Tests can write into real data.** Once a `remember` tool exists, a test line like
"remind me to buy milk" gets saved to the real memory. Snapshot and restore
`memory.md` around test runs, the same as notes and call history.

## When it breaks: symptom → cause

| Symptom | Usual cause |
|---|---|
| Connects, reports healthy, plays silence | reading `reply.audio` as `audio`; it's `data` |
| Personality ignored, `system_prompt: ""` echoed | config sent under `config`; send `session` |
| Replies to itself in a loop | echo — headphones, gate, or AEC |
| A sentence the agent answered is missing from notes | gate sends the notetaker nothing instead of silence |
| Notetaker socket closes `3007` | chunks under 50 ms — buffer to ~100 ms |
| Speaker labels always `None` | reading `speaker`; the field is `speaker_label` |
| "I don't know" about a stored fact | no identity line, or prompt not rebuilt per session |
| Agrees with a wrong date | no instruction to contradict |
| "I don't have a note about that" for a fact in memory, then saves the wrong one | a transcript-search tool returned empty and outranked the prompt — return memory from it |
| Wakes when people merely mention it | wake regex accepts a bare mention |
| Mic dead after restarting the relay | CoreAudio hadn't released the USB device — retry opening it |
| Device on Wi-Fi never reaches the relay | 5 GHz-only network, case-sensitive SSID, or client isolation — see hardware.md |

## Workshop context

For "Voice Agents Night" the target is: everyone leaves with a talking agent that
takes notes and remembers them, running on their own laptop, with hardware as the
stretch. Bias toward the laptop or reSpeaker path for a first build — the Wi-Fi
device path is where the hours go.
