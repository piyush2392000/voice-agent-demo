import base64, json, os, re, tempfile, time

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI

load_dotenv(override=True)

eleven = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
llm_client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")

VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]
LLM_MODEL = os.environ["LLM_MODEL"]
COMPANY = os.getenv("COMPANY_NAME", "Your Company")   # set COMPANY_NAME in .env
STT_MODEL = "scribe_v1"
TTS_MODEL = "eleven_flash_v2_5"
MAX_MSGS = 12        # keep the last N messages so long calls stay fast and cheap
MAX_TOKENS = 400     # reasoning models spend tokens thinking, so leave headroom
# reasoning models (gpt-oss) burn tokens thinking; keep that small
LLM_EXTRA = {"reasoning_effort": "low"} if "gpt-oss" in LLM_MODEL else {}
FALLBACK = "Maaf kijiye, mujhe theek se sunai nahi diya. Kya aap dobara bol sakte hain?"

SYSTEM_PROMPT = f"""You are Vikas, a calling agent for {COMPANY}.
Speak natural Hinglish (Hindi in Roman script mixed with English), like a young Indian professional.
Rules:
- Your name is Vikas and you are a man: use masculine forms (kar sakta hoon). If the caller
  uses another name for you, politely correct it once and move on.
- 1-2 short sentences per turn. One question at a time.
- No lists, no markdown, no emojis. Say numbers as words.
- Your goal: politely find out the caller's need, budget, and timeline.
- Never repeat a question the caller already answered. Once you know the need, budget and
  timeline, thank them and confirm the next step.
- If they are busy, offer a callback time. If they say stop, apologize and end politely."""

SPLIT = re.compile(r"(?<=[.!?।])\s+")
CONNECTORS = {"and", "but", "so", "because", "or", "if", "to", "with", "the", "a", "um", "uh",
              "aur", "lekin", "par", "kyunki", "toh", "ki", "matlab", "mera", "meri",
              "और", "लेकिन", "क्योंकि", "तो", "कि", "मतलब", "मेरा", "मेरी"}

calls = {}  # one state per call_id, so parallel calls don't share memory
stats = {"turns": 0, "holds": 0, "hold_confirmed": 0, "barge_ins": 0, "ttfa": []}
app = FastAPI()


def new_call():
    return {"history": [{"role": "system", "content": SYSTEM_PROMPT}], "pending": "",
            "waits": 0, "cut": None, "committed": False, "last_sents": []}


def unfinished(t):
    """Cheap end-of-turn check on the transcript: did the sentence trail off?"""
    t = t.strip().lower()
    if not t or t.endswith((".", "?", "!", "।")):
        return False
    if t.endswith((",", "-", "…", "...")):
        return True
    return t.split()[-1].strip(",.…-'\"") in CONNECTORS


def tts(text):
    chunks = eleven.text_to_speech.convert(
        voice_id=VOICE_ID, text=text, model_id=TTS_MODEL, output_format="mp3_44100_128")
    return base64.b64encode(b"".join(chunks)).decode()


def line(obj):
    return json.dumps(obj, ensure_ascii=False) + "\n"


def commit(c, sents):
    """Save only what was actually sent/spoken, honoring a barge-in cut."""
    if c["cut"] is not None:
        sents = sents[: c["cut"]]
    c["last_sents"] = sents
    if sents:
        c["history"].append({"role": "assistant", "content": " ".join(sents)})
    c["committed"] = True


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/reset")
def reset(call_id: str):
    calls[call_id] = new_call()
    return {"ok": True}


@app.post("/interrupt")
def interrupt(call_id: str, n: int):
    stats["barge_ins"] += 1
    c = calls.get(call_id)
    if not c:
        return {"ok": False}
    c["cut"] = n
    if c["committed"] and c["history"][-1]["role"] == "assistant":
        c["history"][-1]["content"] = " ".join(c["last_sents"][:n])
    return {"ok": True}


@app.post("/metric")
def metric(ttfa_ms: int):
    stats["ttfa"].append(ttfa_ms)
    return {"ok": True}


@app.get("/metrics")
def metrics():
    t = sorted(stats["ttfa"])
    pct = lambda p: t[min(len(t) - 1, int(p * len(t)))] if t else None
    return {"turns": stats["turns"], "ttfa_ms_p50": pct(0.5), "ttfa_ms_p95": pct(0.95),
            "samples": len(t), "holds": stats["holds"],
            "hold_confirmed": stats["hold_confirmed"], "barge_ins": stats["barge_ins"]}


@app.post("/turn")
def turn(audio: UploadFile, call_id: str = Form(...)):
    data = audio.file.read()
    ext = os.path.splitext(audio.filename or "")[1] or ".webm"
    c = calls.setdefault(call_id, new_call())

    def stream():
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix=ext) as f:
            f.write(data); f.flush(); f.seek(0)
            text = eleven.speech_to_text.convert(file=f, model_id=STT_MODEL).text.strip()
        t1 = time.time()
        if not text:
            yield line({"type": "empty"}); return

        if c["pending"]:
            stats["hold_confirmed"] += 1  # user really was mid-sentence
        full = f'{c["pending"]} {text}'.strip()
        if unfinished(full) and c["waits"] < 2:  # semantic end-of-turn: keep listening
            c["pending"], c["waits"] = full, c["waits"] + 1
            stats["holds"] += 1
            yield line({"type": "wait", "text": full}); return
        c["pending"], c["waits"] = "", 0

        stats["turns"] += 1
        yield line({"type": "user", "text": full})
        c["history"].append({"role": "user", "content": full})
        c["committed"], c["cut"] = False, None
        msgs = [c["history"][0]] + c["history"][1:][-MAX_MSGS:]
        sents = []

        def emit(s):
            s = s.strip()
            if not s:
                return
            sents.append(s)
            audio_b64 = tts(s)
            if len(sents) == 1:
                print(f"STT {t1-t0:.2f}s | first sentence (LLM+TTS) {time.time()-t1:.2f}s")
            yield line({"type": "sentence", "text": s, "audio": audio_b64})

        try:
            buf = ""
            for chunk in llm_client.chat.completions.create(
                    model=LLM_MODEL, messages=msgs, max_tokens=MAX_TOKENS,
                    temperature=0.6, stream=True, **LLM_EXTRA):
                if not chunk.choices:
                    continue
                if chunk.choices[0].finish_reason == "length":
                    print("WARNING: reply was cut off by max_tokens")
                buf += chunk.choices[0].delta.content or ""
                parts = SPLIT.split(buf)
                for s in parts[:-1]:
                    yield from emit(s)
                buf = parts[-1]
            yield from emit(buf)
            if not sents:  # model returned nothing: say something instead of going silent
                print("WARNING: empty LLM reply, using fallback line")
                yield from emit(FALLBACK)
            yield line({"type": "done"})
        finally:
            commit(c, sents)

    return StreamingResponse(stream(), media_type="application/x-ndjson")