import os
import base64
import time
import math

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import pandas as pd
import numpy as np

app = FastAPI(title="Audio Dataset Statistics API")

# --- 1. Allow ANY website/grader to call this API (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Set up the AI client ---
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_CANDIDATES = ["gemini-3.1-flash-lite", "gemini-3-flash", "gemini-2.5-flash"]


def detect_audio_mime(data: bytes) -> str:
    """Guess the audio format from its file signature."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        return "audio/mp3"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    return "audio/wav"  # fallback guess


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


# The shape Gemini must return: column names + rows of string values
# (kept as strings here; we convert types ourselves afterward using
# pandas, which is far more reliable than trusting an LLM's arithmetic).
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "columns": {"type": "array", "items": {"type": "string"}},
        "data": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
        },
    },
    "required": ["columns", "data"],
}


def decode_audio(audio_base64: str) -> bytes:
    raw = audio_base64.strip()
    raw = raw.split(",")[-1] if "," in raw else raw
    raw = raw.replace("\n", "").replace("\r", "").replace(" ", "")
    missing_padding = len(raw) % 4
    if missing_padding:
        raw += "=" * (4 - missing_padding)
    return base64.b64decode(raw)


def transcribe_table_from_audio(audio_bytes: bytes):
    """Send the audio to Gemini and get back a structured table
    (columns + rows), regardless of the spoken language."""
    mime_type = detect_audio_mime(audio_bytes)

    prompt = (
        "Listen carefully to this audio. It contains someone reading out "
        "a small dataset out loud - column names and rows of data values. "
        "The audio may be in Korean or another language; transcribe and "
        "understand it regardless of language, then translate any labels "
        "into their plain values.\n\n"
        "Return the data as a table:\n"
        "- 'columns': the list of column names, in the order mentioned\n"
        "- 'data': a list of rows, each row a list of values (as plain "
        "strings) in the SAME ORDER as 'columns'\n\n"
        "Keep numbers as plain numeric strings (e.g. '42', '3.5') and "
        "categories/text as plain strings."
    )

    response = None
    last_error = None
    for model_name in MODEL_CANDIDATES:
        model_worked = False
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=EXTRACTION_SCHEMA,
                    ),
                )
                model_worked = True
                break
            except Exception as api_error:
                last_error = api_error
                if "429" in str(api_error) or "503" in str(api_error):
                    time.sleep(2 * (attempt + 1))
                    continue
                if "404" in str(api_error) or "NOT_FOUND" in str(api_error):
                    break
                raise
        if model_worked:
            break

    if response is None:
        raise last_error

    import json
    extracted = json.loads(response.text)
    return extracted["columns"], extracted["data"]


def clean_number(value):
    """Convert numpy/pandas numeric types into plain JSON-safe Python
    numbers, turning NaN/inf into None."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        # Show whole numbers as ints, everything else as floats.
        return int(f) if f.is_integer() else round(f, 4)
    except (ValueError, TypeError):
        return value


def compute_statistics(columns, data):
    df = pd.DataFrame(data, columns=columns)

    result = {
        "rows": len(df),
        "columns": list(df.columns),
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": [],
    }

    numeric_cols = []

    for col in df.columns:
        series = df[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        is_numeric = numeric_series.notna().all() and len(series) > 0

        # Mode works for both numeric and categorical columns.
        mode_vals = series.mode()
        result["mode"][col] = clean_number(mode_vals.iloc[0]) if not mode_vals.empty else None

        if is_numeric:
            numeric_cols.append(col)
            df[col] = numeric_series
            result["mean"][col] = clean_number(numeric_series.mean())
            result["std"][col] = clean_number(numeric_series.std())
            result["variance"][col] = clean_number(numeric_series.var())
            result["min"][col] = clean_number(numeric_series.min())
            result["max"][col] = clean_number(numeric_series.max())
            result["median"][col] = clean_number(numeric_series.median())
            result["range"][col] = clean_number(numeric_series.max() - numeric_series.min())
            result["value_range"][col] = [
                clean_number(numeric_series.min()),
                clean_number(numeric_series.max()),
            ]
        else:
            # Categorical column: report the set of distinct values seen.
            unique_vals = sorted(series.dropna().unique().tolist())
            result["allowed_values"][col] = unique_vals
            result["value_range"][col] = unique_vals

    # Correlation matrix across numeric columns only.
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(4)
        corr = corr.fillna(0)
        result["correlation"] = corr.values.tolist()
    else:
        result["correlation"] = []

    return result


# --- A simple "is it alive" route, useful for testing ---
@app.get("/")
def home():
    return {"status": "Audio Dataset Statistics API is running"}


# --- The actual endpoint the grader will call ---
@app.post("/analyze-audio")
def analyze_audio(payload: AudioRequest):
    result, _ = run_analysis(payload.audio_base64)
    return result


# --- TEMPORARY debug endpoint: shows the real error if something breaks.
@app.post("/analyze-audio-debug")
def analyze_audio_debug(payload: AudioRequest):
    result, error = run_analysis(payload.audio_base64)
    if isinstance(result, dict):
        result["_debug_error"] = error
    return result


def run_analysis(audio_base64: str):
    empty_result = {
        "rows": 0, "columns": [], "mean": {}, "std": {}, "variance": {},
        "min": {}, "max": {}, "median": {}, "mode": {}, "range": {},
        "allowed_values": {}, "value_range": {}, "correlation": [],
    }
    try:
        audio_bytes = decode_audio(audio_base64)
        if len(audio_bytes) < 100:
            return empty_result, "decoded audio is too small - check the full base64 string was sent"

        columns, data = transcribe_table_from_audio(audio_bytes)
        result = compute_statistics(columns, data)
        return result, None

    except Exception as e:
        print(f"analyze_audio() error: {e}")
        return empty_result, str(e)
