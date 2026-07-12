from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import os
import pandas as pd
import speech_recognition as sr
import json

app = FastAPI()

# --- 1. 送られてくるデータの形を定義 ---
class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

# --- 2. 統計データを完璧なJSONにするための専用関数 ---
def generate_stats_json(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['number', 'float', 'int']).columns.tolist()
    cat_cols = [col for col in df.columns if col not in numeric_cols]

    # 相関関係をリスト化
    corr_matrix = df[numeric_cols].corr().reset_index()
    corr_list = corr_matrix.to_dict(orient='records')

    # NaN(欠損値)をJSONでエラーにならないように処理する補助関数
    def safe_float(val):
        return float(val) if pd.notna(val) else None

    stats = {
        "rows": len(df),
        "columns": list(df.columns),
        "mean": {col: safe_float(df[col].mean()) for col in numeric_cols},
        "std": {col: safe_float(df[col].std()) for col in numeric_cols},
        "variance": {col: safe_float(df[col].var()) for col in numeric_cols},
        "min": {col: safe_float(df[col].min()) for col in numeric_cols},
        "max": {col: safe_float(df[col].max()) for col in numeric_cols},
        "median": {col: safe_float(df[col].median()) for col in numeric_cols},
        "mode": {col: (df[col].mode().iloc[0] if not df[col].mode().empty else None) for col in df.columns},
        "range": {col: safe_float(df[col].max() - df[col].min()) for col in numeric_cols},
        "allowed_values": {col: df[col].dropna().unique().tolist() for col in cat_cols},
        "value_range": {col: [safe_float(df[col].min()), safe_float(df[col].max())] for col in numeric_cols},
        "correlation": corr_list
    }
    return stats


# --- 3. メインのAPIエンドポイント ---
@app.post("/verify-audio")
async def process_audio(req: AudioRequest):
    tmp_file_path = None
    try:
        # A. Base64の文字列を綺麗にする
        raw_b64 = req.audio_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        
        missing_padding = len(raw_b64) % 4
        if missing_padding:
            raw_b64 += '=' * (4 - missing_padding)

        # B. 暗号を音声データ（バイト）に変換
        audio_bytes = base64.b64decode(raw_b64)

        # C. サーバー内に一時的な音声ファイル(.wav)を作成
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        # D. Googleの無料AIを使って韓国語を文字起こし（AI Pipeを回避！）
        r = sr.Recognizer()
        with sr.AudioFile(tmp_file_path) as source:
            audio_data = r.record(source)
            
        # 韓国語(ko-KR)として聞き取る
        spoken_text = r.recognize_google(audio_data, language="ko-KR")
        print(f"聞き取った韓国語: {spoken_text}")

        # E. 【超重要】聞き取った言葉に合わせてデータを読み込む
        # ※※ ここは教授から渡されている実際のデータセット名に書き換えてください ※※
        if "아이리스" in spoken_text or "iris" in spoken_text.lower():
            # df = pd.read_csv("iris.csv") # 実際は用意されたCSVを読み込む
            pass
        elif "타이타닉" in spoken_text or "titanic" in spoken_text.lower():
            # df = pd.read_csv("titanic.csv")
            pass
        
        # ⚠️ テスト用のダミーデータ（提出時は必ず実際のCSV読み込みに書き換えること！）
        df = pd.DataFrame({
            "A": [10, 20, 30],
            "B": [1.1, 2.2, 3.3],
            "Category": ["cat", "dog", "bird"]
        })

        # F. 計算してJSONを返す
        final_json = generate_stats_json(df)
        return final_json

    except Exception as e:
        print(f"エラー発生: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # お掃除：一時ファイルを削除
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
