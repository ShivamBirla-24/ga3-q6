from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import os
import pandas as pd
from openai import OpenAI
import json

app = FastAPI()

# --- AIの設定 ---
# 教授から指定されたAI Pipeのトークンを使用します
token = os.environ.get("AIPIPE_TOKEN", "missing-token")
client = OpenAI(
    api_key=token,
    base_url="https://aipipe.org/openai/v1" # WhisperもこのPipe経由で使えるか確認してください
)

# --- 1. 送られてくるデータの形を定義 ---
class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

# --- 2. 統計データを完璧なJSONにするための補助関数 ---
# 採点システムが求める「厳密一致」をクリアするための専用関数です
def generate_stats_json(df: pd.DataFrame):
    # 数字が入っている列だけを抽出
    numeric_cols = df.select_dtypes(include=['number', 'float', 'int']).columns.tolist()
    # 文字列などが入っている列
    cat_cols = [col for col in df.columns if col not in numeric_cols]

    # 相関関係（correlation）をリストの形に変換
    corr_matrix = df[numeric_cols].corr().reset_index()
    corr_list = corr_matrix.to_dict(orient='records')

    # 全てのキーを採点システムの要求通りに作成
    stats = {
        "rows": len(df),
        "columns": list(df.columns),
        "mean": {col: float(df[col].mean()) for col in numeric_cols},
        "std": {col: float(df[col].std()) for col in numeric_cols},
        "variance": {col: float(df[col].var()) for col in numeric_cols},
        "min": {col: float(df[col].min()) for col in numeric_cols},
        "max": {col: float(df[col].max()) for col in numeric_cols},
        "median": {col: float(df[col].median()) for col in numeric_cols},
        
        # 最頻値（mode）は複数ある場合があるので、最初の1つを取得
        "mode": {col: (df[col].mode().iloc[0] if not df[col].mode().empty else None) for col in df.columns},
        
        "range": {col: float(df[col].max() - df[col].min()) for col in numeric_cols},
        
        # allowed_valuesは、カテゴリ（文字）列のユニークな値のリスト
        "allowed_values": {col: df[col].dropna().unique().tolist() for col in cat_cols},
        
        # value_rangeは数値の [最小値, 最大値]
        "value_range": {col: [float(df[col].min()), float(df[col].max())] for col in numeric_cols},
        
        "correlation": corr_list
    }
    return stats


# --- 3. メインのAPIエンドポイント ---
@app.post("/verify-audio") # 課題でパスが指定されていない場合はここを変更してください
async def process_audio(req: AudioRequest):
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

        # C. サーバー内に一時的な音声ファイル(.wav)を作成して保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        # D. Whisperを使って音声をテキストに変換
        with open(tmp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        
        spoken_text = transcription.text
        print(f"聞き取った韓国語（または翻訳）: {spoken_text}")

        # 使い終わった一時ファイルを削除（お掃除）
        os.remove(tmp_file_path)

        # E. テキストに基づいてデータを読み込む
        # 【重要】ここは課題の指示に従って変更してください。
        # 例：聞き取ったテキストが「iris」なら iris.csv を読み込むなど
        # ここではダミーとして空のDataFrameを作らず、あなたのデータセットを読み込みます。
        
        # 例: df = pd.read_csv("your_dataset.csv") 
        # ※ 現在はテスト用に適当なデータを作っています。必ず実際のデータに置き換えてください！
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [5.5, 6.6, 7.7, 8.8, 9.9],
            "Category": ["cat", "dog", "cat", "bird", "dog"]
        })

        # F. 完璧なJSONフォーマットを生成して返す
        final_json = generate_stats_json(df)
        return final_json

    except Exception as e:
        print(f"エラー発生: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
