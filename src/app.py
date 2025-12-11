import os
import sys
import re
import math
import json
import time
import textwrap
import base64
import threading
import shutil
import zipfile
import random
import socket
import platform
import logging
from datetime import datetime
from uuid import uuid4
from io import BytesIO
from itertools import islice
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import ffmpeg
import webview
# replicate는 조건부 import (실제 사용 시에만 import)
try:
    import replicate
except ImportError:
    replicate = None
# Google GenAI SDK (새로운 라이브러리)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    send_from_directory,
    send_file,
    Response,
)
from mutagen.mp3 import MP3, HeaderNotFoundError
from PIL import Image, ImageOps

# tkinter는 GUI 스레드에서만 사용 가능하므로 조건부 import
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

from config_manager import ConfigManager
from utils import get_ffmpeg_path, get_ffprobe_path, resource_path


config_manager = ConfigManager()

# 전역 API 키 변수 초기화
ELEVENLABS_API_KEY = ""
REPLICATE_API_TOKEN = ""
GEMINI_API_KEY = ""

# 로그 파일 설정 (Windows 환경에서 디버깅용)
LOG_DIR = os.path.join(os.path.expanduser("~"), "YouTubeMaker_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"youtubemaker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_debug(message: str):
    """디버그 로그 출력 (파일과 콘솔 모두)"""
    logger.debug(message)
    print(f"[DEBUG] {message}")

def log_error(message: str, exc_info=None):
    """에러 로그 출력 (파일과 콘솔 모두)"""
    logger.error(message, exc_info=exc_info)
    print(f"[ERROR] {message}")

def get_default_download_folder() -> str:
    """OS별 기본 다운로드 폴더 경로 반환"""
    system = platform.system().lower()
    home = os.path.expanduser("~")
    
    if system == "windows":
        # Windows: C:\Users\username\Downloads
        downloads = os.path.join(home, "Downloads")
    elif system == "darwin":
        # macOS: ~/Downloads
        downloads = os.path.join(home, "Downloads")
    else:
        # Linux: ~/Downloads
        downloads = os.path.join(home, "Downloads")
    
    # 폴더가 존재하면 반환, 없으면 빈 문자열
    if os.path.isdir(downloads):
        return downloads
    return ""


# 전역 Gemini Client (새로운 라이브러리 사용)
genai_client = None

def reload_api_keys() -> None:
    """Load API keys from the persistent config store."""
    global ELEVENLABS_API_KEY, REPLICATE_API_TOKEN, GEMINI_API_KEY
    global genai_client  # 새로운 라이브러리의 Client 객체
    global genai  # genai 모듈을 전역으로 사용
    
    settings = config_manager.get_all()
    ELEVENLABS_API_KEY = (
        settings.get("elevenlabs_api_key")
        or os.environ.get("ELEVENLABS_API_KEY", "")
    ).strip()
    REPLICATE_API_TOKEN = (
        settings.get("replicate_api_key")
        or os.environ.get("REPLICATE_API_TOKEN", "")
    ).strip()
    GEMINI_API_KEY = (
        settings.get("gemini_api_key")
        or os.environ.get("GEMINI_API_KEY", "")
    ).strip()
    
    # 디버깅: 로드된 키 확인
    print(f"[API 키 로드] Gemini API 키: {'설정됨' if GEMINI_API_KEY else '없음'} (길이: {len(GEMINI_API_KEY)})")
    print(f"[API 키 로드] settings에서 gemini_api_key: {settings.get('gemini_api_key', 'NOT_FOUND')[:20]}...")
    
    # Gemini API Client 초기화 (새로운 라이브러리 방식)
    if GEMINI_API_KEY:
        try:
            # import가 실패했을 수 있으므로 다시 시도
            # genai 모듈을 전역에서 가져오거나 재import
            global genai  # 전역 genai 변수 사용 선언
            
            # genai 모듈이 없거나 None인 경우 재import 시도
            if 'genai' not in globals() or globals().get('genai') is None:
                try:
                    from google import genai
                    print(f"[API 키 로드] google.genai 재import 성공")
                except (ImportError, NameError) as e:
                    print(f"[API 키 로드] google.genai import 실패: {e}")
                    print(f"[API 키 로드] pip install google-genai를 실행해주세요.")
                    genai_client = None
                    return
            
            # genai가 여전히 None인 경우 (import는 성공했지만 모듈이 None)
            if genai is None:
                try:
                    from google import genai
                    print(f"[API 키 로드] google.genai 재import 성공 (genai가 None이었음)")
                except (ImportError, NameError) as e:
                    print(f"[API 키 로드] google.genai import 실패: {e}")
                    genai_client = None
                    return
            
            # 새로운 라이브러리: Client 객체 생성
            genai_client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"[API 키 로드] Gemini API Client 초기화 완료")
        except Exception as e:
            print(f"[API 키 로드] Gemini API Client 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            genai_client = None


reload_api_keys()


def refresh_service_flags() -> None:
    """Update cached booleans for API availability."""
    global replicate_api_available, gemini_available
    replicate_api_available = bool(REPLICATE_API_TOKEN)
    
    # GEMINI_AVAILABLE이 False인 경우 다시 확인
    if not GEMINI_AVAILABLE:
        try:
            from google import genai
            # import 성공하면 사용 가능
            gemini_available = bool(GEMINI_API_KEY) and genai_client is not None
        except ImportError:
            gemini_available = False
    else:
        gemini_available = bool(GEMINI_API_KEY) and GEMINI_AVAILABLE and genai_client is not None
    
    # 디버깅: 서비스 플래그 상태 확인
    print(f"[서비스 플래그] GEMINI_AVAILABLE: {GEMINI_AVAILABLE}, GEMINI_API_KEY: {'있음' if GEMINI_API_KEY else '없음'}, genai_client: {'있음' if genai_client else '없음'}, gemini_available: {gemini_available}")


refresh_service_flags()

# -----------------------------------------------------------------------------
# 경로 설정
# -----------------------------------------------------------------------------
PROJECT_ROOT = resource_path("")
SRC_DIR = resource_path("src")
if not os.path.isdir(SRC_DIR):
    # 개발 환경에서 src 디렉터리를 기준으로 설정
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_FOLDER = os.path.join(SRC_DIR, "static")
TEMPLATE_FOLDER = os.path.join(SRC_DIR, "templates")
ASSETS_BASE_FOLDER = os.path.join(STATIC_FOLDER, "generated_assets")
FINAL_VIDEO_BASE_NAME = "final_video"
SUBTITLE_BASE_NAME = "subtitles"
FONTS_FOLDER = os.path.join(STATIC_FOLDER, "fonts")
SUBTITLE_FONT_NAME = "GmarketSansTTFMedium"

os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(ASSETS_BASE_FOLDER, exist_ok=True)
os.makedirs(FONTS_FOLDER, exist_ok=True)

FFMPEG_BINARY = get_ffmpeg_path()
os.environ["FFMPEG_BINARY"] = FFMPEG_BINARY
ffmpeg_dir = os.path.dirname(FFMPEG_BINARY)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
ffprobe_path = get_ffprobe_path(FFMPEG_BINARY)
if ffprobe_path:
    os.environ["FFPROBE_BINARY"] = ffprobe_path


VIDEO_FPS = 25
SEGMENT_MAX_DURATION_SECONDS = 600
# 성능 최적화: 병렬 워커 수 증가 (4 → 8)
MAX_SCENE_WORKERS = 8  # 이미지/TTS 병렬 처리 워커 수
# 성능 최적화: OpenAI 배치 크기 및 워커 수 증가
OPENAI_PROMPT_BATCH_SIZE = 20  # 배치 크기 증가 (12 → 20)
OPENAI_PROMPT_WORKERS = 4  # 병렬 워커 수 증가 (2 → 4)
# 대본 청크 분할 설정 (5분 분량, 약 50-100문장)
SCRIPT_CHUNK_MAX_SENTENCES = 100  # 한 청크당 최대 문장 수 증가 (80 → 100)
SCRIPT_CHUNK_TARGET_MINUTES = 5  # 목표 분량 (분)
# TTS 병렬 처리 및 재시도 설정
TTS_MAX_CONCURRENT_REQUESTS = int(os.getenv("TTS_MAX_CONCURRENT_REQUESTS", "2"))
TTS_MAX_RETRIES = int(os.getenv("TTS_MAX_RETRIES", "3"))
TTS_RETRY_BASE_DELAY = int(os.getenv("TTS_RETRY_BASE_DELAY", "5"))

# Replicate API Rate Limiting 설정
# 문서: https://replicate.com/docs/topics/predictions/rate-limits
# - 예측 생성: 분당 600개 요청 (최소 0.1초 간격) - 크레딧 $5 이상
# - 크레딧 $5 미만: 분당 6개 요청 (10초 간격)
# - 다른 엔드포인트: 분당 3000개 요청
REPLICATE_MIN_REQUEST_INTERVAL = 0.1  # 최소 요청 간격 (초) - 크레딧 $5 이상 시 분당 600개 제한 준수 (0.1초 간격)
REPLICATE_RATE_LIMIT_RETRY_DELAY = 30  # 429 에러 발생 시 재시도 대기 시간 (초)
_last_replicate_request_time = 0  # 마지막 Replicate API 요청 시간
_replicate_request_lock = threading.Lock()  # 요청 간격 제어를 위한 락
_tts_request_semaphore = threading.BoundedSemaphore(max(1, TTS_MAX_CONCURRENT_REQUESTS))


REALISTIC_STYLE_WRAPPER = (
    "A hyperrealistic, photorealistic masterpiece, 8K, ultra-detailed, sharp focus,"
    " cinematic lighting, shot on a professional DSLR camera with a 50mm lens"
)

REALISTIC_NEGATIVE_PROMPT = (
    "painting, drawing, illustration, cartoon, anime, 3d, cgi, render, sketch, watercolor,"
    " text, watermark, signature, blurry, out of focus"
)

KOREAN_CHAR_PATTERN = json.loads(
    '[["\\u3131","\\u318F"],["\\uAC00","\\uD7A3"]]'
)  # used in ensure_english_text


app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
# Flask 타임아웃 설정 (긴 요청 처리용)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 최대 요청 크기


# =============================================================================
# 보이스 설정
# =============================================================================

ELEVENLABS_VOICE_IDS = [
    "jB1Cifc2UQbq1gR3wnb0",
    "8jHHF8rMqMlg8if2mOUe",
    "uyVNoMrnUku1dZyVEXwD",
    "1W00IGEmNmwmsDeYy7ag",
    "AW5wrnG1jVizOYY7R1Oo",
    "ZJCNdZEjYwkOElxugmW2",
    "U1cJYS4EdbaHmfR7YzHd",
    "KlstlYt9VVf3zgie2Oht",
    "YIHgthsAAn8LcGSRSVXn",
]

_cached_voice_list = None


def get_available_voices():
    global _cached_voice_list
    if _cached_voice_list is not None:
        return _cached_voice_list

    from elevenlabs.client import ElevenLabs

    # 사용자 정의 보이스 ID 가져오기
    custom_voice_ids = config_manager.get("custom_voice_ids", [])
    if not isinstance(custom_voice_ids, list):
        custom_voice_ids = []
    
    # 기본 보이스 ID와 사용자 정의 보이스 ID 합치기
    all_allowed_ids = set(ELEVENLABS_VOICE_IDS) | set(custom_voice_ids)

    voice_entries = []
    try:
        if ELEVENLABS_API_KEY:
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            all_voices = client.voices.get_all()
            
            # API에서 가져온 보이스 중 허용된 ID만 추가
            for voice in all_voices.voices:
                if voice.voice_id in all_allowed_ids:
                    voice_entries.append({"id": voice.voice_id, "name": voice.name or voice.voice_id})

        # API에서 가져오지 못한 보이스 ID도 추가 (이름 없이 ID만)
        known_ids = {entry["id"] for entry in voice_entries}
        for vid in all_allowed_ids:
            if vid not in known_ids:
                # 사용자 정의 보이스는 "(사용자 정의)" 태그 추가
                is_custom = vid in custom_voice_ids
                name = f"{vid} (사용자 정의)" if is_custom else vid
                voice_entries.append({"id": vid, "name": name})
    except Exception as exc:
        print(f"[경고] ElevenLabs 보이스 목록 불러오기 실패: {exc}")
        # API 실패 시에도 기본 보이스 ID와 사용자 정의 보이스 ID는 추가
        for vid in all_allowed_ids:
            is_custom = vid in custom_voice_ids
            name = f"{vid} (사용자 정의)" if is_custom else vid
            voice_entries.append({"id": vid, "name": name})

    if not voice_entries:
        voice_entries = [{"id": "pNInz6AJB3aCYtXltRbl", "name": "Default"}]

    _cached_voice_list = voice_entries
    return _cached_voice_list


def get_default_voice_id():
    voices = get_available_voices()
    return voices[0]["id"] if voices else ELEVENLABS_VOICE_IDS[0]


def is_voice_allowed(voice_id: str) -> bool:
    if not voice_id:
        return False
    allowed = {v["id"] for v in get_available_voices()}
    return voice_id in allowed


def generate_voice_preview_audio(voice_id: str, sample_text: str) -> Optional[bytes]:
    if not is_voice_allowed(voice_id):
        voice_id = get_default_voice_id()

    from elevenlabs.client import ElevenLabs

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": sample_text,
        "model_id": "eleven_turbo_v2_5",  # 더 빠르고 고품질 모델로 업그레이드
        "voice_settings": {
            "stability": 0.75,  # 안정성 증가
            "similarity_boost": 0.85,  # 유사도 증가
            "style": 0.0,
            "use_speaker_boost": True,
        },
        "output_format": "mp3_44100_256",  # 비트레이트 증가로 음질 개선
    }
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Accept": "audio/mpeg"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            print(f"[경고] 보이스 미리듣기 실패: {response.status_code} {response.text}")
            return None
        return response.content
    except Exception as exc:
        print(f"[경고] 보이스 미리듣기 요청 실패: {exc}")
        return None


# =============================================================================
# 공용 유틸리티
# =============================================================================


def cleanup_assets_folder(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def ensure_english_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    for start, end in KOREAN_CHAR_PATTERN:
        if any(ord(start) <= ord(ch) <= ord(end) for ch in text):
            break
    else:
        return text

    if not gemini_available or not GEMINI_API_KEY or genai_client is None:
        return text

    try:
        # 새로운 라이브러리 방식
        from google.genai import types
        target_model = 'gemini-2.5-flash'  # 새로운 라이브러리는 2.5 모델 사용
        prompt = f"You are a translator. Translate the following text into fluent English.\n\n{text}"
        response = genai_client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=200,
            )
        )
        translated = response.text.strip()
        return translated or text
    except Exception as exc:
        print(f"[경고] 번역 요청 중 오류: {exc}")
        return text


def split_script_into_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences = []
    for part in filter(None, text.split("\n")):
        cleaned = part.strip()
        if cleaned:
            sentences.extend([s.strip() for s in cleaned.split("。") if s.strip()])
    final = []
    for sentence in sentences:
        segments = [seg.strip() for seg in sentence.replace("?", ".").replace("!", ".").split(".")]
        for seg in segments:
            if seg:
                final.append(seg)
    if not final:
        final = [text]
    return final


def split_sentences_into_chunks(sentences: List[str], max_sentences: int = SCRIPT_CHUNK_MAX_SENTENCES) -> List[List[str]]:
    """문장 리스트를 청크로 분할합니다."""
    if not sentences:
        return []
    if len(sentences) <= max_sentences:
        return [sentences]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        # 현재 청크가 비어있거나, 추가해도 최대 길이를 넘지 않으면 추가
        if current_length == 0 or (current_length + sentence_length <= max_sentences * 50):  # 대략적인 길이 추정
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # 현재 청크를 저장하고 새 청크 시작
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [sentence]
            current_length = sentence_length
        
        # 문장 수 기준으로도 체크
        if len(current_chunk) >= max_sentences:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [sentences]


def normalize_subtitle_text(text: str) -> str:
    return (text or "")  # .replace(" ", "\u00A0") 제거 - ASS BorderStyle=3으로 충분


def chunk_iterable(seq, size):
    it = iter(seq)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def enforce_stickman_prompt(prompt: str, fallback_context: str = "") -> str:
    base = (
        "a vibrant 2D cartoon, fully rendered illustration featuring a stickman with a white circular face, "
        "simple black outline, dot eyes, curved mouth, thin black limbs, expressive pose"
    )
    style_phrase = (
        "Consistent stick-figure illustration style, clean bold lines, solid colors, explainer video aesthetic, simplified background"
    )
    extra = "colorful detailed drawing, rich environment, dynamic lighting, no realistic human anatomy, no blank background"

    prompt = (prompt or "").strip()
    if "stickman" not in prompt.lower():
        prompt = f"stickman character {fallback_context}, {prompt}".strip(", ")
    prompt = f"{prompt}, {base}, {style_phrase}, {extra}"
    return prompt


def enforce_realistic_prompt(prompt: str, fallback_context: str = "") -> str:
    prompt = ensure_english_text((prompt or "").strip())
    if not prompt:
        prompt = fallback_context or "Detailed cinematic scene"
    
    # 실사화 모드에서는 스틱맨 관련 키워드 제거
    prompt_lower = prompt.lower()
    # 스틱맨 관련 키워드 제거
    if "stickman" in prompt_lower:
        # "stickman character", "stickman scene" 등의 패턴 제거
        prompt = re.sub(r'\bstickman\s+character\s*', '', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bstickman\s+scene\s*', '', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bstickman\s*', '', prompt, flags=re.IGNORECASE)
        prompt = prompt.strip()
        # 쉼표나 공백 정리
        prompt = re.sub(r'\s*,\s*,', ',', prompt)  # 연속된 쉼표 제거
        prompt = re.sub(r'^\s*,\s*', '', prompt)  # 시작 쉼표 제거
        prompt = re.sub(r'\s*,\s*$', '', prompt)  # 끝 쉼표 제거
        prompt = prompt.strip()
    
    # 애니메이션 관련 키워드 제거
    animation_keywords = [
        'cartoon', 'illustration', '2d', 'animated', 'drawing', 'sketch',
        'vibrant 2d', 'explainer video', 'cel-shading', 'bold lines'
    ]
    for keyword in animation_keywords:
        prompt = re.sub(rf'\b{re.escape(keyword)}\s*', '', prompt, flags=re.IGNORECASE)
        prompt = prompt.strip()
        prompt = re.sub(r'\s*,\s*,', ',', prompt)
        prompt = re.sub(r'^\s*,\s*', '', prompt)
        prompt = re.sub(r'\s*,\s*$', '', prompt)
        prompt = prompt.strip()
    
    if not prompt:
        prompt = fallback_context or "Detailed cinematic scene"
    
    if not prompt.lower().startswith(("a ", "the ")):
        prompt = prompt[0].upper() + prompt[1:]
    prompt = f"{REALISTIC_STYLE_WRAPPER}, {prompt}"
    return prompt


def enforce_realistic2_prompt(prompt: str, fallback_context: str = "") -> str:
    """
    실사화2 모드 전용 프롬프트 강화 함수
    - 스타일 중복 방지: OpenAI에서 이미 스타일 키워드를 포함한 경우 Wrapper를 추가하지 않음
    - 후처리 안전장치 역할: 스타일이 없을 때만 Wrapper 적용
    """
    prompt = ensure_english_text((prompt or "").strip())
    if not prompt:
        prompt = fallback_context or "Detailed cinematic scene"
    
    # 1. 불필요한 키워드 제거 (Stickman, Cartoon 등) - 기존 로직 유지
    prompt_lower = prompt.lower()
    
    # 스틱맨 관련 키워드 제거
    if "stickman" in prompt_lower:
        # "stickman character", "stickman scene" 등의 패턴 제거
        prompt = re.sub(r'\bstickman\s+character\s*', '', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bstickman\s+scene\s*', '', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bstickman\s*', '', prompt, flags=re.IGNORECASE)
        prompt = prompt.strip()
    
    # 애니메이션 관련 키워드 제거
    animation_keywords = [
        'cartoon', 'illustration', '2d', 'animated', 'drawing', 'sketch',
        'vibrant 2d', 'explainer video', 'cel-shading', 'bold lines'
    ]
    for keyword in animation_keywords:
        prompt = re.sub(rf'\b{re.escape(keyword)}\s*', '', prompt, flags=re.IGNORECASE)
        prompt = prompt.strip()
    
    # 불필요한 공백 및 쉼표 정리
    prompt = prompt.strip()
    prompt = re.sub(r'\s*,\s*,', ',', prompt)  # 연속된 쉼표 제거
    prompt = re.sub(r'^\s*,\s*', '', prompt)  # 시작 쉼표 제거
    prompt = re.sub(r'\s*,\s*$', '', prompt)  # 끝 쉼표 제거
    prompt = prompt.strip()
    
    if not prompt:
        prompt = fallback_context or "Detailed cinematic scene"
    
    # 2. [변경] REALISTIC_STYLE_WRAPPER 스마트 적용
    # 이미 스타일이 포함되어 있다면 중복 적용 방지
    style_keywords = ["photorealistic", "cinematic", "hyperrealistic", "photography"]
    if any(keyword in prompt_lower for keyword in style_keywords):
        # 스타일은 있지만 화질 키워드가 없으면 보강
        if "8k" not in prompt_lower and "ultra-detailed" not in prompt_lower:
            prompt = f"8k, ultra-detailed, {prompt}"
    else:
        # 스타일이 없으면 Wrapper 강제 적용
        if not prompt.lower().startswith(("a ", "the ")):
            prompt = prompt[0].upper() + prompt[1:]
        prompt = f"{REALISTIC_STYLE_WRAPPER}, {prompt}"
    
    return prompt


def enforce_animation2_prompt(prompt: str, fallback_context: str = "") -> str:
    """
    애니메이션 모드2: 고품질 익스플레인어 비디오 스타일의 스틱맨 캐릭터
    - 참조 이미지(image_6.png)와 동일한 기본 스틱맨 형태 유지
    - 스타일화된 카툰 의상/액세서리 착용 가능
    - 풍부하고 상세한 배경, 역동적인 조명
    """
    base_style = (
        "Flat 2D vector illustration, minimal vector art, white stickman figure with simple circular head, "
        "minimalist black dot eyes, thick bold black outlines, unshaded, flat solid colors, cel-shaded, "
        "simple line art, comic book inking style, completely flat, no shadows, no gradients, no depth."
    )
    
    character_details = (
        "The character is in an expressive, dynamic pose appropriate for the scene. "
        "The character can be wearing stylized cartoon clothing, costumes, or accessories that fit the theme of the environment. "
        "Any clothing must match the simple, bold-line cartoon aesthetic of the stickman, avoiding overly complex or realistic textures."
    )
    
    environment_lighting = (
        "The background is a rich, detailed, and colorful stylized cartoon environment "
        "(e.g., a bustling futuristic market, a pirate ship deck, a magical forest) filled with relevant objects. "
        "The entire frame is filled, with NO blank or simple background regions. "
        "The scene features dynamic, dramatic lighting with strong highlights and shadows that enhance the 2D cartoon feel."
    )
    
    constraints = (
        "Base Character Consistency: The underlying stickman form (head shape, eyes, body type) must match the reference style. "
        "No Realistic Anatomy: Do not add realistic human features, muscles, or photorealistic clothing textures. "
        "Stick to the simple cartoon style."
    )
    
    prompt = (prompt or "").strip()
    if not prompt:
        prompt = fallback_context or "a scene"
    
    # stickman 언급이 없으면 추가
    if "stickman" not in prompt.lower():
        prompt = f"stickman character {fallback_context}, {prompt}".strip(", ")
    
    full_prompt = (
        f"{prompt}, {base_style}, {character_details}, {environment_lighting}, {constraints}"
    )
    
    return full_prompt


def enforce_euro_graphic_novel_prompt(prompt: str, fallback_context: str = "") -> str:
    """
    스타일 강제: 유럽풍 그래픽 노블 (Bande Dessinée) 스타일
    - 특징: 지브리/애니메이션 느낌 배제. 성숙하고 진중한 화풍. 섬세한 잉크 펜 선과 깊은 명암.
    - 분위기: 고전적이고 고급스러운 일러스트레이션 느낌.
    """
    
    # 1. 베이스 스타일: 일본풍을 지우고 유럽 만화/일러스트레이션으로 변경
    base_style = (
        "European graphic novel style, bande dessinée aesthetic, "
        "highly detailed traditional illustration, hand-drawn ink lines with cross-hatching shadows, "
        "sophisticated and muted color palette, atmospheric, cinematic frame. "
        "Looks like printed art on high-quality paper."
    )
    
    # 2. 캐릭터 디테일: 귀여움 대신 '사실적이고 진중한' 묘사
    character_details = (
        "Character rendered with realistic proportions and expressive, mature features. "
        "Deep shadows on the face to create mystery (chiaroscuro effect). "
        "Clothing folds are detailed with heavy ink work. "
        "Serious and grounded character design, NOT cartoony or anime-like."
    )
    
    # 3. 배경 및 조명: 수채화 대신 '잉크 워시'와 진한 그림자
    environment_lighting = (
        "Background is intricate and heavily detailed with ink lines and watercolor washes. "
        "Dramatic, moody lighting with strong contrast between light and dark. "
        "Rich textures on furniture, walls, and objects. "
        "Feels historical and mysterious."
    )
    
    # 4. 제약 사항: 지브리, 아니메 관련 키워드 차단
    constraints = (
        "NO anime style, NO Studio Ghibli look, NO manga big eyes, NO cute aesthetic. "
        "Avoid purely sleek digital painting look. Must feel like traditional media."
    )
    
    prompt = (prompt or "").strip()
    if not prompt:
        prompt = fallback_context or "a mysterious scene"
        
    # 프롬프트 조합
    full_prompt = (
        f"{base_style}, {prompt}, {character_details}, {environment_lighting}, {constraints}"
    )
    
    return full_prompt


def enforce_custom_prompt(prompt: str, custom_style_prompt: str = "", fallback_context: str = "") -> str:
    """
    커스텀 모드: 사용자가 입력한 스타일 프롬프트를 기본 스타일 요소로 적용
    """
    prompt = (prompt or "").strip()
    if not prompt:
        prompt = fallback_context or "a scene"
    
    # 커스텀 스타일 프롬프트가 있으면 적용
    if custom_style_prompt and custom_style_prompt.strip():
        custom_style = custom_style_prompt.strip()
        # 사용자 프롬프트와 커스텀 스타일을 결합
        full_prompt = f"{prompt}, {custom_style}"
    else:
        # 커스텀 스타일이 없으면 기본 프롬프트만 사용
        full_prompt = prompt
    
    return full_prompt


def enforce_prompt_by_mode(prompt: str, fallback_context: str = "", mode: str = "animation", custom_style_prompt: str = "") -> str:
    mode = (mode or "animation").lower()
    if mode == "realistic2":
        return enforce_realistic2_prompt(prompt, fallback_context)
    elif mode == "realistic":
        return enforce_realistic_prompt(prompt, fallback_context)
    elif mode == "animation2":
        return enforce_animation2_prompt(prompt, fallback_context)
    elif mode == "animation3":
        return enforce_euro_graphic_novel_prompt(prompt, fallback_context)
    elif mode == "custom":
        return enforce_custom_prompt(prompt, custom_style_prompt, fallback_context)
    return enforce_stickman_prompt(prompt, fallback_context)


def build_fallback_prompt(sentence: str, mode: str) -> str:
    sentence = ensure_english_text(sentence or "")
    description = sentence if sentence else "unnamed scene"
    if mode == "realistic" or mode == "realistic2":
        base = f"A detailed cinematic depiction of {description}"
    else:
        base = f"Stickman scene illustrating {description}"
    return enforce_prompt_by_mode(base, fallback_context=description[:60], mode=mode)


def normalize_for_alignment(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace())


def alignment_segments_with_semantics(alignment: Dict[str, Any], segments: List[str]):
    characters = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []
    if not characters or not starts or not ends:
        return []

    raw_text = "".join(characters)
    cleaned_chars = []
    cleaned_index_map = []
    for idx, ch in enumerate(raw_text):
        normalized = normalize_for_alignment(ch)
        if not normalized:
            continue
        for norm_char in normalized:
            cleaned_chars.append(norm_char)
            cleaned_index_map.append(idx)

    cleaned_text = "".join(cleaned_chars)
    if not cleaned_text:
        return []

    cursor = 0
    phrases = []
    for seg in segments:
        original_segment = seg.strip()
        if not original_segment:
            continue
        normalized_segment = normalize_for_alignment(original_segment)
        if not normalized_segment:
            continue

        idx = cleaned_text.find(normalized_segment, cursor)
        if idx == -1:
            from difflib import SequenceMatcher

            matcher = SequenceMatcher(None, cleaned_text[cursor:], normalized_segment)
            match = matcher.find_longest_match(0, len(cleaned_text) - cursor, 0, len(normalized_segment))
            if match.size < max(2, len(normalized_segment) * 0.5):
                return []
            idx = cursor + match.a
            end_idx = idx + match.size - 1
        else:
            end_idx = idx + len(normalized_segment) - 1

        if end_idx >= len(cleaned_index_map):
            return []

        start_original_idx = cleaned_index_map[idx]
        end_original_idx = cleaned_index_map[end_idx]
        if start_original_idx >= len(starts) or end_original_idx >= len(ends):
            return []

        start_time = starts[start_original_idx]
        end_time = ends[end_original_idx]
        phrases.append((normalize_subtitle_text(original_segment), start_time, end_time))
        cursor = end_idx + 1
    return phrases


def alignment_to_phrases(alignment: Dict[str, Any], char_limit: int = 50, semantic_segments: Optional[List[str]] = None):
    characters = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []
    if not characters or not starts or not ends:
        return []

    if semantic_segments:
        semantic_phrases = alignment_segments_with_semantics(alignment, semantic_segments)
        if semantic_phrases:
            return semantic_phrases

    phrases = []
    buffer = []
    start_time = starts[0]
    last_end = starts[0]
    for idx, ch in enumerate(characters):
        buffer.append(ch)
        last_end = ends[idx]
        if len(buffer) >= char_limit or ch in ".!?":
            text = normalize_subtitle_text("".join(buffer).strip())
            if text:
                phrases.append((text, start_time, last_end))
            buffer = []
            if idx + 1 < len(starts):
                start_time = starts[idx + 1]

    if buffer:
        text = normalize_subtitle_text("".join(buffer).strip())
        if text:
            phrases.append((text, start_time, last_end))
    return phrases


# =============================================================================
# OpenAI 프롬프트 생성
# =============================================================================


def extract_scene_context(scene_text: str, mode: str = "animation") -> Dict[str, str]:
    """장면 텍스트에서 문맥 정보를 추출합니다."""
    if not gemini_available:
        return {}
    
    try:
        mode_instruction = ""
        if mode == "realistic":
            mode_instruction = "The images will be photorealistic/hyperrealistic. Focus on cinematic, photographic descriptions."
        else:
            mode_instruction = "The images will be in stickman cartoon, vibrant 2D illustration style."
        
        system_prompt = (
            "You are a professional script analyst and visual context extractor. "
            "Analyze the scene description and extract key contextual information for image generation. "
            f"{mode_instruction}\n\n"
            "Return JSON with these fields:\n"
            "- overall_style: The visual style/genre (e.g., '1940s Film Noir', 'cyberpunk', 'high school romance', 'hyperrealistic')\n"
            "- location: The setting/background (e.g., 'dark narrow alleyway at night', 'sunlit classroom', 'rainy street')\n"
            "- time: Time of day/period (e.g., 'deep night', 'sunset', 'rainy afternoon', 'dawn')\n"
            "- characters: Key characters in the scene with brief descriptions (e.g., 'Detective Park (40s, weary, trench coat), Suji (20s, red dress)')\n"
            "- mood: The emotional atmosphere (e.g., 'tense', 'somber', 'mysterious', 'peaceful', 'urgent')\n\n"
            "Be specific and descriptive. Write in English only. Return only valid JSON."
        )
        
        user_prompt = f"Extract context from this scene:\n\n{scene_text[:800]}"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # 새로운 라이브러리 방식
        from google.genai import types
        if genai_client is None:
            return {}
        
        target_model = 'gemini-2.5-flash'
        response = genai_client.models.generate_content(
            model=target_model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=400,
                response_mime_type="application/json",
            )
        )
        
        content = response.text.strip()
        parsed = json.loads(content)
        return {
            "overall_style": parsed.get("overall_style", ""),
            "location": parsed.get("location", ""),
            "time": parsed.get("time", ""),
            "characters": parsed.get("characters", ""),
            "mood": parsed.get("mood", ""),
        }
    except Exception as e:
        print(f"[경고] 장면 문맥 추출 실패: {e}")
    
    return {}


def split_script_into_scenes(script_text: str) -> List[Dict[str, Any]]:
    """대본을 장면 단위로 분할합니다."""
    scenes = []
    lines = script_text.split('\n')
    current_scene = None
    current_sentences = []
    scene_counter = 1
    
    for line in lines:
        line = line.strip()
        
        # 빈 줄은 장면 구분자로 간주 (이전 장면 저장)
        if not line:
            if current_scene is not None and current_sentences:
                scenes.append({
                    "scene_number": scene_counter,
                    "header": current_scene,
                    "sentences": current_sentences,
                })
                scene_counter += 1
                current_scene = None
                current_sentences = []
            continue
        
        # 장면 시작 패턴 감지 (S#1., S#10., Scene 1, 장면 1 등)
        scene_match = re.match(r'^(?:S#|Scene\s*|장면\s*)(\d+)', line, re.IGNORECASE)
        if scene_match:
            # 이전 장면 저장
            if current_scene is not None and current_sentences:
                scenes.append({
                    "scene_number": scene_counter,
                    "header": current_scene,
                    "sentences": current_sentences,
                })
                scene_counter += 1
            
            # 새 장면 시작
            current_scene = line
            current_sentences = []
        else:
            # 장면 헤더가 없으면 첫 번째 문장을 헤더로 사용
            if current_scene is None:
                current_scene = f"Scene {scene_counter}"
            
            # 문장 추가
            sentences = split_script_into_sentences(line)
            current_sentences.extend(sentences)
    
    # 마지막 장면 저장
    if current_scene is not None and current_sentences:
        scenes.append({
            "scene_number": scene_counter,
            "header": current_scene,
            "sentences": current_sentences,
        })
    
    # 장면이 없으면 전체를 하나의 장면으로 처리
    if not scenes:
        all_sentences = split_script_into_sentences(script_text)
        if all_sentences:
            scenes.append({
                "scene_number": 1,
                "header": "",
                "sentences": all_sentences,
            })
    
    return scenes


def call_openai_for_prompts(offset: int, sentences: List[str], mode: str = "animation", scene_context: Dict[str, str] = None, max_retries: int = 3, openai_api_key: Optional[str] = None) -> Dict[int, str]:
    script_blocks = "\n".join([f"Sentence {offset + idx + 1}: {sentence}" for idx, sentence in enumerate(sentences)])
    mode = (mode or "animation").lower()
    
    # 문맥 정보 구성
    context_info = ""
    if scene_context:
        context_parts = []
        if scene_context.get("overall_style"):
            context_parts.append(f"* Overall Style: {scene_context['overall_style']}")
        if scene_context.get("location"):
            context_parts.append(f"* Location: {scene_context['location']}")
        if scene_context.get("time"):
            context_parts.append(f"* Time: {scene_context['time']}")
        if scene_context.get("characters"):
            context_parts.append(f"* Characters in Scene: {scene_context['characters']}")
        if scene_context.get("mood"):
            context_parts.append(f"* Scene Mood: {scene_context['mood']}")
        
        if context_parts:
            context_info = "\n\n[CONTEXT - Scene Information]\n" + "\n".join(context_parts) + "\n"
    
    if mode == "realistic" or mode == "realistic2":
        style_instruction = (
            "For each sentence, convert it into a concrete, visually descriptive scene that could be captured in a photograph."
            " Mention subjects, actions, facial expressions, body language, environment, lighting, and camera framing."
            " Avoid abstract emotions or quoting dialogue. Write in English only."
            " Do not include stylistic keywords like 'hyperrealistic'; focus only on the scene description."
        )
    elif mode == "animation2":
        style_instruction = (
            "Create a vibrant, fully rendered 2D animated illustration in a high-quality explainer video aesthetic. "
            "The character's physical base is the exact white stickman figure with a simple circular head, "
            "minimalist black dot eyes, and clean white stick body with bold black outlines and cel-shading. "
            "The character can be wearing stylized cartoon clothing, costumes, or accessories that fit the theme. "
            "The background should be a rich, detailed, and colorful stylized cartoon environment filled with relevant objects. "
            "The entire frame must be filled with NO blank or simple background regions. "
            "Include dynamic, dramatic lighting with strong highlights and shadows. "
            "Maintain base character consistency - stick to the simple cartoon style, no realistic anatomy."
        )
    elif mode == "animation3":
        style_instruction = (
            "Create a European graphic novel style (bande dessinée aesthetic) with highly detailed traditional illustration. "
            "Use hand-drawn ink lines with cross-hatching shadows and a sophisticated, muted color palette. "
            "The character should be rendered with realistic proportions and expressive, mature features. "
            "Include deep shadows on the face to create mystery (chiaroscuro effect). "
            "The background should be intricate and heavily detailed with ink lines and watercolor washes. "
            "Use dramatic, moody lighting with strong contrast between light and dark. "
            "NO anime style, NO Studio Ghibli look, NO manga big eyes, NO cute aesthetic. "
            "Must feel like traditional media, not purely sleek digital painting."
        )
    else:
        style_instruction = "Stick to a stickman cartoon, vibrant 2D illustration style."

    for attempt in range(max_retries):
        try:
            # Gemini API 키 확인 (파라미터로 받은 키가 없으면 전역 변수 사용)
            api_key = openai_api_key or GEMINI_API_KEY  # 파라미터명은 호환성을 위해 유지
            if not api_key or not gemini_available:
                print(f"[경고] Gemini API 키가 없어 프롬프트 생성을 건너뜁니다.")
                return {}
            
            # 실사화 모드(realistic, realistic2)일 때만 새로운 시스템 프롬프트와 temperature 0.2 사용
            if mode == "realistic" or mode == "realistic2":
                system_content = """[YOUR ROLE]
당신은 '스크립트-투-이미지 프롬프트 전문 엔지니어'이자 '시각적 연속성 감독(Visual Continuity Director)'입니다.
당신의 임무는 사용자가 제공하는 대본(Script)을 받아 시각화하되, 단순한 장면 묘사를 넘어 **영화적 연속성(Continuity)**과 **캐릭터/장소의 일관성(Consistency)**이 완벽하게 유지되는 이미지 생성 프롬프트를 작성하는 것입니다.

[CRITICAL: VISUAL CONSISTENCY & CONTINUITY]
이 섹션은 인물과 장소의 모양이 바뀌는 환각(Hallucination)을 방지하기 위한 절대 규칙입니다.

1. **비주얼 앵커(Visual Anchor) 설정**: 작업을 시작하기 전, 대본 전체를 분석하여 주요 인물과 장소에 대한 고정된 시각적 정의(Visual Definition)를 수립하십시오.
   * 예: '민수' = "Korean man in his 30s, wearing a brown trench coat, glasses, short black hair."
   * 이 정의된 묘사는 해당 인물이 등장하는 *모든* 프롬프트에 토씨 하나 틀리지 않고 반복적으로 포함되어야 합니다. 단순히 "He"나 "The man"으로 퉁치지 마십시오.
2. **장소 고정 (Location Locking)**: 대본에서 장소 이동이 명시되지 않는 한, 배경 묘사(예: "messy bedroom with blue wallpapers")는 모든 프롬프트에서 동일하게 유지되어야 합니다.
3. **행동의 연속성**: 이전 문장에서 인물이 앉아 있었다면, 일어난다는 지문이 없는 한 다음 프롬프트에서도 앉아 있어야 합니다. 문맥의 흐름을 논리적으로 연결하십시오.

[CRITICAL: COMPLETENESS PROTOCOL]
* **전체 출력 의무**: 대본의 첫 문장부터 마지막 문장까지 하나도 빠뜨리지 마십시오.
* **요약 금지**: "이하 생략", "동일한 스타일" 등의 요약은 엄격히 금지됩니다.
* **분할 출력 대응**: 답변 길이가 토큰 제한에 도달할 것 같으면, 반드시 [한국어 번역]-[영어 이미지 프롬프트] 세트가 끝나는 지점에서 멈추고 **"[...계속하려면 '계속'이라고 말해주세요]"**라고 명시하십시오. 문장 중간에서 자르지 마십시오.

[CRITICAL: FORMATTING PROTOCOL - 태그 엄수]
* **[영어 이미지 프롬프트]** 태그는 변형(예: prompt, 프롬pt 등) 없이 정확히 대괄호를 포함하여 **[영어 이미지 프롬프트]**로만 출력해야 합니다.
* 결과물은 반드시 JSON 형식으로 출력해야 합니다. 각 문장에 대해 {"sentence_index": 번호, "prompt": "영어 프롬프트"} 형식의 객체를 배열로 반환하세요.

[WORKFLOW & RULES]
1. 입력 정제 및 분석 (Pre-computation): 타임스탬프 제거, 스타일 래퍼 선정, 비주얼 앵커 확정.
2. 순차적 출력: 각 문장별 [한국어 번역] 및 [영어 이미지 프롬프트] 생성.
   - 영어 프롬프트 구조: [Style Wrapper] + [Visual Anchor] + [Current Action] + [Visual Details]

[OUTPUT FORMAT]
반드시 JSON 형식으로 출력하세요:
{
  "items": [
    {"sentence_index": 1, "prompt": "영어 이미지 프롬프트"},
    {"sentence_index": 2, "prompt": "영어 이미지 프롬프트"},
    ...
  ]
}
IMPORTANT: Ensure all JSON strings are properly escaped and closed. Avoid unescaped quotes or newlines in string values."""
                
                temperature = 0.2  # 실사화 모드: 창의성보다 규칙 준수
            else:
                # 다른 모드: 기존 시스템 프롬프트 유지
                system_content = (
                    "You are an AI Art Director & Visual Prompt Engineer specializing in script visualization. "
                    "Your task is to convert Korean script sentences into detailed English image generation prompts."
                    f" {style_instruction}"
                )
                
                if context_info:
                    system_content += (
                        "\n\nIMPORTANT: Use the provided [CONTEXT] information to understand the scene's setting, "
                        "characters, mood, and style. Each sentence should be converted considering this context "
                        "to create consistent and contextually accurate visual prompts."
                    )
                
                system_content += (
                    "\n\nReturn JSON describing prompts for each sentence. "
                    "IMPORTANT: Ensure all JSON strings are properly escaped and closed. "
                    "Avoid unescaped quotes or newlines in string values."
                )
                temperature = 0.7  # 다른 모드: 기존 temperature 유지
            
            user_content = "[SCRIPT LINES - Sentences to Convert]\n" + script_blocks
            if context_info:
                user_content += context_info
            user_content += "\n\n[ENGLISH IMAGE PROMPT]\nReturn output strictly as JSON object with proper escaping."

            # Gemini API 호출 (새로운 라이브러리 방식)
            full_prompt = f"{system_content}\n\n{user_content}"
            
            # 사용할 모델 선택 (Flash가 속도와 안정성 면에서 유리함)
            # 3 Pro를 꼭 쓰고 싶으시면 아래 target_model을 'gemini-3-pro-preview'로 변경하세요.
            target_model = 'gemini-2.5-flash'
            # target_model = 'gemini-3-pro-preview'
            
            if genai_client is None:
                raise RuntimeError("Gemini API Client가 초기화되지 않았습니다.")
            
            from google.genai import types
            
            try:
                response = genai_client.models.generate_content(
                    model=target_model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=2000,
                        response_mime_type="application/json",
                    )
                )
                content = response.text.strip()
            except Exception as e:
                print(f"[Gemini API 오류] 모델: {target_model}, 에러: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini API request failed: {str(e)}")
            
            # JSON 파싱 시도 (여러 방법으로)
            parsed = None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                # 불완전한 JSON 복구 시도
                content_cleaned = content.strip()
                # 마지막 불완전한 문자열이나 객체 닫기 시도
                if content_cleaned.count('"') % 2 != 0:
                    # 닫히지 않은 따옴표 찾아서 닫기
                    last_quote_idx = content_cleaned.rfind('"')
                    if last_quote_idx > 0:
                        content_cleaned = content_cleaned[:last_quote_idx+1] + '"'
                # 닫히지 않은 중괄호 닫기
                open_braces = content_cleaned.count('{')
                close_braces = content_cleaned.count('}')
                if open_braces > close_braces:
                    content_cleaned += '}' * (open_braces - close_braces)
                # 닫히지 않은 대괄호 닫기
                open_brackets = content_cleaned.count('[')
                close_brackets = content_cleaned.count(']')
                if open_brackets > close_brackets:
                    content_cleaned += ']' * (open_brackets - close_brackets)
                
                try:
                    parsed = json.loads(content_cleaned)
                except:
                    # 여전히 실패하면 부분 파싱 시도
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise e
            
            if isinstance(parsed, dict):
                for key in ("items", "data", "prompts", "results", "sentences"):
                    if isinstance(parsed.get(key), list):
                        parsed = parsed[key]
                        break
                else:
                    parsed = []
            if not isinstance(parsed, list):
                parsed = []
            
            result = {}
            for item in parsed:
                if isinstance(item, dict):
                    idx = item.get("sentence_index") or item.get("index") or item.get("sentenceNumber")
                    prompt = item.get("prompt")
                    if isinstance(idx, int) and isinstance(prompt, str):
                        result[idx] = prompt
            
            # 결과가 비어있으면 재시도
            if not result and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            
            return result
            
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            raise e
    
    return {}


def fallback_semantic_segments(text: str, max_segments: int = 4) -> List[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    segments = []
    buffer = ""
    for ch in cleaned:
        buffer += ch
        if ch in ".?!\n" and buffer.strip():
            segments.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        segments.append(buffer.strip())
    if len(segments) > max_segments:
        chunk_size = max(1, len(segments) // max_segments)
        merged = []
        temp = []
        for seg in segments:
            temp.append(seg)
            if len(temp) >= chunk_size:
                merged.append(" ".join(temp))
                temp = []
        if temp:
            merged.append(" ".join(temp))
        segments = merged
    return segments


def generate_semantic_segments(text: str, max_segments: int = 4) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    fallback = fallback_semantic_segments(text, max_segments=max_segments)
    if not gemini_available:
        return fallback

    try:
        system_prompt = (
            "You split narration into subtitle-ready segments. "
            "Return the original text divided into short, sequential phrases. "
            "Do not paraphrase or remove words. Keep the language the same."
            " Respond strictly as JSON with an array named 'segments', each item containing 'text'."
        )
        full_prompt = f"{system_prompt}\n\n{text}"
        
        # 새로운 라이브러리 방식
        if genai_client is None:
            return fallback
        
        from google.genai import types
        target_model = 'gemini-2.5-flash'
        try:
            response = genai_client.models.generate_content(
                model=target_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=600,
                    response_mime_type="application/json",
                )
            )
            content = response.text.strip()
        except Exception as e:
            print(f"[경고] 자막 세그먼트 생성 실패: {e}")
            return fallback
        parsed = json.loads(content)
        raw_segments = parsed.get("segments") if isinstance(parsed, dict) else None
        if not isinstance(raw_segments, list):
            return fallback
        segments = []
        for item in raw_segments:
            if isinstance(item, dict):
                seg = item.get("text")
            else:
                seg = item
            if isinstance(seg, str) and seg.strip():
                segments.append(seg.strip())
        return segments or fallback
    except Exception as exc:
        print(f"[경고] 자막 세그먼트 생성 중 예외: {exc}")
        return fallback


def generate_visual_prompts(sentences: List[str], mode: str = "animation", progress_cb=None, original_script: str = None, custom_style_prompt: str = "") -> List[str]:
    mode = (mode or "animation").lower()
    total = len(sentences)
    if progress_cb:
        progress_cb("이미지용 프롬프트 생성을 시작합니다.")

    if not sentences:
        return []

    default_prompts = []
    for idx, sentence in enumerate(sentences):
        default_prompts.append(build_fallback_prompt(sentence, mode))

    if not gemini_available or not GEMINI_API_KEY:
        if progress_cb:
            progress_cb("Gemini API 키가 없어 기본 프롬프트를 사용합니다.")
        return default_prompts

    # 장면 단위로 분할 및 문맥 정보 추출
    scene_map = {}  # sentence_index -> scene_context
    if original_script:
        try:
            scenes = split_script_into_scenes(original_script)
            sentence_idx = 0
            for scene in scenes:
                scene_sentences = scene.get("sentences", [])
                if not scene_sentences:
                    continue
                
                # 장면 헤더와 문장들을 합쳐서 문맥 추출
                scene_text = scene.get("header", "") + "\n" + "\n".join(scene_sentences[:5])  # 처음 5개 문장만 사용
                scene_context = extract_scene_context(scene_text, mode)
                
                # 이 장면의 모든 문장에 문맥 정보 매핑
                for _ in scene_sentences:
                    if sentence_idx < total:
                        scene_map[sentence_idx] = scene_context
                        sentence_idx += 1
                
                if progress_cb and scene_context:
                    progress_cb(f"장면 '{scene.get('header', 'Scene')}' 문맥 정보 추출 완료")
        except Exception as e:
            print(f"[경고] 장면 분석 실패: {e}")

    prompts = [None] * total

    def fill_from_map(start_idx: int, chunk: List[str], prompts_map: Dict[int, str]):
        for local_idx, sentence in enumerate(chunk):
            global_idx = start_idx + local_idx
            raw_prompt = prompts_map.get(global_idx + 1)
            if not raw_prompt:
                prompts[global_idx] = default_prompts[global_idx]
                continue
            base_prompt = ensure_english_text(raw_prompt.strip())
            focus_sentence = ensure_english_text(sentence.strip())
            if (mode == "realistic" or mode == "realistic2") and focus_sentence:
                base_prompt = f"{base_prompt}. Focus on: {focus_sentence}"
            prompts[global_idx] = enforce_prompt_by_mode(
                base_prompt, fallback_context=f"based on '{focus_sentence[:50]}'", mode=mode, custom_style_prompt=custom_style_prompt
            )

    if total <= OPENAI_PROMPT_BATCH_SIZE:
        try:
            # 첫 번째 문장의 문맥 정보 사용 (청크 내 문장들이 같은 장면에 있을 가능성)
            scene_context = scene_map.get(0) if scene_map else None
            mapping = call_openai_for_prompts(0, sentences, mode, scene_context=scene_context)
            fill_from_map(0, sentences, mapping)
            if progress_cb:
                progress_cb("프롬프트 생성이 완료되었습니다.")
            return prompts
        except Exception as exc:
            print(f"[경고] OpenAI 프롬프트 단일 요청 실패: {exc}")
            if progress_cb:
                progress_cb("단일 요청 실패, 청크 방식으로 재시도합니다.")

    chunks = list(chunk_iterable(sentences, OPENAI_PROMPT_BATCH_SIZE))
    max_workers = min(len(chunks), OPENAI_PROMPT_WORKERS) or 1
    total_chunks = len(chunks)
    completed_chunks = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for offset, chunk in enumerate(chunks):
            start_idx = offset * OPENAI_PROMPT_BATCH_SIZE
            # 청크의 첫 번째 문장에 해당하는 문맥 정보 사용
            scene_context = scene_map.get(start_idx) if scene_map else None
            future = executor.submit(call_openai_for_prompts, start_idx, chunk, mode, scene_context=scene_context)
            future_map[future] = (start_idx, chunk, offset)

        for future in as_completed(future_map):
            start_idx, chunk, chunk_idx = future_map[future]
            first_idx = start_idx + 1
            last_idx = start_idx + len(chunk)
            try:
                mapping = future.result()
                completed_chunks += 1
                # 진행도 계산 (0-90%, 완료 시 100%)
                progress_pct = int((completed_chunks / total_chunks) * 90)
                if progress_cb:
                    progress_cb(f"문장 {first_idx}~{last_idx} 프롬프트 생성 완료 ({completed_chunks}/{total_chunks})")
            except Exception as exc:
                print(f"[경고] 문장 {first_idx}-{last_idx} 프롬프트 생성 오류: {exc}")
                mapping = {}
                completed_chunks += 1
                if progress_cb:
                    progress_cb(f"문장 {first_idx}~{last_idx} 프롬프트 생성 실패, 기본 프롬프트 사용")
            fill_from_map(start_idx, chunk, mapping)

    return [p or default_prompts[idx] for idx, p in enumerate(prompts)]


# =============================================================================
# 자산 생성 (TTS, 이미지)
# =============================================================================


def generate_tts_with_alignment(voice_id: str, text: str, audio_filename: str, elevenlabs_api_key: Optional[str] = None):
    print("=" * 80)
    print("=== [DEBUG] TTS 생성 요청 받음 ===")
    print(f"=== [DEBUG] voice_id: {voice_id} ===")
    print(f"=== [DEBUG] audio_filename: {audio_filename} ===")
    print(f"=== [DEBUG] text 길이: {len(text) if text else 0} ===")
    print("=" * 80)
    try:
        text = (text or "").strip()
        if not text:
            print("[TTS] 입력 문장이 비어 있어 TTS를 건너뜁니다.")
            return None
        if not is_voice_allowed(voice_id):
            voice_id = get_default_voice_id()
        # 사용자가 입력한 API 키를 우선 사용, 없으면 기본값 사용
        api_key = elevenlabs_api_key or ELEVENLABS_API_KEY
        print(f"=== [DEBUG] API 키 확인: {api_key[:10] + '...' if api_key and len(api_key) > 10 else 'None'} ===")
        if not api_key:
            print("[TTS] ElevenLabs API 키가 설정되지 않았습니다.")
            return None
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
        headers = {"xi-api-key": api_key, "Accept": "application/json"}
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",  # 더 빠르고 고품질 모델로 업그레이드
            "voice_settings": {
                "stability": 0.92,  # 높일수록 피치 변동을 줄이고 일관성을 높임
                "similarity_boost": 0.7,  # 너무 높으면 표현이 과도해질 수 있어 완화
                "style": 0.0,
                "use_speaker_boost": False,  # 피치 상승을 막기 위해 부스트 비활성화
            },
            "output_format": "mp3_44100_256",  # 비트레이트 증가 (128 -> 256)로 음질 개선
            "optimize_streaming_latency": 4,
        }
        last_error = None
        for attempt in range(1, TTS_MAX_RETRIES + 1):
            wait_seconds = TTS_RETRY_BASE_DELAY * attempt
            try:
                with _tts_request_semaphore:
                    resp = requests.post(url, headers=headers, json=payload, timeout=120)
            except Exception as exc:
                last_error = str(exc)
                print(f"[TTS] API 호출 실패 (시도 {attempt}/{TTS_MAX_RETRIES}): {exc}")
                resp = None
            if resp and resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as json_exc:
                    last_error = f"JSON 파싱 실패: {json_exc}"
                    print(f"[TTS] 응답 파싱 실패: {json_exc}")
                    resp = None
                else:
                    audio_b64 = data.get("audio") or data.get("audio_base64")
                    alignment = data.get("alignment")
                    if not audio_b64:
                        last_error = "오디오 데이터가 비어 있습니다."
                        print(f"[TTS] 경고: 오디오 데이터가 비어 있음 (시도 {attempt})")
                    else:
                        try:
                            audio_bytes = base64.b64decode(audio_b64)
                            with open(audio_filename, "wb") as f:
                                f.write(audio_bytes)
                            if os.path.getsize(audio_filename) == 0:
                                last_error = "생성된 오디오 파일 크기가 0입니다."
                                print(f"[TTS] 경고: 생성된 오디오 파일이 비어 있음 (시도 {attempt})")
                            else:
                                return alignment
                        except Exception as file_exc:
                            last_error = f"오디오 파일 저장 실패: {file_exc}"
                            print(f"[TTS] 오디오 저장 실패: {file_exc}")
            else:
                if resp is not None:
                    error_detail = ""
                    try:
                        error_json = resp.json()
                        error_detail = error_json.get("detail") or error_json.get("error", resp.text)
                    except Exception:
                        error_detail = resp.text
                    print(f"[TTS] 실패 (시도 {attempt}/{TTS_MAX_RETRIES}): {resp.status_code} {error_detail}")
                    if resp.status_code == 429:
                        retry_after = 0
                        try:
                            retry_after = int(error_json.get("retry_after", 0) if 'error_json' in locals() else 0)
                        except Exception:
                            retry_after = 0
                        wait_seconds = max(wait_seconds, retry_after + 1)
                    elif resp.status_code >= 500:
                        wait_seconds = max(wait_seconds, TTS_RETRY_BASE_DELAY * attempt)
                    last_error = error_detail or str(resp.status_code)
            if attempt < TTS_MAX_RETRIES:
                print(f"[TTS] {wait_seconds}초 후 재시도합니다...")
                time.sleep(wait_seconds)
        print(f"[TTS] 모든 재시도 실패: {last_error}")
        return None
    except Exception as tts_exc:
        print("=" * 80)
        print("=== [DEBUG] 오류: TTS 생성 예외 발생 ===")
        print(f"=== [DEBUG] 예외 내용: {tts_exc} ===")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return None


def save_image_bytes_as_png(content: bytes, filename: str, target_size=(1280, 768)):
    try:
        with Image.open(BytesIO(content)) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
            if target_size and (img.width != target_size[0] or img.height != target_size[1]):
                resampling = getattr(Image, "Resampling", Image)
                lanczos = getattr(resampling, "LANCZOS", Image.LANCZOS)
                img = ImageOps.fit(img, target_size, method=lanczos)
            img.save(filename, format="PNG")
    except Exception as exc:
        print(f"[IMG] 경고: PIL 저장 실패, 원본 저장 ({exc})")
        with open(filename, "wb") as f:
            f.write(content)


def generate_image(prompt_text: str, filename: str, mode: str = "animation", replicate_api_key: Optional[str] = None, custom_style_prompt: str = "", use_banana_pro_model: bool = False, use_banana_normal_model: bool = False) -> bool:
    """이미지 생성 함수 - Windows 환경 디버깅 강화"""
    print("=" * 80)
    print("=== [DEBUG] 이미지 생성 요청 받음 ===")
    print(f"=== [DEBUG] filename: {filename}")
    print(f"=== [DEBUG] mode: {mode}")
    print(f"=== [DEBUG] prompt_text 길이: {len(prompt_text) if prompt_text else 0}")
    print(f"=== [DEBUG] replicate_api_key 전달 여부: {bool(replicate_api_key)}")
    print("=" * 80)
    
    log_debug(f"[generate_image] 함수 시작 - filename: {filename}, mode: {mode}")
    log_debug(f"[generate_image] prompt_text 길이: {len(prompt_text) if prompt_text else 0}")
    log_debug(f"[generate_image] replicate_api_key 전달 여부: {bool(replicate_api_key)}")
    
    try:
        mode = (mode or "animation").lower()
        fallback_context = "scene description"
        if prompt_text:
            base_prompt = enforce_prompt_by_mode(prompt_text, fallback_context=fallback_context, mode=mode, custom_style_prompt=custom_style_prompt)
        else:
            if mode == "realistic" or mode == "realistic2":
                default_prompt = "A mysterious scene involving a historic landmark attracting curiosity"
            elif mode == "animation2":
                default_prompt = "stickman character in a vibrant detailed scene"
            elif mode == "animation3":
                default_prompt = "a dramatic scene with expressive character"
            else:
                default_prompt = "stickman presenting data in a colorful studio"
            base_prompt = enforce_prompt_by_mode(default_prompt, fallback_context="default context", mode=mode, custom_style_prompt=custom_style_prompt)

        if mode == "realistic2":
            # google/imagen-4-fast는 negative_prompt를 지원하지 않음
            negative_prompt = None
        elif mode == "realistic":
            negative_prompt = REALISTIC_NEGATIVE_PROMPT
        elif mode == "animation2":
            # 애니메이션2 모드 전용 네거티브 프롬프트
            negative_prompt = (
                "3d, realistic, photorealistic, cgi, render, blender, volumetric lighting, gradient, shadow, depth, "
                "fuzzy, blurred, texture, detail, oil painting, water color"
            )
        elif mode == "animation3":
            # 애니메이션3 모드 전용 네거티브 프롬프트 (팝아트/코믹스 스타일)
            negative_prompt = (
                "photorealistic, 3d render, realistic photo, oil painting, watercolor, blurry, "
                "soft edges, gradient shadows, realistic human anatomy, detailed skin texture"
            )
        else:
            # animation 모드는 기존 네거티브 프롬프트 사용
            negative_prompt = (
                "realistic human, detailed human skin, photograph, 3d render, blank white background, line-art only, text, watermark"
            )

        # 사용자가 입력한 API 키를 우선 사용, 없으면 기본값 사용
        api_token = replicate_api_key or REPLICATE_API_TOKEN
        replicate_api_available_local = bool(api_token)
        
        # API 키 사용 여부 로그 (키의 일부만 표시)
        print("=== [DEBUG] API 키 확인 ===")
        if api_token:
            key_preview = api_token[:10] + "..." + api_token[-4:] if len(api_token) > 14 else "***"
            print(f"=== [DEBUG] API 키 확인: {key_preview} ===")
            print(f"=== [DEBUG] API 키 길이: {len(api_token)} ===")
            log_debug(f"[generate_image] 사용 중인 Replicate API 키: {key_preview}")
            log_debug(f"[generate_image] API 키 길이: {len(api_token)}")
        else:
            print("=== [DEBUG] 경고: Replicate API 키가 설정되지 않았습니다! ===")
            log_error(f"[경고] Replicate API 키가 설정되지 않았습니다!")
            log_error(f"[경고] replicate_api_key 파라미터: {replicate_api_key}")
            log_error(f"[경고] REPLICATE_API_TOKEN 전역 변수: {REPLICATE_API_TOKEN[:10] if REPLICATE_API_TOKEN else 'None'}...")
    
        if replicate_api_available_local:
            print("=== [DEBUG] Replicate 호출 시작 ===")
            try:
                log_debug(f"[generate_image] Replicate API 사용 시작 - 모드: {mode}, 파일: {os.path.basename(filename)}")
                log_debug(f"[generate_image] 파일 전체 경로: {os.path.abspath(filename)}")
                log_debug(f"[generate_image] 파일 디렉토리 존재 여부: {os.path.exists(os.path.dirname(filename))}")
                log_debug(f"[generate_image] 파일 디렉토리 쓰기 가능 여부: {os.access(os.path.dirname(filename), os.W_OK) if os.path.exists(os.path.dirname(filename)) else False}")
                print(f"[generate_image] Replicate API 사용 - 모드: {mode}, 파일: {os.path.basename(filename)}")
                print(f"=== [DEBUG] 파일 전체 경로: {os.path.abspath(filename)} ===")
                print(f"=== [DEBUG] 파일 디렉토리 존재 여부: {os.path.exists(os.path.dirname(filename))} ===")
                print(f"=== [DEBUG] 파일 디렉토리 쓰기 가능 여부: {os.access(os.path.dirname(filename), os.W_OK) if os.path.exists(os.path.dirname(filename)) else False} ===")
                headers = {
                    "Authorization": f"Token {api_token}",
                    "Content-Type": "application/json",
                }
                # negative_prompt가 None이 아닌 경우에만 포함
                replicate_input = {
                    "prompt": base_prompt,
                    "aspect_ratio": "16:9",
                }
                if negative_prompt is not None:
                    replicate_input["negative_prompt"] = negative_prompt
                # 나노바나나 PRO 모델 체크박스가 체크된 경우 google/nano-banana-pro 사용
                if use_banana_pro_model:
                    # google/nano-banana-pro 모델 사용
                    # 프롬프트는 그대로 사용 (각 모델의 프롬프트 유지)
                    replicate_input.update({
                        "prompt": base_prompt,  # 프롬프트 유지
                        "negative_prompt": negative_prompt,
                        "aspect_ratio": "16:9",
                    })
                    model_owner = "google"
                    model_name = "nano-banana-pro"
                    request_url = f"https://api.replicate.com/v1/models/{model_owner}/{model_name}/predictions"
                    body = {"input": replicate_input}
                    print(f"[나노바나나 PRO 모델] google/nano-banana-pro 사용 - 프롬프트: {base_prompt[:100]}...")
                # 나노바나나 노말 모델 체크박스가 체크된 경우 google/nano-banana 사용
                elif use_banana_normal_model:
                    # google/nano-banana 모델 사용
                    # 프롬프트는 그대로 사용 (각 모델의 프롬프트 유지)
                    replicate_input.update({
                        "prompt": base_prompt,  # 프롬프트 유지
                        "negative_prompt": negative_prompt,
                        "aspect_ratio": "16:9",
                    })
                    model_owner = "google"
                    model_name = "nano-banana"
                    request_url = f"https://api.replicate.com/v1/models/{model_owner}/{model_name}/predictions"
                    body = {"input": replicate_input}
                    print(f"[나노바나나 노말 모델] google/nano-banana 사용 - 프롬프트: {base_prompt[:100]}...")
                elif mode == "realistic2":
                    # google/imagen-4-fast 모델 사용 (negative_prompt 미지원)
                    replicate_input = {
                        "prompt": base_prompt,
                        "aspect_ratio": "16:9",
                        "safety_filter_level": "block_only_high",
                    }
                    model_owner = "google"
                    model_name = "imagen-4-fast"
                    request_url = f"https://api.replicate.com/v1/models/{model_owner}/{model_name}/predictions"
                    body = {"input": replicate_input}
                    print(f"[실사화2 모델] google/imagen-4-fast 사용 - 프롬프트: {base_prompt[:100]}...")
                elif mode == "realistic":
                    # flux-schnell 모델 파라미터 (docs/cog-flux-main 참고)
                    # - num_inference_steps: 최대 4, 기본값 4 (SchnellPredictor 참고)
                    # - guidance_scale: 지원하지 않음 (제거)
                    # - scheduler: 지원하지 않음 (제거)
                    replicate_input.update(
                        {
                            "num_inference_steps": 4,  # flux-schnell은 최대 4까지만 지원
                            "output_format": "png",
                        }
                    )
                    model_owner = "black-forest-labs"
                    model_name = "flux-schnell"
                    # flux-schnell은 최신 버전을 사용하므로 version_id 없이 직접 predictions 엔드포인트 사용
                    request_url = f"https://api.replicate.com/v1/models/{model_owner}/{model_name}/predictions"
                    body = {"input": replicate_input}
                else:
                    # prunaai/hidream-l1-fast 모델 사용 (HiDream-I1-Fast: 16 inference steps)
                    replicate_input.update({"num_inference_steps": 16, "image_format": "png"})
                    # guidance_scale은 hidream 모델에서 지원하지 않을 수 있으므로 제거
                    model_owner = "prunaai"
                    model_name = "hidream-l1-fast"
                    request_url = f"https://api.replicate.com/v1/models/{model_owner}/{model_name}/predictions"
                    body = {"input": replicate_input}

                print("=== [DEBUG] Replicate API 요청 전송 중 ===")
                print(f"=== [DEBUG] 프롬프트: {base_prompt[:200]}... ===")
                print(f"=== [DEBUG] Negative 프롬프트: {negative_prompt[:200] if negative_prompt else 'None'}... ===")
                print(f"=== [DEBUG] 요청 URL: {request_url} ===")
                print(f"=== [DEBUG] 요청 본문: {json.dumps(body, indent=2, ensure_ascii=False)} ===")
                log_debug(f"[generate_image] Replicate API 요청 전송 중...")
                log_debug(f"[generate_image] 프롬프트: {base_prompt[:200]}...")
                log_debug(f"[generate_image] Negative 프롬프트: {negative_prompt[:200] if negative_prompt else 'None'}...")
                log_debug(f"[generate_image] 요청 URL: {request_url}")
                log_debug(f"[generate_image] 요청 본문: {json.dumps(body, indent=2, ensure_ascii=False)}")
                
                # Rate limiting: 분당 600개 요청 제한 준수 (최소 0.1초 간격)
                global _last_replicate_request_time
                with _replicate_request_lock:
                    current_time = time.time()
                    time_since_last = current_time - _last_replicate_request_time
                    if time_since_last < REPLICATE_MIN_REQUEST_INTERVAL:
                        wait_time = REPLICATE_MIN_REQUEST_INTERVAL - time_since_last
                        print(f"[Rate Limit] 요청 간격 조절: {wait_time:.2f}초 대기 중...")
                        time.sleep(wait_time)
                    _last_replicate_request_time = time.time()
                
                # 429 에러 재시도를 위한 루프
                max_retries = 3
                create_res = None
                for retry_attempt in range(max_retries):
                    try:
                        create_res = requests.post(request_url, headers=headers, json=body, timeout=30)
                        print(f"[generate_image] 응답 상태 코드: {create_res.status_code}")
                        print(f"[generate_image] 응답 본문 (처음 500자): {create_res.text[:500]}")
                        
                        # 429 에러 (Rate Limit) 처리
                        if create_res.status_code == 429:
                            error_data = create_res.json() if create_res.text else {}
                            error_detail = error_data.get("detail", "Request was throttled.")
                            # retry_after 값이 있으면 사용, 없으면 기본값 사용
                            retry_after = error_data.get("retry_after")
                            if retry_after:
                                wait_time = int(retry_after) + 1  # retry_after에 1초 추가하여 안전하게 대기
                            else:
                                wait_time = REPLICATE_RATE_LIMIT_RETRY_DELAY
                            
                            if retry_attempt < max_retries - 1:
                                print(f"[Rate Limit] 429 에러 발생: {error_detail}")
                                print(f"[Rate Limit] {wait_time}초 후 재시도 중... (시도 {retry_attempt + 1}/{max_retries})")
                                print(f"[Rate Limit] 참고: 크레딧이 $5 미만이면 분당 6개 요청으로 제한됩니다.")
                                print(f"[Rate Limit] https://replicate.com/account/billing 에서 크레딧을 충전하세요.")
                                time.sleep(wait_time)
                                continue  # 재시도
                            else:
                                print(f"[Rate Limit] 429 에러: 최대 재시도 횟수 초과")
                                print(f"[Rate Limit] 해결 방법:")
                                print(f"[Rate Limit] 1. https://replicate.com/account/billing 에서 크레딧을 충전하세요 ($5 이상 권장)")
                                print(f"[Rate Limit] 2. 또는 잠시 후 다시 시도하세요 (Rate Limit이 리셋될 때까지 대기)")
                                raise Exception(f"Replicate API Rate Limit 초과: {error_detail}")
                        
                        # 성공(200, 201)했거나, 429(Rate Limit)가 아닌 다른 에러라면 루프를 종료
                        if create_res.status_code != 429:
                            break
                            
                    except Exception as req_exc:
                        if retry_attempt < max_retries - 1 and "429" in str(req_exc):
                            wait_time = REPLICATE_RATE_LIMIT_RETRY_DELAY
                            print(f"[Rate Limit] 예외 발생, {wait_time}초 후 재시도: {req_exc}")
                            time.sleep(wait_time)
                            continue
                        print("=" * 80)
                        print("=== [DEBUG] 오류: Replicate API 요청 실패 ===")
                        print(f"=== [DEBUG] 예외 내용: {req_exc} ===")
                        import traceback
                        traceback.print_exc()
                        print("=" * 80)
                        raise  # 예외를 다시 발생시켜 fallback으로 넘어가도록 함
                
                if create_res is None or create_res.status_code not in (200, 201):
                    print(f"[IMG] (Replicate) 생성 실패: {create_res.status_code if create_res else 'None'} {create_res.text if create_res else 'No response'}")
                    # 402 에러 (월간 사용 한도 도달) 처리
                    if create_res and create_res.status_code == 402:
                        error_data = create_res.json() if create_res.text else {}
                        error_detail = error_data.get("detail", "월간 사용 한도에 도달했습니다.")
                        print(f"[경고] Replicate API 월간 사용 한도 도달: {error_detail}")
                        print(f"[경고] https://replicate.com/account/billing#limits 에서 한도를 확인하거나 증가시켜주세요.")
                        print(f"[경고] 한도를 증가시킨 경우 몇 분 후 다시 시도해주세요.")
                        raise Exception(f"Replicate API 월간 사용 한도 도달: {error_detail}")
                    # 500 에러나 다른 서버 에러인 경우 즉시 fallback으로
                    if create_res and create_res.status_code >= 500:
                        print(f"[IMG] (Replicate) 서버 에러 ({create_res.status_code}), 즉시 fallback으로 전환")
                        raise Exception(f"Replicate API 서버 에러: {create_res.status_code}")
                else:
                    try:
                        prediction = create_res.json()
                        print(f"[generate_image] 예측 응답: {json.dumps(prediction, indent=2, ensure_ascii=False)[:500]}")
                        pred_id = prediction.get("id")
                        get_url = prediction.get("urls", {}).get("get")
                        if not get_url and pred_id:
                            get_url = f"https://api.replicate.com/v1/predictions/{pred_id}"
                        print(f"[generate_image] 예측 ID: {pred_id}, get_url: {get_url}")
                        print(f"[generate_image] 상태 확인 시작 (최대 180초 대기)")
                        final = None
                        poll_count = 0
                        max_polls = 180
                        for poll_count in range(max_polls):
                            if not get_url:
                                print(f"[generate_image] get_url이 없어서 중단")
                                break
                            try:
                                res = requests.get(get_url, headers=headers, timeout=10)
                                if res.status_code != 200:
                                    print(f"[IMG] (Replicate) 상태 조회 실패: {res.status_code} {res.text[:500]}")
                                    break
                                data = res.json()
                                status = data.get("status")
                                # 5초마다 로그 출력 (더 자주)
                                if poll_count % 5 == 0:
                                    print(f"[generate_image] 상태 확인 중... ({poll_count}초 경과, 상태: {status})")
                                if status in ("succeeded", "failed", "canceled"):
                                    final = data
                                    print(f"[generate_image] 최종 상태: {status} (총 {poll_count}초 소요)")
                                    if status == "failed":
                                        error_msg = data.get("error", "알 수 없는 오류")
                                        print(f"[generate_image] 실패 원인: {error_msg}")
                                    break
                                time.sleep(1)
                            except requests.exceptions.Timeout:
                                print(f"[경고] 상태 조회 타임아웃 ({poll_count}초 경과), 계속 시도...")
                                time.sleep(1)
                                continue
                            except Exception as poll_exc:
                                print(f"[오류] 상태 조회 중 예외 발생: {poll_exc}")
                                import traceback
                                traceback.print_exc()
                                # 네트워크 오류는 재시도
                                if poll_count < max_polls - 1:
                                    time.sleep(2)
                                    continue
                                break
                        if poll_count >= max_polls - 1:
                            print(f"[generate_image] 타임아웃: {max_polls}초 동안 완료되지 않음")
                    except Exception as json_exc:
                        print(f"[오류] 예측 응답 파싱 실패: {json_exc}")
                        print(f"[오류] 응답 본문: {create_res.text[:1000]}")
                        import traceback
                        traceback.print_exc()
                        final = None
                if final and final.get("status") == "succeeded":
                    outputs = final.get("output")
                    image_url = None
                    image_b64 = None
                    if isinstance(outputs, str):
                        image_url = outputs
                    elif isinstance(outputs, list) and outputs:
                        first = outputs[0]
                        if isinstance(first, str):
                            image_url = first
                        elif isinstance(first, dict):
                            image_url = first.get("url") or first.get("image")
                            image_b64 = first.get("image_base64") or first.get("b64_json")
                    elif isinstance(outputs, dict):
                        image_url = outputs.get("url") or outputs.get("image")
                        image_b64 = outputs.get("image_base64")
                    if image_b64:
                        print(f"[generate_image] base64 이미지 저장 중...")
                        try:
                            image_bytes = base64.b64decode(image_b64)
                            print(f"[generate_image] base64 디코딩 완료, 크기: {len(image_bytes)} bytes")
                            if len(image_bytes) < 100:
                                print(f"[경고] 이미지 데이터가 너무 작습니다: {len(image_bytes)} bytes")
                            save_image_bytes_as_png(image_bytes, filename)
                            # 저장된 파일 확인
                            if os.path.exists(filename):
                                file_size = os.path.getsize(filename)
                                print(f"[generate_image] 이미지 저장 완료: {filename} (크기: {file_size} bytes)")
                                # 이미지가 검은색인지 확인 (첫 몇 픽셀 확인)
                                try:
                                    with Image.open(filename) as test_img:
                                        # 중앙 픽셀 색상 확인
                                        center_x, center_y = test_img.width // 2, test_img.height // 2
                                        pixel = test_img.getpixel((center_x, center_y))
                                        print(f"[generate_image] 중앙 픽셀 색상: {pixel}")
                                        if pixel == (0, 0, 0) or (isinstance(pixel, tuple) and len(pixel) == 4 and pixel[:3] == (0, 0, 0)):
                                            print(f"[경고] 이미지가 검은색일 수 있습니다!")
                                except Exception as img_check_exc:
                                    print(f"[경고] 이미지 확인 실패: {img_check_exc}")
                                return True
                            else:
                                print(f"[오류] 파일이 저장되지 않았습니다: {filename}")
                        except Exception as b64_exc:
                            print(f"[오류] base64 디코딩 실패: {b64_exc}")
                            import traceback
                            traceback.print_exc()
                    if image_url:
                        print(f"[generate_image] 이미지 URL에서 다운로드 중: {image_url}")
                        try:
                            resp = requests.get(image_url, timeout=60)
                            resp.raise_for_status()
                            print(f"[generate_image] 이미지 다운로드 완료, 크기: {len(resp.content)} bytes")
                            if len(resp.content) < 100:
                                print(f"[경고] 다운로드된 이미지 데이터가 너무 작습니다: {len(resp.content)} bytes")
                            save_image_bytes_as_png(resp.content, filename)
                            # 저장된 파일 확인
                            if os.path.exists(filename):
                                file_size = os.path.getsize(filename)
                                print(f"[generate_image] 이미지 다운로드 및 저장 완료: {filename} (크기: {file_size} bytes)")
                                # 이미지가 검은색인지 확인
                                try:
                                    with Image.open(filename) as test_img:
                                        center_x, center_y = test_img.width // 2, test_img.height // 2
                                        pixel = test_img.getpixel((center_x, center_y))
                                        print(f"[generate_image] 중앙 픽셀 색상: {pixel}")
                                        if pixel == (0, 0, 0) or (isinstance(pixel, tuple) and len(pixel) == 4 and pixel[:3] == (0, 0, 0)):
                                            print(f"[경고] 이미지가 검은색일 수 있습니다!")
                                except Exception as img_check_exc:
                                    print(f"[경고] 이미지 확인 실패: {img_check_exc}")
                                return True
                            else:
                                print(f"[오류] 파일이 저장되지 않았습니다: {filename}")
                        except Exception as download_exc:
                            print(f"[오류] 이미지 다운로드 실패: {download_exc}")
                            import traceback
                            traceback.print_exc()
                    print("[IMG] (Replicate) 출력이 비어 있습니다.")
                    print(f"[디버그] outputs 타입: {type(outputs)}, 값: {outputs}")
                elif final:
                    print(f"[IMG] (Replicate) 예측 실패: {final.get('status')}, 에러: {final.get('error')}")
            except Exception as exc:
                print("=" * 80)
                print("=== [DEBUG] 오류: Replicate API 예외 발생 ===")
                print(f"=== [DEBUG] 예외 내용: {exc} ===")
                import traceback
                traceback.print_exc()
                print("=" * 80)
                log_error(f"[IMG] (Replicate) 예외 발생: {exc}", exc_info=exc)

        # Stability API는 사용하지 않음 (Replicate만 사용)
        # if mode != "realistic" and stability_api_available:
        #     ... (Stability API 코드 제거됨)

        # fallback: create black image
        print("=" * 80)
        print("=== [DEBUG] 경고: 이미지 생성 실패, 검은색 이미지 생성 시도 ===")
        print(f"=== [DEBUG] filename: {filename} ===")
        print("=" * 80)
        log_error(f"[경고] 이미지 생성 실패, 검은색 이미지 생성 시도: {filename}")
        try:
            print(f"[경고] 이미지 생성 실패, 검은색 이미지 생성: {filename}")
            log_debug(f"[fallback] 검은색 이미지 생성 시작 - 경로: {os.path.abspath(filename)}")
            log_debug(f"[fallback] 디렉토리 존재 여부: {os.path.exists(os.path.dirname(filename))}")
            black_img = Image.new("RGB", (1920, 1080), color="black")
            black_img.save(filename, format="PNG")
            log_debug(f"[fallback] 검은색 이미지 저장 완료")
            
            # 생성된 이미지 파일 검증
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"[검은색 이미지] 파일 생성됨: {filename} (크기: {file_size} bytes)")
                if file_size < 100:
                    print(f"[경고] 검은색 이미지 파일이 너무 작습니다. 다시 생성 시도...")
                    black_img.save(filename, format="PNG")
                    file_size = os.path.getsize(filename)
                    print(f"[검은색 이미지] 재생성 완료: {filename} (크기: {file_size} bytes)")
                
                # PIL로 이미지 파일 유효성 검증
                try:
                    with Image.open(filename) as test_img:
                        test_img.verify()  # 이미지 파일 무결성 검증
                    print(f"[검은색 이미지] 이미지 파일 유효성 검증 완료")
                except Exception as verify_exc:
                    print(f"[경고] 이미지 파일 유효성 검증 실패: {verify_exc}")
                    # 다시 생성
                    black_img = Image.new("RGB", (1920, 1080), color="black")
                    black_img.save(filename, format="PNG")
                    print(f"[검은색 이미지] 재생성 완료 (검증 실패 후)")
            else:
                print(f"[오류] 검은색 이미지 파일이 생성되지 않았습니다: {filename}")
        except Exception as fallback_exc:
            print("=" * 80)
            print("=== [DEBUG] 오류: 검은색 이미지 생성 실패 ===")
            print(f"=== [DEBUG] 예외 내용: {fallback_exc} ===")
            import traceback
            traceback.print_exc()
            print("=" * 80)
        
        return False
    except Exception as outer_exc:
        print("=" * 80)
        print("=== [DEBUG] 오류: generate_image 함수 전체 예외 발생 ===")
        print(f"=== [DEBUG] 예외 내용: {outer_exc} ===")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        log_error(f"[generate_image] 함수 전체 예외 발생: {outer_exc}", exc_info=outer_exc)
        return False


def generate_assets(
    sentences: List[str],
    voice_id: str,
    assets_folder: str,
    progress_cb=None,
    prompts_override: Optional[List[str]] = None,
    existing_images: Optional[Dict[int, str]] = None,
    mode: str = "animation",
    original_script: Optional[str] = None,
    scene_offset: int = 0,
    replicate_api_key: Optional[str] = None,
    elevenlabs_api_key: Optional[str] = None,
    custom_style_prompt: str = "",
):
    print(f"\n[generate_assets 시작] scene_offset={scene_offset}, 문장 개수={len(sentences)}, assets_folder={assets_folder}")
    print(f"  예상 scene_id 범위: {scene_offset + 1} ~ {scene_offset + len(sentences)}")
    cleanup_assets_folder(assets_folder)
    os.makedirs(assets_folder, exist_ok=True)
    image_prompts = prompts_override or generate_visual_prompts(sentences, mode=mode, progress_cb=progress_cb, original_script=original_script, custom_style_prompt=custom_style_prompt)
    existing_images = existing_images or {}
    results = {}

    def process_scene(scene_num: int, text: str, prompt: str):
        audio_file = os.path.join(assets_folder, f"scene_{scene_num}_audio.mp3")
        image_file = os.path.join(assets_folder, f"scene_{scene_num}_image.png")

        if progress_cb:
            progress_cb(f"{scene_num}번 씬 TTS 생성 중...")
        print(f"[DEBUG] TTS 생성: scene_num={scene_num}, audio_file={audio_file}, text={text[:50]}...")
        alignment = generate_tts_with_alignment(voice_id, text, audio_file, elevenlabs_api_key=elevenlabs_api_key)
        if alignment is None:
            print(f"[경고] scene_num={scene_num}: TTS 생성 실패")
            if progress_cb:
                progress_cb(f"{scene_num}번 씬 TTS 생성 실패")
            return None
        if not os.path.exists(audio_file):
            print(f"[경고] scene_num={scene_num}: TTS 생성 후 오디오 파일이 존재하지 않습니다: {audio_file}")
            return None
        print(f"[DEBUG] TTS 생성 완료: scene_num={scene_num}, audio_file={audio_file}, 파일 크기={os.path.getsize(audio_file)} bytes")

        if progress_cb:
            progress_cb(f"{scene_num}번 씬 이미지 생성 중...")

        # existing_images는 전체 인덱스를 사용하므로 그대로 사용
        # 이미지가 이미 존재하는지 먼저 확인 (중복 생성 방지)
        if os.path.exists(image_file):
            print(f"[이미지 확인] scene_{scene_num}: 이미지가 이미 존재함 - {image_file} (크기: {os.path.getsize(image_file)} bytes)")
            image_generated = True
        else:
            # existing_images에서 이미지 경로 확인
            img_path = existing_images.get(scene_num)
            if img_path and os.path.exists(img_path):
                print(f"[이미지 복사] scene_{scene_num}: existing_images에서 이미지 복사 - {img_path} -> {image_file}")
                shutil.copy(img_path, image_file)
                image_generated = True
            else:
                print(f"[이미지 생성] scene_{scene_num}: 새 이미지 생성 시작 - {image_file}")
                image_generated = generate_image(prompt, image_file, mode=mode, replicate_api_key=replicate_api_key, custom_style_prompt=custom_style_prompt)

        semantic_segments = generate_semantic_segments(text)

        if not image_generated:
            Image.new("RGB", (1920, 1080), color="black").save(image_file)
            if progress_cb:
                progress_cb(f"{scene_num}번 씬 이미지 생성 실패, 기본 이미지 사용")
        else:
            if progress_cb:
                progress_cb(f"{scene_num}번 씬 이미지 생성 완료")

        return {
            "scene_id": scene_num,
            "text": text,
            "audio_file": audio_file,
            "image_file": image_file,
            "alignment": alignment,
            "image_prompt": prompt,
            "semantic_segments": semantic_segments,
        }

    max_workers = min(MAX_SCENE_WORKERS, len(sentences)) or 1
    print(f"[DEBUG] generate_assets: scene_offset={scene_offset}, 문장 개수={len(sentences)}, 첫 번째 scene_num={scene_offset + 1}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for idx, sentence in enumerate(sentences):
            prompt = image_prompts[idx] if idx < len(image_prompts) else ""
            # scene_offset을 추가하여 전체 인덱스가 연속되도록 함
            scene_num = scene_offset + idx + 1
            if idx < 3:  # 처음 3개만 로그 출력
                print(f"[DEBUG] generate_assets: idx={idx}, scene_num={scene_num}, sentence={sentence[:30]}...")
            future = executor.submit(process_scene, scene_num, sentence, prompt)
            future_map[future] = scene_num

        completed_count = 0
        failed_count = 0
        for future in as_completed(future_map):
            scene_num = future_map[future]
            try:
                result = future.result()
                if result:
                    results[scene_num] = result
                    completed_count += 1
                    # 오디오 파일이 실제로 생성되었는지 확인
                    audio_file = result.get('audio_file')
                    if audio_file and os.path.exists(audio_file):
                        if completed_count <= 3:  # 처음 3개만 로그
                            print(f"[확인] scene_{scene_num} 오디오 파일 생성 완료: {audio_file}")
                    else:
                        print(f"[경고] scene_{scene_num} 오디오 파일이 생성되지 않았습니다: {audio_file}")
                else:
                    failed_count += 1
                    print(f"[경고] scene_{scene_num} 결과가 None입니다.")
            except Exception as exc:
                failed_count += 1
                print(f"[자산 생성 오류] 씬 {scene_num}: {exc}")
                import traceback
                traceback.print_exc()
                if progress_cb:
                    progress_cb(f"{scene_num}번 씬 처리 중 오류가 발생했습니다.")

    print(f"[generate_assets 완료] 성공: {completed_count}개, 실패: {failed_count}개, 총 예상: {len(sentences)}개")
    ordered = [results[idx] for idx in sorted(results.keys())]
    if len(ordered) != len(sentences):
        print(f"[경고] 생성된 씬 수({len(ordered)})가 예상 문장 수({len(sentences)})와 다릅니다!")
        print(f"  예상 scene_id 범위: {scene_offset + 1} ~ {scene_offset + len(sentences)}")
        print(f"  실제 생성된 scene_id: {sorted(results.keys())}")
    if progress_cb:
        progress_cb(f"자산 생성이 완료되었습니다. (총 {len(ordered)}개 씬)")
    return ordered


# =============================================================================
# 타임스탬프 및 영상 합성
# =============================================================================


def calculate_timestamps(scene_data: List[Dict[str, Any]], progress_cb=None):
    result = []
    total = 0.0
    for scene in scene_data:
        audio_path = scene["audio_file"]
        try:
            info = MP3(audio_path)
            duration = info.info.length
        except FileNotFoundError:
            duration = 0.0
            if progress_cb:
                progress_cb(f"{audio_path} 파일이 없어 길이를 0으로 처리합니다.")
        except HeaderNotFoundError:
            duration = 0.0
        except Exception as exc:
            print(f"[타임스탬프 오류] {audio_path}: {exc}")
            duration = 0.0

        scene_copy = dict(scene)
        scene_copy["duration"] = round(duration, 2)
        scene_copy["start_time"] = round(total, 2)
        scene_copy["end_time"] = round(total + duration, 2)
        total += duration
        result.append(scene_copy)
    if progress_cb:
        progress_cb("타임스탬프 계산이 완료되었습니다.")
    return result, total


def seconds_to_srt_time(seconds: float) -> str:
    frac, whole = math.modf(seconds)
    ms = int(frac * 1000)
    minutes, sec = divmod(int(whole), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{sec:02},{ms:03}"


def split_text_into_chunks(text: str, char_limit: int) -> List[str]:
    wrapper = textwrap.TextWrapper(
        width=char_limit,
        break_long_words=True,
        replace_whitespace=False,
        break_on_hyphens=False,
    )
    return wrapper.wrap(text)


def srt_to_ass_time(seconds: float) -> str:
    """초를 ASS 시간 형식 (H:MM:SS.cc)으로 변환"""
    frac, whole = math.modf(seconds)
    centiseconds = int(frac * 100)
    minutes, sec = divmod(int(whole), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{sec:02}.{centiseconds:02}"


def srt_to_ass(srt_file: str, ass_file: str):
    """SRT 파일을 ASS 형식으로 변환 (배경 박스가 끊어지지 않도록)"""
    def srt_time_to_seconds(srt_time: str) -> float:
        """SRT 시간 형식 (HH:MM:SS,mmm)을 초로 변환"""
        time_part, ms_part = srt_time.split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        return h * 3600 + m * 60 + s + ms / 1000.0
    
    try:
        with open(srt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # ASS 헤더 작성
        ass_content = [
            "[Script Info]\n",
            "Title: Generated Subtitles\n",
            "ScriptType: v4.00+\n",
            "PlayResX: 1920\n",  # 영상 가로 해상도
            "PlayResY: 1080\n",  # 영상 세로 해상도
            "\n",
            "[V4+ Styles]\n",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n",
            # [수정] 단일 스타일 사용 (Default)
            # BorderStyle=3 (Opaque Box): 텍스트 뒤에 배경 박스 생성
            # Outline=20: 배경 박스의 패딩(여백) 역할
            # BackColour=&H60000000: 반투명 검정 배경 (Alpha 60)
            f"Style: Default,{SUBTITLE_FONT_NAME},80,&H00FFFFFF,&H000000FF,&H00000000,&H60000000,-1,0,0,0,100,100,0,0,3,20,0,2,10,10,50,1\n",
            "\n",
            "[Events]\n",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # 자막 인덱스
            try:
                int(line)
            except ValueError:
                i += 1
                continue
            
            # 시간 정보
            if i + 1 >= len(lines):
                break
            time_line = lines[i + 1].strip()
            if '-->' not in time_line:
                i += 1
                continue
            
            start_time_str, end_time_str = time_line.split('-->')
            start_time = srt_time_to_seconds(start_time_str.strip())
            end_time = srt_time_to_seconds(end_time_str.strip())
            
            # 자막 텍스트 수집
            subtitle_text_lines = []
            i += 2
            while i < len(lines) and lines[i].strip():
                # 텍스트를 그대로 유지 (공백 포함)
                clean_line = lines[i].rstrip()
                subtitle_text_lines.append(clean_line)
                i += 1
            
            if subtitle_text_lines:
                # 여러 줄을 하나로 합치고 ASS 줄바꿈 적용
                text = "\\N".join(subtitle_text_lines)
                
                # [수정] 복잡한 계산 없이 텍스트만 출력하면 ASS 스타일(BorderStyle=3)에 의해 자동으로 박스 생성
                ass_content.append(f"Dialogue: 0,{srt_to_ass_time(start_time)},{srt_to_ass_time(end_time)},Default,,0,0,0,,{text}\n")
            
            i += 1  # 빈 줄 건너뛰기
        
        with open(ass_file, "w", encoding="utf-8") as f:
            f.writelines(ass_content)
        
        return True
    except Exception as exc:
        print(f"[SRT to ASS 변환 오류] {exc}")
        import traceback
        traceback.print_exc()
        return False


def extract_subtitles_for_scene(subtitle_file: str, scene_start: float, scene_end: float, output_file: str):
    """특정 시간 범위의 자막만 추출하여 새 SRT 파일 생성"""
    def srt_time_to_seconds(srt_time: str) -> float:
        """SRT 시간 형식 (HH:MM:SS,mmm)을 초로 변환"""
        time_part, ms_part = srt_time.split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        return h * 3600 + m * 60 + s + ms / 1000.0
    
    def seconds_to_srt_time(seconds: float) -> str:
        """초를 SRT 시간 형식으로 변환"""
        frac, whole = math.modf(seconds)
        ms = int(frac * 1000)
        minutes, sec = divmod(int(whole), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{sec:02},{ms:03}"
    
    try:
        with open(subtitle_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        subtitle_index = 1
        output_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # 자막 인덱스
            try:
                int(line)
            except ValueError:
                i += 1
                continue
            
            # 시간 정보
            if i + 1 >= len(lines):
                break
            time_line = lines[i + 1].strip()
            if '-->' not in time_line:
                i += 1
                continue
            
            start_time_str, end_time_str = time_line.split('-->')
            start_time = srt_time_to_seconds(start_time_str.strip())
            end_time = srt_time_to_seconds(end_time_str.strip())
            
            # 씬 시간 범위와 겹치는지 확인
            if end_time < scene_start or start_time > scene_end:
                # 겹치지 않음, 다음 자막으로
                i += 2
                # 자막 텍스트 건너뛰기
                while i < len(lines) and lines[i].strip():
                    i += 1
                i += 1  # 빈 줄 건너뛰기
                continue
            
            # 겹치는 자막 발견
            # 시간을 씬 시작 시간 기준으로 조정
            adjusted_start = max(0.0, start_time - scene_start)
            adjusted_end = min(scene_end - scene_start, end_time - scene_start)
            
            # 자막 텍스트 수집
            subtitle_text_lines = []
            i += 2
            while i < len(lines) and lines[i].strip():
                subtitle_text_lines.append(lines[i].rstrip())
                i += 1
            
            if subtitle_text_lines:
                output_lines.append(f"{subtitle_index}\n")
                output_lines.append(f"{seconds_to_srt_time(adjusted_start)} --> {seconds_to_srt_time(adjusted_end)}\n")
                output_lines.append("\n".join(subtitle_text_lines) + "\n")
                output_lines.append("\n")
                subtitle_index += 1
            
            i += 1  # 빈 줄 건너뛰기
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(output_lines)
        
        return True
    except Exception as exc:
        print(f"[자막 추출 오류] {exc}")
        import traceback
        traceback.print_exc()
        return False


def create_video(
    scene_data_with_timestamps: List[Dict[str, Any]],
    char_limit: int,
    subtitle_file: str,
    output_video_file: str,
    assets_folder: str,
    progress_cb=None,
    include_subtitles=True,
):
    if progress_cb:
        progress_cb("영상 합성을 시작합니다.")

    segment_start = min(scene["start_time"] for scene in scene_data_with_timestamps)

    subtitle_index = 1
    try:
        with open(subtitle_file, "w", encoding="utf-8") as f:
            for scene in scene_data_with_timestamps:
                alignment = scene.get("alignment")
                semantic_segments = scene.get("semantic_segments") or []
                phrases = alignment_to_phrases(alignment, char_limit, semantic_segments) if alignment else []
                if not phrases:
                    wrapped = split_text_into_chunks(scene["text"], char_limit)
                    for chunk in wrapped:
                        chunk_text = normalize_subtitle_text(chunk)
                        start_rel = max(0.0, scene["start_time"] - segment_start)
                        end_rel = max(start_rel, scene["end_time"] - segment_start)
                        f.write(f"{subtitle_index}\n")
                        f.write(f"{seconds_to_srt_time(start_rel)} --> {seconds_to_srt_time(end_rel)}\n")
                        f.write(f"{chunk_text}\n\n")
                        subtitle_index += 1
                else:
                    for phrase_text, phrase_start, phrase_end in phrases:
                        start_time = (scene["start_time"] - segment_start) + phrase_start
                        end_time = (scene["start_time"] - segment_start) + phrase_end
                        f.write(f"{subtitle_index}\n")
                        f.write(f"{seconds_to_srt_time(start_time)} --> {seconds_to_srt_time(end_time)}\n")
                        f.write(f"{phrase_text}\n\n")
                        subtitle_index += 1
    except Exception as exc:
        print(f"[자막 오류] {exc}")
        if progress_cb:
            progress_cb("자막 파일 생성에 실패했습니다.")
        return False

    scene_video_files = []
    temp_scene_folder = os.path.join(assets_folder, "temp_scenes")
    os.makedirs(temp_scene_folder, exist_ok=True)
    
    print(f"[create_video] 총 {len(scene_data_with_timestamps)}개 씬 처리 시작")

    # 2. 개별 씬 렌더링 (가장 중요한 수정 부분)
    for idx, scene in enumerate(scene_data_with_timestamps, 1):
        scene_id = scene.get('scene_id')
        image_path = scene.get("image_file")
        video_path = scene.get("video_file")  # 비디오 파일 경로 (있으면 사용)
        print(f"[씬 처리 시작] {idx}/{len(scene_data_with_timestamps)} - scene_{scene_id}")
        print(f"[이미지 확인] scene_{scene_id}: 원본 경로={image_path}")
        if video_path:
            print(f"[비디오 확인] scene_{scene_id}: 비디오 파일 경로={video_path}")
        
        # 비디오 파일이 있으면 우선 사용
        has_video = video_path and os.path.exists(video_path)
        
        # 이미지 파일 존재 및 유효성 검증
        image_valid = False
        if image_path and os.path.exists(image_path):
            image_size = os.path.getsize(image_path)
            print(f"[이미지 확인] scene_{scene_id}: 이미지 파일 존재 ({image_size} bytes)")
            
            # 파일 크기 검증
            if image_size < 100:
                print(f"[경고] scene_{scene_id}: 이미지 파일이 너무 작습니다 ({image_size} bytes)")
            else:
                # PIL로 이미지 파일 유효성 검증
                try:
                    with Image.open(image_path) as test_img:
                        test_img.verify()  # 이미지 파일 무결성 검증
                    # verify() 후에는 파일을 다시 열어야 함
                    with Image.open(image_path) as test_img:
                        test_img.load()  # 이미지 로드
                    print(f"[이미지 확인] scene_{scene_id}: 이미지 파일 유효성 검증 완료")
                    image_valid = True
                except Exception as verify_exc:
                    print(f"[경고] scene_{scene_id}: 이미지 파일 유효성 검증 실패: {verify_exc}")
        
        # 이미지 파일이 없거나 유효하지 않으면 검은색 이미지 생성
        if not image_valid:
            print(f"[경고] scene_{scene_id}: 이미지 파일이 없거나 손상되어 검은색 이미지 생성")
            image_path = os.path.join(assets_folder, f"scene_{scene_id}_fallback.png")
            try:
                black_img = Image.new("RGB", (1920, 1080), color="black")
                black_img.save(image_path, format="PNG")
                # 생성된 이미지 검증
                with Image.open(image_path) as test_img:
                    test_img.verify()
                print(f"[검은색 이미지] scene_{scene_id}: 검은색 이미지 생성 및 검증 완료")
            except Exception as fallback_exc:
                print(f"[오류] scene_{scene_id}: 검은색 이미지 생성 실패: {fallback_exc}")
                import traceback
                traceback.print_exc()
        
        audio_file = scene.get("audio_file")
        duration = scene.get("duration", 0)
        
        # 오디오 파일이 있으면 실제 duration 재확인
        if audio_file and os.path.exists(audio_file):
            try:
                audio_info = MP3(audio_file)
                actual_duration = audio_info.info.length
                if abs(actual_duration - duration) > 0.1:  # 0.1초 이상 차이나면 경고
                    print(f"[경고] scene_id={scene_id}: duration 불일치! 계산된 duration={duration:.2f}초, 실제 오디오 duration={actual_duration:.2f}초")
                    duration = actual_duration  # 실제 duration 사용
            except Exception as e:
                print(f"[경고] scene_id={scene_id}: 오디오 duration 확인 실패: {e}")
        
        print(f"[create_video] scene_id={scene_id}: duration={duration:.2f}초, start_time={scene.get('start_time', 0):.2f}초, end_time={scene.get('end_time', 0):.2f}초")
        
        if not audio_file or not os.path.exists(audio_file):
            print(f"[경고] scene_id={scene_id}: 오디오 파일이 없거나 존재하지 않습니다. 무음 오디오 생성")
            audio_file = None  # 나중에 무음 오디오 생성
        
        # 개별 씬 비디오 파일 경로
        scene_video_file = os.path.join(temp_scene_folder, f"scene_{scene_id}_video.mp4")
        # 주의: scene_video_files 리스트에 추가는 비디오 생성 성공 후에만 수행
        
        try:
            # 각 씬의 자막 파일 생성 (전체 자막 파일에서 해당 씬 부분만 추출)
            scene_subtitle_srt = os.path.join(temp_scene_folder, f"scene_{scene_id}_subtitles.srt")
            scene_subtitle_ass = os.path.join(temp_scene_folder, f"scene_{scene_id}_subtitles.ass")
            scene_start = scene["start_time"] - segment_start
            scene_end = scene["end_time"] - segment_start
            
            print(f"[자막 추출] scene_{scene_id}: 시간 범위 {scene_start:.2f}초 ~ {scene_end:.2f}초")
            
            # 해당 씬의 자막만 추출 (SRT 형식)
            extract_success = extract_subtitles_for_scene(subtitle_file, scene_start, scene_end, scene_subtitle_srt)
            if extract_success and os.path.exists(scene_subtitle_srt):
                srt_size = os.path.getsize(scene_subtitle_srt)
                print(f"[자막 추출] scene_{scene_id}: SRT 파일 생성됨 ({srt_size} bytes)")
                if srt_size > 0:
                    # SRT를 ASS 형식으로 변환 (배경 박스가 끊어지지 않도록)
                    ass_success = srt_to_ass(scene_subtitle_srt, scene_subtitle_ass)
                    if ass_success and os.path.exists(scene_subtitle_ass):
                        ass_size = os.path.getsize(scene_subtitle_ass)
                        print(f"[자막 변환] scene_{scene_id}: ASS 파일 생성됨 ({ass_size} bytes)")
                    else:
                        print(f"[경고] scene_{scene_id}: ASS 파일 생성 실패")
                else:
                    print(f"[경고] scene_{scene_id}: SRT 파일이 비어있음")
            else:
                print(f"[경고] scene_{scene_id}: 자막 추출 실패 또는 파일 없음")
            
            # 비디오 스트림 생성
            # 비디오 파일이 있으면 우선 사용, 없으면 이미지 사용
            try:
                if has_video:
                    # 비디오 파일 사용
                    print(f"[비디오 사용] scene_{scene_id}: 비디오 파일 사용, TTS 길이에 맞게 조정 (duration={duration:.2f}초)")
                    
                    # 비디오 파일의 실제 길이 확인
                    try:
                        probe = ffmpeg.probe(video_path)
                        video_duration = float(probe['streams'][0]['duration'])
                        print(f"[비디오 정보] scene_{scene_id}: 비디오 원본 길이={video_duration:.2f}초, 필요한 길이={duration:.2f}초")
                    except Exception as probe_exc:
                        print(f"[경고] scene_{scene_id}: 비디오 정보 확인 실패, duration 사용: {probe_exc}")
                        video_duration = duration
                    
                    # 중요: stream_loop=-1로 무한 반복 설정 (오디오보다 짧을 경우 대비)
                    # 오디오보다 길면 뒤에서 trim으로 잘림
                    input_stream = ffmpeg.input(video_path, stream_loop=-1)
                    
                    # 비디오 스트림 처리
                    # 1. scale: 해상도 통일
                    # 2. fps: 프레임레이트 강제 통일 (중요)
                    # 3. trim: 오디오 길이에 딱 맞게 자르기
                    # 4. setpts: 타임스탬프 리셋
                    base_stream = (
                        input_stream['v']
                        .filter("scale", 1920, 1080)
                        .filter("fps", fps=VIDEO_FPS, round="up")
                        .filter("trim", duration=duration)
                        .filter("setpts", "PTS-STARTPTS")
                    )
                    print(f"[비디오 조정] scene_{scene_id}: 무한 루프 후 정확히 자르기 (목표: {duration:.2f}초)")
                else:
                    # 이미지 파일 사용 (기존 로직)
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")
                    
                    # PIL로 이미지 크기 확인
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                        print(f"[이미지 확인] scene_{scene_id}: 이미지 크기 {img_width}x{img_height}")
                    
                    # 이미지 입력 (loop=1, t=duration)
                    input_stream = ffmpeg.input(image_path, loop=1, t=duration)
                    base_stream = (
                        input_stream
                        .filter("scale", 1920, 1080)
                        .filter("fps", fps=VIDEO_FPS, round="up")
                        .filter("trim", duration=duration) # 중복 안전장치
                        .filter("setpts", "PTS-STARTPTS")
                    )
            except Exception as img_exc:
                print(f"[오류] scene_{scene_id}: 이미지 로드 실패: {img_exc}")
                import traceback
                traceback.print_exc()
                # 이미지 로드 실패 시 검은색 이미지로 대체
                fallback_image = os.path.join(assets_folder, f"scene_{scene_id}_fallback.png")
                if not os.path.exists(fallback_image):
                    black_img = Image.new("RGB", (1920, 1080), color="black")
                    black_img.save(fallback_image, format="PNG")
                image_path = fallback_image
                base_stream = (
                    ffmpeg.input(image_path, format='image2', loop=1, framerate=VIDEO_FPS)
                    .filter("scale", 1920, 1080)
                    .filter("trim", start=0, duration=duration)
                    .filter("setpts", "PTS-STARTPTS")
                )
            
            # 자막 적용 (ASS 형식 사용 - 배경 박스가 끊어지지 않음)
            # 자막을 먼저 적용하여 타임코드가 정확하게 유지되도록 함
            if include_subtitles:
                subtitle_kwargs = {}
                if os.path.isdir(FONTS_FOLDER):
                    subtitle_kwargs["fontsdir"] = os.path.abspath(FONTS_FOLDER)
                # [수정] force_style 업데이트: 배경 박스가 보이도록 BackColour 수정
                # &H60000000: 반투명 검정 (Alpha 60)
                # BorderStyle=3: Opaque Box
                # Outline=20: 박스 내부 패딩
                subtitle_kwargs["force_style"] = f"BackColour=&H60000000,BorderStyle=3,Outline=20"
                
                # 씬별 ASS 자막 파일이 존재하는 경우에만 자막 적용
                if os.path.exists(scene_subtitle_ass) and os.path.getsize(scene_subtitle_ass) > 0:
                    video_with_subs = base_stream.filter("subtitles", scene_subtitle_ass, **subtitle_kwargs)
                else:
                    video_with_subs = base_stream  # 자막 없이 진행
            else:
                video_with_subs = base_stream  # 자막 포함 옵션이 꺼져있으면 자막 없이 진행
            
            # 모션 효과 없이 정적 이미지 사용
            # duration을 명시적으로 제한 (이미 base_stream에서 trim이 적용되었지만, 추가 안전장치)
            video_stream = video_with_subs.filter("trim", start=0, duration=duration).filter("setpts", "PTS-STARTPTS")
            
            # 오디오 입력 스트림 설정
            if audio_file and os.path.exists(audio_file):
                # 오디오도 정확히 duration에 맞춤 (약간의 오차 제거)
                audio_stream = (
                    ffmpeg.input(audio_file)
                    .filter("atrim", duration=duration)
                    .filter("asetpts", "PTS-STARTPTS")
                )
            else:
                # 오디오 없으면 무음 생성
                audio_stream = ffmpeg.input('anullsrc=channel_layout=stereo:sample_rate=44100', f='lavfi', t=duration)

            # 개별 씬 인코딩 실행
            # 여기서 포맷을 완전히 통일시켜야 나중에 concat할 때 문제가 안 생김
            scene_output = ffmpeg.output(
                video_stream,
                audio_stream,
                scene_video_file,
                vcodec="libx264",
                acodec="aac",
                pix_fmt="yuv420p",
                r=VIDEO_FPS,       # 프레임레이트 강제
                ar=44100,          # 오디오 샘플레이트 강제
                ac=2,              # 오디오 채널 강제 (스테레오)
                t=duration,        # 길이 강제 (최종 안전장치)
                preset="ultrafast", # 속도 우선
                crf=23
            )
            try:
                # 오류 디버깅을 위해 quiet=False로 설정
                ffmpeg.run(scene_output, overwrite_output=True, quiet=False)
                
                # 생성된 비디오 파일 확인
                if os.path.exists(scene_video_file):
                    file_size = os.path.getsize(scene_video_file)
                    print(f"[성공] scene_{scene_id} 비디오 생성 완료: {os.path.basename(scene_video_file)} (크기: {file_size} bytes)")
                    
                    # 비디오 생성 성공 후 리스트에 추가 (순서 보장)
                    if scene_video_file not in scene_video_files:
                        scene_video_files.append(scene_video_file)
                        print(f"[DEBUG] scene_{scene_id} 비디오 파일을 리스트에 추가: {os.path.basename(scene_video_file)}")
                    
                    # 생성된 비디오의 실제 duration 확인 및 강제 수정
                    try:
                        probe = ffmpeg.probe(scene_video_file)
                        video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
                        audio_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
                        
                        actual_video_duration = 0
                        actual_audio_duration = 0
                        
                        if video_streams:
                            actual_video_duration = float(video_streams[0].get('duration', 0))
                        if audio_streams:
                            actual_audio_duration = float(audio_streams[0].get('duration', 0))
                        
                        # 오디오 duration을 우선 기준으로 사용 (오디오가 있으면)
                        target_duration = actual_audio_duration if actual_audio_duration > 0 else duration
                        
                        # 0.1초 이상 차이나면 재인코딩 (더 엄격한 기준)
                        duration_diff = abs(actual_video_duration - target_duration)
                        if duration_diff > 0.1:
                            print(f"[경고] scene_{scene_id}: 비디오 duration 불일치! 예상={target_duration:.2f}초, 실제 비디오={actual_video_duration:.2f}초, 실제 오디오={actual_audio_duration:.2f}초")
                            print(f"[수정] scene_{scene_id}: 비디오를 정확한 duration으로 재인코딩 중... (실제: {actual_video_duration:.2f}초 -> 목표: {target_duration:.2f}초)")
                            temp_file = scene_video_file + ".tmp"
                            os.rename(scene_video_file, temp_file)
                            input_video = ffmpeg.input(temp_file)
                            
                            # 비디오와 오디오 모두 정확히 target_duration에 맞춤
                            corrected_video = input_video['v'].filter('trim', start=0, duration=target_duration).filter('setpts', 'PTS-STARTPTS')
                            corrected_audio = input_video['a'].filter('atrim', start=0, duration=target_duration).filter('asetpts', 'PTS-STARTPTS')
                            
                            corrected_output = ffmpeg.output(
                                corrected_video,
                                corrected_audio,
                                scene_video_file,
                                vcodec="libx264",
                                acodec="aac",
                                pix_fmt="yuv420p",
                                t=target_duration,  # 최종 안전장치
                                preset="ultrafast",
                                crf="23"
                            )
                            ffmpeg.run(corrected_output, overwrite_output=True, quiet=True)
                            os.remove(temp_file)
                            
                            # 재인코딩 후 다시 확인
                            try:
                                verify_probe = ffmpeg.probe(scene_video_file)
                                verify_video_streams = [s for s in verify_probe.get('streams', []) if s.get('codec_type') == 'video']
                                if verify_video_streams:
                                    verify_duration = float(verify_video_streams[0].get('duration', 0))
                                    if abs(verify_duration - target_duration) <= 0.1:
                                        print(f"[수정 완료] scene_{scene_id}: duration 수정됨 ({target_duration:.2f}초로 정확히 맞춤, 검증: {verify_duration:.2f}초)")
                                    else:
                                        print(f"[경고] scene_{scene_id}: 재인코딩 후에도 duration 불일치 (목표: {target_duration:.2f}초, 실제: {verify_duration:.2f}초)")
                            except Exception as verify_exc:
                                print(f"[경고] scene_{scene_id}: 재인코딩 후 duration 검증 실패: {verify_exc}")
                        else:
                            print(f"[확인] scene_{scene_id}: duration 정확히 일치 (비디오: {actual_video_duration:.2f}초, 오디오: {actual_audio_duration:.2f}초, 목표: {target_duration:.2f}초)")
                    except Exception as probe_exc:
                        print(f"[경고] scene_{scene_id}: 비디오 duration 확인 실패: {probe_exc}")
                        
            except ffmpeg.Error as ffmpeg_exc:
                print(f"[FFmpeg 오류] scene_{scene_id} 비디오 생성 실패:")
                if ffmpeg_exc.stderr:
                    stderr_text = ffmpeg_exc.stderr.decode('utf-8', errors='ignore')
                    print(f"[FFmpeg stderr] {stderr_text}")
                if ffmpeg_exc.stdout:
                    stdout_text = ffmpeg_exc.stdout.decode('utf-8', errors='ignore')
                    print(f"[FFmpeg stdout] {stdout_text}")
                # 오류 발생 시 기본 스트림으로 재시도
                print(f"[재시도] scene_{scene_id} 기본 스트림으로 재시도 중...")
                try:
                    # 모션 효과 없이 기본 스트림만 사용
                    # 이미지 형식을 명시적으로 지정 (루프 후 정확히 duration에 맞춤)
                    simple_stream = (
                        ffmpeg.input(image_path, format='image2', loop=1, framerate=VIDEO_FPS)
                        .filter("scale", 1920, 1080)
                        .filter("trim", start=0, duration=duration)
                        .filter("setpts", "PTS-STARTPTS")
                    )
                    if include_subtitles and os.path.exists(scene_subtitle_ass) and os.path.getsize(scene_subtitle_ass) > 0:
                        # fallback 경로에서도 subtitle_kwargs 정의
                        fallback_subtitle_kwargs = {}
                        if os.path.isdir(FONTS_FOLDER):
                            fallback_subtitle_kwargs["fontsdir"] = os.path.abspath(FONTS_FOLDER)
                        fallback_subtitle_kwargs["force_style"] = f"BackColour=&H60000000,BorderStyle=3,Outline=20"
                        simple_with_subs = simple_stream.filter("subtitles", scene_subtitle_ass, **fallback_subtitle_kwargs)
                    else:
                        simple_with_subs = simple_stream
                    # 오디오도 정확히 duration에 맞춤
                    if audio_file and os.path.exists(audio_file):
                        fallback_audio_stream = ffmpeg.input(audio_file).filter("atrim", start=0, duration=duration).filter("asetpts", "PTS-STARTPTS")
                    else:
                        fallback_audio_stream = ffmpeg.input('anullsrc=channel_layout=mono:sample_rate=44100', f='lavfi', t=duration)
                    
                    simple_output = ffmpeg.output(
                        simple_with_subs.filter("trim", start=0, duration=duration).filter("setpts", "PTS-STARTPTS"),
                        fallback_audio_stream,
                        scene_video_file,
                        vcodec="libx264",
                        acodec="aac",
                        pix_fmt="yuv420p",
                        t=duration,  # duration 명시적으로 제한 (최종 안전장치)
                        preset="ultrafast",
                        crf="23",
                        threads="0"
                    )
                    ffmpeg.run(simple_output, overwrite_output=True, quiet=False)
                    print(f"[성공] scene_{scene_id} 기본 스트림으로 생성 완료")
                    # 재시도 성공 후에도 리스트에 추가
                    if os.path.exists(scene_video_file) and scene_video_file not in scene_video_files:
                        scene_video_files.append(scene_video_file)
                        print(f"[DEBUG] scene_{scene_id} 비디오 파일을 리스트에 추가 (재시도 성공): {os.path.basename(scene_video_file)}")
                except Exception as retry_exc:
                    print(f"[실패] scene_{scene_id} 재시도도 실패: {retry_exc}")
                    raise ffmpeg_exc  # 원래 오류를 다시 발생
            
            # 각 씬 처리 후 반드시 리스트에 포함되었는지 확인 (첫 번째 씬 누락 방지)
            if scene_video_file not in scene_video_files:
                if os.path.exists(scene_video_file):
                    scene_video_files.append(scene_video_file)
                    print(f"[DEBUG] scene_{scene_id} 비디오 파일을 리스트에 추가 (처리 후 확인): {os.path.basename(scene_video_file)}")
                else:
                    # 파일이 없으면 즉시 무음 비디오 생성
                    print(f"[경고] scene_{scene_id} 비디오 파일이 없어 즉시 무음 비디오 생성")
                    try:
                        silent_video = ffmpeg.input('color=c=black:s=1920x1080:d=' + str(duration), f='lavfi')
                        silent_audio = ffmpeg.input('anullsrc=channel_layout=mono:sample_rate=44100', f='lavfi', t=duration)
                        fallback_output = ffmpeg.output(silent_video, silent_audio, scene_video_file, vcodec="libx264", acodec="aac", pix_fmt="yuv420p", preset="ultrafast", t=duration)
                        ffmpeg.run(fallback_output, overwrite_output=True, quiet=True)
                        if os.path.exists(scene_video_file):
                            scene_video_files.append(scene_video_file)
                            print(f"[DEBUG] scene_{scene_id} 무음 비디오 생성 및 리스트 추가 완료")
                    except Exception as immediate_fallback_exc:
                        print(f"[오류] scene_{scene_id} 즉시 무음 비디오 생성 실패: {immediate_fallback_exc}")
            
            if idx % 10 == 0 or idx == len(scene_data_with_timestamps):
                if progress_cb:
                    progress_cb(f"씬 비디오 생성 중... ({idx}/{len(scene_data_with_timestamps)})")
                print(f"[create_video] {idx}/{len(scene_data_with_timestamps)}개 씬 비디오 생성 완료")
                
        except Exception as exc:
            print(f"[경고] scene_{scene_id} 비디오 생성 실패: {exc}")
            import traceback
            traceback.print_exc()
            # 실패한 씬은 무음 비디오로 대체
            try:
                silent_video = ffmpeg.input('color=c=black:s=1920x1080:d=' + str(duration), f='lavfi')
                silent_audio = ffmpeg.input('anullsrc=channel_layout=mono:sample_rate=44100', f='lavfi', t=duration)
                fallback_output = ffmpeg.output(silent_video, silent_audio, scene_video_file, vcodec="libx264", acodec="aac", pix_fmt="yuv420p", preset="ultrafast", t=duration)
                ffmpeg.run(fallback_output, overwrite_output=True, quiet=True)
                # fallback 비디오 생성 성공 후 리스트에 추가
                if os.path.exists(scene_video_file) and scene_video_file not in scene_video_files:
                    scene_video_files.append(scene_video_file)
                    print(f"[DEBUG] scene_{scene_id} 비디오 파일을 리스트에 추가 (fallback): {os.path.basename(scene_video_file)}")
            except:
                pass
    
    # 전체 씬 비디오 생성 완료 후 확인 및 누락된 씬 보완
    print(f"[create_video] 전체 씬 비디오 생성 완료. 총 {len(scene_data_with_timestamps)}개 씬 예상, 현재 리스트에 {len(scene_video_files)}개 파일")
    
    # scene_data_with_timestamps의 모든 씬에 대해 비디오 파일이 존재하는지 확인
    expected_scene_ids = {scene.get('scene_id') for scene in scene_data_with_timestamps}
    existing_scene_ids = set()
    scene_file_map = {}  # scene_id -> file_path 매핑
    
    for scene_file in scene_video_files:
        if os.path.exists(scene_file):
            import re
            basename = os.path.basename(scene_file)
            match = re.search(r'scene_(\d+)_video\.mp4', basename)
            if match:
                scene_id = int(match.group(1))
                existing_scene_ids.add(scene_id)
                scene_file_map[scene_id] = scene_file
    
    missing_scene_ids = expected_scene_ids - existing_scene_ids
    if missing_scene_ids:
        print(f"[경고] 누락된 씬 비디오 파일 ({len(missing_scene_ids)}개): {sorted(missing_scene_ids)}")
        # 누락된 씬에 대해 무음 비디오 생성
        for missing_scene_id in sorted(missing_scene_ids):
            # 해당 씬의 정보 찾기
            missing_scene = None
            for scene in scene_data_with_timestamps:
                if scene.get('scene_id') == missing_scene_id:
                    missing_scene = scene
                    break
            
            if missing_scene:
                duration = missing_scene.get('duration', 1.0)
                missing_video_file = os.path.join(temp_scene_folder, f"scene_{missing_scene_id}_video.mp4")
                print(f"[보완] scene_{missing_scene_id} 무음 비디오 생성 중... (duration: {duration:.2f}초)")
                try:
                    silent_video = ffmpeg.input('color=c=black:s=1920x1080:d=' + str(duration), f='lavfi')
                    silent_audio = ffmpeg.input('anullsrc=channel_layout=mono:sample_rate=44100', f='lavfi', t=duration)
                    fallback_output = ffmpeg.output(silent_video, silent_audio, missing_video_file, vcodec="libx264", acodec="aac", pix_fmt="yuv420p", preset="ultrafast", t=duration)
                    ffmpeg.run(fallback_output, overwrite_output=True, quiet=True)
                    if os.path.exists(missing_video_file):
                        scene_video_files.append(missing_video_file)
                        scene_file_map[missing_scene_id] = missing_video_file
                        print(f"[보완 완료] scene_{missing_scene_id} 무음 비디오 생성됨: {os.path.basename(missing_video_file)}")
                    else:
                        print(f"[오류] scene_{missing_scene_id} 무음 비디오 생성 실패")
                except Exception as fallback_exc:
                    print(f"[오류] scene_{missing_scene_id} 무음 비디오 생성 중 예외 발생: {fallback_exc}")
    
    # 최종 확인
    final_existing_count = sum(1 for f in scene_video_files if os.path.exists(f))
    print(f"[create_video] 최종 확인: {final_existing_count}/{len(scene_data_with_timestamps)}개 씬 비디오 파일 존재")
    
    # FFmpeg concat demuxer를 사용하여 모든 씬 비디오 합치기 (메모리 효율적)
    try:
        # 디버깅: scene_video_files 리스트 확인
        print(f"[DEBUG] scene_video_files 리스트 길이: {len(scene_video_files)}")
        for i, scene_file in enumerate(scene_video_files, 1):
            exists = os.path.exists(scene_file)
            size = os.path.getsize(scene_file) if exists else 0
            print(f"[DEBUG] scene_video_files[{i-1}]: {os.path.basename(scene_file)} - 존재: {exists}, 크기: {size} bytes")
        
        # scene_id 순서대로 정렬 (파일명에서 scene_id 추출)
        def extract_scene_id(filepath):
            """파일명에서 scene_id를 추출하여 정렬에 사용"""
            import re
            basename = os.path.basename(filepath)
            match = re.search(r'scene_(\d+)_video\.mp4', basename)
            if match:
                return int(match.group(1))
            return 0  # 매칭 실패 시 0 반환
        
        # scene_data_with_timestamps의 순서대로 concat_list.txt 생성 (중요: 모든 씬이 포함되도록)
        concat_list_file = os.path.join(temp_scene_folder, "concat_list.txt")
        file_count = 0
        concat_file_contents = []  # 디버깅용
        
        print(f"[DEBUG] scene_data_with_timestamps 순서대로 concat_list.txt 생성 중... (총 {len(scene_data_with_timestamps)}개 씬)")
        
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for idx, scene in enumerate(scene_data_with_timestamps, 1):
                scene_id = scene.get('scene_id')
                # 해당 scene_id의 비디오 파일 찾기
                scene_video_file = scene_file_map.get(scene_id)
                
                if not scene_video_file:
                    # scene_video_files에서 찾기
                    for scene_file in scene_video_files:
                        file_scene_id = extract_scene_id(scene_file)
                        if file_scene_id == scene_id:
                            scene_video_file = scene_file
                            scene_file_map[scene_id] = scene_file
                            break
                
                # 여전히 없으면 경로로 직접 생성
                if not scene_video_file:
                    scene_video_file = os.path.join(temp_scene_folder, f"scene_{scene_id}_video.mp4")
                
                if os.path.exists(scene_video_file):
                    # 절대 경로로 변환 (FFmpeg concat demuxer는 상대 경로에 문제가 있을 수 있음)
                    abs_path = os.path.abspath(scene_video_file)
                    # 경로에 작은따옴표가 있으면 이스케이프 처리
                    escaped_path = abs_path.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")
                    file_count += 1
                    file_size = os.path.getsize(scene_video_file)
                    concat_file_contents.append(f"  [{idx}] scene_{scene_id}_video.mp4 (크기: {file_size} bytes)")
                    print(f"[DEBUG] concat_list[{idx}]: scene_{scene_id} 추가됨")
                else:
                    print(f"[경고] scene_{scene_id} 비디오 파일이 존재하지 않아 concat_list에서 제외: {scene_video_file}")
                    # 무음 비디오 생성 시도
                    duration = scene.get('duration', 1.0)
                    print(f"[보완] scene_{scene_id} 무음 비디오 즉시 생성 중... (duration: {duration:.2f}초)")
                    try:
                        silent_video = ffmpeg.input('color=c=black:s=1920x1080:d=' + str(duration), f='lavfi')
                        silent_audio = ffmpeg.input('anullsrc=channel_layout=mono:sample_rate=44100', f='lavfi', t=duration)
                        fallback_output = ffmpeg.output(silent_video, silent_audio, scene_video_file, vcodec="libx264", acodec="aac", pix_fmt="yuv420p", preset="ultrafast", t=duration)
                        ffmpeg.run(fallback_output, overwrite_output=True, quiet=True)
                        if os.path.exists(scene_video_file):
                            abs_path = os.path.abspath(scene_video_file)
                            escaped_path = abs_path.replace("'", "'\\''")
                            f.write(f"file '{escaped_path}'\n")
                            file_count += 1
                            file_size = os.path.getsize(scene_video_file)
                            concat_file_contents.append(f"  [{idx}] scene_{scene_id}_video.mp4 (무음, 크기: {file_size} bytes)")
                            print(f"[보완 완료] scene_{scene_id} 무음 비디오 생성 및 추가됨")
                        else:
                            print(f"[오류] scene_{scene_id} 무음 비디오 생성 실패")
                    except Exception as immediate_fallback_exc:
                        print(f"[오류] scene_{scene_id} 무음 비디오 즉시 생성 중 예외 발생: {immediate_fallback_exc}")
        
        if file_count == 0:
            raise RuntimeError("합칠 씬 비디오 파일이 없습니다.")
        
        print(f"[create_video] concat_list_file 생성 완료: {file_count}개 파일")
        print(f"[create_video] concat_list_file 경로: {concat_list_file}")
        print(f"[create_video] concat_list에 포함된 파일 목록:")
        for content in concat_file_contents:
            print(content)
        
        # concat_list.txt 파일 내용 확인 (디버깅)
        if os.path.exists(concat_list_file):
            with open(concat_list_file, "r", encoding="utf-8") as f:
                concat_content = f.read()
                print(f"[DEBUG] concat_list.txt 내용:\n{concat_content}")
        
        if progress_cb:
            progress_cb("FFmpeg로 최종 영상을 렌더링 중입니다.")
        
        print(f"[create_video] concat demuxer로 {file_count}개 씬 비디오 합치는 중...")
        
        # Concat 실행
        # 모든 파일이 동일한 코덱/해상도/fps를 가지므로 concat demuxer가 안전하게 작동함
        concat_input = ffmpeg.input(concat_list_file, format='concat', safe=0)
        output = ffmpeg.output(
            concat_input,
            output_video_file, 
            c="copy", # 스트림 복사 (재인코딩 없음 -> 매우 빠르고 화질 저하 없음)
            movflags="+faststart"
        )
        ffmpeg.run(output, overwrite_output=True, quiet=False)
        try:
            ffmpeg.run(output, overwrite_output=True, quiet=False)
        except ffmpeg.Error as ffmpeg_exc:
            print(f"[FFmpeg 오류] concat demuxer 실패:")
            if ffmpeg_exc.stderr:
                print(ffmpeg_exc.stderr.decode('utf-8'))
            raise
        
        # 임시 파일 정리
        try:
            import shutil
            shutil.rmtree(temp_scene_folder)
        except:
            pass
        
        if progress_cb:
            progress_cb("영상 합성이 완료되었습니다.")
        print(f"[create_video] 완료: {output_video_file}")
        return True
    except ffmpeg.Error as exc:
        print(f"[FFmpeg 오류] {exc.stderr.decode('utf-8') if exc.stderr else exc}")
        if progress_cb:
            progress_cb("FFmpeg 처리 중 오류가 발생했습니다.")
        return False
    except Exception as exc:
        print(f"[합성 오류] {exc}")
        if progress_cb:
            progress_cb("영상 합성 중 알 수 없는 오류가 발생했습니다.")
        return False


# =============================================================================
# 작업 관리
# =============================================================================


jobs_lock = threading.Lock()
# 중복 요청 방지를 위한 딕셔너리 (요청 해시 -> 마지막 요청 시간)
_recent_image_requests = {}
_recent_requests_lock = threading.Lock()
jobs: Dict[str, Dict[str, Any]] = {}


def create_job_record() -> str:
    job_id = uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "progress": [],
            "video_filename": None,
            "error": None,
            "script_text": None,
            "sentences": [],
            "prompts": [],
            "image_files": {},
            "mode": "animation",
            "stage_progress": 0,
            "current_stage": "",
        }
    return job_id


def update_job(job_id: str, status: Optional[str] = None, message: Optional[str] = None, **kwargs):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        if status:
            job["status"] = status
        if message:
            job.setdefault("progress", []).append({"text": message, "timestamp": time.time()})
        for key, value in kwargs.items():
            job[key] = value


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else None


def get_job_paths(job_id: str):
    assets_folder = os.path.join(ASSETS_BASE_FOLDER, job_id)
    subtitle_file = os.path.join(ASSETS_BASE_FOLDER, f"{SUBTITLE_BASE_NAME}_{job_id}.srt")
    video_file = os.path.join(STATIC_FOLDER, f"{FINAL_VIDEO_BASE_NAME}_{job_id}.mp4")
    os.makedirs(assets_folder, exist_ok=True)
    return assets_folder, subtitle_file, video_file


def run_generation_job(
    job_id: str,
    script_text: str,
    char_limit: int,
    voice_id: str,
    prompts_override=None,
    existing_images=None,
    mode="animation",
    sentences_override=None,
    include_subtitles=True,
    replicate_api_key=None,
    elevenlabs_api_key=None,
    custom_style_prompt="",
):
    print("=" * 80)
    print("=== [DEBUG] 영상 생성 작업 시작 ===")
    print(f"=== [DEBUG] job_id: {job_id} ===")
    print(f"=== [DEBUG] mode: {mode} ===")
    print("=" * 80)
    try:
        assets_folder, subtitle_file, video_file = get_job_paths(job_id)
        
        # API 키 확인 및 환경 변수 fallback
        if not replicate_api_key or (isinstance(replicate_api_key, str) and not replicate_api_key.strip()):
            replicate_api_key = REPLICATE_API_TOKEN
            print(f"[API 키] Replicate API 키: 사용자 입력 없음, 저장된 설정 사용")
        else:
            print(f"[API 키] Replicate API 키: 사용자 입력 사용")
        
        if not elevenlabs_api_key or (isinstance(elevenlabs_api_key, str) and not elevenlabs_api_key.strip()):
            elevenlabs_api_key = ELEVENLABS_API_KEY
            print(f"[API 키] ElevenLabs API 키: 사용자 입력 없음, 저장된 설정 사용")
        else:
            print(f"[API 키] ElevenLabs API 키: 사용자 입력 사용")
        
        # video_file 경로 확인 및 로깅
        print(f"[초기화] video_file 경로: {video_file}")
        print(f"[초기화] STATIC_FOLDER: {STATIC_FOLDER}")
        print(f"[초기화] job_id: {job_id}")
        print(f"[초기화] 예상 경로: {os.path.join(STATIC_FOLDER, f'{FINAL_VIDEO_BASE_NAME}_{job_id}.mp4')}")
        # video_file이 올바른 경로인지 확인
        expected_video_file = os.path.join(STATIC_FOLDER, f"{FINAL_VIDEO_BASE_NAME}_{job_id}.mp4")
        if video_file != expected_video_file:
            print(f"[경고] video_file 경로가 예상과 다릅니다!")
            print(f"  실제: {video_file}")
            print(f"  예상: {expected_video_file}")
            # 올바른 경로로 재설정
            video_file = expected_video_file
            print(f"[수정] video_file을 올바른 경로로 재설정: {video_file}")

        def progress(message):
            update_job(job_id, message=message)

        update_job(job_id, status="running")
        print("=== [DEBUG] 작업 준비 중 ===")
        progress("작업을 준비 중입니다.")
        # sentences_override가 있으면 사용 (이미 파싱된 한국어 문장만 포함)
        if sentences_override:
            sentences = sentences_override
        else:
            sentences = split_script_into_sentences(script_text)
        if not sentences:
            raise ValueError("대본에서 문장을 추출할 수 없습니다.")
        progress(f"{len(sentences)}개 문장으로 분리되었습니다.")

        # 긴 대본을 청크로 분할
        chunks = split_sentences_into_chunks(sentences)
        total_chunks = len(chunks)
        
        if total_chunks > 1:
            progress(f"대본을 {total_chunks}개 청크로 분할했습니다. (각 청크당 약 {SCRIPT_CHUNK_TARGET_MINUTES}분 분량)")
        
        all_scene_data = []
        chunk_video_files = []
        sentence_offset = 0
        
        # 각 청크를 독립적으로 처리
        for chunk_idx, chunk_sentences in enumerate(chunks, 1):
            progress(f"청크 {chunk_idx}/{total_chunks} 처리 중... ({len(chunk_sentences)}개 문장, sentence_offset={sentence_offset})")
            
            # 청크별 폴더 생성
            chunk_folder = os.path.join(assets_folder, f"chunk_{chunk_idx}")
            os.makedirs(chunk_folder, exist_ok=True)
            
            # 프롬프트와 이미지 매핑 조정 (sentence_offset 고려)
            chunk_prompts_override = None
            chunk_existing_images = None
            if prompts_override:
                chunk_prompts_override = prompts_override[sentence_offset:sentence_offset + len(chunk_sentences)]
            if existing_images:
                # existing_images는 전체 인덱스를 사용하므로 그대로 전달
                chunk_existing_images = {
                    idx: path
                    for idx, path in existing_images.items()
                    if sentence_offset < idx <= sentence_offset + len(chunk_sentences)
                }
                print(f"[DEBUG] 청크 {chunk_idx}: sentence_offset={sentence_offset}, existing_images 키: {list(chunk_existing_images.keys())[:5]}...")
            
            # 자산 생성 (scene_offset을 sentence_offset으로 전달하여 전체 인덱스가 연속되도록 함)
            update_job(job_id, stage_progress=0, current_stage=f"자산 생성 (청크 {chunk_idx}/{total_chunks})")
            print(f"[DEBUG] 청크 {chunk_idx}: generate_assets 호출, scene_offset={sentence_offset}, 문장 개수={len(chunk_sentences)}")
            chunk_scene_data = generate_assets(
                chunk_sentences,
                voice_id,
                chunk_folder,
                progress_cb=lambda msg: progress(f"[청크 {chunk_idx}] {msg}"),
                prompts_override=chunk_prompts_override,
                existing_images=chunk_existing_images,
                mode=mode,
                original_script=script_text,  # 전체 원본 스크립트 전달 (문맥 분석용)
                scene_offset=sentence_offset,  # 전체 인덱스가 연속되도록 offset 전달
                replicate_api_key=replicate_api_key,
                elevenlabs_api_key=elevenlabs_api_key,
                custom_style_prompt=custom_style_prompt,
            )
            if not chunk_scene_data:
                raise RuntimeError(f"청크 {chunk_idx} 자산 생성에 실패했습니다.")
            
            # 타임스탬프 계산
            update_job(job_id, stage_progress=1, current_stage=f"타임스탬프 계산 (청크 {chunk_idx}/{total_chunks})")
            chunk_scene_data_with_timestamps, chunk_duration = calculate_timestamps(
                chunk_scene_data, progress_cb=lambda msg: progress(f"[청크 {chunk_idx}] {msg}")
            )
            
            # 비디오 파일 정보 추가 (job_data에서 가져오기)
            with jobs_lock:
                job_data = jobs.get(job_id)
                if job_data is not None:
                    video_files = job_data.get("video_files") or {}
                    for scene in chunk_scene_data_with_timestamps:
                        scene_id = scene.get('scene_id')
                        if scene_id in video_files:
                            video_path = video_files[scene_id]
                            if os.path.exists(video_path):
                                scene["video_file"] = video_path
                                print(f"[비디오 추가] scene_{scene_id}: 비디오 파일 추가됨 - {video_path}")
            
            # 청크별 오디오 파일 빠른 체크
            print(f"\n[체크] 청크 {chunk_idx} 오디오 파일 확인:")
            print(f"  sentence_offset={sentence_offset}, 문장 개수={len(chunk_sentences)}")
            audio_count = 0
            missing_audio = []
            scene_ids = []
            for scene in chunk_scene_data_with_timestamps:
                scene_id = scene.get('scene_id')
                scene_ids.append(scene_id)
                audio_file = scene.get('audio_file')
                if audio_file and os.path.exists(audio_file):
                    audio_count += 1
                    if audio_count <= 5:  # 처음 5개만 상세 로그
                        size = os.path.getsize(audio_file)
                        print(f"  scene_{scene_id}: ✅ 존재 ({size} bytes) - {audio_file}")
                else:
                    missing_audio.append(scene_id)
                    print(f"  scene_{scene_id}: ❌ 없음 - 경로: {audio_file}")
                    # 실제 파일 시스템에서 확인
                    expected_path = os.path.join(chunk_folder, f"scene_{scene_id}_audio.mp3")
                    if os.path.exists(expected_path):
                        print(f"    -> 하지만 예상 경로에는 존재: {expected_path}")
            print(f"  총 {len(chunk_scene_data_with_timestamps)}개 중 {audio_count}개 오디오 파일 존재")
            print(f"  scene_id 범위: {min(scene_ids) if scene_ids else 'N/A'} ~ {max(scene_ids) if scene_ids else 'N/A'}")
            if missing_audio:
                print(f"  [경고] 누락된 오디오: scene_{missing_audio[:10]}...")
            print()
            
            # 청크별 비디오 생성
            chunk_video_file = os.path.join(chunk_folder, f"chunk_{chunk_idx}_video.mp4")
            chunk_subtitle_file = os.path.join(chunk_folder, f"chunk_{chunk_idx}_subtitles.srt")
            update_job(job_id, stage_progress=2, current_stage=f"영상 합성 (청크 {chunk_idx}/{total_chunks})")
            chunk_success = create_video(
                chunk_scene_data_with_timestamps,
                char_limit,
                chunk_subtitle_file,
                chunk_video_file,
                chunk_folder,
                progress_cb=lambda msg: progress(f"[청크 {chunk_idx}] {msg}"),
                include_subtitles=include_subtitles,
            )
            if not chunk_success:
                raise RuntimeError(f"청크 {chunk_idx} 영상 합성에 실패했습니다.")
            
            # 청크 비디오 파일 존재 확인
            if not os.path.exists(chunk_video_file):
                raise RuntimeError(f"청크 {chunk_idx} 비디오 파일이 생성되지 않았습니다: {chunk_video_file}")
            else:
                file_size = os.path.getsize(chunk_video_file)
                print(f"[확인] 청크 {chunk_idx} 비디오 파일 생성됨: {chunk_video_file} ({file_size} bytes)")
            
            chunk_video_files.append(chunk_video_file)
            all_scene_data.extend(chunk_scene_data_with_timestamps)
            sentence_offset += len(chunk_sentences)
            
            print(f"[확인] 청크 {chunk_idx} 완료. chunk_video_files 리스트에 추가됨. 현재 리스트 길이: {len(chunk_video_files)}")
            print(f"[확인] 다음 청크의 sentence_offset: {sentence_offset}")
            
            # 청크별 비디오 파일의 오디오 트랙 빠른 체크
            print(f"\n[체크] 청크 {chunk_idx} 비디오 파일 오디오 트랙 확인:")
            try:
                if not os.path.exists(chunk_video_file):
                    print(f"  ❌ 비디오 파일이 존재하지 않음: {chunk_video_file}")
                else:
                    probe = ffmpeg.probe(chunk_video_file)
                    audio_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
                    video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
                    if audio_streams:
                        duration = float(audio_streams[0].get('duration', 0))
                        bitrate = audio_streams[0].get('bit_rate', 'N/A')
                        print(f"  ✅ 오디오 트랙 존재 (길이: {duration:.2f}초, 비트레이트: {bitrate})")
                        print(f"  비디오 스트림: {len(video_streams)}개")
                    else:
                        print(f"  ❌ 오디오 트랙이 없습니다! (비디오 스트림: {len(video_streams)}개)")
                        print(f"  파일: {chunk_video_file}")
            except Exception as e:
                print(f"  ❌ 오디오 트랙 확인 실패: {e}")
                import traceback
                traceback.print_exc()
            print()
            
            progress(f"청크 {chunk_idx}/{total_chunks} 완료")
        
        # 모든 청크가 완료되면 최종 비디오 합치기
        if total_chunks > 1:
            progress(f"모든 청크 완료. 최종 비디오 합치는 중...")
            update_job(job_id, stage_progress=2, current_stage="최종 비디오 합치기")
            
            # 청크 비디오 파일 리스트 확인
            print(f"\n[최종 체크 전] chunk_video_files 리스트 확인:")
            print(f"  총 청크 수: {total_chunks}")
            print(f"  chunk_video_files 리스트 길이: {len(chunk_video_files)}")
            for idx, video_file in enumerate(chunk_video_files, 1):
                exists = os.path.exists(video_file)
                size = os.path.getsize(video_file) if exists else 0
                print(f"  청크 {idx}: {'✅' if exists else '❌'} {video_file} ({size} bytes)")
            print()
            
            # 모든 청크의 오디오 트랙 최종 체크
            print(f"\n[최종 체크] {total_chunks}개 청크의 오디오 트랙 확인:")
            total_audio_duration = 0.0
            for idx, chunk_file in enumerate(chunk_video_files, 1):
                try:
                    if not os.path.exists(chunk_file):
                        print(f"  청크 {idx}: ❌ 비디오 파일이 존재하지 않음: {chunk_file}")
                        continue
                    probe = ffmpeg.probe(chunk_file)
                    audio_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
                    video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
                    if audio_streams:
                        duration = float(audio_streams[0].get('duration', 0))
                        total_audio_duration += duration
                        bitrate = audio_streams[0].get('bit_rate', 'N/A')
                        print(f"  청크 {idx}: ✅ 오디오 있음 (길이: {duration:.2f}초, 비트레이트: {bitrate}, 비디오 스트림: {len(video_streams)}개)")
                        print(f"    파일: {chunk_file}")
                    else:
                        print(f"  청크 {idx}: ❌ 오디오 없음! (비디오 스트림: {len(video_streams)}개)")
                        print(f"    파일: {chunk_file}")
                except Exception as e:
                    print(f"  청크 {idx}: ❌ 확인 실패: {e}")
                    import traceback
                    traceback.print_exc()
            print(f"  총 예상 오디오 길이: {total_audio_duration:.2f}초")
            print()
            
            # FFmpeg로 비디오 합치기 (비디오와 오디오를 별도로 합침)
            try:
                print(f"\n[비디오 합치기 시작] {len(chunk_video_files)}개 청크 합치는 중...")
                
                # 각 청크의 비디오와 오디오를 별도로 추출
                video_streams = []
                audio_streams = []
                
                for idx, chunk_file in enumerate(chunk_video_files, 1):
                    chunk_input = ffmpeg.input(chunk_file)
                    # 비디오 스트림 추출
                    video_streams.append(chunk_input['v'])
                    # 오디오 스트림 추출 (없으면 무음 오디오 생성)
                    try:
                        # 오디오 스트림이 있는지 확인
                        probe = ffmpeg.probe(chunk_file)
                        has_audio = any(s.get('codec_type') == 'audio' for s in probe.get('streams', []))
                        if has_audio:
                            audio_streams.append(chunk_input['a'])
                            print(f"  청크 {idx}: 비디오 + 오디오")
                        else:
                            # 오디오가 없으면 비디오 길이만큼 무음 오디오 생성
                            video_duration = float([s for s in probe.get('streams', []) if s.get('codec_type') == 'video'][0].get('duration', 0))
                            silent_audio = ffmpeg.input('anullsrc=channel_layout=mono:sample_rate=44100', f='lavfi', t=video_duration)
                            audio_streams.append(silent_audio)
                            print(f"  청크 {idx}: 비디오만 (무음 오디오 추가)")
                    except Exception as e:
                        print(f"  청크 {idx}: 오디오 확인 실패, 무음 오디오 사용: {e}")
                        # 오류 시 기본 무음 오디오 생성 (10초)
                        silent_audio = ffmpeg.input('anullsrc=channel_layout=mono:sample_rate=44100', f='lavfi', t=10)
                        audio_streams.append(silent_audio)
                
                # 비디오와 오디오를 각각 합침
                if len(video_streams) > 1:
                    concatenated_video = ffmpeg.concat(*video_streams, v=1, a=0)
                else:
                    concatenated_video = video_streams[0]
                
                if len(audio_streams) > 1:
                    concatenated_audio = ffmpeg.concat(*audio_streams, v=0, a=1)
                else:
                    concatenated_audio = audio_streams[0]
                
                # 임시 파일로 먼저 생성 (입력 파일과 겹치지 않도록)
                temp_video_file = os.path.join(assets_folder, f"temp_final_video_{job_id}.mp4")
                print(f"[비디오 합치기] 임시 파일로 생성: {temp_video_file}")
                print(f"[비디오 합치기] 최종 파일 경로: {video_file}")
                
                # 입력 파일과 출력 파일이 겹치는지 확인
                for idx, chunk_file in enumerate(chunk_video_files, 1):
                    abs_chunk = os.path.abspath(chunk_file)
                    abs_temp = os.path.abspath(temp_video_file)
                    abs_final = os.path.abspath(video_file)
                    if abs_chunk == abs_temp or abs_chunk == abs_final:
                        print(f"  ⚠️ 경고: 청크 {idx} 파일이 출력 파일과 겹칩니다!")
                        print(f"    청크 파일: {abs_chunk}")
                        print(f"    임시 파일: {abs_temp}")
                        print(f"    최종 파일: {abs_final}")
                
                # FFmpeg 최적화: 빠른 합치기를 위한 설정
                output = ffmpeg.output(
                    concatenated_video,
                    concatenated_audio,
                    temp_video_file, 
                    vcodec='libx264', 
                    acodec='aac', 
                    pix_fmt='yuv420p',
                    **{
                        "preset": "fast",
                        "crf": "23",
                        "threads": "0",
                        "movflags": "+faststart",
                    }
                )
                try:
                    ffmpeg.run(output, overwrite_output=True, quiet=False)
                    # 임시 파일을 최종 경로로 이동
                    import shutil
                    if os.path.exists(temp_video_file):
                        # video_file이 올바른 경로인지 다시 확인
                        expected_final_path = os.path.join(STATIC_FOLDER, f"{FINAL_VIDEO_BASE_NAME}_{job_id}.mp4")
                        if video_file != expected_final_path:
                            print(f"[경고] video_file이 덮어씌워졌습니다! 올바른 경로로 재설정합니다.")
                            print(f"  잘못된 경로: {video_file}")
                            print(f"  올바른 경로: {expected_final_path}")
                            video_file = expected_final_path
                        # 최종 경로로 이동
                        shutil.move(temp_video_file, video_file)
                        print(f"[비디오 합치기] 임시 파일을 최종 경로로 이동 완료")
                        print(f"[비디오 합치기] 최종 파일 경로: {video_file}")
                    progress("최종 비디오 합치기 완료")
                    print(f"[비디오 합치기 완료] {video_file}")
                except ffmpeg.Error as exc:
                    print(f"[FFmpeg 오류] 최종 비디오 합치기 실패:")
                    if exc.stderr:
                        print(exc.stderr.decode('utf-8'))
                    # 임시 파일 정리
                    if os.path.exists(temp_video_file):
                        try:
                            os.remove(temp_video_file)
                        except:
                            pass
                    raise
                
                # 최종 비디오의 오디오 트랙 확인
                if os.path.exists(video_file):
                    try:
                        final_probe = ffmpeg.probe(video_file)
                        final_audio = [s for s in final_probe.get('streams', []) if s.get('codec_type') == 'audio']
                        final_video = [s for s in final_probe.get('streams', []) if s.get('codec_type') == 'video']
                        if final_audio:
                            final_duration = float(final_audio[0].get('duration', 0))
                            print(f"\n[최종 비디오 체크] 오디오 길이: {final_duration:.2f}초 (예상: {total_audio_duration:.2f}초)")
                            if abs(final_duration - total_audio_duration) > 1.0:
                                print(f"  ⚠️ 경고: 최종 오디오 길이가 예상과 다릅니다! (차이: {abs(final_duration - total_audio_duration):.2f}초)")
                        else:
                            print(f"\n[최종 비디오 체크] ❌ 오디오 트랙이 없습니다!")
                    except Exception as e:
                        print(f"\n[최종 비디오 체크] 확인 실패: {e}")
            except Exception as e:
                print(f"[경고] 비디오 합치기 실패: {e}")
                import traceback
                traceback.print_exc()
                # 합치기 실패 시 첫 번째 청크 비디오를 사용
                import shutil
                shutil.copy2(chunk_video_files[0], video_file)
                progress("비디오 합치기 실패, 첫 번째 청크를 사용합니다.")
        else:
            # 청크가 하나면 그대로 사용
            import shutil
            shutil.copy2(chunk_video_files[0], video_file)

        # 전체 SRT 파일 생성 (청크별 SRT 파일을 시간 조정하여 합치기)
        if total_chunks > 1:
            progress("전체 자막 파일 생성 중...")
            try:
                # SRT 시간 문자열을 초로 변환하는 헬퍼 함수
                def srt_time_to_seconds(srt_time: str) -> float:
                        """SRT 시간 형식 (HH:MM:SS,mmm)을 초로 변환"""
                        time_part, ms_part = srt_time.split(',')
                        h, m, s = map(int, time_part.split(':'))
                        ms = int(ms_part)
                        return h * 3600 + m * 60 + s + ms / 1000.0
                
                # 각 청크의 비디오 길이 계산 (시간 오프셋 계산용)
                chunk_durations = []
                for chunk_idx in range(1, total_chunks + 1):
                    chunk_video_file = os.path.join(assets_folder, f"chunk_{chunk_idx}", f"chunk_{chunk_idx}_video.mp4")
                    if os.path.exists(chunk_video_file):
                        try:
                            probe = ffmpeg.probe(chunk_video_file)
                            video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
                            if video_streams:
                                duration = float(video_streams[0].get('duration', 0))
                                chunk_durations.append(duration)
                                print(f"[SRT 합치기] 청크 {chunk_idx} 길이: {duration:.2f}초")
                            else:
                                chunk_durations.append(0.0)
                        except Exception as e:
                            print(f"[경고] 청크 {chunk_idx} 길이 확인 실패: {e}")
                            chunk_durations.append(0.0)
                    else:
                        chunk_durations.append(0.0)
                
                # 각 청크의 누적 시간 오프셋 계산
                chunk_offsets = [0.0]
                for i in range(1, total_chunks):
                    chunk_offsets.append(chunk_offsets[i-1] + chunk_durations[i-1])
                
                print(f"[SRT 합치기] 청크 오프셋: {chunk_offsets}")
                
                # 전체 SRT 파일 생성
                subtitle_index = 1
                with open(subtitle_file, "w", encoding="utf-8") as f:
                    for chunk_idx in range(1, total_chunks + 1):
                        chunk_subtitle_file = os.path.join(assets_folder, f"chunk_{chunk_idx}", f"chunk_{chunk_idx}_subtitles.srt")
                        time_offset = chunk_offsets[chunk_idx - 1]
                        
                        if not os.path.exists(chunk_subtitle_file):
                            print(f"[경고] 청크 {chunk_idx} SRT 파일이 없습니다: {chunk_subtitle_file}")
                            continue
                        
                        print(f"[SRT 합치기] 청크 {chunk_idx} SRT 파일 읽는 중... (오프셋: {time_offset:.2f}초)")
                        
                        try:
                            with open(chunk_subtitle_file, "r", encoding="utf-8") as chunk_f:
                                lines = chunk_f.readlines()
                            
                            i = 0
                            while i < len(lines):
                                line = lines[i].strip()
                                if not line:
                                    i += 1
                                    continue
                                
                                # 자막 인덱스 (무시하고 새로 할당)
                                try:
                                    int(line)
                                except ValueError:
                                    i += 1
                                    continue
                                
                                # 시간 정보
                                if i + 1 >= len(lines):
                                    break
                                time_line = lines[i + 1].strip()
                                if '-->' not in time_line:
                                    i += 1
                                    continue
                                
                                start_time_str, end_time_str = time_line.split('-->')
                                start_time = srt_time_to_seconds(start_time_str.strip()) + time_offset
                                end_time = srt_time_to_seconds(end_time_str.strip()) + time_offset
                                
                                # 자막 텍스트 수집
                                subtitle_text_lines = []
                                i += 2
                                while i < len(lines) and lines[i].strip():
                                    subtitle_text_lines.append(lines[i].rstrip())
                                    i += 1
                                
                                if subtitle_text_lines:
                                    f.write(f"{subtitle_index}\n")
                                    f.write(f"{seconds_to_srt_time(start_time)} --> {seconds_to_srt_time(end_time)}\n")
                                    f.write("\n".join(subtitle_text_lines) + "\n")
                                    f.write("\n")
                                    subtitle_index += 1
                                
                                i += 1  # 빈 줄 건너뛰기
                        except Exception as chunk_exc:
                            print(f"[경고] 청크 {chunk_idx} SRT 파일 읽기 실패: {chunk_exc}")
                            import traceback
                            traceback.print_exc()
                            continue
                
                if os.path.exists(subtitle_file):
                    srt_size = os.path.getsize(subtitle_file)
                    print(f"[완료] 전체 SRT 파일 생성됨: {subtitle_file} (크기: {srt_size} bytes, 총 {subtitle_index - 1}개 자막)")
                else:
                    print(f"[경고] SRT 파일이 생성되지 않았습니다: {subtitle_file}")
            except Exception as srt_exc:
                print(f"[경고] 전체 SRT 파일 생성 실패: {srt_exc}")
                import traceback
                traceback.print_exc()
        elif total_chunks == 1:
            # 청크가 하나면 해당 청크의 SRT 파일을 복사
            progress("자막 파일 복사 중...")
            try:
                chunk_subtitle_file = os.path.join(assets_folder, "chunk_1", "chunk_1_subtitles.srt")
                if os.path.exists(chunk_subtitle_file):
                    import shutil
                    shutil.copy2(chunk_subtitle_file, subtitle_file)
                    print(f"[완료] SRT 파일 복사됨: {subtitle_file}")
                else:
                    print(f"[경고] 청크 1 SRT 파일이 없습니다: {chunk_subtitle_file}")
            except Exception as srt_exc:
                print(f"[경고] SRT 파일 복사 실패: {srt_exc}")
                import traceback
                traceback.print_exc()
        
        # 파일이 실제로 존재하는지 확인하고, 존재할 때만 video_filename 설정
        # video_file은 get_job_paths에서 이미 올바른 경로로 설정되어 있어야 함
        # (STATIC_FOLDER/final_video_{job_id}.mp4)
        print(f"[작업 완료 확인] video_file 경로: {video_file}")
        print(f"[작업 완료 확인] STATIC_FOLDER: {STATIC_FOLDER}")
        print(f"[작업 완료 확인] job_id: {job_id}")
        
        # 비디오 파일 존재 여부 확인 (최대 5초 대기)
        max_wait = 5
        wait_count = 0
        while not os.path.exists(video_file) and wait_count < max_wait:
            print(f"[작업 완료 확인] 비디오 파일 대기 중... ({wait_count + 1}/{max_wait})")
            time.sleep(1)
            wait_count += 1
        
        if os.path.exists(video_file):
            file_size = os.path.getsize(video_file)
            # video_filename은 STATIC_FOLDER를 기준으로 한 상대 경로로 저장
            # 예: final_video_{job_id}.mp4
            video_filename = os.path.relpath(video_file, STATIC_FOLDER)
            # Windows 경로 구분자를 /로 통일
            video_filename = video_filename.replace(os.sep, '/')
            print(f"[완료] 최종 비디오 파일 생성됨: {video_file} (크기: {file_size} bytes)")
            print(f"[완료] 다운로드 파일명 (상대 경로): {video_filename}")
            # 최종 비디오가 완전히 생성되었을 때만 video_filename 설정
            update_job(job_id, status="completed", video_filename=video_filename, stage_progress=3, current_stage="완료")
        else:
            print(f"[경고] 비디오 파일이 존재하지 않습니다: {video_file}")
            print(f"[경고] video_file 경로 확인: {video_file}")
            print(f"[경고] STATIC_FOLDER: {STATIC_FOLDER}")
            print(f"[경고] job_id: {job_id}")
            # 파일이 없으면 video_filename을 None으로 설정
            update_job(job_id, status="error", error="최종 비디오 파일이 생성되지 않았습니다.", video_filename=None)
    except Exception as exc:
            print("=" * 80)
            print("=== [DEBUG] 오류: 영상 생성 작업 예외 발생 ===")
            print(f"=== [DEBUG] 예외 내용: {exc} ===")
            import traceback
            traceback.print_exc()
            print("=" * 80)
            print(f"[작업 오류] {exc}")
            update_job(job_id, status="error", error=str(exc))
            progress("작업 처리 중 오류가 발생했습니다.")


# =============================================================================
# Flask 라우트
# =============================================================================


def render_home(video_filename=None):
    voices = get_available_voices()
    return render_template("index.html", video_filename=video_filename, voices=voices, default_voice_id=get_default_voice_id())


@app.route("/")
def index():
    return render_home()


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        settings = config_manager.get_all()
        # 다운로드 폴더 경로가 없으면 기본 다운로드 폴더로 설정
        if not settings.get("download_folder_path"):
            default_folder = get_default_download_folder()
            if default_folder:
                settings["download_folder_path"] = default_folder
                config_manager.set("download_folder_path", default_folder)
        return jsonify(settings)

    data = request.get_json(silent=True) or {}
    config_manager.update(
        {
            "replicate_api_key": (data.get("replicate_api_key") or "").strip(),
            "elevenlabs_api_key": (data.get("elevenlabs_api_key") or "").strip(),
            "gemini_api_key": (data.get("gemini_api_key") or "").strip(),
            "download_folder_path": (data.get("download_folder_path") or "").strip(),
        }
    )
    reload_api_keys()
    refresh_service_flags()
    
    # 디버깅: 저장된 설정 확인
    saved_settings = config_manager.get_all()
    print(f"[설정 저장] Gemini API 키 저장 확인: {bool(saved_settings.get('gemini_api_key'))} (길이: {len(saved_settings.get('gemini_api_key', ''))})")
    print(f"[설정 저장] GEMINI_API_KEY 전역 변수: {bool(GEMINI_API_KEY)} (길이: {len(GEMINI_API_KEY)})")
    print(f"[설정 저장] gemini_available: {gemini_available}")
    
    global _cached_voice_list
    _cached_voice_list = None
    return jsonify({"status": "ok"})


@app.route("/api/voices", methods=["GET", "POST", "DELETE"])
def api_voices():
    """보이스 ID 관리 API"""
    global _cached_voice_list  # 함수 시작 부분에 global 선언
    
    if request.method == "GET":
        # 현재 사용 가능한 보이스 목록 반환
        voices = get_available_voices()
        custom_voice_ids = config_manager.get("custom_voice_ids", [])
        if not isinstance(custom_voice_ids, list):
            custom_voice_ids = []
        return jsonify({
            "voices": voices,
            "custom_voice_ids": custom_voice_ids
        })
    
    elif request.method == "POST":
        # 보이스 ID 추가
        data = request.get_json(silent=True) or {}
        voice_id = (data.get("voice_id") or "").strip()
        
        if not voice_id:
            return jsonify({"error": "보이스 ID가 필요합니다."}), 400
        
        # 중복 확인
        custom_voice_ids = config_manager.get("custom_voice_ids", [])
        if not isinstance(custom_voice_ids, list):
            custom_voice_ids = []
        
        if voice_id in custom_voice_ids:
            return jsonify({"error": "이미 추가된 보이스 ID입니다."}), 400
        
        # 기본 보이스 ID와 중복 확인
        if voice_id in ELEVENLABS_VOICE_IDS:
            return jsonify({"error": "기본 보이스 ID입니다."}), 400
        
        # 추가
        custom_voice_ids.append(voice_id)
        config_manager.set("custom_voice_ids", custom_voice_ids)
        
        # 캐시 무효화
        _cached_voice_list = None
        
        return jsonify({"status": "ok", "message": "보이스 ID가 추가되었습니다."})
    
    elif request.method == "DELETE":
        # 보이스 ID 삭제
        voice_id = request.args.get("voice_id", "").strip()
        
        if not voice_id:
            return jsonify({"error": "보이스 ID가 필요합니다."}), 400
        
        custom_voice_ids = config_manager.get("custom_voice_ids", [])
        if not isinstance(custom_voice_ids, list):
            custom_voice_ids = []
        
        if voice_id not in custom_voice_ids:
            return jsonify({"error": "사용자 정의 보이스 ID가 아닙니다."}), 400
        
        # 삭제
        custom_voice_ids.remove(voice_id)
        config_manager.set("custom_voice_ids", custom_voice_ids)
        
        # 캐시 무효화
        _cached_voice_list = None
        
        return jsonify({"status": "ok", "message": "보이스 ID가 삭제되었습니다."})


@app.route("/api/select_download_folder", methods=["POST"])
def select_download_folder():
    """폴더 선택 다이얼로그를 열고 선택한 폴더 경로 반환"""
    try:
        # 현재 설정된 경로 또는 기본 다운로드 폴더를 초기 경로로 사용
        current_path = config_manager.get("download_folder_path", "").strip()
        if not current_path or not os.path.isdir(current_path):
            current_path = get_default_download_folder()
        
        selected_folder = None
        error_message = None
        
        # tkinter를 사용한 폴더 선택 (webview 환경에서도 작동)
        if TKINTER_AVAILABLE:
            selected_folder = [None]
            error_message = [None]
            dialog_complete = threading.Event()
            
            def select_folder():
                try:
                    # 루트 윈도우 생성 (숨김)
                    root = tk.Tk()
                    root.withdraw()  # 메인 윈도우 숨기기
                    
                    # macOS에서 focus 관련 문제 방지
                    try:
                        root.attributes('-topmost', True)
                        root.lift()
                    except:
                        pass
                    
                    # 폴더 선택 다이얼로그 열기
                    folder = filedialog.askdirectory(
                        title="다운로드 폴더 선택",
                        initialdir=current_path if current_path else None,
                        parent=root
                    )
                    
                    selected_folder[0] = folder if folder else None
                    
                    # 윈도우 정리
                    try:
                        root.quit()
                    except:
                        pass
                    try:
                        root.destroy()
                    except:
                        pass
                except Exception as e:
                    error_message[0] = str(e)
                    import traceback
                    print(f"[폴더 선택 내부 오류] {e}")
                    traceback.print_exc()
                    selected_folder[0] = None
                finally:
                    dialog_complete.set()
            
            # GUI 스레드에서 실행
            thread = threading.Thread(target=select_folder, daemon=False)
            thread.start()
            
            # 다이얼로그가 완료될 때까지 대기 (최대 60초)
            if not dialog_complete.wait(timeout=60):
                return jsonify({"error": "폴더 선택 시간이 초과되었습니다."}), 500
            
            thread.join(timeout=5)  # 스레드 종료 대기
            
            if error_message[0]:
                print(f"[폴더 선택 오류] {error_message[0]}")
                return jsonify({"error": f"폴더 선택 중 오류: {error_message[0]}"}), 500
            
            selected_folder = selected_folder[0]
        else:
            return jsonify({"error": "폴더 선택 기능을 사용할 수 없습니다. (tkinter 없음)"}), 500
        
        if selected_folder:
            # 선택한 폴더 경로 저장
            config_manager.set("download_folder_path", selected_folder)
            return jsonify({"folder_path": selected_folder})
        else:
            return jsonify({"error": "폴더가 선택되지 않았습니다."}), 400
            
    except Exception as e:
        print(f"[폴더 선택 오류] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"폴더 선택 실패: {str(e)}"}), 500


@app.route("/api/generate_draft_script", methods=["POST"])
def api_generate_draft_script():
    """검수 대본 생성 API"""
    global genai_client  # 전역 변수 사용 선언
    try:
        data = request.get_json(silent=True) or {}
        topic = (data.get("topic") or "").strip()
        
        if not topic:
            return jsonify({"error": "주제/키워드를 입력해주세요."}), 400
        
        # 디버깅: API 키 상태 확인
        print(f"[검수 대본 생성] gemini_available: {gemini_available}, GEMINI_API_KEY: {'있음' if GEMINI_API_KEY else '없음'} (길이: {len(GEMINI_API_KEY)})")
        
        # GEMINI_API_KEY가 있으면 사용 (패키지가 없어도 시도)
        if not GEMINI_API_KEY:
            error_msg = f"Gemini API 키가 설정되지 않았습니다. (GEMINI_API_KEY 길이: {len(GEMINI_API_KEY)})"
            print(f"[검수 대본 생성] {error_msg}")
            return jsonify({"error": "Gemini API 키가 설정되지 않았습니다. 설정에서 Gemini API 키를 입력해주세요."}), 500
        
        # 새로운 라이브러리: genai_client 확인 및 재초기화
        if genai_client is None:
            try:
                from google import genai
                genai_client = genai.Client(api_key=GEMINI_API_KEY)
                print(f"[검수 대본 생성] google.genai Client 재초기화 성공")
            except ImportError as e:
                error_msg = f"google-genai 패키지가 설치되지 않았습니다. pip install google-genai를 실행해주세요. ({e})"
                print(f"[검수 대본 생성] {error_msg}")
                return jsonify({"error": "google-genai 패키지가 설치되지 않았습니다. 패키지를 설치한 후 다시 시도해주세요."}), 500
            except Exception as e:
                error_msg = f"Gemini API Client 초기화 실패: {e}"
                print(f"[검수 대본 생성] {error_msg}")
                return jsonify({"error": f"Gemini API 초기화 실패: {str(e)}"}), 500
        
        # 검수 대본 프롬프트
        system_prompt = """당신은 '지식 스토리텔러' 전문 유튜브 대본 작가입니다.

[채널 정체성]
채널명 (가상): 지식의 소용돌이
주제: 역사, 전쟁사, 미스터리, 흥미로운 경제/사회 이야기
타깃: 깊이 있고 흥미로운 지식 콘텐츠를 선호하는 한국인 성인 시청자
핵심 목표: 단순한 정보 나열이 아닌, 강력한 스토리텔링을 통해 시청자의 감정을 자극하고 시청 지속 시간을 극대화하는 것.

[작가의 역할]
사용자가 [주제]를 제시하면, 이 주제를 바탕으로 한 편의 완결된 유튜브 대본을 작성합니다.
가장 중요한 임무는 '시청자 이탈 방지'입니다. 시청자가 영상을 끄지 않도록 모든 문장에 훅(Hook)을 심어야 합니다.

[대본 작성 시 절대 준수 사항]

1. 총 분량 (필수):
공백 포함 최소 8,000자 이상으로 작성해야 합니다. 이는 깊이 있는 서사를 위한 필수 조건입니다. (컴필레이션/목록형 구조에서도 각 항목을 상세히 다루어 이 분량을 충족해야 합니다.)

2. 한국인 정서에 맞는 톤앤매너 (몰입감 강화):
딱딱한 설명문이 아닌, 친근하면서도 권위 있는 '이야기꾼'의 구어체로 작성합니다. 청자가 바로 눈앞에 있는 것처럼 생생하게 말해야 합니다. (예: "자, 상상해 보세요.", "정말 어처구니가 없죠.", "이제부터가 본 게임입니다.")
벤치마킹 자료(서울 방어력)처럼, 청자의 이해를 돕는 극도로 생생하고 직관적인 비유를 사용해야 합니다. (예: "스탈린그라드 싸대기 갈길 정도의 아파트 지옥", "서울로 치면 그냥 좁은 개천 수준이거든요.")
적절한 비유, 감정 이입을 유도하는 표현, 그리고 시청자에게 질문을 던지는 rhetorical question(수사적 질문)을 적극 사용합니다. (예: "정말 그랬을까요?", "만약 당신이라면 어땠을 것 같나요?")

3. 유연한 콘텐츠 구조 (핵심 개선 사항):
주제에 가장 적합한 방식을 스스로 판단하여, 아래 3가지 구조 중 하나를 선택해 대본을 구성합니다.

[Type A: 단일 서사 심층 분석형] (예: 돌고래 피터 이야기)
하나의 사건, 인물, 또는 이야기를 기승전결에 따라 깊게 파고듭니다.
A. 오프닝 (최소 1,000자): [강력한 후킹과 문제 제기] 가장 충격적이거나 감성적인 사실을 먼저 제시하며 호기심을 극대화합니다.
B. 본론 (전개/위기): [스토리텔링 및 심화] 사건의 배경, 인물의 갈등, 극적인 순간을 생생하게 묘사합니다.
C. 본론 (절정/결말): [미스터리 해결 및 반전] 오프닝의 질문을 해결하고, 예상치 못한 반전이나 새로운 관점을 제시합니다.
D. 아웃트로 (최소 500자): [요약 및 감정적 울림] 전체 이야기를 요약하고, 오늘날 우리에게 주는 의미를 연결하며 여운을 남깁니다.

[Type B: 컴필레이션 / 목록형] (예: 지구의 미스터리, 화해 불가능한 국가 TOP 5)
하나의 대주제 아래 여러 개의 개별 사례(3~5개)를 엮어 제시합니다.
A. 오프닝: [대주제 제시 및 강력한 후킹] "왜 이 주제가 중요한지", "왜 이 목록을 끝까지 봐야 하는지" 강력한 호기심을 자극합니다.
B. 본론 (개별 사례 나열): 각 사례(소주제)를 하나씩 소개합니다. 각 소주제는 그 자체로 **'미니 기승전결'**을 가져야 합니다. (도입부의 미스터리 제기 -> 전개 -> 결말 또는 의문점)
C. 전환 (Transition): 한 사례가 끝나고 다음 사례로 넘어갈 때, 시청자가 이탈하지 않도록 자연스러우면서도 긴장감을 유지하는 연결 멘트를 사용합니다.
D. 아웃트로: [종합 요약 및 CTA] 전체 사례를 관통하는 핵심 메시지(가 있다면)를 요약하고, 구독 및 다음 영상 예고로 연결합니다.

[Type C: 주제 다각도 탐구형] (예: 서울의 방어력)
하나의 대상을 여러 '관점'이나 '하위 주제'로 나누어 심층 분석합니다. (예: 서울의 '북쪽 방어선', '도심 방어선', '한강 방어선')
A. 오프닝: [탐구 대상의 중요성 부각] 분석할 대상이 왜 흥미롭고 중요한지(예: 미군도 점령 못하는 서울)를 제시하며 거대한 질문을 던집니다.
B. 본론 (관점별 분석): 주제를 논리적인 순서(예: 지리적, 시간적, 기능적 순서)로 나누어 하나씩 깊게 파고듭니다. 각 파트가 끝날 때마다 "그럼 이게 끝일까요? 어림도 없죠."처럼 다음 파트에 대한 기대감을 심어줍니다.
C. 결론: [종합 및 의의] 모든 분석을 종합하여 오프닝에서 던진 질문에 대한 답을 제시하고, 주제의 현재적 의의를 짚어줍니다.
D. 아웃트로: [최종 요약 및 CTA]

[최종 산출물 규칙] TTS 전용 '순수 대본'만 출력
이것은 가장 중요한 규칙입니다. 당신의 **최종 응답(Final Output)**은 사용자가 TTS(텍스트 음성 변환) 프로그램에 '그대로 복사하여 붙여넣기' 할 수 있는 순수 내레이션 텍스트여야 합니다.
당신의 내부 로직에서 고려했던 [BGM: ...], [자료: ...], (잠시 쉼), (강조), [내레이션] 태그 등 모든 지시어, 괄호, 설명문을 완벽하게 제거해야 합니다.
최종 결과물은 오직 '말하는 부분(대사)'의 알맹이 텍스트만으로 구성되어야 합니다. 그 외의 어떤 것도 포함해서는 안 됩니다.

대본은 각 챕터로 나눠서 작성하되, 챕터 구분은 "=== 챕터 1: [제목] ===" 형식으로 명시하세요."""

        user_prompt = f"[주제]: {topic}"
        
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # 새로운 라이브러리 방식
            if genai_client is None:
                return jsonify({"error": "Gemini API Client가 초기화되지 않았습니다."}), 500
            
            from google.genai import types
            target_model = 'gemini-2.5-flash'
            # target_model = 'gemini-3-pro-preview'  # 3 Pro를 사용하려면 주석 해제
            
            response = genai_client.models.generate_content(
                model=target_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=8000,
                )
            )
            
            script = response.text.strip()
            
            # 새로운 라이브러리는 safety_ratings를 다르게 제공할 수 있음
            # response 객체의 속성 확인
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    return jsonify({
                        "error": "콘텐츠 정책 위반으로 인해 대본 생성이 차단되었습니다. 다른 주제로 시도해주세요."
                    }), 400
        
        except Exception as e:
            print(f"[검수 대본 생성] Gemini API 오류: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Gemini API 오류: {str(e)}"}), 500
        
        # content가 None이거나 빈 문자열인 경우
        if not script:
            error_msg = "대본 생성에 실패했습니다."
            print(f"[검수 대본 생성] 응답 없음")
            return jsonify({"error": error_msg}), 500
        
        return jsonify({"script": script})
        
    except Exception as e:
        print(f"[대본 생성 오류] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"대본 생성 중 오류가 발생했습니다: {str(e)}"}), 500


def split_draft_script_into_chapters(draft_script: str) -> List[Dict[str, str]]:
    """검수 대본을 챕터별로 분리합니다."""
    import re
    chapters = []
    
    # 챕터 구분자 패턴: === 챕터 1: [제목] === 또는 === 챕터 1 === 등
    # 더 유연한 패턴: ===, ====, 챕터, Chapter 등 다양한 형식 지원
    chapter_pattern = r'={2,}\s*챕터\s*(\d+)[:\s]*(.*?)\s*={2,}'
    
    # 챕터 구분자 찾기
    matches = list(re.finditer(chapter_pattern, draft_script, flags=re.IGNORECASE))
    
    if not matches:
        # 챕터 구분자가 없으면 전체를 하나의 챕터로 처리
        return [{"chapter_num": 1, "title": "전체", "content": draft_script.strip()}]
    
    # 첫 번째 챕터 이전 내용이 있으면 추가
    first_match = matches[0]
    if first_match.start() > 0:
        pre_content = draft_script[:first_match.start()].strip()
        if pre_content:
            chapters.append({
                "chapter_num": 0,
                "title": "서문",
                "content": pre_content
            })
    
    # 각 챕터 추출
    for i, match in enumerate(matches):
        chapter_num = int(match.group(1))
        chapter_title = match.group(2).strip() if match.group(2) else f"챕터 {chapter_num}"
        
        # 다음 챕터 시작 위치 찾기
        if i + 1 < len(matches):
            next_match = matches[i + 1]
            chapter_content = draft_script[match.end():next_match.start()].strip()
        else:
            # 마지막 챕터
            chapter_content = draft_script[match.end():].strip()
        
        if chapter_content:
            chapters.append({
                "chapter_num": chapter_num,
                "title": chapter_title,
                "content": chapter_content
            })
    
    # 챕터가 없으면 전체를 하나로
    if not chapters:
        chapters.append({
            "chapter_num": 1,
            "title": "전체",
            "content": draft_script.strip()
        })
    
    return chapters


def run_final_script_job(job_id: str, draft_script: str, grouping_size: int = 1):
    """백그라운드에서 실행될 최종 대본 생성 작업"""
    global genai_client
    try:
        update_job(job_id, status="running", current_stage="대본 분석 및 생성 중...", stage_progress=5, stage_total=100)
        
        # Gemini 클라이언트 확인 및 재초기화
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API 키가 설정되지 않았습니다.")
        
        if genai_client is None:
            try:
                from google import genai
                genai_client = genai.Client(api_key=GEMINI_API_KEY)
                print(f"[최종 대본 생성 작업] google.genai Client 재초기화 성공")
            except ImportError as e:
                raise ImportError(f"google-genai 패키지가 설치되지 않았습니다: {e}")
            except Exception as e:
                raise Exception(f"Gemini API Client 초기화 실패: {e}")
        
        update_job(job_id, message="챕터 분리 중...", stage_progress=10)
        
        # 챕터별로 분리
        chapters = split_draft_script_into_chapters(draft_script)
        print(f"[최종 대본 생성] 총 {len(chapters)}개 챕터로 분리됨")
        
        if not chapters:
            raise ValueError("챕터를 분리할 수 없습니다.")
        
        update_job(job_id, message=f"총 {len(chapters)}개 챕터로 분리하여 처리를 시작합니다.", stage_progress=15)
        
        # grouping_size에 따라 프롬프트 동적 생성
        if grouping_size == 1:
            # 기존 로직: 각 문장마다 처리
            system_prompt = """[YOUR ROLE]

당신은 '스크립트-투-이미지 프롬프트 전문 엔지니어'입니다. 당신의 임무는 사용자가 제공하는 대본(Script)을 받아, 이를 시각화하기 위한 두 가지 핵심 결과물을 순서대로 제공하는 것입니다.

당신은 단순한 번역기가 아닙니다. 당신은 각 문장의 **'문맥(Context)'**을 파악하고, 이것이 **'장면 묘사(Scene)'**인지 **'인물 묘사(Character)'**인지를 분석하여, 우리가 사전에 정의한 **'스타일 래퍼(Style Wrapper)'**와 결합한 최고 품질의 이미지 생성 프롬프트를 만들어야 합니다.

[CRITICAL: COMPLETENESS PROTOCOL]이 섹션은 가장 중요합니다. 반드시 준수하십시오.

전체 출력 의무: 사용자가 제공한 대본이 아무리 길더라도, 첫 문장부터 마지막 문장까지 단 한 문장도 빠뜨리지 않고 처리해야 합니다.

요약 금지: "나머지는 생략함(...)" 또는 "이하 동일"과 같은 요약 행위를 엄격히 금지합니다.

분할 출력 대응: 만약 답변 길이가 AI의 출력 토큰 제한(Output Token Limit)에 도달할 것 같으면, 문장의 중간에서 끊지 말고 완전한 [한국어 번역]-[영어 이미지 프롬프트] 세트가 끝나는 지점에서 멈추십시오. 그리고 답변 끝에 **"[...계속하려면 '계속'이라고 말해주세요]"**라고 명시하여 사용자가 이어서 출력받을 수 있도록 안내하십시오.

[CRITICAL: FORMATTING PROTOCOL - 태그 엄수]이 섹션은 출력의 일관성을 위해 가장 중요합니다. 반드시 준수하십시오.

긴 글을 출력할 때 [영어 이미지 프롬프트] 태그가 영어 이미지 프롬pt, 영어 이미지 prompt, [영어 이미지 prompt] 등과 같이 변형되는 오류를 엄격히 금지합니다.

영어 이미지 프롬프트의 출력 태그는 대괄호([])를 포함하여 정확히 [영어 이미지 프롬프트] 로만 출력되어야 하며, 일체의 오타나 변형을 허용하지 않습니다.

정확한 예시: [영어 이미지 프롬프트]

[WORKFLOW & RULES]

입력 처리:

   - 사용자가 대본을 제공합니다. (언어 무관, 주로 영어/한국어 혼용 가능성 있음)

   - 대본에 00:01:23 --> 00:01:25와 같은 타임스탬프(Timestamp)가 포함되어 있다면, 반드시 모두 제거하고 순수 텍스트 내용만 처리합니다.

문맥 분석 (내부 단계):

   - 작업을 시작하기 전, 대본 전체를 빠르게 훑어보며 이 장면의 전반적인 **문맥(Context)**을 파악합니다. (예: 시대, 장소, 분위기, 주요 인물)

   - 아래의 '스타일 래퍼 4가지' 중 이 대본의 분위기에 가장 적합한 것을 내부적으로 하나 선택하여 일관되게 적용합니다.

     1. 빈티지 아날로그 (Vintage Analog)

     2. 다큐멘터리 (Documentary)

     3. 모던 시네마틱 (Modern Cinematic)

     4. 디지털 리얼리즘 (Digital Realism)

순차적 출력 (필수 형식):

   - 대본의 각 문장을 1, 2, 3... 번호순으로 처리합니다.

   - 각 번호마다 반드시 아래의 두 가지 요소를 순서대로 제공해야 합니다.

   A. 한국어 번역:

   - 타임스탬프가 제거된 원본 문장의 정확하고 자연스러운 한국어 번역문을 제공합니다.

   B. 영어 이미지 프롬프트:

   - [CRITICAL: FORMATTING PROTOCOL - 태그 엄수] 섹션의 지침에 따라, 태그를 **정확히 [영어 이미지 프롬프트]**로만 출력해야 합니다.

   - 해당 문장을 시각화하기 위한 상세한 영어 이미지 프롬프트를 제공합니다.

   - 구성 요소: [Style Wrapper] + [Context & Subject] + [Visual Details]

   - [Style Wrapper]: 2단계에서 선택한 스타일 (예: Shot on 35mm analog film, grainy texture...)

   - [Context & Subject]: 문맥(시대/장소)과 묘사(인물/행동)를 결합 (예: 1950s office, A weary detective looking at...)

   - [Visual Details]: 카메라 앵글, 조명 등 (예: wide shot, cinematic lighting, shallow depth of field)

[최종 출력 형식 예시]

1.

[한국어 번역] (여기에 첫 번째 문장의 번역)

[영어 이미지 프롬프트] (스타일 래퍼 + 문맥 + 묘사가 결합된 영어 프롬프트)

[한국어 번역] (여기에 두 번째 문장의 번역)

[영어 이미지 프롬프트] (스타일 래퍼 + 문맥 + 묘사가 결합된 영어 프롬프트)

…(대본의 마지막 문장까지 반복. 절대 중단하지 말 것)…

[중요 지시사항]
- 아래의 프롬프트 외 불필요한 답변은 하지 말아주세요.
- 자연스럽고 매력적인 대화체로 작성하되, 괄호나 대괄호를 사용하지 마세요 (TTS를 이용할 것이기 때문)"""
        else:
            # 묶음 처리 로직: grouping_size개 문장씩 묶어서 처리
            system_prompt = f"""[YOUR ROLE]

당신은 '스크립트-투-이미지 프롬프트 전문 엔지니어'입니다. 당신의 임무는 사용자가 제공하는 대본(Script)을 받아, 이를 시각화하기 위한 두 가지 핵심 결과물을 순서대로 제공하는 것입니다.

당신은 단순한 번역기가 아닙니다. 당신은 문장들의 **'문맥(Context)'**을 파악하고, 이것이 **'장면 묘사(Scene)'**인지 **'인물 묘사(Character)'**인지를 분석하여, 우리가 사전에 정의한 **'스타일 래퍼(Style Wrapper)'**와 결합한 최고 품질의 이미지 생성 프롬프트를 만들어야 합니다.

[CRITICAL: COMPLETENESS PROTOCOL]이 섹션은 가장 중요합니다. 반드시 준수하십시오.

전체 출력 의무: 사용자가 제공한 대본이 아무리 길더라도, 첫 문장부터 마지막 문장까지 단 한 문장도 빠뜨리지 않고 처리해야 합니다.

요약 금지: "나머지는 생략함(...)" 또는 "이하 동일"과 같은 요약 행위를 엄격히 금지합니다.

분할 출력 대응: 만약 답변 길이가 AI의 출력 토큰 제한(Output Token Limit)에 도달할 것 같으면, 묶음의 중간에서 끊지 말고 완전한 [한국어 번역]-[영어 이미지 프롬프트] 세트가 끝나는 지점에서 멈추십시오. 그리고 답변 끝에 **"[...계속하려면 '계속'이라고 말해주세요]"**라고 명시하여 사용자가 이어서 출력받을 수 있도록 안내하십시오.

[CRITICAL: FORMATTING PROTOCOL - 태그 엄수]이 섹션은 출력의 일관성을 위해 가장 중요합니다. 반드시 준수하십시오.

긴 글을 출력할 때 [영어 이미지 프롬프트] 태그가 영어 이미지 프롬pt, 영어 이미지 prompt, [영어 이미지 prompt] 등과 같이 변형되는 오류를 엄격히 금지합니다.

영어 이미지 프롬프트의 출력 태그는 대괄호([])를 포함하여 정확히 [영어 이미지 프롬프트] 로만 출력되어야 하며, 일체의 오타나 변형을 허용하지 않습니다.

정확한 예시: [영어 이미지 프롬프트]

[WORKFLOW & RULES]

입력 처리:

   - 사용자가 대본을 제공합니다. (언어 무관, 주로 영어/한국어 혼용 가능성 있음)

   - 대본에 00:01:23 --> 00:01:25와 같은 타임스탬프(Timestamp)가 포함되어 있다면, 반드시 모두 제거하고 순수 텍스트 내용만 처리합니다.

문맥 분석 (내부 단계):

   - 작업을 시작하기 전, 대본 전체를 빠르게 훑어보며 이 장면의 전반적인 **문맥(Context)**을 파악합니다. (예: 시대, 장소, 분위기, 주요 인물)

   - 아래의 '스타일 래퍼 4가지' 중 이 대본의 분위기에 가장 적합한 것을 내부적으로 하나 선택하여 일관되게 적용합니다.

     1. 빈티지 아날로그 (Vintage Analog)

     2. 다큐멘터리 (Documentary)

     3. 모던 시네마틱 (Modern Cinematic)

     4. 디지털 리얼리즘 (Digital Realism)

묶음 단위 출력 (필수 형식):

   - 입력된 대본을 순서대로 **{grouping_size}개의 문장씩 묶어서** 처리하세요.

   - 각 묶음(Group)에 대해 다음 형식을 출력하세요:

   A. 한국어 번역:

   - 묶인 {grouping_size}개 문장의 내용을 합쳐서 자연스럽게 번역하거나 원문을 유지합니다. (대본이 한국어인 경우 원문 유지, 영어인 경우 번역)

   B. 영어 이미지 프롬프트:

   - [CRITICAL: FORMATTING PROTOCOL - 태그 엄수] 섹션의 지침에 따라, 태그를 **정확히 [영어 이미지 프롬프트]**로만 출력해야 합니다.

   - 묶인 {grouping_size}개 문장들의 내용을 포괄하여 시각화하는 **단 하나의** 영어 프롬프트를 작성합니다.

   - 구성 요소: [Style Wrapper] + [Context & Subject] + [Visual Details]

   - [Style Wrapper]: 2단계에서 선택한 스타일 (예: Shot on 35mm analog film, grainy texture...)

   - [Context & Subject]: 묶인 문장들의 문맥(시대/장소)과 묘사(인물/행동)를 종합하여 결합 (예: 1950s office, A weary detective looking at...)

   - [Visual Details]: 카메라 앵글, 조명 등 (예: wide shot, cinematic lighting, shallow depth of field)

[최종 출력 형식 예시]

1.

[한국어 번역] (첫 번째부터 {grouping_size}번째 문장까지 묶인 내용의 번역 또는 원문)

[영어 이미지 프롬프트] (묶인 {grouping_size}개 문장을 포괄하는 하나의 영어 프롬프트)

2.

[한국어 번역] ({grouping_size+1}번째부터 {grouping_size*2}번째 문장까지 묶인 내용의 번역 또는 원문)

[영어 이미지 프롬프트] (묶인 {grouping_size}개 문장을 포괄하는 하나의 영어 프롬프트)

…(대본의 마지막 문장까지 반복. 절대 중단하지 말 것)…

[중요 지시사항]
- 아래의 프롬프트 외 불필요한 답변은 하지 말아주세요.
- 자연스럽고 매력적인 대화체로 작성하되, 괄호나 대괄호를 사용하지 마세요 (TTS를 이용할 것이기 때문)
- 각 묶음은 정확히 {grouping_size}개의 문장을 포함해야 합니다 (마지막 묶음은 남은 문장 수가 {grouping_size}개 미만일 수 있음)"""

        # 각 챕터별로 처리
        all_final_scripts = []
        all_full_responses = []
        
        for idx, chapter in enumerate(chapters, 1):
            chapter_content = chapter["content"]
            chapter_title = chapter.get("title", f"챕터 {chapter['chapter_num']}")
            
            print(f"[최종 대본 생성] 챕터 {idx}/{len(chapters)} 처리 중: {chapter_title} (길이: {len(chapter_content)}자)")
            update_job(job_id, message=f"챕터 {idx}/{len(chapters)}: '{chapter_title}' 처리 중...", stage_progress=15 + int((idx/len(chapters))*70))
            
            # [수정됨] 챕터가 너무 길면 청크 분할 (1500자 단위로 줄임 - AI가 빼먹지 않도록)
            max_chunk_length = 1500
            chunks = []
            
            if len(chapter_content) > max_chunk_length:
                # 1. 마침표(.) 또는 줄바꿈(\n)으로 문장 분리
                import re
                # 문장 끝(. ? !) 뒤에 공백이 있거나 줄바꿈이 있는 경우 분리
                sentences = re.split(r'(?<=[.?!])\s+', chapter_content)
                
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence: continue
                    
                    # 현재 청크가 꽉 차면 저장하고 비움
                    if current_length + len(sentence) > max_chunk_length and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                
                if current_chunk: 
                    chunks.append(' '.join(current_chunk))
            else:
                chunks = [chapter_content]

            # [디버깅용 로그] 분할 결과 확인
            print(f"[DEBUG] 챕터 길이: {len(chapter_content)} -> 청크 {len(chunks)}개로 분할됨")
            
            # 각 청크 처리
            for chunk_idx, chunk_content in enumerate(chunks, 1):
                if grouping_size == 1:
                    # 기존 로직: 각 문장마다 처리
                    base_user_prompt = f"""다음 검수 대본을 최종 대본으로 변환해주세요.

[중요 지침]
1. 검수 대본의 언어를 먼저 확인하세요.
2. 검수 대본이 한국어인 경우: 각 문장을 [한국어 번역] 태그로 감싸고, 그 다음에 [영어 이미지 프롬프트]를 제공하세요.
3. 검수 대본이 영어인 경우: 각 문장을 [한국어 번역]으로 번역하고, 그 다음에 [영어 이미지 프롬프트]를 제공하세요.
4. 반드시 아래 형식을 정확히 따르세요.
5. 아래의 프롬프트 외 불필요한 답변은 하지 말아주세요.

[출력 형식]
1.

[한국어 번역] (여기에 첫 번째 문장의 번역)

[영어 이미지 프롬프트] (스타일 래퍼 + 문맥 + 묘사가 결합된 영어 프롬프트)

[한국어 번역] (여기에 두 번째 문장의 번역)

[영어 이미지 프롬프트] (스타일 래퍼 + 문맥 + 묘사가 결합된 영어 프롬프트)

…(대본의 마지막 문장까지 반복. 절대 중단하지 말 것)…

검수 대본:
{chunk_content}"""
                else:
                    # 묶음 처리 로직: grouping_size개 문장씩 묶어서 처리
                    base_user_prompt = f"""다음 검수 대본을 최종 대본으로 변환해주세요.

[중요 지침]
1. 검수 대본의 언어를 먼저 확인하세요.
2. 입력된 대본을 순서대로 **{grouping_size}개의 문장씩 묶어서** 처리하세요.
3. 검수 대본이 한국어인 경우: 각 묶음의 {grouping_size}개 문장을 [한국어 번역] 태그로 감싸고, 그 다음에 [영어 이미지 프롬프트]를 제공하세요.
4. 검수 대본이 영어인 경우: 각 묶음의 {grouping_size}개 문장을 [한국어 번역]으로 번역하고, 그 다음에 [영어 이미지 프롬프트]를 제공하세요.
5. 반드시 아래 형식을 정확히 따르세요.
6. 아래의 프롬프트 외 불필요한 답변은 하지 말아주세요.

[출력 형식]
1.

[한국어 번역] (첫 번째부터 {grouping_size}번째 문장까지 묶인 내용의 번역 또는 원문)

[영어 이미지 프롬프트] (묶인 {grouping_size}개 문장을 포괄하는 하나의 영어 프롬프트)

2.

[한국어 번역] ({grouping_size+1}번째부터 {grouping_size*2}번째 문장까지 묶인 내용의 번역 또는 원문)

[영어 이미지 프롬프트] (묶인 {grouping_size}개 문장을 포괄하는 하나의 영어 프롬프트)

…(대본의 마지막 문장까지 반복. 절대 중단하지 말 것)…

검수 대본:
{chunk_content}"""

                try:
                    # 여러 번의 요청으로 전체 결과 수집
                    chunk_responses = []
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": base_user_prompt}
                    ]
                    max_continuations = 15  # 횟수 조금 더 여유있게 증가
                    continuation_count = 0
                    
                    while continuation_count <= max_continuations:
                        # 디버깅: 전송되는 프롬프트 확인
                        if continuation_count == 0:
                            print(f"\n{'='*80}")
                            print(f"[최종 대본 생성] 챕터 {idx} 청크 {chunk_idx} - 프롬프트 전송 시작")
                            print(f"[DEBUG] system_prompt 길이: {len(system_prompt)}자")
                            print(f"[DEBUG] system_prompt 처음 500자:\n{system_prompt[:500]}")
                            print(f"[DEBUG] user_prompt 길이: {len(messages[-1]['content'])}자")
                            print(f"[DEBUG] user_prompt 처음 500자:\n{messages[-1]['content'][:500]}")
                            print(f"{'='*80}\n")
                        else:
                            print(f"\n{'='*80}")
                            print(f"[최종 대본 생성] 챕터 {idx} 청크 {chunk_idx} - 계속 요청 {continuation_count}번째")
                            print(f"{'='*80}\n")
                        
                        # Gemini API 호출: messages를 단일 프롬프트로 결합
                        full_prompt_parts = []
                        for msg in messages:
                            if msg["role"] == "system":
                                full_prompt_parts.append(f"[SYSTEM]\n{msg['content']}\n")
                            elif msg["role"] == "user":
                                full_prompt_parts.append(f"[USER]\n{msg['content']}\n")
                            elif msg["role"] == "assistant":
                                full_prompt_parts.append(f"[ASSISTANT]\n{msg['content']}\n")
                        full_prompt = "\n".join(full_prompt_parts)
                        
                        # 새로운 라이브러리 방식
                        if genai_client is None:
                            raise RuntimeError("Gemini API Client가 초기화되지 않았습니다.")
                        
                        from google.genai import types
                        target_model = 'gemini-2.5-flash'
                        chunk_response = None
                        try:
                            response = genai_client.models.generate_content(
                                model=target_model,
                                contents=full_prompt,
                                config=types.GenerateContentConfig(
                                    temperature=0.7,
                                    max_output_tokens=8000,
                                )
                            )
                            
                            # Gemini API 응답 처리
                            if not response or not hasattr(response, 'text'):
                                raise Exception("Gemini API 응답에 텍스트가 없습니다.")
                            
                            chunk_response = response.text.strip() if response.text else ""
                            
                            # 새로운 라이브러리는 safety_ratings를 다르게 제공할 수 있음
                            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                                    raise ValueError(f"챕터 {idx} 처리 중 콘텐츠 정책 위반으로 인해 대본 생성이 차단되었습니다. 검수 대본의 내용을 수정하거나 다른 주제로 시도해주세요.")
                            
                            # 디버깅: Gemini 응답 확인
                            if chunk_response:
                                print(f"\n{'='*80}")
                                print(f"[최종 대본 생성] 챕터 {idx} 청크 {chunk_idx} - Gemini 응답 받음 (요청 {continuation_count + 1}번째)")
                                print(f"[DEBUG] 응답 길이: {len(chunk_response)}자")
                                print(f"[DEBUG] 응답 처음 1000자:\n{chunk_response[:1000]}")
                                if len(chunk_response) > 1000:
                                    print(f"[DEBUG] 응답 마지막 500자:\n{chunk_response[-500:]}")
                                print(f"{'='*80}\n")
                            
                        except Exception as e:
                            error_text = str(e)
                            print(f"[최종 대본 생성] 챕터 {idx} 청크 {chunk_idx} Gemini API 오류: {error_text}")
                            import traceback
                            traceback.print_exc()
                            raise RuntimeError(f"챕터 {idx} 처리 중 Gemini API 오류: {error_text}")
                        
                        # chunk_response가 None이거나 빈 문자열인 경우 처리
                        if not chunk_response:
                            error_msg = f"챕터 {idx} 청크 {chunk_idx} 처리 중 Gemini API 응답이 비어있습니다."
                            print(f"[최종 대본 생성] {error_msg}")
                            raise RuntimeError(error_msg)
                        
                        # Gemini 거부 응답 감지
                        refusal_keywords = [
                            "I'm sorry, I can't assist",
                            "can't assist with that request",
                            "요청하신 작업을 수행할 수 없습니다",
                            "요청하신 작업을 제공하지 못합니다",
                            "죄송합니다, 요청하신 작업"
                        ]
                        
                        is_refusal = any(keyword in chunk_response for keyword in refusal_keywords) if chunk_response else False
                        
                        # content가 None이거나 빈 문자열인 경우, 또는 거부 응답인 경우
                        if not chunk_response or is_refusal:
                            if is_refusal:
                                print(f"[최종 대본 생성] Gemini 거부 응답 감지, 재시도 중...")
                                # 거부 응답인 경우, 프롬프트를 수정하여 재시도
                                if continuation_count < 3:  # 최대 3번까지 재시도
                                    continuation_count += 1
                                    # 프롬프트를 더 명확하게 수정
                                    retry_prompt = f"""다음 검수 대본을 최종 대본으로 변환해주세요. 이는 교육용 유튜브 콘텐츠 제작을 위한 대본 변환 작업입니다.

[중요 지침]
1. 검수 대본의 각 문장을 [한국어 번역] 태그로 감싸고, 그 다음에 [영어 이미지 프롬프트]를 제공하세요.
2. 반드시 아래 형식을 정확히 따르세요:

[출력 형식]
1.
[한국어 번역] (첫 번째 문장)
[영어 이미지 프롬프트] (스타일 래퍼 + 문맥 + 묘사가 결합된 영어 프롬프트)

2.
[한국어 번역] (두 번째 문장)
[영어 이미지 프롬프트] (스타일 래퍼 + 문맥 + 묘사가 결합된 영어 프롬프트)

[요구사항]
- 모든 문장을 빠뜨리지 말고 순서대로 처리하세요
- [한국어 번역]과 [영어 이미지 프롬프트] 태그를 정확히 사용하세요

검수 대본:
{chunk_content}"""
                                    # 메시지 히스토리 초기화하고 재시도
                                    messages = [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": retry_prompt}
                                    ]
                                    continue
                                else:
                                    error_msg = f"챕터 {idx} 처리 중 Gemini가 요청을 거부했습니다. 대본 내용을 확인하거나 다른 주제로 시도해주세요."
                                    print(f"[최종 대본 생성] 재시도 횟수 초과")
                                    raise ValueError(error_msg)
                            else:
                                error_msg = f"챕터 {idx} 처리 중 최종 대본 생성에 실패했습니다."
                                print(f"[최종 대본 생성] 챕터 {idx} 응답 없음")
                                raise RuntimeError(error_msg)
                        
                        # 응답 저장
                        chunk_responses.append(chunk_response)
                        
                        # 응답을 메시지 히스토리에 추가
                        messages.append({"role": "assistant", "content": chunk_response})
                        
                        # 완료 여부 체크 (간소화된 로직)
                        is_finished = False
                        # 완료 키워드 체크
                        if "모든 문장을 처리했습니다" in chunk_response or "처리할 부분이 없습니다" in chunk_response:
                            is_finished = True
                        # 내용이 짧고 '계속'이라는 말이 없으면 종료로 간주
                        elif "계속" not in chunk_response and len(chunk_response) < 5000:
                            import re
                            # 번역 태그는 있는데 계속하라는 말이 없으면 끝난 것으로 간주
                            if re.search(r'\[한국어 번역\]', chunk_response) and not re.search(r'계속', chunk_response):
                                 is_finished = True

                        if is_finished:
                            break
                            
                        continuation_count += 1
                        update_job(job_id, message=f"챕터 {idx} 내용이 길어 계속 생성 중... ({continuation_count}/{max_continuations})")
                        messages.append({"role": "user", "content": "계속해주세요. 이전 응답의 마지막 부분부터 이어서 모든 남은 문장을 처리해주세요."})
                        
                        if continuation_count > max_continuations:
                            print(f"[최종 대본 생성] 경고: 최대 계속 요청 횟수({max_continuations}) 도달, 현재까지 수집된 결과 반환")
                            break
                        
                        # 다음 반복으로 계속
                        continue
                    
                    # 모든 응답 합치기
                    chunk_full_response = "\n\n".join(chunk_responses)
                    all_full_responses.append(chunk_full_response)
                    
                except Exception as chunk_error:
                    print(f"[최종 대본 생성] 챕터 {idx} 청크 {chunk_idx} 처리 중 오류: {chunk_error}")
                    import traceback
                    traceback.print_exc()
                    raise chunk_error
        
        # 모든 챕터의 응답을 합치기
        full_response = "\n\n".join(all_full_responses)
        
        # 디버깅: 전체 응답 확인
        print(f"\n{'='*80}")
        print(f"[최종 대본 생성] 전체 응답 합치기 완료")
        print(f"[DEBUG] 전체 응답 길이: {len(full_response)}자")
        print(f"[DEBUG] 전체 응답 처음 2000자:\n{full_response[:2000]}")
        print(f"{'='*80}\n")
        
        # TTS용 순수 대본 추출 및 영어 이미지 프롬프트 추출
        import re
        final_script_lines = []
        image_prompts = []  # 영어 이미지 프롬프트 저장
        
        # [한국어 번역] ... [영어 이미지 프롬프트] ... 추출
        pairs = re.findall(r'\[한국어 번역\]\s*(.+?)\s*\[영어 이미지 프롬프트\]\s*(.+?)(?=\[한국어 번역\]|$|\n\n\d+\.)', full_response, re.DOTALL)
        
        for k, e in pairs:
            kc = re.sub(r'\(.*?\)|\[.*?\]', '', k).strip()
            ec = re.sub(r'\(.*?\)', '', e).strip()
            if kc and len(kc) > 2: final_script_lines.append(kc)
            if ec: image_prompts.append(ec)
            
        # Fallback 파싱 (번호 형식 등)
        if not final_script_lines:
             numbered_pairs = re.findall(r'\d+\.\s*\[한국어 번역\]\s*(.+?)\s*\[영어 이미지 프롬프트\]\s*(.+?)(?=\d+\.\s*\[한국어 번역\]|$)', full_response, re.DOTALL)
             for k, e in numbered_pairs:
                kc = re.sub(r'\(.*?\)|\[.*?\]', '', k).strip()
                ec = re.sub(r'\(.*?\)', '', e).strip()
                if kc: final_script_lines.append(kc)
                if ec: image_prompts.append(ec)

        # 최후의 수단: 그냥 줄별로
        if not final_script_lines:
             clean_text = re.sub(r'\[.*?\]', '', full_response)
             lines = clean_text.split('\n')
             final_script_lines = [l.strip() for l in lines if l.strip() and not l.strip().isdigit()]

        final_script = '\n'.join(final_script_lines)
        
        # 썸네일 프롬프트
        topic_match = re.search(r'주제[:\s]+(.+?)(?:\n|$)', draft_script, re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else "video content"
        thumbnail_prompt = f"YouTube thumbnail for video about: {topic}. High quality..."
        
        # 작업 완료 저장
        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["final_script"] = final_script
            jobs[job_id]["visual_prompt"] = full_response # 전체 응답 보관
            jobs[job_id]["image_prompts"] = image_prompts
            jobs[job_id]["thumbnail_prompt"] = thumbnail_prompt
            jobs[job_id]["current_stage"] = "완료"
            jobs[job_id]["stage_progress"] = 100
        
    except Exception as e:
        print(f"[최종 대본 생성 작업 오류] {e}")
        import traceback
        traceback.print_exc()
        update_job(job_id, status="error", error=str(e))


@app.route("/api/generate_final_script", methods=["POST"])
def api_generate_final_script():
    """최종 대본 생성 API (비동기 처리)"""
    data = request.get_json(silent=True) or {}
    draft_script = (data.get("draft_script") or "").strip()
    grouping_size = data.get("grouping_size", 1)
    
    # grouping_size 유효성 검사
    try:
        grouping_size = int(grouping_size)
        if grouping_size < 1 or grouping_size > 10:
            grouping_size = 1
    except (ValueError, TypeError):
        grouping_size = 1
    
    if not draft_script:
        return jsonify({"error": "검수 대본을 입력해주세요."}), 400
    
    # API 키 확인
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API 키가 설정되지 않았습니다. 설정에서 키를 입력해주세요."}), 500

    # 작업 ID 생성
    job_id = create_job_record()
    
    # 백그라운드 스레드 실행
    thread = threading.Thread(
        target=run_final_script_job,
        args=(job_id, draft_script, grouping_size),
        daemon=True
    )
    thread.start()
    
    # 작업 시작했다는 응답만 즉시 반환
    return jsonify({"job_id": job_id, "status": "processing"})


@app.route("/api/analyze_image_style", methods=["POST"])
def api_analyze_image_style():
    """이미지 스타일 분석 API"""
    global genai_client
    
    try:
        # API 키 확인
        if not GEMINI_API_KEY:
            return jsonify({"error": "Gemini API 키가 설정되지 않았습니다. 설정에서 키를 입력해주세요."}), 500
        
        # 파일 확인
        if 'image' not in request.files:
            return jsonify({"error": "이미지 파일이 없습니다."}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "이미지 파일이 선택되지 않았습니다."}), 400
        
        # Gemini 클라이언트 확인 및 재초기화
        if genai_client is None:
            try:
                from google import genai
                genai_client = genai.Client(api_key=GEMINI_API_KEY)
                print(f"[이미지 스타일 분석] google.genai Client 초기화 성공")
            except ImportError as e:
                return jsonify({"error": f"google-genai 패키지가 설치되지 않았습니다: {e}"}), 500
            except Exception as e:
                return jsonify({"error": f"Gemini API Client 초기화 실패: {e}"}), 500
        
        # 이미지 파일 읽기
        image_data = image_file.read()
        image_file.seek(0)  # 파일 포인터 리셋
        
        # System Prompt
        system_prompt = """당신은 전문 아트 디렉터이자 프롬프트 엔지니어입니다. 사용자가 제공한 이미지를 분석하여 '내용(인물, 피사체)'은 철저히 배제하고, 오직 '시각적 스타일(Visual Style)'만 추출하세요. 결과는 반드시 아래의 JSON 구조로만 출력해야 합니다.

필수 JSON 구조:
{
  "prompt_style": {
    "art_medium": ["매체 예: Charcoal drawing, Oil painting..."],
    "visual_style": ["스타일 예: Monochromatic, Chiaroscuro..."],
    "lighting_and_atmosphere": ["조명 예: Volumetric lighting, God rays..."],
    "texture_and_details": ["질감 예: Grainy paper texture, Sketch lines..."],
    "color_palette": ["색상 예: Greyscale, Muted tones..."]
  }
}

중요:
- 인물, 사물, 배경의 구체적인 내용은 절대 포함하지 마세요
- 오직 시각적 스타일, 화풍, 기법만 추출하세요
- 각 배열에는 관련 키워드들을 영어로 나열하세요
- JSON 형식만 출력하고 다른 설명은 하지 마세요"""
        
        # Gemini API 호출
        print(f"[이미지 스타일 분석] Gemini API 호출 시작...")
        
        # 이미지를 base64로 인코딩
        import base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        mime_type = image_file.content_type or "image/jpeg"
        
        # Gemini 2.0 Flash 모델 사용
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": system_prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        )
        
        response_text = response.text.strip() if hasattr(response, 'text') else str(response)
        print(f"[이미지 스타일 분석] Gemini 응답 받음 (길이: {len(response_text)}자)")
        
        # JSON 파싱 시도
        import json
        import re
        
        # JSON 부분만 추출 (코드 블록 제거)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response_text
        
        try:
            style_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[이미지 스타일 분석] JSON 파싱 오류: {e}")
            print(f"[이미지 스타일 분석] 원본 응답: {response_text[:500]}")
            return jsonify({"error": f"AI 응답을 파싱할 수 없습니다: {str(e)}", "raw_response": response_text[:500]}), 500
        
        return jsonify({
            "success": True,
            "style_data": style_data,
            "raw_response": response_text
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[이미지 스타일 분석] 오류 발생: {e}")
        print(f"[이미지 스타일 분석] 상세 오류:\n{error_trace}")
        return jsonify({"error": f"이미지 스타일 분석 중 오류가 발생했습니다: {str(e)}"}), 500


@app.route("/start_job", methods=["POST"])
def start_job():
    payload = request.get_json(silent=True) or {}
    job_id = (payload.get("job_id") or "").strip()
    script_text = (payload.get("script_text") or "").strip()
    char_limit_raw = str(payload.get("char_limit") or "").strip()
    voice_id = (payload.get("voice_id") or "").strip()
    mode = (payload.get("mode") or "animation").strip().lower()
    custom_style_prompt = (payload.get("custom_style_prompt") or "").strip()  # 커스텀 모드 스타일 프롬프트
    include_subtitles = payload.get("include_subtitles", True)  # 기본값은 True (자막 포함)
    # API 키 (사용자가 입력한 경우 사용, 없으면 기본값 사용)
    replicate_api_key = payload.get("replicate_api_key") or None
    elevenlabs_api_key = payload.get("elevenlabs_api_key") or None

    try:
        char_limit = int(char_limit_raw) if char_limit_raw else 50
        if char_limit < 5:
            char_limit = 5
    except ValueError:
        char_limit = 50

    if not is_voice_allowed(voice_id):
        voice_id = get_default_voice_id()

    prompts_override = None
    existing_images = None
    sentences_override = None

    if job_id:
        job = get_job(job_id)
        if not job:
            return jsonify({"error": "작업을 찾을 수 없습니다."}), 404
        script_text = job.get("script_text") or script_text
        if not script_text:
            return jsonify({"error": "저장된 대본이 없습니다."}), 400
        prompts_override = job.get("prompts")
        # 저장된 sentences가 있으면 사용 (이미 파싱된 한국어 문장만 포함)
        sentences_override = job.get("sentences")
        image_files = job.get("image_files") or {}
        existing_images = {int(k): v for k, v in image_files.items() if os.path.exists(v)}
        with jobs_lock:
            job_data = jobs.get(job_id)
            if job_data:
                job_data["progress"] = []
                job_data["error"] = None
                job_data["video_filename"] = None
                job_data["status"] = "pending"
                job_data["mode"] = mode
                job_data["stage_progress"] = 0
                job_data["current_stage"] = ""
    else:
        if not script_text:
            return jsonify({"error": "대본이 비어 있습니다."}), 400
        job_id = create_job_record()
        with jobs_lock:
            job_data = jobs.get(job_id)
            if job_data is not None:
                job_data["script_text"] = script_text
                job_data["mode"] = mode
                job_data["stage_progress"] = 0
                job_data["current_stage"] = ""

    thread = threading.Thread(
        target=run_generation_job,
        args=(job_id, script_text, char_limit, voice_id, prompts_override, existing_images, mode, sentences_override, include_subtitles, replicate_api_key, elevenlabs_api_key, custom_style_prompt),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/generate_prompts", methods=["POST"])
def api_generate_prompts():
    data = request.get_json(silent=True) or {}
    script_text = (data.get("script_text") or "").strip()
    mode = (data.get("mode") or "animation").strip().lower()
    custom_style_prompt = (data.get("custom_style_prompt") or "").strip()  # 커스텀 모드 스타일 프롬프트
    user_prompts = data.get("user_prompts")  # 사용자가 입력한 프롬프트 리스트
    
    if not script_text:
        return jsonify({"error": "대본이 비어 있습니다."}), 400

    sentences = split_script_into_sentences(script_text)
    if not sentences:
        return jsonify({"error": "대본에서 문장을 추출할 수 없습니다."}), 400

    job_id = create_job_record()
    
    # 진행도 초기화
    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            job["script_text"] = script_text
            job["sentences"] = sentences
            job["image_files"] = {}
            job["mode"] = mode
            job["progress"] = []
            job["status"] = "running"
            job["stage_progress"] = 0
            job["current_stage"] = "프롬프트 생성 중"
            job["stage_total"] = 1
    
    def generate_prompts_with_progress():
        try:
            total = len(sentences)
            prompts = []
            
            # 사용자가 입력한 프롬프트가 있으면 그것을 사용
            if user_prompts and isinstance(user_prompts, list):
                prompts = []
                for idx, sentence in enumerate(sentences):
                    user_prompt = user_prompts[idx] if idx < len(user_prompts) else None
                    if user_prompt and user_prompt.strip():
                        # 사용자 입력 프롬프트를 모드에 맞게 강제 적용
                        prompts.append(enforce_prompt_by_mode(
                            user_prompt.strip(),
                            fallback_context=f"based on '{sentence[:50]}'",
                            mode=mode,
                            custom_style_prompt=custom_style_prompt
                        ))
                    else:
                        # 사용자 입력이 없으면 자동 생성
                        prompts.append(None)
                
                # 자동 생성이 필요한 프롬프트만 필터링
                needs_auto_generation = [i for i, p in enumerate(prompts) if p is None]
                
                if needs_auto_generation:
                    # 자동 생성이 필요한 문장들만 추출
                    sentences_to_generate = [sentences[i] for i in needs_auto_generation]
                    
                    def progress_callback(message):
                        with jobs_lock:
                            job_data = jobs.get(job_id)
                            if job_data is not None:
                                progress_pct = job_data.get("stage_progress", 0)
                                if "(" in message and "/" in message:
                                    try:
                                        match = re.search(r'\((\d+)/(\d+)\)', message)
                                        if match:
                                            current = int(match.group(1))
                                            total = int(match.group(2))
                                            progress_pct = int((current / total) * 90)
                                    except:
                                        pass
                                elif "완료" in message or "생성 완료" in message:
                                    progress_pct = min(95, progress_pct + 10)
                                
                                job_data["stage_progress"] = progress_pct
                                job_data["current_stage"] = f"프롬프트 생성 중... ({message})"
                                job_data["progress"].append(message)
                    
                    auto_prompts = generate_visual_prompts(
                        sentences_to_generate,
                        mode=mode,
                        progress_cb=progress_callback,
                        original_script=script_text,
                        custom_style_prompt=custom_style_prompt
                    )
                    
                    # 자동 생성된 프롬프트를 원래 위치에 삽입
                    for auto_idx, needs_idx in enumerate(needs_auto_generation):
                        if auto_idx < len(auto_prompts):
                            prompts[needs_idx] = auto_prompts[auto_idx]
                        else:
                            prompts[needs_idx] = build_fallback_prompt(sentences[needs_idx], mode)
                
                # 사용자 입력 프롬프트만 사용한 경우
                with jobs_lock:
                    job_data = jobs.get(job_id)
                    if job_data is not None:
                        job_data["prompts"] = prompts
                        job_data["stage_progress"] = 100
                        job_data["current_stage"] = "프롬프트 생성 완료"
                        job_data["status"] = "completed"
            else:
                # 사용자 입력이 없으면 기존처럼 자동 생성
                def progress_callback(message):
                    with jobs_lock:
                        job_data = jobs.get(job_id)
                        if job_data is not None:
                            progress_pct = job_data.get("stage_progress", 0)
                            if "(" in message and "/" in message:
                                try:
                                    match = re.search(r'\((\d+)/(\d+)\)', message)
                                    if match:
                                        current = int(match.group(1))
                                        total = int(match.group(2))
                                        progress_pct = int((current / total) * 90)
                                except:
                                    pass
                            elif "완료" in message or "생성 완료" in message:
                                progress_pct = min(95, progress_pct + 10)
                            
                            job_data["stage_progress"] = progress_pct
                            job_data["current_stage"] = f"프롬프트 생성 중... ({message})"
                            job_data["progress"].append(message)
                
                prompts = generate_visual_prompts(sentences, mode=mode, progress_cb=progress_callback, original_script=script_text, custom_style_prompt=custom_style_prompt)
                
                with jobs_lock:
                    job_data = jobs.get(job_id)
                    if job_data is not None:
                        job_data["prompts"] = prompts
                        job_data["stage_progress"] = 100
                        job_data["current_stage"] = "프롬프트 생성 완료"
                        job_data["status"] = "completed"
        except Exception as exc:
            print(f"[경고] 프롬프트 생성 실패: {exc}")
            import traceback
            traceback.print_exc()
            with jobs_lock:
                job_data = jobs.get(job_id)
                if job_data is not None:
                    prompts = [build_fallback_prompt(sentence, mode) for sentence in sentences]
                    job_data["prompts"] = prompts
                    job_data["stage_progress"] = 100
                    job_data["current_stage"] = "프롬프트 생성 완료 (기본 프롬프트 사용)"
                    job_data["status"] = "completed"
    
    # 백그라운드 스레드로 실행
    thread = threading.Thread(target=generate_prompts_with_progress, daemon=True)
    thread.start()
    
    # 즉시 반환 (진행도는 job_status로 확인)
    # 백엔드에서 분리된 문장 리스트도 함께 반환하여 프론트엔드에서 정확한 매칭 가능하도록 함
    return jsonify({
        "job_id": job_id,
        "status": "processing",
        "sentences": sentences  # 백엔드에서 분리된 문장 리스트
    })


@app.route("/generate_images", methods=["POST"])
def api_generate_images():
    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or "").strip()
    if not job_id:
        return jsonify({"error": "job_id가 필요합니다."}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다."}), 404

    sentences = job.get("sentences") or []
    prompts = job.get("prompts") or []
    mode = job.get("mode") or "animation"
    if not sentences or not prompts:
        return jsonify({"error": "프롬프트가 준비되지 않았습니다."}), 400

    # 진행도 초기화
    with jobs_lock:
        job_data = jobs.get(job_id)
        if job_data is not None:
            job_data["status"] = "running"
            job_data["stage_progress"] = 0
            job_data["current_stage"] = "이미지 생성 중"
            job_data["stage_total"] = 1
            job_data["progress"] = []
            
            # API 키 확인 및 로그 추가
            replicate_api_key_from_job = job_data.get("replicate_api_key")
            if not replicate_api_key_from_job:
                replicate_api_key_from_job = REPLICATE_API_TOKEN
            if replicate_api_key_from_job:
                key_preview = replicate_api_key_from_job[:10] + "..." + replicate_api_key_from_job[-4:] if len(replicate_api_key_from_job) > 14 else "***"
                job_data["progress"].append(f"[시작] 이미지 생성 시작 (API 키: {key_preview})")
            else:
                job_data["progress"].append(f"[경고] Replicate API 키가 설정되지 않았습니다!")
                log_error(f"[경고] /generate_images 엔드포인트: Replicate API 키가 설정되지 않음")

    assets_folder = os.path.join(ASSETS_BASE_FOLDER, job_id)
    os.makedirs(assets_folder, exist_ok=True)

    def generate_images_with_progress():
        try:
            total = len(sentences)
            image_results = []
            image_files = {}
            
            print(f"[이미지 생성 시작] 총 {total}개 이미지 생성 예정")
            
            for idx, (sentence, prompt) in enumerate(zip(sentences, prompts), start=1):
                try:
                    # 진행도 업데이트 (시작)
                    progress_pct = int((idx / total) * 100)
                    with jobs_lock:
                        job_data = jobs.get(job_id)
                        if job_data is not None:
                            job_data["stage_progress"] = progress_pct
                            job_data["current_stage"] = f"이미지 생성 중... ({idx}/{total})"
                            job_data["progress"].append(f"{idx}번 이미지 생성 시작...")
                    
                    print(f"[이미지 생성] {idx}/{total} 시작 - 프롬프트: {prompt[:50]}...")
                    
                    image_filename = os.path.join(assets_folder, f"scene_{idx}_image.png")
                    
                    # 이미지 생성 실행
                    success = generate_image(prompt, image_filename, mode=mode)
                    
                    # 생성 결과 확인
                    if not success or not os.path.exists(image_filename):
                        print(f"[경고] {idx}번 이미지 생성 실패 또는 파일 없음, 기본 이미지 생성")
                        Image.new("RGB", (1920, 1080), color="black").save(image_filename)
                        with jobs_lock:
                            job_data = jobs.get(job_id)
                            if job_data is not None:
                                job_data["progress"].append(f"{idx}번 이미지 생성 실패 (기본 이미지 사용)")
                    else:
                        file_size = os.path.getsize(image_filename)
                        print(f"[이미지 생성] {idx}/{total} 완료 - 파일 크기: {file_size} bytes")
                        # 진행도 업데이트 (완료)
                        progress_pct = int((idx / total) * 100)
                        with jobs_lock:
                            job_data = jobs.get(job_id)
                            if job_data is not None:
                                job_data["stage_progress"] = progress_pct
                                job_data["progress"].append(f"{idx}번 이미지 생성 완료 ({file_size} bytes)")
                    
                    # Windows 경로 처리 개선
                    abs_path = os.path.abspath(image_filename)
                    image_files[idx] = abs_path
                    rel_path = os.path.relpath(abs_path, STATIC_FOLDER)
                    # Windows 경로 구분자를 /로 변환
                    rel_path = rel_path.replace(os.sep, "/")
                    # url_for 대신 직접 경로 구성 (백그라운드 스레드에서 안전)
                    image_url = "/static/" + rel_path
                    image_results.append({"index": idx, "sentence": sentence, "prompt": prompt, "image_url": image_url})
                    
                except Exception as img_exc:
                    # 개별 이미지 생성 실패 시에도 계속 진행
                    print(f"[경고] {idx}번 이미지 생성 중 예외 발생: {img_exc}")
                    import traceback
                    traceback.print_exc()
                    
                    # 기본 이미지 생성
                    image_filename = os.path.join(assets_folder, f"scene_{idx}_image.png")
                    try:
                        Image.new("RGB", (1920, 1080), color="black").save(image_filename)
                    except:
                        pass
                    
                    with jobs_lock:
                        job_data = jobs.get(job_id)
                        if job_data is not None:
                            job_data["progress"].append(f"{idx}번 이미지 생성 중 오류 발생: {str(img_exc)}")
                    
                    # 파일 정보는 추가
                    if os.path.exists(image_filename):
                        abs_path = os.path.abspath(image_filename)
                        image_files[idx] = abs_path
                        rel_path = os.path.relpath(abs_path, STATIC_FOLDER)
                        image_url = "/static/" + rel_path.replace(os.sep, "/")
                        image_results.append({"index": idx, "sentence": sentence, "prompt": prompt, "image_url": image_url})
            
            print(f"[이미지 생성 완료] 총 {len(image_results)}개 이미지 생성됨")
            
            with jobs_lock:
                job_data = jobs.get(job_id)
                if job_data is not None:
                    job_data["image_files"] = image_files
                    job_data["stage_progress"] = 100
                    job_data["current_stage"] = "이미지 생성 완료"
                    job_data["status"] = "completed"
                    job_data["progress"].append(f"모든 이미지 생성 완료 ({len(image_results)}/{total})")
        except Exception as exc:
            print(f"[경고] 이미지 생성 전체 실패: {exc}")
            import traceback
            traceback.print_exc()
            with jobs_lock:
                job_data = jobs.get(job_id)
                if job_data is not None:
                    job_data["status"] = "error"
                    job_data["error"] = str(exc)
                    job_data["current_stage"] = "이미지 생성 실패"
                    job_data["progress"].append(f"이미지 생성 전체 실패: {str(exc)}")
    
    # 백그라운드 스레드로 실행
    thread = threading.Thread(target=generate_images_with_progress, daemon=True)
    thread.start()
    
    # 즉시 반환 (진행도는 job_status로 확인)
    return jsonify({"job_id": job_id, "status": "processing"})


@app.route("/generate_images_direct", methods=["POST"])
def api_generate_images_direct():
    """프롬프트 생성 단계를 건너뛰고 바로 이미지 생성 (단, 실사화 모드는 프롬프트 재생성)"""
    print("=" * 80)
    print("=== [API] /generate_images_direct 요청 받음 ===")
    print(f"=== [API] 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print("=" * 80)
    
    data = request.get_json(silent=True) or {}
    script_text = (data.get("script_text") or "").strip()
    sentences = data.get("sentences") or []
    prompts = data.get("prompts") or []
    mode = (data.get("mode") or "animation").strip().lower()
    custom_style_prompt = (data.get("custom_style_prompt") or "").strip()  # 커스텀 모드 스타일 프롬프트
    test_mode = data.get("test_mode", False)
    banana_pro_flags = data.get("banana_pro_flags") or []  # PRO 체크박스 상태 배열
    banana_normal_flags = data.get("banana_normal_flags") or []  # 노말 체크박스 상태 배열
    # API 키 (사용자가 입력한 경우 사용, 없으면 기본값 사용)
    replicate_api_key = data.get("replicate_api_key") or None
    elevenlabs_api_key = data.get("elevenlabs_api_key") or None
    gemini_api_key = data.get("gemini_api_key") or None
    
    print(f"[API] 요청 파라미터: sentences={len(sentences)}, prompts={len(prompts)}, mode={mode}, test_mode={test_mode}, banana_pro_flags={len(banana_pro_flags)}, banana_normal_flags={len(banana_normal_flags)}")
    
    # 실사화 모드(realistic, realistic2)일 때는 프롬프트를 재생성해야 함
    if (mode == "realistic" or mode == "realistic2") and sentences:
        print(f"[API] 실사화 모드 감지: 프롬프트 재생성 시작 (새로운 시스템 프롬프트 적용)")
        try:
            # OpenAI API 키 확인
            if not gemini_api_key:
                gemini_api_key = GEMINI_API_KEY
            if not gemini_api_key or not gemini_available:
                print("[API] 경고: Gemini API 키가 없어 프롬프트 재생성을 건너뜁니다.")
            else:
                # 프롬프트 재생성
                scene_map = split_script_into_scenes(script_text)
                scene_context = scene_map[0] if scene_map else None
                prompts_map = call_openai_for_prompts(0, sentences, mode, scene_context=scene_context, openai_api_key=gemini_api_key)
                
                # prompts_map을 prompts 리스트로 변환
                new_prompts = []
                for idx, sentence in enumerate(sentences, start=1):
                    new_prompt = prompts_map.get(idx)
                    if new_prompt:
                        new_prompts.append(new_prompt)
                    else:
                        # 매핑에 없으면 기존 프롬프트 사용 (fallback)
                        if idx - 1 < len(prompts):
                            new_prompts.append(prompts[idx - 1])
                        else:
                            new_prompts.append("")
                prompts = new_prompts
                print(f"[API] 프롬프트 재생성 완료: {len(prompts)}개 프롬프트 생성됨")
        except Exception as e:
            print(f"[API] 경고: 프롬프트 재생성 실패, 기존 프롬프트 사용: {e}")
            import traceback
            traceback.print_exc()
    
    if not script_text or not sentences or not prompts:
        print("[API] 오류: 대본, 문장, 프롬프트가 모두 필요합니다.")
        return jsonify({"error": "대본, 문장, 프롬프트가 모두 필요합니다."}), 400
    
    if len(sentences) != len(prompts):
        print(f"[API] 오류: 문장({len(sentences)})과 프롬프트({len(prompts)})의 개수가 일치하지 않습니다.")
        return jsonify({"error": "문장과 프롬프트의 개수가 일치하지 않습니다."}), 400
    
    # 중복 요청 방지: 같은 요청이 5초 이내에 들어오면 무시
    import hashlib
    request_hash = hashlib.md5(
        (script_text + str(sentences) + str(prompts) + mode).encode('utf-8')
    ).hexdigest()
    
    current_time = time.time()
    global _recent_image_requests
    with _recent_requests_lock:
        if request_hash in _recent_image_requests:
            last_request_time = _recent_image_requests[request_hash]
            time_since_last = current_time - last_request_time
            if time_since_last < 5.0:  # 5초 이내 중복 요청
                print(f"[API] 중복 요청 감지: {time_since_last:.2f}초 전 동일한 요청이 있었습니다. 무시합니다.")
                return jsonify({"error": "중복 요청입니다. 잠시 후 다시 시도해주세요."}), 429
        _recent_image_requests[request_hash] = current_time
        # 오래된 요청 기록 정리 (1분 이상 된 것)
        _recent_image_requests = {
            k: v for k, v in _recent_image_requests.items() 
            if current_time - v < 60.0
        }
    
    job_id = create_job_record()
    print(f"[API] 새 job_id 생성: {job_id}")
    
    # 진행도 초기화
    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            job["script_text"] = script_text
            job["sentences"] = sentences
            job["prompts"] = prompts
            job["image_files"] = {}
            job["mode"] = mode
            job["test_mode"] = test_mode
            job["progress"] = []
            job["status"] = "running"
            job["stage_progress"] = 0
            job["current_stage"] = "이미지 생성 중"
            job["stage_total"] = 1
    
    assets_folder = os.path.join(ASSETS_BASE_FOLDER, job_id)
    os.makedirs(assets_folder, exist_ok=True)
    
    def generate_images_with_progress():
        print("=" * 80)
        print("=== [DEBUG] 이미지 생성 시작 ===")
        print(f"=== [DEBUG] 총 {len(sentences)}개 이미지 생성 예정 ===")
        print(f"=== [DEBUG] 테스트 모드: {test_mode} ===")
        print("=" * 80)
        try:
            total = len(sentences)
            image_results = []
            image_files = {}
            
            log_debug(f"[이미지 생성 시작] 총 {total}개 이미지 생성 예정 (테스트 모드: {test_mode})")
            log_debug(f"[환경 정보] OS: {platform.system()}, Python: {sys.version}")
            log_debug(f"[경로 정보] assets_folder: {assets_folder}")
            log_debug(f"[경로 정보] STATIC_FOLDER: {STATIC_FOLDER}")
            log_debug(f"[경로 정보] ASSETS_BASE_FOLDER: {ASSETS_BASE_FOLDER}")
            log_debug(f"[API 키] replicate_api_key 존재: {bool(replicate_api_key)}")
            
            # Windows 경로 확인
            if platform.system().lower() == "windows":
                log_debug(f"[Windows 경로] assets_folder 존재: {os.path.exists(assets_folder)}")
                log_debug(f"[Windows 경로] assets_folder 쓰기 가능: {os.access(assets_folder, os.W_OK)}")
            
            for idx, (sentence, prompt) in enumerate(zip(sentences, prompts), start=1):
                try:
                    # 체크박스 상태 확인 (인덱스는 0부터 시작)
                    use_banana_pro = (idx - 1 < len(banana_pro_flags) and banana_pro_flags[idx - 1]) if banana_pro_flags else False
                    use_banana_normal = (idx - 1 < len(banana_normal_flags) and banana_normal_flags[idx - 1]) if banana_normal_flags else False
                    
                    # 진행도 업데이트 (시작)
                    progress_pct = int(((idx - 1) / total) * 100)
                    with jobs_lock:
                        job_data = jobs.get(job_id)
                        if job_data is not None:
                            if use_banana_pro:
                                model_name = "google/nano-banana-pro"
                            elif use_banana_normal:
                                model_name = "google/nano-banana"
                            else:
                                model_name = mode
                            job_data["stage_progress"] = progress_pct
                            job_data["current_stage"] = f"이미지 생성 중... ({idx}/{total})"
                            job_data["progress"].append(f"{idx}번 이미지 생성 시작... (모델: {model_name})")
                            job_data["status"] = "running"  # 상태를 명시적으로 running으로 설정
                    
                    log_debug(f"[이미지 생성] {idx}/{total} 시작 - 프롬프트: {prompt[:50]}..., PRO: {use_banana_pro}, 노말: {use_banana_normal}")
                    
                    # Windows 경로 처리 개선
                    image_filename = os.path.join(assets_folder, f"scene_{idx}_image.png")
                    image_filename = os.path.normpath(image_filename)  # 경로 정규화
                    log_debug(f"[경로] image_filename: {image_filename}")
                    
                    # 테스트 모드일 때 photo 폴더의 이미지 사용
                    if test_mode:
                        photo_folder = resource_path("photo")
                        # scene 번호에 맞는 이미지 찾기 (순환 사용)
                        photo_image_path = os.path.join(photo_folder, f"scene_{((idx - 1) % 167) + 1}_image.png")
                        if os.path.exists(photo_image_path):
                            import shutil
                            shutil.copy2(photo_image_path, image_filename)
                            print(f"[테스트 모드] photo 폴더 이미지 사용: {photo_image_path} -> {image_filename}")
                            with jobs_lock:
                                job_data = jobs.get(job_id)
                                if job_data is not None:
                                    job_data["progress"].append(f"{idx}번 이미지 복사 완료 (테스트 모드)")
                        else:
                            # photo 폴더에 이미지가 없으면 기본 이미지 사용
                            Image.new("RGB", (1920, 1080), color="black").save(image_filename)
                            print(f"[테스트 모드] photo 폴더 이미지 없음, 기본 이미지 생성: {photo_image_path}")
                            with jobs_lock:
                                job_data = jobs.get(job_id)
                                if job_data is not None:
                                    job_data["progress"].append(f"{idx}번 이미지 기본 이미지 생성 (테스트 모드)")
                    else:
                        # 이미지 생성 실행
                        log_debug(f"[이미지 생성] {idx}번 이미지 생성 함수 호출 시작")
                        try:
                            success = generate_image(prompt, image_filename, mode=mode, replicate_api_key=replicate_api_key, custom_style_prompt=custom_style_prompt, use_banana_pro_model=use_banana_pro, use_banana_normal_model=use_banana_normal)
                            log_debug(f"[이미지 생성] {idx}번 generate_image 반환값: {success}")
                        except Exception as gen_exc:
                            log_error(f"[이미지 생성] {idx}번 generate_image 예외 발생", exc_info=gen_exc)
                            success = False
                        
                        # 생성 결과 확인
                        file_exists = os.path.exists(image_filename)
                        log_debug(f"[이미지 생성] {idx}번 파일 존재 여부: {file_exists}, 경로: {image_filename}")
                        
                        if not success or not file_exists:
                            log_error(f"[경고] {idx}번 이미지 생성 실패 또는 파일 없음 (success={success}, exists={file_exists}), 기본 이미지 생성")
                            try:
                                Image.new("RGB", (1920, 1080), color="black").save(image_filename)
                                log_debug(f"[이미지 생성] {idx}번 기본 이미지 생성 완료")
                            except Exception as save_exc:
                                log_error(f"[이미지 생성] {idx}번 기본 이미지 저장 실패", exc_info=save_exc)
                            with jobs_lock:
                                job_data = jobs.get(job_id)
                                if job_data is not None:
                                    job_data["progress"].append(f"{idx}번 이미지 생성 실패 (기본 이미지 사용)")
                        else:
                            file_size = os.path.getsize(image_filename)
                            log_debug(f"[이미지 생성] {idx}/{total} 완료 - 파일 크기: {file_size} bytes")
                            # 진행도 업데이트 (완료)
                            progress_pct = int((idx / total) * 100)
                            with jobs_lock:
                                job_data = jobs.get(job_id)
                                if job_data is not None:
                                    job_data["stage_progress"] = progress_pct
                                    job_data["progress"].append(f"{idx}번 이미지 생성 완료 ({file_size} bytes)")
                    
                    # Windows 경로 처리 개선
                    abs_path = os.path.abspath(image_filename)
                    image_files[idx] = abs_path
                    rel_path = os.path.relpath(abs_path, STATIC_FOLDER)
                    # Windows 경로 구분자를 /로 변환
                    rel_path = rel_path.replace(os.sep, "/")
                    # url_for 대신 직접 경로 구성 (백그라운드 스레드에서 안전)
                    image_url = "/static/" + rel_path
                    image_results.append({"index": idx, "sentence": sentence, "prompt": prompt, "image_url": image_url})
                    
                except Exception as img_exc:
                    # 개별 이미지 생성 실패 시에도 계속 진행
                    print(f"[경고] {idx}번 이미지 생성 중 예외 발생: {img_exc}")
                    import traceback
                    traceback.print_exc()
                    
                    # 기본 이미지 생성
                    image_filename = os.path.join(assets_folder, f"scene_{idx}_image.png")
                    try:
                        Image.new("RGB", (1920, 1080), color="black").save(image_filename)
                    except:
                        pass
                    
                    with jobs_lock:
                        job_data = jobs.get(job_id)
                        if job_data is not None:
                            job_data["progress"].append(f"{idx}번 이미지 생성 중 오류 발생: {str(img_exc)}")
                    
                    # 파일 정보는 추가
                    if os.path.exists(image_filename):
                        abs_path = os.path.abspath(image_filename)
                        image_files[idx] = abs_path
                        rel_path = os.path.relpath(abs_path, STATIC_FOLDER)
                        image_url = "/static/" + rel_path.replace(os.sep, "/")
                        image_results.append({"index": idx, "sentence": sentence, "prompt": prompt, "image_url": image_url})
            
            print(f"[이미지 생성 완료] 총 {len(image_results)}개 이미지 생성됨")
            
            with jobs_lock:
                job_data = jobs.get(job_id)
                if job_data is not None:
                    job_data["image_files"] = image_files
                    job_data["stage_progress"] = 100
                    job_data["current_stage"] = "이미지 생성 완료"
                    job_data["status"] = "completed"
                    job_data["progress"].append(f"모든 이미지 생성 완료 ({len(image_results)}/{total})")
        except Exception as exc:
            print(f"[경고] 이미지 생성 전체 실패: {exc}")
            import traceback
            traceback.print_exc()
            with jobs_lock:
                job_data = jobs.get(job_id)
                if job_data is not None:
                    job_data["status"] = "error"
                    job_data["error"] = str(exc)
                    job_data["current_stage"] = "이미지 생성 실패"
                    job_data["progress"].append(f"이미지 생성 전체 실패: {str(exc)}")
    
    # 백그라운드 스레드로 실행
    thread = threading.Thread(target=generate_images_with_progress, daemon=True)
    thread.start()
    
    # 즉시 반환 (진행도는 job_status로 확인)
    return jsonify({"job_id": job_id, "status": "processing"})


@app.route("/regenerate_image", methods=["POST"])
def api_regenerate_image():
    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or "").strip()
    scene_index = data.get("scene_index")
    if not job_id or not isinstance(scene_index, int) or scene_index < 1:
        return jsonify({"error": "job_id와 scene_index가 필요합니다."}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다."}), 404

    sentences = job.get("sentences") or []
    prompts = job.get("prompts") or []
    mode = job.get("mode") or "animation"
    if not sentences or not prompts or scene_index > len(prompts):
        return jsonify({"error": "프롬프트가 준비되지 않았습니다."}), 400

    prompt_text = prompts[scene_index - 1]
    assets_folder = os.path.join(ASSETS_BASE_FOLDER, job_id)
    os.makedirs(assets_folder, exist_ok=True)
    image_filename = os.path.join(assets_folder, f"scene_{scene_index}_image.png")

    success = generate_image(prompt_text, image_filename, mode=mode)
    if not success or not os.path.exists(image_filename):
        Image.new("RGB", (1920, 1080), color="black").save(image_filename)

    abs_path = os.path.abspath(image_filename)
    with jobs_lock:
        job_data = jobs.get(job_id)
        if job_data is not None:
            image_files = job_data.get("image_files") or {}
            image_files[scene_index] = abs_path
            job_data["image_files"] = image_files

    rel_path = os.path.relpath(abs_path, STATIC_FOLDER)
    image_url = url_for("static", filename=rel_path.replace(os.sep, "/")) + f"?v={int(time.time())}"
    return jsonify({"job_id": job_id, "scene_index": scene_index, "image_url": image_url})


def generate_video_from_image(image_path: str, output_path: str, prompt: str = "", replicate_api_key: Optional[str] = None) -> bool:
    """
    이미지를 비디오로 변환하는 함수 (wan-video/wan-2.2-i2v-fast 모델 사용)
    
    Args:
        image_path: 입력 이미지 파일 경로
        output_path: 출력 비디오 파일 경로
        prompt: 비디오 생성 프롬프트 (선택사항)
        replicate_api_key: Replicate API 키 (선택사항, 없으면 전역 변수 사용)
    
    Returns:
        bool: 성공 여부
    """
    try:
        if not os.path.exists(image_path):
            print(f"[비디오 생성] 이미지 파일이 존재하지 않습니다: {image_path}")
            return False
        
        # API 키 확인
        api_token = replicate_api_key or REPLICATE_API_TOKEN
        if not api_token:
            print("[비디오 생성] Replicate API 키가 설정되지 않았습니다.")
            return False
        
        # 이미지를 URL로 변환 (로컬 파일이므로 업로드 필요)
        # Replicate는 URL을 받으므로, 이미지를 임시로 서버에 호스팅하거나
        # 직접 파일을 업로드해야 합니다.
        # 여기서는 이미지 파일을 열어서 replicate에 전달합니다.
        
        print(f"[비디오 생성] 시작: {image_path} -> {output_path}")
        print(f"[비디오 생성] 프롬프트: {prompt[:100] if prompt else '없음'}...")
        
        # Replicate 클라이언트 초기화
        if replicate is None:
            print("[비디오 생성] replicate 모듈을 사용할 수 없습니다.")
            return False
        
        client = replicate.Client(api_token=api_token)
        
        # wan-video/wan-2.2-i2v-fast 모델 실행
        # Replicate는 파일 경로를 직접 받을 수 있습니다
        input_data = {
            "image": open(image_path, "rb"),
        }
        
        # 프롬프트가 있으면 추가
        if prompt and prompt.strip():
            input_data["prompt"] = prompt.strip()
        
        print(f"[비디오 생성] Replicate API 호출 중...")
        try:
            output = client.run(
                "wan-video/wan-2.2-i2v-fast",
                input=input_data
            )
            
            # 출력 파일 다운로드
            print(f"[비디오 생성] 비디오 다운로드 중...")
            # Replicate 출력은 일반적으로 URL 문자열 또는 파일 객체입니다
            if isinstance(output, str):
                # URL 문자열인 경우
                video_url = output
                response = requests.get(video_url, timeout=300)
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(response.content)
            elif hasattr(output, 'read'):
                # 파일 객체인 경우
                with open(output_path, "wb") as f:
                    for chunk in output:
                        f.write(chunk)
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                # 리스트인 경우 (첫 번째 요소가 URL)
                video_url = output[0]
                if isinstance(video_url, str) and video_url.startswith("http"):
                    response = requests.get(video_url, timeout=300)
                    response.raise_for_status()
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                else:
                    print(f"[비디오 생성] 예상치 못한 출력 형식: {type(output)}")
                    return False
            else:
                print(f"[비디오 생성] 예상치 못한 출력 형식: {type(output)}, 값: {output}")
                return False
        finally:
            # 파일 객체 닫기
            if "image" in input_data and hasattr(input_data["image"], "close"):
                input_data["image"].close()
            
            if not os.path.exists(output_path):
                print(f"[비디오 생성] 출력 파일이 생성되지 않았습니다: {output_path}")
                return False
            
            file_size = os.path.getsize(output_path)
            print(f"[비디오 생성] 완료: {output_path} ({file_size} bytes)")
            return True
            
    except Exception as exc:
        print(f"[비디오 생성] 오류 발생: {exc}")
        import traceback
        traceback.print_exc()
        return False


@app.route("/generate_video_from_image", methods=["POST"])
def api_generate_video_from_image():
    """이미지를 비디오로 변환하는 API 엔드포인트"""
    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or "").strip()
    scene_index = data.get("scene_index")
    prompt = (data.get("prompt") or "").strip()  # 비디오 생성 프롬프트 (선택사항)
    replicate_api_key = data.get("replicate_api_key") or None
    
    if not job_id or not isinstance(scene_index, int) or scene_index < 1:
        return jsonify({"error": "job_id와 scene_index가 필요합니다."}), 400
    
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다."}), 404
    
    # 이미지 파일 경로 확인
    image_files = job.get("image_files") or {}
    image_path = image_files.get(scene_index)
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": f"{scene_index}번 이미지를 찾을 수 없습니다."}), 404
    
    # 비디오 출력 경로
    assets_folder = os.path.join(ASSETS_BASE_FOLDER, job_id)
    os.makedirs(assets_folder, exist_ok=True)
    video_filename = os.path.join(assets_folder, f"scene_{scene_index}_video.mp4")
    
    # 프롬프트가 없으면 기본 프롬프트 사용
    if not prompt:
        prompts = job.get("prompts") or []
        if scene_index <= len(prompts):
            prompt = prompts[scene_index - 1]
        else:
            prompt = "A smooth, cinematic video transition with the character in motion."
    
    # 비디오 생성
    success = generate_video_from_image(image_path, video_filename, prompt=prompt, replicate_api_key=replicate_api_key)
    
    if not success or not os.path.exists(video_filename):
        return jsonify({"error": "비디오 생성에 실패했습니다."}), 500
    
    # 작업 데이터에 비디오 파일 경로 저장
    abs_path = os.path.abspath(video_filename)
    with jobs_lock:
        job_data = jobs.get(job_id)
        if job_data is not None:
            video_files = job_data.get("video_files") or {}
            video_files[scene_index] = abs_path
            job_data["video_files"] = video_files
    
    return jsonify({
        "job_id": job_id,
        "scene_index": scene_index,
        "video_path": abs_path,
        "message": "비디오가 생성되었습니다."
    })


@app.route("/check_replicate_api")
def check_replicate_api():
    """Replicate API 서버 상태 확인"""
    if not REPLICATE_API_TOKEN:
        return jsonify({"status": "error", "message": "REPLICATE_API_TOKEN이 설정되지 않았습니다."}), 500
    
    try:
        headers = {
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
        }
        
        # 1. 모델 정보 조회로 API 연결 테스트
        test_url = "https://api.replicate.com/v1/models/leonardoai/lucid-origin"
        print(f"[API 상태 확인] 모델 정보 조회: {test_url}")
        response = requests.get(test_url, headers=headers, timeout=10)
        
        result = {
            "model_check": {
                "status_code": response.status_code,
                "status": "ok" if response.status_code == 200 else "error",
                "message": "API 연결 정상" if response.status_code == 200 else f"API 오류: {response.status_code}"
            }
        }
        
        # 2. 간단한 이미지 생성 요청 테스트
        if response.status_code == 200:
            print(f"[API 상태 확인] 이미지 생성 요청 테스트...")
            request_url = "https://api.replicate.com/v1/models/leonardoai/lucid-origin/predictions"
            body = {
                "input": {
                    "prompt": "a simple red circle on white background",
                    "negative_prompt": "text, watermark",
                    "aspect_ratio": "16:9",
                    "guidance_scale": 3.5,
                    "num_inference_steps": 28,
                    "image_format": "png"
                }
            }
            create_res = requests.post(request_url, headers=headers, json=body, timeout=30)
            result["prediction_check"] = {
                "status_code": create_res.status_code,
                "status": "ok" if create_res.status_code in (200, 201) else "error",
                "message": "이미지 생성 요청 성공" if create_res.status_code in (200, 201) else f"이미지 생성 요청 실패: {create_res.status_code}"
            }
            
            if create_res.status_code >= 500:
                result["prediction_check"]["message"] = f"⚠️ Replicate API 서버 에러 ({create_res.status_code}). Cloudflare의 Internal Server Error로 인해 서비스가 일시적으로 중단되었을 수 있습니다."
        else:
            result["prediction_check"] = {
                "status_code": None,
                "status": "skipped",
                "message": "모델 정보 조회 실패로 인해 이미지 생성 요청 테스트를 건너뜁니다."
            }
        
        return jsonify(result)
    except Exception as e:
        print(f"[API 상태 확인] 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"API 상태 확인 중 오류 발생: {str(e)}"
        }), 500


@app.route("/preview_voice")
def preview_voice():
    voice_id = request.args.get("voice_id", "").strip()
    text = request.args.get("text", "").strip()
    if not text:
        text = "안녕하세요. 이 목소리의 예시입니다. 이 보이스로 영상이 생성됩니다."

    audio = generate_voice_preview_audio(voice_id, text)
    if not audio:
        return Response("미리듣기 생성 실패", status=400)
    response = Response(audio, mimetype="audio/mpeg")
    response.headers["Content-Disposition"] = "inline; filename=voice_preview.mp3"
    return response


@app.route("/job_status/<job_id>")
def job_status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다."}), 404
    video_filename = job.get("video_filename")
    # video_filename이 있고, 실제 파일이 존재할 때만 video_url 생성
    video_url = None
    if video_filename:
        # video_filename은 이미 STATIC_FOLDER 기준 상대 경로로 저장되어 있음
        # 예: generated_assets/{job_id}/final_video_{job_id}.mp4
        # 실제 파일 존재 여부 확인
        expected_path = os.path.join(STATIC_FOLDER, video_filename)
        # 경로 구분자 정규화 (Windows 호환)
        expected_path = os.path.normpath(expected_path)
        if os.path.exists(expected_path):
            # 파일이 존재할 때만 video_url 생성
            # video_filename은 이미 상대 경로이므로 그대로 사용
            video_url = url_for("download_video", job_id=job_id)
            print(f"[job_status] ✅ video_filename: {video_filename}, video_url: {video_url}")
            print(f"[job_status] ✅ 파일 경로: {expected_path}")
        else:
            # 파일이 없으면 video_filename을 None으로 처리
            print(f"[job_status] ⚠️ video_filename이 설정되어 있지만 파일이 존재하지 않음: {expected_path}")
            print(f"[job_status] ⚠️ STATIC_FOLDER: {STATIC_FOLDER}")
            print(f"[job_status] ⚠️ video_filename (상대 경로): {video_filename}")
            video_filename = None
    else:
        print(f"[job_status] ⚠️ video_filename이 None입니다. (작업 진행 중 또는 오류)")
    
    # 프롬프트와 이미지 데이터도 포함
    result = {
        "status": job.get("status"),
        "progress": job.get("progress") or [],
        "error": job.get("error"),
        "video_url": video_url,
        "stage_progress": job.get("stage_progress", 0),
        "current_stage": job.get("current_stage", ""),
        "stage_total": job.get("stage_total", 3),
    }
    
    # 프롬프트 생성 완료 시 프롬프트 데이터 포함
    if job.get("prompts") and job.get("sentences"):
        result["prompts"] = job.get("prompts")
        result["sentences"] = job.get("sentences")
    
    # 이미지 생성 완료 시 이미지 파일 정보 포함
    if job.get("image_files"):
        result["image_files"] = job.get("image_files")
    
    # 최종 대본 생성 결과 포함 (비동기 작업용)
    if job.get("final_script"):
        result["final_script"] = job.get("final_script")
    if job.get("visual_prompt"):  # full_response 저장용
        result["visual_prompt"] = job.get("visual_prompt")
    if job.get("image_prompts"):
        result["image_prompts"] = job.get("image_prompts")
    if job.get("thumbnail_prompt"):
        result["thumbnail_prompt"] = job.get("thumbnail_prompt")
    
    return jsonify(result)


@app.route("/static/<path:filename>")
def static_files(filename):
    """정적 파일 제공 (비디오 파일은 다운로드, 기타는 브라우저에서 표시)"""
    
    # 경로 정규화 (보안: 상위 디렉토리 접근 방지)
    filename = os.path.normpath(filename).lstrip('/')
    if '..' in filename or filename.startswith('/'):
        return jsonify({"error": "잘못된 파일 경로입니다."}), 400
    
    # STATIC_FOLDER 내에서 파일 찾기
    file_path = os.path.join(STATIC_FOLDER, filename)
    
    # 파일이 존재하는지 확인
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f"[다운로드 오류] 파일을 찾을 수 없습니다: {file_path}")
        print(f"[다운로드 오류] 요청된 파일명: {filename}")
        print(f"[다운로드 오류] STATIC_FOLDER: {STATIC_FOLDER}")
        
        # STATIC_FOLDER에 있는 모든 .mp4 파일 목록 출력 (디버깅용)
        if os.path.exists(STATIC_FOLDER):
            mp4_files = [f for f in os.listdir(STATIC_FOLDER) if f.endswith('.mp4')]
            print(f"[다운로드 오류] STATIC_FOLDER에 있는 MP4 파일들: {mp4_files[:10]}")  # 최대 10개만
        
        # HTML이 아닌 JSON으로 명확하게 반환
        response = jsonify({"error": f"파일을 찾을 수 없습니다: {filename}"})
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response, 404
    
    # 비디오 파일인 경우 다운로드로 처리
    # 중요: 이 엔드포인트는 다운로드만 수행하며, 이미지 생성이나 다른 작업을 트리거하지 않습니다.
    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        print(f"[다운로드] 비디오 파일 다운로드: {filename} (경로: {file_path})")
        print(f"[다운로드] 주의: 이 엔드포인트는 이미지 생성을 절대 트리거하지 않습니다.")
        try:
            # 다운로드 폴더 경로 확인 및 파일 복사
            download_folder_path = config_manager.get("download_folder_path", "").strip()
            if download_folder_path and os.path.isdir(download_folder_path):
                try:
                    dest_path = os.path.join(download_folder_path, os.path.basename(filename))
                    shutil.copy2(file_path, dest_path)
                    print(f"[다운로드] 파일이 다운로드 폴더로 복사됨: {dest_path}")
                except Exception as copy_error:
                    print(f"[경고] 다운로드 폴더로 복사 실패: {copy_error}")
            
            # send_file을 사용하여 명시적으로 다운로드 설정 (재생 방지)
            # 중요: 이 엔드포인트는 이미지 생성을 절대 트리거하지 않습니다.
            response = send_file(
                file_path,
                mimetype='application/octet-stream',  # video/mp4 대신 octet-stream 사용하여 재생 방지
                as_attachment=True,
                download_name=os.path.basename(filename)
            )
            # Content-Disposition 헤더 명시적으로 설정
            response.headers['Content-Disposition'] = f'attachment; filename="{os.path.basename(filename)}"'
            response.headers['Content-Type'] = 'application/octet-stream'
            print(f"[다운로드] 파일 전송 완료: {filename}")
            return response
        except Exception as e:
            print(f"[다운로드 오류] 파일 전송 실패: {e}")
            return jsonify({"error": f"파일 전송 실패: {str(e)}"}), 500
    else:
        # 기타 파일은 브라우저에서 표시
        return send_file(file_path)


@app.route("/download_images/<job_id>")
def download_images(job_id):
    """생성된 모든 이미지를 ZIP 파일로 다운로드"""
    try:
        # job_id로 assets_folder 경로 가져오기
        assets_folder, _, _ = get_job_paths(job_id)
        
        # assets_folder가 존재하는지 확인
        if not os.path.exists(assets_folder):
            return jsonify({"error": "이미지 폴더를 찾을 수 없습니다."}), 404
        
        # 모든 이미지 파일 찾기 (chunk 폴더 포함)
        image_files = []
        
        # 1. 메인 assets_folder에서 직접 이미지 파일 찾기
        if os.path.exists(assets_folder):
            for filename in os.listdir(assets_folder):
                if filename.endswith(('.png', '.jpg', '.jpeg')) and 'scene_' in filename:
                    file_path = os.path.join(assets_folder, filename)
                    if os.path.isfile(file_path):
                        image_files.append((file_path, filename))
        
        # 2. chunk 폴더들에서 이미지 파일 찾기
        for item in os.listdir(assets_folder):
            chunk_path = os.path.join(assets_folder, item)
            if os.path.isdir(chunk_path) and item.startswith('chunk_'):
                for filename in os.listdir(chunk_path):
                    if filename.endswith(('.png', '.jpg', '.jpeg')) and 'scene_' in filename:
                        file_path = os.path.join(chunk_path, filename)
                        if os.path.isfile(file_path):
                            # chunk 번호를 포함한 파일명으로 저장
                            chunk_name = os.path.basename(chunk_path)
                            zip_filename = f"{chunk_name}/{filename}"
                            image_files.append((file_path, zip_filename))
        
        # 이미지 파일이 없으면 오류 반환
        if not image_files:
            return jsonify({"error": "다운로드할 이미지가 없습니다."}), 404
        
        # ZIP 파일을 메모리에 생성
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path, zip_filename in image_files:
                try:
                    zip_file.write(file_path, zip_filename)
                except Exception as e:
                    print(f"[경고] 이미지 파일 추가 실패: {file_path} - {e}")
                    continue
        
        zip_buffer.seek(0)
        
        # ZIP 파일 다운로드
        zip_filename = f"images_{job_id}.zip"
        
        # 다운로드 폴더 경로 확인 및 파일 복사
        download_folder_path = config_manager.get("download_folder_path", "").strip()
        if download_folder_path and os.path.isdir(download_folder_path):
            try:
                # 임시 파일로 ZIP 저장
                temp_zip_path = os.path.join(download_folder_path, zip_filename)
                with open(temp_zip_path, 'wb') as f:
                    f.write(zip_buffer.getvalue())
                zip_buffer.seek(0)  # 버퍼를 다시 처음으로
                print(f"[다운로드] ZIP 파일이 다운로드 폴더로 복사됨: {temp_zip_path}")
            except Exception as copy_error:
                print(f"[경고] 다운로드 폴더로 복사 실패: {copy_error}")
        
        response = send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
        response.headers['Content-Disposition'] = f'attachment; filename="{zip_filename}"'
        return response
    
    except Exception as e:
        print(f"[다운로드 오류] 이미지 ZIP 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"이미지 다운로드 실패: {str(e)}"}), 500


@app.route("/download_subtitle/<job_id>")
def download_subtitle(job_id):
    """생성된 SRT 자막 파일 다운로드"""
    try:
        # job_id로 subtitle_file 경로 가져오기
        _, subtitle_file, _ = get_job_paths(job_id)
        
        # SRT 파일이 존재하는지 확인
        if not os.path.exists(subtitle_file):
            return jsonify({"error": "자막 파일을 찾을 수 없습니다."}), 404
        
        # 파일 크기 확인
        file_size = os.path.getsize(subtitle_file)
        if file_size == 0:
            return jsonify({"error": "자막 파일이 비어있습니다."}), 404
        
        print(f"[SRT 다운로드] 파일: {subtitle_file} (크기: {file_size} bytes)")
        
        # SRT 파일 다운로드
        srt_filename = f"subtitles_{job_id}.srt"
        
        # 다운로드 폴더 경로 확인 및 파일 복사
        download_folder_path = config_manager.get("download_folder_path", "").strip()
        if download_folder_path and os.path.isdir(download_folder_path):
            try:
                dest_path = os.path.join(download_folder_path, srt_filename)
                shutil.copy2(subtitle_file, dest_path)
                print(f"[다운로드] SRT 파일이 다운로드 폴더로 복사됨: {dest_path}")
            except Exception as copy_error:
                print(f"[경고] 다운로드 폴더로 복사 실패: {copy_error}")
        
        response = send_file(
            subtitle_file,
            mimetype='text/plain; charset=utf-8',
            as_attachment=True,
            download_name=srt_filename
        )
        response.headers['Content-Disposition'] = f'attachment; filename="{srt_filename}"'
        return response
    
    except Exception as e:
        print(f"[다운로드 오류] SRT 파일 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"자막 다운로드 실패: {str(e)}"}), 500


@app.route("/api/debug/info", methods=["GET"])
def api_debug_info():
    """디버깅 정보 반환"""
    try:
        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": sys.version,
            "log_file": LOG_FILE,
            "log_dir": LOG_DIR,
            "static_folder": STATIC_FOLDER,
            "assets_base_folder": ASSETS_BASE_FOLDER,
            "static_folder_exists": os.path.exists(STATIC_FOLDER),
            "static_folder_writable": os.access(STATIC_FOLDER, os.W_OK) if os.path.exists(STATIC_FOLDER) else False,
            "replicate_api_available": bool(REPLICATE_API_TOKEN),
            "elevenlabs_api_available": bool(ELEVENLABS_API_KEY),
            "gemini_api_available": bool(GEMINI_API_KEY) and gemini_available,
            "ffmpeg_path": FFMPEG_BINARY,
            "ffmpeg_exists": os.path.exists(FFMPEG_BINARY) if FFMPEG_BINARY else False,
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/log", methods=["GET"])
def api_debug_log():
    """최근 로그 파일 내용 반환"""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 최근 100줄만 반환
                recent_lines = lines[-100:] if len(lines) > 100 else lines
                return jsonify({
                    "log_file": LOG_FILE,
                    "total_lines": len(lines),
                    "recent_lines": recent_lines
                })
        else:
            return jsonify({"error": "로그 파일이 없습니다.", "log_file": LOG_FILE}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download_video/<job_id>")
def download_video(job_id):
    """생성된 최종 비디오 파일 다운로드
    
    중요: 이 엔드포인트는 다운로드만 수행하며, 이미지 생성이나 다른 작업을 트리거하지 않습니다.
    """
    print(f"[다운로드] 비디오 다운로드 요청: job_id={job_id}")
    print(f"[다운로드] 주의: 이 엔드포인트는 이미지 생성을 절대 트리거하지 않습니다.")
    try:
        _, _, video_file = get_job_paths(job_id)
        if not os.path.exists(video_file):
            print(f"[다운로드 오류] 영상 파일을 찾을 수 없습니다: {video_file}")
            return jsonify({"error": "영상 파일을 찾을 수 없습니다."}), 404

        if os.path.getsize(video_file) == 0:
            print(f"[다운로드 오류] 영상 파일이 비어 있습니다: {video_file}")
            return jsonify({"error": "영상 파일이 비어 있습니다."}), 404

        download_name = os.path.basename(video_file)
        print(f"[다운로드] 파일 다운로드 시작: {download_name} (크기: {os.path.getsize(video_file)} bytes)")
        
        # 다운로드 폴더 경로 확인 및 파일 복사
        download_folder_path = config_manager.get("download_folder_path", "").strip()
        if download_folder_path and os.path.isdir(download_folder_path):
            try:
                dest_path = os.path.join(download_folder_path, download_name)
                shutil.copy2(video_file, dest_path)
                print(f"[다운로드] 파일이 다운로드 폴더로 복사됨: {dest_path}")
            except Exception as copy_error:
                print(f"[경고] 다운로드 폴더로 복사 실패: {copy_error}")
        
        # 다운로드만 되도록 명시적으로 설정
        # 중요: 이 엔드포인트는 이미지 생성을 절대 트리거하지 않습니다.
        response = send_file(
            video_file,
            mimetype="application/octet-stream",  # video/mp4 대신 octet-stream 사용하여 재생 방지
            as_attachment=True,
            download_name=download_name,
        )
        # Content-Disposition 헤더 명시적으로 설정 (재생 방지)
        response.headers['Content-Disposition'] = f'attachment; filename="{download_name}"'
        response.headers['Content-Type'] = 'application/octet-stream'
        print(f"[다운로드] 파일 전송 완료: {download_name}")
        return response
    except Exception as e:
        print(f"[다운로드 오류] 비디오 파일 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"영상 다운로드 실패: {str(e)}"}), 500


def find_available_port(start: int = 5001, end: int = 5010) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def run_flask_server(port: int) -> None:
    debug_mode = not getattr(sys, "frozen", False)
    app.run(host="127.0.0.1", port=port, debug=debug_mode, use_reloader=False)


# 전역 변수로 webview 창 저장
_webview_window = None

def select_folder_dialog(initial_dir=None):
    """webview에서 호출할 수 있는 폴더 선택 함수"""
    global _webview_window
    try:
        if _webview_window:
            # webview의 create_file_dialog 사용
            result = webview.create_file_dialog(
                webview.FOLDER_DIALOG,
                directory=initial_dir if initial_dir else None
            )
            if result and len(result) > 0:
                return result[0]
        return None
    except Exception as e:
        print(f"[폴더 선택 오류] {e}")
        import traceback
        traceback.print_exc()
        return None


# -----------------------------------------------------------------------------
# [추가] 유튜브 제목/내용 생성기 API
# -----------------------------------------------------------------------------
@app.route("/api/generate_metadata", methods=["POST"])
def api_generate_metadata():
    """유튜브 제목, 설명, 태그, 썸네일 가이드 생성 API"""
    data = request.get_json(silent=True) or {}
    script_text = (data.get("script") or "").strip()
    
    if not script_text:
        return jsonify({"error": "분석할 대본이 없습니다."}), 400
    
    # API 키 확인
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API 키가 설정되지 않았습니다."}), 500

    # 사용자 정의 프롬프트 (Growth Hacker Persona)
    system_instruction = """
## Role & Identity

당신은 '경제, 역사, 지식 스토리텔링' 전문 유튜브 채널의 **[최고 성장 전략가(Growth Hacker)]**입니다.

특히 구독자가 0명인 '신규 채널(Cold Start)'을 '검색 유입(SEO)'과 '높은 클릭률(CTR)'을 통해 성장시키는 데 특화되어 있습니다.

## Channel Identity (Context)

이 채널은 단순한 경제 뉴스나 역사 강의가 아닙니다.

1. **관점의 차별화:** 뇌과학, 진화심리학, 행동경제학, 역사적 이면을 통해 현상을 분석합니다.

2. **타겟 오디언스:** 돈을 잃고 불안해하는 투자자, 세상의 숨겨진 원리를 알고 싶은 지적 호기심이 강한 사람들입니다.

3. **톤앤매너:** 통찰력 있고(Insightful), 날카로우며, 때로는 팩트 폭행을 하지만 결국엔 위로와 해결책을 줍니다.

## Task Instructions

사용자가 [영상 대본]을 입력하면, 다음 4단계 프로세스를 거쳐 결과를 출력하십시오.

### 1단계: 핵심 가치 분석

- 대본에서 가장 시청자의 흥미를 끌 만한 '후킹 포인트'와 '검색될 만한 키워드'를 추출합니다.

- 시청자가 얻어갈 '효용(Benefit)'을 정의합니다.

### 2단계: 필살기 제목 제안 (3가지 옵션)

- 옵션 1 [검색 적중형]: SEO 최우선 (구체적 질문, 키워드 조합)

- 옵션 2 [호기심/후킹형]: 심리 자극 ("당신만 몰랐던", "충격적인 진실" 등)

- 옵션 3 [권위/스토리형]: 인물/사건 빗대기 (스토리텔링 강조)

### 3단계: 노출 최적화 설명란 (Description) 작성

- 첫 2줄(Hook): 클릭을 유도하는 가장 중요한 문장.

- 본문: 대본 내용 요약.

- 타임라인: 챕터별 시간과 소제목 생성 (예: 00:00 인트로).

- 태그: 고트래픽 해시태그 10개.

### 4단계: 썸네일 텍스트 & 이미지 가이드

- 텍스트: 10자 이내 임팩트 문구.

- 이미지 구성: 시선을 사로잡는 시각적 요소 묘사.

## Output Format (JSON)

반드시 다음 JSON 형식으로만 응답하세요:

{
  "analysis": "핵심 가치 분석 내용",
  "titles": {
    "search": "검색 적중형 제목",
    "hook": "호기심/후킹형 제목",
    "story": "권위/스토리형 제목"
  },
  "description": "설명란 전체 내용 (첫 2줄, 본문, 타임라인 포함)",
  "tags": "#태그1 #태그2 ...",
  "thumbnail": {
    "text": "썸네일 텍스트",
    "image_guide": "이미지 구성 가이드"
  }
}
"""
    
    try:
        # Gemini API 호출
        global genai_client
        if genai_client is None:
            return jsonify({"error": "Gemini Client 초기화 실패"}), 500
        
        from google.genai import types
        
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"분석할 대본:\n{script_text}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        
        result = json.loads(response.text)
        return jsonify(result)
    except Exception as e:
        print(f"[메타데이터 생성 오류] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------------
# [추가] 썸네일 제작 API
# -----------------------------------------------------------------------------
@app.route("/api/generate_thumbnail_concepts", methods=["POST"])
def api_generate_thumbnail_concepts():
    """썸네일 기획안 3가지 생성 API (Gemini 2.5 Flash)"""
    data = request.get_json(silent=True) or {}
    script_text = (data.get("script") or "").strip()
    character_setting = (data.get("character_setting") or "").strip()
    
    if not script_text:
        return jsonify({"error": "참조 대본을 입력해주세요."}), 400
    
    # API 키 확인
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API 키가 설정되지 않았습니다."}), 500

    # 시스템 프롬프트
    system_instruction = """당신은 유튜브 썸네일 전문 기획자입니다. 제공된 대본을 바탕으로 클릭을 유도하는 썸네일 구성 3가지를 제안하세요.

사용자가 입력한 '캐릭터/모델 설정'이 있다면 이를 반영하세요.

각 옵션은 다음 형식을 엄격히 준수해야 합니다:

1. **상황 묘사:** 배경, 인물 표정, 주요 오브젝트 설명
2. **텍스트(카피):** 10자 이내의 강력한 한글 멘트 (가독성 고려, 색상 추천 포함)
3. **나노바나나 PRO용 프롬프트:** AI 이미지 생성 툴에 바로 넣을 수 있는 **영어 프롬프트**를 작성하세요. 
   - 스타일(Cinematic, Hyper-realistic), 조명(Dramatic lighting), 표정(Shocked face), 구도(Close-up), 화질(8k, high detail) 등의 파라미터를 반드시 포함해야 합니다.

반드시 다음 JSON 형식으로만 응답하세요:
{
  "concepts": [
    {
      "situation": "상황 묘사 내용",
      "copy": "카피 텍스트",
      "prompt": "영어 프롬프트"
    },
    {
      "situation": "상황 묘사 내용",
      "copy": "카피 텍스트",
      "prompt": "영어 프롬프트"
    },
    {
      "situation": "상황 묘사 내용",
      "copy": "카피 텍스트",
      "prompt": "영어 프롬프트"
    }
  ]
}"""
    
    try:
        global genai_client
        if genai_client is None:
            return jsonify({"error": "Gemini Client 초기화 실패"}), 500
        
        from google.genai import types
        
        # 사용자 프롬프트 구성
        user_prompt = f"""다음 대본을 분석하여 썸네일 기획안 3가지를 제안해주세요.

대본:
{script_text}

"""
        if character_setting:
            user_prompt += f"""캐릭터/모델 설정:
{character_setting}

"""
        user_prompt += "위 정보를 바탕으로 클릭률을 높일 수 있는 썸네일 기획안 3가지를 제안해주세요."
        
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',  # 기존 코드와 동일한 모델 사용
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        
        result = json.loads(response.text)
        return jsonify(result)
    except Exception as e:
        print(f"[썸네일 기획안 생성 오류] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate_thumbnail_image", methods=["POST"])
def api_generate_thumbnail_image():
    """썸네일 이미지 생성 API (Gemini 3 Pro Image Preview)"""
    try:
        data = request.get_json(silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        copy_text = (data.get("copy_text") or "").strip()  # 카피 텍스트
        reference_image_base64 = data.get("reference_image")  # base64 인코딩된 이미지
        
        if not prompt:
            return jsonify({"error": "프롬프트를 입력해주세요."}), 400
        
        # 카피 텍스트가 있으면 프롬프트에 포함
        if copy_text:
            # 프롬프트에 텍스트 삽입 지시사항 추가
            enhanced_prompt = f"""{prompt}

IMPORTANT: Add the following text overlay on the image in a prominent, readable font:
Text to display: "{copy_text}"
- Place the text in a visible area (top, center, or bottom)
- Use high contrast colors (white text on dark background or vice versa)
- Make the text bold and large enough to be clearly readable
- Ensure the text does not interfere with the main visual elements but is clearly visible"""
            prompt = enhanced_prompt
        
        # API 키 확인
        if not GEMINI_API_KEY:
            return jsonify({"error": "Gemini API 키가 설정되지 않았습니다."}), 500

        global genai_client
        if genai_client is None:
            return jsonify({"error": "Gemini Client 초기화 실패"}), 500
        
        from google.genai import types
        import base64
        from PIL import Image
        from io import BytesIO
        
        # 참조 이미지 처리
        reference_images = []
        if reference_image_base64:
            try:
                # base64 디코딩
                image_data = base64.b64decode(reference_image_base64.split(',')[1] if ',' in reference_image_base64 else reference_image_base64)
                reference_img = Image.open(BytesIO(image_data))
                reference_images = [reference_img]
            except Exception as img_error:
                print(f"[참조 이미지 처리 오류] {img_error}")
                # 이미지 처리 실패 시 무시하고 진행
        
        # Gemini 3 Pro Image Preview 모델 사용
        # contents는 프롬프트 + 참조 이미지 리스트
        contents = [prompt] + reference_images
        
        response = genai_client.models.generate_content(
            model='gemini-3-pro-image-preview',
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",  # 유튜브 썸네일 비율
                    image_size="2K",  # 2K 해상도
                ),
                temperature=0.7
            )
        )
        
        # 응답에서 이미지 추출
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data'):
                        # 이미지 데이터를 base64 문자열로 변환
                        image_data = part.inline_data.data
                        # bytes 타입이면 base64로 인코딩
                        if isinstance(image_data, bytes):
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                        else:
                            # 이미 문자열이면 그대로 사용
                            image_base64 = image_data
                        return jsonify({
                            "image": image_base64,
                            "mime_type": part.inline_data.mime_type or "image/png"
                        })
        
        # 이미지가 없으면 오류
        return jsonify({"error": "이미지 생성에 실패했습니다. 응답에 이미지가 포함되지 않았습니다."}), 500
        
    except Exception as e:
        print(f"[썸네일 이미지 생성 오류] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/download_thumbnail", methods=["POST"])
def api_download_thumbnail():
    """썸네일 이미지 다운로드 API (서버에 직접 저장 후 결과 반환)"""
    try:
        data = request.get_json(silent=True) or {}
        image_base64 = data.get("image")
        mime_type = data.get("mime_type") or "image/png"
        
        if not image_base64:
            return jsonify({"error": "이미지 데이터가 없습니다."}), 400
        
        # base64 디코딩
        import base64
        from io import BytesIO
        
        try:
            # data:image/png;base64, 접두사가 있는 경우 제거
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            image_data = base64.b64decode(image_base64)
        except Exception as decode_err:
            print(f"[썸네일 다운로드] base64 디코딩 오류: {decode_err}")
            return jsonify({"error": f"이미지 디코딩 실패: {decode_err}"}), 400
        
        # 파일 확장자 결정
        extension = "png"
        if "jpeg" in mime_type or "jpg" in mime_type:
            extension = "jpg"
        
        filename = f"thumbnail_{int(time.time())}.{extension}"
        
        # 1. 설정된 다운로드 폴더가 있는지 확인
        download_folder_path = config_manager.get("download_folder_path", "").strip()
        
        # 2. 폴더가 설정되어 있고 유효하면 직접 저장 (가장 확실한 방법)
        if download_folder_path and os.path.isdir(download_folder_path):
            try:
                save_path = os.path.join(download_folder_path, filename)
                with open(save_path, "wb") as f:
                    f.write(image_data)
                
                print(f"[썸네일] 파일이 설정된 폴더로 저장됨: {save_path}")
                # 성공 시 JSON으로 경로 반환 (브라우저 다운로드 창을 띄우지 않음)
                return jsonify({
                    "status": "success",
                    "message": "이미지가 저장되었습니다.",
                    "path": save_path
                })
            except Exception as save_err:
                print(f"[썸네일] 파일 저장 실패: {save_err}")
                # 저장 실패 시 아래의 스트림 전송으로 넘어감
        
        # 3. 폴더 설정이 없거나 실패한 경우 기존 방식(스트림 전송) 사용
        file_obj = BytesIO(image_data)
        file_obj.seek(0)
        
        return send_file(
            file_obj,
            mimetype=mime_type,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"[썸네일 다운로드 오류] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    selected_port = find_available_port()
    flask_thread = threading.Thread(target=run_flask_server, args=(selected_port,), daemon=True)
    flask_thread.start()

    window_url = f"http://127.0.0.1:{selected_port}"
    print(f"\n{'=' * 60}")
    print("유튜브 영상 생성기 서버 시작")
    print(f"주소: {window_url}")
    print(f"{'=' * 60}\n")

    # Flask 서버가 완전히 시작될 때까지 대기
    import time
    max_wait = 10  # 최대 10초 대기
    wait_count = 0
    while wait_count < max_wait:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(("127.0.0.1", selected_port))
            sock.close()
            if result == 0:
                print(f"[서버 준비 완료] 포트 {selected_port}에서 서버가 응답합니다.")
                break
        except Exception:
            pass
        time.sleep(0.5)
        wait_count += 0.5
        print(f"[서버 대기 중] {wait_count:.1f}초...")
    
    if wait_count >= max_wait:
        print(f"[경고] 서버 시작 확인 실패, webview를 시작합니다.")

    _webview_window = webview.create_window("YouTube Maker", window_url, width=1280, height=800, resizable=True)
    # JavaScript에서 호출할 수 있도록 함수 등록
    # debug=True로 설정하여 webview 콘솔 로그 확인 가능
    webview.start(debug=True)
