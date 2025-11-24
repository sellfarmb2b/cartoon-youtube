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
from uuid import uuid4
from io import BytesIO
from itertools import islice
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import ffmpeg
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
from dotenv import load_dotenv


# =============================================================================
# 환경 변수 및 경로 설정
# =============================================================================

# 실행 파일 위치 기준으로 경로 설정
if getattr(sys, 'frozen', False):
    # PyInstaller로 빌드된 경우: 실행 파일이 있는 디렉토리
    EXECUTABLE_DIR = os.path.dirname(sys.executable)
else:
    # 개발 환경: 스크립트가 있는 디렉토리
    EXECUTABLE_DIR = os.path.dirname(os.path.abspath(__file__))

# .env(.local) 파일 로드
dotenv_candidates = [
    os.path.join(EXECUTABLE_DIR, ".env.local"),
    os.path.join(EXECUTABLE_DIR, ".env"),
]
for dotenv_path in dotenv_candidates:
    if dotenv_path and os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=False)
# 마지막으로 시스템 환경 변수도 로드 시도 (이미 설정된 값은 유지)
load_dotenv(override=False)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

STATIC_FOLDER = os.path.join(EXECUTABLE_DIR, "static")
ASSETS_BASE_FOLDER = os.path.join(STATIC_FOLDER, "generated_assets")
FINAL_VIDEO_BASE_NAME = "final_video"
SUBTITLE_BASE_NAME = "subtitles"
FONTS_FOLDER = os.path.join(STATIC_FOLDER, "fonts")
SUBTITLE_FONT_NAME = "GmarketSansTTFMedium"

os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(ASSETS_BASE_FOLDER, exist_ok=True)
os.makedirs(FONTS_FOLDER, exist_ok=True)

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

# Replicate API Rate Limiting 설정
# 문서: https://replicate.com/docs/topics/predictions/rate-limits
# - 예측 생성: 분당 600개 요청 (최소 0.1초 간격)
# - 다른 엔드포인트: 분당 3000개 요청
REPLICATE_MIN_REQUEST_INTERVAL = 0.1  # 최소 요청 간격 (초) - 분당 600개 제한 준수
REPLICATE_RATE_LIMIT_RETRY_DELAY = 30  # 429 에러 발생 시 재시도 대기 시간 (초)
_last_replicate_request_time = 0  # 마지막 Replicate API 요청 시간
_replicate_request_lock = threading.Lock()  # 요청 간격 제어를 위한 락


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


# PyInstaller 환경 감지 및 리소스 경로 설정
if getattr(sys, 'frozen', False):
    # PyInstaller로 빌드된 경우
    base_path = sys._MEIPASS
    template_folder = os.path.join(base_path, 'templates')
    static_folder = os.path.join(base_path, 'static')
else:
    # 개발 환경
    base_path = os.path.dirname(os.path.abspath(__file__))
    template_folder = os.path.join(base_path, 'templates')
    static_folder = os.path.join(base_path, 'static')

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)


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

    voice_entries = []
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        all_voices = client.voices.get_all()
        allowed_ids = set(ELEVENLABS_VOICE_IDS)
        for voice in all_voices.voices:
            if voice.voice_id in allowed_ids:
                voice_entries.append({"id": voice.voice_id, "name": voice.name or voice.voice_id})

        # 보이스 목록이 비어 있는 경우, 미리 지정한 아이디라도 그대로 사용
        known_ids = {entry["id"] for entry in voice_entries}
        for vid in ELEVENLABS_VOICE_IDS:
            if vid not in known_ids:
                voice_entries.append({"id": vid, "name": vid})
    except Exception as exc:
        print(f"[경고] ElevenLabs 보이스 목록 불러오기 실패: {exc}")
        voice_entries = [{"id": vid, "name": vid} for vid in ELEVENLABS_VOICE_IDS]

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

    if not OPENAI_API_KEY:
        return text

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a translator. Translate the text into fluent English."},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.1,
                "max_tokens": 200,
            },
            timeout=60,
        )
        if response.status_code != 200:
            print(f"[경고] 번역 실패: {response.status_code} {response.text}")
            return text
        data = response.json()
        translated = data["choices"][0]["message"]["content"].strip()
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
    if not prompt.lower().startswith(("a ", "the ")):
        prompt = prompt[0].upper() + prompt[1:]
    prompt = f"{REALISTIC_STYLE_WRAPPER}, {prompt}"
    return prompt


def enforce_prompt_by_mode(prompt: str, fallback_context: str = "", mode: str = "animation") -> str:
    mode = (mode or "animation").lower()
    if mode == "realistic":
        return enforce_realistic_prompt(prompt, fallback_context)
    return enforce_stickman_prompt(prompt, fallback_context)


def build_fallback_prompt(sentence: str, mode: str) -> str:
    sentence = ensure_english_text(sentence or "")
    description = sentence if sentence else "unnamed scene"
    if mode == "realistic":
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
    if not openai_available:
        return {}
    
    try:
        mode_instruction = ""
        if mode == "realistic":
            mode_instruction = "The images will be photorealistic/hyperrealistic. Focus on cinematic, photographic descriptions."
        else:
            mode_instruction = "The images will be in stickman cartoon, vibrant 2D illustration style."
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional script analyst and visual context extractor. "
                            "Analyze the scene description and extract key contextual information for image generation. "
                            f"{mode_instruction}\n\n"
                            "Return JSON with these fields:\n"
                            "- overall_style: The visual style/genre (e.g., '1940s Film Noir', 'cyberpunk', 'high school romance', 'hyperrealistic')\n"
                            "- location: The setting/background (e.g., 'dark narrow alleyway at night', 'sunlit classroom', 'rainy street')\n"
                            "- time: Time of day/period (e.g., 'deep night', 'sunset', 'rainy afternoon', 'dawn')\n"
                            "- characters: Key characters in the scene with brief descriptions (e.g., 'Detective Park (40s, weary, trench coat), Suji (20s, red dress)')\n"
                            "- mood: The emotional atmosphere (e.g., 'tense', 'somber', 'mysterious', 'peaceful', 'urgent')\n\n"
                            "Be specific and descriptive. Write in English only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Extract context from this scene:\n\n{scene_text[:800]}",
                    },
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
                "max_tokens": 400,
            },
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
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


def call_openai_for_prompts(offset: int, sentences: List[str], mode: str = "animation", scene_context: Dict[str, str] = None, max_retries: int = 3) -> Dict[int, str]:
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
    
    if mode == "realistic":
        style_instruction = (
            "For each sentence, convert it into a concrete, visually descriptive scene that could be captured in a photograph."
            " Mention subjects, actions, facial expressions, body language, environment, lighting, and camera framing."
            " Avoid abstract emotions or quoting dialogue. Write in English only."
            " Do not include stylistic keywords like 'hyperrealistic'; focus only on the scene description."
        )
    else:
        style_instruction = "Stick to a stickman cartoon, vibrant 2D illustration style."

    for attempt in range(max_retries):
        try:
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
            
            user_content = "[SCRIPT LINES - Sentences to Convert]\n" + script_blocks
            if context_info:
                user_content += context_info
            user_content += "\n\n[ENGLISH IMAGE PROMPT]\nReturn output strictly as JSON object with proper escaping."

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": system_content,
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7,
                    "max_tokens": 2000,
                },
                timeout=90,
            )
            if response.status_code != 200:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"OpenAI request failed: {response.status_code} {response.text}")
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
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
    if not openai_available:
        return fallback

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You split narration into subtitle-ready segments. "
                            "Return the original text divided into short, sequential phrases. "
                            "Do not paraphrase or remove words. Keep the language the same."
                            " Respond strictly as JSON with an array named 'segments', each item containing 'text'."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": 600,
            },
            timeout=60,
        )
        if response.status_code != 200:
            print(f"[경고] 자막 세그먼트 생성 실패: {response.status_code} {response.text}")
            return fallback
        data = response.json()
        content = data["choices"][0]["message"]["content"]
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


def generate_visual_prompts(sentences: List[str], mode: str = "animation", progress_cb=None, original_script: str = None) -> List[str]:
    mode = (mode or "animation").lower()
    total = len(sentences)
    if progress_cb:
        progress_cb("이미지용 프롬프트 생성을 시작합니다.")

    if not sentences:
        return []

    default_prompts = []
    for idx, sentence in enumerate(sentences):
        default_prompts.append(build_fallback_prompt(sentence, mode))

    if not OPENAI_API_KEY:
        if progress_cb:
            progress_cb("OpenAI API 키가 없어 기본 프롬프트를 사용합니다.")
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
            if mode == "realistic" and focus_sentence:
                base_prompt = f"{base_prompt}. Focus on: {focus_sentence}"
            prompts[global_idx] = enforce_prompt_by_mode(
                base_prompt, fallback_context=f"based on '{focus_sentence[:50]}'", mode=mode
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
    if not is_voice_allowed(voice_id):
        voice_id = get_default_voice_id()
    # 사용자가 입력한 API 키를 우선 사용, 없으면 기본값 사용
    api_key = elevenlabs_api_key or ELEVENLABS_API_KEY
    if not api_key:
        print("[TTS] ElevenLabs API 키가 설정되지 않았습니다.")
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
    headers = {"xi-api-key": api_key, "Accept": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",  # 더 빠르고 고품질 모델로 업그레이드
        "voice_settings": {
            "stability": 0.75,  # 안정성 증가 (0.5-1.0, 높을수록 일관성)
            "similarity_boost": 0.85,  # 유사도 증가 (0.0-1.0, 높을수록 원본 목소리에 가까움)
            "style": 0.0,
            "use_speaker_boost": True,  # 스피커 부스트 활성화
        },
        "output_format": "mp3_44100_256",  # 비트레이트 증가 (128 -> 256)로 음질 개선
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            print(f"[TTS] 실패: {resp.status_code} {resp.text}")
            return None
        data = resp.json()
        audio_b64 = data.get("audio") or data.get("audio_base64")
        alignment = data.get("alignment")
        if not audio_b64:
            return None
        audio_bytes = base64.b64decode(audio_b64)
        with open(audio_filename, "wb") as f:
            f.write(audio_bytes)
        return alignment
    except Exception as exc:
        print(f"[TTS] API 호출 실패: {exc}")
        return None


replicate_api_available = bool(REPLICATE_API_TOKEN)
# stability_api_available = bool(STABILITY_API_KEY)  # Stability API 사용 안 함
openai_available = bool(OPENAI_API_KEY)


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


def generate_image(prompt_text: str, filename: str, mode: str = "animation", replicate_api_key: Optional[str] = None) -> bool:
    mode = (mode or "animation").lower()
    fallback_context = "scene description"
    if prompt_text:
        base_prompt = enforce_prompt_by_mode(prompt_text, fallback_context=fallback_context, mode=mode)
    else:
        default_prompt = (
            "A mysterious scene involving a historic landmark attracting curiosity"
            if mode == "realistic"
            else "stickman presenting data in a colorful studio"
        )
        base_prompt = enforce_prompt_by_mode(default_prompt, fallback_context="default context", mode=mode)

    if mode == "realistic":
        negative_prompt = REALISTIC_NEGATIVE_PROMPT
    else:
        negative_prompt = (
            "realistic human, detailed human skin, photograph, 3d render, blank white background, line-art only, text, watermark"
        )

    # 사용자가 입력한 API 키를 우선 사용, 없으면 기본값 사용
    api_token = replicate_api_key or REPLICATE_API_TOKEN
    replicate_api_available_local = bool(api_token)
    
    # API 키 사용 여부 로그 (키의 일부만 표시)
    if api_token:
        key_preview = api_token[:10] + "..." + api_token[-4:] if len(api_token) > 14 else "***"
        print(f"[generate_image] 사용 중인 Replicate API 키: {key_preview}")
    else:
        print(f"[경고] Replicate API 키가 설정되지 않았습니다!")
    
    if replicate_api_available_local:
        try:
            print(f"[generate_image] Replicate API 사용 - 모드: {mode}, 파일: {os.path.basename(filename)}")
            headers = {
                "Authorization": f"Token {api_token}",
                "Content-Type": "application/json",
            }
            replicate_input = {
                "prompt": base_prompt,
                "negative_prompt": negative_prompt,
                "aspect_ratio": "16:9",
            }
            if mode == "realistic":
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
                replicate_input.update({"guidance_scale": 3.5, "num_inference_steps": 28, "image_format": "png"})
                request_url = (
                    "https://api.replicate.com/v1/models/leonardoai/lucid-origin/predictions"
                )
                body = {"input": replicate_input}

            print(f"[generate_image] Replicate API 요청 전송 중...")
            print(f"[generate_image] 프롬프트: {base_prompt[:200]}...")
            print(f"[generate_image] Negative 프롬프트: {negative_prompt[:200]}...")
            print(f"[generate_image] 요청 URL: {request_url}")
            print(f"[generate_image] 요청 본문: {json.dumps(body, indent=2, ensure_ascii=False)}")
            
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
                        reset_time = error_data.get("reset_time", REPLICATE_RATE_LIMIT_RETRY_DELAY)
                        
                        if retry_attempt < max_retries - 1:
                            wait_time = REPLICATE_RATE_LIMIT_RETRY_DELAY
                            print(f"[Rate Limit] 429 에러 발생: {error_detail}")
                            print(f"[Rate Limit] {wait_time}초 후 재시도 중... (시도 {retry_attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue  # 재시도
                        else:
                            print(f"[Rate Limit] 429 에러: 최대 재시도 횟수 초과")
                            raise Exception(f"Replicate API Rate Limit 초과: {error_detail}")
                    
                    # 200/201이 아니고 429도 아니면 루프 종료
                    if create_res.status_code not in (200, 201, 429):
                        break
                        
                except Exception as req_exc:
                    if retry_attempt < max_retries - 1 and "429" in str(req_exc):
                        wait_time = REPLICATE_RATE_LIMIT_RETRY_DELAY
                        print(f"[Rate Limit] 예외 발생, {wait_time}초 후 재시도: {req_exc}")
                        time.sleep(wait_time)
                        continue
                    print(f"[오류] Replicate API 요청 실패: {req_exc}")
                    import traceback
                    traceback.print_exc()
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
                    for poll_count in range(180):
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
                            if poll_count % 10 == 0:  # 10초마다 로그 출력
                                print(f"[generate_image] 상태 확인 중... ({poll_count}초 경과, 상태: {status})")
                            if status in ("succeeded", "failed", "canceled"):
                                final = data
                                print(f"[generate_image] 최종 상태: {status} (총 {poll_count}초 소요)")
                                if status == "failed":
                                    error_msg = data.get("error", "알 수 없는 오류")
                                    print(f"[generate_image] 실패 원인: {error_msg}")
                                break
                            time.sleep(1)
                        except Exception as poll_exc:
                            print(f"[오류] 상태 조회 중 예외 발생: {poll_exc}")
                            import traceback
                            traceback.print_exc()
                            break
                    if poll_count >= 179:
                        print(f"[generate_image] 타임아웃: 180초 동안 완료되지 않음")
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
            print(f"[IMG] (Replicate) 예외 발생: {exc}")
            import traceback
            traceback.print_exc()

    # Stability API는 사용하지 않음 (Replicate만 사용)
    # if mode != "realistic" and stability_api_available:
    #     ... (Stability API 코드 제거됨)

    # fallback: create black image
    try:
        print(f"[경고] 이미지 생성 실패, 검은색 이미지 생성: {filename}")
        black_img = Image.new("RGB", (1920, 1080), color="black")
        black_img.save(filename, format="PNG")
        
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
        print(f"[오류] 검은색 이미지 생성 실패: {fallback_exc}")
        import traceback
        traceback.print_exc()
    
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
):
    print(f"\n[generate_assets 시작] scene_offset={scene_offset}, 문장 개수={len(sentences)}, assets_folder={assets_folder}")
    print(f"  예상 scene_id 범위: {scene_offset + 1} ~ {scene_offset + len(sentences)}")
    cleanup_assets_folder(assets_folder)
    os.makedirs(assets_folder, exist_ok=True)
    image_prompts = prompts_override or generate_visual_prompts(sentences, mode=mode, progress_cb=progress_cb, original_script=original_script)
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
        img_path = existing_images.get(scene_num)
        if img_path and os.path.exists(img_path):
            shutil.copy(img_path, image_file)
            image_generated = True
        else:
            image_generated = generate_image(prompt, image_file, mode=mode, replicate_api_key=replicate_api_key)

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
            f"Style: Default,{SUBTITLE_FONT_NAME},80,&H00FFFFFF,&H000000FF,&H00000000,&HFF000000,0,0,0,0,100,100,0,0,3,10,0,2,10,10,50,1\n",
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
                # 텍스트를 하나의 연속된 문자열로 합치기 (띄어쓰기 유지)
                subtitle_text_lines.append(lines[i].rstrip())
                i += 1
            
            if subtitle_text_lines:
                # 여러 줄을 하나로 합치되, 띄어쓰기는 유지
                # ASS 형식에서는 \N으로 줄바꿈
                text = "\\N".join(subtitle_text_lines)
                # ASS 형식의 Dialogue 라인 작성
                # (BorderStyle=3이 이미 설정되어 있어 일반 공백으로도 배경 박스가 끊어지지 않음)
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

    # 메모리 효율적인 방식: 각 씬을 개별 비디오로 생성한 후 concat demuxer로 합침
    scene_video_files = []
    temp_scene_folder = os.path.join(assets_folder, "temp_scenes")
    os.makedirs(temp_scene_folder, exist_ok=True)
    
    print(f"[create_video] 총 {len(scene_data_with_timestamps)}개 씬을 개별 비디오로 생성 중...")
    
    # 각 씬을 개별 비디오로 생성
    for idx, scene in enumerate(scene_data_with_timestamps, 1):
        scene_id = scene.get('scene_id')
        image_path = scene.get("image_file")
        print(f"[씬 처리 시작] {idx}/{len(scene_data_with_timestamps)} - scene_{scene_id}")
        print(f"[이미지 확인] scene_{scene_id}: 원본 경로={image_path}")
        
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
            
            # 비디오 스트림 생성 (기본 스케일)
            # 이미지를 먼저 1920x1080으로 스케일
            # 이미지 형식을 명시적으로 지정하여 안정적인 입력 보장
            try:
                # 이미지 파일이 실제로 존재하고 읽을 수 있는지 확인
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")
                
                # PIL로 이미지 크기 확인
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                    print(f"[이미지 확인] scene_{scene_id}: 이미지 크기 {img_width}x{img_height}")
                
                # FFmpeg 입력: PNG 형식 명시적으로 지정
                base_stream = (
                    ffmpeg.input(image_path, format='image2', loop=1, t=duration, framerate=VIDEO_FPS)
                    .filter("scale", 1920, 1080)
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
                    ffmpeg.input(image_path, format='image2', loop=1, t=duration, framerate=VIDEO_FPS)
                    .filter("scale", 1920, 1080)
                )
            
            # 자막 적용 (ASS 형식 사용 - 배경 박스가 끊어지지 않음)
            # 자막을 먼저 적용하여 타임코드가 정확하게 유지되도록 함
            if include_subtitles:
                subtitle_kwargs = {}
                if os.path.isdir(FONTS_FOLDER):
                    subtitle_kwargs["fontsdir"] = os.path.abspath(FONTS_FOLDER)
                # BackColour와 Outline을 명시적으로 지정하여 검정색 배경 박스가 확실히 표시되도록 함
                # Outline 값이 배경 박스의 패딩 역할을 함
                subtitle_kwargs["force_style"] = f"BackColour=&HFF000000,BorderStyle=3,Outline=10"
                
                # 씬별 ASS 자막 파일이 존재하는 경우에만 자막 적용
                if os.path.exists(scene_subtitle_ass) and os.path.getsize(scene_subtitle_ass) > 0:
                    video_with_subs = base_stream.filter("subtitles", scene_subtitle_ass, **subtitle_kwargs)
                else:
                    video_with_subs = base_stream  # 자막 없이 진행
            else:
                video_with_subs = base_stream  # 자막 포함 옵션이 꺼져있으면 자막 없이 진행
            
            # 모션 효과 없이 정적 이미지 사용
            # duration을 명시적으로 제한
            video_stream = video_with_subs.filter("trim", duration=duration).filter("setpts", "PTS-STARTPTS")
            
            # 오디오 스트림 생성
            if audio_file and os.path.exists(audio_file):
                audio_stream = ffmpeg.input(audio_file)
            else:
                audio_stream = ffmpeg.input('anullsrc=channel_layout=mono:sample_rate=44100', f='lavfi', t=duration)
            
            # 개별 씬 비디오 생성 (duration 명시적으로 제한)
            scene_output = ffmpeg.output(
                video_stream,
                audio_stream,
                scene_video_file,
                vcodec="libx264",
                acodec="aac",
                pix_fmt="yuv420p",
                t=duration,  # duration 명시적으로 제한
                **{
                    "preset": "ultrafast",  # 개별 씬은 빠르게 생성
                    "crf": "23",
                    "threads": "0",
                }
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
                    
                    # 생성된 비디오의 실제 duration 확인
                    try:
                        probe = ffmpeg.probe(scene_video_file)
                        video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
                        if video_streams:
                            actual_video_duration = float(video_streams[0].get('duration', 0))
                            if abs(actual_video_duration - duration) > 0.5:  # 0.5초 이상 차이나면 경고
                                print(f"[경고] scene_{scene_id}: 비디오 duration 불일치! 예상={duration:.2f}초, 실제={actual_video_duration:.2f}초")
                                # duration이 너무 길면 재인코딩 (짧게 자르기)
                                if actual_video_duration > duration + 0.5:
                                    print(f"[수정] scene_{scene_id}: 비디오를 정확한 duration으로 재인코딩 중...")
                                    temp_file = scene_video_file + ".tmp"
                                    os.rename(scene_video_file, temp_file)
                                    input_video = ffmpeg.input(temp_file)
                                    corrected_output = ffmpeg.output(
                                        input_video['v'].filter('trim', duration=duration),
                                        input_video['a'].filter('atrim', duration=duration),
                                        scene_video_file,
                                        vcodec="libx264",
                                        acodec="aac",
                                        pix_fmt="yuv420p",
                                        t=duration,
                                        preset="ultrafast",
                                        crf="23"
                                    )
                                    ffmpeg.run(corrected_output, overwrite_output=True, quiet=True)
                                    os.remove(temp_file)
                                    print(f"[수정 완료] scene_{scene_id}: duration 수정됨")
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
                    # 이미지 형식을 명시적으로 지정
                    simple_stream = (
                        ffmpeg.input(image_path, format='image2', loop=1, t=duration, framerate=VIDEO_FPS)
                        .filter("scale", 1920, 1080)
                    )
                    if include_subtitles and os.path.exists(scene_subtitle_ass) and os.path.getsize(scene_subtitle_ass) > 0:
                        # fallback 경로에서도 subtitle_kwargs 정의
                        fallback_subtitle_kwargs = {}
                        if os.path.isdir(FONTS_FOLDER):
                            fallback_subtitle_kwargs["fontsdir"] = os.path.abspath(FONTS_FOLDER)
                        fallback_subtitle_kwargs["force_style"] = f"BackColour=&HFF000000,BorderStyle=3,Outline=10"
                        simple_with_subs = simple_stream.filter("subtitles", scene_subtitle_ass, **fallback_subtitle_kwargs)
                    else:
                        simple_with_subs = simple_stream
                    simple_output = ffmpeg.output(
                        simple_with_subs,
                        audio_stream,
                        scene_video_file,
                        vcodec="libx264",
                        acodec="aac",
                        pix_fmt="yuv420p",
                        t=duration,  # duration 명시적으로 제한
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
        
        # concat demuxer 사용 (메모리 효율적)
        concat_input = ffmpeg.input(concat_list_file, format='concat', safe=0)
        output = ffmpeg.output(
            concat_input,
            output_video_file,
            vcodec="libx264",  # 재인코딩 (안정성 확보)
            acodec="aac",  # 재인코딩 (안정성 확보)
            pix_fmt="yuv420p",
            **{
                "preset": "fast",  # 빠른 인코딩
                "crf": "23",  # 품질 설정
                "threads": "0",  # 자동 스레드
                "movflags": "+faststart",  # 웹 스트리밍 최적화
            }
        )
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
):
    assets_folder, subtitle_file, video_file = get_job_paths(job_id)
    
    # API 키 확인 및 환경 변수 fallback
    if not replicate_api_key or (isinstance(replicate_api_key, str) and not replicate_api_key.strip()):
        replicate_api_key = REPLICATE_API_TOKEN
        print(f"[API 키] Replicate API 키: 사용자 입력 없음, 환경 변수 사용")
    else:
        print(f"[API 키] Replicate API 키: 사용자 입력 사용")
    
    if not elevenlabs_api_key or (isinstance(elevenlabs_api_key, str) and not elevenlabs_api_key.strip()):
        elevenlabs_api_key = ELEVENLABS_API_KEY
        print(f"[API 키] ElevenLabs API 키: 사용자 입력 없음, 환경 변수 사용")
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

    try:
        update_job(job_id, status="running")
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
            )
            if not chunk_scene_data:
                raise RuntimeError(f"청크 {chunk_idx} 자산 생성에 실패했습니다.")
            
            # 타임스탬프 계산
            update_job(job_id, stage_progress=1, current_stage=f"타임스탬프 계산 (청크 {chunk_idx}/{total_chunks})")
            chunk_scene_data_with_timestamps, chunk_duration = calculate_timestamps(
                chunk_scene_data, progress_cb=lambda msg: progress(f"[청크 {chunk_idx}] {msg}")
            )
            
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


@app.route("/start_job", methods=["POST"])
def start_job():
    payload = request.get_json(silent=True) or {}
    job_id = (payload.get("job_id") or "").strip()
    script_text = (payload.get("script_text") or "").strip()
    char_limit_raw = str(payload.get("char_limit") or "").strip()
    voice_id = (payload.get("voice_id") or "").strip()
    mode = (payload.get("mode") or "animation").strip().lower()
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
        args=(job_id, script_text, char_limit, voice_id, prompts_override, existing_images, mode, sentences_override, include_subtitles, replicate_api_key, elevenlabs_api_key),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/generate_prompts", methods=["POST"])
def api_generate_prompts():
    data = request.get_json(silent=True) or {}
    script_text = (data.get("script_text") or "").strip()
    mode = (data.get("mode") or "animation").strip().lower()
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
                            mode=mode
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
                        original_script=script_text
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
                
                prompts = generate_visual_prompts(sentences, mode=mode, progress_cb=progress_callback, original_script=script_text)
                
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
                        with jobs_lock:
                            job_data = jobs.get(job_id)
                            if job_data is not None:
                                job_data["progress"].append(f"{idx}번 이미지 생성 완료 ({file_size} bytes)")
                    
                    abs_path = os.path.abspath(image_filename)
                    image_files[idx] = abs_path
                    rel_path = os.path.relpath(abs_path, STATIC_FOLDER)
                    # url_for 대신 직접 경로 구성 (백그라운드 스레드에서 안전)
                    image_url = "/static/" + rel_path.replace(os.sep, "/")
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
    """프롬프트 생성 단계를 건너뛰고 바로 이미지 생성"""
    data = request.get_json(silent=True) or {}
    script_text = (data.get("script_text") or "").strip()
    sentences = data.get("sentences") or []
    prompts = data.get("prompts") or []
    mode = (data.get("mode") or "animation").strip().lower()
    test_mode = data.get("test_mode", False)
    # API 키 (사용자가 입력한 경우 사용, 없으면 기본값 사용)
    replicate_api_key = data.get("replicate_api_key") or None
    elevenlabs_api_key = data.get("elevenlabs_api_key") or None
    
    if not script_text or not sentences or not prompts:
        return jsonify({"error": "대본, 문장, 프롬프트가 모두 필요합니다."}), 400
    
    if len(sentences) != len(prompts):
        return jsonify({"error": "문장과 프롬프트의 개수가 일치하지 않습니다."}), 400
    
    job_id = create_job_record()
    
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
        try:
            total = len(sentences)
            image_results = []
            image_files = {}
            
            print(f"[이미지 생성 시작] 총 {total}개 이미지 생성 예정 (테스트 모드: {test_mode})")
            
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
                    
                    # 테스트 모드일 때 photo 폴더의 이미지 사용
                    if test_mode:
                        photo_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo")
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
                        success = generate_image(prompt, image_filename, mode=mode, replicate_api_key=replicate_api_key)
                        
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
                            with jobs_lock:
                                job_data = jobs.get(job_id)
                                if job_data is not None:
                                    job_data["progress"].append(f"{idx}번 이미지 생성 완료 ({file_size} bytes)")
                    
                    abs_path = os.path.abspath(image_filename)
                    image_files[idx] = abs_path
                    rel_path = os.path.relpath(abs_path, STATIC_FOLDER)
                    # url_for 대신 직접 경로 구성 (백그라운드 스레드에서 안전)
                    image_url = "/static/" + rel_path.replace(os.sep, "/")
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
            video_url = f"/static/{video_filename}"
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
    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        print(f"[다운로드] 비디오 파일 다운로드: {filename} (경로: {file_path})")
        try:
            # send_file을 사용하여 명시적으로 다운로드 설정
            response = send_file(
                file_path,
                mimetype='video/mp4',
                as_attachment=True,
                download_name=os.path.basename(filename)
            )
            # Content-Disposition 헤더 명시적으로 설정
            response.headers['Content-Disposition'] = f'attachment; filename="{os.path.basename(filename)}"'
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
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
    
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
        return send_file(
            subtitle_file,
            mimetype='text/plain; charset=utf-8',
            as_attachment=True,
            download_name=srt_filename
        )
    
    except Exception as e:
        print(f"[다운로드 오류] SRT 파일 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"자막 다운로드 실패: {str(e)}"}), 500


if __name__ == "__main__":
    import webbrowser
    from threading import Timer
    
    # PyInstaller로 빌드된 경우 감지
    is_bundled = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
    
    # 포트 찾기 (5001부터 시도, 사용 중이면 다음 포트 사용)
    port = 5001
    import socket
    for p in range(5001, 5010):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        result = sock.connect_ex(('127.0.0.1', p))
        sock.close()
        if result != 0:  # 포트가 비어있음 (연결 실패 = 사용 가능)
            port = p
            break
    # 포트를 찾지 못한 경우 기본값 사용
    if port == 5001:
        # 5001이 사용 중인지 다시 확인
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        if sock.connect_ex(('127.0.0.1', 5001)) == 0:
            # 5001이 사용 중이면 5002 사용
            port = 5002
        sock.close()
    
    def open_browser():
        """브라우저를 자동으로 엽니다."""
        url = f"http://127.0.0.1:{port}"
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"브라우저 열기 실패: {e}")
            print(f"수동으로 다음 주소를 열어주세요: {url}")
    
    # 1초 후 브라우저 열기
    Timer(1.0, open_browser).start()
    
    print(f"\n{'='*60}")
    print(f"유튜브 영상 생성기 서버 시작")
    print(f"주소: http://127.0.0.1:{port}")
    print(f"{'='*60}\n")
    
    # 디버그 모드는 개발 환경에서만
    debug_mode = not is_bundled
    
    try:
        app.run(host='127.0.0.1', port=port, debug=debug_mode, use_reloader=False)
    except KeyboardInterrupt:
        print("\n서버를 종료합니다...")
    except Exception as e:
        print(f"\n서버 시작 오류: {e}")
        if is_bundled:
            input("\n아무 키나 눌러 종료하세요...")
