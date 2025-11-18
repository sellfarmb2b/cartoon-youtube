"""
1단계, 2단계, 3단계, 4단계를 통합한 유튜브 애니메이션 자동화 도구
- 1단계: 대본을 문단으로 분할
- 2단계: 각 문단별로 TTS 오디오와 이미지 생성
- 3단계: 오디오 파일의 재생 시간을 계산하여 타임스탬프 추가
- 4단계: 이미지 + 오디오 + 자막을 합쳐 최종 영상 생성
"""

import os
import json
import math
import requests
import base64
import textwrap
import time
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from mutagen.mp3 import MP3
from mutagen.mp3 import HeaderNotFoundError  # MP3 파일 오류 감지
import ffmpeg  # FFmpeg Python 라이브러리
from dotenv import load_dotenv

# --- 1단계: 대본 분할 함수 ---

def split_script_into_paragraphs(full_script_text):
    """
    긴 대본 텍스트를 '빈 줄'(\n\n)을 기준으로 나누어
    문단들의 리스트(list)로 반환합니다.
    """
    
    # 1. 앞뒤 공백을 먼저 제거합니다.
    clean_text = full_script_text.strip()
    
    # 2. '빈 줄'(\n\n)을 기준으로 텍스트를 나눕니다.
    #    split()은 결과를 리스트(list)로 만들어줍니다.
    paragraphs = clean_text.split('\n\n')
    
    # 3. (선택 사항) 혹시나 사용자가 실수로 빈 줄을 여러 개 넣었을 경우,
    #    비어있는 문단('')을 제거합니다.
    final_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return final_paragraphs

# --- 2단계: 환경 변수 로드 및 API 키 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_candidates = [
    os.path.join(BASE_DIR, ".env.local"),
    os.path.join(BASE_DIR, ".env"),
]
for dotenv_path in dotenv_candidates:
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=False)
load_dotenv(override=False)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# --- 3. API 클라이언트 초기화 (한 번만) ---
print("API 클라이언트 초기화 중...")
eleven_client = None
stability_api_available = False
replicate_api_available = False

try:
    if ELEVENLABS_API_KEY:
        eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        print("✓ ElevenLabs 클라이언트 초기화 완료.")
    else:
        print("⚠ ElevenLabs API 키가 설정되지 않았습니다.")
except Exception as e:
    print(f"⚠ ElevenLabs 클라이언트 초기화 오류: {e}")

# Stability AI API 키 확인 (HTTP 요청으로 직접 사용)
if STABILITY_API_KEY:
    stability_api_available = True
    print("✓ Stability AI API 키가 설정되었습니다. (HTTP 요청 사용)")
else:
    print("⚠ Stability AI API 키가 설정되지 않았습니다.")
    stability_api_available = False

# Replicate API 키 확인
if REPLICATE_API_TOKEN:
    replicate_api_available = True
    print("✓ Replicate API 키가 설정되었습니다.")
else:
    print("⚠ Replicate API 키가 설정되지 않았습니다.")

# --- 4. 헬퍼(Helper) 함수 정의 ---

def generate_tts(text, filename):
    """ElevenLabs API로 TTS를 생성하고 파일로 저장합니다."""
    if eleven_client is None:
        print(f"  [TTS] 건너뜀: API 키가 설정되지 않았습니다.")
        return False
    
    try:
        print(f"  [TTS] 생성 중: {filename}")
        
        # 사용 가능한 voice 목록 가져오기
        voices = eleven_client.voices.get_all()
        if not voices.voices:
            print("  [TTS] 사용 가능한 voice가 없습니다.")
            return False
        
        # 첫 번째 사용 가능한 voice 사용
        voice_id = voices.voices[0].voice_id
        print(f"  [TTS] Voice ID 사용: {voice_id}")
        
        # 최신 ElevenLabs API는 generator를 반환하므로 bytes로 변환
        audio_generator = eleven_client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2", # 다국어 모델
            voice_settings=VoiceSettings(stability=0.7, similarity_boost=0.7)
        )
        # generator를 bytes로 변환
        audio_data = b"".join(audio_generator)
        with open(filename, "wb") as f:
            f.write(audio_data)
        print(f"  [TTS] 저장 완료: {filename}")
        return True
    except Exception as e:
        print(f"  [TTS] 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_image(prompt_text, filename):
    """이미지를 생성하고 파일로 저장합니다. (Replicate 우선, 실패 시 Stability HTTP)"""

    # (전문가 조언) 실제 문단은 프롬프트로 너무 깁니다.
    # Stability/Replicate 모두 영어가 유리합니다. 한국어 포함 시 기본 영어 프롬프트 사용
    if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in prompt_text):
        image_prompt = "digital anime art style, colorful illustration, cute character, friendly atmosphere, bright colors"
    else:
        image_prompt = f"digital anime art style, {prompt_text[:100]}"

    # 1) Replicate 경로 (bytedance/sdxl-lightning-4step)
    if replicate_api_available:
        try:
            print(f"  [IMG] (Replicate) 생성 중: {filename} (프롬프트: {image_prompt}...)")
            headers = {
                "Authorization": f"Token {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json"
            }
            # 16:9에 가까운 허용 해상도(1344x768)를 생성하고, 이후 1920x1080으로 스케일
            body = {
                "version": "6f7a773af6fc3e8de9d5a3c00be77c17308914bf67772726aff83496ba1e3bbe",
                "input": {
                    "prompt": image_prompt,
                    "width": 1344,
                    "height": 768,
                    "num_inference_steps": 4,
                    "guidance_scale": 1.0
                }
            }
            create_res = requests.post("https://api.replicate.com/v1/predictions", headers=headers, json=body)
            if create_res.status_code != 201:
                print(f"  [IMG] (Replicate) 생성 실패: {create_res.status_code} {create_res.text}")
            else:
                pred = create_res.json()
                pred_id = pred.get("id")
                status = pred.get("status")
                get_url = f"https://api.replicate.com/v1/predictions/{pred_id}"
                # 폴링
                for _ in range(60):
                    res = requests.get(get_url, headers=headers)
                    if res.status_code != 200:
                        print(f"  [IMG] (Replicate) 상태 조회 실패: {res.status_code} {res.text}")
                        break
                    data = res.json()
                    status = data.get("status")
                    if status in ("succeeded", "failed", "canceled"):
                        if status == "succeeded":
                            outputs = data.get("output") or []
                            if outputs:
                                # 첫 번째 URL 다운로드
                                img_url = outputs[0]
                                img_bytes = requests.get(img_url).content
                                with open(filename, "wb") as f:
                                    f.write(img_bytes)
                                print(f"  [IMG] (Replicate) 저장 완료: {filename}")
                                return True
                            else:
                                print("  [IMG] (Replicate) 출력이 비어 있습니다.")
                        else:
                            print(f"  [IMG] (Replicate) 상태: {status}")
                        break
                    time.sleep(1)
        except Exception as e:
            print(f"  [IMG] (Replicate) 생성 실패: {e}")

    # 2) Stability HTTP Fallback
    if stability_api_available:
        try:
            print(f"  [IMG] (Stability) 생성 중: {filename} (프롬프트: {image_prompt}...)")
            api_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            }
            payload = {
                "text_prompts": [{"text": image_prompt, "weight": 1.0}],
                "cfg_scale": 7,
                "height": 768,
                "width": 1344,
                "samples": 1,
                "steps": 30,
                "sampler": "K_DPMPP_2M"
            }
            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                if "artifacts" in data and len(data["artifacts"]) > 0:
                    image_data = base64.b64decode(data["artifacts"][0]["base64"])
                    with open(filename, "wb") as f:
                        f.write(image_data)
                    print(f"  [IMG] (Stability) 저장 완료: {filename}")
                    return True
                else:
                    print("  [IMG] (Stability) 생성 실패: 응답에 이미지가 없습니다.")
            else:
                print(f"  [IMG] (Stability) 생성 실패: API 오류 (상태 코드: {response.status_code})")
                print(f"  [IMG] 응답: {response.text}")
        except Exception as e:
            print(f"  [IMG] (Stability) 생성 실패: {e}")

    return False

# --- 3단계: 타임스탬프 계산 함수 ---

def calculate_timestamps(scene_data):
    """
    2단계에서 만든 scene_data 리스트를 받아서,
    각 오디오 파일의 재생 시간을 계산하고 타임스탬프를 추가합니다.
    
    Args:
        scene_data: 2단계에서 생성된 씬 데이터 리스트
        
    Returns:
        scene_data_with_timestamps: 타임스탬프가 추가된 씬 데이터 리스트
        total_duration: 전체 영상의 총 길이 (초)
    """
    print("\n=== 3단계: 타임스탬프 계산 시작 ===")
    
    # 타임스탬프 정보가 추가될 새로운 리스트
    scene_data_with_timestamps = []
    
    # 전체 영상의 총 길이를 추적하는 변수 (단위: 초)
    total_duration_so_far = 0.0
    
    for scene in scene_data:
        audio_path = scene["audio_file"]
        
        try:
            # 1. 오디오 파일 로드
            audio_info = MP3(audio_path)
            
            # 2. 재생 시간(초) 가져오기 (핵심!)
            duration_sec = audio_info.info.length
            
            # 3. 타임스탬프 계산
            start_time = total_duration_so_far
            end_time = total_duration_so_far + duration_sec
            
            # 4. 씬 정보(딕셔너리)에 타임스탬프 추가
            #    (round()는 소수점 2자리까지만 정리)
            scene["duration"] = round(duration_sec, 2)
            scene["start_time"] = round(start_time, 2)
            scene["end_time"] = round(end_time, 2)
            
            # 5. 다음 씬을 위해 총 시간 업데이트
            total_duration_so_far = end_time
            
            # 6. 완성된 씬 정보를 새 리스트에 추가
            #    (딕셔너리를 복사해서 원본을 보존)
            scene_copy = scene.copy()
            scene_data_with_timestamps.append(scene_copy)
            
            print(f"[성공] {audio_path} (길이: {duration_sec:.2f}초, 시작: {start_time:.2f}초, 종료: {end_time:.2f}초)")
            
        except FileNotFoundError:
            print(f"[!!! 오류] 파일을 찾을 수 없습니다: {audio_path}")
            print("    -> 2단계 코드를 먼저 실행해서 오디오 파일을 생성했는지 확인하세요.")
        except HeaderNotFoundError:
            print(f"[!!! 오류] {audio_path} 파일이 손상되었거나 MP3 파일이 아닙니다.")
        except Exception as e:
            print(f"[!!! 오류] {audio_path} 처리 중 알 수 없는 오류: {e}")
    
    print("\n=== 3단계 완료: 모든 타임스탬프 계산됨 ===")
    print("최종 생성된 '씬 데이터' (타임스탬프 포함):")
    print(json.dumps(scene_data_with_timestamps, indent=2, ensure_ascii=False))
    
    print(f"\n======================================")
    print(f"  계산된 총 영상 길이: {total_duration_so_far:.2f} 초")
    print(f"======================================")
    
    return scene_data_with_timestamps, total_duration_so_far

# --- 4단계: 영상 합성 함수 ---

def seconds_to_srt_time(seconds):
    """
    12.345 같은 초 단위를 00:00:12,345 같은 SRT 시간 형식으로 변환합니다.
    """
    # math.modf는 (소수점 부분, 정수 부분)을 반환합니다.
    frac, whole = math.modf(seconds)
    ms = int(frac * 1000)
    
    # 초를 시, 분, 초로 변환
    m, s = divmod(int(whole), 60)
    h, m = divmod(m, 60)
    
    # 00:00:12,345 형식으로 포맷팅
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def wrap_text_by_chars(text, limit=15):
    """텍스트를 글자 수 기준으로 줄바꿈합니다."""
    wrapper = textwrap.TextWrapper(
        width=limit,
        break_long_words=True,
        replace_whitespace=False,
        break_on_hyphens=False
    )
    return wrapper.fill(text)


def create_video(scene_data_with_timestamps, output_folder="generated_assets"):
    """
    3단계에서 만든 scene_data_with_timestamps를 사용하여
    이미지 + 오디오 + 자막을 합쳐 최종 영상 파일을 생성합니다.
    
    Args:
        scene_data_with_timestamps: 타임스탬프가 포함된 씬 데이터 리스트
        output_folder: 출력 폴더 경로
        
    Returns:
        output_video_file: 생성된 영상 파일 경로
    """
    print("\n=== 4단계: 영상 합성 시작 ===")
    
    output_video_file = os.path.join(output_folder, "final_video.mp4")
    subtitle_file = os.path.join(output_folder, "subtitles.srt")

    # 사용자 설정: 자막 줄바꿈 글자 수
    user_defined_char_limit = 20

    # --- 4단계-A: 자막(.srt) 파일 생성 ---
    print(f"--- 4단계-A: '{subtitle_file}' 자막 파일 생성 중 ---")
    print(f"    (적용된 줄바꿈 기준: {user_defined_char_limit}자)")
    try:
        with open(subtitle_file, "w", encoding="utf-8") as f:
            for scene in scene_data_with_timestamps:
                f.write(f"{scene['scene_id']}\n")

                start_srt = seconds_to_srt_time(scene['start_time'])
                end_srt = seconds_to_srt_time(scene['end_time'])

                f.write(f"{start_srt} --> {end_srt}\n")
                cleaned_text = " ".join(scene['text'].split())
                wrapped_text = wrap_text_by_chars(cleaned_text, user_defined_char_limit)
                f.write(f"{wrapped_text}\n\n")
        print(f"'{subtitle_file}' 생성 완료.")
    except Exception as e:
        print(f"자막 파일 생성 실패: {e}")
        return None
    
    # --- 4단계-B: FFmpeg 스트림 정의 (조립법) ---
    print("\n--- 4단계-B: FFmpeg 스트림 정의 중 ---")
    
    # (1) 비디오 스트림 정의 (이미지들을 시간만큼 반복해서 이어붙임)
    video_streams = []
    for scene in scene_data_with_timestamps:
        image_path = scene.get('image_file')
        
        # 이미지가 없는 경우 기본 검은색 이미지 생성
        if image_path is None:
            from PIL import Image
            black_image_path = os.path.join(output_folder, f"black_{scene['scene_id']}.png")
            img = Image.new('RGB', (1920, 1080), color='black')
            img.save(black_image_path)
            image_path = black_image_path
            print(f"  [경고] Scene #{scene['scene_id']}: 이미지가 없어 검은색 이미지를 사용합니다.")
        
        # 상대 경로를 절대 경로로 변환 (필요한 경우)
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.getcwd(), image_path)
        
        # 파일이 존재하는지 확인
        if not os.path.exists(image_path):
            print(f"  [경고] Scene #{scene['scene_id']}: 이미지 파일을 찾을 수 없습니다: {image_path}")
            from PIL import Image
            black_image_path = os.path.join(output_folder, f"black_{scene['scene_id']}.png")
            img = Image.new('RGB', (1920, 1080), color='black')
            img.save(black_image_path)
            image_path = black_image_path
        
        stream = ffmpeg.input(
            image_path,
            loop=1,
            t=scene['duration'],
            framerate=25
        ).filter('scale', 1920, 1080)
        video_streams.append(stream)
    
    # (2) 오디오 스트림 정의
    audio_streams = []
    for scene in scene_data_with_timestamps:
        audio_path = scene['audio_file']
        # 상대 경로를 절대 경로로 변환 (필요한 경우)
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(os.getcwd(), audio_path)
        audio_streams.append(ffmpeg.input(audio_path))
    
    # (3) 비디오와 오디오를 각각 하나로 합침
    concatenated_video = ffmpeg.concat(*video_streams, v=1, a=0)
    concatenated_audio = ffmpeg.concat(*audio_streams, v=0, a=1)
    
    print("비디오/오디오 스트림 정의 완료.")
    
    # --- 4단계-C: 최종 영상 합성 및 자막 입히기 ---
    print("\n--- 4단계-C: 최종 영상 합성 시작 ---")
    print(f"'{output_video_file}' 파일이 생성됩니다. 몇 분 정도 걸릴 수 있습니다...")
    
    try:
        # 자막 파일 경로를 절대 경로로 변환
        subtitle_path = subtitle_file
        if not os.path.isabs(subtitle_path):
            subtitle_path = os.path.join(os.getcwd(), subtitle_path)
        
        # (1) 합쳐진 비디오에 자막 필터(filter) 적용
        # macOS에서는 'AppleGothic' 또는 'Helvetica' 사용
        # Windows에서는 'Malgun Gothic' 사용
        # Linux에서는 'DejaVu Sans' 사용
        import platform
        if platform.system() == 'Darwin':  # macOS
            font_name = "AppleGothic"
        elif platform.system() == 'Windows':
            font_name = "Malgun Gothic"
        else:  # Linux
            font_name = "DejaVu Sans"
        
        video_with_subs = concatenated_video.filter(
            'subtitles',
            subtitle_path,
            force_style=f"FontName={font_name},FontSize=24,PrimaryColour=&H00FFFFFF,BorderStyle=3,OutlineColour=&H00000000,Shadow=0"
        )
        
        # (2) 최종본: 자막 입힌 비디오 + 합쳐진 오디오
        output = ffmpeg.output(
            video_with_subs,         # 비디오 입력
            concatenated_audio,      # 오디오 입력
            output_video_file,       # 출력 파일 이름
            vcodec='libx264',        # 비디오 코덱 (가장 표준)
            acodec='aac',            # 오디오 코덱 (가장 표준)
            pix_fmt='yuv420p',       # 픽셀 포맷 (호환성)
            shortest=None            # 오디오/비디오 중 짧은 쪽 기준이 아님
        )
        
        # (3) 실행! (이때 FFmpeg 프로그램이 실제로 작동함)
        # overwrite_output=True: 이미 파일이 있으면 덮어쓰기
        ffmpeg.run(output, overwrite_output=True, quiet=False)
        
        print("\n--- 4단계 완료! ---")
        print(f"'{output_video_file}' 파일이 성공적으로 생성되었습니다.")
        return output_video_file
        
    except ffmpeg.Error as e:
        print("\n--- [!!! FFmpeg 오류] ---")
        print("FFmpeg 오류가 발생했습니다:")
        if hasattr(e, 'stderr') and e.stderr:
            print(e.stderr.decode('utf-8'))
        else:
            print(str(e))
        print("\n[전문가 조언] 'No such file or directory' 또는 'Error 2' 오류는")
        print("FFmpeg 프로그램이 제대로 설치되지 않았거나 '환경 변수(PATH)'에 등록되지 않은 것입니다.")
        return None
    except FileNotFoundError as e:
        print("\n--- [!!! 파일 없음 오류] ---")
        print(f"파일을 찾을 수 없습니다: {e}")
        print("2단계에서 생성한 이미지/오디오 파일('generated_assets' 폴더)을 찾을 수 없습니다.")
        print("2단계, 3단계 코드가 이 파일과 같은 위치에서 먼저 실행되었는지 확인하세요.")
        return None
    except Exception as e:
        print(f"\n--- [!!! 알 수 없는 오류] ---")
        print(f"오류 발생: {e}")
        return None

# --- 5. 메인 실행 루프 (핵심!) ---

def main():
    """메인 실행 함수"""
    
    # --- 1단계: 대본 입력 및 분할 ---
    print("\n=== 1단계: 대본 분할 ===")
    
    # 사용자가 입력한 긴 대본이라고 가정합니다.
    example_script = """
안녕하세요! 유튜브 애니메이션 자동화 도구입니다.
이것은 첫 번째 문단입니다. 여기서 첫 번째 장면이 시작됩니다.

두 번째 문단입니다.
장면이 바뀌고, 새로운 이미지가 나와야 합니다.

이것은 세 번째 문단입니다.
여기서는 중요한 내용이 강조되어야 합니다.
"""
    
    # 대본을 문단으로 분할
    parsed_paragraphs = split_script_into_paragraphs(example_script)
    
    print(f"총 {len(parsed_paragraphs)}개의 문단으로 분할되었습니다.")
    for i, para in enumerate(parsed_paragraphs):
        print(f"\n[문단 {i+1}]")
        print(para[:50] + "..." if len(para) > 50 else para)
    
    # --- 2단계: 문단별 자산 생성 ---
    print("\n=== 2단계: 문단별 자산 생성 시작 ===")
    
    # 생성된 파일과 정보를 저장할 리스트
    # 이것이 우리 프로젝트의 '데이터베이스'가 됩니다.
    scene_data = []
    
    # 생성된 파일을 저장할 폴더 생성
    output_folder = "generated_assets"
    os.makedirs(output_folder, exist_ok=True)
    
    # 1단계에서 만든 문단 리스트를 하나씩 꺼내서 처리
    for i, paragraph in enumerate(parsed_paragraphs):
        scene_number = i + 1
        print(f"\n--- [Scene #{scene_number}] 처리 중 ---")
        
        # 1. 파일 이름 정의
        audio_filename = os.path.join(output_folder, f"scene_{scene_number}_audio.mp3")
        image_filename = os.path.join(output_folder, f"scene_{scene_number}_image.png")
        
        # 2. TTS 생성
        tts_success = generate_tts(paragraph, audio_filename)
        
        # 3. 이미지 생성
        img_success = generate_image(paragraph, image_filename)
        
        # 4. (매우 중요) 생성된 정보 저장
        # TTS가 성공하면 이미지가 없어도 진행 (이미지는 나중에 추가 가능)
        if tts_success:
            scene_info = {
                "scene_id": scene_number,
                "text": paragraph,
                "audio_file": audio_filename,
                "image_file": image_filename if img_success else None
            }
            scene_data.append(scene_info)
            if not img_success:
                print(f"[경고] Scene #{scene_number}: 이미지가 생성되지 않았지만 오디오는 생성되었습니다.")
        else:
            print(f"[오류] Scene #{scene_number} 생성 중 문제가 발생하여 건너뜁니다.")
    
    # --- 2단계 결과 출력 ---
    print("\n=== 2단계 완료: 모든 자산 생성됨 ===")
    print("생성된 씬(Scene) 데이터:")
    print(json.dumps(scene_data, indent=2, ensure_ascii=False))
    
    # JSON 파일로도 저장 (타임스탬프 없이)
    output_json = os.path.join(output_folder, "scene_data.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(scene_data, f, indent=2, ensure_ascii=False)
    print(f"\n씬 데이터가 {output_json}에 저장되었습니다.")
    
    # --- 3단계: 타임스탬프 계산 ---
    # 오디오 파일이 실제로 생성된 경우에만 타임스탬프 계산
    scene_data_with_timestamps = None
    if scene_data:
        scene_data_with_timestamps, total_duration = calculate_timestamps(scene_data)
        
        # 타임스탬프가 포함된 최종 데이터를 JSON 파일로 저장
        final_output_json = os.path.join(output_folder, "scene_data_with_timestamps.json")
        with open(final_output_json, "w", encoding="utf-8") as f:
            json.dump(scene_data_with_timestamps, f, indent=2, ensure_ascii=False)
        print(f"\n타임스탬프가 포함된 최종 씬 데이터가 {final_output_json}에 저장되었습니다.")
    else:
        print("\n⚠ 오디오 파일이 생성되지 않아 타임스탬프 계산을 건너뜁니다.")
    
    # --- 4단계: 영상 합성 ---
    # 타임스탬프가 계산된 경우에만 영상 합성
    if scene_data_with_timestamps:
        video_file = create_video(scene_data_with_timestamps, output_folder)
        if video_file:
            print(f"\n✅ 최종 영상이 생성되었습니다: {video_file}")
        else:
            print("\n⚠ 영상 생성에 실패했습니다.")
    else:
        print("\n⚠ 타임스탬프 데이터가 없어 영상 합성을 건너뜁니다.")

if __name__ == "__main__":
    main()

