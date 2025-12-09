#!/usr/bin/env python3
"""
create_video 함수 테스트 스크립트
더미 데이터를 사용하여 영상 합성 기능을 테스트합니다.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# src 디렉토리를 경로에 추가
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# 필요한 모듈만 import (전체 app.py를 로드하지 않음)
try:
    from mutagen.mp3 import MP3
except ImportError:
    print("[경고] mutagen이 설치되지 않았습니다. pip install mutagen")
    MP3 = None

try:
    from PIL import Image
except ImportError:
    print("[경고] Pillow가 설치되지 않았습니다. pip install pillow")
    Image = None

try:
    import ffmpeg
except ImportError:
    print("[경고] ffmpeg-python이 설치되지 않았습니다. pip install ffmpeg-python")
    ffmpeg = None

# create_video 함수만 import
# 전체 app.py를 로드하면 많은 의존성이 필요하므로, 
# 필요한 함수들을 직접 복사하거나 간단한 버전을 사용
def import_create_video():
    """create_video 함수를 안전하게 import"""
    try:
        # app.py의 전역 변수들을 설정하지 않고 함수만 가져오기
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", os.path.join(src_path, "app.py"))
        app_module = importlib.util.module_from_spec(spec)
        
        # 필요한 전역 변수만 설정
        app_module.VIDEO_FPS = 25
        app_module.FONTS_FOLDER = os.path.join(src_path, "static", "fonts")
        os.makedirs(app_module.FONTS_FOLDER, exist_ok=True)
        
        # 모듈 실행 (일부 전역 변수는 None으로 설정하여 에러 방지)
        import sys as sys_module
        original_modules = sys_module.modules.copy()
        try:
            spec.loader.exec_module(app_module)
        except Exception as e:
            print(f"[경고] app.py 로드 중 일부 오류 발생 (무시): {e}")
        
        return app_module.create_video
    except Exception as e:
        print(f"[오류] create_video 함수를 import할 수 없습니다: {e}")
        return None

def create_dummy_image(width=1920, height=1080, color=(100, 150, 200), output_path=None):
    """더미 이미지 생성"""
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.png')
    
    img = Image.new('RGB', (width, height), color=color)
    img.save(output_path)
    print(f"[테스트] 더미 이미지 생성: {output_path}")
    return output_path

def create_dummy_audio(duration=3.0, output_path=None):
    """더미 오디오 생성 (무음)"""
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.mp3')
    
    # FFmpeg로 무음 오디오 생성
    (
        ffmpeg
        .input('anullsrc=channel_layout=stereo:sample_rate=44100', f='lavfi', t=duration)
        .output(output_path, acodec='libmp3lame', ac=2, ar=44100)
        .run(overwrite_output=True, quiet=True)
    )
    print(f"[테스트] 더미 오디오 생성: {output_path} (길이: {duration:.2f}초)")
    return output_path

def create_dummy_video(duration=2.0, output_path=None):
    """더미 비디오 생성 (색상 비디오)"""
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.mp4')
    
    # FFmpeg로 색상 비디오 생성
    (
        ffmpeg
        .input('color=c=blue:s=1920x1080:d=' + str(duration), f='lavfi')
        .output(
            output_path,
            vcodec='libx264',
            pix_fmt='yuv420p',
            r=25,
            t=duration
        )
        .run(overwrite_output=True, quiet=True)
    )
    print(f"[테스트] 더미 비디오 생성: {output_path} (길이: {duration:.2f}초)")
    return output_path

def test_create_video():
    """create_video 함수 테스트"""
    print("=" * 80)
    print("create_video 함수 테스트 시작")
    print("=" * 80)
    
    # 필수 모듈 확인
    if MP3 is None or Image is None or ffmpeg is None:
        print("\n[오류] 필수 패키지가 설치되지 않았습니다.")
        print("다음 명령어로 설치해주세요:")
        print("  pip install mutagen pillow ffmpeg-python")
        return False
    
    # create_video 함수 import
    print("\n[초기화] create_video 함수 로드 중...")
    create_video = import_create_video()
    if create_video is None:
        print("[오류] create_video 함수를 로드할 수 없습니다.")
        return False
    
    # 임시 디렉토리 생성
    test_dir = tempfile.mkdtemp(prefix='test_video_')
    assets_folder = os.path.join(test_dir, 'assets')
    os.makedirs(assets_folder, exist_ok=True)
    
    print(f"\n[테스트] 작업 디렉토리: {test_dir}")
    
    try:
        # 더미 데이터 생성
        print("\n[1단계] 더미 데이터 생성 중...")
        
        # 씬 1: 이미지 + 오디오 (3초)
        scene1_image = create_dummy_image(
            color=(255, 100, 100),
            output_path=os.path.join(assets_folder, 'scene_1.png')
        )
        scene1_audio = create_dummy_audio(
            duration=3.0,
            output_path=os.path.join(assets_folder, 'scene_1.mp3')
        )
        
        # 씬 2: 비디오 + 오디오 (비디오가 짧음, 2초 비디오 + 4초 오디오)
        scene2_video = create_dummy_video(
            duration=2.0,
            output_path=os.path.join(assets_folder, 'scene_2.mp4')
        )
        scene2_audio = create_dummy_audio(
            duration=4.0,
            output_path=os.path.join(assets_folder, 'scene_2.mp3')
        )
        
        # 씬 3: 이미지 + 오디오 (5초)
        scene3_image = create_dummy_image(
            color=(100, 255, 100),
            output_path=os.path.join(assets_folder, 'scene_3.png')
        )
        scene3_audio = create_dummy_audio(
            duration=5.0,
            output_path=os.path.join(assets_folder, 'scene_3.mp3')
        )
        
        # 씬 4: 비디오 + 오디오 (비디오가 김, 6초 비디오 + 3초 오디오)
        scene4_video = create_dummy_video(
            duration=6.0,
            output_path=os.path.join(assets_folder, 'scene_4.mp4')
        )
        scene4_audio = create_dummy_audio(
            duration=3.0,
            output_path=os.path.join(assets_folder, 'scene_4.mp3')
        )
        
        # scene_data_with_timestamps 생성
        scene_data = [
            {
                'scene_id': 1,
                'text': '첫 번째 씬입니다. 이미지와 오디오가 3초입니다.',
                'image_file': scene1_image,
                'video_file': None,
                'audio_file': scene1_audio,
                'duration': 3.0,
                'start_time': 0.0,
                'end_time': 3.0,
                'alignment': None,
                'semantic_segments': []
            },
            {
                'scene_id': 2,
                'text': '두 번째 씬입니다. 비디오는 2초지만 오디오는 4초입니다. 비디오가 반복되어야 합니다.',
                'image_file': None,
                'video_file': scene2_video,
                'audio_file': scene2_audio,
                'duration': 4.0,
                'start_time': 3.0,
                'end_time': 7.0,
                'alignment': None,
                'semantic_segments': []
            },
            {
                'scene_id': 3,
                'text': '세 번째 씬입니다. 이미지와 오디오가 5초입니다.',
                'image_file': scene3_image,
                'video_file': None,
                'audio_file': scene3_audio,
                'duration': 5.0,
                'start_time': 7.0,
                'end_time': 12.0,
                'alignment': None,
                'semantic_segments': []
            },
            {
                'scene_id': 4,
                'text': '네 번째 씬입니다. 비디오는 6초지만 오디오는 3초입니다. 비디오가 잘려야 합니다.',
                'image_file': None,
                'video_file': scene4_video,
                'audio_file': scene4_audio,
                'duration': 3.0,
                'start_time': 12.0,
                'end_time': 15.0,
                'alignment': None,
                'semantic_segments': []
            },
        ]
        
        print(f"\n[테스트] 총 {len(scene_data)}개 씬 생성 완료")
        print(f"  - 씬 1: 이미지 (3초)")
        print(f"  - 씬 2: 비디오 2초 + 오디오 4초 (비디오 반복 테스트)")
        print(f"  - 씬 3: 이미지 (5초)")
        print(f"  - 씬 4: 비디오 6초 + 오디오 3초 (비디오 자르기 테스트)")
        
        # 출력 파일 경로
        subtitle_file = os.path.join(test_dir, 'subtitles.srt')
        output_video_file = os.path.join(test_dir, 'test_output.mp4')
        
        print(f"\n[2단계] create_video 함수 실행 중...")
        print(f"  - 자막 파일: {subtitle_file}")
        print(f"  - 출력 비디오: {output_video_file}")
        
        # 진행 콜백 함수
        def progress_callback(message):
            print(f"  [진행] {message}")
        
        # create_video 함수 호출
        result = create_video(
            scene_data_with_timestamps=scene_data,
            char_limit=50,
            subtitle_file=subtitle_file,
            output_video_file=output_video_file,
            assets_folder=assets_folder,
            progress_cb=progress_callback,
            include_subtitles=True
        )
        
        if result:
            print(f"\n[성공] 영상 생성 완료!")
            print(f"  - 출력 파일: {output_video_file}")
            
            # 생성된 파일 확인
            if os.path.exists(output_video_file):
                file_size = os.path.getsize(output_video_file)
                print(f"  - 파일 크기: {file_size / (1024 * 1024):.2f} MB")
                
                # 비디오 정보 확인
                try:
                    probe = ffmpeg.probe(output_video_file)
                    video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
                    audio_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
                    
                    if video_streams:
                        video_duration = float(video_streams[0].get('duration', 0))
                        video_fps = eval(video_streams[0].get('r_frame_rate', '25/1'))
                        print(f"  - 비디오 길이: {video_duration:.2f}초")
                        print(f"  - 프레임레이트: {video_fps:.2f} fps")
                        print(f"  - 해상도: {video_streams[0].get('width')}x{video_streams[0].get('height')}")
                    
                    if audio_streams:
                        audio_duration = float(audio_streams[0].get('duration', 0))
                        print(f"  - 오디오 길이: {audio_duration:.2f}초")
                        print(f"  - 샘플레이트: {audio_streams[0].get('sample_rate')} Hz")
                        print(f"  - 채널: {audio_streams[0].get('channels')}")
                    
                    # 예상 길이와 비교
                    expected_duration = sum(scene['duration'] for scene in scene_data)
                    actual_duration = video_duration if video_streams else 0
                    diff = abs(actual_duration - expected_duration)
                    
                    print(f"\n[검증]")
                    print(f"  - 예상 길이: {expected_duration:.2f}초")
                    print(f"  - 실제 길이: {actual_duration:.2f}초")
                    print(f"  - 차이: {diff:.2f}초")
                    
                    if diff < 0.5:
                        print(f"  ✅ 길이 일치 (오차 0.5초 이내)")
                    else:
                        print(f"  ⚠️  길이 불일치 (오차 0.5초 초과)")
                    
                except Exception as e:
                    print(f"  [경고] 비디오 정보 확인 실패: {e}")
                
                print(f"\n[테스트 완료] 출력 파일을 확인하세요: {output_video_file}")
                print(f"  (테스트 디렉토리는 자동으로 삭제되지 않습니다: {test_dir})")
            else:
                print(f"\n[오류] 출력 파일이 생성되지 않았습니다.")
                return False
        else:
            print(f"\n[실패] 영상 생성 실패")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n[오류] 테스트 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 테스트 디렉토리는 수동으로 확인할 수 있도록 삭제하지 않음
        # 자동 삭제를 원하면 아래 주석 해제
        # shutil.rmtree(test_dir, ignore_errors=True)
        pass

if __name__ == '__main__':
    success = test_create_video()
    sys.exit(0 if success else 1)

