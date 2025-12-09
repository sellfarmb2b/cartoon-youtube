#!/usr/bin/env python3
"""
create_video 함수 간단 테스트 스크립트
더미 데이터를 사용하여 영상 합성 기능을 테스트합니다.

사용법:
  python3 test_create_video_simple.py

필수 패키지:
  pip install mutagen pillow ffmpeg-python
"""

import os
import sys
import tempfile

def check_dependencies():
    """필수 패키지 확인"""
    missing = []
    
    try:
        import mutagen
    except ImportError:
        missing.append("mutagen")
    
    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")
    
    try:
        import ffmpeg
    except ImportError:
        missing.append("ffmpeg-python")
    
    if missing:
        print("=" * 80)
        print("필수 패키지가 설치되지 않았습니다:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n다음 명령어로 설치해주세요:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 80)
        return False
    
    return True

def main():
    """메인 함수"""
    print("=" * 80)
    print("create_video 함수 테스트")
    print("=" * 80)
    
    # 패키지 확인
    if not check_dependencies():
        return 1
    
    # src 디렉토리를 경로에 추가
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.insert(0, src_path)
    
    try:
        # app.py에서 필요한 함수들만 import
        print("\n[1단계] 모듈 로드 중...")
        
        # 필요한 전역 변수 설정
        os.environ.setdefault('ELEVENLABS_API_KEY', '')
        os.environ.setdefault('REPLICATE_API_TOKEN', '')
        os.environ.setdefault('GEMINI_API_KEY', '')
        
        # app.py import (일부 오류는 무시)
        import warnings
        warnings.filterwarnings('ignore')
        
        from app import create_video
        from mutagen.mp3 import MP3
        from PIL import Image
        import ffmpeg
        
        print("  ✅ 모듈 로드 완료")
        
    except Exception as e:
        print(f"  ❌ 모듈 로드 실패: {e}")
        print("\n필요한 패키지를 모두 설치했는지 확인해주세요:")
        print("  pip install -r requirements.txt")
        return 1
    
    # 테스트 데이터 생성
    print("\n[2단계] 더미 데이터 생성 중...")
    
    test_dir = tempfile.mkdtemp(prefix='test_video_')
    assets_folder = os.path.join(test_dir, 'assets')
    os.makedirs(assets_folder, exist_ok=True)
    
    print(f"  작업 디렉토리: {test_dir}")
    
    try:
        # 더미 이미지 생성
        def create_image(color, path):
            img = Image.new('RGB', (1920, 1080), color=color)
            img.save(path)
            print(f"  ✅ 이미지 생성: {os.path.basename(path)}")
        
        # 더미 오디오 생성
        def create_audio(duration, path):
            (
                ffmpeg
                .input('anullsrc=channel_layout=stereo:sample_rate=44100', f='lavfi', t=duration)
                .output(path, acodec='libmp3lame', ac=2, ar=44100)
                .run(overwrite_output=True, quiet=True)
            )
            print(f"  ✅ 오디오 생성: {os.path.basename(path)} ({duration}초)")
        
        # 더미 비디오 생성
        def create_video_file(duration, color, path):
            (
                ffmpeg
                .input(f'color=c={color}:s=1920x1080:d={duration}', f='lavfi')
                .output(path, vcodec='libx264', pix_fmt='yuv420p', r=25, t=duration)
                .run(overwrite_output=True, quiet=True)
            )
            print(f"  ✅ 비디오 생성: {os.path.basename(path)} ({duration}초)")
        
        # 씬 1: 이미지 + 오디오 (3초)
        scene1_image = os.path.join(assets_folder, 'scene_1.png')
        scene1_audio = os.path.join(assets_folder, 'scene_1.mp3')
        create_image((255, 100, 100), scene1_image)
        create_audio(3.0, scene1_audio)
        
        # 씬 2: 비디오(2초) + 오디오(4초) - 비디오 반복 테스트
        scene2_video = os.path.join(assets_folder, 'scene_2.mp4')
        scene2_audio = os.path.join(assets_folder, 'scene_2.mp3')
        create_video_file(2.0, 'blue', scene2_video)
        create_audio(4.0, scene2_audio)
        
        # 씬 3: 이미지 + 오디오 (5초)
        scene3_image = os.path.join(assets_folder, 'scene_3.png')
        scene3_audio = os.path.join(assets_folder, 'scene_3.mp3')
        create_image((100, 255, 100), scene3_image)
        create_audio(5.0, scene3_audio)
        
        # 씬 4: 비디오(6초) + 오디오(3초) - 비디오 자르기 테스트
        scene4_video = os.path.join(assets_folder, 'scene_4.mp4')
        scene4_audio = os.path.join(assets_folder, 'scene_4.mp3')
        create_video_file(6.0, 'red', scene4_video)
        create_audio(3.0, scene4_audio)
        
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
        
        print(f"\n[3단계] create_video 함수 실행 중...")
        print(f"  총 {len(scene_data)}개 씬 처리")
        print(f"  - 씬 1: 이미지 (3초)")
        print(f"  - 씬 2: 비디오 2초 + 오디오 4초 (비디오 반복 테스트)")
        print(f"  - 씬 3: 이미지 (5초)")
        print(f"  - 씬 4: 비디오 6초 + 오디오 3초 (비디오 자르기 테스트)")
        
        subtitle_file = os.path.join(test_dir, 'subtitles.srt')
        output_video_file = os.path.join(test_dir, 'test_output.mp4')
        
        def progress_cb(message):
            print(f"  [진행] {message}")
        
        # create_video 함수 호출
        result = create_video(
            scene_data_with_timestamps=scene_data,
            char_limit=50,
            subtitle_file=subtitle_file,
            output_video_file=output_video_file,
            assets_folder=assets_folder,
            progress_cb=progress_cb,
            include_subtitles=True
        )
        
        if result and os.path.exists(output_video_file):
            print(f"\n[4단계] 결과 확인")
            file_size = os.path.getsize(output_video_file)
            print(f"  ✅ 영상 생성 완료!")
            print(f"  - 출력 파일: {output_video_file}")
            print(f"  - 파일 크기: {file_size / (1024 * 1024):.2f} MB")
            
            # 비디오 정보 확인
            try:
                probe = ffmpeg.probe(output_video_file)
                video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
                audio_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
                
                if video_streams:
                    video_duration = float(video_streams[0].get('duration', 0))
                    print(f"  - 비디오 길이: {video_duration:.2f}초")
                    print(f"  - 해상도: {video_streams[0].get('width')}x{video_streams[0].get('height')}")
                
                if audio_streams:
                    audio_duration = float(audio_streams[0].get('duration', 0))
                    print(f"  - 오디오 길이: {audio_duration:.2f}초")
                
                expected_duration = sum(s['duration'] for s in scene_data)
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
            
            print(f"\n" + "=" * 80)
            print(f"테스트 완료!")
            print(f"출력 파일: {output_video_file}")
            print(f"테스트 디렉토리: {test_dir}")
            print(f"(테스트 디렉토리는 자동으로 삭제되지 않습니다)")
            print("=" * 80)
            return 0
        else:
            print(f"\n❌ 영상 생성 실패")
            return 1
            
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

