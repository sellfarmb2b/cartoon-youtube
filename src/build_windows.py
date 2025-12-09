"""
Windows용 실행 파일 빌드 스크립트
PyInstaller를 사용하여 .exe 파일을 생성합니다.
Windows 환경에서 실행해야 합니다.
"""
import os
import sys
import subprocess
import shutil

def build_windows():
    """Windows용 .exe 파일을 빌드합니다."""
    
    if sys.platform != "win32":
        print("⚠️  이 스크립트는 Windows 환경에서만 실행할 수 있습니다.")
        print("현재 플랫폼:", sys.platform)
        return False
    
    # 필요한 패키지 설치 확인
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller가 설치되지 않았습니다. 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # 기존 빌드 폴더 정리
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("유튜브_영상_생성기.spec"):
        os.remove("유튜브_영상_생성기.spec")
    
    # 템플릿과 static 폴더 확인
    if not os.path.exists("templates"):
        print("⚠️  templates 폴더를 찾을 수 없습니다.")
        return False
    if not os.path.exists("static"):
        print("⚠️  static 폴더를 찾을 수 없습니다.")
        os.makedirs("static", exist_ok=True)
    
    # PyInstaller 명령 실행
    cmd = [
        "pyinstaller",
        "--name=유튜브_영상_생성기",
        "--onefile",
        "--windowed",  # Windows: 콘솔 창 숨김
        "--icon=NONE",  # 아이콘 파일이 있으면 여기에 경로 지정
        "--add-data=templates;templates",  # Windows는 세미콜론 사용
        "--add-data=static;static",
        "--hidden-import=flask",
        "--hidden-import=werkzeug",
        "--hidden-import=requests",
        "--hidden-import=ffmpeg",
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        "--hidden-import=PIL.ImageOps",
        "--hidden-import=mutagen",
        "--hidden-import=mutagen.mp3",
        "--hidden-import=elevenlabs",
        "--hidden-import=elevenlabs.client",
        "--hidden-import=replicate",
        "--hidden-import=openai",
        "--hidden-import=google",
        "--hidden-import=google.genai",
        "--hidden-import=google.genai.types",
        "--collect-all=google.genai",  # google-genai 패키지의 모든 서브모듈과 메타데이터 포함
        "--hidden-import=webbrowser",
        "--hidden-import=socket",
        "--hidden-import=threading",
        "--hidden-import=concurrent.futures",
        "--hidden-import=zipfile",
        "--hidden-import=base64",
        "--hidden-import=json",
        "--hidden-import=uuid",
        "--collect-all=flask",
        "--collect-all=werkzeug",
        "--collect-all=PIL",
        "app.py"
    ]
    
    print("=" * 60)
    print("Windows용 실행 파일 빌드 시작...")
    print("=" * 60)
    print(f"명령: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.check_call(cmd)
        print("\n" + "=" * 60)
        print("✅ Windows 빌드 완료!")
        print("=" * 60)
        print(f"실행 파일 위치: dist\\유튜브_영상_생성기.exe")
        print("\n💡 사용 방법:")
        print("   dist 폴더의 '유튜브_영상_생성기.exe' 파일을 더블클릭하여 실행하세요.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 빌드 실패: {e}")
        return False

if __name__ == "__main__":
    success = build_windows()
    sys.exit(0 if success else 1)

