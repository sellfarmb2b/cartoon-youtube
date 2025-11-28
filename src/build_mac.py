"""
Mac용 실행 파일 빌드 스크립트
PyInstaller를 사용하여 macOS 앱 번들을 생성합니다.
macOS 환경에서 실행해야 합니다.
"""
import os
import sys
import subprocess
import shutil

def build_mac():
    """Mac용 앱 번들을 빌드합니다."""
    
    if sys.platform != "darwin":
        print("⚠️  이 스크립트는 macOS 환경에서만 실행할 수 있습니다.")
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
    if os.path.exists("Launcher.spec"):
        os.remove("Launcher.spec")
    if os.path.exists("YouTubeMaker.spec"):
        os.remove("YouTubeMaker.spec")
    
    # 템플릿과 static 폴더 확인
    if not os.path.exists("src/templates"):
        print("⚠️  src/templates 폴더를 찾을 수 없습니다.")
        return False
    if not os.path.exists("src/static"):
        print("⚠️  src/static 폴더를 찾을 수 없습니다.")
        os.makedirs("src/static", exist_ok=True)
    
    print("=" * 60)
    print("Mac용 앱 번들 빌드 시작...")
    print("=" * 60)
    
    # 1. Launcher 빌드
    print("\n[1/2] Launcher 빌드 중...")
    launcher_cmd = [
        "pyinstaller",
        "--name=Launcher",
        "--onefile",
        # --noconsole 제거 (터미널 창이 보여야 진행 상황을 알 수 있음)
        "--add-data=src/version.json:src",
        "--exclude-module=tkinter",  # tkinter 제외 (macOS에서 모듈 없음 오류 방지)
        "src/launcher.py"
    ]
    print(f"명령: {' '.join(launcher_cmd)}")
    try:
        subprocess.check_call(launcher_cmd)
        print("✅ Launcher 빌드 완료")
    except subprocess.CalledProcessError as e:
        print(f"❌ Launcher 빌드 실패: {e}")
        return False
    
    # 2. YouTubeMaker 빌드
    print("\n[2/2] YouTubeMaker 빌드 중...")
    app_cmd = [
        "pyinstaller",
        "--name=YouTubeMaker",
        "--onedir",
        "--noconsole",
        "--add-data=src/templates:src/templates",
        "--add-data=src/static:src/static",
        "--add-data=bin/mac/ffmpeg:bin/mac",
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
        "--hidden-import=pywebview",
        "--hidden-import=appdirs",
        "--hidden-import=webbrowser",
        "--hidden-import=socket",
        "--hidden-import=threading",
        "--hidden-import=concurrent.futures",
        "--hidden-import=uuid",
        "--exclude-module=tkinter",
        "--exclude-module=matplotlib",
        "--exclude-module=scipy",
        "--exclude-module=pandas",
        "src/app.py"
    ]
    print(f"명령: {' '.join(app_cmd)}")
    try:
        subprocess.check_call(app_cmd)
        print("✅ YouTubeMaker 빌드 완료")
    except subprocess.CalledProcessError as e:
        print(f"❌ YouTubeMaker 빌드 실패: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Mac 빌드 완료!")
    print("=" * 60)
    print(f"실행 파일 위치:")
    print(f"  - dist/Launcher")
    print(f"  - dist/YouTubeMaker/YouTubeMaker")
    print("\n💡 사용 방법:")
    print("   1. dist 폴더의 'Launcher' 파일을 더블클릭하여 실행")
    print("   2. Launcher가 자동으로 YouTubeMaker를 업데이트하고 실행합니다")
    print("   3. YouTubeMaker는 dist/YouTubeMaker/ 폴더 안에 생성됩니다")
    print("\n⚠️  참고: macOS에서 처음 실행 시 보안 경고가 나타날 수 있습니다.")
    print("   '시스템 환경설정 > 보안 및 개인 정보 보호'에서 허용해주세요.")
    return True

if __name__ == "__main__":
    success = build_mac()
    sys.exit(0 if success else 1)

