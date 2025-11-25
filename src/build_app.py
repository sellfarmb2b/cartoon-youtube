"""
PC 앱 빌드 스크립트
PyInstaller를 사용하여 실행 파일을 생성합니다.
"""
import os
import sys
import subprocess
import shutil

def build_app():
    """PyInstaller를 사용하여 앱을 빌드합니다."""
    
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
    if os.path.exists("app.spec"):
        os.remove("app.spec")
    
    # 템플릿과 static 폴더 확인
    if not os.path.exists("templates"):
        print("⚠️  templates 폴더를 찾을 수 없습니다.")
    if not os.path.exists("static"):
        print("⚠️  static 폴더를 찾을 수 없습니다.")
    
    # 경로 구분자 설정
    sep = ";" if sys.platform == "win32" else ":"
    
    # PyInstaller 명령 실행
    cmd = [
        "pyinstaller",
        "--name=유튜브_영상_생성기",
        "--onefile",
        "--noconsole" if sys.platform != "win32" else "--windowed",
        f"--add-data=templates{sep}templates",
        f"--add-data=static{sep}static",
        "--hidden-import=flask",
        "--hidden-import=werkzeug",
        "--hidden-import=requests",
        "--hidden-import=ffmpeg",
        "--hidden-import=PIL",
        "--hidden-import=mutagen",
        "--hidden-import=elevenlabs",
        "--hidden-import=replicate",
        "--hidden-import=openai",
        "--hidden-import=webbrowser",
        "--hidden-import=socket",
        "--collect-all=flask",
        "--collect-all=werkzeug",
        "app.py"
    ]
    
    print("빌드 시작...")
    print(f"명령: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print("\n✅ 빌드 완료!")
        print(f"실행 파일 위치: dist/유튜브_영상_생성기")
        if sys.platform == "win32":
            print("Windows: dist/유튜브_영상_생성기.exe")
        elif sys.platform == "darwin":
            print("macOS: dist/유튜브_영상_생성기.app")
        else:
            print("Linux: dist/유튜브_영상_생성기")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 빌드 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_app()

