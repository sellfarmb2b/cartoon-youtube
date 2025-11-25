import json
import os
import platform
import shutil
import subprocess
import sys
import time

import requests
from packaging import version

from utils import resource_path

# PyInstaller로 빌드된 경우 sys.executable이 실행 파일 경로
def get_launcher_dir():
    """Launcher 실행 파일이 있는 디렉토리 경로 반환"""
    if getattr(sys, 'frozen', False):
        # PyInstaller로 빌드된 경우
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        # 개발 환경
        return os.path.dirname(os.path.abspath(__file__))


REMOTE_VERSION_URL = "https://raw.githubusercontent.com/sellfarmb2b/cartoon-youtube/refs/heads/main/src/version.json"
RELEASE_DOWNLOAD_URL = (
    "https://github.com/sellfarmb2b/cartoon-youtube/releases/download/v{version}/{filename}"
)


class LauncherApp:
    def __init__(self):
        # OS 감지
        self.is_windows = platform.system().lower() == "windows"
        self.executable_name = "YouTubeMaker.exe" if self.is_windows else "YouTubeMaker"

        # 실행 파일 위치 찾기 (여러 위치 시도)
        launcher_dir = get_launcher_dir()
        self.base_dir = resource_path("src")
        
        # 여러 가능한 위치에서 실행 파일 찾기
        possible_dirs = [
            launcher_dir,  # Launcher.exe와 같은 디렉토리 (가장 가능성 높음)
            self.base_dir,  # src 폴더
            os.path.join(launcher_dir, "src"),  # Launcher.exe 옆의 src 폴더
            os.path.dirname(launcher_dir),  # 상위 디렉토리
            os.path.join(os.path.dirname(launcher_dir), "src"),  # 상위 디렉토리의 src
        ]
        
        # 실행 파일 찾기
        self.target_executable = None
        for dir_path in possible_dirs:
            test_path = os.path.join(dir_path, self.executable_name)
            if os.path.exists(test_path):
                self.target_executable = test_path
                self.base_dir = dir_path
                print(f"[Launcher] 실행 파일 찾음: {self.target_executable}")
                break
        
        # 찾지 못한 경우 기본 경로 사용
        if not self.target_executable:
            self.target_executable = os.path.join(launcher_dir, self.executable_name)
            print(f"[Launcher] 실행 파일 기본 경로 설정: {self.target_executable}")
        
        self.local_version_file = os.path.join(self.base_dir, "version.json")

    def _load_local_version(self) -> version.Version:
        try:
            with open(self.local_version_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return version.parse(data.get("version", "0.0.0"))
        except Exception:
            return version.parse("0.0.0")

    def _save_local_version(self, new_version: str) -> None:
        os.makedirs(os.path.dirname(self.local_version_file), exist_ok=True)
        with open(self.local_version_file, "w", encoding="utf-8") as f:
            json.dump({"version": new_version}, f, ensure_ascii=False, indent=2)

    def _fetch_remote_version(self) -> version.Version:
        response = requests.get(REMOTE_VERSION_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        return version.parse(data.get("version", "0.0.0"))

    def _download_release(self, new_version: str) -> None:
        download_url = RELEASE_DOWNLOAD_URL.format(version=new_version, filename=self.executable_name)
        print(f"업데이트 다운로드 중... (v{new_version})")
        temp_path = os.path.join(self.base_dir, f"{self.executable_name}_{new_version}.tmp")

        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 8192

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress_value = (downloaded / total_size) * 100
                    # 진행률을 정수로 표시 (소수점 제거)
                    progress_int = int(progress_value)
                    # 같은 줄에 덮어쓰기 (줄바꿈 없이)
                    print(f"\r업데이트 다운로드 중... ({progress_int}%)", end="", flush=True)
                else:
                    print(f"\r업데이트 다운로드 중... (크기 확인 중)", end="", flush=True)

        print()  # 줄바꿈
        shutil.move(temp_path, self.target_executable)
        self._save_local_version(new_version)
        print("업데이트가 완료되었습니다.")

    def _check_for_updates(self) -> None:
        try:
            print("현재 버전 확인 중...")
            local_version = self._load_local_version()
            print(f"로컬 버전: {local_version}")
            
            try:
                print("원격 버전 확인 중...")
                remote_version = self._fetch_remote_version()
                print(f"원격 버전: {remote_version}")
                
                if remote_version > local_version:
                    self._download_release(str(remote_version))
                else:
                    print("최신 버전입니다!")
                    time.sleep(0.5)
            except requests.RequestException as network_error:
                # 네트워크 오류 시 조용히 무시하고 로컬 버전 실행
                print(f"업데이트 확인 건너뜀 (오프라인 모드): {network_error}")
                time.sleep(0.5)

            self._launch_main_app()
        except Exception as exc:
            # 치명적 오류가 아닌 경우에도 앱 실행 시도
            print(f"업데이트 확인 중 오류 발생, 앱 실행 시도 중...: {exc}")
            time.sleep(0.5)
            try:
                self._launch_main_app()
            except Exception as launch_exc:
                print(f"앱 실행 실패: {launch_exc}")
                input("에러 발생! 엔터를 누르면 종료합니다...")
                sys.exit(1)

    def _launch_main_app(self) -> None:
        # 실행 파일이 없으면 여러 위치에서 다시 찾기
        if not os.path.exists(self.target_executable):
            launcher_dir = get_launcher_dir()
            possible_paths = [
                os.path.join(launcher_dir, self.executable_name),
                os.path.join(self.base_dir, self.executable_name),
                os.path.join(launcher_dir, "src", self.executable_name),
                os.path.join(os.path.dirname(launcher_dir), self.executable_name),
                os.path.join(os.path.dirname(launcher_dir), "src", self.executable_name),
            ]
            
            found = False
            for test_path in possible_paths:
                if os.path.exists(test_path):
                    self.target_executable = test_path
                    self.base_dir = os.path.dirname(test_path)
                    found = True
                    print(f"[Launcher] 실행 파일 재검색 성공: {self.target_executable}")
                    break
            
            if not found:
                error_msg = f"{self.executable_name}를 찾을 수 없습니다.\n검색한 경로:\n" + "\n".join(possible_paths)
                print(error_msg)
                input("에러 발생! 엔터를 누르면 종료합니다...")
                sys.exit(1)

        print("실행 중...")
        try:
            if self.is_windows:
                os.startfile(self.target_executable)  # type: ignore[attr-defined]
            else:
                # macOS: 실행 권한 부여 후 실행
                os.chmod(self.target_executable, 0o755)
                subprocess.Popen([self.target_executable])
        except Exception as exc:
            print(f"애플리케이션 실행에 실패했습니다: {exc}")
            input("에러 발생! 엔터를 누르면 종료합니다...")
            sys.exit(1)


def main():
    print("=" * 60)
    print("YouTube Maker Launcher")
    print("=" * 60)
    print()
    
    try:
        launcher = LauncherApp()
        launcher._check_for_updates()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n치명적 오류 발생: {exc}")
        import traceback
        traceback.print_exc()
        input("에러 발생! 엔터를 누르면 종료합니다...")
        sys.exit(1)


if __name__ == "__main__":
    main()
