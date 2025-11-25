import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import requests
from packaging import version

from utils import resource_path


REMOTE_VERSION_URL = "https://raw.githubusercontent.com/sellfarmb2b/cartoon-youtube/main/src/version.json"
RELEASE_DOWNLOAD_URL = (
    "https://github.com/sellfarmb2b/cartoon-youtube/releases/download/v{version}/{filename}"
)


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YouTube Maker Launcher")
        self.root.geometry("400x250")
        self.root.resizable(False, False)

        # OS 감지
        self.is_windows = platform.system().lower() == "windows"
        self.executable_name = "YouTubeMaker.exe" if self.is_windows else "YouTubeMaker"

        self.base_dir = resource_path("src")
        self.local_version_file = os.path.join(self.base_dir, "version.json")
        self.target_executable = os.path.join(self.base_dir, self.executable_name)

        self._create_widgets()
        threading.Thread(target=self._check_for_updates, daemon=True).start()

    def _create_widgets(self) -> None:
        self.root.configure(bg="#0f172a")

        self.logo_label = tk.Label(
            self.root,
            text="YouTube Maker",
            font=("Segoe UI", 20, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
        )
        self.logo_label.pack(pady=(30, 10))

        self.status_label = tk.Label(
            self.root,
            text="버전을 확인하는 중...",
            font=("Segoe UI", 12),
            fg="#e2e8f0",
            bg="#0f172a",
        )
        self.status_label.pack(pady=10)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=260, mode="determinate")
        self.progress.pack(pady=20)
        self.progress.pack_forget()

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_label.config(text=text))

    def _show_progress(self, value: float) -> None:
        def _update():
            if not self.progress.winfo_ismapped():
                self.progress.pack(pady=20)
            self.progress["value"] = value

        self.root.after(0, _update)

    def _hide_progress(self) -> None:
        self.root.after(0, self.progress.pack_forget)

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
        self._set_status(f"업데이트 중... (v{new_version})")
        temp_path = os.path.join(self.base_dir, f"{self.executable_name}_{new_version}.tmp")

        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 8192

        self._show_progress(0)

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress_value = (downloaded / total_size) * 100
                    self._show_progress(progress_value)
                else:
                    self.progress.config(mode="indeterminate")
                    self.progress.start(10)

        if total_size == 0:
            self.progress.stop()
            self.progress.config(mode="determinate")

        shutil.move(temp_path, self.target_executable)
        self._save_local_version(new_version)
        self._set_status("업데이트가 완료되었습니다.")
        self._hide_progress()

    def _check_for_updates(self) -> None:
        try:
            local_version = self._load_local_version()
            remote_version = self._fetch_remote_version()

            if remote_version > local_version:
                self._download_release(str(remote_version))
            else:
                self._set_status("최신 버전입니다!")
                time.sleep(1)

            self._launch_main_app()
        except requests.RequestException as network_error:
            self._set_status("네트워크 오류가 발생했습니다.")
            messagebox.showerror("오류", f"업데이트 서버에 연결할 수 없습니다.\n{network_error}")
        except Exception as exc:
            self._set_status("업데이트 중 오류가 발생했습니다.")
            messagebox.showerror("오류", f"알 수 없는 오류가 발생했습니다.\n{exc}")
        finally:
            self._hide_progress()

    def _launch_main_app(self) -> None:
        if not os.path.exists(self.target_executable):
            self._set_status(f"{self.executable_name}를 찾을 수 없습니다.")
            return

        try:
            if self.is_windows:
                os.startfile(self.target_executable)  # type: ignore[attr-defined]
            else:
                # macOS: 실행 권한 부여 후 실행
                os.chmod(self.target_executable, 0o755)
                subprocess.Popen([self.target_executable])
        except Exception as exc:
            messagebox.showerror("실행 오류", f"애플리케이션 실행에 실패했습니다.\n{exc}")
            return

        self.root.after(1000, self.root.destroy)


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

