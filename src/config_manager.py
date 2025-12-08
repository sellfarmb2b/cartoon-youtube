import json
import os
from threading import RLock
from typing import Any, Dict, Optional

from appdirs import user_data_dir


class ConfigManager:
    """
    Persist application settings (e.g., API keys) in the OS-specific
    application data directory so they remain available after packaging.
    """

    def __init__(self, app_name: str = "YouTubeMaker", app_author: str = "YouTubeMaker"):
        self.app_name = app_name
        self.app_author = app_author
        self._lock = RLock()
        self._settings: Dict[str, Any] = {
            "replicate_api_key": "",
            "elevenlabs_api_key": "",
            "openai_api_key": "",
            "gemini_api_key": "",
            "download_folder_path": "",
            "custom_voice_ids": [],  # 사용자 정의 보이스 ID 목록
        }

        self.config_dir = user_data_dir(self.app_name, self.app_author)
        os.makedirs(self.config_dir, exist_ok=True)
        self.config_path = os.path.join(self.config_dir, "settings.json")
        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load settings from disk if the file exists."""
        with self._lock:
            if not os.path.exists(self.config_path):
                return
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key in self._settings:
                                # custom_voice_ids는 리스트로 유지
                                if key == "custom_voice_ids":
                                    if isinstance(value, list):
                                        self._settings[key] = value
                                    else:
                                        self._settings[key] = []
                                else:
                                    self._settings[key] = (value or "")
            except Exception:
                # If the file is corrupted, we keep defaults
                pass

    def _save(self) -> None:
        """Persist settings to disk."""
        with self._lock:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str, default: Optional[Any] = "") -> Any:
        with self._lock:
            return self._settings.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._settings)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._settings:
                self._settings[key] = value or ""
                self._save()

    def update(self, data: Dict[str, Any]) -> None:
        changed = False
        with self._lock:
            for key, value in data.items():
                if key in self._settings:
                    normalized = value or ""
                    if self._settings.get(key) != normalized:
                        self._settings[key] = normalized
                        changed = True
            if changed:
                self._save()

