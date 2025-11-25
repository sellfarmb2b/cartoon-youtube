# API 키 사용 분석 보고서

## 1. API 키가 자동으로 채워지는 원인

### 저장된 설정에서 불러오기
- **위치**: `src/config_manager.py`
- **저장 위치**: OS별 Application Support/AppData 폴더의 `settings.json`
  - Windows: `C:\Users\<username>\AppData\Local\YouTubeMaker\YouTubeMaker\settings.json`
  - macOS: `~/Library/Application Support/YouTubeMaker/YouTubeMaker/settings.json`
- **동작**: 앱 시작 시 `ConfigManager._load()`가 저장된 설정 파일을 읽어서 API 키를 불러옴

### 환경 변수에서 불러오기
- **위치**: `src/app.py`의 `reload_api_keys()` 함수
- **우선순위**: 
  1. 저장된 설정 (`config_manager.get_all()`)
  2. 환경 변수 (`os.environ.get()`)
- **환경 변수 이름**:
  - `ELEVENLABS_API_KEY`
  - `REPLICATE_API_TOKEN`
  - `OPENAI_API_KEY`

### 프론트엔드에서 불러오기
- **위치**: `src/templates/index.html`의 `loadSettings()` 함수
- **동작**: 페이지 로드 시 `/api/settings` 엔드포인트를 호출하여 저장된 API 키를 불러옴
- **코드**: 
  ```javascript
  async function loadSettings() {
      const response = await fetch('/api/settings');
      const data = await response.json();
      settingsState = {
          replicate_api_key: data.replicate_api_key || '',
          elevenlabs_api_key: data.elevenlabs_api_key || '',
          openai_api_key: data.openai_api_key || '',
      };
      populateSettingsForm(); // 입력 필드에 자동으로 채움
  }
  ```

## 2. 사용자 입력 API 키가 실제로 사용되는지 확인

### ✅ 이미지 생성 (Replicate API)

**프론트엔드** (`src/templates/index.html`):
```javascript
// 979-1000번 줄
const replicateApiKey = (settingsState.replicate_api_key || '').trim() || null;
const response = await fetch('/generate_images_direct', {
    body: JSON.stringify({ 
        replicate_api_key: replicateApiKey,  // 사용자 입력 API 키 전달
        ...
    })
});
```

**백엔드** (`src/app.py`):
```python
# 3542번 줄
replicate_api_key = data.get("replicate_api_key") or None

# 3634번 줄
success = generate_image(prompt, image_filename, mode=mode, replicate_api_key=replicate_api_key)

# 1182번 줄 (generate_image 함수 내부)
api_token = replicate_api_key or REPLICATE_API_TOKEN  # 사용자 입력 우선 사용
```

**결론**: ✅ 사용자가 입력한 API 키가 우선적으로 사용됨

### ✅ TTS 생성 (ElevenLabs API)

**프론트엔드** (`src/templates/index.html`):
```javascript
// 1437-1447번 줄
const elevenlabsApiKey = (settingsState.elevenlabs_api_key || '').trim() || null;
const payload = {
    elevenlabs_api_key: elevenlabsApiKey,  // 사용자 입력 API 키 전달
    ...
};
```

**백엔드** (`src/app.py`):
```python
# 3161번 줄
elevenlabs_api_key = payload.get("elevenlabs_api_key") or None

# 1538번 줄 (run_generation_job 함수 내부)
alignment = generate_tts_with_alignment(voice_id, text, audio_file, elevenlabs_api_key=elevenlabs_api_key)

# 1060번 줄 (generate_tts_with_alignment 함수 내부)
api_key = elevenlabs_api_key or ELEVENLABS_API_KEY  # 사용자 입력 우선 사용
```

**결론**: ✅ 사용자가 입력한 API 키가 우선적으로 사용됨

### ✅ 프롬프트 생성 (OpenAI API)

**현재 상태**: 프론트엔드에서 OpenAI API 키를 직접 전달하지 않음
- OpenAI API 키는 저장된 설정에서만 불러와서 사용됨
- `reload_api_keys()` 함수를 통해 전역 변수 `OPENAI_API_KEY`에 저장됨

## 3. API 키 우선순위

### 이미지 생성 (Replicate)
1. **사용자 입력** (`replicate_api_key` 파라미터) ← 최우선
2. 저장된 설정 (`config_manager.get("replicate_api_key")`)
3. 환경 변수 (`REPLICATE_API_TOKEN`)

### TTS 생성 (ElevenLabs)
1. **사용자 입력** (`elevenlabs_api_key` 파라미터) ← 최우선
2. 저장된 설정 (`config_manager.get("elevenlabs_api_key")`)
3. 환경 변수 (`ELEVENLABS_API_KEY`)

### 프롬프트 생성 (OpenAI)
1. 저장된 설정 (`config_manager.get("openai_api_key")`)
2. 환경 변수 (`OPENAI_API_KEY`)
- **참고**: 현재 프론트엔드에서 OpenAI API 키를 직접 전달하지 않음

## 4. 하드코딩된 API 키 확인

✅ **하드코딩된 API 키 없음**
- 코드베이스에서 실제 API 키 값이 하드코딩되어 있지 않음
- 모든 API 키는 설정 파일 또는 환경 변수에서 불러옴

## 5. 보안 확인

✅ **안전함**
- API 키는 로컬 설정 파일에만 저장됨
- 서버로 전송되지 않음 (로컬 앱이므로)
- 사용자가 입력한 API 키가 우선적으로 사용됨

## 6. 권장 사항

1. **사용자 입력 API 키 우선 사용**: ✅ 이미 구현됨
2. **OpenAI API 키도 프론트엔드에서 전달하도록 개선 가능**: 현재는 저장된 설정만 사용

