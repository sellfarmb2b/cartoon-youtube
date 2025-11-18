# 유튜브 영상 생성기

대본을 입력하면 자동으로 TTS, 이미지 생성, 영상 합성을 수행하는 AI 기반 유튜브 영상 생성 도구입니다.

## 🚀 빠른 시작

### 1. 사전 요구사항

- **Python 3.8 이상**
- **FFmpeg** (시스템에 설치 필요)
  - Mac: `brew install ffmpeg`
  - Windows: [FFmpeg 다운로드](https://ffmpeg.org/download.html)
  - Linux: `sudo apt install ffmpeg` 또는 `sudo yum install ffmpeg`

### 2. 프로젝트 클론/다운로드

```bash
# Git을 사용하는 경우
git clone <저장소_URL>
cd 유튜브\ 프로젝트

# 또는 ZIP 파일을 다운로드한 경우
# 압축 해제 후 해당 폴더로 이동
```

### 3. 가상 환경 설정

```bash
# 가상 환경 생성
python3 -m venv venv

# 가상 환경 활성화
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 4. 의존성 설치

```bash
pip install -r requirements.txt
```

### 5. 환경 변수 설정

#### 방법 1: `.env.local` 파일 생성 (권장)

프로젝트 루트 디렉토리에 `.env.local` 파일을 만들고 아래 내용을 입력하세요:

```bash
# .env.local 파일 내용
ELEVENLABS_API_KEY="여기에_ElevenLabs_API_키_입력"
STABILITY_API_KEY="여기에_Stability_API_키_입력"
REPLICATE_API_TOKEN="여기에_Replicate_API_토큰_입력"
OPENAI_API_KEY="여기에_OpenAI_API_키_입력"
```

#### 방법 2: 시스템 환경 변수 설정

**Mac/Linux:**
```bash
export ELEVENLABS_API_KEY="여기에_API_키_입력"
export STABILITY_API_KEY="여기에_API_키_입력"
export REPLICATE_API_TOKEN="여기에_API_토큰_입력"
export OPENAI_API_KEY="여기에_API_키_입력"
```

**Windows (PowerShell):**
```powershell
$env:ELEVENLABS_API_KEY="여기에_API_키_입력"
$env:STABILITY_API_KEY="여기에_API_키_입력"
$env:REPLICATE_API_TOKEN="여기에_API_토큰_입력"
$env:OPENAI_API_KEY="여기에_API_키_입력"
```

**Windows (CMD):**
```cmd
set ELEVENLABS_API_KEY=여기에_API_키_입력
set STABILITY_API_KEY=여기에_API_키_입력
set REPLICATE_API_TOKEN=여기에_API_토큰_입력
set OPENAI_API_KEY=여기에_API_키_입력
```

> 💡 **참고**: `.env.example` 파일을 참고하여 필요한 환경 변수를 확인하세요.

### 6. 서버 실행

```bash
python app.py
```

서버가 시작되면 터미널에 다음과 같은 메시지가 표시됩니다:

```
============================================================
유튜브 영상 생성기 서버 시작
주소: http://127.0.0.1:5002
============================================================
```

브라우저에서 `http://127.0.0.1:5002` (또는 표시된 주소)로 접속하세요.

## 📖 사용 방법

1. **대본 입력**: 메인 페이지의 텍스트 영역에 대본을 입력합니다.
   - 형식 1: 일반 텍스트 (자동으로 프롬프트 생성)
   - 형식 2: 구조화된 형식
     ```
     [한국어 번역]
     문장 1
     문장 2
     
     [영어 이미지 프롬프트]
     English prompt 1
     English prompt 2
     ```

2. **모드 선택**: "애니메이션" 또는 "리얼리스틱" 모드를 선택합니다.

3. **이미지 생성**: "이미지 생성" 버튼을 클릭합니다.

4. **영상 생성**: 이미지 생성이 완료되면 "영상 생성" 버튼을 클릭합니다.

5. **다운로드**: 생성된 영상을 다운로드합니다.

## 🔧 문제 해결

### 포트가 이미 사용 중인 경우

서버가 자동으로 다른 포트를 찾습니다. 터미널에 표시된 주소를 확인하세요.

### FFmpeg를 찾을 수 없는 경우

FFmpeg가 시스템 PATH에 있는지 확인하세요:
```bash
ffmpeg -version
```

### API 키 오류

환경 변수가 제대로 설정되었는지 확인하세요:
```bash
# Mac/Linux
echo $ELEVENLABS_API_KEY

# Windows (PowerShell)
echo $env:ELEVENLABS_API_KEY
```

### 의존성 설치 오류

가상 환경이 활성화되어 있는지 확인하고, Python 버전을 확인하세요:
```bash
python --version  # Python 3.8 이상 필요
```

## 📦 실행 파일 빌드

### Mac용 실행 파일 생성

```bash
python build_mac.py
```

생성된 파일: `dist/유튜브_영상_생성기.app`

### Windows용 실행 파일 생성

```bash
python build_windows.py
```

생성된 파일: `dist/유튜브_영상_생성기.exe`

자세한 내용은 `BUILD_GUIDE.md`를 참고하세요.

## 🔐 보안 주의사항

- ⚠️ **절대 `.env.local` 파일을 Git에 커밋하지 마세요!**
- ⚠️ API 키는 환경 변수로만 관리하세요.
- ⚠️ 공개 저장소에 API 키를 업로드하지 마세요.

## 📝 주요 기능

- ✅ 자동 TTS 생성 (ElevenLabs)
- ✅ AI 이미지 생성 (Replicate, Stability AI)
- ✅ 자동 프롬프트 생성 (OpenAI)
- ✅ 자막 자동 생성 및 동기화
- ✅ 영상 자동 합성 (FFmpeg)
- ✅ 긴 영상 청크 분할 처리
- ✅ 테스트 모드 지원

## 🛠️ 기술 스택

- **Backend**: Flask (Python)
- **TTS**: ElevenLabs API
- **이미지 생성**: Replicate API, Stability AI
- **프롬프트 생성**: OpenAI API
- **비디오 처리**: FFmpeg

## 📄 라이선스

이 프로젝트는 개인 사용을 위한 것입니다.

---

**마지막 업데이트**: Git 자동화 규칙 테스트 중

