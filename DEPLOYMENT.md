# PC 앱 배포 가이드

이 가이드는 유튜브 영상 생성기를 PC 앱으로 배포하는 방법을 설명합니다.

## 방법 1: PyInstaller를 사용한 실행 파일 생성 (권장)

### 사전 준비

1. **필요한 패키지 설치**
   ```bash
   pip install pyinstaller
   ```

2. **FFmpeg 설치 확인**
   - Windows: [FFmpeg 다운로드](https://ffmpeg.org/download.html) 후 PATH에 추가
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` (Ubuntu/Debian)

### 빌드 방법

1. **빌드 스크립트 실행**
   ```bash
   python build_app.py
   ```

2. **빌드 결과**
   - Windows: `dist/유튜브_영상_생성기.exe`
   - macOS: `dist/유튜브_영상_생성기.app`
   - Linux: `dist/유튜브_영상_생성기`

### 배포 시 주의사항

1. **FFmpeg 포함**
   - 실행 파일에는 FFmpeg가 포함되지 않습니다.
   - 사용자 PC에 FFmpeg가 설치되어 있어야 합니다.
   - 또는 FFmpeg를 별도로 배포하고 경로를 설정해야 합니다.

2. **API 키 관리**
   - 현재는 코드에 하드코딩되어 있습니다.
   - 배포 전에 환경 변수나 설정 파일로 변경하는 것을 권장합니다.

3. **파일 크기**
   - PyInstaller로 빌드된 실행 파일은 약 100-200MB 정도입니다.
   - 모든 Python 패키지가 포함되기 때문입니다.

## 방법 2: Electron을 사용한 데스크톱 앱

더 나은 사용자 경험을 원한다면 Electron을 사용할 수 있습니다.

### 장점
- 더 작은 파일 크기
- 네이티브 앱처럼 보임
- 자동 업데이트 기능 추가 가능

### 단점
- 추가 개발 시간 필요
- Node.js 환경 필요

## 방법 3: 웹 서버로 배포

사용자가 브라우저에서 접속하도록 웹 서버로 배포할 수도 있습니다.

### 옵션
- **Heroku**: 무료 티어 제공
- **AWS EC2**: 클라우드 서버
- **DigitalOcean**: 간단한 VPS
- **로컬 네트워크**: 회사 내부 네트워크에서만 사용

## 사용자 배포 가이드

### 최소 시스템 요구사항
- Windows 10 이상 / macOS 10.14 이상 / Linux (Ubuntu 18.04 이상)
- 4GB RAM 이상
- 2GB 이상의 여유 디스크 공간
- 인터넷 연결 (API 호출용)

### 설치 방법

1. **실행 파일 다운로드**
   - Windows: `유튜브_영상_생성기.exe` 다운로드
   - macOS: `유튜브_영상_생성기.app` 다운로드
   - Linux: `유튜브_영상_생성기` 다운로드

2. **FFmpeg 설치** (필수)
   - Windows: [FFmpeg 다운로드](https://ffmpeg.org/download.html)
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

3. **실행**
   - 실행 파일을 더블클릭
   - 자동으로 브라우저가 열립니다
   - 브라우저가 열리지 않으면 `http://127.0.0.1:5001` 접속

## 문제 해결

### 포트 충돌
- 다른 포트(5002, 5003 등)를 자동으로 찾아 사용합니다.

### FFmpeg 오류
- FFmpeg가 PATH에 있는지 확인하세요.
- `ffmpeg -version` 명령어로 확인 가능합니다.

### API 키 오류
- `app.py` 파일에서 API 키를 확인하세요.
- 환경 변수로 설정하는 것을 권장합니다.

## 추가 개선 사항

1. **설정 파일 분리**
   - API 키를 별도 설정 파일로 분리
   - 사용자가 직접 입력할 수 있도록 UI 추가

2. **자동 업데이트**
   - GitHub Releases를 통한 자동 업데이트 체크

3. **설치 프로그램**
   - Inno Setup (Windows) 또는 DMG (macOS)로 설치 프로그램 생성

4. **코드 서명**
   - 보안을 위한 디지털 서명 추가

