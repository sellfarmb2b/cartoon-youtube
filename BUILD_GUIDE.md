# 실행 파일 빌드 가이드

이 프로젝트를 Windows와 Mac용 실행 파일로 빌드하는 방법입니다.

## 사전 준비

1. **Python 3.13** 설치 확인
2. **가상환경 활성화** (선택사항이지만 권장)
   ```bash
   source venv/bin/activate  # Mac
   # 또는
   venv\Scripts\activate  # Windows
   ```
3. **필요한 패키지 설치**
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

## Windows용 빌드

### 방법 1: 빌드 스크립트 사용 (권장)

**Windows 환경에서** 다음 명령어를 실행하세요:

```bash
python build_windows.py
```

빌드가 완료되면 `dist/유튜브_영상_생성기.exe` 파일이 생성됩니다.

### 방법 2: 수동 빌드

```bash
pyinstaller --name=유튜브_영상_생성기 --onefile --windowed --add-data="templates;templates" --add-data="static;static" app.py
```

## Mac용 빌드

### 방법 1: 빌드 스크립트 사용 (권장)

**macOS 환경에서** 다음 명령어를 실행하세요:

```bash
python build_mac.py
```

빌드가 완료되면 `dist/유튜브_영상_생성기` 실행 파일과 `dist/유튜브_영상_생성기.app` 앱 번들이 생성됩니다.

### 방법 2: 수동 빌드

```bash
pyinstaller --name=유튜브_영상_생성기 --onefile --noconsole --add-data="templates:templates" --add-data="static:static" app.py
```

## 빌드 결과물

### Windows
- `dist/유튜브_영상_생성기.exe` - 실행 파일

### Mac
- `dist/유튜브_영상_생성기` - 실행 파일
- `dist/유튜브_영상_생성기.app` - 앱 번들

## 주의사항

### Windows
- Windows 환경에서만 Windows용 .exe 파일을 빌드할 수 있습니다.
- 빌드된 .exe 파일은 Windows 10 이상에서 실행 가능합니다.

### Mac
- macOS 환경에서만 Mac용 앱을 빌드할 수 있습니다.
- 처음 실행 시 보안 경고가 나타날 수 있습니다:
  1. "시스템 환경설정 > 보안 및 개인 정보 보호" 열기
  2. "확인 없이 열기" 클릭
  3. 또는 터미널에서 실행: `xattr -cr dist/유튜브_영상_생성기.app`

## 문제 해결

### 빌드 실패 시
1. PyInstaller가 최신 버전인지 확인: `pip install --upgrade pyinstaller`
2. 모든 의존성이 설치되었는지 확인: `pip install -r requirements.txt`
3. `templates`와 `static` 폴더가 존재하는지 확인

### 실행 파일이 작동하지 않을 때
1. 콘솔 모드로 빌드하여 오류 메시지 확인:
   - Windows: `--windowed` 대신 `--console` 사용
   - Mac: `--noconsole` 대신 `--console` 사용
2. 필요한 DLL/라이브러리가 포함되었는지 확인

## 배포

빌드된 실행 파일을 다른 컴퓨터에서 사용하려면:
- Windows: .exe 파일만 배포하면 됩니다.
- Mac: .app 번들을 배포하거나, 실행 파일과 함께 필요한 라이브러리를 포함해야 할 수 있습니다.

