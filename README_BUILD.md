# PC 앱 빌드 가이드

## 빠른 시작

### 1. 빌드 환경 준비

```bash
# 가상환경 활성화 (이미 있다면 생략)
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 앱 빌드

```bash
python build_app.py
```

### 3. 빌드 결과 확인

- **Windows**: `dist/유튜브_영상_생성기.exe`
- **macOS**: `dist/유튜브_영상_생성기.app`
- **Linux**: `dist/유튜브_영상_생성기`

## 빌드 옵션 커스터마이징

`build_app.py` 파일을 수정하여 다음을 변경할 수 있습니다:

- **앱 이름**: `--name` 옵션
- **아이콘**: `--icon=icon.ico` 추가
- **콘솔 창**: `--windowed` (Windows) 또는 `--noconsole` (macOS/Linux)

## 배포 전 체크리스트

- [ ] API 키가 코드에 하드코딩되어 있지 않은지 확인
- [ ] FFmpeg 설치 가이드 준비
- [ ] 사용자 매뉴얼 작성
- [ ] 테스트 실행 파일로 테스트
- [ ] 바이러스 검사 (Windows의 경우)

## 문제 해결

### 빌드 실패
- 모든 의존성이 설치되었는지 확인: `pip list`
- PyInstaller 버전 확인: `pyinstaller --version`

### 실행 파일이 너무 큼
- `--onefile` 대신 `--onedir` 사용 (폴더로 배포)
- 불필요한 패키지 제외

### 실행 시 오류
- 콘솔 창을 표시하여 오류 확인: `--windowed` 제거
- 로그 파일 확인

