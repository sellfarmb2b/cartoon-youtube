@echo off
REM 프로젝트 실행 스크립트 (Windows)

echo ==========================================
echo 유튜브 영상 생성기 실행
echo ==========================================

REM 현재 디렉토리 확인
if not exist "app.py" (
    echo 오류: app.py 파일을 찾을 수 없습니다.
    echo 프로젝트 루트 디렉토리에서 실행해주세요.
    pause
    exit /b 1
)

REM 가상 환경 확인 및 생성
if not exist "venv" (
    echo 가상 환경이 없습니다. 생성 중...
    python -m venv venv
    echo 가상 환경 생성 완료
)

REM 가상 환경 활성화
echo 가상 환경 활성화 중...
call venv\Scripts\activate.bat

REM 의존성 확인
if not exist "venv\Scripts\flask.exe" (
    echo 의존성 설치 중...
    pip install -r requirements.txt
    echo 의존성 설치 완료
)

REM 환경 변수 파일 확인
if not exist ".env.local" if not exist ".env" (
    echo 경고: .env.local 또는 .env 파일이 없습니다.
    echo 환경 변수를 설정해주세요. (.env.example 참고)
    echo.
    pause
)

REM 서버 실행
echo.
echo 서버 시작 중...
echo ==========================================
python app.py
pause

