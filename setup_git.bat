@echo off
REM Git 저장소 초기화 및 원격 저장소 연결 스크립트 (Windows)

echo ==========================================
echo Git 저장소 설정 스크립트
echo ==========================================

REM 현재 디렉토리 확인
if not exist "app.py" (
    echo 오류: app.py 파일을 찾을 수 없습니다.
    echo 프로젝트 루트 디렉토리에서 실행해주세요.
    pause
    exit /b 1
)

REM Git 초기화
if not exist ".git" (
    echo Git 저장소 초기화 중...
    git init
) else (
    echo Git 저장소가 이미 초기화되어 있습니다.
)

REM 파일 추가
echo 변경사항 추가 중...
git add .

REM 커밋
echo 커밋 생성 중...
git commit -m "최신 버전: 자막 배경 박스 수정 포함" || echo 변경사항이 없거나 이미 커밋되어 있습니다.

REM 브랜치 이름 설정
git branch -M main

echo.
echo ==========================================
echo 로컬 Git 저장소 설정 완료!
echo ==========================================
echo.
echo 다음 단계:
echo 1. GitHub/GitLab에서 새 저장소를 생성하세요
echo 2. 아래 명령어를 실행하세요:
echo.
echo    git remote add origin https://github.com/사용자명/저장소명.git
echo    git push -u origin main
echo.
pause

