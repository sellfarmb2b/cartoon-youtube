#!/bin/bash
# 다른 컴퓨터에서 최신 버전 가져오기 스크립트

echo "=========================================="
echo "다른 컴퓨터에서 최신 버전 가져오기"
echo "=========================================="

# 원격 저장소 URL 확인
if [ -z "$1" ]; then
    echo "사용법: $0 <원격_저장소_URL>"
    echo "예시: $0 https://github.com/사용자명/저장소명.git"
    exit 1
fi

REMOTE_URL=$1

# 저장소 클론
echo "📥 저장소 클론 중..."
git clone "$REMOTE_URL" youtube-video-maker
cd youtube-video-maker || exit 1

# 가상 환경 생성
echo "🐍 가상 환경 생성 중..."
python3 -m venv venv

# 가상 환경 활성화
echo "✅ 가상 환경 활성화 중..."
source venv/bin/activate

# 의존성 설치
echo "📦 의존성 설치 중..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ 설치 완료!"
echo "=========================================="
echo ""
echo "다음 명령어로 서버를 실행하세요:"
echo "  cd youtube-video-maker"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "최신 버전 업데이트:"
echo "  git pull origin main"

