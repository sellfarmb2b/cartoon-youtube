#!/bin/bash
# 프로젝트 실행 스크립트 (Mac/Linux)

echo "=========================================="
echo "유튜브 영상 생성기 실행"
echo "=========================================="

# 현재 디렉토리 확인
if [ ! -f "app.py" ]; then
    echo "❌ 오류: app.py 파일을 찾을 수 없습니다."
    echo "프로젝트 루트 디렉토리에서 실행해주세요."
    exit 1
fi

# 가상 환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "📦 가상 환경이 없습니다. 생성 중..."
    python3 -m venv venv
    echo "✅ 가상 환경 생성 완료"
fi

# 가상 환경 활성화
echo "🔧 가상 환경 활성화 중..."
source venv/bin/activate

# 의존성 확인
if [ ! -f "venv/bin/flask" ]; then
    echo "📦 의존성 설치 중..."
    pip install -r requirements.txt
    echo "✅ 의존성 설치 완료"
fi

# 환경 변수 파일 확인
if [ ! -f ".env.local" ] && [ ! -f ".env" ]; then
    echo "⚠️  경고: .env.local 또는 .env 파일이 없습니다."
    echo "환경 변수를 설정해주세요. (.env.example 참고)"
    echo ""
    echo "계속하시겠습니까? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "실행을 취소했습니다."
        exit 1
    fi
fi

# 서버 실행
echo ""
echo "🚀 서버 시작 중..."
echo "=========================================="
python app.py

