#!/bin/bash
# Google Generative AI 패키지 포함하여 재빌드하는 스크립트

echo "=========================================="
echo "YouTubeMaker 재빌드 (google-generativeai 포함)"
echo "=========================================="

# 1. 필수 패키지 설치 확인
echo ""
echo "[1단계] 필수 패키지 설치 확인 중..."
pip3 install google-generativeai pyinstaller

# 2. 패키지 설치 확인
echo ""
echo "[2단계] 패키지 설치 확인 중..."
python3 -c "import google.generativeai; print('✅ google-generativeai 설치됨:', google.generativeai.__version__)" || {
    echo "❌ google-generativeai 패키지 설치 실패"
    exit 1
}

# 3. 기존 빌드 정리
echo ""
echo "[3단계] 기존 빌드 정리 중..."
if [ -d "build" ]; then
    rm -rf build
    echo "  ✅ build 폴더 삭제"
fi
if [ -d "dist" ]; then
    rm -rf dist
    echo "  ✅ dist 폴더 삭제"
fi

# 4. 빌드 실행
echo ""
echo "[4단계] YouTubeMaker 빌드 시작..."
python3 src/build_mac.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 빌드 완료!"
    echo "=========================================="
    echo "실행 파일 위치: dist/YouTubeMaker/YouTubeMaker"
    echo ""
    echo "💡 다음 단계:"
    echo "  1. dist/YouTubeMaker/YouTubeMaker 실행 파일을 실행하세요"
    echo "  2. 또는 dist/Launcher 파일을 실행하세요"
else
    echo ""
    echo "❌ 빌드 실패"
    exit 1
fi

