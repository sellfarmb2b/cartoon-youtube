#!/bin/bash
# Git 저장소 초기화 및 원격 저장소 연결 스크립트

echo "=========================================="
echo "Git 저장소 설정 스크립트"
echo "=========================================="

# 현재 디렉토리 확인
if [ ! -f "app.py" ]; then
    echo "❌ 오류: app.py 파일을 찾을 수 없습니다."
    echo "프로젝트 루트 디렉토리에서 실행해주세요."
    exit 1
fi

# Git 초기화 (이미 초기화되어 있으면 스킵)
if [ ! -d ".git" ]; then
    echo "📦 Git 저장소 초기화 중..."
    git init
else
    echo "✅ Git 저장소가 이미 초기화되어 있습니다."
fi

# 파일 추가
echo "📝 변경사항 추가 중..."
git add .

# 커밋
echo "💾 커밋 생성 중..."
git commit -m "최신 버전: 자막 배경 박스 수정 포함" || echo "⚠️ 변경사항이 없거나 이미 커밋되어 있습니다."

# 브랜치 이름 설정
git branch -M main

echo ""
echo "=========================================="
echo "✅ 로컬 Git 저장소 설정 완료!"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "1. GitHub/GitLab에서 새 저장소를 생성하세요"
echo "2. 아래 명령어를 실행하세요:"
echo ""
echo "   git remote add origin https://github.com/사용자명/저장소명.git"
echo "   git push -u origin main"
echo ""
echo "또는 원격 저장소 URL을 입력하시겠습니까? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "원격 저장소 URL을 입력하세요:"
    read -r remote_url
    git remote add origin "$remote_url" 2>/dev/null || git remote set-url origin "$remote_url"
    echo "원격 저장소에 푸시하시겠습니까? (y/n)"
    read -r push_response
    if [[ "$push_response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        git push -u origin main
    fi
fi

