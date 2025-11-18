# 프로젝트 동기화 가이드

## 방법 1: Git을 사용한 동기화 (권장)

### 현재 컴퓨터에서 설정하기

1. **GitHub/GitLab에 새 저장소 생성**
   - GitHub.com에 로그인
   - "New repository" 클릭
   - 저장소 이름 입력 (예: `youtube-video-maker`)
   - Private 또는 Public 선택
   - "Create repository" 클릭

2. **로컬 저장소에 원격 저장소 연결**
   ```bash
   cd "/Users/rohbyoungkon/Desktop/유튜브 프로젝트"
   git add .
   git commit -m "Initial commit: 최신 버전"
   git branch -M main
   git remote add origin https://github.com/사용자명/저장소명.git
   git push -u origin main
   ```

### 다른 컴퓨터에서 최신 버전 가져오기

1. **프로젝트 클론**
   ```bash
   git clone https://github.com/사용자명/저장소명.git
   cd 저장소명
   ```

2. **가상 환경 설정**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   # 또는
   venv\Scripts\activate  # Windows
   ```

3. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

4. **최신 버전 업데이트 (이후 변경사항 반영 시)**
   ```bash
   git pull origin main
   ```

---

## 방법 2: 파일 직접 복사 (간단한 방법)

### 현재 컴퓨터에서

1. **중요 파일만 압축**
   - `app.py`
   - `templates/` 폴더
   - `requirements.txt`
   - `build_mac.py`, `build_windows.py`
   - `BUILD_GUIDE.md`
   - `.gitignore`
   - `docs/` 폴더 (선택사항)

2. **압축 파일 생성**
   - 위 파일들을 ZIP으로 압축

### 다른 컴퓨터에서

1. **압축 파일 압축 해제**
2. **가상 환경 설정 및 의존성 설치**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 방법 3: 클라우드 동기화 (Dropbox, Google Drive 등)

1. **프로젝트 폴더를 클라우드 폴더에 배치**
2. **다른 컴퓨터에서 클라우드 폴더 동기화**
3. **동기화된 폴더에서 직접 실행**

⚠️ **주의사항**
- `venv/` 폴더는 동기화하지 마세요 (각 컴퓨터에서 새로 생성)
- `static/generated_assets/` 폴더는 동기화하지 마세요 (생성된 파일들)
- API 키가 포함된 파일은 보안에 주의하세요

