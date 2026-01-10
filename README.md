# ☕ AI Smart Kiosk Server with Gemini 2.0

> **Multimodal AI 기반의 음성 대화형 키오스크 서버 프로젝트**
> 실시간 음성 대화, 안면 인식 데이터 수집, 그리고 키오스크 제어를 통합한 지능형 시스템입니다.

> 🏆 **Note:** 본 프로젝트는 **경상국립대학교 RISE-AI 2025 생성형AI 경진대회** 참여를 위해 개발되었습니다.

---

## 📖 Project Overview (프로젝트 개요)

이 프로젝트는 기존 터치 방식의 키오스크를 넘어, **사람과 대화하듯 주문할 수 있는 "Voice-First" 인터페이스**를 구현하는 것을 목표로 합니다.

Google의 최신 **Gemini 2.0 Multimodal Live API**를 활용하여 별도의 STT/TTS 모듈 없이 초저지연(Low Latency) 음성 대화를 처리하며, **Mediapipe**를 통해 주문자의 안면 데이터를 실시간으로 분석하여 주문 성향 데이터베이스를 구축합니다.

---

## 🛠️ Tech Stack (기술 스택)

### 🖥️ Backend (AI Core Server)
핵심 로직이 구동되는 서버 파트입니다.

| 구분 | 기술 / 라이브러리 | 사용 목적 |
| :--- | :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3.10+-blue) | 전체 서버 로직 및 비동기 처리 구현 |
| **LLM / AI** | **Google Gemini 2.0 Flash (Exp)** | 자연어 이해, 음성 입출력(Native Audio), Function Calling (도구 사용) |
| **Vision AI** | **Mediapipe (Face Mesh)** | 468개의 안면 랜드마크(Landmarks) 실시간 추출 및 분석 |
| **Vision Tool** | **OpenCV** | 웹캠 영상 스트림 캡처 및 전처리 |
| **Network** | **Socket (TCP/IP)** | 프론트엔드(키오스크 클라이언트)와의 실시간 명령 송수신 |
| **Concurrency** | **Asyncio & Threading** | 오디오 스트리밍(Async)과 영상 처리(Thread)의 병렬 처리 |

### 🖥️ Frontend (Kiosk Client)
*(※ 본 레포지토리는 서버 코드를 포함하며, 클라이언트는 소켓으로 연결됩니다.)*
* **Role:** 사용자에게 메뉴 UI를 보여주고, 서버로부터 받은 주문 명령을 시각적으로 반영.
* **Communication:** TCP 소켓을 통해 서버와 연결 (`127.0.0.1:9999`).
* **Action:** `add_order` 신호 수신 시 장바구니 UI 업데이트.
* **Reference:** 이 프로젝트의 프론트엔드 구조는 [guaba98/mega_kiosk](https://github.com/guaba98/mega_kiosk) 리포지토리를 참조 및 활용하여 구현되었습니다.

---

## 📊 System Architecture (시스템 구조)

```
graph TD
    User((User/Guest)) -->|Voice| Mic[Microphone]
    User -->|Face| Cam[Webcam]
    
    subgraph "AI Kiosk Server (Python)"
        direction TB
        Mic -->|Audio Stream| AsyncLoop[Asyncio Audio Loop]
        AsyncLoop <-->|Live API (WebSocket)| Gemini[Google Gemini 2.0]
        
        Cam -->|Video Frame| CVThread[Camera Thread]
        CVThread -->|Face Landmarks| GlobalData[(Shared Memory)]
        
        Gemini -->|Tool Call| OrderFunc[add_order_to_kiosk]
        OrderFunc -->|Read| GlobalData
        OrderFunc -->|Save| LogFile[order_logs.json]
    end
    
    OrderFunc -->|Socket Signal| KioskClient[Unity/Web Kiosk Client]
    KioskClient -->|Display| Screen[Kiosk Screen]
```

---

## 🚀 Development Environment Setup (개발 환경 설정)

### 📋 Prerequisites (필수 사항)

#### 1. Python 버전
- **Python 3.10 이상** 필요
- 권장: Python 3.10 ~ 3.12

```bash
python --version  # Python 3.10+ 확인
```

#### 2. 하드웨어 요구사항
- **마이크**: 음성 입력을 위한 마이크 (내장 또는 외장)
- **웹캠**: 안면 인식을 위한 카메라
- **스피커**: AI 음성 응답 출력용

#### 3. API 키 발급
- **Google Gemini API Key** 필요
  - [Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급
  - Gemini 2.0 Flash (Exp) 모델 사용 권한 필요

---

### 📦 Installation (설치 방법)

#### Step 1: 저장소 클론
```bash
git clone https://github.com/dltmdwn0147/AI-Kiosk.git
cd AI-Kiosk
```

#### Step 2: 가상환경 생성 및 활성화 (권장)
```bash
# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### Step 3: 의존성 라이브러리 설치
```bash
# requirements.txt의 모든 패키지 설치
pip install -r requirements.txt
```

**참고**: 일부 시스템에서 `pyaudio` 설치 시 문제가 발생할 수 있습니다.
- **macOS**: `brew install portaudio` 실행 후 `pip install pyaudio`
- **Linux (Ubuntu/Debian)**: `sudo apt-get install portaudio19-dev python3-pyaudio`
- **Windows**: pip로 직접 설치 가능 (`pip install pyaudio`)

#### Step 4: 환경 변수 설정
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 Gemini API 키를 입력합니다:

```bash
# .env 파일 생성
touch .env
```

`.env` 파일 내용:
```env
GEMINI_API_KEY=your_api_key_here
```

**⚠️ 주의**: `.env` 파일은 `.gitignore`에 포함되어 있어 Git에 업로드되지 않습니다. API 키를 절대 공개 저장소에 올리지 마세요!

---

### 🎯 Running the Project (프로젝트 실행)

#### Backend 서버 실행
```bash
cd back
python main.py
```

**실행 시 확인 사항:**
- ✅ Gemini Live API 연결 성공 메시지 확인
- ✅ 웹캠이 정상적으로 작동하는지 확인 (안면 인식 창이 열려야 함)
- ✅ 마이크 권한이 허용되었는지 확인

#### Frontend 키오스크 클라이언트 실행
별도 터미널에서:
```bash
cd front
python mega_kiosk_ver1.py
```

**실행 순서:**
1. **먼저 Backend 서버를 실행** (`back/main.py`)
2. **그 다음 Frontend 클라이언트를 실행** (`front/mega_kiosk_ver1.py`)
3. Frontend는 자동으로 `127.0.0.1:9999` 포트로 Backend에 연결됩니다

---

### 🔧 Troubleshooting (문제 해결)

#### 1. PyAudio 설치 오류
```bash
# macOS
brew install portaudio
pip install pyaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev
pip install pyaudio
```

#### 2. OpenCV (cv2) 웹캠 접근 오류
- 웹캠이 다른 프로그램에서 사용 중인지 확인
- 시스템 권한 설정에서 카메라 접근 권한 허용

#### 3. Gemini API 연결 실패
- `.env` 파일에 올바른 API 키가 입력되었는지 확인
- API 키에 Gemini 2.0 Flash (Exp) 모델 사용 권한이 있는지 확인
- 인터넷 연결 상태 확인

#### 4. 소켓 연결 실패
- Backend 서버가 먼저 실행되었는지 확인
- 포트 `9999`가 다른 프로그램에서 사용 중이지 않은지 확인

---

### 📝 Project Structure (프로젝트 구조)

```
AI-Kiosk/
├── back/                      # Backend 서버 코드
│   ├── main.py               # 메인 서버 로직 (Gemini Live API, 카메라 스레드)
│   ├── mega_coffee_menu.json # 메뉴 데이터 (JSON)
│   ├── excel_to_JSON.py      # 엑셀 → JSON 변환 스크립트
│
├── front/                    # Frontend 키오스크 클라이언트
│   ├── mega_kiosk_ver1.py   # 메인 UI 실행 파일 (PyQt5)
│   ├── shopping_cart.py     # 장바구니 모듈
│   ├── manager_page.py      # 관리자 페이지
│   ├── DATA/                # 데이터베이스 및 CSV 파일
│   │   ├── data.db          # SQLite 데이터베이스
│   │   └── *.csv            # 메뉴 및 주문 데이터
│   └── UI/                  # PyQt5 UI 파일 (.ui)
│
├── requirements.txt          # Python 의존성 패키지 목록
├── .env                      # 환경 변수 (API 키 등) - .gitignore에 포함됨
├── .gitignore               # Git 제외 파일 목록
└── README.md                # 프로젝트 문서 (본 파일)
```

---

### 📚 Additional Resources (추가 자료)

- [Google Gemini API 문서](https://ai.google.dev/docs)
- [Mediapipe Face Mesh 문서](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [PyQt5 공식 문서](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [Python Asyncio 가이드](https://docs.python.org/3/library/asyncio.html)