# ☕ AI Smart Kiosk Server with Gemini 2.0
> **Multimodal AI 기반의 음성 대화형 키오스크 서버 프로젝트** > 실시간 음성 대화, 안면 인식 데이터 수집, 그리고 키오스크 제어를 통합한 지능형 시스템입니다.

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
* **Reference**: 이 프로젝트는 [guaba98/mega_kiosk](https://github.com/guaba98/mega_kiosk) 리포지토리를 참조/활용하여 구현되었습니다.

---

## 🚀 Key Features (핵심 기능)

### 1. Gemini Native Audio Streaming (실시간 음성 대화)
* 기존의 `STT(변환) -> LLM(생성) -> TTS(변환)` 과정을 거치는 느린 방식이 아닙니다.
* **Gemini Live API**를 사용하여 오디오 스트림(PCM)을 직접 AI에게 전송하고 수신합니다.
* 사용자가 말하는 도중에 끼어들어도 인식하는 **Full-Duplex(양방향 통신)** 대화를 지원합니다.

### 2. Intelligent Function Calling (주문 의도 파악)
* 단순한 챗봇이 아닙니다. 사용자의 발화에서 "주문 의도"가 감지되면 AI가 스스로 판단하여 **`add_order_to_kiosk`** 도구를 실행합니다.
* *"아이스 아메리카노 시원한 걸로 하나 줘"* → **`{"menu": "아이스 아메리카노"}`** 추출 및 주문 실행.

### 3. Real-time Face Data Logging (안면 데이터 수집)
* **Mediapipe Face Mesh**를 별도 스레드에서 구동하여 메인 로직의 지연 없이 얼굴을 추적합니다.
* 주문이 확정(장바구니 담기)되는 **정확한 순간(Timestamp)**의 안면 랜드마크(x, y, z 좌표)를 캡처합니다.
* 수집된 데이터는 `JSON` 형태로 로그 파일에 저장되어 추후 **"사용자 맞춤형 메뉴 추천"**이나 **"통계 분석"**에 활용됩니다.

### 4. Async & Thread-Safe Architecture (비동기 및 스레드 안전성)
* **Asyncio:** 오디오 입출력의 끊김 없는 처리를 위해 비동기 루프를 사용합니다.
* **Threading:** 무거운 영상 처리 로직을 별도 스레드로 분리하여 오디오 지연을 방지합니다.
* **Locking:** 전역 변수(안면 데이터) 접근 시 `Threading.Lock`을 사용하여 데이터 무결성을 보장합니다.

---

## 📊 System Architecture (시스템 구조)

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