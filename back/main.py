"""
================================================================================
[Project: AI Kiosk Server Structure Map (Native Audio Version)]

1. 시스템 개요
    - Main Thread (Audio Loop): Gemini Live API를 통한 실시간 음성 스트리밍 및 주문 처리
    - Sub Thread (Camera): Mediapipe를 이용한 실시간 안면 인식 및 데이터 갱신

2. 데이터 흐름도 (Data Flow)

    [run_server()] ----------------------------------------+
        |                                                  |
        | 1. 메뉴 로드                                       | 2. 스레드 시작
        v                                                  v
    [load_menu_data()]                            [camera_thread_func()]
    (JSON 파일 읽기)                                (백그라운드 실행)
        |                                                 |
        +-> [Gemini System Prompt]                        |-- (Write) --> [Global Var: latest_face_data]
                    |                                     |                    (실시간 얼굴 좌표 공유)
                    v                                     |
        [Audio Stream Loop]                               |
        (마이크 입력 <-> 스피커 출력)                           |
                    |                                     |
            (주문 요청 발생 시)                               |
                    v                                     |
            [add_order_to_kiosk()] <------- (Read) -------+
                    |
                    |-- 1. 로그 저장 --> [save_order_log()] --> order_logs.json
                    |
                    +-- 2. 명령 전송 --> [Socket] --> Unity/Kiosk Client
================================================================================
"""
import socket
import asyncio
import os
import sys
import json
import threading
import cv2
import mediapipe as mp
import pyaudio
from datetime import datetime
from dotenv import load_dotenv

from google import genai
from google.genai import types

# 1. 환경 변수 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    sys.exit("❌ 오류: .env 파일에 GEMINI_API_KEY가 없습니다.")

# 2. 오디오 설정 (Gemini Live API 표준)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 512

# --- 전역 변수 설정 ---
latest_face_data = None
face_data_lock = threading.Lock()
kiosk_socket = None


# ------------------------------------------------------------------
# [Function 1] 메뉴 데이터 로드
# 관계: run_server(audio_loop)에서 초기화 단계에 딱 한 번 호출됨.
# 역할: JSON 파일을 읽어 거대 언어 모델(LLM)이 이해할 수 있는 문자열(String)로 변환.
# ------------------------------------------------------------------
def load_menu_data(file_path):
    if not os.path.exists(file_path):
        return "정보 없음"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.dumps(json.load(f), ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ 메뉴 파일 로드 오류: {e}")
        return "정보 없음"


# ------------------------------------------------------------------
# [Function 2] 주문 및 안면 로그 저장
# 관계: add_order_to_kiosk() 내부에서 호출됨 (Helper Function).
# 역할: 
#   1. 주문한 메뉴 이름과 시간 기록
#   2. !!중요!! 주문 시점의 안면 데이터(latest_face_data)를 스냅샷처럼 찍어서 함께 저장.
#   3. 기존 데이터 유무 상관없이 새로운 로그를 파일 끝에 추가(Append).
# ------------------------------------------------------------------
def save_order_log(menu_name, face_data):
    """
    주문 내역과 안면 데이터를 JSON 파일에 누적 저장합니다.
    """
    file_name = "order_logs.json"
    
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "menu": menu_name,
        "face_landmarks": face_data if face_data else "No Face Detected"
    }

    logs = []
    if os.path.exists(file_name):
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except:
            logs = []

    logs.append(log_entry)

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    
    print(f"💾 [System] 주문 및 안면 데이터가 '{file_name}'에 저장되었습니다.")


# ------------------------------------------------------------------
# [Function 3] AI 도구 (Tool) - 주문 실행기
# 관계: Gemini(AI)가 사용자의 주문 의도를 파악했을 때 '스스로' 호출함.
# 역할: 메인 로직과 하드웨어/데이터 로직을 연결하는 다리(Bridge).
# 흐름:
#   1. Global 변수(latest_face_data)에서 현재 손님 얼굴 좌표 가져오기.
#   2. save_order_log()를 호출해 장부에 기록.
#   3. Global 변수(kiosk_socket)를 통해 외부 키오스크(Unity 등)로 주문 명령 전송.
# ------------------------------------------------------------------
def add_order_to_kiosk(menu_name: str):
    global kiosk_socket, latest_face_data
    print(f"\n⚡ [Tool Triggered] AI가 '{menu_name}' 주문을 처리합니다.")
    
    # 1. 안면 데이터 캡처
    current_face = None
    with face_data_lock:
        if latest_face_data:
            current_face = latest_face_data[:] 
    
    # 2. 로그 저장
    save_order_log(menu_name, current_face)
    
    # 3. 키오스크 전송
    msg = "주문 실패 (연결 없음)"
    if kiosk_socket:
        try:
            print(f"   └─ 📡 키오스크 전송: {menu_name}")
            kiosk_socket.sendall(menu_name.encode('utf-8'))
            msg = "주문 성공"
        except Exception as e:
            print(f"   └─ ❌ 전송 실패: {e}")
            pass
    
    # Live API에서는 결과값을 딕셔너리(JSON) 형태로 반환해야 함
    return {"result": msg, "menu": menu_name}


# ------------------------------------------------------------------
# [Function 4] 카메라 및 안면 인식 스레드
# 관계: Main Logic에 의해 별도의 스레드(Daemon Thread)로 시작됨.
# 역할: 메인 대화 루프를 방해하지 않고 '독립적'으로 카메라를 계속 감시함.
# 동작:
#   - OpenCV로 웹캠 영상을 읽음.
#   - Mediapipe로 얼굴 랜드마크(468개 점) 추출.
#   - 추출된 데이터를 전역 변수 'latest_face_data'에 계속 최신화(Update)함.
# ------------------------------------------------------------------
def camera_thread_func():
    global latest_face_data
    
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    print("📸 [Camera] 안면 인식 모듈 시작됨...")

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while True:
            success, image = cap.read()
            if not success: continue

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            image.flags.writeable = True

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    data = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in face_landmarks.landmark]
                    with face_data_lock: latest_face_data = data
            else:
                with face_data_lock: latest_face_data = None

            cv2.imshow('Kiosk Eyes (Server)', image)
            if cv2.waitKey(5) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()
    print("📸 [Camera] 카메라 스레드 종료")


# ------------------------------------------------------------------
# [Function 5] 메인 서버 로직 (Audio Controller)
# 관계: 프로그램의 핵심 비동기 루프 (Main Loop).
# 역할: Gemini Live API와 연결하여 듣기/말하기를 동시에 수행.
# 절차:
#   1. load_menu_data() -> 메뉴판 준비
#   2. PyAudio Stream Open -> 마이크/스피커 준비
#   3. Gemini Live Connect -> AI 연결
#   4. Task Group -> Send Audio(말하기) / Receive Audio(듣기) 동시 실행
# ------------------------------------------------------------------
async def audio_loop():
    # [1] 메뉴 데이터 로드
    menu_data = load_menu_data('mega_coffee_menu.json')
    
    # [2] 시스템 프롬프트 설정
    system_instruction = f"""
    당신은 메가커피 키오스크 음성 AI입니다.
    손님의 말을 듣고 즉시 대답하세요.
    보유 메뉴: {menu_data}
    주문시 반드시 'add_order_to_kiosk' 도구를 호출하세요.
    """

    # [3] 오디오 장치 설정
    p = pyaudio.PyAudio()
    mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    spk_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    print("\n🎧 [System] Gemini Live 연결 시도 중...")

    # [4] Gemini Live 세션 시작
    async with client.aio.live.connect(
        model="gemini-2.0-flash-exp",
        config=types.LiveConnectConfig(
            response_modalities=["AUDIO"], 
            system_instruction=types.Content(parts=[types.Part(text=system_instruction)]),
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name="add_order_to_kiosk",
                    description="손님이 메뉴를 주문하면 실행합니다.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={"menu_name": types.Schema(type="STRING")},
                        required=["menu_name"]
                    )
                )
            ])]
        )
    ) as session:
        print("✅ [Connected] 대화 시작! 말씀하세요.")

        # --- Task A: 마이크 입력 전송 ---
        async def send_audio():
            while True:
                try:
                    data = await asyncio.to_thread(mic_stream.read, CHUNK, exception_on_overflow=False)
                    await session.send(input={"data": data, "mime_type": "audio/pcm"}, end_of_turn=False)
                except Exception as e:
                    print(f"Mic Error: {e}")
                    break

        # --- Task B: AI 응답 수신 및 도구 처리 ---
        async def receive_audio():
            while True:
                async for response in session.receive():
                    # B-1. 오디오 데이터 재생
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data:
                                await asyncio.to_thread(spk_stream.write, part.inline_data.data)
                    
                    # B-2. 도구 호출(Function Calling) 처리
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            if fc.name == "add_order_to_kiosk":
                                result = add_order_to_kiosk(fc.args["menu_name"])
                                # 결과를 AI에게 반환
                                await session.send(input=types.LiveClientToolResponse(
                                    function_responses=[types.FunctionResponse(
                                        name=fc.name, id=fc.id, response=result
                                    )]
                                ))

        # 두 작업을 동시에 실행 (Infinite Loop)
        await asyncio.gather(send_audio(), receive_audio())

    # 정리
    mic_stream.stop_stream()
    mic_stream.close()
    spk_stream.stop_stream()
    spk_stream.close()
    p.terminate()

# --- 실행부 ---
if __name__ == '__main__':
    # 1. 카메라 스레드 시작
    t = threading.Thread(target=camera_thread_func, daemon=True)
    t.start()
    
    # 2. 소켓 서버 시작 (간단화)
    try:
        kiosk_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        kiosk_server.bind(('127.0.0.1', 9999))
        kiosk_server.listen(1)
        kiosk_server.setblocking(False) # 비동기 루프 방해 방지
        print("💡 [Socket] 키오스크 접속 대기 중 (연결 시 주문 전송 가능)...")
        # 실제 환경에선 accept 루프를 별도 스레드나 비동기로 처리해야 함.
        # 여기선 로직 단순화를 위해 리스닝 상태만 유지.
    except Exception as e:
        print(f"Socket Error: {e}")

    # 3. 메인 오디오 루프 시작
    try:
        asyncio.run(audio_loop())
    except KeyboardInterrupt:
        print("\n🚫 프로그램 종료")