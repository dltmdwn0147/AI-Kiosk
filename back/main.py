"""
================================================================================
[Project: AI Kiosk Server - macOS Compatible Version]

1. 시스템 개요 (macOS GUI 정책 반영)
    - Main Thread (Camera): OpenCV 창 출력을 위해 카메라는 반드시 메인 스레드에서 실행
    - Sub Thread (Audio): Gemini Live API 음성 대화는 백그라운드 스레드에서 비동기 실행
    - Sub Thread (Socket): 키오스크 연결 대기 및 통신

2. 데이터 흐름도 (Data Flow)

    [Execution Start]
        |
        +-- (Start Thread) --> [Function 5: Audio Loop] <==> Gemini Live API
        |                       (음성 인식 및 답변 생성)
        |
        +-- (Start Thread) --> [Function 6: Socket Server]
        |                       (키오스크 연결 대기)
        |
        v (Main Thread Blocking)
    [Function 4: Camera Loop]
        |
        |-- [Step 1] OpenCV Read -> Mediapipe (얼굴 위치 추적)
        |-- [Step 2] DeepFace (감정 분석 - 15프레임마다)
        |-- [Step 3] cv2.imshow (화면 출력 - macOS 호환)
        |
        +-- (Write) --> [Global Var: latest_face_data / latest_emotion_data]

    [Event: 주문 발생 시]
        [Audio Loop] -> Gemini Tool Call -> [Function 3: add_order_to_kiosk()]
                                                |
                                                +-- 1. [Function 2: save_order_log()] (JSON 저장)
                                                +-- 2. Socket Send (Unity로 전송)
================================================================================
"""

import socket
import asyncio
import os
import sys
import json
import threading
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import pyaudio
from datetime import datetime
from dotenv import load_dotenv

from google import genai
from google.genai import types

# DeepFace 감정 분석 (선택적 import)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace가 설치되지 않았습니다. 감정 분석 기능이 비활성화됩니다.")

# 1. 환경 변수 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
env_path = os.path.join(root_dir, ".env")

if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    sys.exit("❌ 오류: .env 파일에 GEMINI_API_KEY가 없습니다.")

# 2. 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 512

# --- 전역 변수 설정 ---
latest_face_data = None      # 얼굴 랜드마크
latest_emotion_data = None   # 감정 데이터
face_data_lock = threading.Lock()
emotion_data_lock = threading.Lock()

kiosk_socket = None
socket_lock = threading.Lock()

is_running = True # 전체 프로그램 종료 제어 플래그


# ------------------------------------------------------------------
# [Function 1] 메뉴 데이터 로드
# 역할: JSON 파일을 읽어 거대 언어 모델(LLM)이 이해할 수 있는 문자열로 변환.
# ------------------------------------------------------------------
def load_menu_data(file_path):
    if not os.path.exists(file_path):
        return "정보 없음"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.dumps(json.load(f), ensure_ascii=False, indent=2)
    except Exception:
        return "정보 없음"


# ------------------------------------------------------------------
# [Helper Function] 랜드마크 -> Bounding Box 변환
# 역할: Mediapipe 좌표를 이용해 얼굴 영역을 잘라내기 위한 좌표 계산.
# ------------------------------------------------------------------
def get_face_bbox_from_landmarks(face_landmarks, image_width, image_height):
    if not face_landmarks: return None
    
    x_coords = [lm['x'] * image_width for lm in face_landmarks]
    y_coords = [lm['y'] * image_height for lm in face_landmarks]
    
    # 여유 공간 확보 (Padding)
    min_x = max(0, int(min(x_coords) * 0.9))
    max_x = min(image_width, int(max(x_coords) * 1.1))
    min_y = max(0, int(min(y_coords) * 0.9))
    max_y = min(image_height, int(max(y_coords) * 1.1))
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width < 50 or height < 50: return None # 너무 작은 얼굴 무시
    return (min_x, min_y, width, height)


# ------------------------------------------------------------------
# [Function 2] 주문 및 안면 로그 저장
# 관계: add_order_to_kiosk() 내부에서 호출됨.
# 역할: 주문 시점의 메뉴, 시간, 얼굴 좌표, 감정 상태를 JSON에 기록.
# ------------------------------------------------------------------
def save_order_log(menu_name, face_data, emotion_data=None):
    file_name = "order_logs.json"
    
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "menu": menu_name,
        "face_landmarks": face_data if face_data else "No Face Detected",
        "emotion": emotion_data if emotion_data else "No Emotion Detected"
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
    
    print(f"💾 [System] 로그 저장 완료: {menu_name}")


# ------------------------------------------------------------------
# [Function 3] AI 도구 (Tool) - 주문 실행기
# 관계: Gemini(AI)가 사용자의 주문 의도를 파악했을 때 '스스로' 호출함.
# 역할: 현재 얼굴/감정 데이터를 캡처하고 로그를 저장한 뒤, 키오스크로 신호 전송.
# ------------------------------------------------------------------
def add_order_to_kiosk(menu_name: str):
    global kiosk_socket, latest_face_data, latest_emotion_data
    print(f"\n⚡ [Tool Triggered] AI가 '{menu_name}' 주문을 처리합니다.")
    
    # 1. 데이터 스냅샷 (Thread-Safe)
    current_face = None
    with face_data_lock:
        if latest_face_data: current_face = latest_face_data[:]
    
    current_emotion = None
    with emotion_data_lock:
        if latest_emotion_data:
            current_emotion = latest_emotion_data.copy() if isinstance(latest_emotion_data, dict) else latest_emotion_data

    # 2. 로그 저장을 별도 스레드로 처리 (응답 속도 향상)
    threading.Thread(target=save_order_log, args=(menu_name, current_face, current_emotion)).start()
    
    # 3. 키오스크(Unity) 전송
    msg = "주문 실패 (연결 없음)"
    with socket_lock:
        if kiosk_socket:
            try:
                print(f"   └─ 📡 키오스크 전송: {menu_name}")
                kiosk_socket.sendall(menu_name.encode('utf-8'))
                msg = "주문 성공"
            except Exception as e:
                print(f"   └─ ❌ 전송 실패: {e}")
                pass
    
    return {"result": msg, "menu": menu_name}


# ------------------------------------------------------------------
# [Function 4] 카메라 메인 루프 (Main Thread)
# 중요: macOS에서는 GUI(imshow)가 반드시 Main Thread에 있어야 함.
# 역할: 웹캠 읽기 -> Mediapipe 감지 -> DeepFace 감정 분석 -> 화면 출력
# ------------------------------------------------------------------
def camera_loop_main():
    global latest_face_data, latest_emotion_data, is_running
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    print("📸 [Camera] 카메라 시작 (Running on Main Thread)")

    # Mediapipe 초기화
    face_landmarker = None
    try:
        model_file = os.path.join(current_dir, 'face_landmarker.task')
        if os.path.exists(model_file):
            base_options = python.BaseOptions(model_asset_path=model_file)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE
            )
            face_landmarker = vision.FaceLandmarker.create_from_options(options)
            print("✅ Mediapipe FaceLandmarker 초기화 성공")
        else:
            print("⚠️ 모델 파일 없음 (안면 인식 기능 제한됨)")
    except Exception as e:
        print(f"⚠️ Mediapipe 초기화 오류: {e}")

    frame_count = 0
    
    try:
        while is_running:
            success, image = cap.read()
            if not success: continue

            frame_count += 1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1. Mediapipe 얼굴 감지
            if face_landmarker:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                detection_result = face_landmarker.detect(mp_image)

                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    # 전역 변수 업데이트
                    data = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in landmarks]
                    with face_data_lock: latest_face_data = data
                    
                    # 2. DeepFace 감정 분석 (속도를 위해 15프레임마다 실행)
                    if DEEPFACE_AVAILABLE and frame_count % 15 == 0:
                        h, w, _ = image.shape
                        bbox = get_face_bbox_from_landmarks(data, w, h)
                        if bbox:
                            x, y, bw, bh = bbox
                            # 얼굴 부분만 잘라냄 (Crop)
                            face_crop = image_rgb[y:y+bh, x:x+bw]
                            
                            try:
                                # 얼굴 탐지 스킵(skip)으로 고속 분석
                                res = DeepFace.analyze(img_path=face_crop, actions=['emotion'], 
                                                    detector_backend='skip', enforce_detection=False, silent=True)
                                if isinstance(res, list): res = res[0]
                                emotion = res['dominant_emotion']
                                
                                with emotion_data_lock: 
                                    latest_emotion_data = {'dominant_emotion': emotion}
                            except: pass
                    
                    # 화면에 감정 텍스트 표시
                    if latest_emotion_data:
                        cv2.putText(image, f"Emotion: {latest_emotion_data['dominant_emotion']}", (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    with face_data_lock: latest_face_data = None
            
            # 3. 화면 출력 (반드시 메인 스레드여야 함)
            cv2.imshow('Kiosk Eyes', image)
            
            # ESC 키 종료
            if cv2.waitKey(1) & 0xFF == 27:
                print("🔚 종료 요청 (ESC Key)")
                is_running = False
                break
            
            # 창 닫기 버튼 감지
            if cv2.getWindowProperty('Kiosk Eyes', cv2.WND_PROP_VISIBLE) < 1:
                is_running = False
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("📸 카메라 종료")


# ------------------------------------------------------------------
# [Function 5] 오디오 루프 (Background Thread)
# 역할: 별도 스레드에서 비동기(Asyncio)로 Gemini Live API와 통신.
# ------------------------------------------------------------------
def audio_thread_entry():
    """스레드 타겟용 래퍼 함수"""
    asyncio.run(audio_loop_async())

# [수정된 부분] Function 5: 오디오 루프
async def audio_loop_async():
    global is_running, latest_emotion_data # 전역 변수 참조
    
    menu_data = load_menu_data(os.path.join(current_dir, 'mega_coffee_menu.json'))
    
    # [Point 1] 시스템 프롬프트 강화: 감정 정보가 들어오면 반응하도록 지시
    system_instruction = f"""
    당신은 메가커피 키오스크 음성 AI입니다.
    손님의 말을 듣고 즉시 대답하세요.
    
    [중요] 시스템이 주기적으로 '[System] User Emotion: ...' 형태로 손님의 표정 정보를 보내줍니다.
    - User Emotion: Sad (우울) -> 따뜻한 위로와 함께 달콤한 메뉴를 추천하세요.
    - User Emotion: Angry (화남) -> 정중하게 사과하고 빠르고 간결하게 응대하세요.
    - User Emotion: Happy (행복) -> 밝은 목소리로 신메뉴를 권유해보세요.
    
    보유 메뉴: {menu_data}
    주문시 반드시 'add_order_to_kiosk' 도구를 호출하세요.
    """

    try:
        p = pyaudio.PyAudio()
        mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        spk_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
        
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        print("🎧 [Audio] Gemini 연결 시도 중... (Background Thread)")

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"], 
            system_instruction=types.Content(parts=[types.Part(text=system_instruction)]),
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name="add_order_to_kiosk",
                    description="주문 처리 도구",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={"menu_name": types.Schema(type="STRING")},
                        required=["menu_name"]
                    )
                )
            ])]
        )

        async with client.aio.live.connect(model="gemini-2.0-flash-exp", config=config) as session:
            print("✅ [Audio] 연결 성공! 대화 가능.")

            # Task A: 마이크 -> Gemini
            async def send_audio():
                while is_running:
                    try:
                        data = await asyncio.to_thread(mic_stream.read, CHUNK, exception_on_overflow=False)
                        await session.send(input={"data": data, "mime_type": "audio/pcm"}, end_of_turn=False)
                    except: break

            # Task B: Gemini -> 스피커 & 툴
            async def receive_audio():
                while is_running:
                    async for response in session.receive():
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data:
                                    await asyncio.to_thread(spk_stream.write, part.inline_data.data)
                        
                        if response.tool_call:
                            for fc in response.tool_call.function_calls:
                                if fc.name == "add_order_to_kiosk":
                                    menu = fc.args.get("menu_name", "")
                                    res = add_order_to_kiosk(menu)
                                    await session.send(input=types.LiveClientToolResponse(
                                        function_responses=[types.FunctionResponse(name=fc.name, id=fc.id, response=res)]
                                    ))

            # [Point 2] Task C: 감정 정보 실시간 전송 (NEW!)
            async def send_emotion_context():
                last_sent_emotion = "Neutral"
                while is_running:
                    await asyncio.sleep(2.0) # 2초마다 체크 (너무 자주 보내면 대화가 끊김)
                    
                    current_emo = "Neutral"
                    with emotion_data_lock:
                        if latest_emotion_data and 'dominant_emotion' in latest_emotion_data:
                            current_emo = latest_emotion_data['dominant_emotion']
                    
                    # 감정이 변했거나, 특정 강한 감정(화남, 슬픔)이 지속될 때 AI에게 알림
                    # (여기선 감정이 바뀔 때만 보내도록 설정)
                    if current_emo != last_sent_emotion and current_emo != "Neutral":
                        print(f"🌊 [Context] 감정 변화 감지: {last_sent_emotion} -> {current_emo}")
                        
                        # Gemini에게 텍스트로 상황 전달
                        msg = f"[System] Current User Emotion: {current_emo}"
                        await session.send(input=msg, end_of_turn=True) # end_of_turn=True로 하면 AI가 즉시 반응함
                        
                        last_sent_emotion = current_emo

            # 세 가지 태스크를 동시에 실행
            await asyncio.gather(send_audio(), receive_audio(), send_emotion_context())

    except Exception as e:
        print(f"❌ [Audio] 오류 발생: {e}")
    finally:
        try:
            mic_stream.stop_stream(); mic_stream.close()
            spk_stream.stop_stream(); spk_stream.close()
            p.terminate()
        except: pass


# ------------------------------------------------------------------
# [Function 6] 소켓 서버 (Background Thread)
# 역할: 키오스크 클라이언트(Frontend)의 접속을 대기하고 연결 유지.
# ------------------------------------------------------------------
def socket_server_thread():
    global kiosk_socket
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 9999))
        server.listen(1)
        print("💡 [Socket] 포트 9999 대기 중... (Background Thread)")
        
        while is_running:
            server.settimeout(1.0)
            try:
                conn, addr = server.accept()
                print(f"✅ [Socket] 클라이언트 연결됨: {addr}")
                with socket_lock: kiosk_socket = conn
            except socket.timeout:
                continue
            except:
                break
    except Exception as e:
        print(f"❌ [Socket] 오류: {e}")


# --- 실행부 (Entry Point) ---
if __name__ == '__main__':
    # 1. 오디오 스레드 시작 (Daemon)
    audio_t = threading.Thread(target=audio_thread_entry, daemon=True)
    audio_t.start()

    # 2. 소켓 스레드 시작 (Daemon)
    socket_t = threading.Thread(target=socket_server_thread, daemon=True)
    socket_t.start()

    # 3. 카메라 메인 루프 실행 (Main Thread Blocking)
    # macOS에서는 반드시 여기서 카메라 창을 띄워야 함
    try:
        camera_loop_main()
    except KeyboardInterrupt:
        is_running = False
        print("\n🚫 프로그램 강제 종료")