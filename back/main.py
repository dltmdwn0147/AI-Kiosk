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
import re
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
latest_face_data = None      # (가장 가까운 사람의) 얼굴 랜드마크
latest_emotion_data = None   # (가장 가까운 사람의) 감정 데이터
latest_face_count = 0        # [NEW] 현재 화면에 잡힌 총 인원 수

face_data_lock = threading.Lock()
emotion_data_lock = threading.Lock()

kiosk_socket = None
socket_lock = threading.Lock()

is_running = True # 전체 프로그램 종료 제어 플래그
recent_action_cache = {"text": "", "ts": 0.0}


# ------------------------------------------------------------------
# [Function 1] 메뉴 데이터 로드
# ------------------------------------------------------------------
def load_menu_data(file_path):
    if not os.path.exists(file_path):
        return "정보 없음"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.dumps(json.load(f), ensure_ascii=False, indent=2)
    except Exception:
        return "정보 없음"

def _load_menu_items(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = []
        for m in data:
            if not isinstance(m, dict):
                continue
            items.append({
                "name": m.get("menu_name", ""),
                "temperature": str(m.get("menu_temperature", "")).upper(),
                "price": m.get("menu_price", ""),
            })
        return items
    except Exception:
        return []


# ------------------------------------------------------------------
# [Helper Function] 랜드마크 -> Bounding Box 변환
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
    
    if width < 30 or height < 30: return None # 너무 작은 얼굴 무시
    return (min_x, min_y, width, height)


# ------------------------------------------------------------------
# [Function 2] 주문 및 안면 로그 저장
# ------------------------------------------------------------------
def save_order_log(menu_name, face_data, emotion_data=None):
    file_name = "order_logs.json"
    
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "menu": menu_name,
        "face_landmarks": face_data if face_data else "No Face Detected",
        "emotion": emotion_data if emotion_data else "No Emotion Detected",
        "face_count": latest_face_count # 로그에도 인원수 저장
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
# ------------------------------------------------------------------
def add_order_to_kiosk(menu_name: str):
    return send_kiosk_action("add", menu_name, 1, "")

def send_kiosk_action(action_type: str, menu_name: str, quantity: int = 1, temperature: str = ""):
    global kiosk_socket, latest_face_data, latest_emotion_data
    if not action_type:
        return {"result": "주문 실패", "menu": menu_name}
    payload = {
        "type": action_type,
        "menu_name": menu_name,
        "quantity": int(quantity or 1),
        "temperature": temperature or "",
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    now = time.time()
    if payload_json == recent_action_cache["text"] and now - recent_action_cache["ts"] < 2.0:
        return {"result": "중복 차단", "menu": menu_name}

    current_face = None
    with face_data_lock:
        if latest_face_data:
            current_face = latest_face_data[:]
    current_emotion = None
    with emotion_data_lock:
        if latest_emotion_data:
            current_emotion = latest_emotion_data.copy() if isinstance(latest_emotion_data, dict) else latest_emotion_data
    threading.Thread(target=save_order_log, args=(menu_name, current_face, current_emotion)).start()

    msg = "주문 실패 (연결 없음)"
    with socket_lock:
        if kiosk_socket:
            try:
                print(f"   └─ 📡 키오스크 전송(raw): {payload_json}")
                kiosk_socket.sendall((payload_json + "\n").encode("utf-8"))
                msg = "주문 성공"
            except Exception as e:
                print(f"   └─ ❌ 전송 실패: {e}")
    recent_action_cache["text"] = payload_json
    recent_action_cache["ts"] = now
    return {"result": msg, "menu": menu_name}

def get_menu_price(menu_name: str, temperature: str = ""):
    items = _load_menu_items(os.path.join(current_dir, "mega_coffee_menu.json"))
    if not items:
        return {"menu": menu_name, "price": None}
    def norm(text):
        return re.sub(r"\\s+", "", str(text))
    temp = str(temperature).upper().strip()
    best = None
    for item in items:
        name = item.get("name", "")
        if not name:
            continue
        if norm(name) in norm(menu_name) or norm(menu_name) in norm(name):
            if temp and item.get("temperature") and item.get("temperature") != temp:
                continue
            best = item
            break
    if not best:
        return {"menu": menu_name, "price": None}
    return {"menu": best.get("name", menu_name), "price": best.get("price"), "temperature": best.get("temperature", "")}


# ------------------------------------------------------------------
# [Function 4] 카메라 메인 루프 (Main Thread)
# 특징: 다중 얼굴 인식(20명), 가장 가까운 얼굴 우선 분석, 인원수 카운팅
# ------------------------------------------------------------------
def camera_loop_main():
    global latest_face_data, latest_emotion_data, latest_face_count, is_running
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    print("📸 [Camera] 카메라 시작 (Main Thread - Max 20 faces)")

    # Mediapipe 초기화
    face_landmarker = None
    try:
        model_file = os.path.join(current_dir, 'face_landmarker.task')
        if os.path.exists(model_file):
            base_options = python.BaseOptions(model_asset_path=model_file)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=20, # [설정] 최대 20명까지 감지
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
            h, w, _ = image.shape
            
            # Mediapipe 얼굴 감지
            if face_landmarker:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                detection_result = face_landmarker.detect(mp_image)

                detected_faces_count = 0
                closest_face_landmarks = None
                max_face_area = 0
                closest_face_bbox = None

                # 1. 얼굴이 감지되었는지 확인
                if detection_result.face_landmarks:
                    detected_faces_count = len(detection_result.face_landmarks)
                    
                    # [NEW] 인원수 전역 변수 업데이트
                    with face_data_lock:
                        latest_face_count = detected_faces_count
                    
                    # 2. 여러 얼굴 중 가장 가까운(큰) 얼굴 찾기
                    for landmarks in detection_result.face_landmarks:
                        temp_data = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in landmarks]
                        bbox = get_face_bbox_from_landmarks(temp_data, w, h)
                        
                        if bbox:
                            bx, by, bw, bh = bbox
                            area = bw * bh
                            
                            # 현재까지 가장 큰 얼굴보다 더 크면 메인 유저로 갱신
                            if area > max_face_area:
                                max_face_area = area
                                closest_face_landmarks = landmarks
                                closest_face_bbox = bbox
                            
                            # (시각화) 모든 얼굴에 얇은 회색 박스
                            cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (100, 100, 100), 1)

                    # 3. 메인 유저(가장 가까운 사람) 처리
                    if closest_face_landmarks:
                        # (1) 랜드마크 저장
                        data = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in closest_face_landmarks]
                        with face_data_lock: latest_face_data = data
                        
                        # (시각화) 메인 유저 강조 (초록색 박스)
                        cx, cy, cw, ch = closest_face_bbox
                        cv2.rectangle(image, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)

                        # (2) DeepFace 감정 분석 (15프레임마다 메인 유저만 분석)
                        if DEEPFACE_AVAILABLE and frame_count % 15 == 0:
                            face_crop = image_rgb[cy:cy+ch, cx:cx+cw]
                            try:
                                res = DeepFace.analyze(img_path=face_crop, actions=['emotion'], 
                                                     detector_backend='skip', enforce_detection=False, silent=True)
                                if isinstance(res, list): res = res[0]
                                emotion = res['dominant_emotion']
                                
                                with emotion_data_lock: 
                                    latest_emotion_data = {'dominant_emotion': emotion}
                            except: pass
                else:
                    # 얼굴 없음
                    with face_data_lock: 
                        latest_face_data = None
                        latest_face_count = 0
                    detected_faces_count = 0

                # 4. 화면 UI 표시
                # 인원수
                cv2.putText(image, f"Faces: {detected_faces_count}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # 감정 상태
                if latest_emotion_data and detected_faces_count > 0:
                     emo_text = latest_emotion_data['dominant_emotion']
                     cv2.putText(image, f"Emotion: {emo_text}", (20, 80), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 5. 화면 출력 (macOS 필수)
            cv2.imshow('Kiosk Eyes (Server)', image)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                is_running = False
                break
            if cv2.getWindowProperty('Kiosk Eyes (Server)', cv2.WND_PROP_VISIBLE) < 1:
                is_running = False
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("📸 카메라 루프 종료")


# ------------------------------------------------------------------
# [Function 5] 오디오 루프 (Background Thread)
# 특징: 감정 및 인원수 정보를 Gemini에게 시스템 메시지로 실시간 전달
# ------------------------------------------------------------------
def audio_thread_entry():
    asyncio.run(audio_loop_async())

async def audio_loop_async():
    global is_running, latest_emotion_data, latest_face_count
    
    menu_data = load_menu_data(os.path.join(current_dir, 'mega_coffee_menu.json'))
    
    # [Point 1] 시스템 프롬프트: 인원수와 감정에 따른 행동 지침 정의
    system_instruction = f"""
    당신은 메가커피 키오스크 음성 AI입니다.
    손님의 말을 듣고 즉시 대답하세요.
    
    [실시간 상황 인지 시스템]
    시스템이 주기적으로 아래 정보를 보내줍니다:
    1. User Emotion: (예: Happy, Sad, Angry) - 가장 가까운 손님의 표정
    2. Face Count: (예: 1, 3, 5) - 현재 화면에 보이는 사람 수
    
    [행동 지침]
    1. 인원수 파악:
       - 2명 이상: "두 분이서 오셨네요!", "여러분이서 드시기 좋은 메뉴 추천해드릴까요?"
       - 5명 이상: "손님이 정말 많네요! 매장이 북적북적해요."
       - 1명: 개인 맞춤형 대화 집중.
       
    2. 감정 대응:
       - Sad/Neutral: 따뜻하고 친절하게, 달콤한 메뉴 추천.
       - Angry: 빠르고 신속하게, 불필요한 말 줄이기.
       - Happy: 밝은 텐션으로 인기 메뉴 추천.

    보유 메뉴: {menu_data}
    주문 시 아래 도구를 사용하세요:
    - add_order_to_kiosk(menu_name): 기본 추가
    - kiosk_action(type, menu_name, quantity, temperature): 추가/삭제/증감
    - get_menu_price(menu_name, temperature): 가격 질의
    """

    try:
        p = pyaudio.PyAudio()
        mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        spk_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
        
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        print("🎧 [Audio] Gemini 연결 시도 중...")

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part(text=system_instruction)]),
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name="add_order_to_kiosk",
                    description="주문 추가 도구",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={"menu_name": types.Schema(type="STRING")},
                        required=["menu_name"]
                    )
                ),
                types.FunctionDeclaration(
                    name="kiosk_action",
                    description="장바구니 제어 도구(add/remove/inc/dec)",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "type": types.Schema(type="STRING"),
                            "menu_name": types.Schema(type="STRING"),
                            "quantity": types.Schema(type="NUMBER"),
                            "temperature": types.Schema(type="STRING"),
                        },
                        required=["type", "menu_name"]
                    )
                ),
                types.FunctionDeclaration(
                    name="get_menu_price",
                    description="메뉴 가격 조회 도구",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "menu_name": types.Schema(type="STRING"),
                            "temperature": types.Schema(type="STRING"),
                        },
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

            # Task C: 상황 정보(감정, 인원수) 실시간 주입
            async def send_context_update():
                last_sent_emotion = "Neutral"
                last_sent_count = -1
                
                while is_running:
                    await asyncio.sleep(2.5) # 너무 자주 보내지 않도록 조절
                    
                    # 현재 데이터 스냅샷
                    current_emo = "Neutral"
                    current_cnt = 0
                    
                    with emotion_data_lock:
                        if latest_emotion_data and 'dominant_emotion' in latest_emotion_data:
                            current_emo = latest_emotion_data['dominant_emotion']
                    with face_data_lock:
                        current_cnt = latest_face_count
                    
                    # 변화 감지 시 전송 (인원수 변화 or 감정 변화)
                    if (current_emo != last_sent_emotion and current_emo != "Neutral") or (current_cnt != last_sent_count):
                        
                        print(f"🌊 [Context Update] 감정: {current_emo}, 인원: {current_cnt}명")
                        
                        msg = f"[System] User Emotion: {current_emo}, Face Count: {current_cnt}"
                        await session.send(input=msg, end_of_turn=True)
                        
                        last_sent_emotion = current_emo
                        last_sent_count = current_cnt

            # 3가지 태스크 동시 실행
            await asyncio.gather(send_audio(), receive_audio(), send_context_update())

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
# ------------------------------------------------------------------
def socket_server_thread():
    global kiosk_socket
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 9999))
        server.listen(1)
        print("💡 [Socket] 포트 9999 대기 중...")
        
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
    # 1. 오디오 스레드 시작
    audio_t = threading.Thread(target=audio_thread_entry, daemon=True)
    audio_t.start()

    # 2. 소켓 스레드 시작
    socket_t = threading.Thread(target=socket_server_thread, daemon=True)
    socket_t.start()

    # 3. 카메라 메인 루프 실행 (반드시 메인 스레드)
    try:
        camera_loop_main()
    except KeyboardInterrupt:
        is_running = False
        print("\n🚫 프로그램 강제 종료")
