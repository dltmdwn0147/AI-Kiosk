"""
================================================================================
[Project: AI Kiosk Server - Kiosk Mode (Stable)]
* 목표: 커피 주문 시 거절문(I'm sorry...) 없이 정상적으로 함수 호출/응답
* 변경: "안전 우회" / "거절 금지" / "This is safe" 같은 메타 지시 제거
* 변경: 주문이면 tool 호출, 애매하면 짧게 확인 질문, 무관 요청은 주문으로 유도
* 변경: 감정 컨텍스트 주입에서 [System] 메타 토큰 제거
* 옵션: output_modalities를 ["audio"]로 명시(텍스트 응답 생성 최소화)
* ✅ 추가 수정: 키오스크 전송 메시지에 '\n' 추가(메시지 프레이밍 안정화)
================================================================================
"""

import socket
import asyncio
import os
import sys
import json
import threading
import time
import queue
import urllib.request
import urllib.error
import io
import wave
import re
import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from pandas.core.missing import F
import pyaudio
from datetime import datetime
from dotenv import load_dotenv

import websockets

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace가 설치되지 않았습니다.")

# 1. 환경 변수
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
env_path = os.path.join(root_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("❌ 오류: OPENAI_API_KEY 없음")

# 2. 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024

# --- 전역 변수 ---
latest_face_data = None
latest_emotion_data = None
latest_face_count = 0
face_data_lock = threading.Lock()
emotion_data_lock = threading.Lock()
kiosk_socket = None
socket_lock = threading.Lock()
is_running = True
recent_order_cache = {"text": "", "ts": 0.0}
recent_action_cache = {"text": "", "ts": 0.0}

# --- 헬퍼 함수들 ---
def load_menu_data(file_path):
    if not os.path.exists(file_path):
        return "정보 없음"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f), ensure_ascii=False, indent=2)
    except:
        return "정보 없음"

def get_face_bbox_from_landmarks(face_landmarks, image_width, image_height):
    if not face_landmarks:
        return None
    x_coords = [lm["x"] * image_width for lm in face_landmarks]
    y_coords = [lm["y"] * image_height for lm in face_landmarks]
    min_x = max(0, int(min(x_coords) * 0.9))
    max_x = min(image_width, int(max(x_coords) * 1.1))
    min_y = max(0, int(min(y_coords) * 0.9))
    max_y = min(image_height, int(max(y_coords) * 1.1))
    width = max_x - min_x
    height = max_y - min_y
    if width < 30 or height < 30:
        return None
    return (min_x, min_y, width, height)

def save_order_log(menu_name, face_data, emotion_data=None):
    file_name = "order_logs.json"
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "menu": menu_name,
        "face_landmarks": "Detected",
        "emotion": emotion_data if emotion_data else "None",
        "face_count": latest_face_count,
    }
    try:
        logs = []
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append(log_entry)
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except:
        pass

def add_order_to_kiosk(menu_name: str):
    """
    ✅ 핵심 수정:
    - TCP recv는 메시지 경계를 보장하지 않아서, 키오스크 쪽에서 줄 단위로 끊어 읽을 수 있도록 '\n'을 붙여 전송
    """
    global kiosk_socket, latest_face_data, latest_emotion_data

    current_face = None
    with face_data_lock:
        if latest_face_data:
            current_face = latest_face_data[:]

    current_emotion = None
    with emotion_data_lock:
        if latest_emotion_data:
            current_emotion = latest_emotion_data.copy()

    threading.Thread(
        target=save_order_log, args=(menu_name, current_face, current_emotion)
    ).start()

    return send_kiosk_action("add", menu_name, 1)


def _normalize_for_cache(text: str) -> str:
    return "".join(str(text).split())


def send_order_to_kiosk_safe(menu_text: str):
    """중복 전송을 막고 소켓 전송 실패 시에도 로그를 남기는 헬퍼"""
    if not menu_text:
        return "주문 실패: 빈 메뉴명"
    if str(menu_text).strip().startswith("참고:"):
        return "주문 생략: 참고 문구"

    norm = _normalize_for_cache(menu_text)
    now = time.time()
    if norm and recent_order_cache["text"] == norm and now - recent_order_cache["ts"] < 2.0:
        return f"중복 차단: {menu_text}"

    res = add_order_to_kiosk(menu_text)
    recent_order_cache["text"] = norm
    recent_order_cache["ts"] = now
    return res


def send_kiosk_action(action_type: str, menu_name: str, quantity: int = 1, temperature: str = ""):
    if not action_type:
        return "주문 실패: 타입 없음"
    if action_type in ("add", "inc", "dec", "remove") and not menu_name:
        return "주문 실패: 메뉴 없음"

    payload_dict = {
        "type": action_type,
        "menu_name": menu_name,
        "quantity": int(quantity or 1),
        "temperature": temperature or "",
    }
    payload_json = json.dumps(payload_dict, ensure_ascii=False)

    now = time.time()
    if payload_json == recent_action_cache["text"] and now - recent_action_cache["ts"] < 2.0:
        return f"중복 차단: {payload_json}"

    msg = "주문 실패"
    with socket_lock:
        if kiosk_socket:
            try:
                payload = (payload_json + "\n").encode("utf-8")
                print(f"   └─ 📡 키오스크 전송(raw): {payload!r}")
                kiosk_socket.sendall(payload)
                msg = "주문 성공"
            except Exception as e:
                print(f"   └─ ❌ 키오스크 전송 실패: {e}")

    recent_action_cache["text"] = payload_json
    recent_action_cache["ts"] = now
    return f"{msg}: {payload_json}"


def _is_valid_transcript(text: str) -> bool:
    if not text:
        return False
    cleaned = str(text).strip()
    return cleaned and cleaned.lower() != "none"


def _extract_transcripts_from_item(item: dict):
    transcripts = []
    for content in item.get("content", []) or []:
        if isinstance(content, dict):
            for key in ("transcript", "text"):
                value = content.get(key)
                if _is_valid_transcript(value):
                    transcripts.append(str(value).strip())
            input_audio_tx = content.get("input_audio_transcription", {})
            if isinstance(input_audio_tx, dict):
                value = input_audio_tx.get("text")
                if _is_valid_transcript(value):
                    transcripts.append(str(value).strip())
    return transcripts


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        print(f"❌ [HTTPError] {e} body={body}")
        raise


def _post_bytes(url: str, payload: dict, headers: dict, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        print(f"❌ [HTTPError] {e} body={body}")
        raise


def _extract_response_text(resp_json: dict) -> str:
    texts = []
    for item in resp_json.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in ("output_text", "text"):
                text = content.get("text", "")
                if text:
                    texts.append(text)
    return "\n".join(texts).strip()


def _extract_quantity_korean(text: str) -> int:
    if not text:
        return 0
    cleaned = str(text).replace(" ", "")
    num_map = {
        "한": 1,
        "하나": 1,
        "둘": 2,
        "두": 2,
        "셋": 3,
        "세": 3,
        "넷": 4,
        "네": 4,
        "다섯": 5,
        "여섯": 6,
        "일곱": 7,
        "여덟": 8,
        "아홉": 9,
        "열": 10,
    }
    for key, val in num_map.items():
        if key + "잔" in cleaned or key + "개" in cleaned:
            return val
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return 0
    return 0


def _extract_temperature(text: str) -> str:
    if not text:
        return ""
    lowered = str(text).lower()
    if "아이스" in text or "ice" in lowered or "차가" in text:
        return "ICE"
    if "핫" in text or "hot" in lowered or "뜨거" in text or "따뜻" in text:
        return "HOT"
    return ""


def _play_pcm_audio(pcm_bytes: bytes, rate: int = RATE):
    if not pcm_bytes:
        return
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=rate, output=True)
        stream.write(pcm_bytes)
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()


def _play_wav_audio(wav_bytes: bytes):
    if not wav_bytes:
        return
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    p = pyaudio.PyAudio()
    try:
        fmt = p.get_format_from_width(width)
        stream = p.open(format=fmt, channels=channels, rate=rate, output=True)
        stream.write(frames)
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()


def _build_intent_prompt(user_text: str, menu_json: str, emotion: str) -> str:
    return f"""
[역할]
너는 '메가커피' 키오스크의 주문 처리 AI다.
고객의 발화를 분석하여 정확한 JSON 데이터를 출력해야 한다.

[입력 데이터]
- 보유 메뉴: {menu_json}
- 고객 감정: {emotion}
- 고객 발화: "{user_text}"

[처리 규칙]
1. **메뉴 매칭 (가장 중요)**
   - 고객이 메뉴명을 정확히 말하지 않아도, 문맥과 발음을 통해 보유 메뉴 중 가장 유사한 것으로 매핑한다.
   - 예: "아아" -> "아이스 아메리카노", "따아" -> "따뜻한 아메리카노", "라떼" -> "카페라떼", "스무디" -> (가장 인기 있는 스무디 또는 되물기)
   - 메뉴판에 없는 메뉴를 주문하면 "죄송하지만 해당 메뉴는 없습니다."라고 reply에 적고 actions는 비운다.

2. **온도(Temperature) 처리**
   - 고객 발화에 "아이스/ICE/차가운" 등이 있으면 temperature="ICE".
   - "핫/HOT/따뜻한/뜨거운" 등이 있으면 temperature="HOT".
   - 명시되지 않으면 빈 문자열("")로 둔다.

3. **의도(Intent) 분류**
   - order: 메뉴 주문, 수량 변경, 장바구니 담기
   - query_price: 가격 질문 ("이거 얼마야?")
   - query_info: 메뉴 정보 질문 ("이거 매워?", "카페인 들어있어?")
   - other: 인사, 잡담, 키오스크 사용법 질문 등

4. **Actions 배열 생성**
   - type: "add" (추가), "remove" (삭제), "inc" (수량증가), "dec" (수량감소)
   - 주문 의도가 확실할 때만 생성한다. 애매하면 actions를 비우고 reply로 되묻는다.
   - 수량(quantity) 언급이 없으면 기본값은 1이다.

5. **Reply(응답) 작성 가이드**
   - 무조건 **한국어(존댓말)**로 작성한다.
   - 불필요한 서론(JSON 설명 등)을 빼고, 실제 점원이 손님에게 하듯 자연스럽게 말한다.
   - 감정 데이터({emotion})를 참고하여 톤을 조절한다. (Sad -> 따뜻하고 위로하는 말투, Happy -> 밝고 경쾌한 말투)
   - 예: "아이스 아메리카노 한 잔 담아드릴게요.", "어떤 메뉴를 찾으시나요?"

[출력 요구사항]
- 오직 JSON Schema에 맞는 데이터만 출력한다.
- JSON 외의 잡담이나 마크다운(```json)을 포함하지 않는다.
"""

# ------------------------------------------------------------------
# [Function 4] 카메라 루프
# ------------------------------------------------------------------
def camera_loop_main():
    global latest_face_data, latest_emotion_data, latest_face_count, is_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    print("📸 [Camera] 카메라 시작")

    face_landmarker = None
    try:
        model_file = os.path.join(current_dir, "face_landmarker.task")
        if os.path.exists(model_file):
            base_options = python.BaseOptions(model_asset_path=model_file)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=20,
                min_face_detection_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE,
            )
            face_landmarker = vision.FaceLandmarker.create_from_options(options)
    except:
        pass

    frame_count = 0
    try:
        while is_running:
            success, image = cap.read()
            if not success:
                continue
            frame_count += 1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

            if face_landmarker:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                detection_result = face_landmarker.detect(mp_image)
                detected_faces_count = 0
                closest_face_landmarks = None
                max_face_area = 0
                closest_face_bbox = None

                if detection_result.face_landmarks:
                    detected_faces_count = len(detection_result.face_landmarks)
                    with face_data_lock:
                        latest_face_count = detected_faces_count

                    for landmarks in detection_result.face_landmarks:
                        temp_data = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
                        bbox = get_face_bbox_from_landmarks(temp_data, w, h)
                        if bbox:
                            bx, by, bw, bh = bbox
                            area = bw * bh
                            if area > max_face_area:
                                max_face_area = area
                                closest_face_landmarks = landmarks
                                closest_face_bbox = bbox
                            cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (100, 100, 100), 1)

                    if closest_face_landmarks:
                        data = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in closest_face_landmarks]
                        with face_data_lock:
                            latest_face_data = data
                        cx, cy, cw, ch = closest_face_bbox
                        cv2.rectangle(image, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)

                        if DEEPFACE_AVAILABLE and frame_count % 15 == 0:
                            face_crop = image_rgb[cy : cy + ch, cx : cx + cw]
                            try:
                                res = DeepFace.analyze(
                                    img_path=face_crop,
                                    actions=["emotion"],
                                    detector_backend="skip",
                                    enforce_detection=False,
                                    silent=True,
                                )
                                if isinstance(res, list):
                                    res = res[0]
                                with emotion_data_lock:
                                    latest_emotion_data = {"dominant_emotion": res["dominant_emotion"]}
                            except:
                                pass
                else:
                    with face_data_lock:
                        latest_face_data = None
                        latest_face_count = 0

                cv2.putText(
                    image,
                    f"Faces: {latest_face_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )
                if latest_emotion_data:
                    cv2.putText(
                        image,
                        f"Emotion: {latest_emotion_data.get('dominant_emotion')}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("Kiosk Eyes (Server)", image)
            if cv2.waitKey(1) & 0xFF == 27:
                is_running = False
                break
            if cv2.getWindowProperty("Kiosk Eyes (Server)", cv2.WND_PROP_VISIBLE) < 1:
                is_running = False
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ------------------------------------------------------------------
# [Function 5] 오디오 루프 (OpenAI Realtime API - Kiosk Mode)
# ------------------------------------------------------------------
def audio_thread_entry():
    asyncio.run(audio_loop_async())

async def audio_loop_async():
    global is_running

    menu_data = load_menu_data(os.path.join(current_dir, "mega_coffee_menu.json"))
    try:
        menu_list = json.loads(menu_data)
        menu_items = []
        for m in menu_list:
            if not isinstance(m, dict):
                continue
            name = m.get("menu_name", "")
            temp = str(m.get("menu_temperature", "")).upper()
            price = m.get("menu_price", "")
            menu_items.append({"name": name, "temperature": temp, "price": price})
        menu_names = [m["name"] for m in menu_items if m.get("name")]
        menu_price_map = {(m["name"], m["temperature"]): m["price"] for m in menu_items}
    except Exception:
        menu_items = []
        menu_names = []
        menu_price_map = {}
    ENABLE_EMOTION_CONTEXT = False
    stt_queue = queue.Queue()
    worker_stop = threading.Event()

    system_instruction = f"""
[Role]
You are the hearing interface for a Mega Coffee Kiosk in Korea.
Your primary job is to accurately transcribe user speech into Korean text, specifically focusing on coffee menu items and numbers.

[Transcription Context]
- The user is ordering coffee/beverages.
- Expect words like: "아메리카노", "라떼", "스무디", "에이드", "샷추가", "휘핑", "테이크아웃".
- Expect abbreviations: "아아" (Ice Americano), "따아" (Hot Americano).
- Expect numbers: "한 잔", "두 개", "하나", "둘".

[Instructions]
1. Ignore background noise. Focus only on the user's voice ordering.
2. Even if the user speaks English or another language, transcribe it, but the backend system assumes Korean interaction context.
3. This is a system-level instruction for the audio model context.
"""

    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
    headers = {"Authorization": "Bearer " + api_key, "OpenAI-Beta": "realtime=v1"}
    rest_headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}

    try:
        p = pyaudio.PyAudio()
        mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("🎧 [Audio] OpenAI 연결 중...")

        async with websockets.connect(url, additional_headers=headers) as ws:
            print("✅ [Audio] 연결 성공!")

            session_update = {
                "type": "session.update",
                "session": {
                    "instructions": system_instruction,
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1", "language": "ko", "prompt": "한국어"},
                    "modalities": ["text"],
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 600,
                    },
                },
            }
            await ws.send(json.dumps(session_update))

            def nlu_worker():
                pending = {"menu_name": "", "quantity": 0, "temperature": ""}
                last_text = ""
                last_ts = 0.0
                while not worker_stop.is_set():
                    try:
                        user_text = stt_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    now = time.time()
                    if user_text == last_text and now - last_ts < 2.0:
                        continue
                    last_text = user_text
                    last_ts = now

                    cur_emo = "Neutral"
                    with emotion_data_lock:
                        if latest_emotion_data:
                            cur_emo = latest_emotion_data.get("dominant_emotion", "Neutral")

                    def _normalize_menu(text: str) -> str:
                        cleaned = re.sub(r"\\s+", "", str(text))
                        for token in ("아이스", "ICE", "핫", "HOT", "차가운", "뜨거운", "따뜻한"):
                            cleaned = cleaned.replace(token, "")
                        return cleaned

                    def _find_menu_candidates(text: str, temperature: str = "", full_text: str = ""):
                        if not text or not menu_items:
                            return []
                        norm_text = _normalize_menu(text)
                        raw_full = full_text or text
                        wants_decaf = "디카페인" in raw_full
                        matches = []
                        for item in menu_items:
                            name = item.get("name", "")
                            temp = item.get("temperature", "")
                            norm_name = _normalize_menu(name)
                            if norm_name and (norm_name in norm_text or norm_text in norm_name):
                                if not wants_decaf and "디카페인" in name:
                                    continue
                                if "아메리카노" in name and "아메리카노" in raw_full:
                                    extra = name.replace("아메리카노", "")
                                    if extra and extra not in raw_full:
                                        continue
                                if temperature and temp and temp != temperature:
                                    continue
                                matches.append(item)
                        return matches

                    def _match_menu(text: str, temperature: str = "", full_text: str = "") -> str:
                        candidates = _find_menu_candidates(text, temperature, full_text)
                        if not candidates:
                            return ""
                        candidates.sort(key=lambda x: len(x.get("name", "")), reverse=True)
                        return candidates[0].get("name", "")

                    prompt = _build_intent_prompt(user_text, menu_data, cur_emo)
                    payload = {
                        "model": "gpt-4o-mini",
                        "input": [
                            {
                                "role": "system",
                                "content": [{"type": "input_text", "text": "너는 키오스크 주문 판별기다. JSON만 출력한다."}],
                            },
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": prompt}],
                            },
                        ],
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "kiosk_intent",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "reply": {"type": "string"},
                                        "intent": {"type": "string", "enum": ["order", "query_price", "query_info", "other"]},
                                        "menu_name": {"type": "string"},
                                        "temperature": {"type": "string", "enum": ["ICE", "HOT", ""]},
                                        "quantity": {"type": "integer", "minimum": 0},
                                        "actions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {"type": "string", "enum": ["add", "remove", "inc", "dec"]},
                                                    "menu_name": {"type": "string"},
                                                    "quantity": {"type": "integer", "minimum": 1},
                                                    "temperature": {"type": "string", "enum": ["ICE", "HOT", ""]},
                                                },
                                                "required": ["type", "menu_name", "quantity", "temperature"],
                                                "additionalProperties": False,
                                            },
                                        },
                                    },
                                    "required": ["reply", "intent", "menu_name", "temperature", "quantity", "actions"],
                                    "additionalProperties": False,
                                },
                                "strict": True,
                            }
                        },
                    }
                    try:
                        resp = _post_json("https://api.openai.com/v1/responses", payload, rest_headers)
                        text = _extract_response_text(resp)
                        if not text:
                            continue
                        try:
                            intent = json.loads(text)
                        except json.JSONDecodeError:
                            print(f"⚠️ [NLU] JSON parse failed: {text[:200]}")
                            intent = {"reply": "죄송합니다. 다시 말씀해 주세요.", "actions": []}

                        reply = str(intent.get("reply", "")).strip()
                        intent_type = str(intent.get("intent", "order")).strip() or "order"
                        actions = intent.get("actions") or []
                        menu_name = str(intent.get("menu_name", "")).strip()
                        temperature = str(intent.get("temperature", "")).upper().strip()
                        quantity = int(intent.get("quantity", 0) or 0)

                        confirm_words = ("응", "네", "예", "맞아", "맞습니다", "그래", "그걸로", "그거", "그걸")
                        is_confirm = any(word in user_text for word in confirm_words)
                        effective_text = user_text
                        if "말고" in user_text:
                            effective_text = user_text.split("말고", 1)[1].strip()

                        if not temperature:
                            temperature = _extract_temperature(effective_text)

                        if not menu_name:
                            menu_name = _match_menu(effective_text, temperature, effective_text)
                        else:
                            canonical = _match_menu(menu_name, temperature, effective_text) or _match_menu(
                                effective_text, temperature, effective_text
                            )
                            if canonical:
                                menu_name = canonical

                        if quantity == 0:
                            quantity = _extract_quantity_korean(user_text)

                        if is_confirm and pending["menu_name"]:
                            menu_name = pending["menu_name"]
                            if not temperature:
                                temperature = pending["temperature"]
                        if menu_name:
                            pending["menu_name"] = menu_name
                        if quantity > 0:
                            pending["quantity"] = quantity
                        if temperature:
                            pending["temperature"] = temperature

                        print(
                            f"🧾 [슬롯 상태] menu_name='{pending['menu_name']}', quantity={pending['quantity']}, temperature='{pending['temperature']}'"
                        )

                        price_keywords = ("가격", "얼마", "원", "비용")
                        if any(word in user_text for word in price_keywords):
                            intent_type = "query_price"

                        if intent_type == "order" and is_confirm:
                            if pending["menu_name"] and pending["quantity"] > 0:
                                save_order_log(pending["menu_name"], latest_face_data, latest_emotion_data)
                                send_kiosk_action(
                                    "add", pending["menu_name"], pending["quantity"], pending["temperature"]
                                )
                                reply = f"{pending['menu_name']} {pending['quantity']}잔 준비해 드릴게요."
                                pending = {"menu_name": "", "quantity": 0, "temperature": ""}
                                actions = []

                        if intent_type in ("query_price", "query_info"):
                            matched_menu = pending["menu_name"] or menu_name
                            temperature = pending["temperature"] or temperature
                            candidates = []
                            if matched_menu:
                                candidates = _find_menu_candidates(matched_menu, temperature, effective_text)
                            if not candidates:
                                candidates = _find_menu_candidates(effective_text, temperature, effective_text)

                            if len(candidates) > 1:
                                candidates.sort(key=lambda x: len(x.get("name", "")))
                                matched_menu = candidates[0].get("name", "")
                                if not temperature:
                                    temperature = candidates[0].get("temperature", "")
                            else:
                                if not matched_menu:
                                    matched_menu = candidates[0].get("name", "") if candidates else ""
                                    if not temperature and candidates:
                                        temperature = candidates[0].get("temperature", "")

                            price = ""
                            if matched_menu:
                                if temperature:
                                    price = menu_price_map.get((matched_menu, temperature), "")
                                if not price:
                                    price = menu_price_map.get((matched_menu, ""), "")
                                if not price:
                                    # 온도 미지정일 때는 첫 가격 사용
                                    for item in candidates:
                                        if item.get("name") == matched_menu:
                                            price = item.get("price", "")
                                            break
                            if matched_menu:
                                if price:
                                    temp_suffix = f"({temperature})" if temperature else ""
                                    reply = f"{matched_menu}{temp_suffix}는 {price}원입니다. 주문 도와드릴까요?"
                                else:
                                    reply = f"{matched_menu} 가격 정보를 찾지 못했어요. 주문 도와드릴까요?"
                                pending["menu_name"] = matched_menu
                                pending["temperature"] = temperature or pending["temperature"]
                            else:
                                reply = "어떤 메뉴의 정보를 원하시나요?"
                            actions = []
                        elif actions:
                            for action in actions:
                                act_type = action.get("type")
                                act_menu = action.get("menu_name", "")
                                act_temp = str(action.get("temperature", "")).upper().strip()
                                act_qty = int(action.get("quantity", 1) or 1)
                                if act_type == "add":
                                    save_order_log(act_menu, latest_face_data, latest_emotion_data)
                                send_kiosk_action(act_type, act_menu, act_qty, act_temp)
                            if menu_name and actions[0].get("type") == "add":
                                reply = f"{menu_name} {max(1, quantity)}잔 담아드릴게요."

                        if intent_type == "order" and not actions and pending["menu_name"] and pending["quantity"] > 0:
                            save_order_log(pending["menu_name"], latest_face_data, latest_emotion_data)
                            send_kiosk_action(
                                "add", pending["menu_name"], pending["quantity"], pending["temperature"]
                            )
                            reply = f"{pending['menu_name']} {pending['quantity']}잔 담아드릴게요."
                            pending = {"menu_name": "", "quantity": 0, "temperature": ""}
                        elif intent_type == "order" and not actions and pending["menu_name"] and pending["quantity"] == 0:
                            reply = f"{pending['menu_name']} 몇 잔 준비해드릴까요?"
                        elif intent_type == "order" and not actions and pending["menu_name"] == "" and pending["quantity"] > 0:
                            reply = "어떤 음료로 준비해드릴까요?"

                        if reply:
                            print(f"🗣️ [AI Reply]: {reply}")
                            tts_payload = {
                                "model": "tts-1",
                                "voice": "shimmer",
                                "input": reply,
                                "response_format": "wav",
                            }
                            audio_bytes = _post_bytes(
                                "https://api.openai.com/v1/audio/speech", tts_payload, rest_headers
                            )
                            _play_wav_audio(audio_bytes)
                    except urllib.error.HTTPError as e:
                        print(f"❌ [NLU] HTTPError: {e}")
                    except Exception as e:
                        print(f"❌ [NLU] Error: {e}")

            worker_thread = threading.Thread(target=nlu_worker, daemon=True)
            worker_thread.start()

            # 초기 인사는 TTS로 바로 재생
            try:
                greet_payload = {
                    "model": "tts-1",
                    "voice": "shimmer",
                    "input": "안녕하세요, 메가커피입니다.",
                    "response_format": "wav",
                }
                greet_audio = _post_bytes("https://api.openai.com/v1/audio/speech", greet_payload, rest_headers)
                _play_wav_audio(greet_audio)
            except Exception as e:
                print(f"❌ [TTS] Greeting Error: {e}")

            async def send_audio():
                while is_running:
                    try:
                        data = await asyncio.to_thread(mic_stream.read, CHUNK, exception_on_overflow=False)
                        base64_audio = base64.b64encode(data).decode("utf-8")
                        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64_audio}))
                        await asyncio.sleep(0)
                    except:
                        break

            async def receive_audio():
                last_transcript = ""
                while is_running:
                    try:
                        message = await ws.recv()
                        data = json.loads(message)
                        event_type = data.get("type")
                        if event_type == "error":
                            print(f"❌ [Error] {data.get('error')}")

                        if event_type == "input_audio_buffer.speech_started":
                            print("\n🗣️ [말하는 중...]")
                        elif event_type == "input_audio_buffer.speech_stopped":
                            print("🤐 [말 끝남] -> 서버 처리 중")

                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = data.get("transcript", "").strip()
                            if _is_valid_transcript(transcript) and transcript != last_transcript:
                                last_transcript = transcript
                                print(f"🎤 [User STT]: {transcript}")
                                stt_queue.put(transcript)
                        elif event_type == "input_audio_transcription.completed":
                            transcript = data.get("transcript", "").strip()
                            if _is_valid_transcript(transcript) and transcript != last_transcript:
                                last_transcript = transcript
                                print(f"🎤 [User STT]: {transcript}")
                                stt_queue.put(transcript)
                        elif event_type == "input_audio_transcription.final":
                            transcript = data.get("transcript", "").strip()
                            if _is_valid_transcript(transcript) and transcript != last_transcript:
                                last_transcript = transcript
                                print(f"🎤 [User STT]: {transcript}")
                                stt_queue.put(transcript)

                        elif event_type == "response.created":
                            print("✨ [System] AI가 답변 생성을 시작했습니다.")

                        elif event_type == "response.audio_transcript.done":
                            print(f"🤖 [AI Audio Transcript]: {data.get('transcript', '')}")

                        elif event_type == "response.text.done":
                            pass
                        
                        # 실시간/대체 이벤트에서 transcript 키가 있으면 모두 로깅
                        elif "transcript" in data:
                            transcript = str(data.get("transcript", "")).strip()
                            if _is_valid_transcript(transcript) and transcript != last_transcript:
                                last_transcript = transcript
                                print(f"🎤 [User STT]: {transcript} ({event_type})")
                                stt_queue.put(transcript)
                        elif event_type == "conversation.item.created":
                            item = data.get("item", {})
                            transcripts = _extract_transcripts_from_item(item)
                            if transcripts:
                                for transcript in transcripts:
                                    if transcript != last_transcript:
                                        last_transcript = transcript
                                        print(f"🎤 [User STT]: {transcript} (item)")
                                        stt_queue.put(transcript)
                            elif item.get("role") == "user":
                                pass

                        elif event_type == "response.audio.delta":
                            # Realtime 응답은 사용하지 않는다.
                            pass

                    except websockets.exceptions.ConnectionClosed:
                        break
                    except:
                        break

            async def send_context():
                last_emo = "Neutral"
                while is_running:
                    await asyncio.sleep(5.0)
                    if not ENABLE_EMOTION_CONTEXT:
                        continue
                    cur_emo = "Neutral"
                    with emotion_data_lock:
                        if latest_emotion_data:
                            cur_emo = latest_emotion_data.get("dominant_emotion", "Neutral")

                    if cur_emo != last_emo and cur_emo != "Neutral":
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "input_text",
                                                "text": f"참고: 고객 표정 감정은 {cur_emo}로 추정됨.",
                                            }
                                        ],
                                    },
                                }
                            )
                        )
                        last_emo = cur_emo

            if ENABLE_EMOTION_CONTEXT:
                await asyncio.gather(send_audio(), receive_audio(), send_context())
            else:
                await asyncio.gather(send_audio(), receive_audio())

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        worker_stop.set()
        try:
            mic_stream.stop_stream()
            mic_stream.close()
            p.terminate()
        except:
            pass

def socket_server_thread():
    global kiosk_socket
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 9999))
        server.listen(1)
        print("💡 [Socket] 포트 9999 대기 중...")
        while is_running:
            server.settimeout(1.0)
            try:
                conn, addr = server.accept()
                print(f"✅ [Socket] 클라이언트 연결됨: {addr}")
                with socket_lock:
                    kiosk_socket = conn
            except:
                continue
    except:
        pass

if __name__ == "__main__":
    threading.Thread(target=audio_thread_entry, daemon=True).start()
    threading.Thread(target=socket_server_thread, daemon=True).start()
    try:
        camera_loop_main()
    except KeyboardInterrupt:
        is_running = False
