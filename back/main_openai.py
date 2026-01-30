"""
[Project: AI Kiosk - Final Complete Integrated Version]
* Feature 1: Face/Emotion/Known-User Detection (Camera)
* Feature 2: Session Management (Auto Start/Stop)
* Feature 3: Dynamic Greeting based on Emotion (LLM)
* Feature 4: Full Voice Order System (STT/NLU/TTS)
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
import pyaudio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import websockets
from difflib import SequenceMatcher

# ==========================================
# [1. 환경 설정]
# ==========================================
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace가 설치되지 않았습니다. 감정 인식 기능이 비활성화됩니다.")

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
env_path = os.path.join(root_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
rest_headers = None
if not api_key:
    sys.exit("❌ 오류: OPENAI_API_KEY 없음")

CLASSIFIER_MODEL = os.getenv("OPENAI_CLASSIFIER_MODEL", "gpt-4o-mini")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# ==========================================
# [2. 전역 변수 및 설정]
# ==========================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024
ORDER_LOG_RETENTION_DAYS = 7
RECOMMEND_COUNT = 3
SESSION_TIMEOUT_SEC = 5.0

# 얼굴/감정 상태
latest_face_data = None
latest_emotion_data = None
latest_face_count = 0
face_data_lock = threading.Lock()
emotion_data_lock = threading.Lock()

# 시스템 상태
kiosk_socket = None
socket_lock = threading.Lock()
is_running = True

# 세션 관리
last_face_detected_ts = 0.0
is_in_session = False

# 오디오/통신 캐시
recent_action_cache = {"text": "", "ts": 0.0}
tts_state = {"text": "", "end_ts": 0.0}
tts_lock = threading.Lock()
tts_playing_event = threading.Event()
known_faces_cache = {"faces": [], "ts": 0.0}

# ==========================================
# [3. 상수 데이터 (힌트)]
# ==========================================
KOR_NUM_MAP = {"한": 1, "하나": 1, "한잔": 1, "두": 2, "둘": 2, "두잔": 2, "세": 3, "셋": 3, "서이": 3, "세잔": 3, "네": 4, "넷": 4, "너이": 4, "네잔": 4, "다섯": 5}
TEMPERATURE_ICE_HINTS = ["아이스", "ice", "차가", "차게", "냉", "콜드", "시원", "아이쓰"]
TEMPERATURE_HOT_HINTS = ["핫", "hot", "따뜻", "뜨거", "맨도롱", "따아", "온", "데운", "하스", "핫스", "뜨신"]
NEGATE_OPTION_HINTS = ["옵션 필요없", "필요 없", "없어", "그냥", "기본", "선택 안", "빼줘"]
ORDER_CONFIRM_HINTS = ["주세요", "주문", "담아", "추가", "할게", "그걸로", "이걸로", "줘"]
YES_HINTS = ["네", "응", "맞아", "그래", "오케이", "ok", "확인", "주문해"]
NO_HINTS = ["아니", "아니요", "아냐", "아니야", "아닌데", "아니에요"]
CANCEL_HINTS = ["취소", "처음부터", "리셋", "초기화", "그만", "안할래"]

# ==========================================
# [4. 헬퍼 함수들]
# ==========================================
def load_menu_data(file_path):
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, "r", encoding="utf-8") as f: return json.load(f)
    except: return None

def _normalize_text(s: str) -> str:
    if s is None: return ""
    s = str(s).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", s)

def _contains_any(text: str, hints: list) -> bool:
    t = _normalize_text(text)
    for h in hints:
        if _normalize_text(h) in t: return True
    return False

def _parse_temperature(text: str) -> str:
    t = _normalize_text(text)
    if any(_normalize_text(h) in t for h in TEMPERATURE_ICE_HINTS): return "ICE"
    if any(_normalize_text(h) in t for h in TEMPERATURE_HOT_HINTS): return "HOT"
    return ""

def _parse_quantity(text: str) -> int:
    if not text: return 1
    t = str(text)
    m = re.search(r"(\d+)\s*(잔|개|컵)?", t)
    if m:
        try: return max(1, int(m.group(1)))
        except: pass
    for k, v in KOR_NUM_MAP.items():
        if k in t: return v
    return 1

def _is_echo_text(stt_text: str) -> bool:
    norm_stt = _normalize_text(stt_text)
    if not norm_stt: return False
    with tts_lock:
        last_text = _normalize_text(tts_state["text"])
        end_ts = tts_state["end_ts"]
    if not last_text: return False
    if time.time() - end_ts > 3.0: return False
    return norm_stt in last_text or last_text in norm_stt

def _is_valid_transcript(text: str) -> bool:
    if not text: return False
    return cleaned.lower() != "none" if (cleaned := str(text).strip()) else False

def _post_json(url: str, payload: dict, headers: dict, timeout: int = 10):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"❌ [HTTP Error]: {e}")
        raise

def _post_bytes(url: str, payload: dict, headers: dict, timeout: int = 10):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        print(f"❌ [HTTP Bytes Error]: {e}")
        raise

def _play_wav_audio(wav_bytes: bytes):
    if not wav_bytes: return
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
            stream.write(wf.readframes(wf.getnframes()))
            stream.stop_stream()
            stream.close()
        finally:
            p.terminate()

def play_tts_blocking(reply: str, rest_headers: dict):
    if not reply: return
    tts_playing_event.set()
    with tts_lock:
        tts_state["text"] = reply
        tts_state["end_ts"] = time.time()
    try:
        tts_payload = {"model": "tts-1", "voice": "shimmer", "input": reply, "response_format": "wav"}
        audio_bytes = _post_bytes("https://api.openai.com/v1/audio/speech", tts_payload, rest_headers)
        _play_wav_audio(audio_bytes)
    except Exception as e:
        print(f"❌ [TTS Error]: {e}")
    finally:
        with tts_lock:
            tts_state["end_ts"] = time.time()
        tts_playing_event.clear()

def get_face_bbox_from_landmarks(face_landmarks, image_width, image_height):
    if not face_landmarks: return None
    x = [lm.x * image_width for lm in face_landmarks]
    y = [lm.y * image_height for lm in face_landmarks]
    return (int(min(x)*0.9), int(min(y)*0.9), int((max(x)-min(x))*1.1), int((max(y)-min(y))*1.1))

def save_order_log(menu_name, face_data, emotion_data=None, quantity: int = 1):
    file_name = os.path.join(current_dir, "order_logs.json")
    emo_str = emotion_data.get("dominant_emotion", "None") if emotion_data else "None"
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "menu": menu_name,
        "quantity": int(quantity or 1),
        "face_landmarks": face_data if face_data else [],
        "emotion": emo_str,
    }
    try:
        logs = []
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as f: logs = json.load(f)
        logs.append(entry)
        cutoff = datetime.now() - timedelta(days=ORDER_LOG_RETENTION_DAYS)
        logs = [l for l in logs if datetime.strptime(l["timestamp"], "%Y-%m-%d %H:%M:%S") >= cutoff]
        with open(file_name, "w", encoding="utf-8") as f: json.dump(logs, f, ensure_ascii=False, indent=2)
    except: pass

def get_recent_top_menus(file_path: str, top_k: int = 3) -> list:
    if not os.path.exists(file_path): return []
    try:
        with open(file_path, "r", encoding="utf-8") as f: logs = json.load(f)
    except: return []
    counts = {}
    for l in logs:
        m = str(l.get("menu", "")).strip()
        if m: counts[m] = counts.get(m, 0) + 1
    return [m for m, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]

def get_recent_top_menus_for_face(face_data: list, file_path: str, top_k: int = 3) -> list:
    if not face_data or not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        return []
    counts = {}
    for l in logs:
        logged_face = l.get("face_landmarks")
        if not isinstance(logged_face, list) or not logged_face:
            continue
        if _face_similarity(face_data, logged_face) > 0.01:
            continue
        menu = str(l.get("menu", "")).strip()
        qty = int(l.get("quantity", 1) or 1)
        if menu:
            counts[menu] = counts.get(menu, 0) + max(1, qty)
    return [m for m, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]

def send_kiosk_action(action_type: str, menu_name: str, quantity: int = 1, temperature: str = ""):
    if not action_type: return
    payload = json.dumps({"type": action_type, "menu_name": menu_name, "quantity": int(quantity or 1), "temperature": temperature or ""}, ensure_ascii=False)
    with socket_lock:
        if kiosk_socket:
            try: kiosk_socket.sendall((payload + "\n").encode("utf-8"))
            except: pass
    print(f"   └─ 📡 Kiosk Action: {payload}")

def log_checkout_items(items: list):
    if not items:
        return
    current_face = None
    with face_data_lock:
        if latest_face_data:
            current_face = latest_face_data[:]
    current_emotion = None
    with emotion_data_lock:
        if latest_emotion_data:
            current_emotion = latest_emotion_data.copy() if isinstance(latest_emotion_data, dict) else latest_emotion_data
    for item in items:
        menu = str(item.get("menu_name", "")).strip()
        qty = int(item.get("quantity", 1) or 1)
        if menu:
            save_order_log(menu, current_face, current_emotion, qty)

def _format_checkout_summary(items: list) -> str:
    parts = []
    for item in items:
        name = str(item.get("menu_name", "")).strip()
        qty = int(item.get("quantity", 1) or 1)
        if name:
            parts.append(f"{name} {qty}잔")
    if not parts:
        return "주문 내역이 비어 있어요. 다시 주문해 주세요."
    return "주문하신 내역은 " + ", ".join(parts) + " 맞으실까요?"

def kiosk_listener_thread(conn, addr):
    buffer = ""
    try:
        while is_running:
            data = conn.recv(4096)
            if not data:
                break
            buffer += data.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") == "checkout_preview":
                    items = payload.get("items", [])
                    if isinstance(items, list):
                        checkout_state["items"] = items
                        checkout_state["ts"] = time.time()
                        order_state.awaiting_confirmation = True
                        order_state.confirmation_ts = time.time()
                        summary = _format_checkout_summary(items)
                        print(f"🧾 [Checkout] {summary}")
                        if rest_headers:
                            play_tts_blocking(summary, rest_headers)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        with socket_lock:
            if kiosk_socket is conn:
                kiosk_socket = None
        print(f"🔌 [Socket] Disconnected: {addr}")

def _load_known_faces(file_name: str = None, ttl_sec: int = 10) -> list:
    if file_name is None: file_name = os.path.join(current_dir, "order_logs.json")
    now = time.time()
    if now - known_faces_cache["ts"] < ttl_sec and known_faces_cache["faces"]: return known_faces_cache["faces"]
    faces = []
    if os.path.exists(file_name):
        try:
            with open(file_name, "r", encoding="utf-8") as f: logs = json.load(f)
            for e in logs:
                if e.get("face_landmarks"): faces.append(e["face_landmarks"])
        except: pass
    known_faces_cache["faces"] = faces
    known_faces_cache["ts"] = now
    return faces

def _face_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b): return 1.0
    total = sum((float(la["x"])-float(lb["x"]))**2 + (float(la["y"])-float(lb["y"]))**2 + (float(la["z"])-float(lb["z"]))**2 for la, lb in zip(a, b))
    return total / max(1, len(a))

def _is_known_face(face_data: list, threshold: float = 0.01) -> bool:
    if not face_data: return False
    known_faces = _load_known_faces()
    for known in known_faces:
        if _face_similarity(face_data, known) <= threshold: return True
    return False

# ==========================================
# [5. 클래스 정의]
# ==========================================
class MenuCatalog:
    def __init__(self, raw_menu):
        self.menus = raw_menu.get("menus", []) if isinstance(raw_menu, dict) else (raw_menu if isinstance(raw_menu, list) else [])
        self._name_index = {}
        for m in self.menus:
            n = str(m.get("menu_name", "")).strip()
            if n: self._name_index.setdefault(_normalize_text(n), []).append(m)

    def resolve_menu(self, user_query: str):
        if not user_query: return None, "", []
        norm = _normalize_text(user_query)
        if norm in self._name_index: return self._name_index[norm][0], "", []
        
        clean = norm
        for p in ["아이스", "핫", "따뜻한", "시원한", "차가운", "hot", "ice"]:
            clean = clean.replace(_normalize_text(p), "")
        if clean and clean != norm and clean in self._name_index:
            return self._name_index[clean][0], "", []
        
        for db_name, menus in self._name_index.items():
            if (clean and clean in db_name) or (db_name in clean):
                return menus[0], "", []
        return None, "menu", []

    def _get_menu_price(self, menu):
        return menu.get("menu_price", 0)

class OrderState:
    def __init__(self): self.reset()
    def reset(self):
        self.pending_slot = ""
        self.pending_menu_name = ""
        self.last_question_slot = ""
        self.last_question_ts = 0.0
        self.awaiting_confirmation = False
        self.confirmation_ts = 0.0
        self.menu = None
        self.quantity = 1
        self.temperature = ""
    def set_menu(self, menu): self.menu = menu
    def can_finalize(self): return True

menu_catalog = None
order_state = OrderState()
checkout_state = {"items": [], "ts": 0.0}

# ==========================================
# [6. 핵심 로직 및 LLM]
# ==========================================
def generate_greeting(emotion: str, recommend_menus: list, rest_headers: dict) -> str:
    """
    [NEW] 감정과 추천 메뉴를 기반으로 동적 인사말 생성
    """
    menu_str = ", ".join(recommend_menus) if recommend_menus else "맛있는 커피"
    
    system_prompt = f"""
    너는 메가커피 키오스크의 AI 직원이다.
    CCTV로 파악한 고객의 감정은 현재 '{emotion}' 상태다.
    고객의 감정에 맞춰서 말투와 추천 멘트를 다르게 해야 한다.

    [행동 지침]
    1. **Happy (행복/기쁨):**
       - 톤: 활기차고 높은 텐션, 느낌표(!) 사용.
       - 내용: "오늘 기분 좋아 보이시네요! 시원한 {menu_str} 어떠세요?"
    
    2. **Sad/Neutral (슬픔/무표정/피곤):**
       - 톤: 차분하고 따뜻한 위로의 말투, 부드러운 느낌.
       - 내용: "오늘 조금 지쳐 보이시는데, 달콤한 {menu_str} 한 잔으로 충전하시는 건 어때요?"
    
    3. **Angry (화남/짜증):**
       - 톤: 매우 정중하고 신속하고 간결하게. 군더더기 없이.
       - 내용: "어서 오세요. 시원한 {menu_str} 빠르게 준비해 드릴까요?"

    [제약 사항]
    - 길이는 공백 포함 40자 이내로 짧게 한 문장만 출력.
    - 불필요한 서론(예: "알겠습니다") 생략하고 바로 인사말만 출력.
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "손님이 키오스크 앞에 섰어. 인사해줘."}
        ],
        "max_completion_tokens": 100
    }

    try:
        r = _post_json("https://api.openai.com/v1/chat/completions", payload, rest_headers, timeout=3)
        return r["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ [Greeting Error]: {e}")
        return "안녕하세요, 메가커피입니다. 무엇을 도와드릴까요?"

def classify_text(user_text: str, rest_headers: dict):
    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["order", "price", "recommend", "confirm", "cancel", "reset", "chitchat", "increase", "decrease", "remove_item", "update_order", "unknown"]},
            "order_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "menu_query": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "temperature": {"type": "string", "enum": ["HOT", "ICE", "ANY", ""]}
                    },
                    "required": ["menu_query", "quantity", "temperature"],
                    "additionalProperties": False
                }
            },
            "confidence": {"type": "number"}
        },
        "required": ["intent", "order_items", "confidence"],
        "additionalProperties": False
    }
    prompt = (
        "너는 한국 메가커피 키오스크다. JSON만 출력.\n"
        "1. order: 신규 주문 ('~담아줘', '카푸치노 하나')\n"
        "2. update_order: 정정/변경 ('~로 바꿔줘')\n"
        "3. increase/decrease/remove_item: 추가/감소/삭제\n"
        "4. price/recommend/chitchat/confirm/cancel\n"
        "5. temperature: HOT/ICE/ANY (모르면 ANY)\n"
        "6. order_items: 언급된 모든 메뉴 추출.\n"
    )
    payload = {
        "model": CLASSIFIER_MODEL,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_text}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "cls", "strict": True, "schema": schema}},
        "max_completion_tokens": 512
    }
    try:
        r = _post_json("https://api.openai.com/v1/chat/completions", payload, rest_headers, timeout=5)
        return json.loads(r["choices"][0]["message"]["content"])
    except: return {"intent": "unknown", "order_items": [], "confidence": 0.0}

def chat_fallback(user_text, headers):
    payload = {"model": CHAT_MODEL, "messages": [{"role": "system", "content": "친절한 카페 직원 AI. 모르는 메뉴는 모른다고 해."}, {"role": "user", "content": user_text}]}
    try:
        r = _post_json("https://api.openai.com/v1/chat/completions", payload, headers, timeout=5)
        return r["choices"][0]["message"]["content"]
    except: return "다시 말씀해 주시겠어요?"

def handle_user_text_order_flow(user_text): return "원하시는 메뉴를 말씀해 주세요.", []

def answer_price(user_text: str, cls: dict):
    query = cls.get("order_items", [{}])[0].get("menu_query", user_text)
    resolved, _, _ = menu_catalog.resolve_menu(query)
    if not resolved: return "가격 정보를 찾을 수 없어요.", []
    price = resolved.get("menu_price", 0)
    return f"{resolved['menu_name']} 가격은 {price}원 입니다.", []

def recommend_from_catalog(user_text, cls, top_menus):
    reco = ", ".join(top_menus[:3]) if top_menus else "아메리카노, 라떼"
    return f"추천 메뉴는 {reco} 입니다.", []

def handle_with_classifier(user_text: str, cls: dict, rest_headers: dict, top_menus: list):
    intent = cls.get("intent", "unknown")
    order_items = cls.get("order_items", [])

    # [1] 전체 취소
    if intent in ["cancel", "reset"] or _contains_any(user_text, CANCEL_HINTS):
        order_state.reset()
        return "네, 주문을 취소하고 처음부터 도와드릴게요.", []

    # [1-1] 결제 확인 대기 중 응답 처리
    if order_state.awaiting_confirmation:
        if intent == "confirm" or _contains_any(user_text, YES_HINTS):
            order_state.awaiting_confirmation = False
            if checkout_state["items"]:
                log_checkout_items(checkout_state["items"])
                checkout_state["items"] = []
            send_kiosk_action("checkout_confirm", "", 0, "")
            return "네, 주문 확인되었습니다. 결제 화면으로 안내해 드릴게요.", []
        if _contains_any(user_text, NO_HINTS):
            order_state.awaiting_confirmation = False
            checkout_state["items"] = []
            send_kiosk_action("checkout_cancel", "", 0, "")
            return "알겠습니다. 주문을 다시 확인해 주세요.", []
        return "주문 내역이 맞으신지 알려주세요.", []

    # [2] 문맥 복구 (Context Recovery)
    if order_state.pending_slot == "temperature" and order_state.pending_menu_name:
        is_real_new_menu = False
        for item in order_items:
            q = item.get("menu_query", "").strip()
            if q:
                res, _, _ = menu_catalog.resolve_menu(q)
                if res and str(res.get("menu_name")) not in ["한잔", "두잔"]:
                    is_real_new_menu = True; break
        
        if not is_real_new_menu:
            found_temp = ""
            # GPT 온도 확인
            if not found_temp:
                for item in order_items:
                    t = item.get("temperature")
                    if t in ["HOT", "ICE"]: found_temp = t; break
            # 텍스트 직접 확인
            if not found_temp:
                if any(x in user_text for x in ["아이스", "차가", "냉", "시원", "ice"]): found_temp = "ICE"
                elif any(x in user_text for x in ["핫", "따뜻", "뜨거", "hot"]): found_temp = "HOT"
            
            if found_temp:
                print(f"💡 [Context] 문맥 복구 성공: {order_state.pending_menu_name} -> {found_temp}")
                rec_action = {"type": "add", "menu_name": order_state.pending_menu_name, "quantity": 1, "temperature": found_temp}
                order_state.reset()
                return f"네, {found_temp} {rec_action['menu_name']} 1잔 담아드리겠습니다.", [rec_action]

    # [3] 다중 메뉴 처리
    if intent in ["order", "increase", "decrease", "remove_item", "update_order", "unknown"]:
        if not order_items:
            # Fallback
            if intent == "unknown":
                 if any(x in user_text for x in ["아이스", "따뜻"]): pass
                 else: return handle_user_text_order_flow(user_text)
            else: return "어떤 메뉴를 처리해 드릴까요?", []

        actions, replies = [], []
        for item in order_items:
            raw_q = item.get("menu_query", "").strip()
            qty = int(item.get("quantity") or 1)
            gpt_temp = item.get("temperature", "")
            if not raw_q: continue

            res_menu, _, _ = menu_catalog.resolve_menu(raw_q)
            if not res_menu:
                cleaned = re.sub(r"(아이스|핫|따뜻한|시원한|한잔|두잔|세잔)", "", raw_q).strip()
                if cleaned: res_menu, _, _ = menu_catalog.resolve_menu(cleaned)
            
            if not res_menu: continue
            
            t_name = str(res_menu.get("menu_name", "")).strip()
            final_temp = ""
            
            # 온도 결정
            if gpt_temp in ["HOT", "ICE"]: final_temp = gpt_temp
            elif any(x in raw_q+user_text for x in ["아이스", "차가", "냉"]): final_temp = "ICE"
            elif any(x in raw_q+user_text for x in ["핫", "따뜻", "뜨거"]): final_temp = "HOT"
            elif any(k in t_name for k in ["스무디", "에이드", "프라페", "주스"]): final_temp = "ICE"
            else:
                if intent != "remove_item":
                    order_state.pending_slot = "temperature"
                    order_state.pending_menu_name = t_name
                    return f"{t_name}는 따뜻하게 드릴까요, 시원하게 드릴까요?", []

            act_type = "add"
            if intent == "increase": act_type = "inc"
            elif intent == "decrease": act_type = "dec"
            elif intent == "remove_item": act_type = "remove"; qty = 0
            elif intent == "update_order": act_type = "set"
            
            actions.append({"type": act_type, "menu_name": t_name, "quantity": qty, "temperature": final_temp})
            replies.append(f"{final_temp} {t_name} {qty}잔")

        if not actions: return "말씀하신 메뉴를 찾지 못했어요.", []
        
        full_reply = ", ".join(replies)
        if intent in ["order", "increase", "unknown"]: full_reply += " 담아드리겠습니다."
        elif intent == "remove_item": full_reply += " 삭제했습니다."
        elif intent == "update_order": full_reply += " 변경해 드렸습니다."
        else: full_reply += " 빼드렸습니다."
        return f"네, {full_reply}", actions

    if intent == "price": return answer_price(user_text, cls)
    if intent == "recommend": return recommend_from_catalog(user_text, cls, top_menus)
    if intent == "chitchat": return chat_fallback(user_text, rest_headers), []
    
    return handle_user_text_order_flow(user_text)

# ==========================================
# [7. 스레드 실행 (카메라, 오디오, 소켓)]
# ==========================================
def camera_loop_main():
    global latest_face_data, latest_emotion_data, latest_face_count, is_running, last_face_detected_ts
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    face_landmarker = None
    try:
        base_opts = python.BaseOptions(model_asset_path=os.path.join(current_dir, "face_landmarker.task"))
        opts = vision.FaceLandmarkerOptions(base_options=base_opts, num_faces=10, min_face_detection_confidence=0.5, running_mode=vision.RunningMode.IMAGE)
        face_landmarker = vision.FaceLandmarker.create_from_options(opts)
    except: pass

    frame_count = 0
    try:
        while is_running:
            success, image = cap.read()
            if not success: continue
            frame_count += 1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            
            if face_landmarker:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                res = face_landmarker.detect(mp_img)
                cnt = len(res.face_landmarks) if res.face_landmarks else 0
                
                with face_data_lock:
                    latest_face_count = cnt
                    if cnt > 0:
                        last_face_detected_ts = time.time() # 세션 관리용 갱신

                if res.face_landmarks:
                    for i, landmarks in enumerate(res.face_landmarks):
                        bbox = get_face_bbox_from_landmarks(landmarks, w, h)
                        if bbox:
                            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                            
                            face_data_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
                            if i == 0:
                                with face_data_lock: latest_face_data = face_data_list
                            
                            is_known = _is_known_face(face_data_list)
                            label = "Known" if is_known else "Unknown"
                            color = (0, 255, 0) if is_known else (0, 0, 255)
                            cv2.putText(image, label, (bbox[0], max(20, bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                            if DEEPFACE_AVAILABLE and frame_count % 15 == 0 and i == 0:
                                try:
                                    face_crop = image_rgb[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                                    if face_crop.size > 0:
                                        res_emo = DeepFace.analyze(img_path=face_crop, actions=["emotion"], detector_backend="skip", enforce_detection=False, silent=True)
                                        if isinstance(res_emo, list): res_emo = res_emo[0]
                                        with emotion_data_lock:
                                            latest_emotion_data = {"dominant_emotion": res_emo.get("dominant_emotion", "Neutral")}
                                except: pass

            cv2.putText(image, f"Faces: {latest_face_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            emo_str = "None"
            with emotion_data_lock:
                if latest_emotion_data: emo_str = latest_emotion_data.get("dominant_emotion", "None")
            cv2.putText(image, f"Emotion: {emo_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Kiosk Eyes", image)
            if cv2.waitKey(1) & 0xFF == 27: is_running = False; break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def audio_thread_entry(): asyncio.run(audio_loop_async())

async def audio_loop_async():
    global is_running, menu_catalog, is_in_session, rest_headers

    raw_menu = load_menu_data(os.path.join(current_dir, "mega_coffee_menu.json"))
    menu_catalog = MenuCatalog(raw_menu if raw_menu else [])
    stt_queue = queue.Queue()
    
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
    headers = {"Authorization": "Bearer " + api_key, "OpenAI-Beta": "realtime=v1"}
    r_headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    rest_headers = r_headers

    # ==========================================
    # [세션 관리자] - 감정 기반 동적 인사 및 로그
    # ==========================================
    async def session_manager():
        global is_in_session
        while is_running:
            now = time.time()
            
            # [세션 시작]
            if not is_in_session and (now - last_face_detected_ts < 2.0) and (last_face_detected_ts > 0):
                is_in_session = True
                print("\n" + "="*30)
                print("👋 얼굴 인식 완료, 세션 시작")
                print("="*40 + "\n")
                
                order_state.reset()
                send_kiosk_action("reset", "", 0, "")
                
                # 감정 파악
                current_emotion = "Neutral"
                with emotion_data_lock:
                    if latest_emotion_data:
                        current_emotion = latest_emotion_data.get("dominant_emotion", "Neutral")
                print(f"🧐 [Detected Emotion] {current_emotion}")

                # 추천 메뉴 파악
                current_face = None
                with face_data_lock:
                    if latest_face_data: current_face = latest_face_data[:]
                
                log_path = os.path.join(current_dir, "order_logs.json")
                top_menus = []
                if current_face and _is_known_face(current_face):
                    top_menus = get_recent_top_menus_for_face(current_face, log_path, top_k=RECOMMEND_COUNT)
                if not top_menus:
                    top_menus = get_recent_top_menus(log_path, top_k=RECOMMEND_COUNT)
                
                # 동적 인사말 생성 (LLM 호출)
                greet = generate_greeting(current_emotion, top_menus, r_headers)
                full_greet = f"안녕하세요, 메가커피입니다. {greet}"
                print(f"🗣️ [AI Greeting]: {full_greet}")
                play_tts_blocking(full_greet, r_headers)

            # [세션 종료]
            elif is_in_session and (now - last_face_detected_ts > SESSION_TIMEOUT_SEC):
                is_in_session = False
                print("\n" + "="*40)
                print("🔒 얼굴 인식 종료, 세션 종료")
                print("="*40 + "\n")
                
                order_state.reset()
                send_kiosk_action("reset", "", 0, "")

            await asyncio.sleep(0.5)

    try:
        p = pyaudio.PyAudio()
        mic = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        async with websockets.connect(url, additional_headers=headers) as ws:
            print("✅ [Audio] OpenAI Connected")
            asyncio.create_task(session_manager())

            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": "한국어 음성 인식 전용.",
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1", "language": "ko"},
                    "turn_detection": {"type": "server_vad"}
                }
            }))

            def nlu_worker():
                last_txt = ""
                last_t = 0
                while is_running:
                    try: txt = stt_queue.get(timeout=0.5)
                    except: continue
                    
                    if not is_in_session:
                        print(f"🔇 [Ignored] Session Inactive (No Face): {txt}")
                        continue

                    if txt == last_txt and time.time()-last_t < 1.0: continue
                    last_txt = txt; last_t = time.time()

                    cls = classify_text(txt, r_headers)
                    print(f"🧠 [NLU] {cls.get('intent')} | {cls.get('order_items')}")
                    
                    top = get_recent_top_menus(os.path.join(current_dir, "order_logs.json"))
                    reply, acts = handle_with_classifier(txt, cls, r_headers, top)
                    
                    if acts:
                        for a in acts:
                            if a["type"] == "add":
                                # 주문 로그 저장 시 수량 반영
                                current_face = None
                                with face_data_lock:
                                    if latest_face_data: current_face = latest_face_data[:]
                                save_order_log(a["menu_name"], current_face, latest_emotion_data, a.get("quantity", 1))
                            send_kiosk_action(a["type"], a["menu_name"], a["quantity"], a["temperature"])
                    
                    if reply: play_tts_blocking(reply, r_headers)

            threading.Thread(target=nlu_worker, daemon=True).start()

            async def send_audio():
                while is_running:
                    if tts_playing_event.is_set():
                        await asyncio.sleep(0.1); continue
                    try:
                        data = await asyncio.to_thread(mic.read, CHUNK, exception_on_overflow=False)
                        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(data).decode("utf-8")}))
                    except: break
            
            async def recv_audio():
                while is_running:
                    try:
                        msg = await ws.recv()
                        d = json.loads(msg)
                        if d.get("type") == "conversation.item.input_audio_transcription.completed":
                            txt = d.get("transcript", "").strip()
                            if _is_valid_transcript(txt) and not _is_echo_text(txt):
                                print(f"🎤 [STT] {txt}")
                                stt_queue.put(txt)
                    except: break

            await asyncio.gather(send_audio(), recv_audio())

    except Exception as e: print(f"❌ Error: {e}")
    finally:
        try: mic.close(); p.terminate()
        except: pass

def socket_thread():
    global kiosk_socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 9999))
    s.listen(1)
    while is_running:
        try:
            s.settimeout(1.0)
            c, a = s.accept()
            with socket_lock: kiosk_socket = c
            print(f"✅ [Socket] Connected: {a}")
            threading.Thread(target=kiosk_listener_thread, args=(c, a), daemon=True).start()
        except: continue

if __name__ == "__main__":
    threading.Thread(target=audio_thread_entry, daemon=True).start()
    threading.Thread(target=socket_thread, daemon=True).start()
    camera_loop_main()
