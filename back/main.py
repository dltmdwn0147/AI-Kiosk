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
import math
import difflib
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
from PIL import Image

# ==========================================
# [1. 환경 설정]
# ==========================================
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace가 설치되지 않았습니다. 감정 인식 기능이 비활성화됩니다.")

try:
    import torch
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ torch/torchvision이 설치되지 않았습니다. 연령 추정 기능이 비활성화됩니다.")

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
ALLOW_BARGE_IN = False # 끼어들기 가능 : True, 불가능 : False
FACE_AUDIO_WINDOW_SEC = 1.0 # 얼굴 감지 후 음성 입력 허용 윈도우(초)

# 얼굴/감정 상태
latest_face_data = None
latest_emotion_data = None
latest_face_count = 0
face_data_lock = threading.Lock()
emotion_data_lock = threading.Lock()
age_data_lock = threading.Lock()
latest_age_value = None
latest_age_group = ""

# 시스템 상태
kiosk_socket = None
socket_lock = threading.Lock()
is_running = True

# 세션 관리
last_face_detected_ts = 0.0   # 마지막으로 얼굴이 감지된 시각
session_face_cache = None     # 최근 인식된 얼굴 랜드마크(임시 캐시)
session_face_ts = 0.0         # 최근 얼굴 인식 시각
pending_face_grace_ts = 0.0   # 얼굴 끊김 감지 후 유예 시작 시각
FACE_GRACE_SEC_IDLE = 2.0     # 주문이 없을 때 유예 시간(초)
FACE_GRACE_SEC_ACTIVE = 7.0   # 주문 진행 중 유예 시간(초)
ORDER_ACTIVE_WINDOW_SEC = 30.0  # 최근 주문 활동으로 간주할 시간(초)
last_order_activity_ts = 0.0  # 마지막 주문/수량 변경/결제 요청 시각
is_in_session = False         # 세션 활성 여부

# 오디오/통신 캐시
recent_action_cache = {"text": "", "ts": 0.0}
tts_state = {"text": "", "end_ts": 0.0}
tts_lock = threading.Lock()
tts_playing_event = threading.Event()
tts_stop_event = threading.Event()
known_faces_cache = {"faces": [], "ts": 0.0}
last_action_ts = 0.0      # 마지막 키오스크 액션 시각
last_action_summary = ""  # 마지막 키오스크 액션 요약
last_emotion_label = ""   # 최근 감정 라벨
negative_emotion_start_ts = 0.0  # 부정 감정 지속 시작 시각
last_emotion_prompt_ts = 0.0     # 감정 기반 오류 확인 멘트 출력 시각
EMOTION_ISSUE_SEC = 2.0          # 부정 감정 유지 시간 기준(초)
EMOTION_PROMPT_COOLDOWN_SEC = 15.0  # 감정 확인 멘트 재생 최소 간격(초)

# 연령 추정 모델
AGE_MODEL_PATH = os.getenv("AGE_MODEL_PATH", os.path.join(current_dir, "age_model_epoch_10.pt"))
age_model = None
age_transform = None

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
KIOSK_KEYWORDS = ["주문", "담아", "추가", "빼", "삭제", "취소", "가격", "얼마", "추천", "메뉴", "아이스", "핫", "따뜻", "시원", "결제", "확인", "그걸로", "이걸로", "한잔", "두잔", "세잔"]
CHECKOUT_HINTS = ["결제", "계산", "결재"]
MODIFY_HINTS = ["줄여", "빼", "삭제", "취소", "추가", "늘려", "더", "변경", "바꿔"]
DEC_HINTS = ["줄여", "줄이", "빼", "감소", "빼줘"]
INC_HINTS = ["추가", "더", "늘려", "증가", "더해"]
SET_HINTS = ["바꿔", "변경", "수정", "로 해", "으로 해"]

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

def _has_menu_keyword(text: str) -> bool:
    if not menu_catalog or not getattr(menu_catalog, "menus", None):
        return False
    norm = _normalize_text(text)
    if not norm:
        return False
    for m in menu_catalog.menus:
        name = _normalize_text(m.get("menu_name", ""))
        if name and (name in norm or norm in name):
            return True
    return False

def _has_kiosk_keywords(text: str) -> bool:
    return _contains_any(text, KIOSK_KEYWORDS) or _has_menu_keyword(text)

def _should_process_utterance(text: str, intent: str) -> bool:
    if order_state.awaiting_confirmation:
        if _contains_any(text, YES_HINTS) or _contains_any(text, NO_HINTS):
            return True
    if intent in ["order", "price", "recommend", "confirm", "cancel", "reset", "increase", "decrease", "remove_item", "update_order"]:
        return True
    return _has_kiosk_keywords(text)

def _has_modify_keywords(text: str) -> bool:
    return _contains_any(text, MODIFY_HINTS)

def _pick_action_type(intent: str, idx: int, total_items: int, user_text: str) -> str:
    has_dec = _contains_any(user_text, DEC_HINTS)
    has_inc = _contains_any(user_text, INC_HINTS)
    has_set = _contains_any(user_text, SET_HINTS)
    if intent == "remove_item":
        return "remove"
    if has_dec and has_inc and total_items > 1:
        return "dec" if idx == 0 else "inc"
    if has_dec and not has_inc:
        return "dec"
    if has_inc and not has_dec:
        return "inc"
    if has_set:
        return "set"
    if intent == "increase":
        return "inc"
    if intent == "decrease":
        return "dec"
    if intent == "update_order":
        return "set"
    return "add"

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
    if norm_stt in last_text or last_text in norm_stt:
        print("🤐 [말 끝남] (필터링)")
        return True
    return False

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
            chunk = 1024
            while True:
                if tts_stop_event.is_set():
                    break
                data = wf.readframes(chunk)
                if not data:
                    break
                stream.write(data)
            stream.stop_stream()
            stream.close()
        finally:
            p.terminate()

def play_tts_blocking(reply: str, rest_headers: dict):
    if not reply: return
    tts_stop_event.clear()
    tts_playing_event.set()
    with tts_lock:
        tts_state["text"] = reply
        tts_state["end_ts"] = time.time()
    try:
        print(f"🗣️ [AI Reply]: {reply}")
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
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
        logs.append(entry)
        cutoff = datetime.now() - timedelta(days=ORDER_LOG_RETENTION_DAYS)
        logs = [l for l in logs if datetime.strptime(l["timestamp"], "%Y-%m-%d %H:%M:%S") >= cutoff]
        with open(file_name, "w", encoding="utf-8") as f: json.dump(logs, f, ensure_ascii=False, indent=2)
        known_faces_cache["faces"] = []
        known_faces_cache["ts"] = 0.0
        print(f"✅ [Order Log] 저장 완료: {menu_name} x{entry['quantity']}")
    except Exception as e:
        print(f"❌ [Order Log] 저장 실패: {e}")

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
    # 주문 활동 타임스탬프 갱신 (세션 유예 판단용)
    if action_type in ["add", "inc", "dec", "set", "checkout_request"]:
        global last_order_activity_ts
        last_order_activity_ts = time.time()
    # 최근 액션 기록 (오류 감지용)
    if action_type in ["add", "inc", "dec", "set", "remove"]:
        global last_action_ts, last_action_summary
        last_action_ts = time.time()
        temp_label = f" {temperature}" if temperature else ""
        last_action_summary = f"{menu_name}{temp_label} {int(quantity or 1)}잔"
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
                        global last_order_activity_ts
                        last_order_activity_ts = time.time()
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

def _load_age_model():
    global age_model, age_transform
    if not TORCH_AVAILABLE:
        return
    if not os.path.exists(AGE_MODEL_PATH):
        print(f"⚠️ 연령 모델 파일 없음: {AGE_MODEL_PATH}")
        return
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    ckpt = torch.load(AGE_MODEL_PATH, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    age_model = model
    age_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"✅ 연령 모델 로드 완료: {AGE_MODEL_PATH}")

def _predict_age_group(face_crop_bgr: np.ndarray):
    if not TORCH_AVAILABLE or age_model is None or age_transform is None:
        return None, ""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None, ""
    img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    x = age_transform(pil).unsqueeze(0)
    with torch.no_grad():
        pred = age_model(x).squeeze().item()
    age = max(0, pred)
    group = int(age // 10) * 10
    return age, f"{group}대"

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
        self.last_suggested_menu = ""
        self.last_recommendations = []
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
        r = _post_json("https://api.openai.com/v1/chat/completions", payload, rest_headers, timeout=10)
        return json.loads(r["choices"][0]["message"]["content"])
    except Exception:
        try:
            r = _post_json("https://api.openai.com/v1/chat/completions", payload, rest_headers, timeout=15)
            return json.loads(r["choices"][0]["message"]["content"])
        except Exception:
            return {"intent": "unknown", "order_items": [], "confidence": 0.0}

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
    order_state.last_suggested_menu = resolved.get("menu_name", "")
    price = resolved.get("menu_price", 0)
    return f"{resolved['menu_name']} 가격은 {price}원 입니다.", []

def _tokenize_query(text: str) -> set:
    tokens = re.findall(r"[가-힣]+|[a-z0-9]+", str(text).lower())
    stop = {"그거", "이거", "저거", "뭐", "메뉴", "추천", "주세요", "줘", "하고", "해서", "싶은", "같은", "거", "좀", "그", "이", "저",
            "음료", "마실", "마실거", "주문", "찾아", "맛", "느낌"}
    return {t for t in tokens if t and t not in stop and len(t) > 1}

def _expand_tokens(tokens: set) -> set:
    synonyms = {
        "검은": "진한",
        "검정": "진한",
        "쓴": "쓴맛",
        "쓰다": "쓴맛",
        "쌉싸름": "쓴맛",
        "달달": "달콤",
        "시원": "차가운",
        "차가운": "시원",
        "따뜻": "따뜻한",
        "핫": "hot",
        "아이스": "ice",
    }
    expanded = set(tokens)
    for t in list(tokens):
        if t in synonyms:
            expanded.add(synonyms[t])
    return expanded

def _temperature_hint(text: str) -> str:
    if any(k in text for k in ["아이스", "차가", "시원", "ice"]): return "ICE"
    if any(k in text for k in ["핫", "따뜻", "뜨거", "hot"]): return "HOT"
    return ""

def _is_negative_emotion(emotion: str) -> bool:
    e = str(emotion or "").lower()
    return any(k in e for k in ["angry", "anger", "sad", "fear", "disgust", "frustrat"])

def _detect_order_issue(text: str, intent: str) -> str:
    # 최근 주문 액션 직후 불만/오류 신호가 있으면 확인 질문으로 전환
    action_intents = {"order", "increase", "decrease", "update_order", "remove_item", "confirm", "price"}
    if intent in action_intents:
        return ""
    error_hints = ["아니", "아닌데", "틀렸", "잘못", "왜", "그게 아니", "그거 아니", "다른걸로", "다르게", "이상", "오류"]
    if time.time() - last_action_ts > 10.0:
        return ""
    if not any(h in text for h in error_hints):
        return ""
    current_emotion = ""
    with emotion_data_lock:
        if latest_emotion_data:
            current_emotion = latest_emotion_data.get("dominant_emotion", "")
    if _is_negative_emotion(current_emotion) or any(h in text for h in ["틀렸", "잘못", "아니", "왜"]):
        if last_action_summary:
            return f"방금 {last_action_summary}로 처리했는데, 맞지 않나요? 원하시는 메뉴로 다시 말씀해 주세요."
        return "방금 주문이 잘못된 것 같아요. 원하시는 메뉴로 다시 말씀해 주세요."
    return ""

def _intent_category_from_text(text: str) -> str:
    t = str(text)
    if _contains_any(t, ["달달", "달콤", "부드", "라떼", "바닐라", "흑당", "카라멜"]):
        return "latte"
    if _contains_any(t, ["상큼", "과일", "에이드", "주스", "레몬", "자몽"]):
        return "ade"
    if _contains_any(t, ["진한", "쓴", "쌉싸", "아메리카노", "콜드브루", "에스프레소"]):
        return "coffee"
    if _contains_any(t, ["차", "티", "녹차", "홍차", "캐모마일"]):
        return "tea"
    if _contains_any(t, ["프라페", "스무디", "쉐이크", "빙수"]):
        return "blended"
    return ""

def _embed_text(text: str, headers: dict):
    payload = {"model": "text-embedding-3-small", "input": text}
    r = _post_json("https://api.openai.com/v1/embeddings", payload, headers, timeout=10)
    return r["data"][0]["embedding"]

def _cos_sim(a, b) -> float:
    if not a or not b: return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(x*x for x in b)) or 1.0
    return dot / (na * nb)

def _build_menu_embedding_cache(headers: dict, file_path: str):
    if not headers:
        raise RuntimeError("rest_headers not set")
    menus = menu_catalog.menus if menu_catalog else []
    cache = {"model": "text-embedding-3-small", "items": []}
    for m in menus:
        name = str(m.get("menu_name", "")).strip()
        desc = str(m.get("menu_description", "")).strip()
        if not name:
            continue
        text = f"{name} - {desc}"
        emb = _embed_text(text, headers)
        cache["items"].append({"menu_name": name, "menu_temperature": str(m.get("menu_temperature", "")), "embedding": emb})
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

def _load_menu_embedding_cache(file_path: str):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _llm_reco_constraints(user_text: str, headers: dict):
    schema = {
        "type": "object",
        "properties": {
            "include_terms": {"type": "array", "items": {"type": "string"}},
            "exclude_terms": {"type": "array", "items": {"type": "string"}},
            "avoid_caffeine": {"type": "boolean"},
            "exclude_previous": {"type": "boolean"},
            "desired_category": {"type": "string", "enum": ["drink", "dessert", "any"]},
            "require_decaf": {"type": "boolean"},
            "exclude_categories": {"type": "array", "items": {"type": "string"}},
            "include_categories": {"type": "array", "items": {"type": "string"}},
            "max_decaf_count": {"type": "integer"},
            "temperature": {"type": "string", "enum": ["ICE", "HOT", "ANY"]},
            "count": {"type": "integer"}
        },
        "required": ["include_terms", "exclude_terms", "avoid_caffeine", "exclude_previous", "desired_category", "require_decaf", "exclude_categories", "include_categories", "max_decaf_count", "temperature", "count"],
        "additionalProperties": False
    }
    prompt = (
        "사용자 발화에서 추천 조건을 추출해 JSON으로만 답하세요.\n"
        "- include_terms: 꼭 포함되면 좋은 키워드\n"
        "- exclude_terms: 제외해야 할 키워드/메뉴명\n"
        "- avoid_caffeine: 카페인 제외 요청 여부\n"
        "- require_decaf: 디카페인 요청 여부\n"
        "- exclude_categories: 제외할 카테고리 리스트 (coffee/latte/smoothie/ade/frappe/tea/dessert/bakery/etc)\n"
        "- include_categories: 포함할 카테고리 리스트 (coffee/latte/smoothie/ade/frappe/tea/dessert/bakery/etc)\n"
        "- max_decaf_count: 디카페인 최대 추천 개수(요청 없으면 1)\n"
        "- exclude_previous: '그거 말고/다른 거'처럼 이전 추천 제외 여부\n"
        "- desired_category: dessert(디저트) / drink(음료) / any\n"
        "- temperature: ICE/HOT/ANY\n"
        "- count: 추천 개수(없으면 3)\n"
    )
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_text}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "reco", "strict": True, "schema": schema}},
        "max_completion_tokens": 300
    }
    try:
        r = _post_json("https://api.openai.com/v1/chat/completions", payload, headers, timeout=10)
        return json.loads(r["choices"][0]["message"]["content"])
    except Exception:
        return {"include_terms": [], "exclude_terms": [], "avoid_caffeine": False, "exclude_previous": False, "desired_category": "any", "require_decaf": False, "exclude_categories": [], "include_categories": [], "max_decaf_count": 1, "temperature": "ANY", "count": 3}

def _llm_menu_category(name: str, desc: str, headers: dict) -> str:
    schema = {
        "type": "object",
        "properties": {"category": {"type": "string", "enum": ["drink", "dessert", "other"]}},
        "required": ["category"],
        "additionalProperties": False,
    }
    prompt = (
        "다음 메뉴가 음료인지 디저트인지 분류해 JSON으로만 답하세요.\n"
        "- drink: 음료(커피/라떼/에이드/주스/스무디/프라페/티 등)\n"
        "- dessert: 디저트/베이커리(케이크/쿠키/마카롱/아이스크림/빵/샌드위치/핫도그 등)\n"
        "- other: 기타\n"
        "주의: 스무디/요거트/주스/에이드는 디저트가 아니라 음료로 분류하세요.\n"
    )
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": f"{name} | {desc}"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "cat", "strict": True, "schema": schema}},
        "max_completion_tokens": 100
    }
    try:
        r = _post_json("https://api.openai.com/v1/chat/completions", payload, headers, timeout=10)
        return json.loads(r["choices"][0]["message"]["content"]).get("category", "other")
    except Exception:
        return "other"

def _build_menu_category_cache(headers: dict, file_path: str):
    if not headers:
        raise RuntimeError("rest_headers not set")
    menus = menu_catalog.menus if menu_catalog else []
    cache = {"items": []}
    for m in menus:
        name = str(m.get("menu_name", "")).strip()
        desc = str(m.get("menu_description", "")).strip()
        if not name:
            continue
        category = _llm_menu_category(name, desc, headers)
        cache["items"].append({"menu_name": name, "category": category})
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def _load_menu_category_cache(file_path: str):
    if not os.path.exists(file_path):
        return None

def _llm_menu_subcategory(name: str, desc: str, headers: dict) -> str:
    schema = {
        "type": "object",
        "properties": {"subcategory": {"type": "string", "enum": ["coffee", "latte", "smoothie", "ade", "frappe", "tea", "dessert", "bakery", "etc"]}},
        "required": ["subcategory"],
        "additionalProperties": False,
    }
    prompt = (
        "다음 메뉴를 가장 적합한 카테고리로 분류해 JSON으로만 답하세요.\n"
        "- coffee: 아메리카노/콜드브루/에스프레소/모카 등 커피 기반\n"
        "- latte: 라떼류\n"
        "- smoothie: 스무디/요거트/쉐이크 계열 (음료)\n"
        "- ade: 에이드/주스/과일음료\n"
        "- frappe: 프라페/블렌디드\n"
        "- tea: 차/티\n"
        "- dessert: 디저트류 (케이크/쿠키/마카롱/아이스크림 등)\n"
        "- bakery: 빵/샌드위치/핫도그 등 베이커리/간식\n"
        "- etc: 기타\n"
        "주의: 스무디/요거트/쉐이크는 디저트가 아니라 음료로 분류하세요.\n"
    )
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": f"{name} | {desc}"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "subcat", "strict": True, "schema": schema}},
        "max_completion_tokens": 100
    }
    try:
        r = _post_json("https://api.openai.com/v1/chat/completions", payload, headers, timeout=10)
        return json.loads(r["choices"][0]["message"]["content"]).get("subcategory", "etc")
    except Exception:
        return "etc"

def _build_menu_subcategory_cache(headers: dict, file_path: str):
    if not headers:
        raise RuntimeError("rest_headers not set")
    menus = menu_catalog.menus if menu_catalog else []
    cache = {"items": []}
    for m in menus:
        name = str(m.get("menu_name", "")).strip()
        desc = str(m.get("menu_description", "")).strip()
        if not name:
            continue
        subcat = _llm_menu_subcategory(name, desc, headers)
        cache["items"].append({"menu_name": name, "subcategory": subcat})
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def _load_menu_subcategory_cache(file_path: str):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def recommend_from_catalog(user_text, cls, top_menus):
    try:
        if not menu_catalog or not menu_catalog.menus:
            reco = ", ".join(top_menus[:3]) if top_menus else "아메리카노, 라떼"
            return f"추천 메뉴는 {reco} 입니다.", []

        cache_path = os.path.join(current_dir, "menu_embeddings.json")
        cache = _load_menu_embedding_cache(cache_path)
        cat_path = os.path.join(current_dir, "menu_categories.json")
        cat_cache = _load_menu_category_cache(cat_path)
        subcat_path = os.path.join(current_dir, "menu_subcategories.json")
        subcat_cache = _load_menu_subcategory_cache(subcat_path)
        if not cache:
            # 캐시가 없으면 즉시 fallback (실시간 응답 지연 방지)
            reco = ", ".join(top_menus[:3]) if top_menus else "아메리카노, 라떼"
            return f"추천 메뉴는 {reco} 입니다.", []
        if not cat_cache:
            cat_cache = {"items": []}
        if not subcat_cache:
            subcat_cache = {"items": []}

        constraints = _llm_reco_constraints(user_text, rest_headers)
        query_vec = _embed_text(user_text, rest_headers)

        include_terms = [t for t in constraints.get("include_terms", []) if t]
        exclude_terms = [t for t in constraints.get("exclude_terms", []) if t]
        avoid_caffeine = bool(constraints.get("avoid_caffeine", False))
        require_decaf = bool(constraints.get("require_decaf", False))
        exclude_categories = [c for c in constraints.get("exclude_categories", []) if c]
        include_categories = [c for c in constraints.get("include_categories", []) if c]
        max_decaf_count = int(constraints.get("max_decaf_count", 1) or 1)
        exclude_previous = bool(constraints.get("exclude_previous", False))
        desired_category = constraints.get("desired_category", "any")
        temperature = constraints.get("temperature", "ANY")
        count = int(constraints.get("count", 3) or 3)
        last_recos = set(order_state.last_recommendations or [])
        if not include_categories and desired_category in ["dessert", "drink"]:
            include_categories = ["dessert", "bakery"] if desired_category == "dessert" else ["coffee", "latte", "smoothie", "ade", "frappe", "tea", "etc"]

        # 최소한의 퍼지 매칭으로 제외 대상 보정
        menu_names = [it.get("menu_name", "") for it in cache.get("items", []) if it.get("menu_name")]
        def _fuzzy_expand(terms, cutoff=0.75):
            out = set()
            for t in terms:
                matches = difflib.get_close_matches(t, menu_names, n=3, cutoff=cutoff)
                out.update(matches)
            return out
        fuzzy_excludes = _fuzzy_expand(exclude_terms)

        scored = []
        category_map = {it.get("menu_name"): it.get("category") for it in cat_cache.get("items", [])}
        subcategory_map = {it.get("menu_name"): it.get("subcategory") for it in subcat_cache.get("items", [])}
        for item in cache.get("items", []):
            name = item.get("menu_name", "")
            temp = str(item.get("menu_temperature", "")).upper()
            desc = ""
            for m in (menu_catalog.menus if menu_catalog else []):
                if str(m.get("menu_name", "")).strip() == name:
                    desc = str(m.get("menu_description", "")).strip()
                    break
            if desired_category in ["drink", "dessert"]:
                if category_map.get(name) and category_map.get(name) != desired_category:
                    continue
            if exclude_categories and subcategory_map.get(name) in exclude_categories:
                continue
            if include_categories and subcategory_map.get(name) not in include_categories:
                continue
            if temperature != "ANY" and temp and temp != temperature:
                continue
            if exclude_previous and name in last_recos:
                continue
            if exclude_terms and any(t in name for t in exclude_terms):
                continue
            if fuzzy_excludes and name in fuzzy_excludes:
                continue
            if require_decaf and "디카페인" not in name:
                continue
            if avoid_caffeine and any(k in name for k in ["아메리카노", "콜드브루", "에스프레소", "모카", "커피", "라떼", "카페", "디카페인"]):
                continue
            if include_terms and not any(t in (name + " " + desc) for t in include_terms):
                continue
            sim = _cos_sim(query_vec, item.get("embedding", []))
            scored.append((sim, name))

        scored.sort(key=lambda x: x[0], reverse=True)
        names = []
        seen = set()
        decaf_used = 0
        for _, n in scored:
            if n in seen:
                continue
            if not require_decaf and "디카페인" in n and decaf_used >= max_decaf_count:
                continue
            seen.add(n)
            if "디카페인" in n:
                decaf_used += 1
            names.append(n)
            if len(names) >= count:
                break

        if names:
            order_state.last_suggested_menu = names[0]
            order_state.last_recommendations = names
            return f"추천 메뉴는 {', '.join(names)} 입니다.", []
    except Exception as e:
        print(f"❌ [Recommend Error]: {e}")

    reco = ", ".join(top_menus[:3]) if top_menus else "아메리카노, 라떼"
    if top_menus:
        order_state.last_suggested_menu = top_menus[0]
        order_state.last_recommendations = top_menus[:3]
    return f"추천 메뉴는 {reco} 입니다.", []

def handle_with_classifier(user_text: str, cls: dict, rest_headers: dict, top_menus: list):
    intent = cls.get("intent", "unknown")
    order_items = cls.get("order_items", [])

    # [0-1] 디카페인 메뉴 존재 여부 질문은 짧게 응답
    if "디카페인" in str(user_text) and intent in ["chitchat", "price", "recommend", "unknown"]:
        resolved, _, _ = menu_catalog.resolve_menu(user_text) if menu_catalog else (None, "", [])
        base_name = ""
        if resolved:
            base_name = str(resolved.get("menu_name", "")).replace("디카페인", "").strip()
        elif menu_catalog:
            # 문장 안 메뉴명 추출
            for m in menu_catalog.menus:
                name = str(m.get("menu_name", "")).strip()
                if name and name in str(user_text) and "디카페인" not in name:
                    base_name = name
                    break
        if base_name and menu_catalog:
            decaf_name = f"디카페인 {base_name}".strip()
            if any(str(m.get("menu_name", "")).strip() == decaf_name for m in menu_catalog.menus):
                return f"네, {decaf_name} 있습니다.", []
        return "확인해 드릴게요.", []

    # [0] 전체 삭제/초기화 직접 감지
    if _contains_any(user_text, ["모든메뉴", "전메뉴", "전체메뉴"]) or (
        _contains_any(user_text, ["모든", "전부", "전체"])
        and _contains_any(user_text, ["메뉴", "주문", "장바구니"])
        and "다른" not in str(user_text)
    ) or (
        "다" in str(user_text)
        and "다른" not in str(user_text)
        and _contains_any(user_text, ["삭제", "취소", "비워", "없애", "리셋", "초기화"])
    ):
        order_state.reset()
        return "네, 장바구니를 모두 비워드릴게요.", [{"type": "reset", "menu_name": "", "quantity": 0, "temperature": ""}]

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
            if intent == "remove_item":
                if _contains_any(user_text, ["전체", "모두", "다", "전부", "싹", "리셋", "초기화"]):
                    order_state.reset()
                    return "네, 장바구니를 모두 비워드릴게요.", [{"type": "reset", "menu_name": "", "quantity": 0, "temperature": ""}]
                return "어떤 메뉴를 삭제해 드릴까요?", []
            if intent == "unknown":
                 if any(x in user_text for x in ["아이스", "따뜻"]): pass
                 else: return handle_user_text_order_flow(user_text)
            else: return "어떤 메뉴를 처리해 드릴까요?", []

        actions, replies = [], []
        total_items = len(order_items)
        for idx, item in enumerate(order_items):
            raw_q = item.get("menu_query", "").strip()
            qty = int(item.get("quantity") or 1)
            gpt_temp = item.get("temperature", "")
            if not raw_q:
                continue
            if raw_q in ["그거", "그걸", "이거", "이걸", "저거", "저걸"] and order_state.last_suggested_menu:
                raw_q = order_state.last_suggested_menu

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

            act_type = _pick_action_type(intent, idx, total_items, user_text)
            if act_type == "remove":
                qty = 0
            
            actions.append({"type": act_type, "menu_name": t_name, "quantity": qty, "temperature": final_temp})
            replies.append(f"{final_temp} {t_name} {qty}잔")

        if not actions: return "말씀하신 메뉴를 찾지 못했어요.", []
        
        if intent in ["order", "increase", "unknown"]:
            short_reply = "네, 처리해드렸어요."
        elif intent == "remove_item":
            short_reply = "네, 삭제했습니다."
        elif intent == "update_order":
            short_reply = "네, 변경했습니다."
        elif intent == "decrease":
            short_reply = "네, 수량을 줄였습니다."
        else:
            short_reply = "네, 처리해드렸어요."
        return short_reply, actions

    if intent == "price": return answer_price(user_text, cls)
    if intent == "recommend": return recommend_from_catalog(user_text, cls, top_menus)
    if intent == "chitchat": return chat_fallback(user_text, rest_headers), []
    
    return handle_user_text_order_flow(user_text)

# ==========================================
# [7. 스레드 실행 (카메라, 오디오, 소켓)]
# ==========================================
def camera_loop_main():
    global latest_face_data, latest_emotion_data, latest_face_count, is_running, last_face_detected_ts, session_face_cache, session_face_ts, pending_face_grace_ts
    global latest_age_value, latest_age_group
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    face_landmarker = None
    try:
        base_opts = python.BaseOptions(model_asset_path=os.path.join(current_dir, "face_landmarker.task"))
        opts = vision.FaceLandmarkerOptions(base_options=base_opts, num_faces=10, min_face_detection_confidence=0.5, running_mode=vision.RunningMode.IMAGE)
        face_landmarker = vision.FaceLandmarker.create_from_options(opts)
    except: pass

    frame_count = 0
    age_last_print_ts = 0.0
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
                        if res.face_landmarks and res.face_landmarks[0]:
                            session_face_cache = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in res.face_landmarks[0]]
                            session_face_ts = last_face_detected_ts
                            pending_face_grace_ts = 0.0

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

                            if TORCH_AVAILABLE and age_model is not None and i == 0:
                                try:
                                    face_crop_bgr = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                                    age_val, age_group = _predict_age_group(face_crop_bgr)
                                    if age_group:
                                        with age_data_lock:
                                            latest_age_value = age_val
                                            latest_age_group = age_group
                                        now_ts = time.time()
                                        if now_ts - age_last_print_ts > 1.0:
                                            age_last_print_ts = now_ts
                                            print(f"🧓 [Age] {age_group} (예측 {age_val:.1f})")
                                except Exception:
                                    pass

            cv2.putText(image, f"Faces: {latest_face_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            emo_str = "None"
            with emotion_data_lock:
                if latest_emotion_data: emo_str = latest_emotion_data.get("dominant_emotion", "None")
            cv2.putText(image, f"Emotion: {emo_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            age_group_text = ""
            with age_data_lock:
                if latest_age_group:
                    age_group_text = latest_age_group
            if age_group_text:
                cv2.putText(image, f"Age: {age_group_text}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

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
    # 임베딩/카테고리 캐시 백그라운드 생성 (최초 1회)
    def _ensure_embedding_cache():
        cache_path = os.path.join(current_dir, "menu_embeddings.json")
        if not os.path.exists(cache_path):
            try:
                _build_menu_embedding_cache(rest_headers, cache_path)
                print("✅ [Embeddings] menu_embeddings.json 생성 완료")
            except Exception as e:
                print(f"❌ [Embeddings] 생성 실패: {e}")
        cat_path = os.path.join(current_dir, "menu_categories.json")
        if not os.path.exists(cat_path):
            try:
                _build_menu_category_cache(rest_headers, cat_path)
                print("✅ [Categories] menu_categories.json 생성 완료")
            except Exception as e:
                print(f"❌ [Categories] 생성 실패: {e}")
        subcat_path = os.path.join(current_dir, "menu_subcategories.json")
        if not os.path.exists(subcat_path):
            try:
                _build_menu_subcategory_cache(rest_headers, subcat_path)
                print("✅ [Subcategories] menu_subcategories.json 생성 완료")
            except Exception as e:
                print(f"❌ [Subcategories] 생성 실패: {e}")
    
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
    headers = {"Authorization": "Bearer " + api_key, "OpenAI-Beta": "realtime=v1"}
    r_headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    rest_headers = r_headers
    threading.Thread(target=_ensure_embedding_cache, daemon=True).start()

    # ==========================================
    # [세션 관리자] - 감정 기반 동적 인사 및 로그
    # ==========================================
    async def session_manager():
        global is_in_session, pending_face_grace_ts, session_face_cache, session_face_ts, last_order_activity_ts
        global last_emotion_label, negative_emotion_start_ts, last_emotion_prompt_ts
        while is_running:
            now = time.time()
            
            # [세션 시작]
            if not is_in_session and (now - last_face_detected_ts < 2.0) and (last_face_detected_ts > 0):
                is_in_session = True
                print("\n" + "="*30)
                print("👋 얼굴 인식 완료, 세션 시작")
                print("="*40 + "\n")

                order_state.reset()
                last_order_activity_ts = 0.0
                send_kiosk_action("reset", "", 0, "")
                send_kiosk_action("open_main", "", 0, "")
                
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
                face_top_menus = []
                is_known = False
                if current_face and _is_known_face(current_face):
                    is_known = True
                    face_top_menus = get_recent_top_menus_for_face(current_face, log_path, top_k=RECOMMEND_COUNT)
                global_top_menus = get_recent_top_menus(log_path, top_k=RECOMMEND_COUNT)
                emotion_menus = global_top_menus or face_top_menus

                # 동적 인사말 생성 (LLM 호출)
                is_neutral = str(current_emotion).lower() in ["neutral", "none"]
                if is_neutral:
                    base_greet = "오늘은 어떤 메뉴 도와드릴까요?"
                else:
                    base_greet = generate_greeting(current_emotion, emotion_menus, r_headers)
                if is_known and face_top_menus:
                    prev_str = ", ".join(face_top_menus[:2])
                    full_greet = f"안녕하세요, 메가커피입니다. {base_greet} 아니면 이전에 주문하신 {prev_str}로 도와드릴까요?"
                else:
                    full_greet = f"안녕하세요, 메가커피입니다. {base_greet}"
                print(f"🗣️ [AI Greeting]: {full_greet}")
                threading.Thread(target=play_tts_blocking, args=(full_greet, r_headers), daemon=True).start()

            # [감정 기반 오류 확인 멘트]
            if is_in_session:
                current_emotion = ""
                with emotion_data_lock:
                    if latest_emotion_data:
                        current_emotion = latest_emotion_data.get("dominant_emotion", "")
                is_neg = _is_negative_emotion(current_emotion)
                if is_neg and current_emotion != last_emotion_label:
                    negative_emotion_start_ts = now
                    print(f"⚠️ [Emotion] Negative detected: {current_emotion}")
                    # 부정 감정이 감지되면 즉시 오류 확인 멘트 (조건 충족 시)
                    if (
                        now - last_action_ts < 12.0
                        and not order_state.awaiting_confirmation
                        and not tts_playing_event.is_set()
                        and (now - last_emotion_prompt_ts >= EMOTION_PROMPT_COOLDOWN_SEC)
                    ):
                        last_emotion_prompt_ts = now
                        if last_action_summary:
                            prompt = f"방금 {last_action_summary}로 처리했는데, 맞지 않나요? 원하시는 메뉴로 다시 말씀해 주세요."
                        else:
                            prompt = "혹시 주문이 잘못되었나요? 원하시는 메뉴로 다시 말씀해 주세요."
                        threading.Thread(target=play_tts_blocking, args=(prompt, r_headers), daemon=True).start()
                elif not is_neg:
                    negative_emotion_start_ts = 0.0
                last_emotion_label = current_emotion

                if (
                    is_neg
                    and negative_emotion_start_ts
                    and (now - negative_emotion_start_ts >= EMOTION_ISSUE_SEC)
                    and (now - last_action_ts < 12.0)
                    and not order_state.awaiting_confirmation
                    and not tts_playing_event.is_set()
                    and (now - last_emotion_prompt_ts >= EMOTION_PROMPT_COOLDOWN_SEC)
                ):
                    last_emotion_prompt_ts = now
                    prompt = "혹시 주문이 잘못되었나요? 원하시는 메뉴로 다시 말씀해 주세요."
                    threading.Thread(target=play_tts_blocking, args=(prompt, r_headers), daemon=True).start()

            # [세션 종료 - 얼굴 유예 처리]
            elif is_in_session and (now - last_face_detected_ts > SESSION_TIMEOUT_SEC):
                if pending_face_grace_ts == 0.0:
                    pending_face_grace_ts = now
                # 주문 진행 여부에 따라 유예 시간을 다르게 적용
                has_active_order = False
                try:
                    recent_order_activity = (now - last_order_activity_ts) < ORDER_ACTIVE_WINDOW_SEC
                    has_active_order = bool(
                        order_state.pending_menu_name
                        or order_state.awaiting_confirmation
                        or recent_order_activity
                    )
                except Exception:
                    has_active_order = False
                grace_sec = FACE_GRACE_SEC_ACTIVE if has_active_order else FACE_GRACE_SEC_IDLE
                if pending_face_grace_ts and (now - pending_face_grace_ts < grace_sec):
                    # 유예 기간: 얼굴이 다시 잡히면 유지
                    if session_face_cache and latest_face_data:
                        if last_face_detected_ts > pending_face_grace_ts and _face_similarity(latest_face_data, session_face_cache) <= 0.01:
                            pending_face_grace_ts = 0.0
                            continue
                    await asyncio.sleep(0.1)
                    continue
                is_in_session = False
                pending_face_grace_ts = 0.0
                last_order_activity_ts = 0.0
                print("\n" + "="*40)
                print("🔒 얼굴 인식 종료, 세션 종료")
                print("="*40 + "\n")

                tts_stop_event.set()
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
                    "turn_detection": {
                        "type": "server_vad",
                        "silence_duration_ms": 1200} # 300 : 약 0.3초 / 600 : 약 0.6초 / 1000 : 약 1초
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

                    t_start = time.time()
                    cls = classify_text(txt, r_headers)
                    t_nlu = time.time()
                    intent = cls.get("intent", "unknown")
                    issue_reply = _detect_order_issue(txt, intent)
                    if issue_reply:
                        print(f"🧠 [NLU] {intent} | {cls.get('order_items')}")
                        t_tts = time.time()
                        play_tts_blocking(issue_reply, r_headers)
                        t_after_tts = time.time()
                        t_end = time.time()
                        timings = {
                            "nlu": (t_nlu - t_start),
                            "logic": 0.0,
                            "actions": 0.0,
                            "tts": (t_after_tts - t_tts),
                            "total": (t_end - t_start),
                        }
                        print(f"⏱️ [Timing] NLU {timings['nlu']:.3f}s | Logic {timings['logic']:.3f}s | Actions {timings['actions']:.3f}s | TTS {timings['tts']:.3f}s | Total {timings['total']:.3f}s")
                        continue
                    if not _should_process_utterance(txt, intent):
                        print(f"🔇 [Filtered] {txt}")
                        continue
                    print(f"🧠 [NLU] {intent} | {cls.get('order_items')}")
                    
                    top = get_recent_top_menus(os.path.join(current_dir, "order_logs.json"))
                    t_logic = time.time()
                    try:
                        reply, acts = handle_with_classifier(txt, cls, r_headers, top)
                    except Exception as e:
                        print(f"❌ [NLU Worker Error]: {e}")
                        reply, acts = "죄송합니다. 다시 말씀해 주세요.", []
                    t_after_logic = time.time()
                    
                    if acts:
                        for a in acts:
                            send_kiosk_action(a["type"], a["menu_name"], a["quantity"], a["temperature"])
                    checkout_requested = _contains_any(txt, CHECKOUT_HINTS) and not order_state.awaiting_confirmation
                    if checkout_requested:
                        send_kiosk_action("checkout_request", "", 0, "")

                    if reply and not checkout_requested:
                        t_tts = time.time()
                        play_tts_blocking(reply, r_headers)
                        t_after_tts = time.time()
                    else:
                        t_tts = None
                        t_after_tts = None

                    t_end = time.time()
                    timings = {
                        "nlu": (t_nlu - t_start),
                        "logic": (t_after_logic - t_logic),
                        "actions": (t_end - t_after_logic) if not t_tts else (t_tts - t_after_logic),
                        "tts": (t_after_tts - t_tts) if (t_tts and t_after_tts) else 0.0,
                        "total": (t_end - t_start),
                    }
                    print(f"⏱️ [Timing] NLU {timings['nlu']:.3f}s | Logic {timings['logic']:.3f}s | Actions {timings['actions']:.3f}s | TTS {timings['tts']:.3f}s | Total {timings['total']:.3f}s")

            threading.Thread(target=nlu_worker, daemon=True).start()

            async def send_audio():
                while is_running:
                    if not ALLOW_BARGE_IN and tts_playing_event.is_set():
                        await asyncio.sleep(0.1); continue
                    # 얼굴이 감지된 상태에서만 음성 입력 허용
                    if time.time() - last_face_detected_ts > FACE_AUDIO_WINDOW_SEC:
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
                        if d.get("type") == "input_audio_buffer.speech_started":
                            print("🗣️ [말하는 중...]")
                        if d.get("type") == "input_audio_buffer.speech_stopped":
                            print("🤐 [말 끝남]")
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
    if "--build-embeddings" in sys.argv:
        raw_menu = load_menu_data(os.path.join(current_dir, "mega_coffee_menu.json"))
        menu_catalog = MenuCatalog(raw_menu if raw_menu else [])
        rest_headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        cache_path = os.path.join(current_dir, "menu_embeddings.json")
        _build_menu_embedding_cache(rest_headers, cache_path)
        print("✅ [Embeddings] menu_embeddings.json 생성 완료")
        sys.exit(0)
    if "--build-categories" in sys.argv:
        raw_menu = load_menu_data(os.path.join(current_dir, "mega_coffee_menu.json"))
        menu_catalog = MenuCatalog(raw_menu if raw_menu else [])
        rest_headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        cat_path = os.path.join(current_dir, "menu_categories.json")
        _build_menu_category_cache(rest_headers, cat_path)
        print("✅ [Categories] menu_categories.json 생성 완료")
        sys.exit(0)
    if "--build-subcategories" in sys.argv:
        raw_menu = load_menu_data(os.path.join(current_dir, "mega_coffee_menu.json"))
        menu_catalog = MenuCatalog(raw_menu if raw_menu else [])
        rest_headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        subcat_path = os.path.join(current_dir, "menu_subcategories.json")
        _build_menu_subcategory_cache(rest_headers, subcat_path)
        print("✅ [Subcategories] menu_subcategories.json 생성 완료")
        sys.exit(0)

    _load_age_model()
    threading.Thread(target=audio_thread_entry, daemon=True).start()
    threading.Thread(target=socket_thread, daemon=True).start()
    camera_loop_main()
