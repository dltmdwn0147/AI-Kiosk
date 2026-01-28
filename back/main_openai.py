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

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace가 설치되지 않았습니다.")

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

CLASSIFIER_MODEL = os.getenv("OPENAI_CLASSIFIER_MODEL", "gpt-4o-mini")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024
ORDER_LOG_RETENTION_DAYS = 7

latest_face_data = None
latest_emotion_data = None
latest_face_count = 0
face_data_lock = threading.Lock()
emotion_data_lock = threading.Lock()
kiosk_socket = None
socket_lock = threading.Lock()
is_running = True

recent_action_cache = {"text": "", "ts": 0.0}
tts_state = {"text": "", "end_ts": 0.0}
tts_lock = threading.Lock()
tts_playing_event = threading.Event()

known_faces_cache = {"faces": [], "ts": 0.0}

KOR_NUM_MAP = {
    "한": 1, "하나": 1, "한잔": 1, "한 잔": 1,
    "두": 2, "둘": 2, "두잔": 2, "두 잔": 2,
    "세": 3, "셋": 3, "서이": 3, "세잔": 3, "세 잔": 3,
    "네": 4, "넷": 4, "너이": 4, "네잔": 4, "네 잔": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}

# 기존 리스트를 아래 내용으로 교체하세요.
TEMPERATURE_ICE_HINTS = ["아이스", "ice", "차가", "차게", "냉", "콜드", "시원", "아이쓰", "아스크림"]

# "하스", "학", "하시" 등 '핫'으로 들릴 수 있는 오타 추가
TEMPERATURE_HOT_HINTS = [
    "핫", "hot", "따뜻", "뜨거", "맨도롱", "따아", "온", "데운", 
    "하스", "핫스", "학", "하시", "히트", "뜨신"
]

NEGATE_OPTION_HINTS = [
    "옵션 필요없", "옵션 필요 없", "옵션 없어", "옵션 없", "없어", "필요없", "필요 없",
    "그냥", "기본", "선택 안", "선택안", "추가 안", "추가안", "빼줘", "빼", "없애"
]

VARIANT_MODIFIERS = [
    "디카페인", "꿀", "헤이즐넛", "바닐라", "카라멜", "초코", "연유", "시럽", "스테비아",
    "라이트", "제로", "저당", "무가당"
]

ORDER_CONFIRM_HINTS = ["주세요", "주문", "담아", "추가", "할게", "할께", "그걸로", "이걸로", "이대로", "줘", "주문해"]
YES_HINTS = ["네", "응", "맞아", "맞아요", "그래", "그거", "그걸로", "이걸로", "이대로", "좋아", "오케이", "ok", "okay", "확인", "주문해", "주문해줘", "주문할게"]
CANCEL_HINTS = ["취소", "처음부터", "리셋", "초기화", "그만", "안할래", "주문 취소"]

def load_menu_data(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-z가-힣]", "", s)
    return s

def _contains_any(text: str, hints: list) -> bool:
    t = _normalize_text(text)
    for h in hints:
        if _normalize_text(h) in t:
            return True
    return False

def _parse_temperature(text: str) -> str:
    t = _normalize_text(text)
    if any(_normalize_text(h) in t for h in TEMPERATURE_ICE_HINTS):
        return "ICE"
    if any(_normalize_text(h) in t for h in TEMPERATURE_HOT_HINTS):
        return "HOT"
    return ""

def _parse_quantity(text: str) -> int:
    if not text:
        return 1
    t = str(text)
    m = re.search(r"(\d+)\s*(잔|개|杯|컵|주문|개요|개만|잔만)?", t)
    if m:
        try:
            q = int(m.group(1))
            return max(1, q)
        except:
            pass
    for k, v in KOR_NUM_MAP.items():
        if k in t:
            return v
    return 1

def _is_order_like(text: str) -> bool:
    if not text:
        return False
    t = _normalize_text(text)
    return any(_normalize_text(h) in t for h in ORDER_CONFIRM_HINTS)

def _is_yes_like(text: str) -> bool:
    if not text:
        return False
    t = _normalize_text(text)
    return any(_normalize_text(h) in t for h in YES_HINTS)

def _normalize_text_for_echo(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(text)).lower()

def _is_echo_text(stt_text: str) -> bool:
    norm_stt = _normalize_text_for_echo(stt_text)
    if not norm_stt:
        return False
    with tts_lock:
        last_text = _normalize_text_for_echo(tts_state["text"])
        end_ts = tts_state["end_ts"]
    if not last_text:
        return False
    if time.time() - end_ts > 3.0:
        return False
    return norm_stt in last_text or last_text in norm_stt

def _is_valid_transcript(text: str) -> bool:
    if not text:
        return False
    cleaned = str(text).strip()
    return cleaned and cleaned.lower() != "none"

def _post_json(url: str, payload: dict, headers: dict, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _post_bytes(url: str, payload: dict, headers: dict, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

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

def play_tts_blocking(reply: str, rest_headers: dict):
    if not reply:
        return
    tts_playing_event.set()
    with tts_lock:
        tts_state["text"] = reply
        tts_state["end_ts"] = time.time()
    tts_payload = {"model": "tts-1", "voice": "shimmer", "input": reply, "response_format": "wav"}
    try:
        audio_bytes = _post_bytes("https://api.openai.com/v1/audio/speech", tts_payload, rest_headers)
        _play_wav_audio(audio_bytes)
    except Exception as e:
        print(f"❌ [TTS Error]: {e}")
    finally:
        with tts_lock:
            tts_state["end_ts"] = time.time()
        tts_playing_event.clear()

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
    file_name = os.path.join(current_dir, "order_logs.json")
    cutoff = datetime.now() - timedelta(days=ORDER_LOG_RETENTION_DAYS)
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "menu": menu_name,
        "face_landmarks": face_data if face_data else "None",
        "emotion": emotion_data if emotion_data else "None",
        "face_count": latest_face_count,
    }
    try:
        logs = []
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
            filtered = []
            for entry in logs:
                ts = entry.get("timestamp", "")
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                if dt >= cutoff:
                    filtered.append(entry)
            logs = filtered
        logs.append(log_entry)
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ [Order Log] 저장 실패: {e}")

def get_recent_top_menus(file_path: str, top_k: int = 3, days: int = 30) -> list:
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except:
        return []
    cutoff = datetime.now() - timedelta(days=days)
    counts = {}
    for entry in logs if isinstance(logs, list) else []:
        ts = entry.get("timestamp", "")
        menu = str(entry.get("menu", "")).strip()
        if not menu:
            continue
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except:
            continue
        if dt < cutoff:
            continue
        counts[menu] = counts.get(menu, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [m for m, _ in ranked[:top_k]]

def send_kiosk_action(action_type: str, menu_name: str, quantity: int = 1, temperature: str = ""):
    if not action_type:
        return "주문 실패: 타입 없음"
    payload_dict = {"type": action_type, "menu_name": menu_name, "quantity": int(quantity or 1), "temperature": temperature or ""}
    payload_json = json.dumps(payload_dict, ensure_ascii=False)
    now = time.time()
    if payload_json == recent_action_cache["text"] and now - recent_action_cache["ts"] < 2.0:
        return f"중복 차단: {payload_json}"
    msg = "주문 실패"
    with socket_lock:
        if kiosk_socket:
            try:
                payload = (payload_json + "\n").encode("utf-8")
                print(f"   └─ 📡 키오스크 전송: {payload_dict}")
                kiosk_socket.sendall(payload)
                msg = "주문 성공"
            except Exception as e:
                print(f"   └─ ❌ 키오스크 전송 실패: {e}")
    recent_action_cache["text"] = payload_json
    recent_action_cache["ts"] = now
    return f"{msg}: {payload_json}"

def _load_known_faces(file_name: str = None, ttl_sec: int = 10) -> list:
    if file_name is None:
        file_name = os.path.join(current_dir, "order_logs.json")
    now = time.time()
    if now - known_faces_cache["ts"] < ttl_sec and known_faces_cache["faces"]:
        return known_faces_cache["faces"]
    faces = []
    if os.path.exists(file_name):
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                logs = json.load(f)
            for entry in logs:
                face = entry.get("face_landmarks")
                if isinstance(face, list) and face:
                    faces.append(face)
        except Exception:
            faces = []
    known_faces_cache["faces"] = faces
    known_faces_cache["ts"] = now
    return faces

def _face_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    total = 0.0
    for la, lb in zip(a, b):
        dx = float(la.get("x", 0)) - float(lb.get("x", 0))
        dy = float(la.get("y", 0)) - float(lb.get("y", 0))
        dz = float(la.get("z", 0)) - float(lb.get("z", 0))
        total += dx * dx + dy * dy + dz * dz
    return total / max(1, len(a))

def _is_known_face(face_data: list, threshold: float = 0.01) -> bool:
    if not face_data:
        return False
    known_faces = _load_known_faces()
    if not known_faces:
        return False
    for known in known_faces:
        if _face_similarity(face_data, known) <= threshold:
            return True
    return False

class MenuCatalog:
    def __init__(self, raw_menu):
        self.menus = []
        # JSON 구조 파싱 (유연하게)
        if isinstance(raw_menu, dict):
            if isinstance(raw_menu.get("menus"), list):
                self.menus = raw_menu["menus"]
            elif isinstance(raw_menu.get("data"), list):
                self.menus = raw_menu["data"]
            else:
                for k in ["menu", "items", "list"]:
                    if isinstance(raw_menu.get(k), list):
                        self.menus = raw_menu[k]
                        break
        elif isinstance(raw_menu, list):
            self.menus = raw_menu
        
        self.menus = [m for m in self.menus if isinstance(m, dict)]
        
        # [검색 최적화] 검색용 인덱스 생성
        # key: 정규화된 이름 (공백제거), value: 메뉴 객체 리스트
        self._name_index = {}
        for m in self.menus:
            name = str(m.get("menu_name", "")).strip()
            if not name:
                continue
            norm_name = self._normalize(name) # "아메리카노"
            self._name_index.setdefault(norm_name, []).append(m)

    def _normalize(self, text: str) -> str:
        # 공백 제거 및 소문자화, 특수문자 제거
        s = str(text).lower()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[^0-9a-z가-힣]", "", s)
        return s

    def _remove_prefixes(self, text: str) -> str:
        # 온도/상태 관련 수식어 제거 (검색 정확도 향상용)
        prefixes = ["아이스", "핫", "따뜻한", "따뜻", "뜨거운", "시원한", "차가운", "냉", "hot", "ice"]
        norm = self._normalize(text)
        for p in prefixes:
            norm = norm.replace(self._normalize(p), "")
        return norm

    def resolve_menu(self, user_query: str):
        """
        사용자 발화(user_query)를 바탕으로 DB에서 가장 정확한 메뉴를 찾는다.
        return: (찾은 메뉴 객체, 필요 슬롯, 후보 리스트)
        """
        if not user_query:
            return None, "menu", []

        # 1단계: 있는 그대로 정확한 매칭 시도 ("메가리카노" -> "메가리카노")
        norm_query = self._normalize(user_query)
        if norm_query in self._name_index:
            return self._name_index[norm_query][0], "", []

        # 2단계: 수식어(아이스/핫) 떼고 매칭 시도 ("아이스 아메리카노" -> "아메리카노")
        clean_query = self._remove_prefixes(user_query)
        if clean_query and clean_query != norm_query:
            if clean_query in self._name_index:
                # 찾았다! (예: "아메리카노")
                return self._name_index[clean_query][0], "", []

        # 3단계: 포함 관계 검색 (Fuzzy Search) - "바닐라" 라고만 했을 때
        candidates = []
        for db_name, menus in self._name_index.items():
            # DB이름이 쿼리에 포함되거나, 쿼리가 DB이름에 포함될 때
            if (clean_query and clean_query in db_name) or (db_name in clean_query):
                candidates.extend(menus)
        
        if candidates:
            # 후보 중 가장 짧은 이름(기본 메뉴)을 우선시 (예: "라떼" 검색 시 "카페라떼" > "고구마라떼")
            candidates.sort(key=lambda x: len(x.get("menu_name", "")))
            return candidates[0], "", []

        return None, "menu", []

def _extract_default_option_ids(menu: dict) -> list:
    option_ids = []
    options = menu.get("options", [])
    if not isinstance(options, list):
        return option_ids
    for ok in options:
        if not isinstance(ok, dict):
            continue
        details = ok.get("option_details", [])
        if not isinstance(details, list):
            continue
        default = None
        for d in details:
            if not isinstance(d, dict):
                continue
            if str(d.get("option_name", "")).strip() == "선택x":
                default = d
                break
        if default is None and details:
            default = details[0] if isinstance(details[0], dict) else None
        if default and "option_id" in default:
            option_ids.append(default["option_id"])
    return option_ids

def _match_option_ids_from_text(menu: dict, user_text: str) -> list:
    selected_ids = []
    options = menu.get("options", [])
    if not isinstance(options, list):
        return selected_ids
    t = _normalize_text(user_text)
    want_default = any(_normalize_text(h) in t for h in map(_normalize_text, NEGATE_OPTION_HINTS))
    for ok in options:
        if not isinstance(ok, dict):
            continue
        details = ok.get("option_details", [])
        if not isinstance(details, list) or not details:
            continue
        if want_default:
            for d in details:
                if isinstance(d, dict) and str(d.get("option_name", "")).strip() == "선택x":
                    if "option_id" in d:
                        selected_ids.append(d["option_id"])
                    break
            else:
                d0 = details[0] if isinstance(details[0], dict) else None
                if d0 and "option_id" in d0:
                    selected_ids.append(d0["option_id"])
            continue
        matched = None
        for d in details:
            if not isinstance(d, dict):
                continue
            oname = str(d.get("option_name", "")).strip()
            if oname and _normalize_text(oname) in t and oname != "선택x":
                matched = d
                break
        if matched and "option_id" in matched:
            selected_ids.append(matched["option_id"])
        else:
            for d in details:
                if isinstance(d, dict) and str(d.get("option_name", "")).strip() == "선택x":
                    if "option_id" in d:
                        selected_ids.append(d["option_id"])
                    break
            else:
                d0 = details[0] if isinstance(details[0], dict) else None
                if d0 and "option_id" in d0:
                    selected_ids.append(d0["option_id"])
    return selected_ids

def _has_any_named_option_in_text(menu: dict, user_text: str) -> bool:
    options = menu.get("options", [])
    if not isinstance(options, list) or not options:
        return False
    tnorm = _normalize_text(user_text)
    for ok in options:
        if not isinstance(ok, dict):
            continue
        details = ok.get("option_details", [])
        if not isinstance(details, list):
            continue
        for d in details:
            if not isinstance(d, dict):
                continue
            oname = str(d.get("option_name", "")).strip()
            if oname and oname != "선택x" and _normalize_text(oname) in tnorm:
                return True
    return False

def _get_menu_price(menu: dict):
    for k in ["price", "menu_price", "menuPrice", "cost", "amount", "menu_amount", "menu_cost"]:
        if k in menu:
            try:
                v = menu.get(k)
                if isinstance(v, (int, float)):
                    return int(v)
                if isinstance(v, str):
                    digits = re.sub(r"[^0-9]", "", v)
                    if digits:
                        return int(digits)
            except:
                pass
    return None

class OrderState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pending_slot = ""
        self.pending_menu_name = ""
        self.pending_temp = ""
        self.menu = None
        self.quantity = 1
        self.temperature = ""
        self.option_ids = []
        self.last_question_slot = ""
        self.last_question_ts = 0.0
        self.awaiting_confirmation = False
        self.confirmation_ts = 0.0

    def set_menu(self, menu: dict):
        self.menu = menu
        self.temperature = str(menu.get("menu_temperature", "")).strip() or self.temperature
        self.pending_menu_name = str(menu.get("menu_name", "")).strip()
        self.option_ids = _extract_default_option_ids(menu)

    def can_finalize(self):
        if not self.menu:
            return False
        mt = str(self.menu.get("menu_temperature", "")).strip()
        if mt in ["ICE", "HOT"] and not self.temperature:
            return False
        return True

menu_catalog = None
order_state = OrderState()

def _build_reply_for_slot(slot: str, base_menu_name: str = "") -> str:
    if slot == "menu":
        return "원하시는 메뉴를 말씀해 주세요."
    if slot == "temperature":
        if base_menu_name:
            return f"{base_menu_name}는 핫과 아이스 중 어떤 걸로 드릴까요?"
        return "온도는 핫으로 드릴까요, 아이스로 드릴까요?"
    if slot == "options":
        return "추가 옵션은 필요하신가요? 필요 없으시면 '기본으로'라고 말씀해 주세요."
    return "무엇을 도와드릴까요?"

def _finalize_action():
    m = order_state.menu
    if not m:
        return None
    menu_name = str(m.get("menu_name", "")).strip()
    temp = order_state.temperature or str(m.get("menu_temperature", "")).strip()
    qty = int(order_state.quantity or 1)
    return {"type": "add", "menu_name": menu_name, "quantity": max(1, qty), "temperature": temp if temp in ["ICE", "HOT"] else ""}

def _handle_pending_slot(user_text: str):
    if order_state.pending_slot == "temperature":
        temp = _parse_temperature(user_text)
        if temp:
            order_state.temperature = temp
            order_state.pending_slot = ""
            order_state.last_question_slot = ""
            return True
        return False
    if order_state.pending_slot == "menu":
        order_state.pending_slot = ""
        order_state.last_question_slot = ""
        return True
    if order_state.pending_slot == "options":
        if _contains_any(user_text, NEGATE_OPTION_HINTS):
            order_state.option_ids = _extract_default_option_ids(order_state.menu or {})
            order_state.pending_slot = ""
            order_state.last_question_slot = ""
            return True
        if order_state.menu:
            order_state.option_ids = _match_option_ids_from_text(order_state.menu, user_text)
            order_state.pending_slot = ""
            order_state.last_question_slot = ""
            return True
        order_state.pending_slot = ""
        order_state.last_question_slot = ""
        return True
    return False

def handle_user_text_order_flow(user_text: str):
    global order_state, menu_catalog
    user_text = (user_text or "").strip()
    if not user_text:
        return "", []

    order_state.quantity = _parse_quantity(user_text) or order_state.quantity

    if order_state.awaiting_confirmation and order_state.can_finalize():
        now = time.time()
        if now - order_state.confirmation_ts <= 30.0:
            candidates = menu_catalog.find_best_name_candidates(user_text)
            if not candidates and (_is_yes_like(user_text) or _is_order_like(user_text)):
                action = _finalize_action()
                reply = f"네, {action['temperature']} {action['menu_name']} {action['quantity']}잔 준비해 드리겠습니다."
                order_state.reset()
                return reply, [action]
        else:
            order_state.awaiting_confirmation = False

    if order_state.pending_slot:
        ok = _handle_pending_slot(user_text)
        if not ok:
            slot = order_state.pending_slot
            base_name = order_state.pending_menu_name or ""
            now = time.time()
            if order_state.last_question_slot == slot and now - order_state.last_question_ts < 12.0:
                order_state.pending_slot = ""
            else:
                order_state.last_question_slot = slot
                order_state.last_question_ts = now
                return _build_reply_for_slot(slot, base_name), []
        if order_state.menu and order_state.pending_slot != "options":
            if _contains_any(user_text, NEGATE_OPTION_HINTS):
                order_state.option_ids = _extract_default_option_ids(order_state.menu)
            else:
                order_state.option_ids = _match_option_ids_from_text(order_state.menu, user_text)
        if order_state.can_finalize() and (_is_order_like(user_text) or _is_yes_like(user_text)):
            action = _finalize_action()
            reply = f"네, {action['temperature']} {action['menu_name']} {action['quantity']}잔 준비해 드리겠습니다."
            order_state.reset()
            return reply, [action]
        if order_state.can_finalize():
            action = _finalize_action()
            order_state.awaiting_confirmation = True
            order_state.confirmation_ts = time.time()
            reply = f"{action['temperature']} {action['menu_name']} {action['quantity']}잔 맞으실까요? 맞으면 '주문해줘'라고 말씀해 주세요."
            return reply, []
        missing = "menu"
        if not order_state.menu:
            missing = "menu"
        elif not order_state.temperature and str(order_state.menu.get("menu_temperature", "")).strip() in ["ICE", "HOT"]:
            missing = "temperature"
        else:
            missing = ""
        if missing:
            order_state.pending_slot = missing
            order_state.last_question_slot = missing
            order_state.last_question_ts = time.time()
            return _build_reply_for_slot(missing, order_state.pending_menu_name), []

    temp_in_text = _parse_temperature(user_text)
    resolved_menu, need_slot, need_data = menu_catalog.resolve_menu(user_text)

    if need_slot == "menu":
        order_state.pending_slot = "menu"
        order_state.last_question_slot = "menu"
        order_state.last_question_ts = time.time()
        return _build_reply_for_slot("menu"), []

    if need_slot == "temperature":
        base_name = need_data[0] if need_data else ""
        order_state.pending_slot = "temperature"
        order_state.pending_menu_name = base_name
        order_state.last_question_slot = "temperature"
        order_state.last_question_ts = time.time()
        order_state.pending_temp = ""
        order_state.menu = None
        order_state.temperature = ""
        order_state.quantity = _parse_quantity(user_text) or 1
        order_state.awaiting_confirmation = False
        return _build_reply_for_slot("temperature", base_name), []

    if resolved_menu:
        order_state.set_menu(resolved_menu)
        order_state.awaiting_confirmation = False
        if temp_in_text:
            order_state.temperature = temp_in_text
        has_options = isinstance(order_state.menu.get("options", []), list) and bool(order_state.menu.get("options"))
        if has_options:
            if _contains_any(user_text, NEGATE_OPTION_HINTS):
                order_state.option_ids = _extract_default_option_ids(order_state.menu)
            elif _has_any_named_option_in_text(order_state.menu, user_text):
                order_state.option_ids = _match_option_ids_from_text(order_state.menu, user_text)
            else:
                order_state.option_ids = _extract_default_option_ids(order_state.menu)

        if order_state.can_finalize() and (_is_order_like(user_text) or _is_yes_like(user_text)):
            action = _finalize_action()
            reply = f"네, {action['temperature']} {action['menu_name']} {action['quantity']}잔 준비해 드리겠습니다."
            order_state.reset()
            return reply, [action]

        if order_state.can_finalize():
            action = _finalize_action()
            order_state.awaiting_confirmation = True
            order_state.confirmation_ts = time.time()
            reply = f"{action['temperature']} {action['menu_name']} {action['quantity']}잔 맞으실까요? 맞으면 '주문해줘'라고 말씀해 주세요."
            return reply, []

    order_state.pending_slot = "menu"
    order_state.last_question_slot = "menu"
    order_state.last_question_ts = time.time()
    order_state.awaiting_confirmation = False
    return "원하시는 메뉴를 다시 한번 말씀해 주실래요?", []

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
            "context": {
                "type": "object",
                "properties": {
                    "weather": {"type": "string", "enum": ["cold", "hot", ""]},
                    "throat": {"type": "string", "enum": ["dry", "sore", ""]},
                    "caffeine": {"type": "string", "enum": ["ANY", "LOW", "NONE", "HIGH", ""]},
                    "sweetness": {"type": "string", "enum": ["ANY", "LOW", "MID", "HIGH", ""]}
                },
                "additionalProperties": False,
                "required": ["weather", "throat", "caffeine", "sweetness"]
            },
            "confidence": {"type": "number"}
        },
        "required": ["intent", "order_items", "context", "confidence"],
        "additionalProperties": False
    }

    prompt = (
        "너는 한국 메가커피 키오스크 입력 문장을 분류하는 분류기다.\n"
        "반드시 JSON만 출력한다.\n"
        "\n"
        "[Intent 가이드]\n"
        "- order: 신규 주문\n"
        "- update_order: 수량 변경/정정. **'~하고 ~해줘' 같은 연결 문장도 모두 추출해야 함**.\n"
        "  (예: '아아는 1잔으로 바꾸고 라떼는 2잔으로 해줘' -> intent: update_order, order_items: [{menu: '아아', qty: 1}, {menu: '라떼', qty: 2}])\n"
        "- increase/decrease: 추가/감소\n"
        "- remove_item: 삭제\n"
        "- cancel/reset: 전체 초기화\n"
        "\n"
        "[추출 규칙]\n"
        "1. order_items: 사용자가 언급한 **모든 메뉴**를 빠짐없이 배열로 추출하라.\n"
        "2. temperature: 문맥상 명시되었거나 메뉴 특성상 유추 가능하면 HOT/ICE, 모르면 ANY.\n"
        "3. quantity: 없으면 기본 1.\n"
    )

    payload = {
        "model": CLASSIFIER_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "classification_task",
                "strict": True,
                "schema": schema
            }
        },
        "max_completion_tokens": 512
    }

    try:
        r = _post_json("https://api.openai.com/v1/chat/completions", payload, rest_headers, timeout=10)
        if "choices" in r and len(r["choices"]) > 0:
            content = r["choices"][0]["message"]["content"]
            # 디버그: GPT가 뭘 뽑았는지 콘솔에 출력 (필수)
            parsed = json.loads(content)
            print(f"🧠 [GPT Debug] Intent: {parsed.get('intent')}, Items: {parsed.get('order_items')}")
            return parsed
        return {"intent": "unknown", "order_items": [], "context": {"weather": "", "throat": "", "caffeine": "", "sweetness": ""}, "confidence": 0.0}
    except Exception as e:
        print(f"❌ [Classifier Error]: {e}")
        return {"intent": "unknown", "order_items": [], "context": {"weather": "", "throat": "", "caffeine": "", "sweetness": ""}, "confidence": 0.0}

def chat_fallback(user_text: str, rest_headers: dict, extra_hint: str = "") -> str:
    # [수정됨] 거짓말 방지(Hallucination Prevention) 프롬프트 강화
    sys_prompt = (
        "너는 메가커피 키오스크 음성 도우미다.\n"
        "사용자의 질문에 짧고 친절하게 답하고, 필요하면 메뉴 주문으로 자연스럽게 유도한다.\n"
        "가격은 시스템이 별도로 처리할 수 있으니, 모르면 '해당 메뉴를 말씀해 주시면 확인해 드릴게요'라고 한다.\n"
        "\n"
        "[매우 중요 - 거짓말 금지]\n"
        "사용자가 말한 단어가 메뉴판에 없는 생소한 단어(예: 오타, 잘못된 발음)라면, 절대 아는 척하거나 맛 설명을 지어내지 마라.\n"
        "차라리 '죄송하지만 잘 못 알아들었습니다. 다시 말씀해 주시겠어요?'라고 되물어라.\n"
        "너무 길게 말하지 말고 1~2문장으로 답한다."
    )
    
    user_content = user_text if not extra_hint else f"{user_text}\n\n(추가 힌트)\n{extra_hint}"
    
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content}
        ],
        "max_completion_tokens": 200
    }
    
    try:
        r = _post_json("https://api.openai.com/v1/chat/completions", payload, rest_headers, timeout=10)
        
        if "choices" in r and len(r["choices"]) > 0:
            return r["choices"][0]["message"]["content"].strip()
            
    except Exception as e:
        print(f"❌ [Chat Error]: {e}")
    
    return "죄송합니다. 다시 한번 말씀해 주시겠어요?"

def recommend_from_catalog(user_text: str, cls: dict, top_from_logs: list):
    t = _normalize_text(user_text)
    temp = (cls.get("temperature") or "").strip()
    if not temp or temp == "ANY":
        temp = _parse_temperature(user_text) or ""
    want_hot = (temp == "HOT") or ("cold" == (cls.get("context", {}) or {}).get("weather")) or ("따뜻" in user_text)
    want_ice = (temp == "ICE") or ("hot" == (cls.get("context", {}) or {}).get("weather")) or ("시원" in user_text)

    candidates = []
    for m in menu_catalog.menus if menu_catalog else []:
        name = str(m.get("menu_name", "")).strip()
        if not name:
            continue
        mt = str(m.get("menu_temperature", "")).strip()
        if want_hot and mt and mt != "HOT":
            continue
        if want_ice and mt and mt != "ICE":
            continue
        candidates.append(name)

    if ("차" in user_text) or ("티" in user_text) or ("throat" in (cls.get("context", {}) or {}) and (cls.get("context", {}) or {}).get("throat") in ["dry", "sore"]):
        tea_like = [n for n in candidates if ("티" in n or "차" in n)]
        if tea_like:
            candidates = tea_like

    uniq = []
    seen = set()
    for n in candidates:
        if n in seen:
            continue
        seen.add(n)
        uniq.append(n)

    picked = []
    for m in top_from_logs[:3]:
        if m in seen and m not in picked:
            picked.append(m)

    for n in uniq:
        if len(picked) >= 3:
            break
        if n not in picked:
            picked.append(n)

    if not picked:
        return "원하시는 메뉴를 말씀해 주시면 추천해 드릴게요.", []

    txt = " / ".join(picked[:3])
    return f"추천드리자면 {txt} 어떠세요? 원하시는 메뉴를 말씀해 주세요.", picked[:3]

def answer_price(user_text: str, cls: dict):
    # 1. GPT가 추출한 'menu_query'(예: "디카페인 티라미수 라떼")로 먼저 검색
    query = cls.get("menu_query", "").strip()
    target_text = query if query else user_text
    
    print(f"🔍 [Price Search] 검색어: '{target_text}'")

    # 시도 1: 있는 그대로 검색
    resolved_menu, _, _ = menu_catalog.resolve_menu(target_text)
    
    # 시도 2: 검색 실패 시, '디카페인/아이스/라떼' 같은 수식어를 제거하고 핵심 단어로 재검색
    if not resolved_menu:
        # 수식어 제거 (정규식으로 흔한 수식어 삭제)
        cleaned_text = re.sub(r"(디카페인|아이스|핫|따뜻한|시원한|차가운|뜨거운)", "", target_text).strip()
        if cleaned_text and cleaned_text != target_text:
            print(f"   └─ 재검색(수식어 제거): '{cleaned_text}'")
            resolved_menu, _, _ = menu_catalog.resolve_menu(cleaned_text)

    # 시도 3: 그래도 없으면 문장 전체(user_text)에서 검색 (최후의 수단)
    if not resolved_menu and target_text != user_text:
        print(f"   └─ 재검색(문장 전체): '{user_text}'")
        resolved_menu, _, _ = menu_catalog.resolve_menu(user_text)

    # --- 결과 처리 ---
    if not resolved_menu:
        return "어떤 메뉴 가격을 확인해 드릴까요? 메뉴명을 정확히 말씀해 주세요.", []
    
    name = str(resolved_menu.get("menu_name", "")).strip()
    price = _get_menu_price(resolved_menu)
    
    if price is None:
        return f"죄송합니다. {name}의 가격 정보를 찾지 못했어요.", []
    
    return f"{name}의 가격은 {price:,}원입니다.", []

def handle_with_classifier(user_text: str, cls: dict, rest_headers: dict, top_menus: list):
    intent = (cls.get("intent") or "unknown").strip()
    conf = float(cls.get("confidence") or 0.0)
    order_items = cls.get("order_items", [])
    
    # [1] 전체 취소
    if intent in ["cancel", "reset"]:
        order_state.reset()
        return "네, 주문을 취소하고 처음부터 도와드릴게요. 원하시는 메뉴를 말씀해 주세요.", []

    # [2] 다중 메뉴 처리 (핵심 로직)
    if intent in ["order", "increase", "decrease", "remove_item", "update_order"]:
        if not order_items:
            # 메뉴 언급 없이 명령만 있는 경우
            if intent == "increase": return "어떤 메뉴를 추가해 드릴까요?", []
            if intent == "decrease": return "어떤 메뉴를 줄여드릴까요?", []
            if intent == "remove_item": return "어떤 메뉴를 삭제해 드릴까요?", []
            if intent == "update_order": return "어떤 메뉴를 변경해 드릴까요?", []
            return handle_user_text_order_flow(user_text)

        final_actions = []
        reply_parts = []
        
        for item in order_items:
            raw_query = item.get("menu_query", "").strip()
            gpt_qty = int(item.get("quantity") or 1)
            gpt_temp = item.get("temperature", "") # GPT가 파악한 온도 (HOT/ICE/ANY)

            if not raw_query: continue

            # [메뉴 검색] Catalog가 수식어를 떼고 '본질적인 이름'을 찾아줍니다.
            resolved, _, _ = menu_catalog.resolve_menu(raw_query)
            
            if not resolved:
                print(f"⚠️ [Menu Not Found] '{raw_query}' 검색 실패")
                continue 
                
            # DB에 있는 정확한 메뉴명 (예: "아메리카노", "바닐라라떼")
            target_menu_name = str(resolved.get("menu_name", "")).strip()
            
            # [온도 결정 로직]
            # 1순위: GPT가 "ICE"라고 명확히 분류했으면 그걸 쓴다.
            # 2순위: 사용자가 "아이스", "차가운"이라고 말했으면 그걸 쓴다.
            # 3순위: 메뉴 자체의 기본 속성(예: 스무디는 무조건 ICE)을 따른다.
            
            final_temp = ""
            
            # (1) GPT 판단 우선
            if gpt_temp in ["HOT", "ICE"]:
                final_temp = gpt_temp
            # (2) 사용자 발화에서 힌트 찾기 (GPT가 놓쳤을 경우 대비)
            elif any(x in raw_query for x in ["아이스", "차가운", "냉", "시원"]):
                final_temp = "ICE"
            elif any(x in raw_query for x in ["핫", "따뜻", "뜨거운"]):
                final_temp = "HOT"
            # (3) 메뉴판 기본값 (DB에 고정된 온도가 있다면)
            else:
                db_temp = str(resolved.get("menu_temperature", "")).strip().upper()
                if db_temp in ["HOT", "ICE"]:
                    final_temp = db_temp
                else:
                    # 아예 모르면 ICE (한국인 국룰) 또는 HOT? -> 여기선 빈값으로 둬서 Unity가 처리하거나 기본 ICE
                    final_temp = "ICE" 

            # Action 생성
            action_data = {
                "menu_name": target_menu_name,
                "quantity": gpt_qty,
                "temperature": final_temp
            }

            if intent == "order":
                final_actions.append({**action_data, "type": "add"})
                reply_parts.append(f"{final_temp} {target_menu_name} {gpt_qty}잔")
            elif intent == "increase":
                final_actions.append({**action_data, "type": "inc"})
                reply_parts.append(f"{target_menu_name} {gpt_qty}잔 더")
            elif intent == "decrease":
                final_actions.append({**action_data, "type": "dec"})
                reply_parts.append(f"{target_menu_name} {gpt_qty}잔 빼고")
            elif intent == "remove_item":
                # 삭제는 온도가 중요하지 않지만 매칭을 위해 보냄
                final_actions.append({**action_data, "type": "remove", "quantity": 0})
                reply_parts.append(f"{target_menu_name} 삭제하고")
            elif intent == "update_order":
                final_actions.append({**action_data, "type": "set"})
                reply_parts.append(f"{target_menu_name} {gpt_qty}잔으로 변경하고")

        if not final_actions:
            return "말씀하신 메뉴를 메뉴판에서 찾을 수 없었어요. 정확한 메뉴명을 말씀해 주시겠어요?", []

        # 응답 메시지 정리
        full_reply = ", ".join(reply_parts)
        if intent in ["order", "increase"]:
             full_reply += " 담아드리겠습니다."
        elif intent == "decrease":
             full_reply = full_reply.rstrip(" 빼고") + " 빼드렸습니다."
        elif intent == "remove_item":
             full_reply = full_reply.rstrip(" 삭제하고") + " 삭제했습니다."
        elif intent == "update_order":
             full_reply = full_reply.rstrip(" 변경하고") + " 변경해 드렸습니다."

        return f"네, {full_reply}", final_actions

    # [3] 확인
    if intent == "confirm":
        if order_state.can_finalize():
            action = _finalize_action()
            reply = f"네, {action['temperature']} {action['menu_name']} {action['quantity']}잔 준비해 드리겠습니다."
            order_state.reset()
            return reply, [action]
        return "확인할 주문이 없어요.", []

    # [4] 기타
    if intent == "price" and conf >= 0.55: return answer_price(user_text, cls)
    if intent == "recommend" and conf >= 0.55: return recommend_from_catalog(user_text, cls, top_menus)[0], []
    if intent == "chitchat" and conf >= 0.55: return chat_fallback(user_text, rest_headers), []

    if intent == "unknown" and conf < 0.55:
        if _contains_any(user_text, ["얼마", "가격", "price", "원"]): return answer_price(user_text, cls)
        if _contains_any(user_text, ["취소", "빼줘", "지워"]): return "어떤 메뉴를 취소해 드릴까요?", []
        if _contains_any(user_text, ["바꿔", "정정", "변경"]): return "어떤 메뉴를 어떻게 변경해 드릴까요?", []
        return handle_user_text_order_flow(user_text)

    return handle_user_text_order_flow(user_text)

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
            options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=10, min_face_detection_confidence=0.5, running_mode=vision.RunningMode.IMAGE)
            face_landmarker = vision.FaceLandmarker.create_from_options(options)
    except:
        face_landmarker = None

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
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    detection_result = face_landmarker.detect(mp_image)
                except:
                    detection_result = None

                detected_faces_count = 0
                closest_face_landmarks = None
                max_face_area = 0
                closest_face_bbox = None

                if detection_result and detection_result.face_landmarks:
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
                        is_known = _is_known_face(data)
                        known_text = "Known: True" if is_known else "Known: False"
                        cv2.putText(image, known_text, (cx, max(20, cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_known else (0, 0, 255), 2)

                        if DEEPFACE_AVAILABLE and frame_count % 15 == 0:
                            face_crop = image_rgb[cy:cy + ch, cx:cx + cw]
                            try:
                                res = DeepFace.analyze(img_path=face_crop, actions=["emotion"], detector_backend="skip", enforce_detection=False, silent=True)
                                if isinstance(res, list):
                                    res = res[0]
                                with emotion_data_lock:
                                    latest_emotion_data = {"dominant_emotion": res.get("dominant_emotion", "Neutral")}
                            except:
                                pass
                else:
                    with face_data_lock:
                        latest_face_data = None
                        latest_face_count = 0

                cv2.putText(image, f"Faces: {latest_face_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if latest_emotion_data:
                    cv2.putText(image, f"Emotion: {latest_emotion_data.get('dominant_emotion')}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Kiosk Eyes (Server)", image)
            if cv2.waitKey(1) & 0xFF == 27:
                is_running = False
                break
            if cv2.getWindowProperty("Kiosk Eyes (Server)", cv2.WND_PROP_VISIBLE) < 1:
                is_running = False
                break
    finally:
        try:
            if face_landmarker:
                face_landmarker.close()
        except:
            pass
        cap.release()
        cv2.destroyAllWindows()

def audio_thread_entry():
    asyncio.run(audio_loop_async())

async def audio_loop_async():
    global is_running, menu_catalog

    raw_menu = load_menu_data(os.path.join(current_dir, "mega_coffee_menu.json"))
    if raw_menu is None:
        print("❌ mega_coffee_menu.json 로드 실패")
        return
    menu_catalog = MenuCatalog(raw_menu)

    stt_queue = queue.Queue()
    worker_stop = threading.Event()

    system_instruction = """
[역할]
당신은 대한민국 메가커피 키오스크의 '청각 인터페이스(Hearing Interface)'입니다.
당신의 유일한 임무는 사용자의 한국어 발화를 텍스트로 정확하게 받아쓰는(Transcribe) 것입니다.

[받아쓰기 핵심 가이드]
1. 있는 그대로 적으세요: 사용자가 사투리, 줄임말, 비표준어를 사용해도 문법을 고치지 말고 들리는 발음 그대로 한글로 적으세요.
2. 핵심 용어 주의:
   - 숫자 방언: "서이"(3), "너이"(4)
   - 제주 방언: "맨도롱"(따뜻한)
   - 메뉴 줄임말: "아아", "따아", "라떼", "샷추가"
3. 잡음 무시: 배경 소음은 무시하고 사람의 목소리에만 집중하세요.
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
                    "input_audio_transcription": {"model": "whisper-1", "language": "ko", "prompt": "한국어, 제주방언, 사투리"},
                    "modalities": ["text"],
                    "turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 600},
                },
            }
            await ws.send(json.dumps(session_update))

            order_log_path = os.path.join(current_dir, "order_logs.json")
            top_menus = get_recent_top_menus(order_log_path, top_k=3, days=30)
            if top_menus:
                reco_text = " / ".join(top_menus)
                greet_text = f"안녕하세요, 메가커피입니다. 이전에 자주 주문하신 {reco_text} 어떠세요? 원하시는 메뉴를 말씀해 주세요."
            else:
                greet_text = "안녕하세요, 메가커피입니다. 원하시는 메뉴를 말씀해 주세요."

            print(f"🗣️ [AI Reply]: {greet_text}")
            play_tts_blocking(greet_text, rest_headers)

            def nlu_worker():
                last_text = ""
                last_ts = 0.0
                while not worker_stop.is_set():
                    try:
                        user_text = stt_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    now = time.time()
                    if user_text == last_text and now - last_ts < 1.5:
                        continue
                    last_text = user_text
                    last_ts = now

                    try:
                        cls = classify_text(user_text, rest_headers)
                    except:
                        cls = {"intent": "unknown", "menu_query": "", "temperature": "", "quantity": 1, "context": {"weather": "", "throat": "", "caffeine": "", "sweetness": ""}, "confidence": 0.0}

                    reply, actions = handle_with_classifier(user_text, cls, rest_headers, top_menus)

                    if actions:
                        for action in actions:
                            act_type = action.get("type")
                            act_menu = action.get("menu_name", "")
                            act_qty = action.get("quantity", 1)
                            act_temp = action.get("temperature", "")
                            if act_type == "add":
                                save_order_log(act_menu, latest_face_data, latest_emotion_data)
                                send_kiosk_action(act_type, act_menu, act_qty, act_temp)
                            else:
                                send_kiosk_action(act_type, act_menu, act_qty, act_temp)

                    if reply:
                        print(f"🗣️ [AI Reply]: {reply}")
                        play_tts_blocking(reply, rest_headers)

            worker_thread = threading.Thread(target=nlu_worker, daemon=True)
            worker_thread.start()

            async def send_audio():
                while is_running:
                    try:
                        if tts_playing_event.is_set():
                            await asyncio.to_thread(mic_stream.read, CHUNK, exception_on_overflow=False)
                            await asyncio.sleep(0)
                            continue
                        data = await asyncio .to_thread(mic_stream.read, CHUNK, exception_on_overflow=False)
                        
                        # TTS 재생 중이면 서버로 오디오를 보내지 않음 (에코 방지 핵심)
                        if tts_playing_event.is_set():
                            await asyncio.sleep(0.01)
                            continue

                        base64_audio = base64.b64encode(data).decode("utf-8")
                        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64_audio}))
                        await asyncio.sleep(0)
                    except Exception as e:
                        print(f"⚠️ [Audio Send Error]: {e}")
                        break

            async def receive_audio():
                last_transcript = ""
                while is_running:
                    try:
                        message = await ws.recv()
                        data = json.loads(message)
                        event_type = data.get("type")

                        if event_type == "input_audio_buffer.speech_started":
                            print("\n🗣️ [말하는 중...]")
                        elif event_type == "input_audio_buffer.speech_stopped":
                            print("🤐 [말 끝남] -> 서버 처리 중")

                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = data.get("transcript", "").strip()
                            if _is_valid_transcript(transcript) and transcript != last_transcript:
                                # TTS가 읽은 문장이 다시 들리는 에코 현상인지 필터링
                                if _is_echo_text(transcript):
                                    print(f"🔇 [Echo Filter] TTS 에코 무시됨: {transcript}")
                                    continue
                                
                                last_transcript = transcript
                                print(f"🎤 [User STT]: {transcript}")
                                stt_queue.put(transcript)
                        
                        elif event_type == "error":
                            err_msg = data.get("error", {}).get("message", "Unknown Error")
                            print(f"❌ [OpenAI Realtime Error]: {err_msg}")

                    except websockets.exceptions.ConnectionClosed:
                        print("🔌 [OpenAI] 소켓 연결 종료")
                        break
                    except Exception as e:
                        print(f"❌ [Receive Error]: {e}")
                        break

            # 송신(Mic)과 수신(Realtime API) 루프 동시 실행
            await asyncio.gather(send_audio(), receive_audio())

    except Exception as e:
        print(f"❌ [Audio Loop Error]: {e}")
    finally:
        worker_stop.set()
        try:
            if mic_stream.is_active():
                mic_stream.stop_stream()
            mic_stream.close()
            p.terminate()
        except:
            pass
        print("🛑 [Audio] 오디오 리소스 해제 완료")

def socket_server_thread():
    global kiosk_socket
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 9999))
        server.listen(1)
        print("💡 [Socket] 포트 9999 대기 중 (Unity 연결용)...")
        while is_running:
            server.settimeout(1.0)
            try:
                conn, addr = server.accept()
                print(f"✅ [Socket] 클라이언트 연결됨: {addr}")
                with socket_lock:
                    kiosk_socket = conn
            except socket.timeout:
                continue
            except Exception as e:
                print(f"❌ [Socket Accept Error]: {e}")
                time.sleep(1)
    except Exception as e:
        print(f"❌ [Socket Server Error]: {e}")

if __name__ == "__main__":
    # 1. 오디오 처리 스레드 (OpenAI Realtime + NLU)
    audio_t = threading.Thread(target=audio_thread_entry, daemon=True)
    audio_t.start()

    # 2. 소켓 서버 스레드 (Unity 통신)
    socket_t = threading.Thread(target=socket_server_thread, daemon=True)
    socket_t.start()

    # 3. 메인 스레드: 카메라 루프 (OpenCV GUI 표시를 위해 메인에서 실행)
    try:
        camera_loop_main()
    except KeyboardInterrupt:
        print("\n👋 프로그램 종료 요청됨.")
        is_running = False
    
    print("시스템 종료 중...")
