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
known_faces_cache = {"faces": [], "ts": 0.0}

tts_playing_event = threading.Event()
mic_stream_lock = threading.Lock()

KOR_NUM_MAP = {
    "한": 1, "하나": 1, "한잔": 1, "한 잔": 1,
    "두": 2, "둘": 2, "두잔": 2, "두 잔": 2,
    "세": 3, "셋": 3, "서이": 3, "세잔": 3, "세 잔": 3,
    "네": 4, "넷": 4, "너이": 4, "네잔": 4, "네 잔": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}

TEMPERATURE_ICE_HINTS = ["아이스", "ice", "차가", "차게", "냉", "콜드", "시원"]
TEMPERATURE_HOT_HINTS = ["핫", "hot", "따뜻", "뜨거", "맨도롱", "따아", "온", "데운"]

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

PRICE_HINTS = [
    "가격", "얼마", "값", "비용", "얼마야", "얼마에요", "얼마예요", "얼마임",
    "가격 알려", "가격좀", "가격 좀", "가격이 뭐", "가격이뭐", "가격이 얼마", "얼마 하지",
    "얼마입니까", "얼마야?", "얼마에요?", "얼마예요?"
]

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

def _is_price_query(text: str) -> bool:
    if not text:
        return False
    t = _normalize_text(text)
    return any(_normalize_text(h) in t for h in PRICE_HINTS)

def _extract_price(menu: dict):
    if not isinstance(menu, dict):
        return None
    for k in ["price", "menu_price", "cost", "amount", "menu_cost", "menuAmount", "menuPrice"]:
        if k in menu and menu.get(k) not in [None, ""]:
            return menu.get(k)
    for k in ["menu_price_won", "price_won", "won"]:
        if k in menu and menu.get(k) not in [None, ""]:
            return menu.get(k)
    return None

def _format_price(price_val):
    if price_val is None:
        return None
    try:
        if isinstance(price_val, str):
            s = price_val.strip()
            s2 = re.sub(r"[^\d]", "", s)
            if s2:
                n = int(s2)
                return f"{n:,}원"
            return s
        if isinstance(price_val, (int, float)):
            n = int(price_val)
            return f"{n:,}원"
    except:
        pass
    return str(price_val)

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
        print(f"✅ [Order Log] 저장 완료: {file_name} (total={len(logs)})")
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
                print(f"   └─ 📡 키오스크 전송: {payload_dict}")
                kiosk_socket.sendall(payload)
                msg = "주문 성공"
            except Exception as e:
                print(f"   └─ ❌ 키오스크 전송 실패: {e}")

    recent_action_cache["text"] = payload_json
    recent_action_cache["ts"] = now
    return f"{msg}: {payload_json}"

def _load_known_faces(file_name: str = "order_logs.json", ttl_sec: int = 10) -> list:
    now = time.time()
    if now - known_faces_cache["ts"] < ttl_sec and known_faces_cache["faces"]:
        return known_faces_cache["faces"]
    faces = []
    path = file_name
    if not os.path.isabs(path):
        path = os.path.join(current_dir, file_name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
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

def _is_valid_transcript(text: str) -> bool:
    if not text:
        return False
    cleaned = str(text).strip()
    return cleaned and cleaned.lower() != "none"

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

class MenuCatalog:
    def __init__(self, raw_menu):
        self.menus = []
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
        self._name_index = {}
        for m in self.menus:
            name = str(m.get("menu_name", "")).strip()
            if not name:
                continue
            self._name_index.setdefault(name, []).append(m)

    def all_names(self):
        return list(self._name_index.keys())

    def _score_similarity(self, a: str, b: str) -> float:
        na, nb = _normalize_text(a), _normalize_text(b)
        if not na or not nb:
            return 0.0
        if na in nb or nb in na:
            return 1.0
        return SequenceMatcher(None, na, nb).ratio()

    def find_best_name_candidates(self, user_text: str, top_k: int = 8):
        ut = _normalize_text(user_text)
        if not ut:
            return []
        exact = []
        for name in self.all_names():
            if _normalize_text(name) and _normalize_text(name) in ut:
                exact.append((1.0, name))
        if exact:
            exact.sort(key=lambda x: x[0], reverse=True)
            return [n for _, n in exact[:top_k]]

        scored = []
        for name in self.all_names():
            s = self._score_similarity(user_text, name)
            if s > 0.35:
                scored.append((s, name))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]

    def get_menus_by_name(self, name: str):
        return list(self._name_index.get(name, []))

    def _is_modified_variant(self, menu_name: str) -> bool:
        n = str(menu_name).strip()
        for mod in VARIANT_MODIFIERS:
            if n.startswith(mod):
                return True
        return False

    def _user_wants_modifier(self, user_text: str) -> bool:
        t = _normalize_text(user_text)
        for mod in VARIANT_MODIFIERS:
            if _normalize_text(mod) in t:
                return True
        return False

    def pick_variant_for_base(self, menus_same_temp: list, user_text: str):
        if not menus_same_temp:
            return None

        t = _normalize_text(user_text)

        if self._user_wants_modifier(user_text):
            hits = []
            for m in menus_same_temp:
                mn = str(m.get("menu_name", ""))
                if any(_normalize_text(mod) in t and mn.startswith(mod) for mod in VARIANT_MODIFIERS):
                    hits.append(m)
            if hits:
                return hits[0]

        if any(_normalize_text(x) in t for x in ["일반", "기본", "그냥"]):
            plains = [m for m in menus_same_temp if not self._is_modified_variant(m.get("menu_name", ""))]
            if plains:
                return plains[0]

        plains = [m for m in menus_same_temp if not self._is_modified_variant(m.get("menu_name", ""))]
        if plains:
            return plains[0]
        return menus_same_temp[0]

    def resolve_menu(self, user_text: str, forced_name: str = "", forced_temp: str = ""):
        candidates = [forced_name] if forced_name else self.find_best_name_candidates(user_text)
        if not candidates:
            return None, "menu", []

        best_name = candidates[0]
        menus = self.get_menus_by_name(best_name)
        if not menus:
            return None, "menu", []

        temp = forced_temp or _parse_temperature(user_text)
        temps = sorted(list({str(m.get("menu_temperature", "")).strip() for m in menus if str(m.get("menu_temperature", "")).strip()}))

        if len(temps) >= 2 and not temp:
            return None, "temperature", [best_name]

        if temp:
            menus_same_temp = [m for m in menus if str(m.get("menu_temperature", "")).strip() == temp]
            if menus_same_temp:
                picked = self.pick_variant_for_base(menus_same_temp, user_text)
                return picked, "", []
            picked = self.pick_variant_for_base(menus, user_text)
            return picked, "", []

        picked = self.pick_variant_for_base(menus, user_text)
        return picked, "", []

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
    return {
        "type": "add",
        "menu_name": menu_name,
        "quantity": max(1, qty),
        "temperature": temp if temp in ["ICE", "HOT"] else "",
    }

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

def handle_user_text(user_text: str):
    global order_state, menu_catalog

    user_text = (user_text or "").strip()
    if not user_text:
        return "", []

    if menu_catalog is not None and _is_price_query(user_text):
        resolved_menu, need_slot, need_data = menu_catalog.resolve_menu(user_text)
        if not resolved_menu:
            if need_slot == "temperature" and need_data:
                base_name = need_data[0]
                return f"{base_name} 가격을 알려드릴까요? 핫/아이스 중 어떤 온도 가격을 원하세요?", []
            return "어떤 메뉴의 가격을 알고 싶으신가요? 예: '아이스 아메리카노 가격 얼마야?'", []
        p = _extract_price(resolved_menu)
        fp = _format_price(p)
        if fp:
            return f"{resolved_menu.get('menu_name','해당 메뉴')} 가격은 {fp}입니다.", []
        return f"{resolved_menu.get('menu_name','해당 메뉴')}의 가격 정보가 메뉴 데이터에 없습니다.", []

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

    if _contains_any(user_text, ["취소", "처음부터", "리셋"]):
        order_state.reset()
        return "네, 주문을 취소하고 처음부터 도와드릴게요. 원하시는 메뉴를 말씀해 주세요.", []

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
                num_faces=10,
                min_face_detection_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE,
            )
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
                        cv2.putText(
                            image,
                            known_text,
                            (cx, max(20, cy - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0) if is_known else (0, 0, 255),
                            2,
                        )

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

    p = None
    mic_stream = None

    async def stop_mic_stream():
        nonlocal mic_stream
        with mic_stream_lock:
            s = mic_stream
        if s:
            try:
                if s.is_active():
                    s.stop_stream()
            except:
                pass

    async def start_mic_stream():
        nonlocal mic_stream
        with mic_stream_lock:
            s = mic_stream
        if s:
            try:
                if not s.is_active():
                    s.start_stream()
            except:
                pass

    async def play_tts_and_pause_mic(ws, text: str):
        if not text:
            return
        tts_playing_event.set()
        try:
            try:
                await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            except:
                pass
            await stop_mic_stream()
            tts_payload = {
                "model": "tts-1",
                "voice": "shimmer",
                "input": text,
                "response_format": "wav",
            }
            audio_bytes = _post_bytes("https://api.openai.com/v1/audio/speech", tts_payload, rest_headers)
            _play_wav_audio(audio_bytes)
        except Exception as e:
            print(f"❌ [TTS Error]: {e}")
        finally:
            await asyncio.sleep(0.15)
            await start_mic_stream()
            tts_playing_event.clear()

    try:
        p = pyaudio.PyAudio()
        mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("🎧 [Audio] OpenAI 연결 중...")

        async with websockets.connect(url, additional_headers=headers) as ws:
            print("✅ [Audio] 연결 성공!")

            loop = asyncio.get_running_loop()

            session_update = {
                "type": "session.update",
                "session": {
                    "instructions": system_instruction,
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1", "language": "ko", "prompt": "한국어, 제주방언, 사투리"},
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

                    reply, actions = handle_user_text(user_text)

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
                        asyncio.run_coroutine_threadsafe(play_tts_and_pause_mic(ws, reply), loop)

            worker_thread = threading.Thread(target=nlu_worker, daemon=True)
            worker_thread.start()

            try:
                order_log_path = os.path.join(current_dir, "order_logs.json")
                top_menus = get_recent_top_menus(order_log_path, top_k=3, days=30)
                if top_menus:
                    reco_text = " / ".join(top_menus)
                    greet_text = f"안녕하세요, 메가커피입니다. 이전에 자주 주문하신 {reco_text} 어떠세요? 원하시는 메뉴를 말씀해 주세요."
                else:
                    greet_text = "안녕하세요, 메가커피입니다. 원하시는 메뉴를 말씀해 주세요."
                await play_tts_and_pause_mic(ws, greet_text)
            except Exception as e:
                print(f"❌ [TTS] Greeting Error: {e}")

            async def send_audio():
                while is_running:
                    try:
                        if tts_playing_event.is_set():
                            await asyncio.sleep(0.03)
                            continue
                        with mic_stream_lock:
                            s = mic_stream
                        if not s:
                            await asyncio.sleep(0.05)
                            continue
                        data = await asyncio.to_thread(s.read, CHUNK, exception_on_overflow=False)
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

                    except websockets.exceptions.ConnectionClosed:
                        break
                    except:
                        break

            await asyncio.gather(send_audio(), receive_audio())

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        worker_stop.set()
        try:
            with mic_stream_lock:
                s = mic_stream
            if s:
                try:
                    s.stop_stream()
                except:
                    pass
                try:
                    s.close()
                except:
                    pass
            if p:
                try:
                    p.terminate()
                except:
                    pass
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
