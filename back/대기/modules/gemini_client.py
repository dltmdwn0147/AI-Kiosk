import os
import json
import asyncio
from dotenv import load_dotenv
from google import genai # 신규 라이브러리
from google.genai import types

# 1. 환경변수 로드 (.env 찾기 강화)
# 현재 파일 위치: code/back/modules/gemini_client.py
# .env 위치: code/.env (세 단계 상위)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../..")) # code 폴더 찾기
env_path = os.path.join(root_dir, ".env")

# 만약 루트에서 못 찾으면, 혹시 모를 상위 탐색 (안전장치)
if not os.path.exists(env_path):
    load_dotenv() # 기본 탐색
else:
    load_dotenv(env_path)

MY_API_KEY = os.getenv("GEMINI_API_KEY")

class GeminiClient:
    def __init__(self, api_key=None):
        """
        Gemini API 클라이언트 (Google GenAI SDK v1.0 사용)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ API Key가 없습니다. .env 파일을 확인해주세요.")

        # [변경] 신규 클라이언트 초기화 방식
        self.client = genai.Client(api_key=self.api_key)
        
        # 메뉴 데이터 로드
        self.menu_data = self._load_menu_data()
        
        # 시스템 프롬프트 준비
        self.system_instruction = self._create_system_prompt()
        
        # [변경] 모델 설정 (Config 객체 사용)
        self.config = types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
            system_instruction=self.system_instruction
        )
        
        # 채팅 세션 (v1에서는 chat 객체가 조금 다릅니다. 여기서는 간단히 모델명 저장)
        self.model_name = "gemini-1.5-flash-002" # 최신 모델 추천 (또는 gemini-1.5-flash)
        self.chat = self.client.chats.create(
            model=self.model_name,
            config=self.config
        )

        print(f"🧠 [Gemini Client] 신규 SDK 모델 준비 완료 (메뉴 {len(self.menu_data)}개)")

    def _load_menu_data(self):
        """메뉴 파일 로드"""
        # 경로 보정: code/back/modules/ 에서 실행되므로 프로젝트 루트 기준 경로를 찾아야 함
        # 편의상 절대경로로 계산
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        paths_to_check = [
            os.path.join(project_root, "data/processed/menu.json"),
            os.path.join(project_root, "data/menu.json")
        ]
        
        for path in paths_to_check:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        print(f"📂 [System] 메뉴 데이터 로드: {path}")
                        return json.load(f)
                except Exception:
                    continue
        
        print("⚠️ [Warning] 메뉴 파일을 찾을 수 없습니다. 빈 리스트 사용.")
        return []

    def _create_system_prompt(self):
        menu_str = json.dumps(self.menu_data, ensure_ascii=False, indent=2)
        return f"""
        당신은 카페 키오스크 AI '지니'입니다.
        [메뉴 데이터] {menu_str}
        
        [JSON 응답 형식]
        {{ "text": "응대 멘트", "command": {{ "type": "ADD_CART", "payload": "값" }} }}
        """

    async def get_response(self, user_input):
        print(f"📤 [To Gemini]: {user_input}")
        try:
            # [변경] 신규 SDK 비동기 호출 방식
            # 주의: google-genai 라이브러리의 비동기 지원이 아직 실험적일 수 있어 동기로 호출하거나
            # async 래퍼를 써야 할 수도 있습니다. 여기서는 표준 호출 사용.
            
            # v1.0 SDK는 동기 호출이 기본이나, async 사용 시 client.aio 사용
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=user_input,
                config=self.config
            )
            
            response_json = json.loads(response.text)
            print(f"📥 [From Gemini]: {response_json}")
            return response_json

        except Exception as e:
            print(f"❌ [Error]: {e}")
            return {"text": "오류가 발생했습니다.", "command": None}

# --- 테스트 실행 코드 ---
if __name__ == "__main__":
    async def test_main():
        # [수정됨] 논리 오류 수정: 키가 '없으면' 에러
        if not MY_API_KEY:
            print("⚠️ API 키가 없습니다! .env 위치를 확인하세요.")
            return

        client = GeminiClient(api_key=MY_API_KEY)
        
        # 질문 테스트
        await client.get_response("아이스 아메리카노 한 잔 줘")

    asyncio.run(test_main())