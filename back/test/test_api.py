import asyncio
import os
from dotenv import load_dotenv
from google import genai

# 1. 환경변수 및 API 키 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../..")) # code 폴더 찾기
env_path = os.path.join(root_dir, ".env")

if not os.path.exists(env_path):
    load_dotenv()
else:
    load_dotenv(env_path)

MY_API_KEY = os.getenv("GEMINI_API_KEY")
if not MY_API_KEY:
    raise ValueError("API 키가 없습니다.")

# [수정 1] http_options 제거 (SDK가 자동으로 'v1alpha' 등을 선택하게 함)
client = genai.Client(api_key=MY_API_KEY, http_options={'api_version': 'v1alpha'})

async def main():
    # [수정 2] Live API를 지원하는 '실험용(exp)' 모델 사용
    model_id = "gemini-2.0-flash"
    
    config = {"response_modalities": ["TEXT"]}

    print(f"📡 {model_id} 모델에 실시간(Live) 연결 시도 중...")

    try:
        # Live API 연결 시작
        async with client.aio.live.connect(model=model_id, config=config) as session:
            print("✅ [연결 성공] 세션이 시작되었습니다.")
            
            message = "안녕? 너는 실시간으로 대화가 가능하니?"
            print(f"📤 나: {message}")
            
            # 메시지 전송
            await session.send(input=message, end_of_turn=True)

            # 응답 수신 루프
            async for response in session.receive():
                # 텍스트 데이터가 들어왔는지 확인
                if response.server_content is not None:
                    model_turn = response.server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.text:
                                print(f"🤖 Gemini: {part.text}")
                                return # 응답 하나만 받고 종료 (테스트용)
                
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main())