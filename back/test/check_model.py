import os
from dotenv import load_dotenv
from google import genai

# 환경변수 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ API 키가 없습니다. .env 파일을 확인하세요.")
    exit()

try:
    client = genai.Client(api_key=api_key)
    print(f"🔍 API Key 확인 완료. 사용 가능한 모델 목록을 불러옵니다...\n")

    print("--- [사용 가능한 모델 이름] ---")
    
    # 모델 목록 가져오기
    for model in client.models.list():
        # 모델 이름에 'gemini'가 들어간 것만 출력 (보기 편하게)
        if "gemini" in model.name:
            print(f"👉 {model.name}")
            
    print("\n--------------------------------")
    print("위 목록 중 하나를 복사해서 main.py의 model='...' 부분에 넣으세요.")

except Exception as e:
    print(f"❌ 오류 발생: {e}")