import google.generativeai as genai
import pandas as pd
import json
import os
import time
from dotenv import load_dotenv

# [설정] API 키 입력 (또는 환경변수)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") 
if not API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

genai.configure(api_key=API_KEY)

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_FILE = os.path.join(BASE_DIR, "data", "raw", "menu.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "menu_enhanced.json")

def analyze_menu_with_gemini(df):
    """
    단순 엑셀 데이터를 Gemini에게 보내서 '관계성'과 '홍보 멘트'를 추가한 JSON을 받음
    """
    # 1. 엑셀 데이터를 CSV 문자열로 변환 (AI에게 보여주기 위해)
    csv_data = df.to_csv(index=False)

    # 2. 분석가 AI를 위한 프롬프트 작성
    prompt = f"""
    당신은 최첨단 AI 키오스크의 **시스템 로직 설계자(System Logic Architect)**입니다.
    아래 제공된 [메뉴 리스트]를 분석하여, 실제 AI 모델이 키오스크를 제어할 때 참고할 **'제약 조건(Constraints)'과 '옵션 로직(Logic)'이 포함된 JSON 데이터**를 생성하세요.

    [분석 요구사항]
    1. **기본 정보 유지:** 'id', 'name', 'price'는 원본 그대로 유지하십시오.
    2. **description (필수):** 메뉴에 대한 간략한 설명 (응대용).
    3. **options (핵심):** 해당 메뉴에서 선택 가능한 옵션 목록 (예: HOT/ICE, 샷추가, 사이즈업 등).
        - 메뉴의 특성을 고려하여 합리적인 옵션을 자동으로 생성하십시오.
    4. **constraints (가장 중요):** 해당 메뉴 주문 시 시스템이 반드시 지켜야 할 제약 조건.
        - 예: "샷 추가는 최대 3회까지만 가능", "ICE 전용 메뉴는 HOT 변경 불가".
       - **고객이 무리한 요구(예: 샷 100번 추가)를 할 때 AI가 근거로 삼을 규칙**을 명시하십시오.
    5. **recommendation_target:** 이 메뉴와 함께 추천하면 좋은 메뉴의 ID (관계성).

    [메뉴 리스트]
    {csv_data}

    [출력 포맷 예시 (Strict JSON)]
    [
        {{
        "id": 101,
        "name": "아이스 아메리카노",
        "price": 4500,
        "description": "산미와 고소함이 조화로운 시원한 커피",
        "options": ["샷 추가(500원)", "연하게", "시럽 추가"],
        "constraints": [
            "샷 추가는 컵 용량 제한으로 인해 최대 3회(총 4샷)까지만 가능합니다.",
            "얼음 없이(No Ice) 주문 시 음료 양이 줄어들 수 있음을 안내해야 합니다.",
            "기본적으로 차가운 음료이므로 '따뜻하게' 옵션 요청 시 거절하고 따뜻한 아메리카노(ID:102)를 안내해야 합니다."
        ],
        "recommendation_target": [105, 106]
        }}
    ]

    위 포맷을 준수하여 오직 JSON 리스트만 출력하십시오.
    """

    print("🧠 [Smart Converter] Gemini가 메뉴 관계성을 분석 중입니다... (약 5~10초 소요)")

    # 3. 모델 호출 (똑똑한 분석을 위해 Pro 모델 권장, Flash도 가능)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro", 
        generation_config={
            "response_mime_type": "application/json", # <--- "야, 딴소리 말고 무조건 JSON만 줘!"
            "temperature": 0.1,                       # <--- "상상하지 말고 있는 데이터 그대로 분석해!"
        }
    )
    
    response = model.generate_content(prompt)
    
    # 4. 결과 반환
    return json.loads(response.text)

def run():
    # 1. 엑셀 읽기
    if not os.path.exists(EXCEL_FILE):
        print("엑셀 파일이 없습니다. utils/converter.py를 먼저 실행해서 샘플을 만드세요.")
        return
    
    df = pd.read_excel(EXCEL_FILE)
    
    # 2. AI 분석 실행
    try:
        enhanced_data = analyze_menu_with_gemini(df)
        
        # 3. 저장
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ [Success] 똑똑한 메뉴 데이터가 생성되었습니다: {OUTPUT_FILE}")
        print("   이제 gemini_client.py가 이 파일을 읽으면 훨씬 똑똑하게 대답합니다!")
        
    except Exception as e:
        print(f"❌ [Error] 분석 실패: {e}")

if __name__ == "__main__":
    run()