import pandas as pd
import json
import os

def excel_to_json(excel_file_path, output_json_path):
    # 파일 존재 여부 확인
    if not os.path.exists(excel_file_path):
        print(f"❌ 오류: '{excel_file_path}' 파일을 찾을 수 없습니다.")
        return

    print(f"📂 '{excel_file_path}' 파일을 읽는 중...")

    try:
        # 1. 옵션 데이터 로드 (ID -> Name 매핑용)
        option_df = pd.read_excel(excel_file_path, sheet_name='옵션')
        option_map = dict(zip(option_df['번호'], option_df['옵션 이름']))

        # 2. 메뉴 데이터 로드 (헤더 처리를 위해 header=None으로 로드)
        raw_menu_df = pd.read_excel(excel_file_path, sheet_name='메뉴', header=None)
        
    except ValueError as e:
        print(f"❌ 엑셀 읽기 오류: {e}")
        return

    # 컬럼명 재구성 (메뉴 시트 구조 반영)
    main_columns = raw_menu_df.iloc[0].values
    sub_columns = raw_menu_df.iloc[1].values

    final_columns = []
    option_start_index = -1

    for i, col in enumerate(main_columns):
        if col == '옵션':
            option_start_index = i
            final_columns.append(sub_columns[i])
        elif option_start_index != -1 and i > option_start_index:
            final_columns.append(sub_columns[i])
        else:
            final_columns.append(col)

    # 데이터프레임 정리 (Row 2부터 실제 데이터)
    menu_df = raw_menu_df.iloc[2:].copy()
    menu_df.columns = final_columns
    menu_df = menu_df.reset_index(drop=True)
    
    # 메뉴 이름 없는 행 제거
    menu_df = menu_df.dropna(subset=['메뉴 이름'])

    # 3. JSON 변환
    menu_data_list = []
    option_kinds = final_columns[option_start_index:] # 옵션 종류들

    for index, row in menu_df.iterrows():
        # 가격 및 ID 처리 (숫자 변환 안전하게)
        try:
            price = int(float(row['가격'])) if pd.notna(row['가격']) else 0
        except:
            price = 0
            
        try:
            m_id = int(float(row['번호'])) if pd.notna(row['번호']) else 0
        except:
            m_id = 0

        menu_item = {
            "menu_id": m_id,  # [추가됨] 메뉴 번호
            "menu_name": row['메뉴 이름'],
            "menu_price": price,
            "menu_temperature": row['온도(Hot/Ice)'],
            "menu_description": row['메뉴 설명'],
            "options": []
        }
        
        # 옵션 매핑
        for kind in option_kinds:
            val = row[kind]
            
            if pd.isna(val) or val == 0 or val == '0':
                continue
            
            ids = []
            if isinstance(val, str):
                ids = [int(float(x.strip())) for x in val.split(',')]
            elif isinstance(val, (int, float)):
                ids = [int(val)]
                
            details = []
            for opt_id in ids:
                if opt_id in option_map:
                    details.append({
                        "option_id": opt_id,
                        "option_name": option_map[opt_id]
                    })
            
            if details:
                menu_item["options"].append({
                    "option_kind": kind,
                    "option_details": details
                })
        
        menu_data_list.append(menu_item)

    # 4. JSON 파일 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(menu_data_list, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 변환 완료! '{output_json_path}' 파일 생성됨.")
    print(f"   총 {len(menu_data_list)}개의 메뉴 처리 완료.")

# --- 실행 ---
excel_file = '메가커피 가게 데이터.xlsx'
output_file = 'mega_coffee_menu.json'

excel_to_json(excel_file, output_file)