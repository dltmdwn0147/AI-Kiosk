import pandas as pd

# 1. CSV 파일 읽어오기
# 'input.csv' 부분에 변환하려는 파일명을 넣으세요.
df = pd.read_csv('/Users/iseungju/Desktop/RISE-AI/code/front/DATA/drinks_menu.csv')

# 2. 엑셀 파일로 저장하기
# index=False는 불필요한 행 번호(0, 1, 2...)가 엑셀에 들어가는 것을 방지합니다.
df.to_excel('output.xlsx', index=False)

print("변환이 완료되었습니다.")