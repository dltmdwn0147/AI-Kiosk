import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== 1. macOS 한글 폰트 설정 =====
mpl.rcParams["font.family"] = "AppleGothic"   # 맥 기본 한글 폰트
mpl.rcParams["axes.unicode_minus"] = False    # 마이너스 기호 깨짐 방지

# ===== 2. 데이터 설정 (예시 비중) =====
labels = ["비대면 주문", "주문시간 감소", "주문·결제 플로우 편의성"]
sizes  = [40, 35, 25]  # 100% 기준 임의 설정

colors = ["#ffb3ba", "#baffc9", "#bae1ff"]

# ===== 3. 원그래프 그리기 =====
plt.figure(figsize=(6, 6))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct="%.1f%%",   # 퍼센트 표시
    startangle=90,
    colors=colors,
    counterclock=False
)

plt.title("키오스크 주요 장점 3가지 비중 (예시)")
plt.tight_layout()
plt.show()
