import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# macOS 한글 폰트 설정
mpl.rcParams["font.family"] = "AppleGothic"
mpl.rcParams["axes.unicode_minus"] = False

# === 그래프 2: 주문 시간 감소 효과 ===
methods = ["대면 주문", "키오스크 주문"]
order_time_index = [100, 60]  # 대면=100, 키오스크≈60 (약 40% 감소)

x = np.arange(len(methods))
bar_width = 0.6

plt.figure(figsize=(6, 5))
bars = plt.bar(x, order_time_index, width=bar_width, color=["#c0504d", "#4bacc6"])

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.xticks(x, methods)
plt.ylabel("주문 시간 지수(대면 주문=100)")
plt.title("키오스크 도입에 따른 주문 시간 감소 (예시, 약 40% 감소)")

plt.ylim(0, 120)
plt.tight_layout()
plt.show()
