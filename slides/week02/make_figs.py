"""2주차 — 핸즈온 Ch1 강의용 그림.

교재 그림을 스캔하지 않고 합성 데이터와 matplotlib으로 같은 내용을 흑백 재현한다.
fig_fit은 3주차와 같은 방식(난수 시드 42)으로 생성해 두 주차의 그림을 일치시킨다.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'legend.fontsize': 13,
    'font.family': 'Pretendard',
    'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.prop_cycle': mpl.cycler(color=['0.0', '0.45', '0.7'],
                                  linestyle=['-', '--', ':']),
    'lines.linewidth': 2, 'legend.frameon': False,
})

# fig1 — 과대적합·과소적합 개념 (책 그림 1-23 취지, 합성 데이터)
rng = np.random.default_rng(42)
x = np.sort(rng.uniform(0, 3, 30))
y = np.sin(1.5 * x) + rng.normal(0, 0.25, 30)
xs = np.linspace(0.02, 2.98, 300)
fig, ax = plt.subplots(figsize=(7.2, 4.55))
ax.scatter(x, y, s=30, color='0.6', edgecolors='none', zorder=3)
for deg, label in [(1, '1차 — 과소적합'), (4, '4차 — 적절'),
                   (20, '20차 — 과대적합')]:
    coef = np.polyfit(x, y, deg)
    ax.plot(xs, np.polyval(coef, xs), label=label)
ax.set_ylim(-1.8, 1.8)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('특성'); ax.set_ylabel('타깃')
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig('fig_fit.png', dpi=200)

# fig2 — 시간 순서 분할과 무작위 분할의 구조 비교 (데이터 값이 아닌 분할 방식의 모식도)
fig, ax = plt.subplots(figsize=(7.2, 4.55))
months = ['4월', '5월', '6월', '7월', '8월']

# 위쪽 막대: 시간 순서 분할 — 8월 이전 학습, 8월 평가
ax.broken_barh([(0, 4)], (1.55, 0.5), facecolors='0.35', edgecolor='white')
ax.broken_barh([(4, 1)], (1.55, 0.5), facecolors='0.8', edgecolor='white',
               hatch='//')
ax.text(2.0, 1.8, '학습', ha='center', va='center', color='white', fontsize=13)
ax.text(4.5, 1.8, '평가', ha='center', va='center', color='0.1', fontsize=13)

# 아래쪽 막대: 무작위 분할 — 평가 구간이 전 기간에 흩어짐
edges = np.linspace(0, 5, 26)
test_idx = {2, 6, 10, 14, 18, 22}
train_spans = [(edges[i], edges[i + 1] - edges[i])
               for i in range(25) if i not in test_idx]
test_spans = [(edges[i], edges[i + 1] - edges[i])
              for i in range(25) if i in test_idx]
ax.broken_barh(train_spans, (0.55, 0.5), facecolors='0.35', edgecolor='white')
ax.broken_barh(test_spans, (0.55, 0.5), facecolors='0.8', edgecolor='white',
               hatch='//')

handles = [mpl.patches.Patch(facecolor='0.35', label='학습 구간'),
           mpl.patches.Patch(facecolor='0.8', hatch='//', label='평가 구간')]
ax.legend(handles=handles, loc='upper center', ncol=2,
          bbox_to_anchor=(0.5, 0.18))

ax.set_yticks([1.8, 0.8])
ax.set_yticklabels(['시간 순서 분할', '무작위 분할'])
ax.set_xticks(np.arange(0.5, 5.5, 1))
ax.set_xticklabels(months)
ax.set_xlabel('2022년')
ax.set_xlim(0, 5)
ax.set_ylim(0.0, 2.4)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
fig.tight_layout()
fig.savefig('fig_split.png', dpi=200)

print('saved 2 figs')
