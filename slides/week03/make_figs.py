"""3주차 — 핸즈온 Ch1·Ch2 강의용 그림 (California housing 데이터로 직접 생성).

교재 그림을 스캔 캡처하지 않고, 공개 데이터(ageron/data housing.csv)와
matplotlib으로 같은 내용을 흑백 재현한다.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'legend.fontsize': 13,
    'font.family': 'Pretendard',
    'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.prop_cycle': mpl.cycler(color=['0.0', '0.45', '0.7'],
                                  linestyle=['-', '--', ':']),
    'lines.linewidth': 2, 'legend.frameon': False,
})

housing = pd.read_csv('data/housing.csv')

# fig1 — 주요 수치 특성 히스토그램 (책 그림 2-8에 해당, 불릿 옆 3:2 자리)
cols = ['median_income', 'housing_median_age',
        'median_house_value', 'population']
labels = ['중간 소득 (만 달러)', '주택 연식 (년)', '중간 주택 가격 (달러)', '인구']
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.55))
for ax, c, lb in zip(axes.ravel(), cols, labels):
    ax.hist(housing[c], bins=50, color='0.55', edgecolor='white')
    ax.set_xlabel(lb, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_yticks([])
fig.tight_layout()
fig.savefig('fig_hist.png', dpi=200)

# fig2 — 지리적 산점도: 위치·인구·주택 가격 (책 그림 2-13에 해당)
fig, ax = plt.subplots(figsize=(7.2, 4.55))
sc = ax.scatter(housing['longitude'], housing['latitude'],
                s=housing['population'] / 80,
                c=housing['median_house_value'],
                cmap='Greys', alpha=0.5, edgecolors='none')
ax.set_xlabel('경도'); ax.set_ylabel('위도')
cb = fig.colorbar(sc, ax=ax)
cb.set_label('중간 주택 가격 (달러)', fontsize=12)
cb.ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig('fig_geo.png', dpi=200)

# fig3 — 중간 소득 vs 중간 주택 가격 (책 그림 2-15에 해당, 상한선 확인)
fig, ax = plt.subplots(figsize=(7.2, 4.55))
ax.scatter(housing['median_income'], housing['median_house_value'],
           s=4, color='0.3', alpha=0.15, edgecolors='none')
ax.set_xlabel('중간 소득 (만 달러)')
ax.set_ylabel('중간 주택 가격 (달러)')
ax.set_xlim(0, 16)
fig.tight_layout()
fig.savefig('fig_income.png', dpi=200)

# fig4 — 과대적합/과소적합 개념 (책 그림 1-23 취지, 합성 데이터)
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

# fig5 — 소득 카테고리 히스토그램 (계층적 샘플링 근거, 책 그림 2-9에 해당)
income_cat = pd.cut(housing['median_income'],
                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                    labels=['1', '2', '3', '4', '5'])
fig, ax = plt.subplots(figsize=(7.2, 4.55))
counts = income_cat.value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values, color='0.55',
       edgecolor='black')
ax.set_xlabel('소득 카테고리')
ax.set_ylabel('구역 수')
fig.tight_layout()
fig.savefig('fig_income_cat.png', dpi=200)

# fig6 — 꼬리 두꺼운 분포의 로그 변환 (책 그림 2-17에 해당)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.55))
axes[0].hist(housing['population'], bins=50, color='0.55',
             edgecolor='white')
axes[0].set_xlabel('인구 (원본)')
axes[1].hist(np.log(housing['population']), bins=50, color='0.55',
             edgecolor='white')
axes[1].set_xlabel('log(인구)')
for ax in axes:
    ax.set_ylabel('구역 수')
fig.tight_layout()
fig.savefig('fig_log.png', dpi=200)

print('saved 6 figs')
