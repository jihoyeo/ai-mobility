"""4주차 — 핸즈온 Ch4(+Ch3 발췌) 강의용 그림 (합성·공개 데이터로 직접 생성).

교재 그림을 캡처하지 않고 같은 내용을 흑백으로 재현한다.
실행하면 코드 슬라이드에 쓸 수치도 함께 출력한다.
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

rng = np.random.default_rng(42)

# ── 책 4.1의 선형 데이터: y = 4 + 3x + 잡음 ──────────────────────────
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

from sklearn.linear_model import LinearRegression, SGDRegressor
lin = LinearRegression().fit(X, y)
print('LinearRegression  intercept, coef =',
      lin.intercept_.round(3), lin.coef_.ravel().round(3))

sgd = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None,
                   eta0=0.01, random_state=42)
sgd.fit(X, y.ravel())
print('SGDRegressor      intercept, coef =',
      sgd.intercept_.round(3), sgd.coef_.round(3))

# fig_gd_lr — 학습률 3종의 손실 곡선 (배치 GD, 불릿 옆 3:2 자리)
Xb = np.c_[np.ones((m, 1)), X]


def gd_losses(eta, n_iter=60):
    theta = np.zeros((2, 1))
    losses = []
    for _ in range(n_iter):
        losses.append(float(np.mean((Xb @ theta - y) ** 2)))
        grad = 2 / m * Xb.T @ (Xb @ theta - y)
        theta -= eta * grad
    return np.array(losses)


fig, ax = plt.subplots(figsize=(7.2, 4.55))
for eta, label in [(0.002, '학습률 0.002 (너무 작음)'),
                   (0.1, '학습률 0.1 (적당)'),
                   (0.52, '학습률 0.52 (너무 큼)')]:
    ax.semilogy(np.clip(gd_losses(eta), None, 3e3), label=label)
ax.set_xlabel('반복 횟수')
ax.set_ylabel('MSE 손실 (로그 스케일)')
ax.set_ylim(0.5, 3e3)
ax.legend()
fig.tight_layout()
fig.savefig('fig_gd_lr.png', dpi=200)
plt.close(fig)

# ── 책 4.3의 2차 데이터: y = 0.5x² + x + 2 + 잡음 ────────────────────
np.random.seed(42)
mq = 100
Xq = 6 * np.random.rand(mq, 1) - 3
yq = 0.5 * Xq ** 2 + Xq + 2 + np.random.randn(mq, 1)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import learning_curve

# fig_lc — 학습 곡선: 과소적합(1차) vs 과대적합(10차) (전체 폭 2.5:1)
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=True)
for ax, deg, title in [(axes[0], 1, '1차 (과소적합)'),
                       (axes[1], 10, '10차 (과대적합)')]:
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    sizes, tr, va = learning_curve(
        model, Xq, yq.ravel(), cv=5,
        scoring='neg_root_mean_squared_error',
        train_sizes=np.linspace(0.06, 1.0, 20))
    ax.plot(sizes, -tr.mean(axis=1), label='훈련 RMSE')
    ax.plot(sizes, -va.mean(axis=1), label='검증 RMSE')
    ax.set_xlabel(f'훈련 샘플 수 — {title}')
    ax.set_ylim(0, 3.5)
axes[0].set_ylabel('RMSE')
axes[0].legend()
fig.tight_layout()
fig.savefig('fig_lc.png', dpi=200)
plt.close(fig)

# fig_reg — 10차 다항 + 릿지 규제 강도 3종 (불릿 옆 3:2 자리)
from sklearn.linear_model import Ridge
np.random.seed(1)
ms = 24
Xs = 6 * np.random.rand(ms, 1) - 3
ys = 0.5 * Xs ** 2 + Xs + 2 + np.random.randn(ms, 1)
xg = np.linspace(-3, 3, 300).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(7.2, 4.55))
ax.plot(Xs, ys, 'o', color='0.6', ms=6, ls='none', label='훈련 데이터')
for alpha, label in [(0, 'α = 0 (규제 없음)'),
                     (1e-3, 'α = 0.001'),
                     (10, 'α = 10')]:
    model = make_pipeline(PolynomialFeatures(10), StandardScaler(),
                          Ridge(alpha=alpha) if alpha else LinearRegression())
    model.fit(Xs, ys)
    ax.plot(xg, model.predict(xg), label=label)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-2, 12)
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig('fig_reg.png', dpi=200)
plt.close(fig)

# fig_logit — 붓꽃 꽃잎 너비 vs P(버지니카) (불릿 옆 3:2 자리)
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris(as_frame=True)
Xi = iris.data[['petal width (cm)']].values
yi = (iris.target == 2).astype(int)          # Iris virginica 여부
log_reg = LogisticRegression(random_state=42).fit(Xi, yi)

xw = np.linspace(0, 3, 500).reshape(-1, 1)
proba = log_reg.predict_proba(xw)
boundary = xw[proba[:, 1] >= 0.5][0, 0]
print('결정 경계 (petal width) =', round(boundary, 2), 'cm')

fig, ax = plt.subplots(figsize=(7.2, 4.55))
ax.plot(xw, proba[:, 1], label='버지니카일 확률')
ax.plot(xw, proba[:, 0], label='버지니카가 아닐 확률')
ax.axvline(boundary, color='0.5', lw=1.2, ls=':')
ax.text(boundary + 0.06, 0.87, f'결정 경계 {boundary:.2f}cm',
        fontsize=12, color='0.3')
ax.plot(Xi[yi == 1], np.ones(yi.sum()) * 1.02, marker='^',
        ls='none', color='0.0', ms=5, label='버지니카 샘플')
ax.plot(Xi[yi == 0], np.zeros((yi == 0).sum()) - 0.02, marker='s',
        ls='none', color='0.6', ms=4, label='그 외 샘플')
ax.set_xlabel('꽃잎 너비 (cm)')
ax.set_ylabel('확률')
ax.legend(fontsize=11, loc='center right')
fig.tight_layout()
fig.savefig('fig_logit.png', dpi=200)
plt.close(fig)

print('그림 4장 저장 완료')
