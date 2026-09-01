"""5주차 — 핸즈온 Ch6·Ch7 강의용 그림 (공개·합성 데이터로 직접 생성).

교재 그림을 캡처하지 않고 같은 내용을 흑백으로 재현한다.
실행하면 코드 슬라이드에 쓸 수치도 함께 출력한다.
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

# fig_tree — 붓꽃 결정 트리 (깊이 2, 불릿 옆 3:2 자리)
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
Xi = iris.data[:, 2:]                      # 꽃잎 길이·너비
yi = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(Xi, yi)

fig, ax = plt.subplots(figsize=(7.2, 4.55))
plot_tree(tree_clf, ax=ax, impurity=True, rounded=True,
          feature_names=['꽃잎 길이 (cm)', '꽃잎 너비 (cm)'],
          class_names=['세토사', '버시컬러', '버지니카'],
          fontsize=11)
fig.tight_layout()
fig.savefig('fig_tree.png', dpi=200)
plt.close(fig)

# fig_tree_reg — 회귀 트리의 계단형 예측 (불릿 옆 3:2 자리)
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
m = 200
Xr = np.sort(6 * np.random.rand(m, 1) - 3, axis=0)
yr = 0.5 * Xr ** 2 + Xr + 2 + np.random.randn(m, 1)
xg = np.linspace(-3, 3, 600).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(7.2, 4.55))
ax.plot(Xr, yr, 'o', color='0.75', ms=4, ls='none', label='훈련 데이터')
for depth, label in [(2, '깊이 2 (구간 4개)'),
                     (None, '깊이 제한 없음 (과대적합)')]:
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    reg.fit(Xr, yr)
    ax.plot(xg, reg.predict(xg), label=label)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig('fig_tree_reg.png', dpi=200)
plt.close(fig)

# ── moons 데이터: 투표·배깅·랜덤 포레스트 수치 ───────────────────────
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (VotingClassifier, BaggingClassifier,
                              RandomForestClassifier,
                              GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

Xm, ym = make_moons(n_samples=500, noise=0.30, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(Xm, ym, random_state=42)

single_tree = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
print('단일 결정 트리      :', round(single_tree.score(Xte, yte), 3))

voting = VotingClassifier(estimators=[
    ('lr', LogisticRegression(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('svc', SVC(random_state=42))])
voting.fit(Xtr, ytr)
for name, clf in voting.named_estimators_.items():
    print(f'  {name:3s}             :', round(clf.score(Xte, yte), 3))
print('직접 투표 앙상블    :', round(voting.score(Xte, yte), 3))

bag = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                        n_estimators=500, max_samples=100,
                        oob_score=True, random_state=42)
bag.fit(Xtr, ytr)
print('배깅 500그루        :', round(bag.score(Xte, yte), 3),
      '/ OOB', round(bag.oob_score_, 3))

# fig_imp — 캘리포니아 주택 RF 특성 중요도 (불릿 옆 3:2 자리)
from sklearn.ensemble import RandomForestRegressor

housing = pd.read_csv('../week03/data/housing.csv')
X = housing.drop(columns='median_house_value')
X = pd.get_dummies(X, columns=['ocean_proximity'])
X = X.fillna(X.median(numeric_only=True))
y = housing['median_house_value']

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns)
imp = imp.sort_values().tail(8)
name_ko = {
    'median_income': '중간 소득', 'longitude': '경도', 'latitude': '위도',
    'housing_median_age': '주택 연식', 'population': '인구',
    'total_rooms': '방 개수', 'total_bedrooms': '침실 개수',
    'households': '가구 수', 'ocean_proximity_INLAND': '내륙 여부',
    'ocean_proximity_<1H OCEAN': '해안 1시간 이내',
}
imp.index = [name_ko.get(c, c) for c in imp.index]
print('RF 중요도 상위:', dict(imp.round(3).tail(4)))

fig, ax = plt.subplots(figsize=(7.2, 4.55))
ax.barh(imp.index, imp.values, color='0.4')
ax.set_xlabel('특성 중요도 (합계 1)')
fig.tight_layout()
fig.savefig('fig_imp.png', dpi=200)
plt.close(fig)

# fig_gbrt — 그레이디언트 부스팅의 누적 예측 (불릿 옆 3:2 자리)
gbrt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                 max_depth=2, random_state=42)
gbrt.fit(Xr, yr.ravel())
stages = {1: None, 5: None, 100: None}
for i, pred in enumerate(gbrt.staged_predict(xg), start=1):
    if i in stages:
        stages[i] = pred.copy()

fig, ax = plt.subplots(figsize=(7.2, 4.55))
ax.plot(Xr, yr, 'o', color='0.75', ms=4, ls='none', label='훈련 데이터')
for n, pred in stages.items():
    ax.plot(xg, pred, label=f'트리 {n}그루까지 합산')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig('fig_gbrt.png', dpi=200)
plt.close(fig)

print('그림 4장 저장 완료')
