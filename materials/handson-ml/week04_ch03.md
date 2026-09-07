# 핸즈온 ML Ch3 — 분류

- 교재: 『핸즈온 머신러닝 3판』 1부 3장 (pp. 143~175)
- 사용 주차: 4주차 전반 (분류 평가 지표)
- 관련 슬라이드: [slides/week04/deck.md](../../slides/week04/deck.md)
- 교재 노트북: `03_classification.ipynb`

이 수업에서 이 장은 **발췌**로 다룬다. 프로젝트의 타깃은 회귀(ETA·통행시간)지만, 3.3절의 오차 행렬·정밀도·재현율·임곗값 트레이드오프는 "지표를 왜 하나만 보면 안 되는가"를 가장 명확하게 보여주는 부분이라 그대로 가져간다. 3.6·3.7절(다중 레이블·다중 출력)은 개념만 확인한다.

## 장의 구성

| 절 | 제목 | 교재 쪽 | 이 수업에서 |
|---|---|---:|---|
| 3.1 | MNIST | 143 | 데이터 소개 |
| 3.2 | 이진 분류기 훈련 | 147 | 훑고 지나감 |
| 3.3 | **성능 측정** | 148 | **집중** |
| 3.4 | 다중 분류 | 163 | OvR·OvO 개념만 |
| 3.5 | **오류 분석** | 167 | **집중** |
| 3.6 | 다중 레이블 분류 | 171 | 개념만 |
| 3.7 | 다중 출력 분류 | 173 | 개념만 |

---

## 3.1 MNIST

고등학생과 미국 인구 조사국 직원들이 손으로 쓴 **70,000개의 작은 숫자 이미지**인 MNIST 데이터셋을 사용한다. 학습용으로 아주 많이 사용되기 때문에 머신러닝 분야의 'Hello World'라 불린다.

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
X.shape   # (70000, 784)
y.shape   # (70000,)
```

`sklearn.datasets` 패키지의 함수는 대부분 세 종류다.

| 접두어 | 용도 |
|---|---|
| `fetch_*` | 실전 데이터셋을 다운로드 |
| `load_*` | 사이킷런에 번들로 포함된 소규모 데이터셋 로드 |
| `make_*` | 테스트에 유용한 가짜 데이터셋 생성 |

이미지가 70,000개 있고 각 이미지에는 784개의 특성이 있다. 이미지가 28×28 픽셀이기 때문이다. 각각의 특성은 단순히 0(흰색)부터 255(검은색)까지의 픽셀 강도를 나타낸다.

```python
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()
y[0]     # '5'
```

`fetch_openml()`이 반환한 MNIST 데이터셋은 **이미 훈련 세트(앞쪽 60,000개)와 테스트 세트(뒤쪽 10,000개)로 나뉘어 있다.**

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

훈련 세트는 이미 섞여 있어서 모든 교차 검증 폴드를 비슷하게 만든다(하나의 폴드라도 특정 숫자가 누락되면 안 된다). 어떤 학습 알고리즘은 훈련 샘플의 순서에 민감해서 많은 비슷한 샘플이 연이어 나타나면 성능이 나빠진다.

> **NOTE(교재 각주).** 어떤 경우에는 섞는 것이 좋지 않다. **주식 가격이나 날씨 예보 같은 시계열 데이터를 다룰 때**다. 15장에서 이런 경우를 살펴본다.

> **모빌리티 적용.** 이 각주가 우리 프로젝트에 직접 해당한다. BIS·통행속도 데이터는 섞으면 안 된다.

---

## 3.2 이진 분류기 훈련

문제를 단순화해서 숫자 5만 식별한다. 이 '5-감지기'는 '5'와 '5 아님' 두 개의 클래스를 구분하는 **이진 분류기**다.

```python
y_train_5 = (y_train == '5')   # 5는 True고, 다른 숫자는 모두 False
y_test_5 = (y_test == '5')

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])   # array([ True])
```

`SGDClassifier`는 **확률적 경사 하강법(SGD)** 분류기다. 매우 큰 데이터셋을 효율적으로 처리할 수 있고, 한 번에 하나씩 훈련 샘플을 독립적으로 처리할 수 있어 온라인 학습에 잘 들어맞는다. (SGD 자체는 4장에서 다룬다.)

---

## 3.3 성능 측정

> 분류기 평가는 회귀 모델보다 훨씬 어렵기 때문에 여기서는 이 주제에 많은 지면을 할애할 것입니다.

### 3.3.1 교차 검증을 사용한 정확도 측정

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.95035, 0.96035, 0.9604 ])
```

**모든 교차 검증 폴드에 대해 정확도가 95% 이상이다. 아주 놀랍지 않은가?** 너무 흥분하지 말고 모든 이미지를 가장 많이 등장하는 클래스(여기서는 음성 클래스, 즉 '5 아님')로 분류하는 더미 분류기를 만들어 비교한다.

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))    # False — True로 예측된 것이 없습니다.

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.90965, 0.90965, 0.90965])
```

**정확도가 90% 이상 나왔다.** 이미지의 10% 정도만 숫자 5이기 때문에 무조건 '5 아님'으로 예측하면 정확히 맞출 확률이 90%다.

> 이 예제는 **정확도를 분류기의 성능 측정 지표로 선호하지 않는 이유**를 보여준다. 특히 **불균형한 데이터셋**을 다룰 때(즉, 어떤 클래스가 다른 것보다 월등히 많은 경우) 더욱 그렇다.

> **교차 검증 구현 (교재 박스).** 사이킷런이 제공하는 기능보다 교차 검증 과정을 더 많이 제어해야 할 때는 직접 구현하면 된다.
>
> ```python
> from sklearn.model_selection import StratifiedKFold
> from sklearn.base import clone
>
> skfolds = StratifiedKFold(n_splits=3)   # 데이터셋이 미리 섞여 있지 않다면
>                                         # shuffle=True를 추가하세요.
> for train_index, test_index in skfolds.split(X_train, y_train_5):
>     clone_clf = clone(sgd_clf)
>     X_train_folds = X_train[train_index]
>     y_train_folds = y_train_5[train_index]
>     X_test_fold = X_train[test_index]
>     y_test_fold = y_train_5[test_index]
>     clone_clf.fit(X_train_folds, y_train_folds)
>     y_pred = clone_clf.predict(X_test_fold)
>     n_correct = sum(y_pred == y_test_fold)
>     print(n_correct / len(y_pred))   # 0.95035, 0.96035, 0.9604
> ```

### 3.3.2 오차 행렬

오차 행렬의 기본 아이디어는 **모든 A/B 쌍에 대해 클래스 A의 샘플이 클래스 B로 분류된 횟수를 세는 것**이다.

오차 행렬을 만들려면 실제 타깃과 비교할 수 있도록 예측값을 만들어야 한다. **테스트 세트로 예측을 만들 수 있지만 여기서 사용하면 안 된다**(테스트 세트는 프로젝트의 맨 마지막에 분류기가 출시 준비를 마치고 나서 사용된다). 대신 `cross_val_predict()`를 사용한다.

```python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

`cross_val_predict()`는 k-폴드 교차 검증을 수행하지만 평가 점수가 아니라 **각 테스트 폴드에서 얻은 예측을 반환**한다. 즉 훈련 세트의 모든 샘플에 대해 **깨끗한**(모델이 훈련하는 동안 보지 못했던 데이터에 대한) 예측을 얻는다.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)
cm
# array([[53892,   687],
#        [ 1891,  3530]])
```

행은 **실제 클래스**, 열은 **예측한 클래스**를 나타낸다.

| | 예측: 5 아님 | 예측: 5 |
|---|---:|---:|
| **실제: 5 아님** | 53,892 (진짜 음성 TN) | 687 (거짓 양성 FP, **1종 오류**) |
| **실제: 5** | 1,891 (거짓 음성 FN, **2종 오류**) | 3,530 (진짜 양성 TP) |

완벽한 분류기라면 진짜 양성과 진짜 음성만 가지고 있을 것이므로 오차 행렬의 **주대각선**(왼쪽 위에서 오른쪽 아래로)만 0이 아닌 값이 된다.

**[식 3-1] 정밀도(precision)** — 양성 예측의 정확도

```
정밀도 = TP / (TP + FP)
```

가장 간단한 방법은 제일 확신이 높은 샘플에 대해 양성 예측을 하고 나머지는 모두 음성 예측을 하는 분류기를 만드는 것이다. 이 양성 예측이 맞는다면 정밀도는 100%다(1/1). 당연히 이런 분류기는 다른 모든 양성 샘플을 무시하기 때문에 그리 유용하지 않다.

**[식 3-2] 재현율(recall)** — 분류기가 정확하게 감지한 양성 샘플의 비율. **민감도** 또는 **진짜 양성 비율(TPR)**이라고도 한다.

```
재현율 = TP / (TP + FN)
```

### 3.3.3 정밀도와 재현율

```python
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)   # 0.8370879772350012  == 3530 / (687 + 3530)
recall_score(y_train_5, y_train_pred)      # 0.6511713705958311  == 3530 / (1891 + 3530)
```

**정확도에서 봤을 때만큼 멋져 보이지는 않는다.** 5로 판별된 이미지 중 83.7%만 정확하고, 전체 숫자 5에서 65.1%만 감지했다.

**[식 3-3] F₁ 점수** — 정밀도와 재현율의 **조화 평균**

```
F₁ = 2 / (1/정밀도 + 1/재현율) = 2 × (정밀도 × 재현율) / (정밀도 + 재현율)
   = TP / (TP + (FN + FP)/2)
```

보통의 평균은 모든 값을 동일하게 취급하지만 **조화 평균은 낮은 값에 훨씬 더 높은 비중을 둔다.** 결과적으로 F₁ 점수가 높아지려면 재현율과 정밀도가 모두 높아야 한다.

```python
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)   # 0.7325171197343846
```

**하지만 F₁ 점수가 높은 것이 항상 바람직한 것은 아니다.** 상황에 따라 정밀도가 중요할 수도 있고 재현율이 중요할 수도 있다. 교재의 두 예시가 이 절의 핵심이다.

| 상황 | 원하는 것 | 이유 |
|---|---|---|
| 어린아이에게 안전한 동영상을 걸러내는 분류기 | **높은 정밀도**, 낮은 재현율 감수 | 좋은 동영상이 많이 제외되더라도 정말 나쁜 동영상이 노출되면 안 된다 |
| 감시 카메라로 좀도둑을 잡아내는 분류기 | **높은 재현율**(99%), 정밀도 30%도 감수 | 경비원이 잘못된 호출을 종종 받겠지만 거의 모든 좀도둑을 잡는다 |

안타깝게도 정밀도와 재현율 모두 얻을 수는 없다. 정밀도를 올리면 재현율이 줄고 그 반대도 마찬가지다. 이를 **정밀도/재현율 트레이드오프**라 한다.

### 3.3.4 정밀도/재현율 트레이드오프

`SGDClassifier`는 **결정 함수**를 사용하여 각 샘플의 점수를 계산한다. 점수가 **결정 임곗값**보다 크면 샘플을 양성 클래스에 할당하고 그렇지 않으면 음성 클래스에 할당한다.

교재의 그림 3-4가 보여주는 것: 임곗값을 올리면 재현율은 낮아지고 (보통) 정밀도는 높아진다.

| 임곗값 위치 | 정밀도 | 재현율 |
|---|---|---|
| 왼쪽(낮음) | 6/8 = 75% | 6/6 = 100% |
| 가운데 | 4/5 = 80% | 4/6 = 67% |
| 오른쪽(높음) | 3/3 = 100% | 3/6 = 50% |

**사이킷런에서 임곗값을 직접 지정할 수는 없지만 예측에 사용한 점수는 확인할 수 있다.** `predict()` 대신 `decision_function()`을 호출한다.

```python
y_scores = sgd_clf.decision_function([some_digit])
y_scores          # array([2164.22030239])
threshold = 0
y_some_digit_pred = (y_scores > threshold)    # array([ True])

threshold = 3000
y_some_digit_pred = (y_scores > threshold)    # array([False])
```

적절한 임곗값을 정하려면 먼저 `cross_val_predict()`로 훈련 세트의 모든 샘플의 **결정 점수**를 구한다.

```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

> **NOTE(교재).** 그림 3-5에서 정밀도 곡선이 재현율 곡선보다 왜 더 울퉁불퉁한지 의아할 수 있다. 이는 **임곗값을 올리더라도 정밀도가 낮아질 때가 가끔 있기 때문이다**(일반적으로는 높아져야 한다). 반면 **재현율은 임곗값이 올라감에 따라 줄어들 수밖에 없어** 부드러운 곡선이 된다.

정밀도 90%를 달성하는 것이 목표라면, 정밀도가 최소 90%가 되는 가장 낮은 임곗값을 찾는다.

```python
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision      # 3370.0194991439557

y_train_pred_90 = (y_scores >= threshold_for_90_precision)

precision_score(y_train_5, y_train_pred_90)   # 0.9000345901072293
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)  # 0.4799852425751706
```

정밀도 90%를 달성한 분류기를 만들었다. 보다시피 **임곗값을 충분히 크게 지정하기만 하면 거의 모든 정밀도의 분류기를 손쉽게 만들 수 있다.** 하지만 재현율이 너무 낮다면 높은 정밀도의 분류기는 전혀 유용하지 않다. 많은 애플리케이션에 재현율 48%는 훌륭한 값이 전혀 아니다.

> **TIP(교재).** 누군가 **"99% 정밀도를 달성하자"**라고 말하면 반드시 **"재현율 얼마에서?"**라고 물어봐야 한다.

### 3.3.5 ROC 곡선

**수신기 조작 특성(ROC) 곡선**은 정밀도/재현율 곡선과 매우 비슷하지만, 정밀도에 대한 재현율 곡선이 아니라 **거짓 양성 비율(FPR)에 대한 진짜 양성 비율(TPR, 재현율의 다른 이름)**의 곡선이다.

- FPR(폴-아웃)은 양성으로 잘못 분류된 음성 샘플의 비율이다.
- FPR = 1 − TNR이고, TNR(진짜 음성 비율)을 **특이도**라고 한다.
- **그러므로 ROC 곡선은 민감도(재현율)에 대한 1−특이도 그래프다.**

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

여기에도 트레이드오프가 있다. 재현율(TPR)이 높을수록 분류기가 만드는 거짓 양성 비율(FPR)이 늘어난다. 점선은 완전한 랜덤 분류기의 ROC 곡선이며, **좋은 분류기는 이 점선에서 최대한 멀리 떨어져 있어야 한다(왼쪽 위 모서리).**

**곡선 아래의 면적(AUC)**을 측정해 분류기들을 비교할 수 있다. 완벽한 분류기는 ROC의 AUC가 1이고, 완전한 랜덤 분류기는 0.5다.

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)    # 0.9604938554008616
```

> **TIP(교재).** ROC 곡선이 정밀도/재현율(PR) 곡선과 비슷해서 어떤 것을 사용해야 할지 궁금할 수 있다. 일반적으로 **양성 클래스가 드물거나 거짓 음성보다 거짓 양성이 더 중요할 때 PR 곡선을 사용하고 그렇지 않으면 ROC 곡선을 사용한다.** 예를 들어 ROC 곡선(그리고 ROC의 AUC 점수)을 보면 매우 좋은 분류기라고 생각할 수 있지만, 이는 음성(5 아님)에 비해 양성(5)이 매우 적기 때문이다. 반면 PR 곡선은 분류기의 성능이 얼마만큼 개선될 수 있는지 보여준다.

`RandomForestClassifier`와 비교해본다. `RandomForestClassifier`는 작동 방식 때문에 `decision_function()`을 제공하지 않고, 대신 각 샘플에 대한 클래스 확률을 반환하는 `predict_proba()`를 제공한다.

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_probas_forest[:2]
# array([[0.11, 0.89],
#        [0.99, 0.01]])
```

> **CAUTION(교재).** 이는 실제 확률이 아닌 **추정 확률**이다. 예를 들어 모델이 50%에서 60% 사이의 추정 확률로 양성으로 분류한 모든 이미지를 살펴보면 약 94%가 실제로 양성이다. 이 경우 모델의 추정 확률이 너무 낮았지만 모델 역시 **과신**할 수 있다. `sklearn.calibration` 패키지에는 추정 확률을 보정하여 실제 확률에 훨씬 가깝게 만드는 도구가 포함되어 있다.

```python
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)

y_train_pred_forest = y_probas_forest[:, 1] >= 0.5   # 양성 클래스 확률 ≥ 50%
f1_score(y_train_5, y_train_pred_forest)             # 0.9242275142688446
roc_auc_score(y_train_5, y_scores_forest)            # 0.9983436731328145
```

정밀도와 재현율 점수를 계산하면 **99.1% 정밀도와 86.6% 재현율**이 나온다. 랜덤 포레스트의 PR 곡선이 SGD 분류기의 곡선보다 훨씬 더 좋아 보인다.

> **모빌리티 적용.** 회귀 프로젝트에서도 이 절의 사고방식을 그대로 쓴다. "MAE 3분"이라는 단일 숫자만 보고할 것이 아니라 **"지연 5분 이상을 얼마나 잡아내는가(재현율), 그때 오탐은 얼마인가(정밀도)"**를 함께 봐야 한다. ETA 예측을 "정시/지연" 이진 판정으로 바꿔 오차 행렬을 그려보는 것이 좋은 오차 분석 방법이다.

---

## 3.4 다중 분류

이진 분류기는 두 개의 클래스를 구별하는 반면 **다중 분류기**(또는 다항 분류기)는 둘 이상의 클래스를 구별할 수 있다.

- `LogisticRegression`, `RandomForestClassifier`, `GaussianNB` 같은 일부 알고리즘은 여러 개의 클래스를 **직접 처리**할 수 있다.
- `SGDClassifier`, `SVC` 같은 다른 알고리즘은 **이진 분류만 가능**하다.

이진 분류기를 여러 개 사용해 다중 클래스를 분류하는 기법이 두 가지 있다.

| 전략 | 방식 | 필요한 분류기 수 (클래스 N개) |
|---|---|---|
| **OvR**(one-versus-the-rest, OvA) | 특정 숫자 하나만 구분하는 이진 분류기 10개를 훈련. 결정 점수 중 가장 높은 것을 선택 | N |
| **OvO**(one-versus-one) | 0과 1 구별, 0과 2 구별, 1과 2 구별 등 각 숫자의 조합마다 훈련 | N × (N−1) / 2 (MNIST는 45개) |

**OvO 전략의 주요 장점은 각 분류기의 훈련에 전체 훈련 세트 중 구별할 두 클래스에 해당하는 샘플만 있으면 된다는 점이다.** (서포트 벡터 머신 같은) 일부 알고리즘은 훈련 세트의 크기에 민감해서 큰 훈련 세트에서 몇 개의 분류기를 훈련시키는 것보다 작은 훈련 세트에서 많은 분류기를 훈련시키는 쪽이 빠르므로 OvO를 선호한다. **하지만 대부분의 이진 분류 알고리즘에서는 OvR을 선호한다.**

사이킷런은 알고리즘에 따라 자동으로 OvR 또는 OvO를 실행한다.

```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])   # y_train_5가 아닌 y_train을 사용합니다.
svm_clf.predict([some_digit])                 # array(['5'], dtype=object)

some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)
# array([[ 3.79, 0.73, 6.06, 8.3 , -0.29, 9.3 , 1.75, 2.77, 7.21, 4.82]])
class_id = some_digit_scores.argmax()    # 5
svm_clf.classes_[class_id]               # '5'
```

강제로 지정하려면 `OneVsOneClassifier`나 `OneVsRestClassifier`를 사용한다.

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
len(ovr_clf.estimators_)     # 10
```

`SGDClassifier`로 다중 분류를 하면 사이킷런이 OvR 전략을 사용해 10개의 이진 분류기를 훈련한다.

```python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# array([0.87365, 0.85835, 0.8689 ])
```

**입력의 스케일을 조정하면 정확도를 89.1% 이상으로 높일 수 있다.**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
# array([0.8983, 0.891 , 0.9018])
```

---

## 3.5 오류 분석

가능성이 높은 모델을 하나 찾았다고 가정하고 이 모델의 성능을 향상시킬 방법을 찾는다. **한 가지 방법은 생성된 오류의 종류를 분석하는 것이다.**

```python
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()
```

클래스가 10개라서 오차 행렬에 상당히 많은 숫자가 포함되므로 읽기 어렵다. 컬러 그래프로 나타내면 분석하기가 훨씬 쉽다.

**5번 행과 5번 열의 대각선에 있는 셀은 다른 숫자보다 약간 더 어둡게 보인다.** 이는 모델이 5에서 더 많은 오류를 범했거나 데이터 집합에 다른 숫자보다 5가 적기 때문일 것이다. 따라서 각 값을 **해당 클래스의 총 이미지 수로 나누어(행의 합으로 나누어) 오차 행렬을 정규화하는 것이 중요하다.**

```python
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true", values_format=".0%")
plt.show()
```

이제 5 이미지의 **82%**만 올바르게 분류되었다는 것을 쉽게 알 수 있다. 모델이 5 이미지에서 가장 많이 범한 오류는 8로 잘못 분류한 것인데 전체 5의 10%에서 이 오류가 발생했다. 하지만 8은 2%만 5로 잘못 분류되었다. 즉 **오차 행렬은 일반적으로 대칭이 아니다.**

오류를 더 눈에 띄게 만들려면 **올바른 예측에 대한 가중치를 0으로 설정**한다.

```python
sample_weight = (y_train_pred != y_train)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%")
```

**클래스 8의 열이 매우 밝아진 것으로 보아 많은 이미지가 8로 잘못 분류되었음을 알 수 있다.** 사실 이는 거의 모든 클래스에서 가장 많이 발생하는 분류 오류다.

> **CAUTION.** 이 그래프에서 백분율을 해석하는 방법에 주의해야 한다. **올바른 예측을 제외했다는 점을 기억해야 한다.** 예를 들어 7번 행, 9번 열의 36%는 모든 7 이미지 중 36%가 9로 잘못 분류되었다는 뜻이 아니다. 이는 **모델이 7 이미지에서 발생한 오류 중 36%가 9로 잘못 분류되었다**는 의미다. 실제로는 7 이미지 중 3%만이 9로 잘못 분류되었다.

`normalize="pred"`로 지정하면 열 단위로 정규화할 수 있다. 예를 들어 **잘못 분류된 7의 56%가 실제로는 9**라는 것을 알 수 있다.

오차 행렬을 분석하면 분류기의 성능 향상 방안에 관한 인사이트를 얻을 수 있다. 여기서는 8로 잘못 분류되는 것을 줄이도록 개선할 필요가 있다.

- 8처럼 보이는(하지만 실제로 8은 아닌) 숫자의 훈련 데이터를 더 많이 모아서 실제 8과 구분하도록 분류기를 학습시킬 수 있다.
- 분류기에 도움될 만한 특성을 더 찾아볼 수 있다. 예를 들어 동심원의 수를 세는 알고리즘을 작성한다(8은 2개, 6은 1개, 5는 0개).
- 동심원과 같은 패턴이 드러나도록 이미지를 전처리해볼 수 있다.

개별 오류를 분석하는 것도 좋다. 3과 5의 샘플을 오차 행렬 스타일로 그려보면(그림 3-11), 분류기가 잘못 분류한 숫자의 일부는 정말 잘못 쓰여 있어서 사람도 분류하기 어려울 것 같지만 대부분의 잘못 분류된 이미지는 확실한 오류로 보인다.

> 이 예제는 선형 모델인 `SGDClassifier`를 사용한다는 점을 기억해두자. 선형 분류기는 클래스마다 픽셀에 가중치를 할당하고 새로운 이미지에 대해 단순히 픽셀 강도의 가중치 합을 클래스의 점수로 계산한다. 그러므로 **몇 개의 픽셀만 다른 3과 5를 모델이 쉽게 혼동한다.**

3과 5의 주요 차이는 위쪽 선과 아래쪽 호를 이어주는 작은 직선의 위치다. 다시 말해 **분류기는 이미지의 위치나 회전 방향에 매우 민감하다.** 간단한 접근 방식은 훈련 이미지를 약간 이동시키거나 회전된 변형 이미지로 훈련 집합을 보강하는 것이다. 이를 **데이터 증식**(data augmentation)이라 한다.

> **모빌리티 적용.** 이 절이 이 장에서 프로젝트로 가장 직접 이어진다. **"평균 지표 하나가 아니라, 어디서 어떤 종류의 오차가 나는지 행렬로 펼쳐 본다."** ETA에서는 (실제 소요시간 구간) × (예측 구간)으로 행렬을 만들어보면 특정 시간대·특정 구간에 오차가 몰려 있는지 바로 보인다.

---

## 3.6 다중 레이블 분류

분류기가 샘플마다 여러 개의 클래스를 출력해야 할 때도 있다. 얼굴 인식 분류기가 앨리스와 찰리가 있는 사진을 보면 `[1, 0, 1]`을 출력해야 한다('앨리스 있음, 밥 없음, 찰리 있음'). 이처럼 **여러 개의 이진 꼬리표를 출력하는 분류 시스템**을 다중 레이블 분류 시스템이라 한다.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])    # array([[False,  True]])
```

평가는 각 레이블의 F₁ 점수를 구하고 평균 점수를 계산하는 방법이 있다.

```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")   # 0.976410265560605
```

이 코드는 **모든 레이블의 가중치가 같다고 가정한 것이다.** 레이블에 클래스의 **지지도**(타깃 레이블에 속한 샘플 수)를 가중치로 주려면 `average="weighted"`로 설정한다.

`SVC`처럼 기본적으로 다중 레이블 분류를 지원하지 않는 분류기라면 레이블당 하나의 모델을 학습시키는 전략이 가능하다. 그러나 이 전략은 **레이블 간의 의존성을 포착하기 어렵다.** 큰 숫자(7, 8, 9)는 짝수보다 홀수일 가능성이 두 배 높지만, '홀수' 레이블에 대한 분류기는 '큰 값' 레이블 분류기가 무엇을 예측했는지 알 수 없다. 이 문제를 해결하기 위해 모델을 **체인**으로 구성할 수 있다.

```python
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])
chain_clf.predict([some_digit])   # array([[0., 1.]])
```

---

## 3.7 다중 출력 분류

**다중 출력 다중 클래스 분류**(간단히 다중 출력 분류)는 다중 레이블 분류에서 한 레이블이 다중 클래스가 될 수 있도록 일반화한 것이다.

교재의 예는 이미지에서 잡음을 제거하는 시스템이다. 잡음이 많은 숫자 이미지를 입력으로 받아 깨끗한 숫자 이미지를 픽셀의 강도를 담은 배열로 출력한다. 분류기의 출력이 다중 레이블(픽셀당 한 레이블)이고 각 레이블은 값을 여러 개 가진다(0부터 255까지 픽셀 강도).

```python
np.random.seed(42)
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
```

> **NOTE(교재).** 이 예에서처럼 **분류와 회귀 사이의 경계는 때때로 모호하다.** 확실히 픽셀 강도 예측은 분류보다 회귀와 비슷하다. 더욱이 다중 출력 시스템이 분류 작업에 국한되지도 않는다.

---

## 연습문제 (교재 원문)

1. MNIST 데이터셋으로 분류기를 만들어 테스트 세트에서 97% 정확도를 달성해보세요. (힌트: `KNeighborsClassifier`가 이 작업에 아주 잘 맞습니다. `weights`와 `n_neighbors` 하이퍼파라미터로 그리드 서치를 시도해보세요.)
2. MNIST 이미지를 (왼, 오른, 위, 아래) 어느 방향으로든 한 픽셀 이동시킬 수 있는 함수를 만들어보세요. 그런 다음 훈련 세트에 있는 각 이미지에 대해 네 개의 이동된 복사본을 만들어 훈련 세트에 추가하세요. 마지막으로 이 확장된 데이터셋에서 최선의 모델을 훈련시키고 테스트 세트에서 정확도를 측정해보세요. 인위적으로 훈련 세트를 늘리는 이 기법을 **데이터 증식** 또는 **훈련 세트 확장**이라고 합니다.
3. 타이타닉 데이터셋에 도전해보세요.

---

## 수업용 정리

| 개념 | 4주차에서 사용하는 형태 |
|---|---|
| 불균형 데이터에서 정확도의 함정 | 더미 분류기 90% 예시 그대로 |
| 오차 행렬 (TN/FP/FN/TP) | 표 하나로 정리 |
| 정밀도 vs 재현율 | 동영상 필터 / 좀도둑 감지 두 예시 |
| F₁ 점수는 조화 평균 | 낮은 값에 비중을 둔다는 점 |
| 결정 임곗값 트레이드오프 | "99% 정밀도? 재현율 얼마에서?" |
| PR 곡선 vs ROC 곡선 | 양성이 드물면 PR |
| 정규화된 오차 행렬 | 프로젝트 오차 분석의 원형 |

## 확인 문제

1. 5의 비율이 10%인 데이터에서 정확도 90%가 왜 의미 없는가?
2. 오차 행렬에서 1종 오류와 2종 오류는 각각 무엇인가?
3. 재현율을 높이면 정밀도는 어떻게 되는가? 그 이유는?
4. 양성 클래스가 매우 드물 때 ROC 곡선 대신 PR 곡선을 쓰는 이유는?
5. `cross_val_predict()`가 반환하는 예측이 "깨끗하다"는 것은 무슨 뜻인가?

## 관련 자료

- [Ch2 — 머신러닝 프로젝트 처음부터 끝까지](week03_ch02.md)
- [Ch4 — 모델 훈련](week04_ch04.md)
- [주차별 일정](../../schedule.md)
- [공식 실습 코드](https://github.com/rickiepark/handson-ml3)
