# 핸즈온 ML Ch2 — 머신러닝 프로젝트 처음부터 끝까지

- 교재: 『핸즈온 머신러닝 3판』 1부 2장 (pp. 68~141)
- 사용 주차: 3주차 (End-to-End 프로젝트)
- 관련 슬라이드: [slides/week03/deck.md](../../slides/week03/deck.md)
- 교재 노트북: `02_end_to_end_machine_learning_project.ipynb`

교재는 이 장에서 "부동산 회사에 막 고용된 데이터 과학자"를 가정하고 가상의 예제 프로젝트를 처음부터 끝까지 진행한다. 부동산 비즈니스를 배우는 것이 목적이 아니라 **머신러닝의 주요 단계를 설명하는 것**이 목적이다.

## 교재가 제시하는 8단계

1. 큰 그림을 봅니다.
2. 데이터를 구합니다.
3. 데이터로부터 인사이트를 얻기 위해 탐색하고 시각화합니다.
4. 머신러닝 알고리즘을 위해 데이터를 준비합니다.
5. 모델을 선택하고 훈련시킵니다.
6. 모델을 미세 튜닝합니다.
7. 솔루션을 제시합니다.
8. 시스템을 론칭하고, 모니터링하고, 유지 보수합니다.

## 장의 구성

| 절 | 제목 | 교재 쪽 |
|---|---|---:|
| 2.1 | 실제 데이터로 작업하기 | 68 |
| 2.2 | 큰 그림 보기 | 70 |
| 2.3 | 데이터 가져오기 | 76 |
| 2.4 | 데이터 이해를 위한 탐색과 시각화 | 93 |
| 2.5 | 머신러닝 알고리즘을 위한 데이터 준비 | 101 |
| 2.6 | 모델 선택과 훈련 | 125 |
| 2.7 | 모델 미세 튜닝 | 130 |
| 2.8 | 론칭, 모니터링, 시스템 유지 보수 | 136 |
| 2.9 | 직접 해보세요! | 140 |

---

## 2.1 실제 데이터로 작업하기

머신러닝을 배울 때는 인공적으로 만들어진 데이터셋이 아닌 **실제 데이터셋**으로 실험해보는 것이 가장 좋다. 교재가 소개하는 데이터 출처는 다음과 같다.

| 구분 | 목록 |
|---|---|
| 공개 데이터 저장소 | OpenML, 캐글, PapersWithCode, UC 어바인 머신러닝 저장소, 아마존 AWS 데이터셋, 텐서플로 데이터셋 |
| 메타 포털 | 데이터 포털(dataportals.org), 오픈 데이터 모니터(opendatamonitor.eu) |
| 목록 페이지 | 위키백과 머신러닝 데이터셋 목록, Quora, 데이터셋 서브레딧 |

이 장에서 사용하는 것은 StatLib 저장소의 **캘리포니아 주택 가격 데이터셋**이다. 1990년 캘리포니아 인구 조사 데이터를 기반으로 하며, 교재는 교육 목적으로 범주형 특성을 추가하고 몇 가지 특성을 제외했다.

> **모빌리티 적용.** 이 수업은 3주차에 이 housing 데이터로 절차를 익히고, 5주차부터 데이콘 제주 통행속도 데이터, 프로젝트에서 서울시 BIS 데이터로 같은 절차를 반복한다.

---

## 2.2 큰 그림 보기

첫 번째 일은 캘리포니아 인구 조사 데이터를 사용해 주택 가격 모델을 만드는 것이다. 데이터는 **블록 그룹**(block group)마다 인구, 중간 소득, 중간 주택 가격을 담고 있다. 블록 그룹은 미국 인구 조사국이 샘플 데이터를 발표하는 데 사용하는 최소한의 지리적 단위이며 보통 600~3,000명의 인구를 나타낸다. 교재는 이를 간단히 **구역**이라 부른다.

> **TIP(교재).** 잘 훈련된 데이터 과학자로서 첫 번째로 할 일은 머신러닝 프로젝트 체크리스트를 준비하는 것이다. 〈부록 B〉에 준비된 것을 사용해도 된다.

### 2.2.1 문제 정의

상사에게 첫 번째로 할 질문은 **'비즈니스의 목적이 정확히 무엇인가요?'**이다. 모델을 만드는 것이 최종 목적은 아닐 것이다. 목적을 아는 것은 다음을 결정하기 때문에 아주 중요하다.

- 문제를 어떻게 구성할지
- 어떤 알고리즘을 선택할지
- 모델 평가에 어떤 성능 지표를 사용할지
- 모델 튜닝을 위해 얼마 만큼의 노력을 투여할지

이 예제에서 모델의 출력(구역의 중간 주택 가격 예측)은 다른 여러 신호와 함께 다음 머신러닝 시스템에 입력으로 사용된다. 뒤따르는 시스템은 해당 지역에 투자할 가치가 있는지 결정한다. 이 결정이 수익과 직결되기 때문에 올바르게 예측하는 것이 매우 중요하다.

> **파이프라인 (교재 박스).** 데이터 처리 **컴포넌트**들이 연속되어 있는 것을 데이터 **파이프라인**이라 한다. 보통 컴포넌트들은 비동기적으로 작동한다. 각 컴포넌트는 많은 데이터를 추출해 처리하고 그 결과를 다른 데이터 저장소로 보낸다. 각 컴포넌트는 완전히 독립적이며, 컴포넌트 사이의 인터페이스는 데이터 저장소뿐이다. 이는 시스템을 이해하기 쉽게 만들고, 각 팀은 각자의 컴포넌트에 집중할 수 있다. 한 컴포넌트가 중단되더라도 하위 컴포넌트는 문제가 생긴 컴포넌트의 마지막 출력을 사용해 (적어도 한동안은) 평상시와 같이 계속 작동할 수 있어 시스템이 매우 견고해진다. 한편 **모니터링이 적절히 되지 않으면 고장 난 컴포넌트를 한동안 모를 수 있다.** 데이터가 만들어진 지 오래되면 전체 시스템의 성능이 떨어진다.

두 번째 질문은 **'현재 솔루션은 어떻게 구성되어 있나요?'**이다. 현재 상황은 문제 해결 방법에 관한 정보를 제공할 뿐 아니라 **참고 성능**으로도 사용할 수 있다. 이 예제에서 현재는 전문가가 수동으로 추정하고 있으며, 실제 가격과 30% 이상 벗어나는 경우가 많다.

이제 시스템을 설계할 수 있다. 교재의 결론은 다음과 같다.

| 질문 | 답 | 근거 |
|---|---|---|
| 지도 방식 | 지도 학습 | 레이블(구역의 중간 주택 가격)된 훈련 샘플이 있다 |
| 작업 유형 | 회귀 | 모델이 값을 예측해야 한다 |
| 더 구체적으로 | **다중 회귀** | 예측에 사용할 특성이 여러 개다 |
| 그리고 | **단변량 회귀** | 구역마다 하나의 값을 예측한다 (여러 값이면 다변량 회귀) |
| 학습 방식 | 배치 학습 | 데이터에 연속적인 흐름이 없고, 데이터가 메모리에 들어갈 만큼 작다 |

> **TIP(교재).** 데이터가 매우 크면 (맵리듀스 기술을 사용하여) 배치 학습을 여러 서버로 분할하거나 온라인 학습 기법을 사용할 수 있다.

> **모빌리티 적용.** ETA 프로젝트를 같은 표로 채워보면 지도학습 / 회귀 / 다중 회귀 / 단변량 회귀 / 배치 학습이 된다. 다만 "현재 솔루션"에 해당하는 것이 BIS가 이미 제공하는 도착예정시간이므로, 이를 **입력 특성에서 제외**하고 **비교 대상(참고 성능)으로만** 사용한다.

### 2.2.2 성능 측정 지표 선택

회귀 문제의 전형적인 성능 지표는 **평균 제곱근 오차(RMSE)**다. 오차가 커질수록 이 값은 더욱 커지므로 예측에 얼마나 많은 오차가 있는지 가늠하게 해준다.

**[식 2-1] 평균 제곱근 오차(RMSE)**

```
RMSE(X, h) = sqrt( (1/m) * Σ (h(x⁽ⁱ⁾) − y⁽ⁱ⁾)² )
```

> **표기법 (교재 박스).** 이 책 전체에 걸쳐 사용하는 표기법이다.
> - `m`은 RMSE를 측정할 데이터셋에 있는 샘플 수다.
> - `x⁽ⁱ⁾`는 데이터셋에 있는 i번째 샘플(레이블 제외)의 전체 특성값의 벡터이고, `y⁽ⁱ⁾`는 해당 레이블이다.
> - `X`는 데이터셋에 있는 모든 샘플의 모든 특성값(레이블은 제외)을 포함하는 행렬이다. 샘플 하나가 하나의 행이며, i번째 행은 `x⁽ⁱ⁾`의 전치와 같다.
> - `h`는 시스템의 예측 함수이며 **가설**(hypothesis)이라고도 한다. 예측값은 `ŷ⁽ⁱ⁾ = h(x⁽ⁱ⁾)`로 쓴다.
> - 스칼라값이나 함수는 기울어진 소문자(`m`, `y⁽ⁱ⁾`, `h`), 벡터는 굵은 소문자(`x⁽ⁱ⁾`), 행렬은 굵은 대문자(`X`)로 쓴다.

RMSE는 일반적으로 회귀 문제에서 선호되지만, **이상치로 보이는 구역이 많다면 평균 절대 오차(MAE)**를 고려해볼 수 있다.

**[식 2-2] 평균 절대 오차(MAE)**

```
MAE(X, h) = (1/m) * Σ |h(x⁽ⁱ⁾) − y⁽ⁱ⁾|
```

RMSE와 MAE 모두 예측값의 벡터와 타깃값의 벡터 사이의 거리를 재는 방법이며, 거리 측정에는 여러 가지 **노름**(norm)을 사용할 수 있다.

| 노름 | 계산 | 별칭 |
|---|---|---|
| ℓ₂ (‖·‖₂) | 제곱항을 합한 것의 제곱근 = RMSE | 유클리드 노름 |
| ℓ₁ (‖·‖₁) | 절댓값의 합 = MAE | 맨해튼 노름 |
| ℓ₀ | 벡터에 있는 0이 아닌 원소의 수 | — |
| ℓ∞ | 벡터에서 가장 큰 절댓값 | — |

**노름의 지수가 클수록 큰 값의 원소에 치우치며 작은 값은 무시된다.** 그래서 RMSE가 MAE보다 조금 더 이상치에 민감하다. 하지만 (종 모양 분포의 양 끝단처럼) 이상치가 매우 드물면 RMSE가 잘 맞아 일반적으로 널리 사용된다.

> **모빌리티 적용.** 이 수업의 프로젝트 주 지표는 **MAE**다. ETA 오차는 분 단위로 직접 해석되고, 사고나 신호 대기로 인한 극단적 지연이 드물지 않기 때문이다. RMSE는 보조 지표로만 본다.

### 2.2.3 가정 검사

마지막으로 지금까지 만든 가정을 나열하고 검사해보는 것을 권한다. 이 과정에서 심각한 문제를 일찍 발견할 수도 있다.

교재의 예: 시스템이 출력한 구역의 가격이 그대로 다음 머신러닝 시스템의 입력으로 사용될 거라 가정했는데, 하위 시스템에서 이 값을 ('저렴', '보통', '고가' 같은) 카테고리로 바꾸고 가격 대신 카테고리를 사용한다면 어떨까? 이런 때는 정확한 가격을 구하는 것이 전혀 중요하지 않고 **이 문제는 회귀가 아니라 분류 작업**이 된다. 몇 달 동안 회귀 시스템을 구축하고 나서야 이런 사실을 깨닫는 것은 아무도 원치 않는다.

---

## 2.3 데이터 가져오기

교재의 모든 예제 코드는 오픈 소스이며 주피터 노트북으로 깃허브(`https://github.com/rickiepark/handson-ml3`)에서 제공된다. 컴퓨터에 아무것도 설치하지 않고 온라인에서 주피터 노트북을 바로 실행할 수 있는 무료 서비스인 **구글 코랩**으로 실행한다고 가정한다.

### 2.3.3 대화식 환경의 편리함과 위험 (교재 요지)

주피터 노트북은 유연하지만 대가가 따른다. **잘못된 순서로 셀을 실행하기가 매우 쉽고**, 어떤 셀을 실행하는 것을 잊어버릴 수 있다. 각 노트북의 첫 번째 코드 셀은 설정 코드(패키지 임포트)를 담고 있으므로 가장 먼저 실행해야 한다.

> **TIP(교재).** 이상한 에러가 발생하면 [런타임] → [런타임 다시 시작]으로 런타임을 다시 시작하고 노트북의 모든 코드를 처음부터 다시 실행하라. 이렇게 하면 문제가 해결되는 경우가 많다.

> **CAUTION(교재).** 구글 코랩은 대화식 서비스다. 노트북을 장기간 실행하지 않으면 런타임이 종료되고 데이터가 모두 사라진다. 중요한 데이터를 생성했다면 종료 전에 다운로드하거나 구글 드라이브를 마운트해서 저장하라.

### 2.3.5 데이터 다운로드

데이터를 수동으로 내려받아 압축을 푸는 대신 **함수를 작성하는 것이 일반적으로 낫다**. 특히 데이터가 정기적으로 바뀌는 경우에 유용하며, 여러 기기에 데이터셋을 설치해야 할 때도 편리하다.

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
```

### 2.3.6 데이터 구조 훑어보기

```python
housing.head()
```

각 행은 하나의 구역을 나타낸다. 특성은 `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value`, `ocean_proximity` 등 10개다.

```python
housing.info()
```

`info()`는 전체 행 수, 각 특성의 데이터 타입, 널이 아닌 값의 개수를 확인하는 데 유용하다. 교재의 출력에서 확인되는 사실은 다음과 같다.

- 데이터셋에 **20,640개**의 샘플이 있다. 머신러닝 프로젝트치고는 상당히 적지만 처음 시작하기에는 적당한 크기다.
- `total_bedrooms`는 **20,433개**만 널값이 아니다. **207개 구역이 이 특성을 갖고 있지 않다.**
- `ocean_proximity`만 빼고 모든 특성이 숫자형이다. 타입이 `object`이고 값이 반복되는 것으로 보아 범주형 특성일 것이다.

```python
housing["ocean_proximity"].value_counts()
# <1H OCEAN 9136 / INLAND 6551 / NEAR OCEAN 2658 / NEAR BAY 2290 / ISLAND 5

housing.describe()
```

`describe()`는 숫자형 특성의 요약 정보를 보여준다. `count`에서 **널값은 제외된다**(`total_bedrooms`의 count가 20,640이 아니라 20,433). `std`는 표준 편차, `25%`/`50%`/`75%`는 **백분위수**다.

```python
import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(12, 8))
plt.show()
```

교재가 히스토그램(그림 2-8)에서 읽어낸 네 가지가 이 절의 핵심이다.

1. **중간 소득이 US 달러로 표현되어 있지 않다.** 데이터를 취합한 팀에 확인해보니 스케일을 조정하고 상한 15, 하한 0.5가 되도록 만들었다고 한다. 머신러닝에서는 전처리된 데이터를 다루는 경우가 흔하고 이것이 문제가 되지는 않지만 **데이터가 어떻게 계산된 것인지 반드시 이해하고 있어야 한다.**
2. **중간 주택 연도와 중간 주택 가격 역시 최댓값과 최솟값을 한정했다.** 중간 주택 가격의 경우 타깃 속성으로 사용되기 때문에 심각한 문제가 될 수 있다. 가격이 한곗값을 넘어가지 않도록 머신러닝 알고리즘이 학습할지도 모른다. 클라이언트 팀과 검토해서 정확한 예측값이 필요하다면 (a) 한곗값 밖의 구역에 대한 정확한 레이블을 구하거나 (b) 훈련 세트에서 이런 구역을 제거한다(테스트 세트에서도 제거한다).
3. **특성들의 스케일이 서로 많이 다르다.**
4. **많은 히스토그램의 오른쪽 꼬리가 더 길다.** 이런 형태는 일부 머신러닝 알고리즘에서 패턴을 찾기 어렵게 만든다. 나중에 좀 더 종 모양의 분포가 되도록 변형한다.

> **CAUTION(교재).** 데이터를 더 깊게 들여다보기 전에 **테스트 세트를 따로 떼어놓아야 한다. 그리고 테스트 세트를 절대 들여다보면 안 된다.**

### 2.3.7 테스트 세트 만들기

우리 뇌는 매우 과대적합되기 쉬운 패턴 감지 시스템이다. 테스트 세트를 들여다본다면 테스트 세트에서 겉으로 드러난 어떤 패턴에 속아 특정 모델을 선택하게 될지도 모른다. 이를 **데이터 스누핑(data snooping) 편향**이라 한다.

가장 단순한 구현은 다음과 같다.

```python
import numpy as np

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```

문제는 **프로그램을 다시 실행하면 다른 테스트 세트가 생성된다**는 점이다. 여러 번 계속하면 결국 전체 데이터셋을 보는 셈이 된다. 해결책은 세 가지다.

| 해결책 | 한계 |
|---|---|
| 첫 실행에서 테스트 세트를 저장하고 다음 실행에서 불러온다 | 데이터셋이 업데이트되면 문제 |
| `np.random.seed(42)`로 난수 초깃값을 고정한다 | 데이터셋이 업데이트되면 문제 |
| **샘플의 식별자를 사용해 테스트 세트로 보낼지 정한다** | 샘플이 고유하고 변경 불가능한 식별자를 가져야 함 |

세 번째 방법은 각 샘플의 식별자 해시값을 계산해 해시 최댓값의 20%보다 작거나 같은 샘플만 테스트 세트로 보낸다. 이렇게 하면 데이터셋이 갱신되더라도 테스트 세트가 동일하게 유지된다.

```python
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
```

주택 데이터셋에는 식별자 컬럼이 없으므로 행의 인덱스를 ID로 쓰거나, 위도와 경도처럼 안정적인 특성을 연결해 ID를 만든다.

```python
housing_with_id = housing.reset_index()  # index 열이 추가됨
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
```

사이킷런의 가장 간단한 함수는 `train_test_split`이다.

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

#### 계층적 샘플링

지금까지는 순수한 **랜덤 샘플링**이다. 데이터셋이 충분히 크다면 괜찮지만 그렇지 않으면 샘플링 편향이 생길 가능성이 크다.

교재의 비유: 설문 조사 기관에서 1,000명에게 질문을 하려 할 때 그냥 전화번호부에서 1,000명을 랜덤으로 뽑지 않는다. 미국 인구의 51.1%가 여성이고 48.9%가 남성이라면 잘 구성된 설문 조사는 샘플에서도 이 비율을 유지해야 한다. 이를 **계층적 샘플링**(stratified sampling)이라 하며, 전체 인구를 **계층**(strata)이라는 동질의 그룹으로 나누고 각 계층에서 올바른 수의 샘플을 추출한다.

전문가가 중간 소득이 중간 주택 가격 예측에 매우 중요하다고 했으므로, 연속적인 중간 소득으로 카테고리 특성을 만든다. **계층별로 데이터셋에 충분한 샘플 수가 있어야 한다.** 그렇지 않으면 계층의 중요도를 추정하는 데 편향이 발생하므로, 너무 많은 계층으로 나누어서는 안 되며 각 계층이 충분히 커야 한다.

```python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```

사이킷런의 분할기 클래스를 사용하거나, 하나의 분할만 필요하면 `train_test_split`의 `stratify` 매개변수로 간편하게 만든다.

```python
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]

# 또는
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
```

교재의 그림 2-10이 이 절의 결론이다. 계층 샘플링으로 만든 테스트 세트는 전체 데이터셋의 소득 카테고리 비율과 거의 같지만, 순수한 랜덤 샘플링은 비율이 많이 달라졌다.

| 소득 카테고리 | 전체 % | 계층 샘플링 % | 랜덤 샘플링 % | 계층 오차 % | 랜덤 오차 % |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.98 | 4.00 | 4.24 | 0.36 | 6.45 |
| 2 | 31.88 | 31.88 | 30.74 | −0.02 | −3.59 |
| 3 | 35.06 | 35.05 | 34.52 | −0.01 | −1.53 |
| 4 | 17.63 | 17.64 | 18.41 | 0.03 | 4.42 |
| 5 | 11.44 | 11.43 | 12.09 | −0.08 | 5.63 |

`income_cat`은 다시 사용하지 않으므로 열을 삭제해 데이터를 원래 상태로 되돌린다.

```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

> **모빌리티 적용.** ETA·통행속도 데이터는 시간 순서를 갖는다. 무작위 분할이나 계층적 분할을 쓰면 미래 정보가 훈련 세트로 새어 들어간다(데이터 누수). 교재의 "식별자 해시" 아이디어와 같은 목적을 **날짜 기준 분할**로 달성한다.
>
> ```python
> df = df.sort_values("observed_at")
> train = df[df["observed_at"] < "2026-08-01"]
> valid = df[(df["observed_at"] >= "2026-08-01") & (df["observed_at"] < "2026-08-08")]
> test  = df[df["observed_at"] >= "2026-08-08"]
> ```

---

## 2.4 데이터 이해를 위한 탐색과 시각화

먼저 테스트 세트를 떼어놓았는지 확인하고 **훈련 세트에 대해서만** 탐색한다. 훈련 세트가 매우 크다면 탐색을 위한 세트를 별도로 샘플링할 수도 있다. 나중에 되돌릴 수 있도록 원본의 복사본을 만든다.

```python
housing = strat_train_set.copy()
```

### 2.4.1 지리적 데이터 시각화하기

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
plt.show()
```

캘리포니아 지역을 잘 나타내지만 특별한 패턴을 찾기는 힘들다. `alpha` 옵션을 0.2로 주면 데이터 포인트가 밀집된 영역을 잘 보여준다.

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
```

주택 가격까지 나타낸다. 원의 반지름은 구역의 인구(매개변수 `s`), 색상은 가격(매개변수 `c`)이다.

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()
```

주택 가격은 **지역(바다와 인접한 곳) 및 인구 밀도와 관련성이 높다**. 군집 알고리즘을 사용해 주요 군집을 찾고 군집의 중심까지의 거리를 재는 특성을 추가할 수 있다. 다만 북부 캘리포니아 지역의 해안가는 주택 가격이 그리 높지 않으므로 **간단한 규칙을 적용하기는 어렵다**.

### 2.4.2 상관관계 조사하기

데이터셋이 너무 크지 않으므로 모든 특성 간의 **표준 상관계수**(피어슨의 r)를 `corr()`로 계산한다.

```python
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
```

교재의 출력:

| 특성 | 상관계수 |
|---|---:|
| median_house_value | 1.000000 |
| median_income | 0.688380 |
| total_rooms | 0.137455 |
| housing_median_age | 0.102175 |
| households | 0.071426 |
| total_bedrooms | 0.054635 |
| population | −0.020153 |
| longitude | −0.050859 |
| latitude | −0.139584 |

상관관계의 범위는 −1부터 1까지다. 1에 가까우면 강한 양의 상관관계, −1에 가까우면 강한 음의 상관관계, 0에 가까우면 선형적인 상관관계가 없다는 뜻이다.

특성 사이의 상관관계를 확인하는 다른 방법은 판다스의 `scatter_matrix`다.

```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
```

중간 소득이 가장 유용해 보이므로 확대해서 본다.

```python
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
```

그림 2-15에서 교재가 지적하는 사실:

- 상관관계가 매우 강하다. 위쪽으로 향하는 경향이 뚜렷하고 포인트가 너무 퍼져 있지 않다.
- 가격의 한곗값 $500,000에서 수평선이 잘 보인다.
- 그 외에도 $450,000, $350,000, $280,000 근처와 그 아래에 직선에 가까운 형태가 나타난다. **알고리즘이 이런 이상한 형태를 학습하지 않도록 해당 구역을 제거할 수 있다.**

> **CAUTION(교재).** 상관계수는 **선형적인** 상관관계만 측정한다. 비선형적인 관계는 잡을 수 없다(x가 0에 가까워지면 y가 증가하는 경우). 그림 2-16의 마지막 줄 그래프들은 두 축이 완전히 독립적이지 않음에도 상관계수가 0이다. 또한 상관계수는 **기울기와 상관없다**. 인치 단위의 키는 피트나 나노미터 단위의 키와 상관계수가 1이다.

### 2.4.3 특성 조합으로 실험하기

머신러닝 알고리즘용 데이터를 준비하기 전에 마지막으로 할 수 있는 일은 **특성을 여러 가지로 조합**해보는 것이다.

교재의 논리가 좋은 예시다. 어떤 구역의 방 개수는 가구 수를 모른다면 그다지 유용하지 않다. 진짜 필요한 것은 **가구당 방 개수**다. 비슷하게 전체 침실 개수도 그 자체로는 유용하지 않고 **방 개수와 비교하는 게 낫다**. 가구당 인원도 흥미로운 조합일 것이다.

```python
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
```

상관관계 행렬을 다시 확인하면:

| 특성 | 상관계수 |
|---|---:|
| median_income | 0.688380 |
| **rooms_per_house** | **0.143663** |
| total_rooms | 0.137455 |
| housing_median_age | 0.102175 |
| … | … |
| **bedrooms_ratio** | **−0.256397** |

새로운 `bedrooms_ratio` 특성은 전체 방 개수나 침실 개수보다 중간 주택 가격과의 상관관계가 **훨씬 높다**. 침실/방의 비율이 낮은 집이 더 비싼 경향이 있다. 가구당 방 개수도 구역 내 전체 방 개수보다 더 유용하다.

> 이 탐색 단계는 완벽하지 않아도 된다. **빠르게 시작해서 인사이트를 얻는 것이 합리적인 첫 번째 프로토타입을 만드는 데 도움이 되며, 이는 반복적인 과정이다.**

> **모빌리티 적용.** "합계보다 비율·단위당 값"이라는 이 장의 교훈을 그대로 옮기면 다음이 된다.
> - 남은 정류장 수, 남은 거리
> - 최근 3개 구간의 평균 통행시간
> - 동일 노선·시간대의 과거 중앙값
> - 정류장 체류시간
> - 구간 길이당 통행시간(= 1/속도)

---

## 2.5 머신러닝 알고리즘을 위한 데이터 준비

수동으로 하는 대신 **함수를 만들어 자동화**해야 하는 이유:

- 어떤 데이터셋에 대해서도 데이터 변환을 손쉽게 반복할 수 있다.
- 향후 프로젝트에 재사용 가능한 변환 라이브러리를 점진적으로 구축할 수 있다.
- 실제 시스템에서 알고리즘에 새 데이터를 주입하기 전에 이 함수를 사용해 변환할 수 있다.
- 여러 가지 데이터 변환을 쉽게 시도해볼 수 있고 어떤 조합이 가장 좋은지 확인하는 데 편리하다.

예측 변수와 타깃값에 같은 변형을 적용하지 않기 위해 분리한다.

```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

### 2.5.1 데이터 정제

`total_bedrooms`의 누락된 값을 고치는 방법은 세 가지다.

1. 해당 구역을 제거한다.
2. 전체 특성을 삭제한다.
3. 누락된 값을 어떤 값으로 채운다(0, 평균, 중간값 등). 이를 **대체**(imputation)라 한다.

```python
housing.dropna(subset=["total_bedrooms"], inplace=True)      # 옵션 1
housing.drop("total_bedrooms", axis=1, inplace=True)         # 옵션 2
median = housing["total_bedrooms"].median()                  # 옵션 3
housing["total_bedrooms"].fillna(median, inplace=True)
```

**옵션 3이 데이터를 최대한 유지**하므로 이를 선택한다. 다만 판다스 코드 대신 사이킷런의 `SimpleImputer`를 사용한다. 이 클래스는 각 특성의 중간값을 저장하고 있어 훈련 세트뿐 아니라 검증 세트, 테스트 세트, 새로운 데이터에 있는 누락된 값을 대체할 수 있다.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)

imputer.statistics_          # 각 특성의 중간값
X = imputer.transform(housing_num)
```

> `total_bedrooms` 특성에만 누락된 값이 있지만, **나중에 시스템이 서비스될 때 새로운 데이터에서 어떤 값이 누락될지 확신할 수 없으므로 모든 수치형 특성에 imputer를 적용하는 것이 바람직하다.**

전략은 `"median"` 외에 `"mean"`, `"most_frequent"`, `"constant"`(`fill_value=...`)가 있고, 마지막 두 방법은 수치가 아닌 데이터를 지원한다.

> **TIP(교재).** `sklearn.impute` 패키지는 더 강력한 클래스도 제공한다.
> - `KNNImputer`: 누락된 값을 이 특성에 대한 k-최근접 이웃의 평균으로 대체한다.
> - `IterativeImputer`: 특성마다 회귀 모델을 훈련하여 다른 모든 특성을 기반으로 누락된 값을 예측한다.

> **사이킷런의 설계 철학 (교재 박스).** 이 박스는 이후 모든 장에서 쓰이므로 그대로 옮긴다.
> - **일관성**: 모든 객체는 일관되고 단순한 인터페이스를 공유한다.
>   - **추정기**(estimator): 데이터셋을 기반으로 일련의 모델 파라미터들을 추정하는 객체. 추정 자체는 `fit()` 메서드에 의해 수행된다. 추정 과정에서 필요한 다른 매개변수들은 모두 하이퍼파라미터로 간주되고 인스턴스 변수로 저장된다.
>   - **변환기**(transformer): 데이터셋을 변환하는 추정기. 변환은 `transform()` 메서드가 수행하고 변환된 데이터셋을 반환한다. 모든 변환기는 `fit()`과 `transform()`을 연달아 호출하는 것과 동일한 `fit_transform()` 메서드도 가지고 있다(이따금 최적화되어 있어서 더 빠르다).
>   - **예측기**(predictor): 주어진 데이터셋에 대해 예측을 만들 수 있는 추정기. `predict()` 메서드는 새로운 데이터셋을 받아 상응하는 예측값을 반환하고, `score()` 메서드는 테스트 세트를 사용해 예측의 품질을 측정한다.
> - **검사 가능**: 모든 추정기의 하이퍼파라미터는 공개 인스턴스 변수로 직접 접근할 수 있고(`imputer.strategy`), 모든 학습된 모델 파라미터는 접미사로 밑줄을 붙여 공개 인스턴스 변수로 제공된다(`imputer.statistics_`).
> - **클래스 남용 방지**: 데이터셋을 별도의 클래스가 아니라 넘파이 배열이나 사이파이 희소 행렬로 표현한다.
> - **조합성**: 기존의 구성 요소를 최대한 재사용한다. 여러 변환기를 연결한 다음 마지막에 추정기 하나를 배치한 `Pipeline` 추정기를 쉽게 만들 수 있다.
> - **합리적인 기본값**: 대부분의 매개변수에 합리적인 기본값을 지정해두었다.

사이킷런 변환기는 판다스 데이터프레임이 입력되더라도 넘파이 배열을 출력한다. 필요하면 다시 감싼다.

```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

### 2.5.2 텍스트와 범주형 특성 다루기

`ocean_proximity`는 가능한 값을 제한된 개수로 나열한 **범주형 특성**이다. 먼저 `OrdinalEncoder`로 숫자로 바꿔본다.

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
ordinal_encoder.categories_
# [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'], dtype=object)]
```

**이 표현 방식의 문제는 머신러닝 알고리즘이 가까이 있는 두 값을 떨어져 있는 두 값보다 더 비슷하다고 생각한다는 점이다.** `'bad'`, `'average'`, `'good'`, `'excellent'`처럼 순서가 있는 카테고리라면 괜찮지만 `ocean_proximity`에는 해당되지 않는다(카테고리 0과 1보다 카테고리 0과 4가 확실히 더 비슷하다).

해결책은 카테고리별 이진 특성을 만드는 **원-핫 인코딩**이다. 새로 만든 특성을 **더미**(dummy) 특성이라고도 한다.

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```

기본적으로 출력은 넘파이 배열이 아니라 사이파이 **희소 행렬**이다. 희소 행렬은 0이 대부분인 행렬을 매우 효율적으로 표현하며 내부적으로 0이 아닌 값과 그 위치만 저장한다. 밀집 배열로 바꾸려면 `toarray()`를 호출하거나 생성 시 `sparse_output=False`로 지정한다.

#### `get_dummies()` 대신 `OneHotEncoder`를 쓰는 이유

판다스에도 `get_dummies()`가 있지만, **`OneHotEncoder`의 장점은 어떤 카테고리로 훈련되었는지 기억한다는 점**이다. 모델을 제품에 적용할 때 훈련과 완전히 동일한 특성이 주입되어야 하기 때문에 매우 중요하다.

- `get_dummies()`는 본 카테고리만큼만 열을 출력한다. 알 수 없는 카테고리(`"<2H OCEAN"`)를 주입해도 아무 문제 없이 변환된 결과를 출력한다.
- `OneHotEncoder`는 학습된 카테고리마다 하나의 열을 순서대로 출력하고, **알 수 없는 카테고리를 감지하면 예외를 발생**시킨다. 원한다면 `handle_unknown="ignore"`로 지정해 그냥 0으로 나타낼 수 있다.

> **TIP(교재).** 카테고리 특성이 담을 수 있는 카테고리 수가 많다면(국가 코드, 직업, 생물 종류 등) 원-핫 인코딩은 많은 수의 입력 특성을 만든다. 이는 훈련 속도를 느리게 하고 성능을 감소시킬 수 있다. 이런 현상이 나타나면 **범주형 입력값을 이 특성과 관련된 숫자형 특성으로 바꾸고 싶을 것이다.** 예를 들어 `ocean_proximity` 특성을 해안까지의 거리로 바꿀 수 있다(비슷하게 국가 코드는 인구와 1인당 GDP로 바꿀 수 있다). 다른 방법으로 `category_encoders` 패키지가 제공하는 인코더 중 하나를 사용할 수 있다. 또는 신경망을 사용해 각 카테고리를 **임베딩**이라 부르는 학습 가능한 저차원 벡터로 바꿀 수도 있는데, 이는 **표현 학습**의 한 예다(13장·17장 참고).

> **모빌리티 적용.** 노선 ID, 정류장 ID는 정확히 위의 TIP이 경고하는 고차원 범주형 특성이다. 5주차까지는 원-핫과 트리 모델로 다루고, 10주차 MLP부터는 **임베딩**으로 바꾼다. 이 TIP이 그 전환의 근거다.

데이터프레임으로 훈련하면 추정기는 열 이름을 `feature_names_in_` 속성에 저장하고, 이후 입력되는 데이터프레임이 동일한 열 이름을 갖는지 확인한다. 변환기는 `get_feature_names_out()` 메서드도 제공한다.

### 2.5.3 특성 스케일과 변환

**특성 스케일링**은 데이터에 적용할 가장 중요한 변환 중 하나다. 몇 가지를 제외하고 머신러닝 알고리즘은 입력된 숫자 특성들의 스케일이 많이 다르면 제대로 작동하지 않는다. 전체 방 개수의 범위는 6에서 39,320인 반면 중간 소득의 범위는 0에서 15까지다. 스케일링하지 않으면 대부분의 모델은 중간 소득을 무시하고 방 개수에 더 초점을 맞출 것이다.

> **CAUTION(교재).** 모든 추정기와 마찬가지로 **스케일링은 훈련 데이터로만 수행해야 한다.** 훈련 세트 이외의 어떤 것에도 `fit()`이나 `fit_transform()`을 사용해서는 안 된다. 훈련 세트 값은 항상 특정 범위로 스케일링되지만 새로운 데이터에 이상치가 있다면 이 범위 밖으로 스케일링될 것이다. 이를 원치 않는다면 `MinMaxScaler`의 `clip` 매개변수를 `True`로 지정하라.

| 방법 | 내용 | 특징 |
|---|---|---|
| min-max 스케일링(정규화) | 최솟값을 뺀 후 최댓값과 최솟값의 차이로 나눠 0~1 범위로 | `feature_range`로 범위 변경 가능. 신경망은 −1~1 선호 |
| 표준화 | 평균을 뺀 후 표준 편차로 나눔 (평균 0, 표준 편차 1) | 특정 범위로 제한하지 않지만 **이상치에 영향을 덜 받는다** |

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```

교재의 대비가 명확하다. 어떤 구역의 중간 소득이 (잘못 입력되어) 100이라 하면, min-max 스케일링은 이 이상치를 1로 매핑하고 **다른 모든 값을 0~0.15로 만들어버리지만** 표준화는 크게 영향받지 않는다.

> **TIP(교재).** 희소 행렬을 밀집 행렬로 바꾸지 않고 스케일링하고 싶다면 `StandardScaler`에서 `with_mean` 하이퍼파라미터를 `False`로 지정하라. 평균을 빼지 않고 표준 편차로 나누기만 한다.

#### 꼬리가 두꺼운 분포 다루기

특성 분포의 꼬리가 두꺼울 때 min-max 스케일링과 표준화는 대부분의 값을 작은 범위로 압축한다. 머신러닝 모델은 일반적으로 이런 값을 좋아하지 않으므로, **스케일링하기 전에 꼬리를 줄이도록 데이터를 먼저 변환하고 분포가 대략적으로 대칭이 되도록** 만들어야 한다.

| 상황 | 변환 |
|---|---|
| 오른쪽 꼬리가 두꺼운 양수 특성 | 제곱근(또는 0~1 사이 거듭제곱) |
| **멱법칙 분포**처럼 꼬리가 아주 길고 두꺼움 | **로그 변환** |

`population` 특성은 대략적으로 멱법칙을 따른다. 10,000명이 있는 구역은 1,000명이 있는 구역보다 빈도가 지수적으로 줄어들지 않고 단지 10배 낮을 뿐이다. 로그값으로 바꾸면 가우스 분포에 매우 가까워진다(그림 2-17).

#### 버킷타이징

또 다른 방법은 특성을 **버킷타이징**하는 것이다. 분포를 거의 동일한 크기의 버킷으로 자르고 특성값을 해당하는 버킷의 인덱스로 바꾼다. 거의 동일한 크기의 버킷을 사용하면 거의 균등 분포의 특성을 만들 수 있어 추가 스케일링이 필요 없다.

`housing_median_age`처럼 **멀티모달 분포**(모드라 부르는 정점이 두 개 이상 나타나는 분포)일 때 버킷타이징이 도움이 될 수 있다. 하지만 이때 **버킷 ID를 수치가 아니라 카테고리로 다룬다.** 즉 버킷 인덱스를 `OneHotEncoder` 등으로 인코딩해야 한다(따라서 너무 많은 버킷을 사용하고 싶지는 않을 것이다). 이렇게 하면 회귀 모델이 특성값의 여러 범주에 대해 다양한 규칙을 쉽게 학습할 수 있다. 예를 들어 35년 전에 지어진 집은 유행이 지난 독특한 스타일을 가지고 있기 때문에 같은 해에 지어진 다른 집보다 더 저렴하다.

#### RBF 유사도 특성

멀티모달 분포를 변환하는 또 다른 방법은 (적어도 주요 모드에 대해) 중간 주택 연도와 특정 모드 사이의 **유사도를 나타내는 특성을 추가**하는 것이다. 유사도 측정은 일반적으로 **방사 기저 함수(RBF)**를 사용한다. 가장 널리 사용되는 RBF는 입력값이 고정 포인트에서 멀어질수록 출력값이 지수적으로 감소하는 **가우스 RBF**다. 주택 연도 x와 35 사이의 가우스 RBF 유사도는 `exp(−γ(x−35)²)`이다. 하이퍼파라미터 γ(감마)는 x가 35에서 멀어짐에 따라 유사도 값이 얼마나 빠르게 감소하는지 결정한다.

```python
from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
```

#### 타깃값 변환

지금까지는 입력 특성만 보았지만 **타깃값도 변환이 필요할 수 있다.** 타깃 분포의 꼬리가 두껍다면 타깃을 로그값으로 바꿀 수 있다. 하지만 이렇게 하면 회귀 모델이 중간 주택 가격 자체가 아니라 **중간 주택 가격의 로그를 예측**하게 된다. 원래 값을 얻으려면 모델 예측에 지수 함수를 적용해야 한다.

대부분의 사이킷런 변환기에는 역변환을 수행하는 `inverse_transform()` 메서드가 있다.

```python
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)
```

더 간단한 방법은 `TransformedTargetRegressor`를 사용하는 것이다. 회귀 모델과 레이블 변환기를 전달하고 스케일링되지 않은 원본 레이블로 훈련하면, 자동으로 레이블을 스케일링해 훈련하고 `predict()` 시 역변환까지 수행한다.

```python
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
```

### 2.5.4 사용자 정의 변환기

훈련이 필요 없는 변환은 넘파이 배열을 받아 변환된 배열을 출력하는 함수를 `FunctionTransformer`로 감싸면 된다.

```python
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])

ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
```

`fit()`에서 파라미터를 학습해야 하는 **훈련 가능한 변환기**가 필요하면 사용자 정의 클래스를 작성한다. 사이킷런은 덕 타이핑에 의존하므로 특정 클래스를 상속할 필요가 없고, 필요한 것은 `fit()`(self를 반환), `transform()`, `fit_transform()` 세 개의 메서드뿐이다. `TransformerMixin`을 상속하면 `fit_transform()`이 자동 생성되고, `BaseEstimator`를 상속하면 `get_params()`와 `set_params()`를 얻는다.

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # *args나 **kwargs를 사용하지 않습니다!
        self.with_mean = with_mean

    def fit(self, X, y=None):            # 사용하지 않더라도 y를 넣어야 합니다.
        X = check_array(X)               # X가 부동소수점 배열인지 확인합니다.
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1] # 모든 추정기는 fit()에서 이를 저장합니다.
        return self                      # 항상 self를 반환합니다!

    def transform(self, X):
        check_is_fitted(self)            # 학습된 속성이 있는지 확인합니다.
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
```

주의 사항(교재 원문 요지):

- 사이킷런 파이프라인은 `X`와 `y` 두 개의 매개변수를 가진 메서드가 필요하다. 그래서 `y`를 사용하지 않지만 `y=None`이 필요하다.
- 모든 사이킷런 추정기는 `fit()` 안에서 `n_features_in_`을 설정하고 `transform()`이나 `predict()`에 전달된 데이터의 특성 개수가 동일한지 확인한다.
- `fit()` 메서드는 `self`를 반환해야 한다.
- 데이터프레임이 전달될 때 `fit()` 안에서 `feature_names_in_`을 설정해야 하고, `get_feature_names_out()`과 `inverse_transform()`을 제공해야 완전하다.

하나의 사용자 변환기가 구현 안에서 **다른 추정기를 사용**할 수도 있다. 교재는 `fit()` 안에서 `KMeans`로 핵심 클러스터를 식별하고 `transform()`에서 `rbf_kernel()`로 각 샘플이 클러스터 중심과 얼마나 유사한지 측정하는 변환기를 보여준다.

```python
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self                      # 항상 self를 반환합니다!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"클러스터 {i} 유사도" for i in range(self.n_clusters)]
```

```python
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
                                           sample_weight=housing_labels)
```

> **TIP(교재).** `sklearn.utils.estimator_checks` 모듈의 `check_estimator()` 함수에 사용자 정의 추정 객체를 전달하여 사이킷런 API를 준수하는지 확인할 수 있다.

### 2.5.5 변환 파이프라인

변환 단계는 올바른 순서대로 실행되어야 한다. 사이킷런의 `Pipeline` 클래스가 이를 도와준다.

```python
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
```

`Pipeline` 생성자는 이름/추정기 쌍의 리스트를 받는다. 이름은 이중 밑줄(`__`)을 포함하지 않으면서 고유하다면 어떤 것도 가능하다. 이중 밑줄은 나중에 하이퍼파라미터 튜닝에 사용된다. **추정기는 마지막을 제외하고 모두 변환기여야 한다.**

이름을 짓는 게 귀찮다면 `make_pipeline()`을 사용한다. 클래스 이름을 소문자로 바꾸어 이름을 자동 지정한다.

```python
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
```

파이프라인의 `fit()`을 호출하면 모든 변환기의 `fit_transform()`을 순서대로 호출하면서 한 단계의 출력을 다음 단계의 입력으로 전달한다. 마지막 단계에서는 `fit()`만 호출한다. 파이프라인 객체는 **마지막 추정기와 동일한 메서드를 제공**한다.

> **TIP(교재).** 주피터 노트북에서 `import sklearn`과 `sklearn.set_config(display="diagram")`을 실행하면 모든 사이킷런 추정기가 인터랙티브한 다이어그램으로 표현된다. 파이프라인을 시각화하는 데 특히 유용하다.

수치형과 범주형 열을 하나의 변환기로 처리하려면 `ColumnTransformer`를 쓴다.

```python
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
```

> **TIP(교재).** 튜플에 변환기를 사용하는 대신 삭제하고 싶은 특성이 있다면 `"drop"`으로, 변환을 적용하지 않을 특성이 있다면 `"passthrough"`로 지정할 수 있다. 기본적으로 나머지 열(나열되지 않은 열)은 삭제된다. 나머지 열을 다르게 처리하고 싶다면 `remainder` 하이퍼파라미터에 변환기(또는 `"passthrough"`)를 지정할 수 있다.

모든 특성 이름을 일일이 나열하는 것이 번거롭다면 `make_column_selector`와 `make_column_transformer`를 사용한다.

```python
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing)
```

> **NOTE(교재).** `OneHotEncoder`는 희소 행렬을 반환하지만 `num_pipeline`은 밀집 행렬을 반환한다. 희소 행렬과 밀집 행렬이 섞여 있을 때 `ColumnTransformer`는 최종 행렬의 밀집 정도(0이 아닌 원소의 비율)를 추정한다. 밀집도가 임곗값(기본 `sparse_threshold=0.3`)보다 낮으면 희소 행렬을 반환한다.

#### 최종 전처리 파이프라인

교재가 이 장에서 실험한 모든 변환을 하나의 파이프라인으로 정리한 것이다. **이 파이프라인이 할 일과 그 이유**를 교재가 목록으로 정리한 부분이 곧 이 장의 요약이다.

- 대부분의 머신러닝 알고리즘은 누락된 값을 기대하지 않으므로, 수치형 특성은 누락된 값을 중간값으로, 범주형 특성은 가장 많이 등장하는 카테고리로 바꾼다.
- 대부분의 머신러닝 알고리즘은 수치 입력만 받으므로 범주형 특성을 원-핫 인코딩한다.
- 비율 특성 `bedrooms_ratio`, `rooms_per_house`, `people_per_house`를 계산하여 추가한다. 이런 특성은 중간 주택 가격과 상관관계가 높으므로 머신러닝 모델에 도움이 되기를 기대해볼 수 있다.
- 몇 가지 클러스터 유사도 특성을 추가한다. 위도와 경도보다 모델에 더 유용할 가능성이 높다.
- 대부분의 모델은 균등 분포나 가우스 분포에 가까운 특성을 선호하기 때문에 꼬리가 두꺼운 분포를 띠는 특성을 로그값으로 바꾼다.
- 대부분의 머신러닝 알고리즘은 모든 특성이 대체로 동일한 스케일을 가질 때 잘 작동하므로 모든 수치 특성을 표준화한다.

```python
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # get_feature_names_out에 사용

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                           "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
],
    remainder=default_num_pipeline)  # 남은 특성: housing_median_age
```

```python
housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape        # (16512, 24)
preprocessing.get_feature_names_out()
```

---

## 2.6 모델 선택과 훈련

### 2.6.1 훈련 세트에서 훈련하고 평가하기

```python
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
```

훈련 세트에 적용하고 처음 다섯 개 예측과 레이블을 비교한다.

```python
housing_predictions = lin_reg.predict(housing)
housing_predictions[:5].round(-2)
# array([243700., 372400., 128800.,  94400., 328300.])
housing_labels.iloc[:5].values
# array([458300., 483800., 101700.,  96100., 361800.])
```

작동하긴 하지만 항상 맞지는 않는다. 첫 번째 예측은 $200,000달러 이상 많이 벗어났고, 두 개는 25% 정도, 다른 두 개는 10% 미만이다.

```python
from sklearn.metrics import mean_squared_error

lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
lin_rmse    # 68687.89176589991
```

대부분 구역의 중간 주택 가격이 $120,000에서 $265,000 사이인데 예측 오차가 $68,628인 것은 매우 만족스럽지 못하다. **이는 모델이 훈련 데이터에 과소적합된 사례다.** 특성들이 좋은 예측을 만들 만큼 충분한 정보를 제공하지 못했거나 모델이 충분히 강력하지 못하다는 뜻이다. 이 모델은 규제를 사용하지 않았으므로 "규제 감소" 옵션은 제외된다.

더 복잡한 모델인 `DecisionTreeRegressor`를 시도한다.

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
tree_rmse    # 0.0
```

> 잠깐만요, 뭐죠!? 오차가 전혀 없나요? 이 모델이 진짜 완벽할 수 있나요? **물론 모델이 데이터에 심하게 과대적합되었을 가능성이 높다.**

확신이 드는 모델을 론칭하기 전까지는 테스트 세트를 사용하지 않을 것이므로, 훈련 세트의 일부분으로 훈련하고 다른 일부분을 모델 검증에 사용해야 한다.

### 2.6.2 교차 검증으로 평가하기

사이킷런의 **k-폴드 교차 검증**은 훈련 세트를 **폴드**라 불리는 중복되지 않는 10개의 서브셋으로 랜덤 분할한다. 결정 트리 모델을 10번 훈련하고 평가하는데, 매번 다른 폴드를 선택해 평가에 사용하고 나머지 9개 폴드는 훈련에 사용한다.

```python
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
```

> **CAUTION(교재).** 사이킷런의 교차 검증 기능은 `scoring` 매개변수에 (낮을수록 좋은) 비용 함수가 아니라 (클수록 좋은) 효용 함수를 기대한다. 그래서 `neg_mean_squared_error` 함수는 RMSE의 음숫값을 출력한다. 따라서 RMSE를 얻기 위해 마이너스 부호를 추가한다.

```python
pd.Series(tree_rmses).describe()
# count    10.000000
# mean  66868.027288
# std    2060.966425
# min   63649.536493
# 50%   66801.953094
# max   70094.778246
```

결정 트리 결과가 이전만큼 좋아 보이지 않는다. 실제로 거의 선형 회귀 모델만큼 나쁘다. 교차 검증으로 성능을 추정하는 것뿐 아니라 **이 추정이 얼마나 정확한지(표준 편차)까지 측정할 수 있다.** 검증 세트를 하나만 사용했다면 이런 정보를 얻지 못했을 것이다. 단점은 모델을 여러 번 훈련시켜야 해서 비용이 비싸다는 점이다.

| 모델 | 교차 검증 RMSE 평균 | 표준 편차 |
|---|---:|---:|
| 선형 회귀 | 69,858 | 4,182 |
| 결정 트리 | 66,868 | 2,061 |
| **랜덤 포레스트** | **47,019** | **1,034** |

결정 트리는 훈련 오차가 작고(실제로 0) 검증 오차는 높기 때문에 **과대적합**이다.

```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)
```

랜덤 포레스트는 이 작업에 아주 잘 맞아 보인다. 하지만 `RandomForestRegressor`를 훈련하고 **훈련 세트에서 RMSE를 측정하면 약 17,474**를 얻는다. 이는 매우 낮은 값으로 **여전히 많이 과대적합되어 있다.** 해결 방법은 모델을 단순화하거나, 제한을 하거나(규제), 더 많은 훈련 데이터를 모으는 것이다.

> 랜덤 포레스트를 더 깊이 살펴보기 전에 **하이퍼파라미터 조정에 너무 많은 시간을 들이지 않고 여러 종류의 머신러닝 알고리즘에서 다양한 모델을 시도해봐야 한다. 가능성 있는 2~5개 정도의 모델을 선정하는 것이 목적이다.**

> **모빌리티 적용.** 시계열 데이터에는 무작위 k-폴드 대신 `TimeSeriesSplit`이나 날짜 구간을 직접 지정한 분할을 사용한다.
>
> ```python
> from sklearn.model_selection import TimeSeriesSplit
> tscv = TimeSeriesSplit(n_splits=5)
> ```

---

## 2.7 모델 미세 튜닝

### 2.7.1 그리드 서치

가장 단순한 방법은 만족할 만한 하이퍼파라미터 조합을 찾을 때까지 수동으로 조정하는 것이다. 이는 매우 지루한 작업이며 많은 경우의 수를 탐색할 시간이 부족할 수도 있다. 대신 `GridSearchCV`를 사용한다.

```python
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                          scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
```

파이프라인이나 `ColumnTransformer`가 추정기를 겹겹이 감싸고 있더라도 **이중 밑줄로 경로를 지정해 모든 하이퍼파라미터를 지정할 수 있다.** `"preprocessing__geo__n_clusters"`는 파이프라인의 `"preprocessing"` → `ColumnTransformer`의 `"geo"` 변환기(`ClusterSimilarity`) → 그 변환기의 `n_clusters`를 의미한다.

> **TIP(교재).** 사이킷런 파이프라인으로 전처리 단계를 감싸면 모델의 하이퍼파라미터와 함께 **전처리 하이퍼파라미터도 튜닝**할 수 있다. 두 하이퍼파라미터는 상호 작용하는 경우가 많기 때문에 바람직하다. 예를 들어 `n_clusters`를 증가시키면 `max_features`도 증가시켜야 한다. 파이프라인 변환기를 훈련하는 데 계산 비용이 많이 든다면 `memory` 매개변수에 캐싱 디렉터리 경로를 지정할 수 있다.

`param_grid`에는 두 개의 딕셔너리가 있다. 첫 번째에서 3 × 3 = 9개, 두 번째에서 2 × 3 = 6개를 평가하므로 총 **15개의 조합**을 탐색한다. 3-폴드 교차 검증이므로 총 **15 × 3 = 45번 훈련**이 일어난다.

```python
grid_search.best_params_
# {'preprocessing__geo__n_clusters': 15, 'random_forest__max_features': 6}
```

> **TIP(교재).** 15가 `n_clusters`에 대한 탐색 범위의 최댓값이기 때문에 **더 큰 값을 지정하여 다시 검색해봐야 한다.** 계속 점수가 향상될 가능성이 있다.

`grid_search.best_estimator_`로 최상의 추정기를 얻는다. `refit=True`(기본값)이면 교차 검증으로 최적 추정기를 찾은 다음 **전체 훈련 세트로 다시 훈련**시킨다. 평가 점수는 `grid_search.cv_results_`로 얻는다.

최상의 모델에 대한 평균 테스트 RMSE 점수는 **44,042**로, 기본 하이퍼파라미터로 얻은 점수(47,019)보다 좋다.

### 2.7.2 랜덤 서치

그리드 서치는 비교적 적은 수의 조합을 탐구할 때 좋다. 하지만 **하이퍼파라미터 탐색 공간이 커지면 `RandomizedSearchCV`가 종종 선호된다.** 가능한 모든 조합을 시도하는 대신 각 반복마다 하이퍼파라미터에 임의의 수를 대입하여 지정한 횟수만큼 평가한다. 주요 장점 두 가지:

- 하이퍼파라미터 값이 연속적이면(또는 이산적이지만 가능한 값이 많다면) 랜덤 서치를 1,000번 실행했을 때 각 하이퍼파라미터마다 1,000개의 다른 값을 탐색한다. 반면 그리드 서치는 하이퍼파라미터에 대해 나열한 몇 개의 값만 탐색한다.
- 어떤 하이퍼파라미터가 성능 면에서 큰 차이를 만들지 못하지만 아직 그 사실을 모른다고 하자. 10개의 가능한 값이 있을 때 이를 그리드 서치에 추가하면 훈련이 10배 더 오래 걸린다. 하지만 이 하이퍼파라미터를 랜덤 서치에 추가하면 **탐색 시간이 더 늘어나지 않는다.**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing, housing_labels)
```

사이킷런은 `HalvingRandomSearchCV`와 `HalvingGridSearchCV`도 제공한다. 첫 번째 반복에서 많은 후보를 **제한된 자원**(훈련 세트의 작은 일부분 등)으로 훈련하고, 최상의 후보만 다음 단계로 넘어가 더 많은 자원을 사용하게 하여 튜닝 시간을 단축한다.

### 2.7.3 앙상블 방법

모델을 세밀하게 튜닝하는 또 다른 방법은 **최상의 모델을 연결**해보는 것이다. 결정 트리의 앙상블인 랜덤 포레스트가 결정 트리 하나보다 성능이 좋은 것처럼, 모델의 그룹(앙상블)이 최상의 단일 모델보다 더 나은 성능을 발휘할 때가 많다. **특히 개별 모델이 각기 다른 형태의 오차를 만들 때 그렇다.** 이 주제는 7장에서 자세히 다룬다.

### 2.7.4 최상의 모델과 오차 분석

최상의 모델을 분석하면 문제에 대한 좋은 인사이트를 얻는 경우가 많다. `RandomForestRegressor`는 각 특성의 상대적인 중요도를 알려준다.

```python
final_model = rnd_search.best_estimator_    # 전처리 포함됨
feature_importances = final_model["random_forest"].feature_importances_

sorted(zip(feature_importances,
           final_model["preprocessing"].get_feature_names_out()),
       reverse=True)
```

교재의 출력 상위 항목:

| 중요도 | 특성 |
|---:|---|
| 0.1869 | `log__median_income` |
| 0.0748 | `cat__ocean_proximity_INLAND` |
| 0.0693 | `bedrooms__ratio` |
| 0.0545 | `rooms_per_house__ratio` |
| 0.0526 | `people_per_house__ratio` |
| 0.0382 | `geo__Cluster 0 similarity` |
| … | … |
| 0.00015 | `cat__ocean_proximity_NEAR BAY` |
| 7.3e-05 | `cat__ocean_proximity_ISLAND` |

이 정보를 바탕으로 덜 중요한 특성들을 제외할 수 있다.

> **TIP(교재).** `sklearn.feature_selection.SelectFromModel` 변환기는 자동으로 가장 덜 유용한 특성을 제거할 수 있다.

시스템이 특정한 오차를 만들었다면 **왜 그런 문제가 생겼는지 이해해야 한다.** 그리고 추가 특성을 포함시키거나, 불필요한 특성을 제거하거나, 이상치를 제외하는 등 해결 방법을 찾아야 한다.

교재가 이 절 마지막에 붙인 문단이 중요하다.

> 이제 모델이 평균적으로 잘 작동하는 것뿐만 아니라 **시골이든 도시든, 부유하든 가난하든, 북쪽이든 남쪽이든, 소수 민족이든 아니든 모든 구역에서 잘 작동하는지 확인할 차례다.** 각 범주에 대한 검증 세트를 만들려면 약간의 노력이 필요하지만 중요한 작업이다. 모델이 전체 범주에서 제대로 실행되지 않는다면 이 문제가 해결될 때까지 모델을 배포해서는 안 된다. 아니면 해당 범주에 대해서는 이 모델을 사용해 예측을 만들지 말아야 한다. 득보다 실이 많기 때문이다.

> **모빌리티 적용.** 이 문단이 프로젝트 14주차 "오차 분석"의 근거다. 전체 MAE만 보지 말고 **시간대별(출퇴근/한산), 정류장 구간별, 요일별**로 나눠 오차를 확인한다. 출근 시간대에만 크게 틀리는 모델은 평균 MAE가 좋아도 쓸 수 없다.

### 2.7.5 테스트 세트로 시스템 평가하기

어느 정도 모델을 튜닝하면 마침내 만족할 만한 모델을 얻게 된다. **이 과정에 특별히 다른 점은 없다.** 테스트 세트의 특성과 레이블을 사용해 `final_model`을 실행하여 데이터를 변환하고 예측을 만든 다음 평가한다.

```python
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)   # 41424.40026462184
```

일반화 오차의 **점 추정**만으로 론칭을 결정하기에 충분하지 않은 경우가 있다. 현재 제품 시스템에 있는 모델보다 불과 0.1% 더 높다면 어떨까? `scipy.stats.t.interval()`로 95% **신뢰 구간**을 계산할 수 있다.

```python
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
# array([39275.40861216, 43467.27680583])
```

구간은 39,275와 43,467 사이로 꽤 크며 이전 점 추정값 41,424는 대략 중간에 해당한다.

> 하이퍼파라미터 튜닝을 많이 했다면 교차 검증을 사용해 측정한 것보다 성능이 조금 낮은 것이 보통이다. 우리 시스템이 검증 데이터에서 좋은 성능을 내도록 세밀하게 튜닝되었기 때문에 새로운 데이터셋에는 잘 작동하지 않을 가능성이 크다. 이 예제에서는 테스트 RMSE가 검증 RMSE보다 낮기 때문에 성능이 낮아지지는 않았다. **하지만 이런 경우가 생기더라도 테스트 세트에서 성능 수치를 좋게 하려고 하이퍼파라미터를 튜닝하려 시도해서는 안 된다.** 그렇게 향상된 성능은 새로운 데이터에 일반화되기 어렵다.

론칭 직전 단계에서는 (학습한 것, 한 일과 하지 않은 일, 수립한 가정, 시스템 제한 사항 등을 강조하면서) 솔루션과 문서를 출시하고, 깔끔한 도표와 기억하기 쉬운 제목(예: '수입의 중간값이 주택 가격 예측의 가장 중요한 지표다')으로 발표 자료를 만들어야 한다.

---

## 2.8 론칭, 모니터링, 시스템 유지 보수

### 배포

가장 기본적인 방법은 훈련된 최상의 모델을 저장하고 제품 환경으로 이 파일을 전달하여 로드하는 것이다.

```python
import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")
```

> **TIP(교재).** 원하는 모델로 쉽게 돌아올 수 있도록 **실험한 모든 모델을 저장하는 것이 좋다.** 검증 점수와 검증 세트에 대한 실제 예측도 저장할 수 있다. 이렇게 하면 여러 종류의 모델이 만든 점수와 오류의 종류를 비교하기 쉽다.

모델을 로드하려면 모델이 사용하는 **모든 사용자 정의 클래스와 함수를 먼저 임포트해야 한다**(이 코드를 제품 환경으로 전달해야 한다는 의미다).

```python
import joblib
[...]  # import KMeans, BaseEstimator, TransformerMixin, rbf_kernel, ...

def column_ratio(X): [...]
def ratio_name(function_transformer, feature_names_in): [...]
class ClusterSimilarity(BaseEstimator, TransformerMixin): [...]

final_model_reloaded = joblib.load("my_california_housing_model.pkl")
new_data = [...]
predictions = final_model_reloaded.predict(new_data)
```

배포 형태는 웹 애플리케이션이 직접 `predict()`를 호출하는 방식, 전용 웹 서비스로 모델을 감싸는 방식(그림 2-20), 구글 버텍스 AI 같은 클라우드에 배포하는 방식이 있다. 웹 서비스로 감싸면 주 애플리케이션을 건드리지 않고 모델을 새 버전으로 업그레이드하기 쉽고, 로드 밸런싱으로 규모를 확장하기도 쉽다.

### 모니터링

**배포가 마지막이 아니다.** 일정 간격으로 시스템의 실시간 성능을 체크하고 성능이 떨어졌을 때 알림을 보낼 수 있는 모니터링 코드를 작성해야 한다.

- 인프라에서 특정 컴포넌트가 고장나는 경우 성능이 빠르게 감소할 수 있다.
- 반면에 성능이 아주 느리게 감소해서 긴 시간 동안 쉽게 눈에 띄지 않을 수 있다. **이는 모델 부패로 인해 매우 흔하게 발생하는 일이다.** 작년 데이터로 훈련된 모델이라면 오늘 데이터에는 적용할 수 없을 것이다.

모니터링 방법은 상황에 따라 다르다. 후속 시스템의 지표로 모델 성능을 추정할 수도 있다(추천 시스템이라면 추천 상품의 매일 판매량). 사람의 분석이 필요할 수도 있다(전문가 또는 크라우드소싱 플랫폼의 작업자).

데이터가 계속 변화하면 데이터셋을 업데이트하고 모델을 정기적으로 다시 훈련해야 한다. 자동화할 수 있는 작업은 다음과 같다.

- 정기적으로 새로운 데이터를 수집하고 레이블을 단다.
- 모델을 훈련하고 하이퍼파라미터를 자동으로 미세 튜닝하는 스크립트를 작성한다. 작업에 따라 매일 또는 매주 자동으로 이 스크립트를 실행할 수 있다.
- 업데이트된 테스트 세트에서 새로운 모델과 이전 모델을 평가하는 스크립트를 하나 더 작성한다. 성능이 감소하지 않으면 새로운 모델을 제품에 배포한다. **이 스크립트는 테스트 세트의 여러 서브셋에서 모델의 성능을 테스트해야 한다**(가난한 구역 또는 부유한 구역, 시골 또는 도시 등).

또한 **모델의 입력 데이터 품질을 평가해야 한다.** 점점 더 많은 입력에서 한 특성이 누락되거나, 평균 또는 표준 편차가 훈련 세트와 멀어지거나, 범주형 특성이 새로운 카테고리를 포함하는 경우 알람을 울릴 수 있다.

마지막으로 **만든 모든 모델을 백업**해야 한다. 새로운 모델이 어떤 이유로 올바르지 않게 작동하는 경우 이전 모델로 빠르게 롤백하기 위한 절차와 도구를 준비해야 한다. 비슷하게 **모든 버전의 데이터셋을 백업**해야 새로운 버전의 데이터셋이 오염되었을 때 롤백할 수 있다.

> 이처럼 머신러닝은 매우 많은 시스템과 연관되어 있다. 19장에서 논의하겠지만 이를 **MLOps**라 부른다. 따라서 첫 번째 머신러닝 프로젝트를 제품으로 만들고 배포하는 데 많은 노력과 시간이 든다고 놀라지 마라. 이 모든 시스템이 준비되고 나면 아이디어를 제품으로 구현하는 일이 훨씬 빨라질 것이다.

---

## 2.9 직접 해보세요!

> 대부분의 작업은 **데이터 준비 단계, 모니터링 도구 구축, 사람의 평가 파이프라인 세팅, 주기적인 모델 학습 자동화**로 이루어진다. 물론 머신러닝 알고리즘도 중요하지만 전체 프로세스에 익숙해져야 한다. **고수준 알고리즘을 탐색하느라 시간을 모두 허비해서 전체 프로세스 구축에 충분한 시간을 투자하지 못하는 것보다 서너 개의 알고리즘만으로라도 전체 프로세스를 올바로 구축하는 편이 더 낫다.**

## 연습문제 (교재 원문)

이 장에서 소개한 주택 가격 데이터셋을 사용해 문제를 풀어보세요.

1. 서포트 벡터 머신 회귀(`sklearn.svm.SVR`)를 `kernel="linear"`(하이퍼파라미터 `C`를 바꿔가며)나 `kernel="rbf"`(하이퍼파라미터 `C`와 `gamma`를 바꿔가며) 등의 다양한 하이퍼파라미터 설정으로 시도해보세요. 서포트 벡터 머신은 대용량 데이터셋에 적용하기가 쉽지 않으므로 훈련 세트의 처음 5,000개 샘플만 사용해 모델을 훈련하고 3-폴드 교차 검증을 사용하세요. 최상의 SVR 모델은 무엇인가요?
2. `GridSearchCV`를 `RandomizedSearchCV`로 바꿔보세요.
3. 가장 중요한 특성을 선택하는 `SelectFromModel` 변환기를 준비 파이프라인에 추가해보세요.
4. `fit()` 메서드 안에서 k-최근접 이웃 회귀(`sklearn.neighbors.KNeighborsRegressor`)를 훈련하고 `transform()` 메서드에서 이 모델의 예측을 반환하는 사용자 정의 변환기를 만들어보세요. 이 변환기의 입력으로 위도와 경도를 사용하고 예측 결과를 하나의 특성으로 전처리 파이프라인에 추가하세요.
5. `GridSearchCV`를 사용해 준비 단계의 옵션을 자동으로 탐색해보세요.
6. `StandardScalerClone` 클래스를 처음부터 다시 구현하세요. 그다음 `inverse_transform()` 메서드를 추가하세요. 그다음 특성 이름을 지원하는 기능을 추가하세요.

---

## 수업용 정리

| 교재 단계 | 3주차 실습에서 하는 일 | ETA 프로젝트로 옮길 때 |
|---|---|---|
| 문제 정의 | 회귀·다중회귀·단변량회귀·배치 판정 | BIS 도착예정시간은 특성이 아니라 비교 기준 |
| 지표 선택 | RMSE vs MAE, 노름의 의미 | 주 지표 MAE |
| 테스트 세트 분리 | `train_test_split`, 계층적 샘플링 | **날짜 기준 분할** |
| 탐색·시각화 | 지리적 산점도, `corr()`, `scatter_matrix` | 노선별·시간대별 통행시간 분포 |
| 특성 조합 | 비율 특성 3개 | 남은 정류장 수, 최근 구간 통행시간 |
| 전처리 | `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `ColumnTransformer` | 노선·정류장은 고차원 범주형 → 임베딩 예고 |
| 모델 훈련 | 선형회귀 → 결정 트리 → 랜덤 포레스트 | 중앙값 기준 모델 → XGBoost |
| 교차 검증 | `cross_val_score(cv=10)` | `TimeSeriesSplit` |
| 미세 튜닝 | `GridSearchCV`, `RandomizedSearchCV` | 랜덤 서치 우선 |
| 오차 분석 | `feature_importances_`, 범주별 성능 확인 | 시간대·구간별 오차 |
| 최종 평가 | 테스트 세트 1회, 신뢰 구간 | 마지막 주까지 미룬다 |

## 확인 문제

1. 검증 데이터와 테스트 데이터의 역할 차이는 무엇인가?
2. 범주형 특성에 `OrdinalEncoder`로 정수 코드를 바로 부여하면 어떤 문제가 생길 수 있는가?
3. 전처리기를 전체 데이터에 먼저 `fit`하면 왜 데이터 누수인가?
4. ETA 데이터에 무작위 분할이 부적절한 이유는 무엇인가?
5. `get_dummies()` 대신 `OneHotEncoder`를 쓰는 이유는 무엇인가?
6. 결정 트리의 훈련 RMSE가 0인데 교차 검증 RMSE가 66,868인 것은 무엇을 의미하는가?

## 관련 자료

- [Ch1 — 한눈에 보는 머신러닝](week02_ch01.md)
- [Ch3 — 분류](week04_ch03.md)
- [주차별 일정](../../schedule.md)
- [공식 실습 코드](https://github.com/rickiepark/handson-ml3)
