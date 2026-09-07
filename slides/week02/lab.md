---
course: AI 모빌리티 · 가천대학교 스마트시티학과
week: 2주차 실습
title: 개발 환경과 AI 코딩 도구
subtitle: 설치하고, 실행하고, 결과 확인하기
presenter: 여지호 교수 · 스마트시티학과
date: 2026년 9월 9일 (수)
contact: jihoyeo@gachon.ac.kr · 오피스 아워 수 17:00–20:00
footer: AI 모빌리티 · 2026 가을학기
closing: 다음 주 실습에서 이어서
closing_sub: 핸즈온 Ch2 주택 가격 예제
toc: true
toc_label: Week 02 Lab
---

<!-- 덱 설계
시간: 50분
원칙: 설명보다 설치와 실행에 시간을 쓴다. 설치가 어려우면 Google Colab으로 바로 전환한다.
-->

# 오늘 할 일

## 순서

1. GitHub 계정 확인
2. Antigravity 설치와 로그인
3. Codex 설치와 로그인
4. 교재 데이터 읽기

> 한 단계씩 함께 진행하고 완료 여부를 확인합니다.

## 사용할 환경

- 편집기: Google Antigravity
- 코딩 도구: OpenAI Codex
- 실행 환경: 로컬 Python 또는 Google Colab
- 패키지: pandas·NumPy·scikit-learn

> 로컬 설치 문제를 오늘 모두 해결하려고 시간을 쓰지 않습니다. 실행이 안 되면 Colab에서 수업을 계속합니다.

# 첫 실행

## 교재의 주택 데이터 읽기

```python
import pandas as pd

base = "https://raw.githubusercontent.com/ageron/data/main"
housing = pd.read_csv(f"{base}/housing/housing.csv")

housing.info()
housing.head()
housing.describe()
```

> 다음 주 Ch2에서 사용할 데이터입니다. 교재 노트북과 같은 순서로 크기, 열, 결측값, 요약 통계를 확인합니다.

## 실행 뒤 확인할 것

- 20,640행인가?
- 열은 10개인가?
- `total_bedrooms`의 결측값은 몇 개인가?
- 범주형 열은 무엇인가?

> AI가 만든 코드도 이 질문에 답할 수 있어야 합니다. 실행에 성공했다는 사실만으로 맞는 코드라고 판단하지 않습니다.

# 마무리

## 다음 주 준비

- 세 패키지가 오류 없이 불러와지는지 확인
- 핸즈온 Ch2 읽기
- 공식 Ch2 노트북 열어 보기

> 다음 실습부터는 교재의 `02_end_to_end_machine_learning_project.ipynb`를 함께 실행합니다.
