---
course: AI 모빌리티 · 가천대학교 스마트시티학과
week: 2주차 실습
title: 개발 환경과 AI 코딩 도구
subtitle: 설치, 첫 실행, 결과 확인
presenter: 여지호 교수 · 스마트시티학과
date: 2026년 9월 9일 (수)
contact: jihoyeo@gachon.ac.kr · 오피스 아워 수 17:00–20:00
footer: AI 모빌리티 · 2026 가을학기
closing: 다음 주 실습에서 이어서
closing_sub: End-to-End 프로젝트 노트북 (핸즈온 Ch2)
toc: true
toc_label: Week 02 Lab
---

<!-- 덱 설계
청중: 2주차 화요일 이론(핸즈온 Ch1)을 마친 2학년. 파이썬 숙련도 혼재, 설치 경험 없는 학생 다수
시간: 수 실습 50분 → 본문 6장. 설명은 짧게, 대부분의 시간을 설치와 확인에 쓴다
메시지: 코드를 쓰는 시간보다 결과를 확인하는 시간이 길어지는 것이 정상이다
목표: (1) 두 도구를 설치하고 로그인한다 (2) pandas로 데이터를 읽고 요약한다 (3) AI가 만든 코드에서 확인할 세 가지를 익힌다
출처: schedule.md 실습 환경 · 교재 주택 데이터(핸즈온 Ch2에서 사용)
제외: 파이썬 문법 강의, 실습 데이터 소개(5주차)
-->

# 오늘 할 일

## 실습 순서
- GitHub 계정 생성 — 과제 제출에도 사용
- Antigravity 설치와 로그인 (Google 계정)
- Codex 설치와 로그인 — 막히면 웹 버전
- pandas 첫 실행 — 3주차에 쓸 교재 데이터
> 50분. 설치에서 막히는 학생이 반드시 나오므로 순서대로 확인하며 진행한다.

# 도구

## 사용할 도구
- 에디터: Google Antigravity — VS Code 기반, 에이전트 내장
- 코딩 에이전트: OpenAI Codex — 터미널과 웹에서 사용
- 대비책: Google Colab — 설치 없이 브라우저에서 실행
- 분석: Python, pandas, scikit-learn
> 셋 다 무료 플랜으로 시작. 한도가 자주 바뀌므로 과제는 한도를 전제하지 않는다.

## 파이썬 문법 학습 방식
- 별도 문법 강의 없음 — 필요한 시점에 도구로 해결
- 확보한 시간은 결과를 확인하는 훈련에 배분
- 배열 연산은 6주차 신경망 구현에서 별도 설명
> 파이썬이 처음인 학생도 오늘부터 바로 시작할 수 있게 한다.

# 첫 실행

## pandas로 데이터 읽기
- 3주차 End-to-End에서 쓸 교재 데이터로 미리 연습

```python
import pandas as pd

base = "https://raw.githubusercontent.com/ageron/data/main"
df = pd.read_csv(f"{base}/housing/housing.csv")

print(df.shape)                          # (결과) (20640, 10)
print(df["median_house_value"].mean())   # (결과) 206855.82
print(df.isna().sum().sum())             # (결과) 207
```
> 두 도구에 같은 요청을 넣고 결과가 같은지 본다. 값이 다르면 그 자리에서 원인을 찾는다.

## AI 코드 확인 항목
- 행 수: 읽어들인 결과가 원본과 일치하는가
- 결측: 삭제인지 대체인지, 대체면 어떤 값인지
- 필터: 조건이 의도한 범위와 일치하는가
- 확인 없이 넘어간 코드가 오차의 출처
> 코드를 쓰는 시간보다 확인하는 시간이 길어지는 것이 정상이다.

# 마무리

## 오늘 확인할 것
- 세 패키지 import 성공 — pandas, numpy, scikit-learn
- 데이터 20,640행을 읽고 요약 통계 출력
- 두 도구 모두에서 같은 결과 확인
- 결측 207개를 어떻게 처리할지 도구에 물어보기
> 막힌 학생은 Colab으로 전환하고 다음 주까지 로컬 환경을 정리한다.
