# 랭체인으로 LLM 기반의 AI 서비스 개발하기 - 서지영

## 목차

1. LLM 훑어보기
2. LLM 활용하기
3. RAG 훑어보기
4. 랭체인 익숙해지기
5. 랭체인으로 RAG 구현하기
    1. LLM을 이용한 간단한 챗봇
    2. 랭체인과 챗GPT로 RAG 기반의 챗봇
    3. PDF를 요약해주는 웹사이트
    4. PDF 파일에 대한 독립형 질문을 하는 챗봇
    5. 대화형 챗봇
    6. 번역 서비스
    7. 메일 작성기
    8. LLM을 이용해서 CSV 파일 분석
6. LLM을 이용한 서비스 알아보기

---

## 1. LLM 훑어보기

### 언어 모델(Language Model)

- 통계적 언어 모델
- 신경망 언어 모델
- 트랜스포머

### LLM 생태계

- Infrastructure Layer
- Application Layer

### LLM, GAI, SLM

- Generative AI
- prompt, completion
- Small Language Model

### LLM 생성 과정

1. **데이터 수집 및 준비**
    - 데이터 수집 > 데이터 정제 > 데이터 전처리 > 데이터 형식 변경
    - 데이터 전처리 : 토큰화(tokenization), 정규화
2. **모델 설계**
    - 신경망 아키텍처(주로 트랜스포머) 구축, hyperparameter(계층수, 학습률, 배치 크기...) 설정
3. **모델 학습**
    - 모델링 : 주어진 데이터를 기반으로 일반화된 패턴이나 규칙을 만드는 것.
4. **평가 및 검증**
    - 정확도(accuracy)
    - 정밀도(precision)
    - 재현율(recall)
    - F1 점수(F1 score)
    - ROC-curve/AUC
5. **배포 및 유지 보수**

### LLM 생성 후 추가 고려 사항

- 윤리적 고려 및 보정 : 책임감 있는 AI(Responsible AI)
- 지속적 모니터링

---

## 2. LLM 활용하기

### LLM 활용 방법

1. 파인튜닝(Fine-Tuning)
2. RAG