# 🧠 Deconstructing NLP: Mathematical Foundations & Transfer Learning Optimization
> **Implementing Attention from Scratch, Pipeline Analysis, and Performance Engineering with KLUE-BERT**

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)


## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 자연어 처리(NLP) 모델의 '동작 원리 구현'과 '실무적 성능 최적화'를 입증하는 엔지니어링 포트폴리오입니다. 라이브러리에 의존하지 않는 **Attention 메커니즘의 수학적 설계**를 시작으로, **Hugging Face 추론 파이프라인의 내부 공정을 수동으로 해체**합니다. 최종적으로 사전 학습된 모델을 **전이 학습(Transfer Learning)을 통해 정확도를 50%에서 89%까지 개선**하고 모델의 사회적 편향성을 분석한 과정을 기록했습니다.


## 🎯 핵심 목표 (Motivation)
| 구분 | 세부 내용 |
| :--- | :--- |
| **수학적 설계** | 논문에 제시된 $Q, K, V$ 행렬 연산을 Numpy로 구현하여 트랜스포머의 핵심 엔진을 파악. |
| **추론 공정 해체** | `pipeline()` API의 블랙박스를 걷어내고 Tokenizer부터 Logits 생성까지의 텐서 흐름을 파악. |
| **전이 학습 및 성능 도약** | 베이스 모델을 NSMC(영화 리뷰) 데이터에 맞춰 미세 조정하여 무작위 추측 수준의 성능을 실무급으로 개선. |
| **비판적 모델 검증** | Masked LM을 활용하여 모델이 내포한 사회적 편향성을 정량적으로 탐색하고 윤리적 한계를 정의. 


## 📂 프로젝트 구조 (Project Structure)
```text
📂 nlp-foundations
├── 📄 README.md
├── 📄 .gitignore
├── 📄 LICENSE
├── 📂 src
│   ├── 📄 01_attention_scratch.py         # 어텐션 스코어 및 가중치 수학적 구현
│   ├── 📄 02_pipeline_deconstruction.py    # 추론 파이프라인 단계별 수동 제어
│   └── 📄 03_model_analysis_and_tuning.py      # 전이 학습 및 성능 최적화 프로세스
```


## 🛠️ 주요 알고리즘 및 기술적 구현 (Technical Implementation)

### 1. Mathematical Reconstruction of Attention
Numpy를 활용해 어텐션 메커니즘의 연산 흐름을 재구성하여 고차원 데이터의 상호작용 원리를 증명했습니다.

| 구현 단계 | 핵심 로직 및 수식 | 기술적 포인트 |
| :--- | :--- | :--- |
| **Step 1. Score** | $Score = (Q \cdot K^T) / \sqrt{d_k}$ | Dot-product 유사도 산출 및 Scaling을 통한 수치적 안정성 확보 |
| **Step 2. Weight** | $Weight = \text{Softmax}(Score)$ | 소프트맥스 함수를 활용한 토큰별 확률 분포 및 가중치 할당 |
| **Step 3. Context** | $Context = Weight \cdot V$ | 가중치가 반영된 최종 문맥 벡터(Context Vector) 도출 |

### 2. Inference Pipeline Deconstruction
Hugging Face의 추론 과정을 직접 설계하여 모델 제어권을 확보했습니다.

| 프로세스 순서 | 수행 내용 | 활용 모듈 |
| :--- | :--- | :--- |
| **1. Tokenization** | 텍스트를 모델 입력 규격에 맞는 정수 인덱스 및 어텐션 마스크로 변환 | `AutoTokenizer` |
| **2. Forward Pass** | 모델의 연산 그래프를 통과하여 분류를 위한 Logits값 추출 | `AutoModelForSequenceClassification` |
| **3. Activation** | Logits에 Softmax를 적용하여 최종 클래스별 신뢰도 확률 도출 | `torch.nn.functional.softmax` |

### 3. Performance Engineering: The 89% Accuracy Leap
전이 학습 전후의 성능 지표 비교를 통해 실무적 가치 창출 능력을 증명합니다.

| 분석 지표 | 학습 전 (Base Model) | 학습 후 (Fine-tuned) | 비고 |
| :--- | :--- | :--- | :--- |
| **정확도(Accuracy)** | **약 50.2%** | **약 89.1%** | 도메인 특화 학습을 통한 성능 도약 |
| **특이사항** | 무작위 추측 수준 | 반어법 및 구어체 문맥 파악 가능 | NSMC 데이터셋 최적화 완료 |


## 🔍 비판적 탐색 및 분석 (Critical Analysis)
| 분석 항목 | 세부 분석 내용 |
| :--- | :--- |
| **사회적 편향성 탐색** | `[MASK]` 추론 결과, 성별이나 직업군에 따라 모델이 특정 고정관념을 출력하는 편향성을 확인. |
| **도메인 적응성 한계** | NSMC(영화 리뷰)로 학습된 모델이 뉴스나 법률 등 타 도메인에서는 성능이 저하되는 현상을 포착. |
| **대응 전략 제안** | 이터 증강(Augmentation) 및 가드레일 설정을 통한 모델의 편향성 완화 방안을 탐구. |


## 📊 학습 개념의 직관적 해석 (Analogies)
| 개념 | 비유 (Analogy) | 기술적 의미 설명 |
| :--- | :--- | :--- |
| **Attention** | **형광펜** | 문장에서 결과 예측에 결정적인 영향을 주는 단어에 '하이라이트'를 칠하는 과정 |
| **Q, K, V** | **도서관 검색** | 검색어(Q)로 제목(K)을 찾아 내용(V)을 가져오는 매칭 메커니즘 |
| **Masked LM** | **빈칸 채우기** | 앞뒤 문맥을 통해 생략된 정보를 유추하며 언어의 구조를 학습하는 방식 |


## 💡 회고록 (Retrospective)

1. 기술적 성취와 구현의 즐거움이번 프로젝트를 통해 단순히 transformers 라이브러리의 pipeline() 함수를 사용하는 단계를 넘어, 그 내부에서 흐르는 텐서의 차원과 연산 과정을 손으로 직접 그려볼 수 있는 수준까지 성장했습니다. Numpy로 Attention을 직접 구현하며 텐서 연산의 흐름을 완벽히 이해하게 되었습니다. 특히 차원의 제곱근으로 나누어주는 Scaling 과정이 Softmax의 기울기 소실을 막기 위해 왜 필수적인지 수식적으로 증명한 과정이 가장 인상 깊었습니다.
2. 동일한 모델이라도 도메인 데이터(NSMC)로 전이 학습을 수행했을 때, 적절한 데이터셋으로 전이 학습을 진행하면 정확도가 50%에서 89%로 치솟는 과정을 목격하며 "좋은 알고리즘만큼 중요한 것은 양질의 데이터와 적절한 Fine-tuning 전략"이라는 사실을 느꼈습니다. 단순히 모델을 돌려보는 것보다 하이퍼파라미터를 조정하고 학습 곡선을 모니터링하는 과정의 중요성을 깨달았습니다.
3. 비판적 탐색 과정에서 발견한 '반어법 추론의 어려움'과 '도메인 편향성' 문제는 저에게 큰 과제를 안겨주었습니다. 또한 모델의 성능 수치 이면에 숨겨진 사회적 편향성을 직접 코드로 확인하면서, 책임감 있는 AI 개발의 중요성을 체감했습니다. 실무에서는 단순히 정확도가 높다고 바로 배포하는 것이 아니라, 모델 배포 전 반드시 거쳐야 할 검증 프로세스로서 '비판적 탐색'의 가치를 재정의하게 되었습니다.
