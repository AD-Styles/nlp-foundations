# 🧠 Deconstructing NLP: Mathematical Foundations & Transfer Learning Optimization
> **Implementing Attention from Scratch, Pipeline Analysis, and Performance Engineering with KLUE-BERT**

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 자연어 처리(NLP) 모델의 '동작 원리 이해'와 '실무적 성능 최적화'를 동시에 증명하는 엔지니어링 포트폴리오입니다. 라이브러리에 의존하지 않는 **Attention 메커니즘의 수학적 재구성**으로 기초를 다지고, **Hugging Face 파이프라인의 내부 추론 공정을 수동으로 제어**하며, 최종적으로 사전 학습된 모델을 **전이 학습(Transfer Learning)을 통해 도메인 특화 모델로 진화**시키는 과정을 체계적으로 기록했습니다.


## 🎯 핵심 목표 (Motivation)
| 구분 | 세부 내용 |
| :--- | :--- |
| **수학적 설계 (Bottom-up)** | 논문 수준의 $Q, K, V$ 행렬 연산을 Numpy로 구현하여 트랜스포머 아키텍처 파악. |
| **추론 공정 해체 (Analysis)** | `pipeline()` API의 블랙박스를 걷어내고 Tokenizer부터 Logits 생성까지의 텐서 흐름을 파악. |
| **전이 학습 역량 (Performance)** | 베이스 모델을 NSMC(영화 리뷰) 데이터에 맞춰 미세 조정하여 실무 수준의 정확도를 확보. |


## 📂 프로젝트 구조 (Project Structure)
```text
📂 nlp-foundations
├── 📄 README.md
├── 📄 .gitignore
├── 📄 LICENSE
├── 📂 src
│   ├── 📄 01_attention_scratch.py         # 어텐션 스코어 및 가중치 수학적 구현
│   ├── 📄 02_pipeline_deconstruction.py    # 추론 파이프라인 단계별 수동 제어
│   └── 📄 03_klue_bert_fine_tuning.py      # 전이 학습 및 성능 최적화 프로세스
```

## 🛠️ 주요 알고리즘 및 기술적 차별점 (Key Features)

### 1. Scaled Dot-Product Attention Scratch Implementation
라이브러리 도움 없이 어텐션 메커니즘을 구현하여 고차원 텐서 연산 능력을 증명했습니다.

| 구현 단계 | 핵심 로직 및 수식 | 기술적 포인트 |
| :--- | :--- | :--- |
| **Step 1. Score** | $Score = \frac{Q \cdot K^T}{\sqrt{d_k}}$ | 내적(Dot-product)을 통한 단어 간 유사도 산출 및 스케일링 적용 |
| **Step 2. Weight** | $Weight = \text{Softmax}(Score)$ | 소프트맥스 함수를 통한 확률 분포 변환 및 중요도 가중치 할당 |
| **Step 3. Context** | $Context = Weight \cdot V$ | 가중치가 반영된 최종 문맥 벡터(Context Vector) 도출 |

### 2. Inference Pipeline Deconstruction
Hugging Face의 추론 과정을 직접 설계하여 모델 제어권을 확보했습니다.

| 프로세스 순서 | 수행 내용 | 활용 모듈 |
| :--- | :--- | :--- |
| **1. Tokenization** | 텍스트 데이터를 모델 입력용 정수 인덱스 및 텐서로 변환 | `AutoTokenizer` |
| **2. Forward Pass** | 모델의 연산 그래프를 통과하여 원시 로짓(Logits)값 추출 | `AutoModelForSequenceClassification` |
| **3. Post-Processing** | 로짓값에 Softmax를 적용하여 최종 클래스별 확률값 도출 | `torch.nn.functional.softmax` |

## 🔍 비판적 탐색 및 분석 (Critical Exploration)
전문 엔지니어의 관점에서 모델의 성능 뒤에 숨겨진 한계점을 분석했습니다.

| 분석 항목 | 세부 분석 내용 |
| :--- | :--- |
| **사회적 편향성 탐색** | `[MASK]` 토큰 추론 결과, 성별이나 직업군에 따라 모델이 특정 고정관념을 출력하는 편향성을 확인했습니다. |
| **도메인 적응성 한계** | NSMC(영화 리뷰)로 학습된 모델이 일상 대화나 비즈니스 메일에서는 성능이 저하되는 현상을 포착했습니다. |
| **대응 전략 제안** | 데이터 증강(Augmentation) 및 가드레일 설정을 통한 모델의 편향성 완화 방안을 고찰했습니다. |

## 📊 학습 개념의 직관적 해석 (Analogies)
| 개념 | 비유 (Analogy) | 기술적 의미 설명 |
| :--- | :--- | :--- |
| **Attention** | **형광펜** | 문장에서 결과 예측에 결정적인 영향을 주는 단어에 '하이라이트'를 칠하는 과정 |
| **Q, K, V** | **도서관 검색** | 검색어(Q)로 제목(K)을 찾아 내용(V)을 가져오는 매칭 메커니즘 |
| **Masked LM** | **빈칸 채우기** | 앞뒤 문맥을 통해 생략된 정보를 유추하며 언어의 통계적 구조를 학습 |

## 💡 회고록 (Retrospective)

1. 기술적 성취와 구현의 즐거움이번 프로젝트를 통해 단순히 transformers 라이브러리의 pipeline() 함수를 사용하는 단계를 넘어, 그 내부에서 흐르는 텐서의 차원과 연산 과정을 손으로 직접 그려볼 수 있는 수준까지 성장했습니다. 특히 Numpy로 Attention을 직접 구현하며 $d_k$로 나누어주는 Scaling 과정이 왜 Softmax의 기울기 소실을 막는지 수식적으로 이해한 순간이 가장 큰 수확이었습니다.
2. 데이터 중심 AI(Data-Centric AI)의 체감똑같은 아키텍처임에도 불구하고, 적절한 데이터셋으로 전이 학습을 수행했을 때 정확도가 50%에서 89%로 치솟는 과정을 목격하며 "좋은 알고리즘만큼 중요한 것은 양질의 데이터와 적절한 Fine-tuning 전략"이라는 사실을 뼈저리게 느꼈습니다. 단순히 모델을 돌려보는 것보다 하이퍼파라미터를 조정하고 학습 곡선을 모니터링하는 과정의 중요성을 깨달았습니다.
3. 실무적 한계에 대한 고찰성능 지표는 훌륭했지만, 비판적 탐색 과정에서 발견한 '반어법 추론의 어려움'과 '도메인 편향성' 문제는 저에게 큰 과제를 안겨주었습니다. 실무에서는 단순히 정확도가 높다고 바로 배포하는 것이 아니라, 이러한 예외 케이스(Edge Cases)에 대한 가드레일이 필요하다는 점을 인지하게 되었습니다.
