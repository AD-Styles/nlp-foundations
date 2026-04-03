# 🧠 NLP Foundations: Deep Dive into Model Internals

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 자연어 처리(NLP) 모델의 내부 동작 원리를 완벽하게 장악하는 것을 목표로 합니다. 단순히 라이브러리를 호출하는 수준을 넘어, **Attention Mechanism을 분석**하고 **Hugging Face 파이프라인의 추론 공정(Inference Process)을 수동으로 분해**하여 엔지니어링 역량을 증명합니다. 또한, 언어 모델의 사회적 편향성을 비판적으로 탐색하여 기술적/윤리적 완성도를 높였습니다.

## 🎯 핵심 목표 (Motivation)
| 구분 | 세부 내용 |
| :--- | :--- |
| **수학적 설계 (Bottom-up)** | 논문에 제시된 $Q, K, V$ 연산 수식을 Numpy 기반 코드로 직접 구현하여 모델의 근본 원리를 이해합니다. |
| **시스템 구조 해체 (Analysis)** | `pipeline()` API의 내부 동작을 3단계(Encoding → Forward → Softmax)로 분해하여 데이터 흐름을 제어합니다. |
| **비판적 사고 (Critical View)** | 사전 학습된 모델(`klue/bert-base`)이 가진 사회적 편향성을 정량적으로 분석하고 모델의 한계를 정의합니다. |

## 📂 프로젝트 구조 (Project Structure)
```text
📂 nlp-foundations
├── 📄 README.md
├── 📄 .gitignore
├── 📄 LICENSE
├── 📂 src
│   ├── 📄 01_attention_scratch.py        # 어텐션 스코어 및 가중치 직접 구현
│   ├── 📄 02_pipeline_deconstruction.py   # 파이프라인 3단계 수동 추론 프로세스
│   └── 📄 03_model_bias_analysis.py       # 언어 모델의 사회적 편향성 탐색
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
