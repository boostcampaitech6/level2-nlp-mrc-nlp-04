# **📝 Project 소개**

| 항목 | 내용 |
| --- | --- |
| 프로젝트 주제 | 문장 속 단어(Entity)에 대한 속성과 관계를 예측하는 관계 추출(Relation Extraction) 업무를 수행 |
| 프로젝트 구현 내용 | • 단어 간 관계를 의미하는 30개 라벨 각각에 대해, subject와 object가 해당 클래스에 속할 확률을 예측<br>• 평가 지표로는 1) no-relation class를 제외한 micro F1 score, 2) 모든 클래스에 대한 AUPRC가 사용 |
| 진행 기간 |  2024년 1월 3일 ~ 2024년 1월 18일 |

### ⚙️ 개발 환경 및 협업 환경

<img width="700" alt="Lv1. 대회 Wrap Up Report.png" src="https://github.com/boostcampaitech6/level2-klue-nlp-04/assets/67735022/f63c6ef2-0ccd-4f01-9d0c-2816ba63a813">

> **Notion, Slack, Zoom** 을 통해 회의를 진행했으며, 코드의 경우는 모듈화하여 Make파일로 자동화하여 관리했습니다. 이때, **Github**을 통해 코드 공유를 진행했으며, **Wandb**를 이용해 실시간으로 실험을 관리했습니다. 
> 

# **👨‍👩‍👧‍👦 Team & Members** 소개


### 💁🏻‍♂️ Members

| 구희찬 | 김민석 | 손유림 | 오태연 | 이상수 | 최예진 |
| --- | --- | --- | --- | --- | --- |
| <img src='https://avatars.githubusercontent.com/u/67735022?v=4' height=100 width=100></img> | <img src='https://avatars.githubusercontent.com/u/63552400?v=4' height=100 width=100></img> | <img src='https://avatars.githubusercontent.com/u/139030224?v=4' height=100 width=100></img> | <img src='https://avatars.githubusercontent.com/u/122765534?v=4' height=100 width=100></img> | <img src='https://github.com/boostcampaitech6/level2-klue-nlp-04/assets/118837517/344540c3-a093-4cb8-a694-61164a7380f8' height=100 width=100></img> | <img src='https://avatars.githubusercontent.com/u/69586041?v=4' height=100 width=100></img> |
| <a href="https://github.com/kooqooo" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/maxseats" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/alwaysday4u" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/ohbigkite" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/SangSusu-git" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/yeh-jeans" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> |

### 👸🏻 Members' Role

> 전반적인 프로젝트 과정을 모두 경험할 수 있도록 분업을 하여 협업을 진행했으며, 초기 개발 환경 구축 단계부터 Github을 연동하여 세부 task 마다 issue와 branch를 생성하여 각자의 task를 수행했다. 이를 통해 코드 버전 관리와 코드 공유가 원활하게 이루어질 수 있었다.
> 

| 이름 | 역할 |
| --- | --- |
| **구희찬**  | 환경 설정 및 템플릿 관리, 베이스라인 코드 리팩토링, 데이터 라벨 검수, 모델 아키텍쳐 개선(3-classification, LSTM layer), Github issue 관리 및 PR merge, 하이퍼파라미터 튜닝 실험, 검증 데이터셋 비율 변경 실험, 사용성 개선 및 사후분석용 유틸 제작  |
| **김민석**  | 환경 설정 및 템플릿 관리, 베이스라인 모델 탐색 및 실험, 모델 아키텍쳐 개선(LSTM layer, Focal Loss 적용 및 실험), 전처리(query 추가), 검증 데이터셋 생성 및 코드 작성, 전반적인 방법론 정리, 실험 결과 사후 분석  |
| **손유림**  | 베이스라인 모델 실험, 사전 조사 및 방법론 정리, 모델 개선 관련 논문 발제(entity marking, TAPT), 전처리 및 증강 시도(entity marking, 역번역, 한자 제거, MLM), 모델링 실험(3-classification)  |
| **오태연**  | 전반적인 방법론 정리 및 노션 관리, WandB 관리, EDA 및 데이터 분석, 전처리 및 증강 시도(label reverse, MLM증강, 소스 스페셜 토큰 추가), tokenizer 분석, 모델 아키텍쳐 개선(LSTM layer), 실험 결과 사후 분석 코드 작성 및 결과 비교  |
| **이상수**  | 베이스라인 모델 실험 및 비교, 모델 아키텍쳐 개선(3-classification, Huggingface Roberta 코드 분석 및 LSTM layer 추가), 전처리 및 증강 시도(역번역 및 entity 추출), 하이퍼파라미터 튜닝 실험, 데이터 라벨 분석 및 실험 결과 사후 분석  |
| **최예진**  | 모델 아키텍쳐 개선(3-classification, Early Stopping, Trainer 코드 분석), 전처리 및 증강 시도(data copy, backtranslation), 검증 데이터셋 생성 및 코드 작성, 데이터 분석 및 실험 결과 사후 분석, 앙상블 코드 작성  |

# 💾 데이터 소개

### 데이터셋 통계


- 전체 데이터에 대한 통계는 다음과 같습니다.
    - `train.csv` : 총 32470개
    - `test_data.csv` : 총 7765개 (정답 라벨 blind = 100으로 임의 표현)

<img width="510" src=https://github.com/boostcampaitech6/level2-klue-nlp-04/assets/67735022/190e5528-e171-43ab-95d3-39fd88d2a4bd>

### **Data 예시**




![16bc4f53-f355-4b9d-968f-657bb5d9b5e5](https://github.com/boostcampaitech6/level2-klue-nlp-04/assets/67735022/9ce4af17-e6f8-42d3-95ed-eaa0b15221a9)

- column 1: 샘플 순서 id
- column 2: sentence.
- column 3: subject_entity
- column 4: object_entity
- column 5: label
- column 6: 샘플 출처

## 💡 Methods

| 분류 | 내용 |
| --- | --- |
| **모델** | • 실험한 모델 : 최종적으로 `klue/roberta-large` 사용<br>`klue/roberta-large`, `klue/roberta-base`,  `klue/roberta-small`, `monologg/koelectra-base-v3-discriminator`, `snunlp/KR-ELECTRA-discriminator`, `beomi/kcbert`, `xlm/roberta-large`, `kykim/bert-kor-base`, `kykim/electra-kor-base` <br>• LSTM layer 추가 : Classification 단계에서 LSTM layer를 추가해줌으로써 일부 토큰의 결과 벡터만을 사용하던 기존 구조 개선, 문장 전체 벡터를 활용할 수 있는 LSTM layer를 추가 |
| **데이터 전처리** | • (Typed) Entity Marker : entity의 위치 정보를 marker로 제공하고 entity의 유형을 제공해서 학습 성능 향상을 시도<br>• 데이터 Query 추가하기 : BERT의 QA Task 학습 방식을 적용하고자 함- sentence 앞 부분에 질문 형태의 쿼리 추가 (예시 : [SUB]와 [OBJ]의 관계는 무엇인가? [SEP] [sentence] [SEP])<br>• Source 스페셜 토큰 추가 : 소스별 타겟값의 분포가 다른 것을 확인, 쿼리문 앞에 3가지 소스 스페셜 토큰을 추가해줌 - [W_PED],[W_TR], [POL]<br>• 한자 제거 : 토큰 결과의 UNK 최소화를 위함. 가장 많이 UNK로 토큰화되었던 한자어 제거 |
| **데이터 증강 및 조정** | • Label Reverse 증강 : 서로 상충되는 의미의 라벨과, subject와 object를 바꿔도 괜찮은 라벨의 경우 subject와 object를 반대로 swap하여 데이터 증강, 10939개의 데이터 증가<br>• Back-Translation 증강 : GoogleTrans 라이브러리를 활용해 문장을 영어로 번역한 후, 이를 다시 한국어로 번역하여 데이터 증강<br>• MLM 증강 :  BERT 기반 모델들의 MLM 학습 방식에서 착안, [MASK] 부분이 기존 문장과 다른 새로운 token으로 패러프레이징 될 것임을 가정하고,증강에 활용 |
| **아키텍쳐 보완** | 1. 과적합 방지<br>• Early Stopping : patience 조정<br>• Hyperparameter Tuning : epoch, learning_rate, batch_size, load_best_model 등<br>2. 불포 불균형 해결<br>• binning 모델링<br>• 특정 라벨 증강 시도 및 no_relation 라벨 undersamping<br>• source별 불균형 해소 시도<br>• Loss Function 변경 (Focal Loss) |
| **검증 전략** | • 9:1, 8:2, 95:5 비율과 random, stratify의 방식으로 valid set 생성해서 평가 <br>• 최종적으로 리더보드에 제출하여 모델 성능 검증<br>• Valid set에 대한 predict 값과 정답값을 비교하는 difference.csv 파일 및 히트맵을 생성하여 정성평가 |
| **앙상블 방법** | • 데이터 전처리와 모델링 기법, 증강 데이터 적용 후 학습한 모델 중 가장 성능이 좋은 모델 10개를 선정하여 soft voting 앙상블을 진행<br>• 성능이 좋은 모델들 중 최대한 다양한 b종류의 모델과 여러 데이터셋이 섞이도록 Soft Voting, Weighted Voting 진행<br>• 성능 개선 : micro f1 75.1084(단일모델 최고) →76.4576 (앙상블) |

## 📂 폴더 구조

```bash
📦level2-nlp-mrc-nlp-04
├── README.md
├── code
│   ├── README.md
│   ├── arguments.py
│   ├── custom_model.py
│   ├── inference.py
│   ├── inference_bm25.py
│   ├── inference_es.py
│   ├── korquad_finetuning.ipynb
│   ├── requirements.txt
│   ├── retrieval.py
│   ├── retrieval_bm25.py
│   ├── retrieval_es.py
│   ├── train.py
│   ├── train_cnn.py
│   ├── trainer_qa.py
│   ├── utils
│   │   ├── datetime_helper.py
│   │   ├── file_name_utils.py
│   │   ├── hyper_parameters.py
│   │   ├── logging_utils.py
│   │   └── question 전처리.ipynb
│   └── utils_qa.py
├── data
├── eval.sh
├── inference.sh
├── run.sh
└── train.sh
```

## 🛠️ 사용법

1. `dataset`의 하위 디렉토리인 `test`에 `test.csv`, `train`에  `train.csv`파일을 준비한다.
2. `code` 디렉토리로 이동하고 `split_valid_*.py` 파일을 실행하여 validation 데이터를 생성한다.
3. `config` 디렉토리의 `default_config.yaml`을 복사하여 `config.yaml` 파일을 생성하고 원하는 Hyperpatameter를 설정한다.
4. 상위 디렉토리인 `level2-klue-nlp-04`으로 이동하여 `make run`을 입력하면 학습과 함께 추론이 완료된다.
    1. ‘config’로 시작하는 파일을 만들고 `make all`을 입력하면 여러 config 파일을 시작한다.
    - 예시
        
        ```bash
         ┣ config
         ┃ ┣ config.yaml
         ┃ ┣ config2.yaml
         ┃ ┣ config3.yaml
         ┃ ┣ config4.yaml
         ┃ ┗ default_config.yaml
        ```
        

## 📜 발표 자료

[발표 자료.pdf](https://github.com/boostcampaitech6/level2-klue-nlp-04/files/14020715/default.pdf)

## 📝 Wrap-Up Report

[KLUE_NLP_팀 리포트(04조).pdf](https://github.com/boostcampaitech6/level2-klue-nlp-04/files/14020703/KLUE_NLP_.ED.8C.80_.EB.A6.AC.ED.8F.AC.ED.8A.B8.04.EC.A1.B0.pdf)
