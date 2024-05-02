# **📝 Project 소개**

| 항목 | 내용 |
| --- | --- |
| 프로젝트 주제 | 기계 독해 MRC (Machine Reading Comprehension) 중 ‘Open-Domain Question Answering’ 를 주제로, 주어진 질의와 관련된 문서를 탐색하고, 해당 문서에서 적절한 답변을 찾거나 생성하는 task를 수행 |
| 프로젝트 구현 내용 | • Retrieval 단계와 Reader 단계의 two-stage 구조 사용 <br>• 평가 지표로는 EM Score(Exact Match Score)이 사용되었고, 모델이 예측한 text와 정답 text가 글자 단위로 완전히 똑같은 경우에만 점수 부여 |
| 진행 기간 |  2024년 2월 7일 ~ 2024년 2월 22일 |

# **최종 리더보드**
<img width="1216" alt="image" src="https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-04/assets/69586041/e6de43e7-31c5-41ea-8664-c75bb8cd4285">
최종 리더보드 순위 2위

### ⚙️ 개발 환경 및 협업 환경

![image](https://github.com/boostcampaitech6/level2-klue-nlp-04/assets/63552400/faf13da8-8251-4efb-873c-feb8905c2308)


> **Notion, Slack, Zoom** 을 통해 회의를 진행했으며, **Github**을 통해 코드 공유를 및 Issues 기능을 이용한 관리를 진행. **Wandb**를 이용해 실시간으로 실험 관리. 
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
|구희찬| 베이스라인 코드 리팩토링, Retriever 모델 개선(DPR), Reader 모델 개선(PLM 선정 및 하이퍼파라미터 튜닝), 깃허브 관리 |
|김민석| 베이스라인 코드 기능 추가(slack 연동), Retriever 모델 개선(DPR, SPR), 데이터 전처리(Question row 일반명사 추가) |
|손유림| EDA, Retriever 모델 개선(SPR), Reader 모델 개선(PLM 선정 및 하이퍼파라미터 튜닝), 데이터 후처리, 앙상블 코드 작성(Hard Voting) |
|오태연| Retriever 모델 개선(SPR), Reader 모델 개선(Dataset Fine-Tuning, Curriculum Learning, CNN Layer 추가) |
|이상수| Retriever 모델 개선(DPR), Reader 모델 개선(Dataset Fine-Tuning, Curriculum Learning) |
|최예진&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 베이스라인 코드 기능 추가(wandb 연동), Retriever 모델 개선(하이퍼파라미터 튜닝), 앙상블 코드 작성(Soft Voting) |

# 💾 데이터 소개
 json 형식의 데이터셋이 제공되었고, train 데이터셋은 질문의 고유 id, 질문, 답변 텍스트 및 답변의 시작 위치, 답변이 포함된 문서, 문서의 제목, 문서의 고유 id를 포함하고 있다. 이때 train_dataset 경로 내 파일은 3952개의 샘플을 포함하는 train 데이터와 240개의 샘플을 포함하는 validation 데이터셋으로 구성되어 있다.

### 데이터셋 정보
![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-04/assets/69586041/af4ca733-e3de-4ecd-8f02-a07c7470924d)
좌측은 train dataset, 우측은 validation dateset 시각화의 결과이고, 각각 context length, question length, answer length를 의미.

- 데이터셋에 대한 정보
    - `train.csv` : 총 3952개
    - `vaildation.csv`: 240개
    - `test_data.csv` : 총 600개 (이중 240개의 데이터셋을 대상으로 리더보드 점수 채점)

- column 1: 고유 id
- column 2: 질문
- column 3: 답변 text
- column 4: 답변이 포함된 문서
- column 5: 문서의 제목
- column 6: 문서 고유 id 


## 💡 Methods

| 분류 | 내용 |
| --- | --- |
| **Retriver 모델 개선** |• Sparse Passage Retrieval : 단어사전(BoW)을 구축하여 TF-IDF 기반으로 중요도를 계산하여 평가지표 EM의 성능을 높이기 위해 정확한 단어를 비교하기 위한 방법. TF-IDF 계열 중 SOTA를 달성한 것으로 알려진 BM25 알고리즘을 적용하여 성능 개선을 시도함. 모든 상황에서 유의미한 성능 개선을 보임. <br><br> • Dense Passage Retrieval : Sparse Embedding 은 중요한 단어가 정확히 일치하는 상황에서 유리하나, Dense Embedding은 단어의 유사성 혹은 맥락 파악에 유리하고 학습으로 임베딩을 만들어서 추가적인 학습이 가능하다는 특징을 가지고 있음. 따라서  postive passage의 유사도를 높이면서, negative passage의 유사도를 낮추는 방식으로 학습을 시도 했으나, 실제로는 결과가 좋지 못했음. 원인에 대해서는 wrap-up report 참조 |
| **Reader 모델 개선 - PLM모델** | • 실험한 모델 : 최종적으로 `klue/roberta-large` 사용<br>`klue/roberta-large`, `klue/roberta-base`, `monologg/koelectra-base-v3-finetuned-koquad`,`RoBertForQuestionAnswering` |
| **Reader 모델 개선** | • Korquad dataset 추가 : Reader 모델 성능 개선을 위해 데이터 증강이 필요하다소 판단. 그중 대회 데이터셋과 KorQuAD 1.0 데이터셋이 context 길이나 데이터 출처가 유사하다고 파악하여 KorQuAD 1.0 데이터셋을 활용해 데이터 추가하여 학습을 진행.  기본 베이스라인 Roberta-large 기준 EM 56.6700 → 59.5800으로 성능 상승을 확인 <br> <br> • Korquad Fine-Tuning : 위의 KorQuAD 1.0 데이터셋을 추가하여 학습한 경우, 방법론 조합에 따라 성능 떨어지는 경우가 발생. 이를 해결하기 위해 train 데이터보다 지문 길이가 긴 데이터를 제외하고 roberta-large를 KorQuAD 1.0 로 fine-tuning 진행. 결과적으로 제대로된 단어를 추출하지 못했기 때문에 최종 제출에는 사용하지 않음. <br> <br> • CNN Layer 추가  : 기존 RobertaForQuestionAnswering의 경우 마지막에 Linear layer만을 통과하는데, 이 앞에 CNN layer 를 추가해 근접 백터간의 연간 정보까지 학습되도록 함. 결과적으로 약 3점의 성능개선을 보임. | 
| **데이터 전처리** | • Entity Marker : 명사들을 쉼표로 구분해서 question 앞에 추가<br><br>• 특수기호 제거 : 일부 특수 기호(<>, 책 제목 기호, 불필요한 말따옴표 등) 제거 |
| **앙상블 방법** | • 모델의 다양성과 성능을 고려하여, Hard Voting과 Soft Voting의 두 가지 방식으로 앙상블을 진행. 실험 결과, 전반적으로 Hard Voting 방식이 더 높은 성능을 유도하였고, 이에 따라 최종 제출 결과에도 Hard Voting 방식이 적용 <br><br>• Hard Voting : 앙상블한 파일에서 단순히 가장 자주 등장하는 단어를 선택하는 방식. 이때, 최빈도 단어가 2개 이상일 시 랜덤으로 결과를 출력 <br><br> • Soft Voting: nbest_predictions.json에서 제공하는 단어별 확률값을 활용해서, 각 파일에서 단어의 확률값을 평균낸 후 가장 높은 값을 선택하는 방식 |

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


## 🛠️ 사용방법
1. aistages의 링크를 통해 데이터 폴더를 다운받는다.
2. `tar -zxvf data.tar.gz` 로 압축을 해제한다.
3. data 폴더를 복사 + 붙여넣기 한다.
4. `sh run.sh` or `chmod +x *.sh`로 실행권한 설정 이후에 `./run.sh`


## 📜 발표 자료
[MRC대회발표_PPT.pdf](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-04/files/15184120/MRC._PPT.pdf)

## 📝 Wrap-Up Report
[MRC_NLP_팀 리포트(04조).docx](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-04/files/15184125/MRC_NLP_.04.1.docx)
