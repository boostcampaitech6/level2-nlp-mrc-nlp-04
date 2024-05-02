# **📝 Project 소개**

| 항목 | 내용 |
| --- | --- |
| 프로젝트 주제 | 기계 독해 MRC (Machine Reading Comprehension) 중 ‘Open-Domain Question Answering’ 를 주제로, 주어진 질의와 관련된 문서를 탐색하고, 해당 문서에서 적절한 답변을 찾거나 생성하는 task를 수행 |
| 프로젝트 구현 내용 | • etrieval 단계와 reader 단계의 two-stage 구조 사용 <br>• 평가 지표로는 EM Score(Exact Match Score)이 사용됨, 모델이 예측한 text와 정답 text가 글자 단위로 완전히 똑 같은 경우에만 점수가 부여 |
| 진행 기간 |  2024년 2월 7일 ~ 2024년 2월 22일 |

### ⚙️ 개발 환경 및 협업 환경

![image](https://github.com/boostcampaitech6/level2-klue-nlp-04/assets/63552400/faf13da8-8251-4efb-873c-feb8905c2308)


> **Notion, Slack, Zoom** 을 통해 회의를 진행했으며, **Github**을 통해 코드 공유를 및 Issues 기능을 이용한 관리를 진행. **Wandb**를 이용해 실시간으로 실험을 관리했습니다. 
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
| **구희찬**  | 베이스라인 코드 리팩토링, Retriever 모델 개선(DPR), Reader 모델 개선(PLM 선정 및 하이퍼파라미터 튜닝), 깃허브 관리 |
| **김민석**  | 베이스라인 코드 기능 추가(slack 연동), Retriever 모델 개선(DPR, SPR), 데이터 전처리(Question row 형태소 추가) |
| **손유림**  | EDA, Retriever 모델 개선(SPR), Reader 모델 개선(PLM 선정 및 하이퍼파라미터 튜닝), 데이터 후처리, 앙상블 코드 작성(Hard Voting) |
| **오태연**  | Retriever 모델 개선(SPR), Reader 모델 개선(Dataset Fine-Tuning, Curriculum Learning, CNN Layer 추가) |
| **이상수**  | Retriever 모델 개선(DPR), Reader 모델 개선(Dataset Fine-Tuning, Curriculum Learning) |
| **최예진**  | 베이스라인 코드 기능 추가(wandb 연동), Retriever 모델 개선(하이퍼파라미터 튜닝), 앙상블 코드 작성(Soft Voting) |

# 💾 데이터 소개
 json 형식의 데이터셋이 제공, rain 데이터셋은 질문의 고유 id, 질문, 답변 텍스트 및 답변의 시작 위치, 답변이 포함된 문서, 문서의 제목, 문서의 고유 id를 포함하고 있다. 이때 train_dataset 경로 내 파일은 3952개의 샘플을 포함하는 train 데이터와 240개의 샘플을 포함하는 validation 데이터셋으로 구성되어 있음.

### 데이터셋 통계


- 전체 데이터에 대한 통계는 다음과 같습니다.
    - `train.csv` : 총 3952개
    - `test_data.csv` : 총 600개 (이중 240개의 데이터셋을 대상으로 리더보드 점수 채점)
<img width="510" src=https://github.com/boostcampaitech6/level2-klue-nlp-04/assets/67735022/190e5528-e171-43ab-95d3-39fd88d2a4bd>

### **Data 예시**




![16bc4f53-f355-4b9d-968f-657bb5d9b5e5](https://github.com/boostcampaitech6/level2-klue-nlp-04/assets/67735022/9ce4af17-e6f8-42d3-95ed-eaa0b15221a9)

- column 1: 샘플 순서 id
- column 2: sentence.
- column 3: subject_entity
- column 4: object_entity
- column 5: label
- column 6: 샘플 출처


### 사용방법
1. aistages의 링크를 통해 데이터 폴더를 다운받는다.
2. `tar -zxvf data.tar.gz` 로 압축을 해제한다.
3. data 폴더를 복붙한다.
4. `sh run.sh` or `chmod +x *.sh`로 실행권한 설정 이후에 `./run.sh`
