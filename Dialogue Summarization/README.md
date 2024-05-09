# Dialogue Summarization

> 일상 대화를 주어진 요약문과 유사하게 요약하는 모델 개발 프로젝트
 

## 📝 프로젝트 개요

- 2명에서 최대 7명이 등장하는 일상 대화 데이터를 3개의 요약문과 유사하게 요약하는 모델을 개발한 팀 프로젝트입니다.

- 일상 대화 요약은 학교 생활, 직장, 치료, 쇼핑, 여가, 여행 등 광범위한 일상 생활 주제를 다루며, 최소 2턴에서 최대 60턴으로 구성된 대화를 요약합니다. 이는 다양한 분야에서 대화 자동화 및 효율적인 정보 전달을 가능하게 할 수 있습니다.

- **팀원**: 6명 
- **담당 역할**: 
    - **Data Augmentation**: AEDA, Back-Translation, 외부 데이터(AI Hub) 활용 등 데이터 증강 기법 적용
    - **모델링**: KoBART 및 다양한 사전학습 모델들을 활용한 Baseline 구축 및 하이퍼파라미터 튜닝  
- **팀 프로젝트 기간**: 2024년 3월 15일 ~ 2024년 3월 21일
- **팀 프로젝트 소속**: Upstage AI Lab 


## 🛠️ Stacks  
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/Hugging Face-yellow?style=for-the-badge&logo=Hugging Face&logoColor=white"> <img src="https://img.shields.io/badge/Weights & Biases-FFBE00?style=for-the-badge&logo=Weights & Biases&logoColor=white">

## 📊 Data Description

### Dataset Overview

- 학습 데이터셋:  
    - train.csv: 12,457개의 대화(Dialogue)와 요약문(Summary)으로 구성
- 평가 데이터셋:
    - test.csv: 499개의 대화(Dialogue)로 구성 


### EDA Insights

- 대화의 최소/최대 턴 수: 2 ~ 60턴
- 데이터 번역체 특성: 영어를 한글로 번역한 번역투의 부자연스러운 문장들로 이루어짐
- 최다 등장 화자 수: 7명



## 🔍 프로젝트 진행 단계 및 세부 과정

### 1. EDA 및 문제 정의  
- 대화 데이터의 특성 파악
    - 평균 대화 길이: 28.4턴, 평균 요약문 길이: 18.7어절
    - 데이터 번역체 특성: 영어를 한글로 번역한 총 12,457개의 대화-요약문 쌍 데이터
- 문제 정의
    - 영한 번역투 대화 데이터를 자연스러운 한국어 요약문으로 변환하는 과제
    - Rouge-1,2,L F1 Score를 통한 토큰 단위 유사도 향상에 초점

### 2. 데이터 전처리 및 증강
- 데이터 정제
    - 정규표현식을 통한 특수문자, 이모티콘 제거 및 문장 분절
    - Komoran, KKma, Mecab 형태소 분석기를 활용한 토큰화 및 불용어 제거  
- 데이터 증강   
    - AEDA 기법 적용: 학습 데이터 약 2만 5천개로 증강
    - Back-Translation (Papago API): 영한 증강 데이터 3천개 추가 생성
    - AI Hub 한국어 대화 요약 데이터 2만 8천개 활용 

### 3. 모델링 및 학습
- 사전학습 모델 비교 실험
    - SKT-KoBART, BART-r3f, KoBART 요약 모델 등 6종 실험
    - KoBART 계열 모델의 우수한 성능 확인 (F1 0.51) 
- 하이퍼파라미터 튜닝
    - Batch Size (16,32), Learning Rate (1e-4, 5e-5, 2e-5), Epoch (20,30,50) 
    - Weight Decay (0.01, 0.1), Warmup Ratio (0.05, 0.1) 등 그리드 서치 수행
- 최적 파라미터 설정으로 모델 재학습    

### 4. 모델 평가 및 분석
- Rouge 자동 평가 및 Human Evaluation 수행   
    - 모델별 Rouge Score 산출 및 우수 모델 선정
    - 무작위 샘플 30개에 대해 3명의 참가자가 5점 척도 평가 진행 (평균 4.2점)
- 오류 유형 분석 및 개선 방안 논의
    - 고유명사, 숫자 표현 오류에 대한 사후 처리 적용 (F1 +0.8) 
    - 대화 문맥 정보 반영을 위한 Pre-training 기법 적용 제안  

### 5. 결과 정리 및 제출
- 실험 결과 종합 및 분석 리포트 작성
    - 모델별 실험 세팅 및 하이퍼파라미터, 학습 결과 정리
    - 데이터 증강, 하이퍼파라미터 튜닝 등을 통한 성능 향상 과정 기록
- 최종 모델 학습 및 테스트 데이터 예측 결과 제출    



![image](https://github.com/ANGHOOO/Upstage-AI-Stages/assets/103275370/fd0cb685-33f9-4b53-9eaa-9ab902fbb590)


## 🏆 Results

| Model | Final Result | LB Rouge1 | LB Rouge2 | LB Rouge L | 
|:-----:|:------------:|:---------:|:---------:|:----------:|
| KoBART-base-v1 (Baseline) | 41.5032 | 0.5088 | 0.3141 | 0.4221 |
| BART-r3f | 15.7081 | 0.2066 | 0.1000 | 0.1647 |
| KoBART-summary-v1 | 40.4117 | 0.4996 | 0.3029 | 0.4099 |
| KoBART-summary-v2 | 41.4013 | 0.5101 | 0.3160 | 0.4160 |


- Baseline 모델인 KoBART-base-v1 모델 대비 KoBART-summary-v2 모델의 Rouge-1 F1 Score 0.0013 향상
- 영어 데이터셋 기반 BART 계열 모델은 성능이 현저히 낮음
- 최종 순위 8위 달성
<img width="961" alt="image" src="https://github.com/ANGHOOO/Upstage-AI-Stages/assets/103275370/4bda31c2-c9fa-41e5-8ab1-35e7e5c79174">

## 📜 Conclusion

- 한국어 특화 언어 모델인 KoBART 계열의 모델들이 높은 성능을 보임  
- 부자연스러운 번역체 한글 데이터의 특성상 문장 다듬기 등 추가적인 데이터 전처리의 필요성 대두
- 대화의 문맥 정보와 화자의 감정을 반영할 수 있는 모델 개발의 중요성 인지

## 🌟 잘했던 점

- 다양한 데이터 증강 기법들을 적용하여 제한된 학습 데이터 환경을 개선하고자 노력하였다. 
- 프로젝트의 목표와 평가지표를 명확히 정의하고 그에 맞는 문제 해결 방향을 설정하였다.
- 팀원 모두가 적극적으로 참여하고 아이디어를 공유하며 협업하였다.

## 😥 시도했으나 잘 되지 않았던 것들  

- mT5, T5, Pegasus 계열의 다국어 모델들을 시도해보았으나 config error 등으로 최종 결과를 얻지 못하였다.
- BART-r3f 모델을 사용하였으나 베이스라인 모델 대비 성능이 크게 떨어졌다. 

## 😢 한계점 및 아쉬웠던 점들

- 프로젝트 기간이 짧아 충분한 실험과 분석을 진행하기 어려웠다. 
- 대화 데이터의 문맥 정보와 화자의 감정을 모델에 반영하는 방법을 고민하였으나 실제 적용하지 못하였다.
- 학습 리소스의 한계로 인해 대용량 언어 모델들을 충분히 활용하지 못한 점이 아쉽다.

## 💡 경진대회를 통해 배운 점 또는 시사점

- 자연어처리 태스크의 전반적인 파이프라인(데이터 전처리, 모델링, 학습, 평가)을 경험할 수 있었다.  
- 다양한 데이터 증강 기법들을 배우고 실제로 적용해볼 수 있는 기회였다.
- 팀원들과의 협업을 통해 문제 해결 능력과 커뮤니케이션 스킬을 기를 수 있었다.

## 🔮 What's Next  

- 대화의 문맥 정보와 화자의 감정을 반영할 수 있는 모델 아키텍처 연구
- Multilingual 모델을 활용한 영어-한국어 간 번역 및 요약 생성 파이프라인 구축  
- 실제 산업 도메인의 대화 데이터를 활용한 특화 요약 모델 개발

## 🔗 Reference

- [KoBART](https://github.com/SKT-AI/KoBART)
- [Data Augmentation using Pre-trained Transformer Models](https://arxiv.org/pdf/2003.02245.pdf) 
