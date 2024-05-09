# House Price Prediction Competition


## 📝 프로젝트 개요

  
- 서울시 아파트 실거래가 데이터를 활용하여 정확하고 일반화된 모델을 개발하여 아파트 시장의 동향을 미리 예측하는 팀 프로젝트입니다.
- 부동산 시장은 다양한 요인에 의해 가격이 변동되며, 이를 예측함으로써 구매자와 판매자 모두에게 유용한 정보를 제공할 수 있습니다. 이번 프로젝트는 서울시의 아파트 실거래가 데이터를 기반으로 예측 모델을 개발하여 부동산 시장의 동향을 파악하고자 합니다.

- **팀원**: 5명
- **역할**:
  - **EDA & 데이터 전처리**: 아파트 실거래가에 대한 이상치 탐색 수행, 도로명 주소를 기반으로 외부 데이터를 통한 결측치 대체
  - **Feature Engineering**: '복도유형' 카테고리 변수, '일정 거리 내의 정류장 개수', '역세권 여부' 변수 생성
  - **모델링**: LightGBM 모델을 이용한 베이스라인 코드 작성, 고가 아파트 모델 성능 고도화
- **팀 프로젝트 기간**: 2024년 1월 15일 ~ 2024년 1월 25일  
- **Repository**: [House Price Prediction](https://github.com/UpstageAILab/upstage-ml-regression-07)


## 📊 Data Description

### Dataset Overview

- **출처**: Upstage 측에서 제공  
- **기간**: 2007년 1월 1일 ~ 2023년 9월 30일  
- **데이터 수**: 1,118,822  
- **특성 수**: 102개의 특성 (50개 이상의 추가 피처 생성)  
- **변수 설명**: 부동산의 위치, 크기, 건축 연도, 주변 시설 및 교통 편의성 등의 정보를 포함함  


##  🛠️ Stacks 


  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=NumPy&logoColor=white"> <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"> <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
 <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white"> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"> 


## 🔍 프로젝트 진행 단계 

1. **데이터 분석**  
    - EDA를 진행하면서 데이터에 대한 이해 ('왜?'라는 부분에 집중하여 결측치와 이상치를 파악)  
2. **모델 선정 및 베이스라인 구축**  
    - LightGBM을 활용하여 베이스라인 모델을 선정하고 기본적인 코드 작성  
3. **피처 엔지니어링**  
    - 각자 다양한 피처들을 나눠서 생성하고 파생변수를 추가  
4. **피처 선택 및 과적합 방지**  
    - Feature Selection을 통해 모델의 과적합을 방지하고 성능 개선  
5. **하이퍼파라미터 튜닝**  
    - Weights & Biases Sweep을 이용하여 Hyper Parameter Tuning 수행  
6. **모델링 및 예측**  
    - 실거래가에 따라 높은 실거래가와 낮은 실거래가를 예측하는 2개의 모델로 각각 예측 수행  

<img src="https://github.com/ANGHOOO/Upstage-AI-Stages/assets/103275370/a2e7432c-5bcc-496a-8821-076efe0f8d3f">

## 🪜 프로젝트 세부 과정

### 1. 데이터 전처리

#### a. 결측치 처리 및 데이터 클리닝
- 서울시 건축물 대장 데이터와 외부 데이터를 활용하여 데이터셋의 결측치를 처리
- 사용 라이브러리: `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`

#### b. 피처 엔지니어링
- EDA를 바탕으로 다음과 같은 파생변수 생성:
  - '꼭대기층 여부', '이전가격', '전용면적', '아파트 평균높이', '연GDP', '층', '계약년월', 'y', '한강거리', '500m이내 정류장 수', '건물나이'
- 사용 라이브러리: `pandas`, `numpy`, `LightGBM`, `WandB`

### 2. 모델 구성

#### a. LightGBM 모델
- 저가 아파트와 고가 아파트 예측 모델을 분리하여 구성
  - 저가 아파트 모델: 실거래가가 30억 미만인 데이터 사용
  - 고가 아파트 모델: 실거래가가 30억 이상인 데이터 사용
- Optuna 라이브러리를 이용해 최적의 하이퍼파라미터 탐색
  - `learning_rate`: 0.05
  - `num_leaves`: 31
  - `max_depth`: -1

## 🏆 Results

| 내용 | 저가 아파트 | 고가 아파트 |
|------|------------|------------|
| LightGBM 모델의 최종 Train RMSE | 4958.61 | 5493.75 |
| LightGBM 모델의 최종 Valid RMSE | 9761.01 | 26855.08 |
| LightGBM 모델의 최종 Test RMSE | 10015.53 | 10015.53 |
| 최종 Public RMSE | 104262.69 | 104262.69 |
| 최종 Private RMSE | 86148.23 | 86148.23 |

- 모델을 통해 아파트 가격 예측에 도움을 주었지만, 정확도 향상에는 한계가 있었음  
  - 저가 아파트 모델이 고가 아파트 모델에 비해 예측 정확도가 높음  
  - 고가 아파트 모델은 고가 아파트 가격의 변동성 때문에 예측에 어려움을 겪음  
  - 평균적인 가격 예측보다는 극단적인 가격 예측에서 성능 저하가 나타남  

#### Private LB
<img width="1094" alt="Private Leaderboard" src="https://github.com/UpstageAILab/upstage-ml-regression-07/assets/46295610/8c5a662b-bc08-40e2-8204-2081150386da">

### Presentation
[AI Lab House Price Prediction](https://drive.google.com/file/d/1rLKcLZ6QxLMrfo_h3RMY4oZuMC4I0aqG/view?usp=sharing)

## 📜 Conclusion
- 도메인 지식과 EDA를 바탕으로 파생변수를 생성하고, 이를 모델에 활용함으로써 예측 성능 향상에 기여 
- LightGBM 모델과 하이퍼파라미터 튜닝을 통해 제한된 리소스 환경에서도 우수한 성능의 모델 구축 
- 저가/고가 아파트 분리 모델링, 리더보드 패턴 분석 등 다양한 접근 방식의 실험을 통해 문제 해결의 실마리 포착 


## 🌟 잘했던 점

1. 모델의 성능을 올리기에 집중하지 않고, 팀원들과 일주일간 EDA를 진행하여 데이터를 이해하고 공유  
2. 변수들을 분담하여 파생변수를 생성함으로써 모든 변수에 대해 의미 있을 것 같은 파생변수를 생성  

## 😥 시도했으나 잘 되지 않았던 것들

1. 개인적으로 의미 있다고 생각되는 파생변수를 생성해 실험해 보았지만, 성능 개선으로 이어지지 않는 경우가 많았음  
2. 최대한 많은 데이터를 사용해서 모델을 학습하려 했지만, 시계열적 특성을 잘 반영하지 못하여 최근 3년의 데이터로 학습한 모델보다 성능이 좋지 않았음  

## 😢 한계점 및 아쉬웠던 점들

1. 실거래가가 높은 아파트와 낮은 아파트의 기준을 단순히 target 값의 분포를 기준으로 30억으로 정했음  
    - 해당 기준을 좀 더 명확하게 정할 필요가 있었음  

2. 항상 '왜?'라는 부분을 염두에 두고 진행하려 했지만, 점수에 집중하는 경우가 많았음  

3. 대회 기간이 2주로 짧았기 때문에 더 많은 실험을 해보지 못했음  

4. 모델을 검증하는 과정에서 시계열 데이터에 사용되는 TimeSeries Split 방법을 사용해보지 않았음  

5. 프로젝트를 구조화했지만, 실험 파이프라인을 체계적으로 작성하지 않아, 대회가 끝난 후 코드를 살펴보는데 어려움이 있었음  

6. 깃헙 커밋 컨벤션을 따로 지정하지 않아, 팀원들끼리 같은 컨벤션을 사용하지 못함  

## 💡 경진대회를 통해 배운 점 또는 시사점

1. ML 프로젝트의 전반적인 파이프라인을 경험해 볼 수 있었음  

2. 실험 도중 기존 데이터와 모델의 성능에만 사로잡혀 있었지만, 팀원들은 외부 데이터를 어떻게 활용할지,  
   우리의 모델이 어떤 점을 잘 예측하지 못하는지에 대해 생각했음  
   - 팀원들을 통해 많은 것을 배우고, 생각하는 관점도 넓어졌음  

3. Weights & Biases의 사용법과 Sweep을 통해 하이퍼파라미터 튜닝하는 방법을 배웠음  

4. Feature Selection의 여러 방법론에 대한 라이브러리 사용 방법을 배웠음  
