# Document Type Classification

> 금융, 의료, 보험 등 다양한 산업 분야의 문서 처리 자동화를 위한 문서 유형 이미지 분류 모델 개발
 

## 📝 프로젝트 개요

- 문서 이미지를 분류하는 팀 프로젝트로, 17개 종류의 문서 유형을 식별하는 모델을 개발했습니다.

- 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.
- **팀원**: 5명
- **담당 역할**:
    - **EDA & 데이터 전처리**: 문서의 다양한 변형과 노이즈 패턴을 분석하고, Albumentation, Augraphy 라이브러리를 이용한 Image Augmentation 적용
    - **모델링**: ResNet 및 EfficientNet 모델을 활용한 베이스라인 구축 및 하이퍼 파라미터 튜닝, TTA를 이용한 성능 고도화
- **팀 프로젝트 기간**: 2024년 2월 5일 ~ 2024년 2월 19일  
- **팀 프로젝트 소속**: Upstage AI Lab
- **팀 프로젝트 Repo**: [Document Type Classification](https://github.com/UpstageAILab/upstage-cv-classification-cv2)


##  🛠️ Stacks 
 <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=NumPy&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white"> <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white"> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"> 

## 📊 Data Description

### Dataset Overview

- 학습 데이터셋:
  - train 폴더: 1570장의 이미지 
  - train.csv: 1570개 이미지의 파일명과 정답 클래스
- 평가 데이터셋: 
  - test 폴더: 3140장의 이미지 
  - sample_submission.csv: 3140개 이미지 파일명
- meta.csv: 17개 클래스의 번호와 클래스명

※ 데이터 저작권 문제로 인해 예시 이미지는 포함하지 않았습니다.

### EDA Insights

- Train/Test 데이터 간 분포 차이 존재 
  - Test 데이터의 주요 변형: Rotate, Crop, Flip, 겹쳐짐, 빛번짐
  - 다양한 유형의 노이즈와 이미지 Mix-up 존재

## 🔍 프로젝트 진행 단계 및 세부 과정

### 1. EDA 및 문제 정의
- Train/Test 데이터 간 분포 차이 파악
  - Test 데이터의 주요 변형: Rotate, Crop, Flip
  - 다양한 유형의 노이즈와 이미지 Mix-up 존재
- 문제 정의
  - Train/Test 데이터의 분포 불일치로 인한 모델 성능 저하
  - 제한된 학습 데이터 환경에서의 모델 일반화 성능 확보

### 2. 데이터 전처리 및 증강
- Test 데이터와 유사한 변형 및 노이즈를 적용한 Data Augmentation 수행
  - Rotate, Crop, Flip 등 기본 변형 적용
  - 다양한 유형의 노이즈 패턴 탐색 및 적용
- Data Augmentation을 통한 학습 데이터 확장
  - 1570개 이미지 데이터 → 52608개 이미지 데이터
    
### 3. 모델링 및 학습
- Backbone Model 선정
  - ResNet50, EfficientNet 계열의 다양한 모델 실험
  - 데이터 특성 및 리소스 제약을 고려하여 EfficientNet-B4 선정
- 모델 학습 및 검증
  - K-fold Ensemble, Stratified K-fold 등 다양한 학습 전략 적용
  - 하이퍼파라미터 튜닝, Learning Rate Scheduling, TTA 등을 통한 모델 최적화

### 4. 모델 평가 및 분석
- 학습된 모델의 성능 평가
  - LB F1-score 기준 모델 성능 비교 및 분석
- 모델 예측 결과 분석
  - 잘 분류하지 못하는 클래스 파악 및 원인 분석
- 추가 개선 방향 도출  

### 5. 결과 정리 및 제출
- 최종 모델 선정 및 결과 정리
  - 각 실험 결과 종합 및 최종 모델 선정
- 리더보드 제출 및 순위 확인
  - 최종 모델의 예측 결과를 제출하고 순위 확인

<img width="962" alt="image" src="https://github.com/ANGHOOO/Upstage-AI-Stages/assets/103275370/f28af955-9d57-4edb-a71f-4b239705d393">

## 🏆 Results

| Solution | LB F1-score |
|:--------:|:-----------:|
| ResNet50 (Baseline) | 0.3806 |
| Data Augmentation (1570 -> 2.5만) | 0.6206 -> 0.8692 |
| EfficientNet-B4 | 0.9040 |
| Selective Multi-Class Classification | 0.9158 |
| K-fold Ensemble | 0.9293 |
| Data Augmentation (2.5만 -> 5만) | 0.8692 -> 0.9340 |
| Test Time Augmentation | 0.9379 |

- Albumentation, Augraphy 라이브러리를 이용한 Data Augmentation과 EfficientNetB4로 Backbone Model 변경, 성능 향상을 위한 Ensemble, TTA 적용을 통해 최종 Private LB F1-score 0.9400 달성
- 최종 순위 5위 (상위 10%) 달성
<img width="782" alt="image" src="https://github.com/ANGHOOO/Upstage-AI-Stages/assets/103275370/7e89531a-0bcb-4eeb-b8b4-5ecf0322f29d">

## 📜 Conclusion

- 철저한 EDA를 통한 데이터 특성 파악과 그에 기반한 Data Augmentation 전략이 성능 향상에 크게 기여
- Backbone Model 탐색을 통해 EfficientNet 계열 모델이 제한된 리소스 환경에서도 좋은 성능을 보임을 확인
- 다양한 기법 적용 실험과 검증을 통해 모델 성능 극대화 
- Divide & Conquer 전략을 통한 체계적인 문제 해결 과정의 중요성 인지

## 🌟 잘했던 점

- 프로젝트 구조화와 구글 스프레드시트를 통해 효율적으로 실험할 수 있도록 설계하였다.
- EDA, Data Preprocessing, Modeling, Inference의 Computer Vision Task의 파이프라인을 반복적으로 수행하며 가설을 입증하며 성능 향상을 이루었다.  
- 팀원들 중 누구 하나 불평불만 없이 서로의 의견을 존중해 주었고 모두 다 열심히 참여해서 좋은 분위기 속에서 대회를 마무리하였다.

## 😥 시도했으나 잘 되지 않았던 것들

- 비교적 최신 모델인 ViT 계열 모델들을 학습해 보았지만 성능이 잘 나오지 않았다.
- Inference 분석 단계에서 우리의 모델이 잘 맞추지 못하는 클래스에 대해 따로 학습시키는 방식인 Selective Multi-Class Classification을 수행하였지만 생각보다 큰 성능 향상을 이루지 못하였다.

## 😢 한계점 및 아쉬웠던 점들

- 상위권 팀들의 솔루션을 분석한 결과 기존에 잘 알려진 모델들이 아닌 최신 논문을 읽고 모델을 구현하여 성능 향상을 이룬 팀이 있었는데 기존 SOTA 모델들의 사용에 집중한 것 같아 아쉬움이 남는다. 
- 이전 Tabular Data 경진대회에 비해 학습 시간이 엄청 오래 걸려서 생각보다 다양한 실험을 못한 것이 아쉽고, 컴퓨팅 자원의 중요성을 느낄 수 있었다.
- 연휴 기간에 진행한 대회여서 시간이 많이 부족해서 모델 앙상블을 진행하지 못한 점이 매우 아쉽다.

## 💡 경진대회를 통해 배운 점 또는 시사점

- Computer Vision Project의 전반적인 파이프라인을 경험해 볼 수 있었다.  
- 다양한 Augmentation 기법을 배웠고 이를 적용해 보고 실험할 수 있었다.
- CNN Backbone 모델과 Transformer 계열 모델의 차이를 알 수 있었고 각 모델들의 특성에 대해 자세히 배울 수 있었다.

## 🔮 What's Next
- 실험 과정과 결과에 대한 문서화 강화 
- Vision Transformer 등 대용량 데이터에서 강점을 보이는 모델 아키텍처 적용 및 검증
- 실제 비즈니스 데이터 확보를 통한 모델의 산업 적용 가능성 타진

## 🔗 Reference
- [Intriguing Properties of Vision Transformers](https://arxiv.org/pdf/2105.10497.pdf)
- [Mixed-Precision Training of Deep Neural Networks](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/)

## 📁 Project Management

- [제출 기록 스프레드시트](https://docs.google.com/spreadsheets/d/1iyIL6euOwvQgpm0kUp4lYCq8qOL-CKlKiaClfn8ZcTw/edit#gid=0)
