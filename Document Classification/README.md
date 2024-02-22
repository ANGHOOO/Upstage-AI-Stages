## Team

| ![이명진](https://github.com/UpstageAILab/upstage-ml-regression-07/assets/116725865/0fca01d5-ef06-429b-86b5-314c4a0844e9) | ![서재현](https://github.com/UpstageAILab/upstage-ml-regression-07/assets/116725865/df7f4ae6-afdb-4624-b130-88f68ce6ad1a) | ![신주용](https://github.com/UpstageAILab/upstage-ml-regression-07/assets/116725865/6d047df8-b648-4b4b-8180-db4d61d3a30d) | ![이영훈](https://github.com/UpstageAILab/upstage-ml-regression-07/assets/116725865/fed64993-4c58-4223-9e10-8a942e1a03c2) | ![이준형](https://github.com/UpstageAILab/upstage-ml-regression-07/assets/116725865/66b20948-1eec-4273-9eb4-9d393221cfb8) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이명진](https://github.com/myeongwang)             |            [서재현](https://github.com/SeoJaeHyeon)             |            [신주용](https://github.com/kimddong23)             |            [이영훈](https://github.com/ANGHOOO)             |            [이준형](https://github.com/Perelman-0)             |
|                            **팀장, EDA, Feature Engineering, Modeling**                             |                           **EDA, Feature Engineering, Modeling**                           |                         **EDA, Feature Engineering, Modeling**                           |                            **EDA, Feature Engineering, Modeling**                      |                            **EDA, Feature Engineering, Modeling**                             |



##  :clipboard: Stacks 
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=NumPy&logoColor=white"> <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"> 
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white"> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"> <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=Slack&logoColor=white"> <img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=Notion&logoColor=white"> 

## 1. Competiton Info
![스크린샷 2024-02-19 오후 8 48 57](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/2e73550d-6fb7-459d-aa39-fcebb848d454)

**우선 데이터 저작권에 의해 해당 대회에 사용된 이미지 데이터를 첨부할 수 없어서 본 README에서는 학습 데이터에 대한 사진이 없음을 알려드립니다.**

### Overview

- **Document Type Classification | 문서 타입 분류**
- computer vision domain에서 중요한 태스크인 이미지 분류 대회



![스크린샷 2024-02-19 오후 7 45 13](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/60af49f0-8e8b-45b3-ad96-10a0d45f955d)

이미지 분류란 주어진 이미지를 여러 클래스 중 하나로 분류하는 작업입니다. 
이러한 이미지 분류는 의료, 패션, 보안 등 여러 현업에서 기초적으로 활용되는 태스크입니다. 
딥러닝과 컴퓨터 비전 기술의 발전으로 인한 뛰어난 성능을 통해 현업에서 많은 가치를 창출하고 있습니다.

그 중, 이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 
문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 
이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

이번 대회에 사용될 데이터는 총 17개 종의 문서로 분류되어 있습니다. 
1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됩니다. 
특히, 현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최대한 줄였습니다. 또한 현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축하였습니다.


이번 대회를 통해서 문서 타입 데이터셋을 이용해 이미지 분류를 모델을 구축합니다. 
주어진 문서 이미지를 입력 받아 17개의 클래스 중 정답을 예측하게 됩니다. 
computer vision에서 중요한 backbone 모델들을 실제 활용해보고, 좋은 성능을 가지는 모델을 개발할 수 있습니다. 
그 밖에 학습했던 여러 테크닉들을 적용해 볼 수 있습니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

- input : 3140개의 이미지

- output : 주어진 이미지의 클래스


### Timeline

- ex) February 05, 2024 - Start Date
- ex) February 19, 2024 - Final submission deadline

### Evaluation

- 평가지표 : Macro F1

F1 score는 Precision과 Recall의 조화 평균을 의미합니다. 클래스마다 개수가 불균형할 때 모델의 성능을 더욱 정확하게 평가할 수 있습니다.
수식은 다음과 같습니다.

![스크린샷 2024-02-19 오후 8 23 52](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/b2a29c3b-86fe-49fe-b361-67b80ae859a2)

[참고자료] https://www.linkedin.com/pulse/understanding-confusion-matrix-tanvi-mittal/

Macro F1 score는 multi classification을 위한 평가 지표로 클래스 별로 계산된 F1 score를 단순 평균한 지표입니다. 


## 2. Data descrption

### Dataset overview

#### 학습 데이터셋 정보

주어진 학습 데이터에 대한 정보는 다음과 같습니다.

- train [폴더]: 1570장의 이미지가 저장되어 있습니다.

- train.csv [파일]: 1570개의 행으로 이루어져 있습니다. train/ 폴더에 존재하는 1570개의 이미지에 대한 정답 클래스를 제공합니다.

- ID: 학습 샘플의 파일명

- target: 학습 샘플의 정답 클래스 번호

- meta.csv [파일]:17개의 행으로 이루어져 있습니다.
- target 17개의 클래스 번호입니다.
- class_name 클래스 번호에 대응하는 클래스 이름입니다.

#### 평가 데이터셋 정보

평가 데이터에 대한 정보는 다음과 같습니다.

- test [폴더]: 3140장의 이미지가 저장되어 있습니다.

- sample_submission.csv [파일]: 3140개의 행으로 이루어져 있습니다.

- ID 평가 샘플의 파일명이 저장되어 있습니다.

- target 예측 결과가 입력될 컬럼입니다. 값이 전부 0으로 저장되어 있습니다.

- 그 밖에 평가 데이터는 학습 데이터와 달리 랜덤하게 Rotation 및 Flip 등이 되었고 훼손된 이미지들이 존재합니다.


### EDA 
#### 정의된 문제 해결을 위해 테스트 데이터에 적용되어 있는 변형에는 어떤 것이 있는지를 중점적으로 파악

- 문서 자체의 변형에는 Rotate, Crop, Flip이 많은 부분을 차지


- 다양한 유형의 노이즈가 있었고, 다른 유형의 이미지가 Mix-up 되어 있는 경우도 존재


### 문제 해결방향

> Rotate, Crop, Flip을 기본적으로 진행하고 테스트 데이터와 유사한 노이즈 패턴을 찾아서 적용하자.

  > 위 데이터를 잘 학습할 수 있는 BackBone Model를 찾고, 여러 기법을 통해 모델의 성능을 높이자

## 4.Soulution 01 - Data Augmentation

<img width="1015" alt="스크린샷 2024-02-19 오후 8 33 11" src="https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/4bdab723-70ea-48c1-81f9-235226721bd0">

- 실제 초반에는 2.5만개 학습데이터를 구축하고 기본 모델을 돌린 결과, 0.6206 -> 0.8692로 증가
- 추가로 최종 5만개를 학습한 결과, 0.8692 -> 0.9340으로 증가해서 Valid data 지표와 리더보드상의 지표의 간격을 낮출 수 있었습니다.

<img width="833" alt="스크린샷 2024-02-19 오후 8 33 37" src="https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/168cc5ee-063d-4861-8f40-6883d70726a1">



## 5.Soulution 02 - BACKBONE MODEL

- 기존 Baseline으로 설정한 ResNet50 모델의 경우 리더보드 기준 f1 score가 0.86 근처에서 좀처럼 향상이 되지 않음.
- 여러 backbone model(EfficientNet, Resnet101, ViT Model) 테스트 결과, depth, width, resolution에 대해
  compound-scaling 방식을 통해 balance 하게 uniform한 값으로 scale-up 해주는 방식인 EfficientNet 사용이 학습 시간/ 성능 관점에서 높게 나옴
- 모델의 크기가 늘어남에 따라 입력 이미지의 사이즈도 늘려주는 방식을 통해 ResNet50 일 때 LB : 0.8583에서 EfficientNet_B4 LB : 0.9040으로 향상
  
    <img width="728" alt="스크린샷 2024-02-19 오후 8 36 52" src="https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/1c853c0e-e027-475a-8c44-d9f90e8a7073">

<img width="794" alt="스크린샷 2024-02-19 오후 8 35 31" src="https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/e2501d8a-86fc-4314-93a0-c6c44c1b2db0">

## 6. Soulution 03 - TECHNIQUES

- #### HYPER PARAMETER TUNING 
  
<img width="612" alt="스크린샷 2024-02-19 오후 8 38 41" src="https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/417eb1f7-9aa8-4cf5-a1ca-4c8366e95603">

- #### LEARNING RATE SCHEDULER

<img width="694" alt="스크린샷 2024-02-19 오후 8 39 02" src="https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/ac0ec76e-bc51-441e-bace-ace699b4313e">

- #### SELECTIVE MULTI-CLASS CLASSIFICATION

<img width="693" alt="스크린샷 2024-02-19 오후 8 39 49" src="https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/116725865/0af63908-9622-4236-a4e1-d20d44cd4833">

- #### K-FOLD ENSEMBLE
  
<img width="649" alt="스크린샷 2024-02-19 오후 8 40 15" src="https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/189add39-2fc6-4f4e-b94f-4b1945c538b2">

- #### MIXED PRECISION TRAINING

<img width="425" alt="스크린샷 2024-02-19 오후 8 40 48" src="https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/d214d5d1-b773-4331-9102-27b009496a87">

- #### TTA

<img width="1027" alt="스크린샷 2024-02-19 오후 8 41 13" src="https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/a066992f-78b7-4966-9e8e-55088d9f1dc7">

- #### VISON TRANSFORMER

## 7. Result

### Leader Board

#### Private LB

![스크린샷 2024-02-20 오후 8 15 16](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/46295610/4379fa03-a16f-4ca7-bbe8-1f4a2658c8a6)



### Presentation

[CV 2조 발표자료 pdf.pdf](https://github.com/UpstageAILab/upstage-cv-classification-cv2/files/14330590/CV.2.pdf.pdf)


## etc

### Meeting Log
- Notion
  
  https://www.notion.so/e7e7e43d3c8f47e7b0205de5c1d688ad?v=90279b42a3c04fc09048647a696dbf0d

- 제출 및 실험 기록
  
  https://docs.google.com/spreadsheets/d/1iyIL6euOwvQgpm0kUp4lYCq8qOL-CKlKiaClfn8ZcTw/edit#gid=0  

- 대회 마지막 주는 Zoom 실시간 미팅(10:00am ~ 19:00pm)   

### Reference
- Intriguing Properties of Vision Transformers

  https://arxiv.org/pdf/2105.10497.pdf

- Mixed-Precision Training of Deep Neural Networks

  https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/  
  
