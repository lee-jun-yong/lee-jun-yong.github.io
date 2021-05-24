---
layout: single
title: "캐글스터디 1회차 : kaggle 평가지표"
---

```python
!pip install IPython
from IPython.display import Image

#주피터노트북 사진 호출법 (해당 작성 폴더에 사진 파일이 있어야함)
#1. 결과값 Image('back.jpg')
#2. MARKDOWN값 마크다운 창으로 변환후 ![title](1.jpg)
```


# 캐글대회 평가지표

평가지표는 학습시킨 모델의 성능이나 그 예측 결과의 좋고 나쁨을 측정하는 지표이다.
실전에서는 경진 대회 참가자가 구현한 모델에 따른 예측 결과를 제출하면, 각 대회마다 정해진 평가지표로 점수(score)가 산출되며 이를 기준으로 순위가 정해진다.

###### 목차

1. 회귀의 평가지표
2. 이진분류의 평가지표
3. 다중클래스의 평가지표
4. 추천 문제의 평가지표

### 1. 회귀의 평가지표

#### RMSE(제곱근평균제곱오차)

![1](https://user-images.githubusercontent.com/84025932/119320073-2540b700-bcb6-11eb-99ba-3a8f74e01b1b.jpg)

●RMSE의 값을 최소화했을 때의 결과가, 오차가 정규분포를 따른다는 전제하에 구할 수 있는 최대가능도방법과같아지는 등 통계학적으로도 큰 의미를 가지고 있는 평가지표이다.

●하나의 대푯값으로 예측을 실시한다고 가정했을 때 평가지표 RMSE를 최소화하는 예측값이바로 평균값이다.

●이상치를 제외한 처리 등을 미리 해두지않으면 이상치에 과적합한 모델을 만들 가능성이 있다.

●scikit-learn에서 metrics 모듈의 mean_squared_error를 이용

#### RMSLE(제곱근평균제곱로그오차)

![2](https://user-images.githubusercontent.com/84025932/119320143-38538700-bcb6-11eb-9c6f-5abd76ae094a.jpg)

● 예측할 대상인 목적변수를 정한다. 이 변수의 값에 로그를 취한 값을 새로운 목적변수로 삼는다. 그 후 이에 대한 RMSE를 최소화하면 RMSLE가 최소화된다.

● 목적변수의 분포가 한쪽으로 치우치면 큰 값의 영향력이 일반적인 RMSE보다 강해지기 때문에 이를 방지하려고 사용한다.

● numpy의 log1p함수 이용

● 이 지표는 비율에 주목한다.

#### 결정계수

![3](https://user-images.githubusercontent.com/84025932/119320147-38ec1d80-bcb6-11eb-8416-4e11cdedf566.jpg)

● 결정계수의 최댓값은 1이므로 1에 가까워질수록 모델 성능이 높은 예측으로 볼 수 있다.

● 사이킷 런에서 metrics 모율의 r2_score 함수로 계산할 수 있다.

● scikit-learn에서 metrics 모듈의 r2_score 함수를 이용

#### MAE(평균절대오차)

![4](https://user-images.githubusercontent.com/84025932/119320148-3984b400-bcb6-11eb-9640-d6e698dd0ca6.jpg)

●MAE는 이상치의 영향을 상대적으로 줄여주는 평가에 적절한 함수이다.

●하나의 대표값으로예측할 때 MAE를 최소화하는 예측값은중앙값이다.

●scikit-learn에서 metrics 모듈의 mean_absolute_error함수 이용

### 2. 이진 분류의 평가지표

##### 혼동행렬

![5](https://user-images.githubusercontent.com/84025932/119320149-3984b400-bcb6-11eb-8ecc-ec25b9f6c35c.jpg)

혼동행렬(confusion matrix)은 모델의 성능을 평가할 때 주로 사용하는 지표이다. 평가지표는 아니지만 양성인지 음성인지를 예측값으로하는 평가지표에서 자주 활용된다.

●TP(True Positive, 참 양성): 예측이 정확하다(True), 예측값양성(Positive), 실제값 양성(Positive)

●TN(True Negative, 참 음성): 예측이 정확하다.(True), 예측값음성(Negative), 실제값음성(Negative)

●FP(False Positive,거짓 양성): 예측이 틀렸다.(False), 예측값양성(Positive), 실제값음성(Negative)

●FN(False Negative, 거짓 음성): 예측이 틀렸다.(False), 예측값음성(Negative),실젯값양성(Positive)

### 이진분류 평가지표 (혼동행렬 사용지표)

#### 정밀도와 재현율

![6](https://user-images.githubusercontent.com/84025932/119320151-3a1d4a80-bcb6-11eb-9380-ad03c32fac57.jpg)

![7](https://user-images.githubusercontent.com/84025932/119320152-3a1d4a80-bcb6-11eb-837c-0a66a2e43f15.jpg)

●정밀도(precision)는 양성으로 예측한 값 중에 실젯값도양성일 비율, 재현율(recall)은 실제값이양성인 것 중에 예측값이양성일 비율이다.

●각각의 값의 범위는 0부터 1사이이며 1에 가까워질수록 좋은 점수이다.

●정밀도와 재현율은 어느 한 쪽의 값을 높이려 할 때 다른 쪽의 값은 낮아지는 트레이드 오프관계이다. 따라서 둘 중 하나만을 경진 대회의 지표로 삼는 일은 없다.

●잘못된 예측(오답)을 줄이고 싶다면 정밀도를 중시하고, 실제 양성인 데이터를 양성으로 올바르게 예측하고 싶다면 재현율을 중시하면 된

●사이킷런 metrics 모듈의 precision_score함수와 recall_score함수를 이용

#### F1-score와 FB-score

![8](https://user-images.githubusercontent.com/84025932/119320154-3ab5e100-bcb6-11eb-8806-d8a8302af671.jpg)

![9](https://user-images.githubusercontent.com/84025932/119320155-3ab5e100-bcb6-11eb-9239-c6a4b0e38298.jpg)

●F1-score는 앞서 설명한 정밀도와 재현율의 조화 평균으로 계산되는 지표이다.

●정밀도와 재현율의 균형을 이루는 지표로 실무에서도 자주 쓰이며 F점수(F score)라고도 한다.

●FB-score는 F1-score에서 구한 정밀도와 재현율의 균형에서 계수 B(베타)에 따라 재현율에 가중치를 주어 조정한 지표이다. 

●scikit-learn에서 metrics 모듈의 f1_score 함수와 
Fbeta_score함수를 이용

#### MCC(매튜상관계수)

![10](https://user-images.githubusercontent.com/84025932/119320157-3b4e7780-bcb6-11eb-8d45-00cc14c69389.jpg)

●이 지표는 -1부터 +1 사이 범위의 값을 가진다. +1일 때는 완벽한 예측, 0일때는 랜덤한예측, -1일때는완전 반대 예측을 한것이다.

●scikit-learn에서 metrics 모듈의 mattews_corrcoef함수 이용

### 이진분류 평가지표(혼동행렬 미사용) 

#### Log loss(로그손실)

![11](https://user-images.githubusercontent.com/84025932/119320160-3b4e7780-bcb6-11eb-9a79-a6fc517ef17b.jpg

●이 식에서 yi는 양성인지 아닌지를 표시하는 레이블(양성:1, 음성:0)이고, pi는 각 행 데이터가 양성일 예측 확률을 나타낸다.

●pi는 실젯값을예측하는 확률로, 실젯값이 양성일 경우는  pi이고 음성일 경우는 1- p이다.

●scikit-learn에서 metrics 모듈의 log_loss 함수 이용

#### AUC(ROC 곡선아래 면적)

![ROC](https://user-images.githubusercontent.com/84025932/119320167-3c7fa480-bcb6-11eb-8c8d-3a867fd9382c.JPG)

###### -거짓 양성 비율(FPR):실제 거짓인 행 데이터를 양성으로 잘못 예측한 비율(혼동행렬 요소로 FP/(FP+TN))
###### -참 양성 비율(T PR):실제 참인 행 데이터를 양성으로 올바르게 예측한 비율(혼동행렬 요소로 TP/(TP+FN))

출처: https://medium.com/@unfinishedgod/r-%EC%8B%A0%EC%9A%A9%EB%B6%84%EC%84%9D-%EC%98%88%EC%B8%A1%EC%A0%81-%EB%B6%84%EC%84%9D-roc-curve%EC%9D%84-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-db6e119c5b49

●ROC곡선은 예측값을양성으로 판단하는 임계값을 1에서 0으로 움직일 때의 거짓 양성 비율(False positive rate)(FPR)과 참 양성 비율(True positive rate)(TPR)을 그래프의 (x,y)축으로 나타낼 수 있다

●모든 행 데이터를 정확하게 예측했을 경우 AUC는 1이다.

●예측값이반대일 경우(1.0-본래 예측값일경우) AUC는 1.0-원래의 AUC가 된다.

●AUC는 양성과 음성을 각각 랜덤 선택했을 때 양성 예측값이음성 예측값보다클 확률이라고 정의할 수 있다.

●AUC의 값에 영향을 미치는 요소는 각 행 데이터 예측값의대소 관계뿐이다.

●양성이 매우 적은 불균형 데이터의 경우, 양성인 예측값을얼마나 높은 확률로 예측할 수 있을지가 AUC에 크게 영향을 미친다.

●지니 계수는 Gini=2AUC-1로 계산하며 AUC와 선형 관계이다. 따라서 평가지표가 지니 계수라면 평가지표가 AUC라고 할 수 있다.

●scikit-learn에서 metrics 모듈의 roc_auc_score 함수 이용

### 3. 다중클래스 지표

#### Multi-class accuracy

●이진 분류의 정확도를 다중 클래스로 확장한 것이다.

●예측이 올바른 비율을 나타내는 지표로, 예측이 정답인 행 데이터 수를 모든 행 데이터 수로 나눈 결과이다.

●scikit-learn에서 metrics 모듈의 accuracy_score 함수 이용

#### Multi-class logloss

![12](https://user-images.githubusercontent.com/84025932/119320163-3be70e00-bcb6-11eb-84be-114eddb95fe5.jpg)

●M은 클래스의 수 

●y(i,m)는 행데이터 i가 클래스 m에 속할 경우 1, 그렇지 않을 경우 0이된다.

●p(i,m)는 행 데이터 i가 클래스 m에 속하는 예측 확률

● Metrics 모듈의 log_loss 함수로 계산 할 수 있다. 단 이진 분류와는 log_loss 함수에 부여하는 배열의 형태가 다르다.

#### Mean-F1, Macro-F1, Micro-F1

●Mean-F1에서는 행 데이터 단위로 F1-score를 계산하고 그 평균값이 평가지표 점수가 된다.
ID가 1인 행 데이터는 (TP,TN,FP,FN)=(1,0,1,1)이 되고  F1-score는 0.5이다.

●macro-F1에서는 각 클래스별 F1-score를 계산하고 이들의 평균값을 평가지표 점수로 삼는다.

●micro-F1에서는 행데이터 x 클래스의 각 쌍에 대해 TP,TN,FP,FN중 어디에 해당하는지 카운트 합니다.

●클래스 1(1,2,3클래스 중)은 (T P,TN,FP,FN)=(2,2,0,1)이 되므로 여기서 F점수(F score)를 계산하면 0.8이다. 이것을 각 클래스에서 평균한 값이 점수가 된다.

#### QWK

![13](https://user-images.githubusercontent.com/84025932/119320165-3be70e00-bcb6-11eb-8db4-fec8e5de544b.jpg)

 ● O(i,j)는 실젯값의클래스가 I, 예측값의클래스가 j인 행 데이터 수로, 이것을 행렬의 형태로 나열하면 다중 클래스에서의 혼동행렬이 된다.

● E(I,J)는 실젯값의클래스와 예측값의클래스 분포가 서로 독립적인 관계일 때, 혼동행렬의 각 셀(i,j)에 속하는 행 데이터 수의 기대치이다. 실제      
값이 i인 비율 X 예측값이 j인 비율 X 데이터 전체의 행 데이터 수로 계산한다.

● W(I,J)는 실젯값과 예측값차의 제곱((i-j)의 제곱)이다. 

● 실제값과크게 동떨어진 클래스를 예측해버리면 이 값은 제곱으로 커지므로, 예측을 크게 빗나가버릴 경우 큰 패널티가부과된다. 

### 4. 추천문제 평가지표

#### MAP@K

![14](https://user-images.githubusercontent.com/84025932/119320166-3c7fa480-bcb6-11eb-899c-8c2de12a8624.jpg)

●mi는 행 데이터 i가 속한 클래스의 수를 나타낸다.

●Pi(K)는 행 데이터 i에 대해 K(1<=k<=K)번째까지의 예측값으로계산되는 정밀도이다. 다만 k번째 예측값이정답일 경우에만 값을 
취하고 그 외에는 0이 된다.

●K개의 예측값과실제 정답 수가 같아도, 정답인 예측값이순서에 맞지 않게 뒤로 밀리면 점수는 낮아진다.

●예측값의순서가 중요하고 완전한 예측을 실시 했을때는 1, 완전히 잘못된 예측을 실시했을 때는 0이 된다.


"데이터가 뛰어노는 AI놀이터, 캐글" 한빛미디어 발췌
오류가 있을시 dothe7847@nate.com 연락부탁드립니다.

---
comments:True
---
<script src="https://utteranc.es/client.js"
        repo="lee-jun-yong/blog-comments"
        issue-term="pathname"
        theme="github-light"
        crossorigin="anonymous"
        async>
</script>
