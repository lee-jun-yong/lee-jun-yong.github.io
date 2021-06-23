---
layout: single
title: "캐글스터디 4회차 : 경진대회 주요 모델 평가 "
---

# Chapter 5 모델평가

## 모델평가란?

+ 예측 모델을 구축하는 주된 목적
+	모델의 일반화 성능: 미지의 데이터에 대한 예측 능력
+	모델의 일반화 성능을 개선하려면 당연히 그 모델의 일반화 성능을 알 수 있는 방법 필요
+	보통 학습 데이터를 학습 데이터와 검증 데이터로 분할 후, 검증 데이터 예측 성능을 평가지표에 기반한 점수로 나타내 평가
+	검증 데이터는 몇 가지 방법으로 나눌 수 있음
+	이때 적절한 평가를 진행하려면 학습 데이터와 테스트 데이터의 성질을 고려한 방법으로 나눠야 함
+	검증: 데이터를 적절히 나눠 모델의 일반화 성능을 평가하는 작업 자체

## 검증방법

### 홀드아웃 검증

![15](https://user-images.githubusercontent.com/84025932/123093419-a2527e00-d466-11eb-880d-4e0fc1cdb413.jpg)


+	학습용 데이터로 모델을 학습시키고, 따로 남겨둔 검증용 데이터로 모델을 평가하는 방법
+	사이킷런 model_selection 모듈의 train_test_split 함수와 kfold클래스를 이용해 홀드아웃 검증으로 데이터 분할 가능
+	홀드아웃 검증은 교차 검증과 비교해 데이터를 효율적으로 사용하지 못하는 단점이 있음
+	검증 데이터가 적으면 평가를 신뢰할 수 없지만, 검증 데이터가 늘어나면 학습용 데이터가 줄어들어 원래 모델의 예측 성능이 떨어짐
+	테스터 데이터 예측 시에는 학습 데이터 전체에서 모델을 다시 구축할 수 있지만,  학습할 때의 모델과 최종 모델의 데이터 수가 다르면 최적의 하이퍼파라미터나 특징이 달라질 수도 있으므로, 검증에서도 학습 데이터는 어느 정도 확보하는 편이 바람직함
# 홀드아웃(hold-out)방법으로 검증을 수행

#### 홀드아웃 코드 구현

+ (1)train_test_split 방식


```python
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Model 클래스를 정의
# Model 클래스는 fit으로 학습하고 predict로 예측값 확률을 출력

# train_test_split 함수를 이용하여 홀드아웃 방법으로 분할
tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y,
                                          test_size=0.25, random_state=71, shuffle=True)

# 학습 실행, 검증 데이터 예측값 출력, 점수 계산
model = Model()
model.fit(tr_x, tr_y, va_x, va_y)
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(score)
```

+ (2)kfold 방식


```python
# KFold 클래스를 이용하여 홀드아웃 방법으로 검증 데이터를 분할

from sklearn.model_selection import KFold

# KFold 클래스를 이용하여 홀드아웃 방법으로 분할
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
print(tr_idx, va_idx)

tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
```


### 교차검증

![16](https://user-images.githubusercontent.com/84025932/123093421-a383ab00-d466-11eb-9de2-b90391b35064.jpg)


사진출처 : <https://velog.io/@lsmmay322/Model-Selection>

+ 학습 데이터를 분할하고 홀드아웃 검증 절차를 여러 번 반복함으로써 매회 검증 학습에 이용할 데이터의 양을 유지하면서도 검증 평가에 필요한 데이터를 학습 데이터 전체로 가능
+	분할된 데이터를 폴드라고 하고 분할된 수를 폴드 수라고 함
+	교차 검증의 폴드 수는 n_splits 인수로 지정함
+	폴드 수를 늘릴 수록 학습 데이터의 양을 더 확보할 수 있으므로 데이터 전체로도 학습시켰을 때와 유사한 모델 성능으로 평가 가능
+	연산 시간이 늘어나므로 트레이드 오프가 됨

#### 교차검증 코드 구현

```python
# 교차 검증
# -----------------------------------
# 교차 검증 방법으로 데이터 분할

from sklearn.model_selection import KFold

# KFold 클래스를 이용하여 교차 검증 분할을 수행
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# 교차 검증을 수행

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# Model 클래스를 정의
# Model 클래스는 fit으로 학습하고, predict로 예측값 확률을 출력

scores = []

# KFold 클래스를 이용하여 교차 검증 방법으로 분할
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 학습 실행, 검증 데이터의 예측값 출력, 점수 계산
    model = Model()
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    scores.append(score)

# 각 폴더의 점수 평균을 출력
print(np.mean(scores))
```

### 층화 K-겹 검증

+	분류 문제에서 폴드마다 포함되는 클래스의 비율을 서로 맞출 때가 자주 있는데 이것을 층화추출이라고 부른다. 
+	StratifiedKFold 클래스로 층화추출을 통한 검증을 수행할 수 있다.
+	Kfold 클래스와 달리 층화추출을 위해 split 메서드의 인수에 목적변수를 입력해야 한다.
+	홀드아웃 검증으로 층화추출을 하고 싶을 때는 train_test_split 함수의 stratify인수에 목적변수를 지정한다.

#### 층화 K-겹 검증 코드 구현

```python
# Stratified K-Fold
# -----------------------------------
from sklearn.model_selection import StratifiedKFold

# StratifiedKFold 클래스를 이용하여 층화추출로 데이터 분할
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x, train_y):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
```

### 그룹 k-겹 검증


경진 대회에 따라서는 학습 데이터와 테스트 데이터가 랜덤으로 분할되지 않을 때도 있다.


+ 학습 데이터와 테스트 데이터에 동일한 고객 데이터가 포함되지 않도록 분할한다.
+ 위 경우 단순히 랜덤하게 데이터를 분할하여 검증하면 본래의 성능보다 과대 평가될 우려가 있어서 고객 단위로 데이터를 분할해야한다.
+ 검증에서도 고객 단위로 데이터를 분할해준다.
+ 그룹 k-겹 검증은 사이킷런의 GroupKFold 클래스를 이용한다.


#### 그룹 k-겹 검증 코드 구현

```python
# GroupKFold
# -----------------------------------
# 4건씩 같은 유저가 있는 데이터였다고 가정한다.
train_x['user_id'] = np.arange(0, len(train_x)) // 4
# -----------------------------------

from sklearn.model_selection import KFold, GroupKFold

# user_id열의 고객 ID 단위로 분할
user_id = train_x['user_id']
unique_user_ids = user_id.unique()

# KFold 클래스를 이용하여 고객 ID 단위로 분할
scores = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_group_idx, va_group_idx in kf.split(unique_user_ids):
    # 고객 ID를 train/valid(학습에 사용하는 데이터, 검증 데이터)로 분할
    tr_groups, va_groups = unique_user_ids[tr_group_idx], unique_user_ids[va_group_idx]

    # 각 샘플의 고객 ID가 train/valid 중 어느 쪽에 속해 있느냐에 따라 분할
    is_tr = user_id.isin(tr_groups)
    is_va = user_id.isin(va_groups)
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]
```


### LOO검증

+ 경진 대회에서는 드문 경우이지만 학습 데이터의 데이터 수가 극히 적을 때가 있다.
+ 데이터가 적으면 가능한 한 많은 데이터를 사용하려 하고 학습에 걸리는 연산 시간도 짧으므로 폴드 수를 늘리는 방법을 고려할 수 있다.
+ Kfold 클래스에서 n_splits 인수에 데이터 행의 수를 지정하기만 하면 되지만 LOO검증을 수행하는 LeaveOneOut 클래스도 있다.
+ LOO검증의 경우 GBDT나 신경망과 같이 순서대로 학습을 진행하는 모델에서 조기종료를 사용하면 검증 데이터에 가장 최적의 포인트에서 학습을 멈출 수 있어 모델의 성능이 과대 평가된다.


#### LOO검증 코드 구현

```python
# leave-one-out
# -----------------------------------
# 데이터가 100건밖에 없는 것으로 간주
train_x = train_x.iloc[:100, :].copy()
# -----------------------------------
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
for tr_idx, va_idx in loo.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
```


-------------------------------------------

"데이터가 뛰어노는 AI놀이터, 캐글" 한빛미디어 인용

오류가 있을시 dothe7847@nate.com 연락부탁드립니다.


<script src="https://utteranc.es/client.js"
        repo="lee-jun-yong/blog-comments"
        issue-term="pathname"
        theme="github-light"
        crossorigin="anonymous"
        async>
</script>



