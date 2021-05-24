---
layout:single
title:"캐글 스터디 2회차 :특징생성"
---

```python
# ---------------------------------
# 데이터 등 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수, test_x는 테스트 데이터
# pandas의 DataFrame, Series의 자료형 사용(numpy의 array로 값을 저장하기도 함.)

train = pd.read_csv(r'C:\Users\leez\Desktop\kagglebook-main\input\sample-data\train.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv(r'C:\Users\leez\Desktop\kagglebook-main\input\sample-data\test.csv')


# 설명용으로 학습 데이터와 테스트 데이터의 원래 상태를 복제해 두기
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()


# 학습 데이터와 테스트 데이터를 반환하는 함수
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x

num_cols = ['age', 'height', 'weight', 'amount',
            'medical_info_a1', 'medical_info_a2', 'medical_info_a3', 'medical_info_b1']
```

# 캐글 스터디 2회차:특징생성

캐글 대회에서 모델의 성능을 높이는 가장중요한 특징을 만드는 방법에 대해 알아보았다.

### 목차
1. 결측값처리

    ①결측값 인채 처리
    
    ②대푯값으로 결측값 채우기
    
    ③결측값을 이용해 새로운 특징생성
    
    ④결측값을 가진 행 변수를 제외
    
2. 수치형변수 변환

    ①표준화(StandardScaler)
    
    ②최소최대스케일링(MinMaxScaler)
    
    ③로그변환, log(x+1) 변환,절댓값 로그 변환(비선형변환)
    
    ④박스-칵스 변환(비선형변환)
    
    ⑤여-존슨 변환(비선형변환)
    
    ⑥클리핑
    
    ⑦구간분할
    
    ⑧순위로 변환
    
    ⑨RankGauss
    
3. 범주형변수 변환

    ①원-핫 인코딩(One-hot-encoding)
    
    ②레이블 인코딩(label encoding)
    
    ③특징 해싱(feature hashing)
    
    ④프리퀀시인코딩(frequency encoding)
    
    ⑤타깃 인코딩(target encoding)
    
    ⑥임베딩
    
    ⑦순서변수의 취급
    
4. 날짜 및 시간변수 변환

## 1. 결측값처리

### ①결측값인 채 처리

●GBDT모델은 결측값을채우지 않고도 그대로 쓸 수 있다.

●사이킷런의랜덤 포레스트와같이 결측값을그대로 취급할 수 없는 라이브러리도 있다.
이럴때는 결측값으로 쉽게 얻을수 없는값을 넣어 결측값임을 인증하는 것(ex -9999)

●결측값이존재한다면 결측값인지 아닌지에따라 데이터를 나누어 처리한다.

###  ②대푯값으로 결측값 채우기

● 수치형 변수에서는 평균값과 중앙값으로 채울수 있다.

● 범주형 변수의 값에 따라 그룹별로 평균을 구할 경우 데이터 샘플 수가 극단적으로 적은 범주가 있으면 그 평균값에는 별로 믿음이 가지 않는다. 이러한 경우 분자와 분모에 정수항을 더하여 계산하는 베이즈평균이라는 방법이 있다.


베이즈평균 

![1](https://user-images.githubusercontent.com/84025932/119360825-fc381a80-bce5-11eb-8d34-5d8250aa8260.jpg)

값 m의 데이터를 C개 측정했다고 보고 평균 계산에 추가한다. 데이터 수가 적을 때는 m에 근접하고 충분히 많을 때는 해당 범주의 평균에 근접한다.

●해당 특징의 범줏값중에 가장많은 수를 대푯값으로 변경하는 방법이있다.

###    ③결측값을 이용해 새로운 특징생성


●결측값이 아무런 이유 없이 임의로 만들어지는 경우는 드물다. 어떤한이유로 결측값이발생했을 때는 그러한 상황 자체가 정보를 포함하므로, 해당 결측값으로새로운 특징을 생성할 수 있다.

●이때 간단한 방법은 결측 여부를 나타내는 두 값(0 또는 1)을 갖는 변수를 생성하는 것이다. 결측값을채우더라도 해당 변수들을 따로 만들어두면추가된 정보를 사용할 수 있습니다. 결측 상태의 변수가 여러 개라면 각각에 대해 두 값을 갖는 변수를 생성한다.

●여러 개의 변수에서 결측값의조합을 조사하여 몇 개의 패턴으로 분류할 수 있다면, 어느 패턴인지를 하나의 특징으로 삼는다.

●pandas 모듈의 read.csv함수에서 na_values인수로 결측값을지정할 수 있다.
Ex) pd.read_csv(‘train.csv’,na_values=[‘’,‘NA’,-1,9999])

●pandas 모듈의 replace 함수를 이용해 결측값을 다른 수치나 문자로 변경할 수 있다.
Ex) data[‘col1’].replace(-1,np.nan)

 ###   ④결측값을 가진 행 변수를 제외


●경진대회에서 주어진 데이터로부터 예측에 유효한 정보를 얻는게 최대한 중요하므로,데이터를 제거하는 것은 좋은 방법이아니다.

## 2. 수치형변수 변환


수치형 변수는 기본적으로 모델 입력에 그대로 사용 할수있지만, 적절히 변환하거나 가공하면 더 효과적인 특징 생성 가능
GBDT 등 트리 모델에 기반을 둔 모델에서 대소 관계가 저장되는 변환은 학습에 거의 영향을 주지 않으므로 다음 방법들에 적용의 의미가 없다

###  ①표준화(StandardScaler) 


정의 : 선형변환을 통해 변수의 평균을 0, 표준편차를 1로 만드는 방법

선형변환: 가장 기본적인 변환 방법으로, 곱셈과 덧셈만으로 변숫값의범위를 변경하는 변환


●표준화 수식 

![2](https://user-images.githubusercontent.com/84025932/119359872-086fa800-bce5-11eb-8ac7-b6c15f454ee7.jpg)


변수의 평균값과 표준편차를 기준으로 표준화한다.

● 사이킷런 preprocessing 모듈의 StandardScaler 클래스에서 표준화 가능하다.

● 0 또는 1의 두 값으로 나타나는 변수는 0과 1의 비율이 어느 한 쪽으로 치우치면 표준편차가 작으므로, 변환한 뒤에 0 또는 1 중에 어느 한 쪽의 절댓값이 커질 가능성이 있다.

● 이들 두 값을 갖는 이진변수에 대해서는 표준화 실시하지 않아도 된다.

#### 표준화를 사용하는 경우


→ 선형 회귀나 로지스틱 회귀 등의 선형 모델은 값의 범위가 큰 변수일수록 회귀 계수가 작다
  표준화하지않으면 그런 변수의 정규화가 어려워지므로 표준화를 이용한다.
  
→ 신경망에서도 변수들 간의 값의 범위가 크게 차이나는 상태로는 학습이 잘 진행되지 않을 때가 많으므로 표준화를 이용한다.


```python
# 표준화
train_x, test_x = load_data()

from sklearn.preprocessing import StandardScaler

# 학습 데이터를 기반으로 복수 열의 표준화를 정의(평균 0, 표준편차 1)
scaler = StandardScaler()
scaler.fit(train_x[num_cols])

# 표준화를 수행한 후 각 열을 치환
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])
```

● 학습 데이터에서 각 변수의 평균값과 표준편차를 fit 메서드로 계산해 기억한 뒤 이를 통해 학습 데이터와 테스트 데이터를 변환함

###     ②최소-최대스케일링(MinMaxScaler)


정의: 변숫값이취하는 범위를 특정 구간으로 변환하는 방법

●최소-최대스케일링 수식

![3](https://user-images.githubusercontent.com/84025932/119359873-09083e80-bce5-11eb-97f7-39d2c335e48a.jpg)


●사이킷런의 MinMaxScaler클래스로 실시 가능하다.

●변환 후의 평균이 정확히 0이 되지 않고 이상치의 악영향을 받기 더 쉬워 표준화가 더 많이 쓰인다.

#### 최소최대스케일링을 사용하는 경우

→ 이미지 데이터의 각 픽셀값등은 처음부터 0~255로 범위가 정해진 변수


```python
# Min-Max 스케일링
train_x, test_x = load_data()

from sklearn.preprocessing import MinMaxScaler

# 학습 데이터를 기반으로 여러 열의 최소-최대 스케일링 정의
scaler = MinMaxScaler()
scaler.fit(train_x[num_cols])

# 정규화(0~1) 변환 후의 데이터로 각 열을 치환
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])
```

● 최소최대스케일링을 호출하고 fit과 transform으로 적용한다.


##### 비선형변환

● 표준화와 최소-최대 스케일링은 선형변환이므로변수의 분포가 유동적일 뿐 형태 그 자체는 변하지 않는다.

● 비선형변환을 통해 변수의 분포 형태를 바꾸는 편이 바람직한 경우도 있다

###     ③로그변환, log(x+1) 변환,절댓값 로그 변환(비선형변환)


●특정 금액이나 횟수를 나타내는 변수에서는 어느 한 방향으로 치우쳐 뻗은 분포가 되기 쉬우므로
로그 변환을 한다.	

●값에 0이 포함될때는 numpy 모듈의 log1p를 이용해서 log(x+1) 변환을 한다.

●값에 음수가 포함될때는 절댓값에 로그 변환을 곱한 뒤 원래의 부호를 더한다.


```python
# 로그 변환
# -----------------------------------
x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

# 단순히 값에 로그를 취함
x1 = np.log(x)

# 1을 더한 뒤에 로그를 취함
x2 = np.log1p(x)

# 절댓값의 로그를 취한 후, 원래의 부호를 추가
x3 = np.sign(x) * np.log(np.abs(x))
```

###   ④박스-칵스 변환(비선형변환)  

정의: 로그 변환을 일반화, 박스-칵스변환의 매개변수 λ= 0 일 때가 로그 변환	

●박스-칵스 변환 수식

![4](https://user-images.githubusercontent.com/84025932/119359876-09083e80-bce5-11eb-903e-1d8976a8730a.jpg)


```python
# Box-Cox 변환

train_x, test_x = load_data()
# 양의 정숫값만을 취하는 변수를 변환 대상으로 목록에 저장
# 또한, 결측값을 포함하는 경우는 (~(train_x[c] <= 0.0)).all() 등으로 해야 하므로 주의
pos_cols = [c for c in num_cols if (train_x[c] > 0.0).all() and (test_x[c] > 0.0).all()]

from sklearn.preprocessing import PowerTransformer

# 학습 데이터를 기반으로 복수 열의 박스-칵스 변환 정의
pt = PowerTransformer(method='box-cox')
pt.fit(train_x[pos_cols])

# 변환 후의 데이터로 각 열을 치환
train_x[pos_cols] = pt.transform(train_x[pos_cols])
test_x[pos_cols] = pt.transform(test_x[pos_cols])

```

● λ값을 따로 명시할 필요가 없다 (λ값은 정규분포에 근접하도록 라이브러리 측에서 최적에 값을 추정해준다)

![5](https://user-images.githubusercontent.com/84025932/119359877-09a0d500-bce5-11eb-94c2-b0f95cdda6ad.jpg)

↑ 박스-칵스 변환전 후 분포도

###     ⑤여-존슨 변환(비선형변환)


●음의 값을 갖는 변수에도 적용할수 있는 변환
●여-존슨 변환 공식

![6](https://user-images.githubusercontent.com/84025932/119359879-0a396b80-bce5-11eb-8252-76f0d94e81ed.jpg)

```python
# Yeo-Johnson변환

train_x, test_x = load_data()

from sklearn.preprocessing import PowerTransformer

# 학습 데이터를 기반으로 복수 열의 여-존슨 변환 정의
pt = PowerTransformer(method='yeo-johnson')
pt.fit(train_x[num_cols])

# 변환 후의 데이터로 각 열을 치환
train_x[num_cols] = pt.transform(train_x[num_cols])
test_x[num_cols] = pt.transform(test_x[num_cols])
```

● λ값을 따로 명시할 필요가 없다 (λ값은 정규분포에 근접하도록 라이브러리 측에서 최적에 값을 추정해준다)

###    ⑥클리핑


정의 : 수치형 변수에는 이상치가 포함되기도 하지만, 상한과 하한을 설정한 뒤 해당 범위를 벗어나는 값은 상한값과 하한값으로 치환함으로써 일정 범위를 벗어난 이상치 제외 가능하다.

● 분포를 확인한 뒤 적당한 임곗값을설정할 수도 있지만, 분위점을 임곗값으로삼아 기계적으로 이상치 치환 가능
● pandas 모듈이나 numpy 모듈의 clip함수 이용 가능


```python
# clipping

train_x, test_x = load_data()

# 열마다 학습 데이터의 1%, 99% 지점을 확인
p01 = train_x[num_cols].quantile(0.01)
p99 = train_x[num_cols].quantile(0.99)

# 1％점 이하의 값은 1%점으로, 99%점 이상의 값은 99%점으로 클리핑
train_x[num_cols] = train_x[num_cols].clip(p01, p99, axis=1)
test_x[num_cols] = test_x[num_cols].clip(p01, p99, axis=1)
```

![7](https://user-images.githubusercontent.com/84025932/119359882-0a396b80-bce5-11eb-8905-9dec8051f16e.jpg)

↑클리핑 전후 산포도

###     ⑦구간분할

정의: 수치형 변수를 구간별로 나누어 범주형 변수로 변환하는 방법

● 같은 간격으로 분할 / 분위점을 이용해 분할 / 구간 구분을 지정해 분할 등

● 데이터에 대한 사전 지식이 있고 어떤 구간으로 나눠야 하는지 알고 있다면 더 효과적인 방법

● 구간분할 시 순서 있는 범주형 변수가 되므로, 순서 그대로 수치화 가능& 범주형 변수로서 원-핫 인코딩 등 적용 가능하다.

● pandas 모듈의 cut 함수 와 numpy 모듈의 digitize 함수를 이용할수 있다.


```python
# 구간분할

x = [1, 7, 5, 4, 6, 3]

# 팬더스 라이브러리의 cut 함수로 구간분할 수행

# bin의 수를 지정할 경우
binned = pd.cut(x, 3, labels=False)
print(binned)
# [0 2 1 1 2 0] - 변환된 값은 세 구간(0, 1, 2)를 만들고 원본 x의 값이 어디에 해당되는지 나타냄

# bin의 범위를 지정할 경우(3.0 이하, 3.0보다 크고 5.0보다 이하, 5.0보다 큼)
bin_edges = [-float('inf'), 3.0, 5.0, float('inf')]
binned = pd.cut(x, bin_edges, labels=False)
print(binned)
# [0 2 1 1 2 0] - 변환된 값은 세 구간을 만들고 원본 x의 값이 어디에 해당되는지 나타냄
```

    [0 2 1 1 2 0]
    [0 2 1 1 2 0]
    

### ⑧순위로 변환


정의:수치형 변수를 대소 관계에 따른 순위로 변환하는 방법

●순위를 행 데이터의 수로 나누면 0~1범위에 들어가고, 값의 범위가 행 데이터의 수에 의존하지 않아 다루기 쉽다.

●수치의 크기나 간격 정보를 버리고 대소 관계만을 얻어내는 방법

●pandas 모듈의 rank 함수 사용,numpy 모듈의 argsort 함수 2회 적용


```python
# 순위로 변환

x = [10, 20, 30, 0, 40, 40]

# 팬더스의 rank 함수로 순위 변환
rank = pd.Series(x).rank()
print(rank.values)
# 시작이 1, 같은 순위가 있을 경우에는 평균 순위가 됨
# [2. 3. 4. 1. 5.5 5.5]

# 넘파이의 argsort 함수를 2회 적용하는 방법으로 순위 변환
order = np.argsort(x)
rank = np.argsort(order)
print(rank)
# 넘파이의 argsort 함수를 2회 적용하는 방법으로 순위 변환
# [1 2 3 0 4 5]
```

    [2.  3.  4.  1.  5.5 5.5]
    [1 2 3 0 4 5]
    

###     ⑨RankGauss

정의: 수치형 변수를 순위로 변환한 뒤 순서를 유지한 채 반강제로 정규분포가 되도록 변환하는 방법

●신경망에서 모델을 구축할 때의 변환으로서 일반적인 표준화보다 좋은 성능을 나타낸다.

●사이킷런 preprocessing 모듈의 QuantileTransformer 클래스에서 n_quantiles을 충분히 크게 한 뒤 output_distribution=‘normal’로 지정시이 변환 실시 가능하다.



```python
# RankGauss

train_x, test_x = load_data()

from sklearn.preprocessing import QuantileTransformer

# 학습 데이터를 기반으로 복수 열의 RankGauss를 통한 변환 정의
transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

# 변환 후의 데이터로 각 열을 치환
train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])

```

![8](https://user-images.githubusercontent.com/84025932/119359886-0ad20200-bce5-11eb-9d6e-9092e76e276a.jpg)


↑ RankGauss 전후 분포도

●output_distribution=‘uniform’이라고 하면 균등분포에 가깝게 변환됨

## 3. 범주형변수 변환


### 범주형변수

정의: 몇 개의 동일한 성질이 갖는 부류나 범위로 나눌 수있는변수

● 앞선 수치형 변수와 더불어 범주형 변수 또한 대표적인 변수

● 범주형 변수는 많은 머신러닝모델에서 그대로 분석에 쓸수없으므로 모델마다 적합한 형태로 변환하여 사용 

● 범주형 변수는 주로 문자열의 형태를 가지지만 데이터상 수치라 하더라도 값이나 크기나 순서가  의미가 없을때는변수형 범주로 취급


###     ①원-핫 인코딩(One-hot-encoding)


![슬라이드4](https://user-images.githubusercontent.com/84025932/119359887-0b6a9880-bce5-11eb-977d-1b36e09ec4d7.JPG)


#### pandas의 get_dummies를 이용한 원핫인코딩

![슬라이드5](https://user-images.githubusercontent.com/84025932/119359890-0b6a9880-bce5-11eb-8737-5af5a3438e9f.JPG)
![슬라이드6](https://user-images.githubusercontent.com/84025932/119359894-0c032f00-bce5-11eb-9be9-a9ef6e9a61be.JPG)

#### scikit-learn의onehotencorder를 이용한 원핫 인코딩

![슬라이드7](https://user-images.githubusercontent.com/84025932/119359896-0c9bc580-bce5-11eb-98e4-c13b077225b1.JPG)
![슬라이드8](https://user-images.githubusercontent.com/84025932/119359899-0d345c00-bce5-11eb-90af-c36bdba6a8e7.JPG)

#### 원-핫 인코딩(One-hot-encoding)의 단점

◦ 원-핫 인코딩의 경우 특징의 개수가 범주형 변수 레벨의 개수에 따라 증가한다는 결점이있다. 

◦ 레벨의 종류가 많을때는값이 0인 특징들이 많아져 쓸모있는정보가 적어진다. 

◦ 결과적으로 계산시간이 증가하고 메모리가 증가하여 모델 성능에 악영향을 준다.

#### 원-핫 인코딩의 결점 보완방안


◦ 원-핫 인코딩 이외의 다른 인코딩 방법 검토

◦ 임의의 규칙을 활용한 그룹화로 범주형 변수의 레벨 개수(종류) 줄이기

◦ 빈도가 낮은 범주들을 모두 ‘기타 범주’에 한데 모아 정리하기

###     ②레이블 인코딩(label encoding)


![슬라이드10](https://user-images.githubusercontent.com/84025932/119359904-0dccf280-bce5-11eb-9e51-968e000041e6.JPG)

#### scikit-learn의 labelencorder를 이용한 레이블 인코딩 


![슬라이드11](https://user-images.githubusercontent.com/84025932/119359906-0e658900-bce5-11eb-9dbc-d4973294f1df.JPG)
![슬라이드12](https://user-images.githubusercontent.com/84025932/119359909-0e658900-bce5-11eb-8da7-8e367ed0161c.JPG)

###  ③특징 해싱(feature hashing)

![슬라이드13](https://user-images.githubusercontent.com/84025932/119359911-0efe1f80-bce5-11eb-9529-4e7f6353d7e1.JPG)

#### scikit-learn의 FeatureHasher를 이용한 특징해싱


![슬라이드14](https://user-images.githubusercontent.com/84025932/119359914-0f96b600-bce5-11eb-83e3-6d051b614a4e.JPG)
![슬라이드15](https://user-images.githubusercontent.com/84025932/119359916-102f4c80-bce5-11eb-879c-b1d4d50445ff.JPG)

###   ④프리퀀시인코딩(frequency encoding)


![슬라이드16](https://user-images.githubusercontent.com/84025932/119359917-102f4c80-bce5-11eb-9e29-2198910b2574.JPG)
![슬라이드17](https://user-images.githubusercontent.com/84025932/119359924-11f91000-bce5-11eb-8b2a-4e900d37950b.JPG)

###     ⑤타깃 인코딩(target encoding)

![슬라이드18](https://user-images.githubusercontent.com/84025932/119359927-11f91000-bce5-11eb-9e04-7c563b919b38.JPG)

#### 타깃 인코딩(1)타깃 인코딩용 폴드분할

![슬라이드19](https://user-images.githubusercontent.com/84025932/119359929-1291a680-bce5-11eb-947e-a03af209a89e.JPG)
![슬라이드20](https://user-images.githubusercontent.com/84025932/119359931-132a3d00-bce5-11eb-817e-a61391897822.JPG)

#### 타깃 인코딩(2)교차 검증

![슬라이드21](https://user-images.githubusercontent.com/84025932/119359938-145b6a00-bce5-11eb-8b0b-76892e8b48b5.JPG)
![슬라이드22](https://user-images.githubusercontent.com/84025932/119359940-14f40080-bce5-11eb-94a3-46f32126f930.JPG)

#### 타겟 인코딩 모델별 목적변수 평균구하는법


●회귀 - 목적변수의 평균

●이진분류 - 양성일때 1, 음성일때 0 으로 하여 평균

●다중 클래스 분류- 클래스 수만큼 이진 분류가 있다 가정하고, 클래스 수 만큼 타깃 인코딩의 특징 생성

●이상치 존재시 - 평균값보다 중앙값을 이용

●평가지표가 RMSLE - 로그 변환 후 목적변수의 평균을 계산

#### 타깃 인코딩 데이터 정보 노출(1)

![슬라이드24](https://user-images.githubusercontent.com/84025932/119359947-16252d80-bce5-11eb-9701-e0b9f5d3cf4f.JPG)

#### 타깃 인코딩 데이터 정보 노출(2)

![슬라이드25](https://user-images.githubusercontent.com/84025932/119359949-16bdc400-bce5-11eb-9c3d-7e05326214e3.JPG)
![슬라이드26](https://user-images.githubusercontent.com/84025932/119359958-17eef100-bce5-11eb-87d5-fc39ab041add.JPG)

### ⑥임베딩
    
![슬라이드27](https://user-images.githubusercontent.com/84025932/119359961-18878780-bce5-11eb-89b7-46591537c4f2.JPG)

### ⑦순서변수의 취급

![슬라이드28](https://user-images.githubusercontent.com/84025932/119359965-19201e00-bce5-11eb-902a-a9eacbfc222f.JPG)

## 4. 날씨 및 시간변수 변환 

날짜 변수와 시간 변수로 만들 수 있는 특징


연월,월일

  ●시간 정보를 24시간이 아닌 몇 시간 단위의 구간으로 그룹화하여 과적합을피하게 해주는 특징이다.

주수,월일,일수

  ●주수로는 계절적인 경향을 더 잘 파악할 수 있으나 과적합의위험이 크다

  ●월일은 연도가 달라지면 같은 날짜라도 경향이 바뀐다.
  
사분기

●월을 사분기에 포함시켜 나타낸다

요일 ,공휴일,휴일

  ●사람의 행동은 요일에 따라 변하는 경우가 많다.

  ●요일을 레이블 인코딩또는 원 핫 인코딩을통해 표현한다. 

  ●공휴일의 전 날 또는 다음 날 인가에 따라서도 사람의 행동이 변한다.

특정 기념일

  ●공휴일이나 휴일과 마찬가지로 기념일의 전 후일을 기준으로 일련의 경향이 나타난다.

시,분,초

  ●시간을 특징로하루의 주기적인 움직임을 표현할 수 있다.
  
시간차

  ●예측하려는 데이터와 어느 시점에서의 시간차를 특징으로 삼을 수 있다.


```python

```
