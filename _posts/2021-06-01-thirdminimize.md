---
layout: single
title: "캐글스터디 3회차 : 차원축소기법과 자연어처리 기법 "
---


## 차원축소기법

### 1. PCA(주성분분석)

+ 주성분분석은 차원축소의 가장 대표적인 기법으로, 다차원 데이터를 분산이 큰 방향에서부터 순서대로 축을 다시 잡는 방법이다. 

+ 각 특징이 정규분포를 따르는 조건을 가정하므로 왜곡된 분포를 가진 변수를 주성분분석에 적용하는 것은 좋지 않다.

+ 차원축소로서 특잇값분해(SVD)는 PCA와 거의 같은 의미이다. 

+ 주성분분석은 사이킷런 decomposition 모듈의 PCA 및 TruncatedSVD 클래스에서 시행할 수 있다. 



```python
# -----------------------------------
# PCA
# 데이터는 표준화 등의 스케일을 갖추기 위한 전처리가 이루어져야 함
train_x, test_x = load_standarized_data()
from sklearn.decomposition import PCA

# 학습 데이터를 기반으로 PCA에 의한 변환을 정의
pca = PCA(n_components=5)
pca.fit(train_x)

# 변환 적용
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)

# -----------------------------------
# TruncatedSVD 특이값 분해
# 표준화된 데이터를 사용
train_x, test_x = load_standarized_data()
from sklearn.decomposition import TruncatedSVD

# 데이터는 표준화 등의 스케일을 갖추기 위한 전처리가 이루어져야 함

# 학습 데이터를 기반으로 SVD를 통한 변환 정의
svd = TruncatedSVD(n_components=5, random_state=71)
svd.fit(train_x)

# 변환 적용
train_x = svd.transform(train_x)
test_x = svd.transform(test_x)
```

↑ 사이킷런 decomposition 모듈의 PCA 및 TruncatedSVD 클래스 이용

### 2. NMF(음수 미포함 행렬 분해)

+ 음수 미포함 행렬 분해는 음수를 포함하지 않은 행렬 데이터를, 음수를 포함하지 않은 행렬들의 곱의 형태로 만드는 방법이다.

+ 음수가 아닌 데이터에만 사용할 수 있지만 PCA와는 달리 벡터의 합 형태로 나타낼 수 있다.


```python
# NMF
# 비음수의 값이기 때문에 MinMax스케일링을 수행한 데이터를 이용
train_x, test_x = load_minmax_scaled_data()
from sklearn.decomposition import NMF

# 데이터는 음수가 아닌 값으로 구성
# 학습 데이터를 기반으로 NMF에 의한 변환 정의
model = NMF(n_components=5, init='random', random_state=71)
model.fit(train_x)

# 변환 적용
train_x = model.transform(train_x)
test_x = model.transform(test_x)
```

### 3. LDA(잠재 디리클레 할당)

+ 잠재 디리클레 할당(LDA)는 자연어 처리에서 문서를 분류하는 토픽 모델에서 쓰이는 기법이다.

+ 각 문서를 행으로, 각 단어를 열로 하여 해당 문서에 해당 단어가 몇 번이나 나타나는지 보여주는 단어-문서 카운트 행렬을 작성한다. 이 때, 분류할 토픽의 수를 지정한다.

+ LDA는 **베이즈 추론** 을 이용하여 행렬에서 각 문서를 확률적으로 토픽으로 분류한다.

>베이즈 추론(Bayesian inference):
>통계적 추론의 한 방법으로, 추론 대상의 사전 확률과 추가적인 정보를 통해 해당 대상의 사후 확률을 추론하는 방법이다.

+ LDA를 적용하면 단어-문서 카운트 행렬, 문서가 각 토픽에 소속될 확률을 나타내는 행렬, 각 토픽의 단어 분포를 나타내는 행렬이 나타난다.



```python
# LatentDirichletAllocation (LDA)
# MinMax스케일링을 수행한 데이터를 이용
# 카운트 행렬은 아니지만, 음수가 아닌 값이면 계산 가능
train_x, test_x = load_minmax_scaled_data()
from sklearn.decomposition import LatentDirichletAllocation

# 데이터는 단어-문서의 카운트 행렬 등으로 함

# 학습 데이터를 기반으로 LDA에 의한 변환을 정의
model = LatentDirichletAllocation(n_components=5, random_state=71)
model.fit(train_x)

# 변환 적용
train_x = model.transform(train_x)
test_x = model.transform(test_x)
```

### 4.LDA(선형판별분석)

+ 선형판별분석(LDA) 는 지도 학습의 분류 문제에서 차원축소를 실시하는 방법이다.

+ 학습 데이터를 잘 분류할 수 있는 저차원의 특징 공간을 찾고, 원래 특징을 그 공간에 
투영함으로써 차원을 줄인다.

+ 학습 데이터가 n행의 행 데이터와 f개의 특징으로 이루어진 n X f 행렬이라 할 때 f X K의 변환 행렬을
곱함으로써 n X K 행렬로 변환한다.

+ 차원축소 후의 차원 수 k는 클래스 수보다 줄어들고, 이진 분류일 때는 변환 후에 1차원 값이 된다.



```python
# LinearDiscriminantAnalysis (LDA)
# 표준화된 데이터를 사용
train_x, test_x = load_standarized_data()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 데이터는 단어-문서의 카운트 행렬 등으로 함

# 학습 데이터를 기반으로 LDA에 의한 변환을 정의
lda = LDA(n_components=1)
lda.fit(train_x, train_y)

# 변환 적용
train_x = lda.transform(train_x)
test_x = lda.transform(test_x)
```

### 5. t-SNE

+ t-SNE는 차원축소의 한 방법으로, 데이터를 2차원 평면상에 압축하여 시각화하기 위해 사용된다.

+ 비선형 관계를 파악할 수 있으므로 원래의 특징에 t-SNE로 표현된 압축 결과를 더하면 모델 성능이 올라갈 수 있다.


```python
train_x, test_x = load_standarized_data()
import sklearn
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)

data = pd.concat([train_x,test_x])
tsne_data = tsne.fit_transform([data])
```



### 6.UMAP

+ UMAP은 t-SNE와 마찬가지로 원래의 특징 공간상에서 가까운 점이 압축 후에도 가까워지도록 표현한다.

+ 실행시간이 매우 빠르다는 것이 장점이다.


```python
# UMAP
# 표준화된 데이터를 사용
train_x, test_x = load_standarized_data()
import umap

# 데이터는 표준화 등의 스케일을 갖추는 전처리가 이루어져야 함

# 학습 데이터를 기반으로 UMAP에 의한 변환을 정의
um = umap.UMAP()
um.fit(train_x)

# 변환 적용
train_x = um.transform(train_x)
test_x = um.transform(test_x)
```

### 7. 오토인코더

+ 오토인코더는 신경망을 이용한 차원 압축 방법이다.

+ 입력 차원보다 작은 중간층을 이용해 입력과 같은 값으로 출력하는 신경망을 학습함으로써 원래의 데이터를 재현할 수 있는 더 저차원의 표현을 학습한다.


### 8. 군집화

+ 군집화(clustering)는 데이터를 여러 그룹으로 나누는 비지도 학습이다.

+ 데이터가 어느 그룹에 분류 되었는지에 대한 값을 특징으로 할 수 있고, 클러스터 중심으로부터의 거리를 특징으로 할 수도 있다.

+ 군집을 수행하는 알고리즘은 K-Means, DBSCAN, 병합군집이 있다



```python
# 클러스터링
# 표준화된 데이터를 사용
train_x, test_x = load_standarized_data()
from sklearn.cluster import MiniBatchKMeans

# 데이터는 표준화 등의 스케일을 갖추는 전처리가 이루어져야 함

# 학습 데이터를 기반으로 Mini-Batch K-Means를 통한 변환 정의
kmeans = MiniBatchKMeans(n_clusters=10, random_state=71)
kmeans.fit(train_x)

# 해당 클러스터를 예측
train_clusters = kmeans.predict(train_x)
test_clusters = kmeans.predict(test_x)

# 각 클러스터 중심까지의 거리를 저장
train_distances = kmeans.transform(train_x)
test_distances = kmeans.transform(test_x)
```

## 자연어 처리 기법

### 1. Bag-of-word(BoW)


- 문장 등의 텍스트를 단어로 분할하고, 각 단어의 출현 수를 순서에 상관없이 단순하게 세는 방식이다.

- 사이킷런 feature_extraction.text 모듈의 CountVectorizer에서 처리할 수 있다.

### 2. n-gram

- Bow에서 분할하는 단위를, 단어가 아닌 연속되는 단어 뭉치 단위로 끊는 방법이다.
>예를 들어 ‘This is a sentence’라는 문장에서 [this, is, a, sentence]라는 4개의 단어를 추출할 수 있지만 2-gram에서는 [This-is, is-a, a-sentence]라는 3개의 단어 뭉치를 추출한다.
- 단어 분할에 비해 텍스트에 포함된 정보를 유지하기는 좋지만, 출현 가능한 종류의 수가 크게 늘어날 뿐만 아니라 희소 데이터가 된다.


### 3. tf-idf

- BoW에서 작성한 단어-문서 카운트 행렬을 변환하는 기법이다.

 >> 단어 빈도(TF): 어떤 텍스트에서의 특정 단어의 출현 비율

 >> 역문서 빈도(IDF): tf와 반대되는 개념으로 특정 단어가 나타나는 문서의 수.

- CounterVectorizer 클래스 등으로 작성된 행렬에 사이킷런 feature_extraction.text 모듈의 Tfidf Transformer를 적용함으로써 이 기법을 처리할 수 있다.


### 4. 단어 임베딩

- 단어를 수치 벡터로 변환하는 방법을 단어 임베딩이라고 한다.


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
