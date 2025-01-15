---
title:  "[머신러닝]RandomForest"
categories:
  - ML
---  
# 랜덤포레스트란?(RandomForest)
- 기존에 의사결정나무는 뿌리노드를 잘못 잡으면 결과가 이상해졌다.
- 또한 과적합 등의 문제가 많이 일어났다.
- 랜덤포레스트 모델은 여러개의 독립적인 트리를 생성하여 이러한 문제들을 해결하였다.
- 의사결정나무 설명 링크 : https://bnacho.github.io/ml/머신러닝-의사결정나무/

# 랜덤포레스트로 분류하기(RandomForestClassifier)
- 의사결정나무를 기초로 하기에 분류와 회귀가 모두 가능하다.
- 이번 챕터에서는 분류의 예제를 보여준다.

## 랜덤포레스트분류(RandomForestCalssifier)의 주요 파라미터
- n_estimators : 생성할 트리의 개수
- max_depth : 트리의 깊이
- min_samples_split : 노드를 분할하기 위한 최소 샘플 수
- min_samples_leaf : 리프 노드에 있어야 할 최소 샘플 수
- max_leaf_nodes : 최대 리프 노드 수
- max_features : 각 분할에서 고려할 최대 특성 수
- random_state : 결과를 재현 가능하도록 설정
- class_weight : 클래스 가중치를 지정하여 불균형 데이터를 처리
- criterion : 트리의 분할 품질을 평가하는 기준


```python
# 라이브러리 로드
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mglearn

from sklearn.ensemble import RandomForestClassifier
```


```python
# 데이터 생성
data = load_iris()
X = data.data[:, :2]
y = data.target
```


```python
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```


```python
# 모델 생성 및 학습
rf = RandomForestClassifier(n_estimators = 5, random_state= 42)
rf.fit(X_train, y_train)
```




```python
# 예측 및 정확도 측정
y_pred = rf.predict(X_test)

print("정확도 : ", accuracy_score(y_test, y_pred))
```

    정확도 :  0.7111111111111111
    


```python
# 각 트리마다 결정경계 시각화
fig, axes = plt.subplots(2, 3, figsize=(20,10) )
for i, (ax, tree) in enumerate( zip( axes.ravel(), rf.estimators_ ) ):
    ax.set_title("tree {}".format(i) )
    mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)

axes[-1, -1].set_title("Random forest")
mglearn.plots.plot_2d_separator(rf, X, fill=True, alpha=0.5, ax=axes[-1,-1] )
mglearn.discrete_scatter(X[:,0], X[:,1], y)
```





    
![output_8_1](https://github.com/user-attachments/assets/98941d99-2254-4d4e-9532-ed4b83952251)
    


# 랜덤포레스트로 회귀하기(RandomForestRegressor)
- 의사결정나무를 기초로 하기에 분류와 회귀가 모두 가능하다.
- 이번 챕터에서는 회귀의 예제를 보여준다.

## 랜덤포레스트회귀(RandomForestRegressor)의 주요 파라미터
- n_estimators : 생성할 트리의 개수
- max_depth : 트리의 깊이
- min_samples_split : 노드를 분할하기 위한 최소 샘플 수
- min_samples_leaf : 리프 노드에 있어야 할 최소 샘플 수
- max_leaf_nodes : 최대 리프 노드 수
- max_features : 각 분할에서 고려할 최대 특성 수
- random_state : 결과를 재현 가능하도록 설정
- class_weight : 클래스 가중치를 지정하여 불균형 데이터를 처리
- criterion : 트리의 분할 품질을 평가하는 기준


```python
# 라이브러리 로드
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mglearn

from sklearn.ensemble import RandomForestRegressor
```


```python
# 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
```


```python
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```


```python
# 모델 생성 및 학습
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```








```python
# 예측 및 정확도 측정
y_pred = rf.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
```

    Mean Squared Error: 1.1205671165564837
    


```python
plt.figure(figsize=(8, 6))

# 학습 데이터 시각화
plt.scatter(X_train, y_train, color='blue', label='Train Data', alpha=0.6)

# 테스트 데이터와 예측값 시각화
plt.scatter(X_test, y_test, color='green', label='Test Data', alpha=0.6)  # x와 y 크기 맞춤
plt.scatter(X_test, y_pred, color='red', label='Predicted Values', alpha=0.6)  # x와 y 크기 맞춤

# 모델의 결정 경계 시각화
X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)  # 격자 좌표 생성
y_range_pred = rf.predict(X_range)
plt.plot(X_range, y_range_pred, color='black', label='Model Prediction')

plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('RandomForestRegressor Prediction')
plt.legend()
plt.show()
```


    
![output_16_0](https://github.com/user-attachments/assets/c954c69b-2535-456d-a490-a95824a15207)
    

