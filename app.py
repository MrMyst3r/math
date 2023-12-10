
import streamlit as st

# 데이터셋과 시각화용 라이브러리
import mglearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs

# 데이터 분류와 모델 형성에 사용되는 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler         
from sklearn.neural_network import MLPClassifier


st.markdown(
    f"""
    <style>
        .centered-text {{
            text-align: center;
        }}
    </style>
    """
    , unsafe_allow_html=True
)

st.markdown("<h1 class='centered-text'>창원 중앙고 1학년 수학 탐구</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='centered-text'>10921 이현수</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='centered-text'>MLP 인공신경망의 합성함수로의 분석</h2>", unsafe_allow_html=True)

# 랜덤 시드
random_list = list(range(1,101))
default_index_random = random_list.index(42) if 42 in random_list else 0
selected_random = st.sidebar.selectbox("랜덤 시드", random_list, index=default_index_random)

# 노이즈
noise_list = list(range(1,5))
default_index_noise = noise_list.index(2) if 1 in noise_list else 0
selected_noise = st.sidebar.selectbox("노이즈 정도", noise_list, index=default_index_noise)

X, y = make_moons(n_samples=1000, noise= 0.15*selected_noise, random_state=selected_random)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Cluster 1')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Cluster 2')
plt.title('Random Data with Noise')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
st.pyplot(plt)

# 은닉층 형태
hidden1_list = list(range(1, 21))
default_index_hidden1 = hidden1_list.index(10) if 10 in hidden1_list else 0
selected_hidden1 = st.sidebar.selectbox("첫번째 은닉층 노드 개수", hidden1_list, index=default_index_hidden1)

hidden2_list = list(range(0, 11))
default_index_hidden2 = hidden2_list.index(0) if 0 in hidden2_list else 0
selected_hidden2 = st.sidebar.selectbox("두번째 은닉층 노드 개수", hidden2_list, index=default_index_hidden2)

if selected_hidden2 == 0:
    mlp = MLPClassifier(solver='lbfgs', activation='relu', hidden_layer_sizes=(selected_hidden1,)).fit(X_train, y_train)
else:
    mlp = MLPClassifier(solver='lbfgs', activation='relu', hidden_layer_sizes=(selected_hidden1, selected_hidden2)).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill = True, alpha= 0.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.title('MLP Classifier')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xticks()
plt.yticks()
plt.legend()
st.pyplot(plt)

from sklearn.metrics import r2_score

y_pred = mlp.predict(X_test)
y_train_pred = mlp.predict(X_train)

# R2 결정 계수
r2 = r2_score(y_test, y_pred)

st.subheader(f'테스트용 데이터에 대한 R2결정계수 : {round(float(r2), 4)}')

# 모델에서 가중치, 편향
weights = mlp.coefs_
biases = mlp.intercepts_

# 입력 데이터
X_sample = X_test[0] 

# 은닉층과 출력층 연산
layer_output = []
layer_output.append(X_sample)  # 입력층 출력 추가

for i in range(len(weights)):
    layer_input = np.dot(layer_output[i], weights[i]) + biases[i]
    if i == len(weights) - 1:
        layer_output.append(1 / (1 + np.exp(-layer_input)))  # 출력층은 시그모이드 함수 적용
    else:
        layer_output.append(np.maximum(0, layer_input))  # 은닉층은 ReLU 함수 적용

# 출력
for i, output in enumerate(layer_output):
    st.write(f"층 {i} 출력(soft label):", output)

# 모델의 예측 값
y_pred = mlp.predict(X_test)
st.write("모델의 예측 값(hard label):", y_pred[0])

# 인덱스 찾기
sample_index = np.where((X_test == X_sample).all(axis=1))[0][0]

# 실제값 가져오기
actual_value = y_test[sample_index]
st.write("실제값:", actual_value)


code = '''
import streamlit as st

# 데이터셋과 시각화용 라이브러리
import mglearn
import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs

# 데이터 분류와 모델 형성에 사용되는 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler         
from sklearn.neural_network import MLPClassifier


st.markdown(
    f"""
    <style>
        .centered-text {{
            text-align: center;
        }}
    </style>
    """
    , unsafe_allow_html=True
)

st.markdown("<h1 class='centered-text'>창원 중앙고 1학년 수학 탐구</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='centered-text'>10921 이현수</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='centered-text'>MLP 인공신경망의 합성함수로의 분석</h2>", unsafe_allow_html=True)

# 랜덤 시드
random_list = list(range(1,101))
default_index_random = random_list.index(42) if 42 in random_list else 0
selected_random = st.sidebar.selectbox("랜덤 시드", random_list, index=default_index_random)

# 노이즈
noise_list = list(range(1,5))
default_index_noise = noise_list.index(2) if 1 in noise_list else 0
selected_noise = st.sidebar.selectbox("노이즈 정도", noise_list, index=default_index_noise)

X, y = make_moons(n_samples=1000, noise= 0.15*selected_noise, random_state=selected_random)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Cluster 1')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Cluster 2')
plt.title('Random Data with Noise')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
st.pyplot(plt)

# 은닉층 형태
hidden1_list = list(range(1, 21))
default_index_hidden1 = hidden1_list.index(10) if 10 in hidden1_list else 0
selected_hidden1 = st.sidebar.selectbox("첫번째 은닉층 노드 개수", hidden1_list, index=default_index_hidden1)

hidden2_list = list(range(0, 11))
default_index_hidden2 = hidden2_list.index(0) if 0 in hidden2_list else 0
selected_hidden2 = st.sidebar.selectbox("두번째 은닉층 노드 개수", hidden2_list, index=default_index_hidden2)

if selected_hidden2 == 0:
    mlp = MLPClassifier(solver='lbfgs', activation='relu', hidden_layer_sizes=(selected_hidden1,)).fit(X_train, y_train)
else:
    mlp = MLPClassifier(solver='lbfgs', activation='relu', hidden_layer_sizes=(selected_hidden1, selected_hidden2)).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill = True, alpha= 0.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.title('MLP Classifier')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xticks()
plt.yticks()
plt.legend()
st.pyplot(plt)

from sklearn.metrics import r2_score

y_pred = mlp.predict(X_test)
y_train_pred = mlp.predict(X_train)

# R2 결정 계수
r2 = r2_score(y_test, y_pred)

st.subheader(f'테스트용 데이터에 대한 R2결정계수 : {round(float(r2), 4)}')

# 모델에서 가중치, 편향
weights = mlp.coefs_
biases = mlp.intercepts_

# 입력 데이터
X_sample = X_test[0] 

# 은닉층과 출력층 연산
layer_output = []
layer_output.append(X_sample)  # 입력층 출력

for i in range(len(weights)):
    layer_input = np.dot(layer_output[i], weights[i]) + biases[i]
    if i == len(weights) - 1:
        layer_output.append(1 / (1 + np.exp(-layer_input)))  # 출력층 시그모이드 함수
    else:
        layer_output.append(np.maximum(0, layer_input))  # 은닉층 ReLU 함수

# 출력
for i, output in enumerate(layer_output):
    st.write(f"층 {i} 출력(soft label):", output)

# 모델의 예측 값
y_pred = mlp.predict(X_test)
st.write("모델의 예측 값(hard label):", y_pred[0])

# 인덱스 찾기
sample_index = np.where((X_test == X_sample).all(axis=1))[0][0]

# 실제값 가져오기
actual_value = y_test[sample_index]
st.write("실제값:", actual_value)
'''
st.code(code, language='Python')

