import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, f1_score
from sklearn.datasets import make_blobs
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split 

# 데이터 경로
path = "d:/study_data/_data/gas/"
path_save = "d:/study_data/_save/gas/"

# train 데이터
train_csv = pd.read_csv(path + 'train_data.csv', index_col=0)
print(train_csv)
print(train_csv.shape) #(2463, 7)

# test 데이터
test_csv = pd.read_csv(path + 'test_data.csv', index_col=0)
print(test_csv) 
print(test_csv.shape) #(7389, 7)

# 결측치 처리
print('결측치 숫자 : ',train_csv.isnull().sum())  # 결측치 없음

# 데이터 분리
x = train_csv.copy()
scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=2)
pca_x = pca.fit_transform(x)

kmeans = KMeans(n_clusters=3)
kmeans.fit(pca_x)

centers = kmeans.cluster_centers_

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(3):
    ax.scatter(pca_x[kmeans.labels_ == i, 0], pca_x[kmeans.labels_ == i, 1], label=f"Cluster {i+1}")
ax.scatter(centers[:, 0], centers[:, 1], s=100, c='black', label='Centroids')
ax.set_title('KMeans Clustering')
ax.legend()
plt.show()

train_clusters = kmeans.labels_
print(train_clusters)

train_csv['cluster'] = train_clusters
print(train_csv)

# test 데이터 전처리 과정 (생략)
test_x = scaler.transform(test_csv)
test_pca_x = pca.transform(test_x)
test_clusters = kmeans.predict(test_pca_x)

# macro f1 score 계산
macro_f1_score = f1_score(test_csv['type'], test_clusters, average='macro')
print("macro f1 score: ", macro_f1_score)

# 제출 파일 생성
submission = pd.read_csv(path +'answer_sample.csv', index_col='type') # index_col을 'type'으로 변경
submission['label'] = test_clusters
submission.to_csv(path_save+'submit_04.csv')