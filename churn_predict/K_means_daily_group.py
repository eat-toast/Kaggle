from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import copy
import numpy as np
import pandas as pd

np.random.seed(2)


train_day1  = pd.DataFrame(data = {'datekey': ['2019-07-18', '2019-07-18', '2019-07-18', '2019-07-18', '2019-07-18', '2019-07-18', '2019-07-18', '2019-07-18'],
                               'nid' : range(8),
                               'x1' : np.random.randint(high=10, low = 0, size = 8),
                               'x2' : np.random.randint(high=10, low = 0, size = 8),
                               })
train_day1.datekey = pd.to_datetime(train_day1.datekey)


train_day2 = copy.deepcopy(train_day1)
train_day2.datekey = train_day2.datekey + pd.Timedelta(days = 1)

# 하루를 차이로 변수가 +1 씩 변경되었을 때, K-means의 군집 매칭 시키기 (Day1 <=> Day2)
features = ['x1', 'x2']
train_day2[features] = train_day1[features] +1

kmeans_day1 = KMeans(n_clusters=3, random_state=1).fit(train_day1[features]) # Day1 군집
label_day1 = kmeans_day1.labels_


kmeans_day2 = KMeans(n_clusters=3, random_state=0).fit(train_day2[features]) # Day2 군집
label_day2 = kmeans_day2.labels_

next_cluster = []
for i in range(3):
    temp = []
    for j in range(3):
        x = np.array( [ kmeans_day1.cluster_centers_[i, :] ]) # Day1 군집의 중심
        y = np.array( [ kmeans_day2.cluster_centers_[j, :] ]) # Day2 군집의 중심

        temp.append( cosine_similarity(x, y)[0][0] ) # Cos 유사도

    next_cluster.append( np.argmax(temp) ) # 다음날 k-means 군집 번호

# Day1 기준으로 Day(n) 까지 군집 나열하기
train_day1['cluster'] = label_day1
train_day2['cluster'] = label_day2
train_day2['cluster'].replace(range(3), next_cluster, inplace = True)

# Day1 Day2 데이터 합치기
train = pd.concat([train_day1, train_day2], axis = 0)


train.groupby('nid')['cluster'].nunique()