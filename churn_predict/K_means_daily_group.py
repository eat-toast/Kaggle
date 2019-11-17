import copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import itertools
# Hyper parameter
num_cluster = 3
np.random.seed(10)
input_day = 3
cluster_list = list( range(input_day ) )


# 군집 중심 정보 저장
temp_center = list(itertools.chain.from_iterable(itertools.repeat(i, num_cluster ) for i in cluster_list ))
train_center = pd.DataFrame(data = {'day': temp_center }, index = range(num_cluster * 3))
train_center['x1_center'] = 0
train_center['x2_center'] = 0
train_center['cluster'] = list(range(num_cluster)) * input_day


train  = pd.DataFrame(data = {'datekey': ['2019-07-18'] * 100,
                               'nid' : range(100),
                               'x1' : np.random.normal(30, 1, size = 100),
                               'x2' : np.random.normal(30, 1, size = 100),
                               })

train.datekey = pd.to_datetime(train.datekey) # datetime 형식으로 변환

train_day2 = copy.deepcopy(train) # day2 데이터 생성
train_day2.datekey = train_day2.datekey + pd.Timedelta(days = 1) # datetime 형식 변환 + add day1
train_day2.x1 = np.random.normal(50, 1, size = 100)
train_day2.x2 = np.random.normal(50, 1, size = 100)


train_day3 = copy.deepcopy(train_day2) # day3 데이터 생성
train_day3.datekey = train_day3.datekey + pd.Timedelta(days = 1) # datetime 형식 변환 + add day1
train_day3.x1 = np.random.normal(10, 5, size = 100)
train_day3.x2 = np.random.normal(10, 5, size = 100)

# Scale 맞춰주기
def zero_to_one(x):
    max_x = max(x)
    min_x = min(x)

    return ( x - min_x ) / (max_x - min_x)

train[['scale_x1', 'scale_x2']] = train[['x1', 'x2']].apply(lambda x : zero_to_one(x))
train_day2[['scale_x1', 'scale_x2']] = train_day2[['x1', 'x2']].apply(lambda x : zero_to_one(x))
train_day3[['scale_x1', 'scale_x2']] = train_day3[['x1', 'x2']].apply(lambda x : zero_to_one(x))

# 하루를 차이로 변수가 +1 씩 변경되었을 때, K-means의 군집 매칭 시키기 (Day1 <=> Day2)
features = ['scale_x1', 'scale_x2']
# train_day2[features] = train[features] +1

kmeans_day1 = KMeans(n_clusters=num_cluster, random_state=1).fit(train[features]) # Day1 군집
label_day1 = kmeans_day1.labels_
train['cluster'] = label_day1

train_center.loc[train_center.day == 0, ['x1_center', 'x2_center'] ] = kmeans_day1.cluster_centers_

def day_day_cluster(day_n_df, features):
   """
   :param day1_df:  기준이 되는 첫날 df
   :param day_n_df:  첫날과 비교하는 df
   :param features:  사용할 feature 모음
   :return: day_n_df + cluster 정보
   """
   # 군집비교는 cluser 수만큼 동시에 비교하여야 한다. 2019-11-04

    global kmeans_day1, train

    kmeans_day_n = KMeans(n_clusters=num_cluster, random_state=0).fit(day_n_df[features]) # Day_n 군집
    label_day_n = kmeans_day_n.labels_


    next_cluster = []
    for i in range(num_cluster):
        temp = []
        for j in range(num_cluster):
            x = np.array( [ kmeans_day1.cluster_centers_[i, :] ]) # Day1 군집의 중심
            y = np.array( [ kmeans_day_n.cluster_centers_[j, :] ]) # Day2 군집의 중심

            temp.append( cosine_similarity(x, y)[0][0] ) # Cos 유사도

        next_cluster.append( np.argmax(temp) ) # 다음날 k-means 군집 번호

    # Day1 기준으로 Day(n) 까지 군집 나열하기
    day_n_df['cluster'] = label_day_n
    day_n_df['cluster'].replace(range(num_cluster), next_cluster, inplace = True)

    # Day1 Day_n 데이터 합치기
    train = pd.concat([train, day_n_df], axis = 0)

    return kmeans_day_n.cluster_centers_


train_center.loc[train_center.day == 1, ['x1_center', 'x2_center'] ] = day_day_cluster(day_n_df= train_day2, features = features)
print(train.shape)

#
train_center.loc[train_center.day == 2, ['x1_center', 'x2_center'] ] = day_day_cluster(day_n_df= train_day3, features = features)
print(train.shape)

import seaborn as sns
import matplotlib.pyplot as plt

# 중심점 집합

f, ax = plt.subplots(nrows = 1, ncols = 4)

sns.scatterplot(x = 'scale_x1', y = 'scale_x2', data = train.loc[train.datekey == '2019-07-18'] , hue = 'cluster', s = 100
                , style = 'cluster' , ax = ax[0] )

sns.scatterplot(x = 'scale_x1', y = 'scale_x2', data = train.loc[train.datekey == '2019-07-19'] , hue = 'cluster', s = 100
                , style = 'cluster' , ax = ax[1] )

sns.scatterplot(x = 'scale_x1', y = 'scale_x2', data = train.loc[train.datekey == '2019-07-20'] , hue = 'cluster', s = 100
                , style = 'cluster' , ax = ax[2] )


sns.scatterplot(x = 'scale_x1', y = 'scale_x2', data = train , hue = 'cluster', s = 100
                , style = 'cluster' , ax = ax[3] )
ax[1].set_ylim(0, 1)
ax[1].set_xlim(0, 1)
ax[2].set_ylim(0, 1)
ax[2].set_xlim(0, 1)
# 군집별 특성 확인

# Day1 - 군집1
train.loc[train.datekey == '2019-07-18'].groupby('cluster').scale_x1.mean()
train.loc[train.datekey == '2019-07-19'].groupby('cluster').scale_x1.mean()
train.loc[train.datekey == '2019-07-20'].groupby('cluster').scale_x1.mean()