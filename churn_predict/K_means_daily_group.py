import copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Hyper parameter
num_cluster = 3
np.random.seed(2)

train  = pd.DataFrame(data = {'datekey': ['2019-07-18', '2019-07-18', '2019-07-18', '2019-07-18', '2019-07-18', '2019-07-18', '2019-07-18', '2019-07-18'],
                               'nid' : range(8),
                               'x1' : np.random.randint(high=10, low = 0, size = 8),
                               'x2' : np.random.randint(high=10, low = 0, size = 8),
                               })

train.datekey = pd.to_datetime(train.datekey) # datetime 형식으로 변환 

train_day2 = copy.deepcopy(train) # day2 데이터 생성
train_day2.datekey = train_day2.datekey + pd.Timedelta(days = 1) # datetime 형식 변환 + add day1

# 하루를 차이로 변수가 +1 씩 변경되었을 때, K-means의 군집 매칭 시키기 (Day1 <=> Day2)
features = ['x1', 'x2']
train_day2[features] = train[features] +1

kmeans_day1 = KMeans(n_clusters=num_cluster, random_state=1).fit(train[features]) # Day1 군집
label_day1 = kmeans_day1.labels_
train['cluster'] = label_day1

def day_day_cluster(day_n_df, features):
   """
   :param day1_df:  기준이 되는 첫날 df
   :param day_n_df:  첫날과 비교하는 df
   :param features:  사용할 feature 모음
   :return: day_n_df + cluster 정보
   """
    global kmeans_day1
    global train

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
    #return train

day_day_cluster(day_n_df= train_day2, features = features)
print(train.shape)

#
train_day3 = copy.deepcopy(train_day2)
train_day3.datekey = train_day3.datekey + pd.Timedelta(days = 1)

day_day_cluster(day_n_df= train_day3, features = features)
print(train.shape)
