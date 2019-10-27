import pandas as pd
import numpy as np


train = pd.DataFrame(data = {'datekey': ['2019-07-18', '2019-07-19', '2019-07-21', '2019-07-22', '2019-07-23', '2019-07-24', '2019-07-25', '2019-07-18'],
                             'nid' : [1, 1, 1, 1, 1, 1, 1, 2],
                             'x1' : [10, 20, 30, 40, 50, 60, 70, 80]})

# train에 있는 datekey를 date 형식으로 변환하기
train['datekey'] = train['datekey'].apply(lambda x : pd.to_datetime(str(x), format = '%Y-%m-%d'))


# nid별 최소 접속일, 최대 접속일 구하기.
nid_min_datkey = pd.DataFrame( data = train.groupby('nid')['datekey'].min() ) ; nid_min_datkey.reset_index(level=0, inplace=True)
nid_max_datkey =pd.DataFrame( train.groupby('nid')['datekey'].max() ) ; nid_max_datkey.reset_index(level=0, inplace=True)


nid_date_range = pd.merge(nid_min_datkey, nid_max_datkey, how = 'inner', on='nid' )
nid_date_range .columns = ['nid', 'min_datekey', 'max_datekey']

# nid별 최소 최대 datekey를 이용해 비어 있는 날짜 추가하기
nid_list = list(set(train.nid))

df = train[train.nid == nid_list[0] ]
df.set_index('datekey', inplace = True, drop= True)
df = df.reindex( pd.date_range(start = '2019-07-18', end = '2019-07-25'))

df.loc[df.nid.isnull(), 'x1'] = 0
df.loc[df.nid.isnull(), 'nid'] = nid_list[0]

# Y값 계산하기
df = df.rename_axis('datekey').reset_index()
df['y'] = True
window = 3

# 1. row수가 window보다 작으면 모두 False 부여
nrow = df.shape[0]
if nrow  < window :
    df['y'] = False
else:
    df.iloc[:(nrow - 2), ]['y'] = True
    df.iloc[(nrow - 2):, ]['y'] = False
