# 이 커널은 xgb와 lgb를 kfold에 적용한 예시.

#  나중에 EDA와 섞어서 실험해보면 좋겠다.
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc

pd.set_option('display.max_columns', 200)
print('loading files...')

train = pd.read_csv('Porto/data/train.csv', na_values = -1, nrows=1000) # 이번 커널은 -1 (누락)를 모두 Nan으로 변경해 주고 시작한다.
test = pd.read_csv('Porto/data/test.csv', na_values = -1, nrows = 1000)

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')] # 왜 calc변수는 제거하지?
train = train.drop(col_to_drop , axis = 1)
test = test.drop(col_to_drop , axis = 1)

for c in train.select_dtypes(include = ['float64']).columns:
    # select_dtypes 은 DF 중 원하는 타입의 정보만 가져 올 수 있다.
    # float64 형식의 컬럼만 가져온다.
    before = train.memory_usage(index = True).sum()
    train[c] = train[c].astype(np.float32)
    test[c] = test[c].astype(np.float32)
    after = train.memory_usage(index = True).sum()

    # 이번주에 데이터 용량 줄여주는 코드를 본적 있는데 그거와 연관이 있을까?
    print( 'before change : {}, after : {}'.format(before, after))
    # 용량이 확실히 줄어들었다.

for c in train.select_dtypes(include=['int64']).columns:
    train[c] = train[c].astype(np.int8)
    test[c] = test[c].astype(np.int8)

print('train.shape : ', train.shape, 'test.shape: ',test.shape)

## Custom OBJ function
def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arrange(len(y))], dtype = np.float) # np.c_[np.array([1,2,3]), np.array([4,5,6])] # c_함수는 두 벡터를 컬럼으로 붙히는 역할
    g = g[np.lexsort((g[:, 2], -1*g[:, 1]))] # lexsort 잘 모르겠네..
    gs = g[:, 0].colsum().sum() / g[:,0].sum()
    gs -= (len(y)+1) / 2
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y,y)

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y,y)
    return 'gini', score, True


## XGB
params = {'eta' : 0.02 ,'max_depth' : 4, 'subsample' : 0.9, 'colsample_bytree' : 0.9,
          'objective' : 'binary:logistic', 'eval_metric':'auc', 'silent' : True}

X = train.drop(['id', 'target'], axis = 1)
features = X.columns
X = X.values
sub = test['id'].to_frame()
sub['target'] = 0

nround = 200 # need to change to 2000 (이정도 까지는 학습해도 된다는 거 같다)
kfold = 2    # need to change to 5    (Kfold도 마찬가지로..)




