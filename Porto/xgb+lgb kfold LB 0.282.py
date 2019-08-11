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

train = pd.read_csv('Porto/data/train.csv', na_values = -1, nrows=100000) # 이번 커널은 -1 (누락)를 모두 Nan으로 변경해 주고 시작한다.
test = pd.read_csv('Porto/data/test.csv', na_values = -1)

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
    # print( 'before change : {}, after : {}'.format(before, after))
    # 용량이 확실히 줄어들었다.

for c in test.select_dtypes(include=['int64']).columns:
    train[c] = train[c].astype(np.int8)
    test[c] = test[c].astype(np.int8)

print('train.shape : ', train.shape, 'test.shape: ',test.shape)

## Custom OBJ function
def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y))], dtype = np.float) # np.c_[np.array([1,2,3]), np.array([4,5,6])] # c_함수는 두 벡터를 컬럼으로 붙히는 역할
    g = g[np.lexsort((g[:, 2], -1*g[:, 1]))] # lexsort 잘 모르겠네..
    gs = g[:, 0].cumsum().sum() / g[:,0].sum()
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
y = train['target'].values
sub = test['id'].to_frame()
sub['target'] = 0

nround = 2000 # need to change to 2000 (이정도 까지는 학습해도 된다는 거 같다)
kfold = 2    # need to change to 5    (Kfold도 마찬가지로..)

skf = StratifiedKFold(n_splits = kfold, random_state= 0)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_model = xgb.train(params, d_train, nround, watchlist, early_stopping_rounds=100,
                          feval=gini_xgb, maximize=True, verbose_eval=100)
    sub['target'] += xgb_model.predict(xgb.DMatrix(test[features].values),  ntree_limit = xgb_model.best_ntree_limit) / (2 * kfold) # ntree_limit : train에서 사용할 최대 성능 트리

gc.collect()

# lgb
params = {'metric': 'auc', 'learning_rate': 0.01, 'max_depth': 10, 'max_bin': 10, 'objective': 'binary',
          'feature_fraction': 0.8, 'bagging_fraction': 0.9, 'bagging_freq': 10, 'min_data': 500}

skf = StratifiedKFold(n_splits=kfold, random_state=1)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' lgb kfold: {}  of  {} : '.format(i + 1, kfold))
    X_train, X_eval = X[train_index], X[test_index]
    y_train, y_eval = y[train_index], y[test_index]
    lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nround,
                          lgb.Dataset(X_eval, label=y_eval), verbose_eval=100,
                          feval=gini_lgb, early_stopping_rounds=100)
    sub['target'] += lgb_model.predict(test[features].values, num_iteration=lgb_model.best_iteration) / (2 * kfold)

sub.to_csv('sub10.csv', index=False, float_format='%.5f')
gc.collect()

sub.head(2)


import seaborn as sns
sns.distplot(sub['target'])