# 모듈 불러오기
from bayes_opt import BayesianOptimization
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np
# For Saving
from bayes_opt.observer import JSONLogger # 저장 logger를 달아야지만 저장이 된다.
from bayes_opt.event import Events

# seed 고정!
np.random.seed(0)
n_samples, n_features = 100, 1 # 100개의 데이터와 1개의 변수 생성

X = np.random.randn(n_samples, n_features) # shape = (100 , 1)
y = np.random.randn(n_samples) # shape = (100 , )

# CV 를 이용한, Ridge 파라미터 찾기
def Ridge_cv(alpha):
    '''
    :param alpha: Ridge's 하이퍼 파라미터
    :return: -RMSE --> 최대화를 위해 음수 부호를 붙힘
    '''

    RMSE = cross_val_score(Ridge(alpha=alpha), X, y, scoring='neg_mean_squared_error', cv=5).mean()

    return -RMSE

# 파라미터를 탐색할 공간
# Ridge는 0 ~ 10 사이에서 적절한 값을 찾는다.
pbounds = {'alpha': ( 0, 10 )}

# 베이지안 옵티마이제이션 객체를 생성
Ridge_BO = BayesianOptimization( f = Ridge_cv, pbounds  = pbounds , verbose=2, random_state=1 )

# 저장 logger 생성
logger = JSONLogger(path="./Ridge.json") # 이전 point 는 바라보지 않는다.
Ridge_BO.subscribe(Events.OPTMIZATION_STEP, logger) # 하나만 저장하는 logger를 전체 저장하는 것 같다.

# 메소드를 이용해 최대화!
Ridge_BO.maximize(init_points=2, n_iter = 10)

Ridge_BO.max # 찾은 파라미터 값 확인

# Loading
from bayes_opt.util import load_logs

new_optimizer = BayesianOptimization( f=Ridge_cv,
                                      pbounds={"alpha": (0, 10)},
                                      verbose=2,
                                      random_state=7)

print(len(new_optimizer.space)) # 새로 만든 optimizer 에는 저장된 것이 없다.

load_logs(new_optimizer, logs=["./Ridge.json"])

print( new_optimizer.max == Ridge_BO.max )
