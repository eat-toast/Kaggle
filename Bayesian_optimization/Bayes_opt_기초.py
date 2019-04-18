import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression

X = np.random.rand(100)
Y = X + np.random.rand(100)*0.1

# reshape
X = X.reshape((-1,1))
Y = Y.reshape((-1,1))

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


def black_box_function(x,y):
    return -x**2 - (y-1) **2 +1

# 2. Getting Started
from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

# 세부 사항 설정
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1
)

# 최대화!! (최소화는 없는 것 같다)
optimizer.maximize(init_points=2, n_iter=5 )
    # n_iter: 반복 횟수 (많을 수록 정확한 값을 얻을 수 있다)
    # init_points: 초기 랜덤 설정 값


# 결과 확인
print(optimizer.max)

optimizer.res # 이전 history 를 확인 할 수 있다.

# 2.1 Changing bounds
optimizer.set_bounds(new_bounds={"x": (-2, 3)})

# 이후 절차는 동일히다.
optimizer.maximize( init_points=0, n_iter=5)

# 3. Guiding the optimization

optimizer.probe( params={"x": -3.0, "y": 2}, lazy=True)

optimizer.probe(
    params=[-0.3, 0.1],
    lazy=True
)

# 4. Saving, loading and restarting

# 4.1 Saving progress
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

logger = JSONLogger(path="./logs.json") # 이전 point 는 바라보지 않는다.
optimizer.subscribe(Events.OPTMIZATION_STEP, logger) # 하나만 저장하는 logger를 전체 저장하는 것 같다.

optimizer.maximize(init_points=2, n_iter = 3)


# 4.2 Loading progress
from bayes_opt.util import load_logs

new_optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={"x": (-2, 2), "y": (-2, 2)},
    verbose=2,
    random_state=7,
)
print(len(new_optimizer.space)) # 새로 만든 optimizer 에는 저장된 것이 없다.

load_logs(new_optimizer, logs=["./logs.json"]);

print("New optimizer is now aware of {} points.".format(len(new_optimizer.space)))
new_optimizer.maximize(
    init_points=0,
    n_iter=10,
)
