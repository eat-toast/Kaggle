import numpy as np
import matplotlib
from matplotlib import pyplot  as plt
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
# For Saving
from bayes_opt.observer import JSONLogger # 저장 logger를 달아야지만 저장이 된다.
from bayes_opt.event import Events

matplotlib.rc('font', family = 'Malgun Gothic')

iris = datasets.load_iris()
X = iris.data
y = iris.target

def SVM_rbf_cv(gamma, C):
    model = svm.SVC(kernel = 'rbf', gamma=gamma, C = C)
    RMSE = cross_val_score(model, X, y, scoring='accuracy', cv=5).mean()
    return RMSE

# 주어진 범위 사이에서 적절한 값을 찾는다.
pbounds = {'gamma': ( 0.001, 1000 ), "C": (0.001, 1000)}

# 베이지안 옵티마이제이션 객체를 생성
SVM_rbf_BO = BayesianOptimization( f = SVM_rbf_cv, pbounds = pbounds, verbose = 2, random_state = 1 )

# 저장 logger 생성
logger = JSONLogger(path="./SVM_rbf.json")
SVM_rbf_BO.subscribe(Events.OPTMIZATION_STEP, logger)

# 메소드를 이용해 최대화!
SVM_rbf_BO.maximize(init_points=2, n_iter = 10)

SVM_rbf_BO.max # 찾은 파라미터 값 확인

# Loading
from bayes_opt.util import load_logs

new_optimizer = BayesianOptimization( f=SVM_rbf_cv,
                                      pbounds={'gamma': ( 0.01, 10 ), "C": (0.1, 10)},
                                      verbose=2,
                                      random_state=7)

print(len(new_optimizer.space)) # 새로 만든 optimizer 에는 저장된 것이 없다.

load_logs(new_optimizer, logs=["./SVM_rbf.json"])

print( new_optimizer.max == SVM_rbf_BO.max )



def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

fig, sub = plt.subplots(1, 2)
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

model = svm.SVC(kernel = 'rbf', gamma=SVM_rbf_BO.max['params']['gamma'], C = SVM_rbf_BO.max['params']['C'])
clf = model.fit(X[:,:2],y)
plot_contours(sub[0], clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

sub[0].scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
sub[0].set_ylabel('y label here')
sub[0].set_xlabel('x label here')
sub[0].set_title('베이즈 최적화')


model = svm.SVC(kernel = 'rbf', gamma=0.1, C =1 )
clf = model.fit(X[:,:2],y)
plot_contours(sub[1], clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

sub[1].scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
sub[1].set_ylabel('y label here')
sub[1].set_xlabel('x label here')
sub[1].set_title('일반')
plt.show()