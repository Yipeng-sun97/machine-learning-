import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN


def KNN_iris():
    iris=load_iris()
    #特征工程标准化
    transfer = StandardScaler()
    data = transfer.fit_transform(iris.data)
    #划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, iris.target, random_state=6)
    #KNN
    estimator = KNN(n_neighbors=3)
    estimator.fit(x_train,y_train)
    #评估
    y_predict = estimator.predict(x_test)
    print('预测值:', y_predict)
    score=estimator.score(x_test, y_test)
    print('准确率:', score)



def KNN_iris_Grid_CV():
    iris=load_iris()
    #特征工程标准化
    transfer = StandardScaler()
    data = transfer.fit_transform(iris.data)
    #划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, iris.target, random_state=6)
    #KNN
    estimator = KNN()
    #加入网格搜索和交叉验证
    param_dict={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)

    estimator.fit(x_train, y_train)
    #评估
    y_predict = estimator.predict(x_test)
    print('预测值:', y_predict)
    score=estimator.score(x_test, y_test)
    print('准确率:', score)
    #最佳参数
    print('best param:', estimator.best_params_)
    print('best score:', estimator.best_score_)
    print('best estimator:', estimator.best_estimator_)
    print('cv result:', estimator.cv_results_)

if __name__=='__main__':
    KNN_iris_Grid_CV()



