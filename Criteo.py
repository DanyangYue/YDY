#! python 3

import pandas as pd
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn.linear_model import LogisticRegression
from numpy import genfromtxt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

# 训练集与测试集的提取
app_train = pd.read_csv('train.csv')
app_test = pd.read_csv('test.csv')

#读取实际点击情况
submission_path = r"submission.csv"
submission = genfromtxt(submission_path,delimiter=",")
y_true = submission[1:,-1]


#对数据进行处理
#对train集进行处理：删除不需要的类别特征和I12特征,删除ID，提出Label
train_labels = app_train['Label']
train_del=[0,1,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
app_train.drop(app_train.columns[train_del],axis=1,inplace=True)

#对test集进行处理：处理方法同上，提出ID
submit = app_test[['Id']]
test_del = [0,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
app_test.drop(app_test.columns[test_del],axis=1,inplace=True)

train = app_train.copy()
test = app_test.copy()
# print("train:",train)
features = list(train.columns)

#进行数据填充，填充中位数median/平均值mean/众数most_frequent
imputer = Imputer(strategy='mean')
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)


#进行数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

# print("normalize_train:\n",train)
# print("normalize_test:\n",test)
# print('Training data shape: ', train.shape)
# print('Testing data shape: ', test.shape)

# 调用GBDT分类模型。
grd = GradientBoostingClassifier(n_estimators=10)
# 调用one-hot编码。
grd_enc = OneHotEncoder()
# 调用LR分类模型。
grd_lr = LogisticRegression(penalty='l1',C=0.2)
'''使用X_train训练GBDT模型，后面用此模型构造特征'''
grd.fit(train,train_labels)
# fit one-hot编码器
grd_enc.fit(grd.apply(train)[:, :, 0])

''' 
使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
'''
grd_lr.fit(grd_enc.transform(grd.apply(train)[:, :, 0]), train_labels)

log_reg_prob = grd_lr.predict_proba(grd_enc.transform(grd.apply(test)[:, :, 0]))[:,1]

print(log_reg_prob)

#Logarithmic Loss评估函数
def logloss(y_true, y_pred, eps=1e-15):
    import numpy as np

    # Prepare numpy array data
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert (len(y_true) and len(y_true) == len(y_pred))

    # Clip y_pred between eps and 1-eps
    p = np.clip(y_pred, eps, 1-eps)
    loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1-p))

    return loss / len(y_true)

print ("Use log_loss(),the result is",format(logloss(y_true, log_reg_prob)))
submit['Label'] = log_reg_prob
# 输出到文件
submit.to_csv('submit.csv', index = False)