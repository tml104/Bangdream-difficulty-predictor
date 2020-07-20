import csv
import numpy as np
from keras.datasets import boston_housing
from keras import layers
from keras import models
from keras.optimizers import sgd
import matplotlib.pyplot as plt

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    iSGD=sgd(learning_rate=0.01)
    model.compile(optimizer=iSGD,loss='mse',metrics=['mae'])
    return model

#加载训练数据
songs=[]
with open('csv_bang/bangdreamsong.csv') as f:
    f_csv=csv.reader(f)
    for row in f_csv:
        songs.append(row)

#提取训练数据与测试数据
a=[x[2:] for x in songs[1:]]
a=np.asarray(a,dtype='float32')
#np.random.shuffle(a)
print(a)

test_data=a[:281,1:]
test_targets=a[:281,0]
train_data=a[281:,1:]
train_targets=a[281:,0]
print('test_data,test_targets,train_data,train_targets SHAPE:',test_data.shape,test_targets.shape,train_data.shape,train_targets.shape)

#数据标准化
mean=train_data.mean(axis=0)
std=train_data.std(axis=0)
train_data-=mean
test_data-=mean
train_data/=std
test_data/=std

#创建模型并训练
num_epochs=300

model=build_model()
history=model.fit(train_data,train_targets,epochs=num_epochs,batch_size=16)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)

#打印测试集损失和平均绝对误差
print('test_mse_score',test_mse_score)
print('test_mae_score',test_mae_score)

#预测测试集结果
#print(model.predict(test_data))

#预测 “ヒトリノ夜” 四个难度的预测结果：
b=np.asarray([
[97,7.63,3.59,174,632,6.5,0.45],
[97,7.07,3.33,174,437,4.5,0.47],
[97,6.65,3.13,174,240,2.5,0.45],
[97,6.35,2.99,174,160,1.6,0.46]
]).astype('float32')
b-=mean
b/=std
print(model.predict(b))


'''
“ヒトリノ夜” 四个难度的参数
[
[97,7.63,3.59,174,632,6.5,0.45],
[97,7.07,3.33,174,437,4.5,0.47],
[97,6.65,3.13,174,240,2.5,0.45],
[97,6.35,2.99,174,160,1.6,0.46]
]

预测结果
[[24.927454]
 [16.998707]
 [13.436136]
 [ 8.783671]]

实际结果
[25,17,14,9]
'''