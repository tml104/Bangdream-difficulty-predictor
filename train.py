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
    iSGD=sgd(learning_rate=0.001)
    model.compile(optimizer=iSGD,loss='mse',metrics=['mae']) #rmsprop sgd
    return model

def smooth_curve(points,factor=0.9):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous=smoothed_points[-1]
            smoothed_points.append(previous*factor+(1-factor)*point)
        else:
            smoothed_points.append(point)
    return smoothed_points

songs=[]
with open('csv_bang/bangdreamsong.csv') as f:
    f_csv=csv.reader(f)
    for row in f_csv:
        songs.append(row)

#print(songs)
#第一行描述各个列代表什么
#第二行开始：Rank	Title	R	Time	Score	Eff	BPM	N	NPS	SR
#Rank、Title不用，R作为输出，后面的全部作为输入

#a=np.asarray(songs[1:][2:],dtype='float32')
a=[x[2:] for x in songs[1:]]
a=np.asarray(a,dtype='float32')
#np.random.shuffle(a)
print(a)

test_data=a[:281,1:]
test_targets=a[:281,0]
train_data=a[281:,1:]
train_targets=a[281:,0]
print('test_data,test_targets,train_data,train_targets SHAPE:',test_data.shape,test_targets.shape,train_data.shape,train_targets.shape)

mean=train_data.mean(axis=0)
std=train_data.std(axis=0)
train_data-=mean
test_data-=mean
train_data/=std
test_data/=std

#K
k=4 
num_val_samples=len(train_data)//k
num_epochs=500
#all_score=[]
all_mae_history=[]

for i in range(k):
    print('processing fold #',i)
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data=np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets=np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)

    model=build_model()
    history=model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=1,validation_data=(val_data,val_targets))

    #val_mse,val_mae=model.evaluate(val_data,val_targets)
    #all_score.append(val_mae)

    mae_history=history.history['val_mae']
    all_mae_history.append(mae_history)

average_mae_history=[np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]


plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

smooth_mae_history=smooth_curve(average_mae_history[10:])
plt.clf()
plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()