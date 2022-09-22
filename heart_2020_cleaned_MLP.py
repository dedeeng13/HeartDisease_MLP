#!/usr/bin/env python
# -*- coding=utf-8 -*-
# 增加筆數看看
__author__ = "DDENG"

"""
 資料來源：kaggle
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

"""
import xlrd
import xlwt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
# from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import model_from_json
import seaborn as sns
import tensorflow as tf


#read data
df = pd.read_csv('heart_2020_cleaned_new.csv')
print(df.head())

#############
# ---資料拆切
# ---決定X 分類 和Y分類 要用的欄位
dfX=df[['Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','Diabetic','PhysicalActivity','GenHealth','SleepTime','Asthma','KidneyDisease','SkinCancer']]
dfY=df['HeartDisease']


# ---均一化
dfX = (dfX - dfX.min()) / (dfX.max() - dfX.min())


X=dfX.to_numpy()
Y=dfY.to_numpy()
X_train ,X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size=0.1)

# ---換變數名稱
x_train = X_train[:, :]
x_test =X_test[:, :]
y_train  = Y_train
y_test = Y_test


dim=x_train.shape[1]
category=2
t=2
# ---one-hotEconding
y_train2=tf.keras.utils.to_categorical(y_train, num_classes=(category))
y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))


# ---讀取模型架構
try:
    with open('model_HeartDisease.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # ---讀取模型權重
        model.load_weights("model_HeartDisease.h5")
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy'])
except IOError:
    # ---建立模型
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=200,
        activation=tf.nn.relu,
        input_dim=dim))
    model.add(tf.keras.layers.Dense(units=40*t,
        activation=tf.nn.relu ))
    model.add(tf.keras.layers.Dense(units=60*t,
        activation=tf.nn.relu ))
    model.add(tf.keras.layers.Dense(units=80*t,
        activation=tf.nn.relu ))
    model.add(tf.keras.layers.Dense(units=category,
        activation=tf.nn.softmax ))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy'])
    model.fit(x_train, y_train2,
              epochs=500*t,
              batch_size=64)

# ---測試
model.summary()

score = model.evaluate(x_test, y_test2, batch_size=64)
print("score:",score)

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])



# ---保存模型架構
with open("model_HeartDisease.json", "w") as json_file:
   json_file.write(model.to_json())
# ---保存模型權重
model.save_weights("model_HeartDisease.h5")


 




# ---畫seaborn 圖
# 刪除身高體重
dfZ=df[['Smoking','AlcoholDrinking','Stroke','DiffWalking','Diabetic','PhysicalActivity','GenHealth','SleepTime','Asthma','KidneyDisease','SkinCancer','HeartDisease']]
sns.set_theme(style="ticks")
sns.pairplot(dfZ, hue="HeartDisease")
plt.savefig("seaborn_HeartDisease.jpg")
# plt.show()




