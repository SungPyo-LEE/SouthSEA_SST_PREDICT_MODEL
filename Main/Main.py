#사용자 정의 클래스
from Util.Util import Util
from Layer.MultiClassNetWork import *

#외부 프레임워크
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import sys

util = Util()

df = util.get_file(Path+"/Raw_data.csv")

#print(df)

#print(df.keys())


###########################################################################
#Index(['YEAR', 'MONTH', 'KS_SST', 'EQ_SOI', 'UWND', 'SouthSea',
#       'PAC_IND_SST_DIFF', 'EAST_PACIFIC', 'INDIA', 'KS_Anomaly', 'SouthSEA',
#       'EastSea'],
#      dtype='object')
#YEAR 연도
#MONTH 월
# KS_SST 대한해협 수온
# EQ_SOI 인도양과 동태평양의 기압차이
# UWND 동서방향 바람의 세기
# SouthSea 남해 수온
# PAC_IND_SST_DIFF 인도양과 동태평양 수온 차이
# EAST_PACIFIC 동태평양 수온
# INDIA 인도양 수온
# KS_Anomaly 대한해협 수온 아노말리
# SouthSEA 남해 수온 아노말리
# EastSea 동해 수온 아노말리
###########################################################################


###########################################################################
#Data PreProcessing
#대한해협 y값 형성, //-1, 0 ,1 ,2//
ks_ano = util.convert_list(df, 'KS_Anomaly')
Ks_result = util.high_or_low(ks_ano)
##

#남해 y값 형성, //-1, 0 ,1 ,2//
ss_ano = util.convert_list(df, 'SouthSEA')
ss_result = util.high_or_low(ss_ano)
##

# 매개변수를 3개로 하는 다중분류 문제, 해결을 위해 신경망 사용

##학습데이터 구축.
# 매개변수 가져오기
m = len(ks_ano)
x = []
eq_soi = util.convert_list(df, 'EQ_SOI')
uwnd = util.convert_list(df, 'UWND')
sst_diff = util.convert_list(df, 'PAC_IND_SST_DIFF')

#2차원 데이터로 구축
for i in range(m):
    x1=[]
    x1.append(eq_soi[i])
    x1.append(uwnd[i])
    x1.append(sst_diff[i])
    x.append(x1)
x = np.array(x)

# 출력값, 6달 뒤의 대한해협 수온과 남해의 수온
y1 = np.array(Ks_result)
y2 = np.array(ss_result)

## 다중분류니까 Tensorflow를 이용하자... 원핫인코딩에 사용

# 학습데이터, 결과데이터 나누기, 7:3 비율
x_train, x_val, y1_train, y1_val = train_test_split(x, y1, stratify=y1,
                                                  test_size=0.2, random_state=42)

x_train, x_val, y2_train, y2_val = train_test_split(x, y2, stratify=y2,
                                                  test_size=0.2, random_state=42)

#원-핫-인코딩
y1_train_encoded = tf.keras.utils.to_categorical(y1_train)
y1_val_encoded = tf.keras.utils.to_categorical(y1_val)

y2_train_encoded = tf.keras.utils.to_categorical(y2_train)
y2_val_encoded = tf.keras.utils.to_categorical(y2_val)
#END PREPROCESSING
###########################################################################

###########################################################################
#Modeling
## units=100 : 은닉층 뉴런이 10개 // 다중분류 알고리즘
fc = MultiClassNetwork(units=30, batch_size=30)
fc.fit(x_train, y1_train_encoded,
       x_val=x_val, y_val=y1_val_encoded, epochs=40)
###########################################################################

print(fc.score(x_val, y1_val_encoded)) #0.78 성공률

plt.plot(fc.losses)
plt.plot(fc.val_losses)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.legend(['train_loss', 'val_loss'])
plt.show()
#plt.savefig('fig1.png', dpi=300)

#대한해협 수온
##

#Modeling
## units=100 : 은닉층 뉴런이 10개 // 다중분류 알고리즘
fc = MultiClassNetwork(units=30, batch_size=30)
fc.fit(x_train, y2_train_encoded,
       x_val=x_val, y_val=y2_val_encoded, epochs=40)
###########################################################################

print(fc.score(x_val, y2_val_encoded)) #0.82 성공률

plt.plot(fc.losses)
plt.plot(fc.val_losses)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.legend(['train_loss', 'val_loss'])
plt.show()
#plt.savefig('fig1.png', dpi=300)
