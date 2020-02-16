#사용자 정의 클래스
from Util.Util import Util
from Layer.MultiClassNetWork import *
#외부 프레임워크
import tensorflow as tf
from sklearn.model_selection import train_test_split

#TensorFlow model 이용, CNN
from tensorflow.keras import layers

import matplotlib.pyplot as plt

util = Util()

df = util.get_file("C:/Users/sunng/PycharmProjects/South_Sea_SST_ML/Raw_data.csv")

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
y1 = Ks_result
y2 = ss_result

## 다중분류니까 Tensorflow를 이용하자... 원핫인코딩에 사용

# 학습데이터, 결과데이터 나누기, 7:3 비율
x_train, x_val, y_train, y_val = train_test_split(x, y1, stratify=y1,
                                                  test_size=0.2, random_state=42)

#원-핫-인코딩
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)
#END PREPROCESSING
###########################################################################

###########################################################################
#Modeling
model = tf.keras.Sequential()
## 은닉층을 정의함. 유닛개수 100개, input_shape : 샘플수를 제외한 입력형태를 정의함.
model.add(layers.Dense(100, activation='sigmoid', input_shape=(367,)))
## 출력층을 정의함. 유닛수는 10개
model.add(layers.Dense(10, activation='softmax'))

## 최적화알고리즘은 sgd(경사하강법), 손실함수는 크로스엔트로피, metrics : 훈련과정기록으로 정확도를 남기기 위함.
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train, y_train_encoded, epochs=30,
                    validation_data=(x_val, y_val_encoded))

###########################################################################