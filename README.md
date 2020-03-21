# South_Sea_SST_ML
남해의 여섯달 뒤에 수온을 예측 해보자


## 연구 방법
1. 인도양과 동태평양의 기압 차이(x1)과 인도양과 동태평양의 수온 차이(x2), Ucompoent(x3)를 매개변수로 하여 수온 예측값을 구하여 보자.
2. 비교 시기는 적도 태평양이 한국에 영향을 미칠 것이라 예상되는 6달 뒤로.

## 데이터 획득
1. NOAA에서 제공되는 ICOADS-1 위성 수치자료 획득.
2. Degree 1로 Resolution이 굉장히 넓다.
* 차후 이미지 처리를 이용하면 좋을 것 같음

## 데이터 전처리

YEAR 연도
MONTH 월
KS_SST 대한해협 수온
EQ_SOI 인도양과 동태평양의 기압차이
UWND 동서방향 바람의 세기
SouthSea 남해 수온
PAC_IND_SST_DIFF 인도양과 동태평양 수온 차이
EAST_PACIFIC 동태평양 수온
INDIA 인도양 수온
KS_Anomaly 대한해협 수온 아노말리
SouthSEA 남해 수온 아노말리
EastSea 동해 수온 아노말리

1. 2차원 배열로 정리함
2. 학습데이터 결과데이터는 7:3 비율로 나눔
3. 활성화 함수는 시그모이드, 출력함수는 소프트맥스

## 결과

대한해협 손실함수
![대한해협](./Result/Korean_strait_loss_fig.png)
Score: 0.78


남해 손실합수
![남해](./Result/South_sea_loss_fig.png)
Score: 0.83

## Tensorflow 이용
모델 정의. 
손실함수 : 크로스 엔트로피
최적화 알고리즘 : 경사하강법
은닉층 100개, 출력층 3개
* 출력층 정의 : 2 - 수온 매우 높음, 1 - 수온 높음, 0 - 수온 예상 변동 없음, -1 - 수온 하강 예상

대한해협 손실함수
![Tensor_KS](./Result/KS_Tensorflow.png)
Val_accuracy : 0.78

남해 손실함수
![Tensor_SS](./Result/SS_Tensorflow.png)
Val_accuracy : 0.83

## 고쳐야 될 점
1. 매개변수를 조절하면서 해보자.. (바람변수를 뺀다던가)
2. 도드라지는 변화가 없어서 사실 맞추기 어렵다..
3. 그래도 자연과학에서 83%면 높은 것이 아닐까?

## 시계열 데이터
1. RNN을 이용해보자
2. LSTM 모델을 이용해보자
