< Dataset >
test : test에 이용할 데이터
train : train에 이용할 데이터
activity_labels.txt : Labeling index

body_acc_x_train : 중력가속도를 포함한 몸의 가속도 데이터 (x,y,z 공통)
body_gyro_x_train : calibration 되어있는 각속도 데이터 (x,y,z 공통)
removeG_acc_x_train : 중력가속도를 제거한 몸의 가속도 데이터 (x,y,z 공통)
y_train : train에 이용할 정답 Label

body_acc_x_test : 중력가속도를 포함한 몸의 가속도 데이터 (x,y,z 공통)
body_gyro_x_test : calibration 되어있는 각속도 데이터 (x,y,z 공통)
removeG_acc_x_test : 중력가속도를 제거한 몸의 가속도 데이터 (x,y,z 공통)
y_test : test에 이용할 정답 Label

< 프로그램 구동방법 >
1. config.py -> BASE_ADDRESS를 압축을 푼 절대경로로 설정 ( 문자열 마지막에 역슬래스 2번을 해줘야 정상 작동 )
2. loader.py -> y_train, y_test 각각 파일의 절대경로 지정