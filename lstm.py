'''
Class : Make LSTM Model 
Writer : Cho jun ho
Last Modified Date : 21.07.29
Modification details : Add Comments and Modify Final
'''

#################### Import Library ####################
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from loader import Loader
import numpy as np
import random
########################################################

#################### Tensorflow Set ####################
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
########################################################


def run():
    data = Loader() # Data load

    # Data shuffle
    ##############################################################
    tmp = [ [x,y] for x, y in zip(data.x_train, data.y_train)]
    random.shuffle(tmp)
    data.x_train=[n[0] for n in tmp]
    data.y_train=[n[1] for n in tmp]
    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)
    ##############################################################
    


    # Train
    ##############################################################
    epochs = 20     # 총 학습 횟수
    batch_size = 32 # 1회 학습당 학습시킬 데이터 갯수
    n_hidden = 264  # 데이터셋이 적어 hidden layer를 deep하게 설계
    pv = 0.1        # 너무 많은 drop out은 학습률을 저하시킴

    model = Sequential()
    model.add(LSTM(n_hidden, input_shape=(132, 9))) # 1번 학습당 9개종류의 데이터가 132개씩 input
    model.add(BatchNormalization())
    model.add(Dropout(pv))
    model.add(Dense(6, activation='softmax'))       # 6가지 종류를 분류해야하므로 activation 함수 softmax 사용
    model.compile(loss='categorical_crossentropy',  # 다중 분류 loss 함수 categorical_crossentropy
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(x=data.x_train, y=data.y_train,       # 학습시작
                epochs=epochs, 
                batch_size=batch_size, 
                validation_data=(data.x_test,data.y_test))
    model.summary()                                 # 모델 summary 출력
    ##############################################################

    # Check Score
    ##############################################################
    score = model.evaluate(data.x_test, data.y_test)    # test data를 통한 validation 진행
    print(score)                                        # 결과 출력
    ##############################################################


if __name__ == '__main__':
    run()