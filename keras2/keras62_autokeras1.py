import autokeras as ak
import tensorflow as tf
import time
##########################
#1. 데이터
(x_train, y_train), (x_test,y_test)= \
                            tf.keras.datasets.mnist.load_data()


model = ak.ImageClassifier(
    overwrite=False,   #
    max_trials=2
)
###
#3. 콤파일 훈련
start =time.time()
model.fit(x_train,y_train, epochs =10, validation_split=0.15)
end = time.time()

#4. 평가 예측
y_predict = model.predict(x_test)
results = model.evaluate(x_test,y_test)
print('결과 :', results)
print(' 걸린시간 :',round(end-start, 4))

#5. 최적의 모델 출력
best_model = model.export_model()
print(best_model.summary())

#최적의 모델 저장
path = './_save/autokeras/'
best_model.save(path + "keras62_autokeras1.h5")



