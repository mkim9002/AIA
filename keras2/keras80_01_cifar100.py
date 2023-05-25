# 10 개 파일 뿌라스로 만들어
# 쉐이프 오류 인것은 내용 명시하고 추가 모델 만들어
# 공동 fully connected layer 구성하지 말고
# GAP 로 바로 출력 떄릴것

# 01. VGG19
# 02. Xception
# 03. ResNet50
# 04. ResNet101
# 05. InceptionV3
# 06. InceptionResNetV2
# 07. DenseNet121
# 08. MobileNetV2
# 09. NASNetMobile
# 10. EfficientNetB0

import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import VGG19, Xception, ResNet50, ResNet101, InceptionV3, InceptionResNetV2, DenseNet121, MobileNetV2, NASNetMobile
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
# from efficientnet.tfkeras import EfficientNetB0

# CIFAR-100 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 데이터 전처리
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

# 모델 선택 및 구성
models = [
    VGG19(weights='imagenet', include_top=False),
    # Xception(weights='imagenet', include_top=False),
    # ResNet50(weights='imagenet', include_top=False),
    # ResNet101(weights='imagenet', include_top=False),
    # InceptionV3(weights='imagenet', include_top=False),
    # InceptionResNetV2(weights='imagenet', include_top=False),
    # DenseNet121(weights='imagenet', include_top=False),
    # MobileNetV2(weights='imagenet', include_top=False),
    # NASNetMobile(weights='imagenet', include_top=False),
    # EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
]

for i, model in enumerate(models):
    # GAP 레이어 추가
    x = GlobalAveragePooling2D()(model.output)
    output = Dense(100, activation='softmax')(x)

    # 새로운 모델 생성
    model = Model(inputs=model.input, outputs=output) 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 학습
    model.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test, y_test))

    # 모델 평가
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print("===================================")

