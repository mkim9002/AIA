from tensorflow.keras.applications import ResNet50  #전이학습모델
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
# model = ResNet50(weights= None)
# model = ResNet50(weights='경로')

# path = 'D:/archive/training_set/training_set/dogs/dog.6.jpg'
path = 'C:/Users/bitcamp/Pictures/Saved Pictures/나2.JPG'


img = image.load_img(path, target_size=(224,224))
print(img)

x = image.img_to_array(img)
print("======================= img.img_to_array(img)===========================")
print(x, '\n', x.shape) #(224, 224, 3)
print(np.min(x), np.max(x))  #0.0 255.0

# x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
# print(x.shape)  #(1, 224, 224, 3)

x =np.expand_dims(x, axis=0)
print(x.shape) #(1, 224, 224, 3)

############################ -155 에서 155 사이로 정규화 ##############
print("======================= preprocess_ input(x)  ===========================")

x = preprocess_input(x)
print(x.shape)
print(np.min(x), np.max(x))  #-123.68 151.061

print("==================================================")
x_pred = model.predict(x)
print(x_pred, '\n', x_pred.shape)

print("결과는 :", decode_predictions(x_pred, top=5)[0])

