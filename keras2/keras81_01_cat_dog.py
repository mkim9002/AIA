#최종목표인 acc 도출
#기존꺼와 전이학습의 성능비교\
#무조건 전이학습이 이겨야 한다!!

#본인 사진 개인지 고양이 인지 구별

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Path to the image
# path = 'D:/archive/training_set/training_set/dogs/dog.6.jpg'
path = 'C:/Users/bitcamp/Pictures/Saved Pictures/나2.JPG'

# Load and preprocess the image
img = image.load_img(path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions using the model
preds = model.predict(x)

# Decode the predictions and print the results
decoded_preds = decode_predictions(preds, top=5)[0]
for pred in decoded_preds:
    print(f"{pred[1]}: {pred[2]*100:.2f}%")
