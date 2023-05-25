from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Path to the image
path = 'D:/horse-or-human/horse-or-human/train/humans/human02-17.png'

# path = 'C:/Users/bitcamp/Pictures/Saved Pictures/나2.JPG'

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
