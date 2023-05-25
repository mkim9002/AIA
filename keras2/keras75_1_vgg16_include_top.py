import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

# model = VGG16() #include_top = True, input_shape=(224, 224,3)
model = VGG16(weights='imagenet', include_top=True,
              input_shape=(224, 224 ,3))

######
model.summary()

print(len(model.weights)) #32
print(len(model.trainable_weights)) #32

##### include_top = True ########

#1. FC layer 원래꺼 쓴다
#2. input_shize=(224,224,3) 고정값, 바꿀수 없다




##### include_top = False  ########
#1. FC layer 원래꺼 쓴다
#2. input_shize=(224,224,3) 고정값, 바꿀수 없다


#  block5_conv1 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_conv2 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_conv3 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0

# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0






