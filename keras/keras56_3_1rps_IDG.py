from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import time
path = 'd:/study_data/_data/rps/'
save_path = 'd:/study_data/_save/rps/'

datagen = ImageDataGenerator(rescale=1./255)
start = time.time()
rps = datagen.flow_from_directory('d:/study_data/_data/rps/', target_size=(150, 150), batch_size=1000, class_mode='categorical', color_mode='rgb', shuffle=True)

rps_x = rps[0][0]
rps_y = rps[0][1]
print(rps_y)

end = time.time()
rps_x_train, rps_x_test, rps_y_train, rps_y_test = train_test_split(rps_x, rps_y, train_size=0.7, shuffle=True, random_state=123)

print(rps_x_train.shape)
print(rps_x_test.shape)
print(rps_y_train.shape)
print(rps_y_test.shape)

np.save(save_path + 'keras56_rps_x_train.npy', arr=rps_x_train)
np.save(save_path + 'keras56_rps_x_test.npy', arr=rps_x_test)
np.save(save_path + 'keras56_rps_y_train.npy', arr=rps_y_train)
np.save(save_path + 'keras56_rps_y_test.npy', arr=rps_y_test)