from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import time
path = 'd:/study_data/_data/men_woman/'
save_path = 'd:/study_data/_save/men_woman/'

datagen = ImageDataGenerator(rescale=1./255)
start = time.time()
men_woman = datagen.flow_from_directory('d:/study_data/_data/men_woman/', target_size=(150, 150), batch_size=1000, class_mode='binary', color_mode='rgb', shuffle=True)

men_woman_x = men_woman[0][0]
men_woman_y = men_woman[0][1]

end = time.time()
print(end - start)

men_woman_x_train, men_woman_x_test, men_woman_y_train, men_woman_y_test = train_test_split(men_woman_x, men_woman_y, train_size=0.7, shuffle=True, random_state=123)

print(men_woman_x_train.shape)
print(men_woman_x_test.shape)
print(men_woman_y_train.shape)
print(men_woman_y_test.shape)

np.save(save_path + 'keras56_men_woman_x_train.npy', arr=men_woman_x_train)
np.save(save_path + 'keras56_men_woman_x_test.npy', arr=men_woman_x_test)
np.save(save_path + 'keras56_men_woman_y_train.npy', arr=men_woman_y_train)
np.save(save_path + 'keras56_men_woman_y_test.npy', arr=men_woman_y_test)