from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import time
path = 'd:/study_data/_data/horse-or-human/'
save_path = 'd:/study_data/_save/horse-or-human/'

datagen = ImageDataGenerator(rescale=1./255)
start = time.time()
horse_human = datagen.flow_from_directory('d:/study_data/_data/horse-or-human/', target_size=(250, 250), batch_size=1000, class_mode='binary', color_mode='rgb', shuffle=True)

horse_human_x = horse_human[0][0]
horse_human_y = horse_human[0][1]

end = time.time()
print(end - start)

horse_human_x_train, horse_human_x_test, horse_human_y_train, horse_human_y_test = train_test_split(horse_human_x, horse_human_y, train_size=0.7, shuffle=True, random_state=123)

print(horse_human_x_train.shape)
print(horse_human_x_test.shape)
print(horse_human_y_train.shape)
print(horse_human_y_test.shape)

np.save(save_path + 'keras56_horse_human_x_train.npy', arr=horse_human_x_train)
np.save(save_path + 'keras56_horse_human_x_test.npy', arr=horse_human_x_test)
np.save(save_path + 'keras56_horse_human_y_train.npy', arr=horse_human_y_train)
np.save(save_path + 'keras56_horse_human_y_test.npy', arr=horse_human_y_test)