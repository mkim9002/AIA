import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
path = 'd:/study_data/_data/dog_breed/dog_v1/'
save_path = 'd:/study_data/_save/dog_breed/'
datagen = ImageDataGenerator(rescale= 1./255) #스케일러 해준거.

start = time.time()
dog_breed = datagen.flow_from_directory(path,
            target_size=(150,150),
            batch_size=3000,
            class_mode='categorical',
            color_mode= 'rgba',
            shuffle= True)

dog_breed_x = dog_breed[0][0]
dog_breed_y = dog_breed[0][1]

end = time.time()
print(end - start, 2)
dog_breed_x_train, dog_breed_x_test, dog_breed_y_train, dog_breed_y_test = train_test_split(
    dog_breed_x, dog_breed_y, train_size= 0.7, shuffle= True, random_state=1557
)

print(dog_breed_x_train.shape) #(721, 150, 150, 4)
print(dog_breed_x_test.shape)  #(309, 150, 150, 4)
print(dog_breed_y_train.shape) #(721, 5)
print(dog_breed_y_test.shape)  #(309, 5)

np.save(save_path + 'keras56_dog_breed_x_train.npy', arr = dog_breed_x_train)
np.save(save_path + 'keras56_dog_breed_x_test.npy', arr = dog_breed_x_test)
np.save(save_path + 'keras56_dog_breed_y_train.npy', arr = dog_breed_y_train)
np.save(save_path + 'keras56_dog_breed_y_test.npy', arr = dog_breed_y_test)