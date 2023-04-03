from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

np.random.seed(0)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
        rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)
augment_size = 40000
# randidx  = np.random.randint(60000,size=40000)
randidx  = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)   #[36536 46210 56951 ... 51744 57631 14037]
print(randidx.shape)  #(40000,)
print(np.min(randidx), np.max(randidx))  #0 59998

x_augmented = x_train[randidx].copy()
y_augmented = x_train[randidx].copy()
print(x_augmented)
print(x_augmented.shape, y_augmented.shape)  #(40000, 28, 28) (40000, 28, 28)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)

# x_augmented = train_datagen.flow(
#     x_augmented,y_augmented, batch_size=augment_size, shuffle=False
# )
# print(x_augmented)  #<keras.preprocessing.image.NumpyArrayIterator object at 0x0000029D763D3B50>
# print(x_augmented[0][0].shape) #(40000, 28, 28, 1)


x_augmented = train_datagen.flow(
    x_augmented,y_augmented, batch_size=augment_size, shuffle=False
).next()[0]

print(x_augmented)
print(x_augmented.shape) #(40000, 28, 28, 1)

print(x_augmented.shape)
print(np.max(x_train), np.min(x_train))
print(np.max(x_augmented),np.min(x_augmented))


x_train = np.concatenate([x_train/255., x_augmented], axis=0)   #concatenate 이어주다
print(x_train.shape, y_train.shape)

# x_train = x_train + x_augmented
# print(x_train.shape)

