import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# the model
model = tf.keras.Sequential()
model.add(layers.InputLayer(input_shape=(224, 224, 3)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, (3, 3),  padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3),  padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(120, activation='softmax'))

model.summary()

# using ImageDataGenerator to process the images and generalize the model
# rescale to normalize data between 0 - 1
train_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# for testing we only have to normalize
test_data_gen = ImageDataGenerator(rescale=1./255)

batch_size = 20

# loading training set
train_generator = train_data_gen.flow_from_directory(
    "/dogs/images",
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
    )

# loading testing set
test_generator = test_data_gen.flow_from_directory(
    directory=r"/dogs/testing/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=20,
    class_mode='categorical',
    shuffle=False,
)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# load weights from previous training
model.load_weights("/dogs/weights/first_run.h5")

# for only testing, change epochs to 0
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="/dogs/log")
model.fit_generator(train_generator, callbacks=[tensorboard], epochs=4, steps_per_epoch=921)

# testing the model
score = model.evaluate_generator(test_generator, steps=108)
print('\n', 'Test accuracy:', score[1])

# save model
model.save_weights("/dogs/weights/first_run.h5")

