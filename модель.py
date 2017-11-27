from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np


# Каталог с данными для обучения
train_dir = 'img/train'
# Каталог с данными для проверки
val_dir = 'img/val'
# Каталог с данными для тестирования
test_dir = 'img/test'
# Размеры изображения
img_width, img_height = 224, 224
# Размерность тензора на основе изображения для входных данных в нейронную сеть
input_shape = (img_width, img_height, 3)

# Количество эпох
epochs = 5
# Размер мини-выборки
batch_size = 2
# Количество изображений для обучения
nb_train_samples = 1520
# Количество изображений для проверки
nb_validation_samples = 800
# Количество изображений для тестирования
nb_test_samples = 50


model = Sequential()
#слой свертки
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
#слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#слой преобразования из двумерного в одномерное
model.add(Flatten())
#полносвязный слой
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#выходной слой
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)


scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Точность на тестовых данных: %.2f%%" % (scores[1]*100))

# Для сохранения результатов обученной сети
print("Сохраняем сеть")
# Сохраняем сеть для последующего использования
# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("neuron.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("neuron.h5")
print("Сохранение сети завершено")
