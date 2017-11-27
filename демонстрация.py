import numpy as np
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing import image
import matplotlib.pyplot as plt
from scipy.misc import toimage
get_ipython().magic('matplotlib inline')



# Список классов
classes = ['кот', 'собака']



print("Загружаю сеть из файлов")
# Загружаем данные об архитектуре сети
json_file = open("neuron.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель
loaded_model = model_from_json(loaded_model_json)
# Загружаем сохраненные веса в модель
loaded_model.load_weights("neuron.h5")
print("Загрузка сети завершена")



loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



#img = image.load_img('test/0.jpg', target_size=(224, 224))
img = image.load_img('424642_rotvejler_1680x1050.jpg', target_size=(224, 224))
plt.imshow(img)
plt.show()

nop = np.array([None])
resized_img = np.append(nop, img)

x = image.img_to_array(img)
x=255-x
x/=255
x = np.expand_dims(x, axis=0)



prediction = loaded_model.predict(x)



print(prediction)
print(classes[np.argmax(prediction)])