
import shutil
import os

# Каталог с набором данных
data_dir = 'img/all3'
# Каталог с данными для обучения
train_dir = 'img/train'
# Каталог с данными для проверки
val_dir = 'img/val'
# Каталог с данными для тестирования
test_dir = 'img/test'
# Часть набора данных для тестирования
test_data_portion = 0.05 
# Часть набора данных для проверки
val_data_portion = 0.95 
# Количество элементов данных в одном классе
nb_images = 1600


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "cats"))
    os.makedirs(os.path.join(dir_name, "dogs"))


create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)


# функция для копирования изображений
def copy_images(start_index, end_index, source_dir, dest_dir):
    # начальная точка, конечная точка, приращение
    for i in range(start_index+1, end_index+1, 2):
        shutil.copy2(os.path.join(source_dir, str(i) + ".jpg"), os.path.join(dest_dir, "cats"));
    print(start_index, end_index);             
    for i in range(start_index, end_index, 2):
        shutil.copy2(os.path.join(source_dir, str(i) + ".jpg"), os.path.join(dest_dir, "dogs"));
    print(start_index, end_index);     

start_val_data_idx = int(nb_images * (val_data_portion))# 1520
start_test_data_idx = int(nb_images * (test_data_portion)) # 80
print(start_val_data_idx)
print(start_test_data_idx)


# вызов функций копирования
copy_images(0, start_val_data_idx, data_dir, train_dir)
#copy_images(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
copy_images(0, start_val_data_idx, data_dir, val_dir)
copy_images(start_val_data_idx, nb_images, data_dir, test_dir)





