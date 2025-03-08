import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL
import scipy

# Путь к папке с изображениями
data_dir = 'Images'

# Размер изображений
img_height, img_width = 224, 224
batch_size = 32

# Генератор для обучения
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% данных для валидации
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

'''
# Загрузка предобученной модели MobileNetV2 без верхних слоев
base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')

# Замораживаем базовую модель
base_model.trainable = False

# Создаем модель
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

epochs = 20

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

model.save('dog_breed_model.h5')
 '''

# Загрузка модели
model = load_model('dog_breed_model.h5')


# Функция для предсказания
def predict_breed(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return train_generator.class_indices, predicted_class


# Пример использования
img_path = '1.jpg'
class_indices, predicted_class = predict_breed(img_path)
print(f"Predicted class: {list(class_indices.keys())[predicted_class[0]]}")
