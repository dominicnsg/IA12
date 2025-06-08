from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from sklearn.metrics import classification_report
import os

# Configurações
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATA_DIR = 'data'
MODEL_PATH = 'dogs_vs_cats_model.keras'

def evaluate_model():
    model = load_model(MODEL_PATH)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test_set'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    loss, accuracy = model.evaluate(test_set)
    print(f'\nAcurácia no Teste: {accuracy:.4f}')
    print(f'Loss no Teste: {loss:.4f}\n')
    
    y_pred = (model.predict(test_set) > 0.5).astype(int)
    print(classification_report(
        test_set.classes,
        y_pred,
        target_names=test_set.class_indices.keys()
    ))

def predict_image(image_path):
    model = load_model(MODEL_PATH)
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]
    class_name = 'Cachorro' if pred > 0.5 else 'Gato'
    confidence = pred if pred > 0.5 else 1 - pred
    
    print(f'\nImagem: {os.path.basename(image_path)}')
    print(f'Predição: {class_name} (Confiança: {confidence:.2%})')

if __name__ == '__main__':
    evaluate_model()  # Avaliação com o conjunto de teste
    predict_image('images/gato1.jpg')  
    predict_image('images/gato2.jpg')
    predict_image('images/gato3.jpg') 
    predict_image('images/cachorro1.jpg')
    predict_image('images/cachorro2.jpg')
    predict_image('images/cachorro3.jpg')