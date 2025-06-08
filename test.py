from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


# Inicialização do modelo treinado salvo
modelo = load_model('modelo_cnn.keras')

# ----------------------------- Predições com o conjunto de teste --------------------------------

# Pre-processamento das imagens no conjunto de teste 
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'data/test_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # importante para manter a ordem dos rótulos
)


# Metrícas de avaliação gerais
loss, accuracy = modelo.evaluate(test_set)
print(f'Acurácia no conjunto de teste: {accuracy:.4f}')
print(f'Loss no conjunto de teste: {loss:.4f}')

# Predição do conjunto de teste
y_pred_probs = modelo.predict(test_set)
y_pred = (y_pred_probs > 0.5).astype(int)

y_true = test_set.classes

print(classification_report(y_true, y_pred, target_names=test_set.class_indices.keys()))

#----------------------------- Predições com imagens especificas ------------------------------

# carregar imagem para predição
img = image.load_img('images/gato3.jpeg', target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# predição
pred = modelo.predict(img_array)
print("Predição:", pred, "Classe:", "Cachorro" if pred > 0.5 else "Gato")
