from flask import Flask, request, jsonify
import os, io
import numpy as np
from PIL import Image

import mxnet as mx
from mxnet import nd
from gluoncv.model_zoo import get_model  # <- OJO: GluonCV

app = Flask(__name__)

# Clases de CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Contexto (CPU por defecto)
ctx = mx.cpu()

# Modelo preentrenado de GluonCV
net = get_model('cifar_resnet20_v1', classes=10, pretrained=True)
net.collect_params().reset_ctx(ctx)
net.hybridize(static_alloc=True, static_shape=True)

# (Opcional) normalización típica de CIFAR-10
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

def preprocess_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read()))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((32, 32))               # CIFAR-10 es 32x32

    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W,C] en [0,1]
    # Normaliza por canal (opcional pero recomendado si el modelo lo espera)
    arr = (arr - CIFAR10_MEAN) / CIFAR10_STD
    arr = arr.transpose(2, 0, 1)             # [C,H,W]
    nd_arr = nd.array(arr).expand_dims(0)    # (1,C,H,W)
    return nd_arr

@app.route('/')
def home():
    return "¡API del clasificador de imágenes Kubermatic DL en ejecución!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'img' not in request.files:
        return jsonify({'error': 'No se proporcionó imagen'}), 400

    file = request.files['img']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    x = preprocess_image(file).as_in_context(ctx)

    with mx.autograd.predict_mode():
        logits = net(x)
        probs = nd.softmax(logits)[0]
        ind = int(nd.argmax(probs, axis=0).asscalar())
        prob = float(probs[ind].asscalar())

    prediction = f'La imagen de entrada se clasifica como [{class_names[ind]}], con probabilidad {prob:.3f}.'
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port)




