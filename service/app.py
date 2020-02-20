import tensorflow as tf
from flask import Flask, request
from PIL import Image
from flask_cors import CORS, cross_origin
import numpy as np

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
import io

app = Flask(__name__)
import flask
CORS(app, support_credentials=True)


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

# ALways refere model.py for right model architecture which we used for training
def create_model():
    base_model = tf.keras.applications.VGG16(
        input_shape=IMG_SHAPE,
        include_top=False
    )

    # Trainable classification head
    maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    # Layer classification head with feature detector
    model = tf.keras.Sequential([
        base_model,
        maxpool_layer,
        prediction_layer
    ])

    return model

model = create_model()

# Trained model.
model.load_weights("../trained_model/weights.150-0.38.hdf5")


def load_img(content):
    img = tf.io.decode_image(
        content,
        channels=3,
        dtype=tf.float32
    )
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.reshape(img, (1, IMAGE_SIZE, IMAGE_SIZE, 3), name=None)
    return img

@app.route('/classify', methods=['POST'])
@cross_origin(supports_credentials=True)
def classify():
    data = {"success": False}
    if request.method == "POST":
        img = load_img(request.data)
        prediction = float(model.predict(img)[0][0])
        prediction = np.round(prediction, 4)

       # data["prob"] = prediction
        data["predictions"] = [ {
            "label": "Invoice",
            "prob": np.round((1-prediction) * 100, 4)
            },
            {
                "label": "Non-Invoice",
                "prob": np.round(prediction * 100, 4)
                }
                ]
       # if prediction >= 0.5:
       #     data["label"] = "Non-Invoice"
       # else:
       #     data["label"] = "Invoice"
        data["success"] = True
    response = flask.jsonify(data)
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)



