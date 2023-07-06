import tensorflow as tf
import tensorflow_hub as hub
from keras.models import load_model
from pathlib import Path
import numpy as np
import config
import os


FRAME_HT = 224
FRAME_WD = 224
FRAME_NUM = 8
TENSORFLOW_HUB_URL_LABELS = "https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt"
TENSORFLOW_HUB_URL_MODEL = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'Activity_recognition.h5')

def get_labels():

    labels_path = tf.keras.utils.get_file(
                fname=os.path.join(os.getcwd(),  'labels.txt'),
                origin=config.TENSORFLOW_HUB_URL_LABELS
                )
    
    labels_path = Path(labels_path)

    lines = labels_path.read_text().splitlines()
    KINETICS_600_LABELS = np.array([line.strip() for line in lines])

    return KINETICS_600_LABELS


def get_model():
    encoder = hub.KerasLayer(TENSORFLOW_HUB_URL_MODEL, trainable=True)

    inputs = tf.keras.layers.Input(
                        shape=[FRAME_NUM, FRAME_HT, FRAME_WD, 3],
                        dtype=tf.float32,
                        name='image'
                        )

    # [batch_size, 600] 
    outputs = encoder(dict(image=inputs))

    model = tf.keras.Model(inputs, outputs, name='movinet')

    return model

KINETICS_600_LABELS = get_labels()
MODEL = get_model()
