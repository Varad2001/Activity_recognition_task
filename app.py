import gradio as gr
import utils
from config import KINETICS_600_LABELS, MODEL
from logger import logging


def get_predictions(video_path):

    logging.info(f">>> Getting predictions for video file : {video_path}")

    video, _ = utils.preprocess_video(video_path)
    model = MODEL
    probs = model(video)
    labels = utils.get_top_k(probs, label_map=KINETICS_600_LABELS)

    logging.info(f"Getting predictions successful : {labels}")

    return labels


label = gr.components.Label(num_top_classes=5)
vd = gr.components.Video()

logging.info(">>> Launching the gradio app...  ")

iface = gr.Interface(fn=get_predictions, inputs=vd, outputs=label)
iface.launch(share=True)

logging.info(">>> Launched successfully.")
