import gradio as gr
import utils
from config import KINETICS_600_LABELS, MODEL

def get_predictions(video_path):
    video, frame_list = utils.preprocess_video(video_path)
    model = MODEL
    probs = model(video)
    labels = utils.get_top_k(probs, label_map=KINETICS_600_LABELS)
    return labels

label = gr.components.Label(num_top_classes=5)
vd = gr.components.Video()
iface = gr.Interface(fn=get_predictions, inputs=vd, outputs=label)
iface.launch(debug=True)