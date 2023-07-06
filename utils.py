import tensorflow as tf
import cv2
import numpy as np
import config
from logger import logging



def preprocess_video(video_path : str) -> tuple[tf.Tensor, list] :
    """
    Preprocess the video by keeping the required number of frames, 
    resizing the frames and normalizing the frames.

    params : 
    video_path : path of the video file

    returns :

    Returns tuple (input_tensor, frame_list)

    input_tensor : video with required number of frames and size
    frame_list : list of required number of frames 
    """

    logging.info(">>> Preprocessing the video....")

    # load the video
    video_capture = cv2.VideoCapture(video_path)

    # the number of frames in the original video
    original_number_of_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    # gap between two consecutive frames to capture
    frame_interval = int(original_number_of_frames / config.FRAME_NUM)

    new_video , frame_list = [] , []
    for i in range(0, config.FRAME_NUM  ):
      video_capture.set(cv2.CAP_PROP_POS_FRAMES, i*frame_interval)
      success, frame = video_capture.read()

      if not success :
        logging.info("video loading failed")
        break
      
      frame_list.append(frame)
      # Resize the Frame to fixed height and width.
      resized_frame = cv2.resize(frame, (config.FRAME_HT, config.FRAME_WD))
      
      # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
      normalized_frame = resized_frame / 255
      
      # Append the normalized frame into the frames list
      new_video.append(normalized_frame)

    new_video_array = np.asarray(new_video)

    input_tensor = tf.expand_dims(new_video_array, axis=0)


    video_capture.release()

    logging.info("Video processing successful.")

    return input_tensor, frame_list


# Get top_k labels and probabilities
def get_top_k(probs, label_map,k=5 ):
    """Outputs the top k model labels and probabilities on the given video.

    Args:
        probs: probability tensor of shape (num_frames, num_classes) that represents
        the probability of each class on each frame.
        k: the number of top predictions to select.
        label_map: a list of labels to map logit indices to label strings.

    Returns:
        a tuple of the top-k labels and probabilities.
    """
    # Sort predictions to find top_k
    indices = tf.argsort(probs, direction='DESCENDING').numpy()[0][:k]
    # collect the labels of top_k predictions
    labels = tf.gather(label_map, indices).numpy()
    # decode lablels
    labels = [label.decode('utf8') for label in labels]
    # top_k probabilities of the predictions
    top_probs = tf.gather(probs[0], indices).numpy()
    
    output = dict()
    for label, prob in zip(labels, top_probs):
        output[label] = float(prob) / 100
    return output
