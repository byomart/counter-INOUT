import cv2
import numpy as np



def set_VideoWriter(cap, output_filename):
    '''
    Crea un objeto VideoWriter para escribir un video con las propiedades del video de entrada.

    Args:
        cap (cv2.VideoCapture): Objeto VideoCapture que abre el video de entrada.
        output_filename (str): Nombre del archivo de video de salida.

    Returns:
        cv2.VideoWriter: Objeto VideoWriter configurado con las propiedades del video de entrada.
    '''
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    vidObject = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    return vidObject


def draw_boxes_for_class(frame, results, class_name, config):
    '''
    Dibuja los cuadros delimitadores y etiquetas para una clase específica en los resultados de detección.

    Args:
        results (ultralytics.engine.results.Results): Objeto de resultados del modelo de detección.
        class_name (str): Nombre de la clase para la cual se deben dibujar los cuadros delimitadores y etiquetas.

    Returns:
        None
    '''
    for result in results:
        for i in range(len(result.boxes.xyxy)):
            box = result.boxes.xyxy[i]
            cls = int(result.boxes.cls[i])

            if result.names[cls] == class_name:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), config['draw']['box_color'], config['draw']['box_width'])
                cv2.putText(frame, f'{class_name} - {result.boxes.conf[i]:.2f}', (int(x1), int(y1 - 10)), eval(config['draw']['text_font']), config['draw']['text_fontsize'], config['draw']['text_color'],  config['draw']['text_width'])

    return frame




from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))