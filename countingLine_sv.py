from ultralytics import YOLO
import supervision as sv
from   tqdm import tqdm
import numpy as np
import yaml
import cv2

from utils import set_VideoWriter


# Config file
yaml_file = 'config_SV.yaml'
with open(yaml_file, 'r') as f:
    config = yaml.safe_load(f)

# Models
model_name = config['model']['detection']
detection_model = YOLO(model_name)

# Paths
source = config['assets']['input_video']
output_filename = config['assets']['output_filename']


# Crossing Line
# START = sv.point(config['line']['start']) 
# END = sv.point(config['line']['end']) 


###########################################################################

box_annotator = sv.BoxCornerAnnotator()
label_annotator = sv.LabelAnnotator()
color_annotator = sv.ColorAnnotator()
halo_annotator = sv.HaloAnnotator()
percentage_bar_annotator = sv.BoundingBoxAnnotator()


cap = cv2.VideoCapture(source)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
pbar = tqdm(total=frame_count, desc='Procesando frames', unit='frames' )
videoWriterObject = set_VideoWriter(cap, output_filename)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = detection_model(frame)
    CLASS_NAMES_DICT = detection_model.model.names

    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
    # custom labels
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id, _
        in detections
    ]
    # annotate  
    boxed_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        )
    annotated_frame = label_annotator.annotate(
        scene=boxed_frame,
        detections=detections,
        labels=labels
        )
    annotated_frame = color_annotator.annotate(
        scene=annotated_frame.copy(),
        detections=detections   
        )


    #cv2.imshow('Supervision processed video', annotated_frame)
    videoWriterObject.write(annotated_frame)

    pbar.update(1)    

    # 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):     
        break


cap.release()
videoWriterObject.release()
cv2.destroyAllWindows()
pbar.close()
