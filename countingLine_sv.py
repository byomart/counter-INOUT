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

# Line
START = sv.Point((config['line']['x1']),(config['line']['y1']))
END = sv.Point((config['line']['x2']),(config['line']['y2']))
COLOR = config['line']['color']
TEXT_IN = config['line']['text_in']
TEXT_OUT = config['line']['text_out']
THICKNESS = config['line']['thickness']
TEXT_THICKNESS = config['line']['text_thickness']
TEXT_SCALE = config['line']['text_scale']

line_counter = sv.LineZone(start=START, end=END)
line_annotator = sv.LineZoneAnnotator(
    thickness= THICKNESS, 
    text_thickness= TEXT_THICKNESS,
    color = sv.Color(255, 150, 150), 
    text_scale= TEXT_SCALE,
    custom_in_text= TEXT_IN,
    custom_out_text= TEXT_OUT
)

# track object
tracker = sv.ByteTrack(track_thresh=0.05, track_buffer=30, match_thresh=0.75, frame_rate=24)

# annotations objects
box_annotator = sv.BoxCornerAnnotator()
label_annotator = sv.LabelAnnotator()
color_annotator = sv.ColorAnnotator()


###########################################################################

cap = cv2.VideoCapture(source)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
pbar = tqdm(total=frame_count, desc='Procesando frames', unit='frames' )
videoWriterObject = set_VideoWriter(cap, output_filename)
cv2.namedWindow('Supervision processed video', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Supervision processed video', 2000, 2000)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    print(frame.shape)
    results = detection_model(frame, verbose=False, device='cpu', conf=0.35, iou=0.6, imgsz=1280)[0]

    detections = sv.Detections.from_ultralytics(results)
    # only consider selected class
    detections = detections[np.isin(detections.class_id, 0)]
    detections = tracker.update_with_detections(detections)


    # custom labels
    labels = [
        f"#{tracker_id} - {results.names[class_id]} - {confidence:0.2f}"
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

    # count 'people' crossing the line 
    line_counter.trigger(detections)
    line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)
    
    
    #cv2.imshow('Supervision processed video', annotated_frame)
    videoWriterObject.write(annotated_frame)
    pbar.update(1)    
    
    # 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):     
        break

pbar.close()
cap.release()
videoWriterObject.release()
cv2.destroyAllWindows()

