from ultralytics import YOLO
from   tqdm import tqdm
import yaml
import cv2

from utils import set_VideoWriter, draw_boxes_for_class

# Config file
yaml_file = 'config.yaml'
with open(yaml_file, 'r') as f:
    config = yaml.safe_load(f)

# Models
model_name = config['model']['detection']
detection_model = YOLO(model_name)

# Paths
source = config['assets']['runners_video']
output_filename = config['assets']['output_filename']


###########################################################################


cap = cv2.VideoCapture(source)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
pbar = tqdm(total=frame_count, desc='Procesando frames', unit='frames' )
videoWriterObject = set_VideoWriter(cap, output_filename)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # detecting people, drawing detection boxes on the frames, composing the video
    results = detection_model(frame)
    processed_frame = draw_boxes_for_class(frame, results, 'person', config)
    videoWriterObject.write(processed_frame)
    #cv2.imshow('Input video processing display', processed_frame)
    pbar.update(1)    


    # 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):     
        break


cap.release()
videoWriterObject.release()
cv2.destroyAllWindows()
pbar.close()
