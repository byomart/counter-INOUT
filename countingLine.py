from ultralytics import YOLO
import supervision
from   tqdm import tqdm
import yaml
import cv2

from supervision.draw.color import ColorPalette
# from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from utils import set_VideoWriter, draw_boxes_for_class, detections2boxes



from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())



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


# Crossing Line
START = sv.poingt(config['line']['start']) 
END = sv.point(config['line']['end']) 


###########################################################################


cap = cv2.VideoCapture(source)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
pbar = tqdm(total=frame_count, desc='Procesando frames', unit='frames' )
videoWriterObject = set_VideoWriter(cap, output_filename)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # # detecting people, drawing detection boxes on the frames, composing the video
    results = detection_model(frame)
    # processed_frame = draw_boxes_for_class(frame, results, 'person', config)
    # #videoWriterObject.write(processed_frame)
    # cv2.imshow('Input video processing display', processed_frame)
    # pbar.update(1)    
        
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )

    print(detections)











    # 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):     
        break


cap.release()
videoWriterObject.release()
cv2.destroyAllWindows()
pbar.close()
