from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import numpy as np
import yaml
import cv2

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

# Initiate polygon zone
video_info = sv.VideoInfo.from_video_path(source)
width, height = video_info.resolution_wh
polygon = np.array([
    [width // 2 - 500, height // 2 + 500],
    [width // 2 + 500, height // 2 + 500],
    [width // 2 + 500, height // 2 - 500],
    [width // 2 - 500, height // 2 - 500]
])
# polygon = np.array([[500, 1500],[1500, 1500],[1500, 1000],[500, 1000]])

video_info = sv.VideoInfo.from_video_path(source)
zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

# initiate annotators
bbox_annotator = sv.BoundingBoxAnnotator()
box_annotator = sv.LabelAnnotator(text_scale=1)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)
# cv2.namedWindow('Supervision processed video', cv2.WINDOW_NORMAL)
tracker = sv.ByteTrack(track_thresh=0.05, track_buffer=30, match_thresh=0.75, frame_rate=24)


def process_frame(frame: np.ndarray, _) -> np.ndarray:
    # detection
    results = detection_model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 49]
    detections = tracker.update_with_detections(detections)


    zone.trigger(detections=detections)
    
    # annotate
    frame = bbox_annotator.annotate(scene=frame, detections=detections)
    
    # custom labels
    labels = [
        f"#{tracker_id} - {results.names[class_id]} - {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id, _
        in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)

    # Display the frame
    cv2.imshow('Supervision processed video', frame)
    
    # Wait for a short period to display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return frame

    return frame

# Process the video and get the last frame
sv.process_video(source_path=source, target_path=output_filename, callback=process_frame)

# Close all OpenCV windows
cv2.destroyAllWindows()



