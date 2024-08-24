import sys
import argparse
from enum import Enum
import os
import cv2
from typing import Iterator, List
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
import torch
import time
from inference import get_model
from roboflow import Roboflow
from dotenv import load_dotenv
load_dotenv()
   
rf = os.environ.get("ROBOFLOW_API_KEY")
PLAYER_DETECTION_MODEL_ID = "hockey-players-pcp2f/2"

PUCK_CLASS_ID = 0
GOALIE_CLASS_ID = 1
SKATER_CLASS_ID = 2
REF_CLASS_ID = 3

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']


## CONFIG == IceRinkConfiguration()


""" should be completed along with the Config ^^
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
"""
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

class Mode(Enum):
    SKATER_DETECTION = 'SKATER_DETECTION'

# frame-by-frame skater detection, using Roboflow inference API
def run_skater_detection(src_vid_path: str, device:str) -> Iterator[np.ndarray]:
    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
   
    
    PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=rf)
    
    frame_generator = sv.get_video_frames_generator(source_path=src_vid_path)

    for frame in frame_generator:
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame

    

def main(src_vid_path: str, target_vid_path: str, device: str, mode: Mode) -> None:
    if mode == Mode.SKATER_DETECTION:
        frame_generator = run_skater_detection(
            src_vid_path=src_vid_path, device=device
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")
    
    video_info = sv.VideoInfo.from_video_path(src_vid_path)
    with sv.VideoSink(target_vid_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--src_vid_path', type=str, required=True)
    parser.add_argument('--target_vid_path', type=str, required=True)
    parser.add_argument('--device', type=str, required=True, default='cpu')
    parser.add_argument('--mode', type=Mode, required=True, default='SKATER_DETECTION')

    args = parser.parse_args()

    main(
        src_vid_path=args.src_vid_path, 
        target_vid_path=args.target_vid_path,
        device=args.device, 
        mode=args.mode
    )