
import argparse
import cv2
from counter import ROICounter
from detector import YOLODetector
from tracker import SortTracker
from util import display_frame, draw_count, draw_roi, draw_tracks

DEFAULT_COUNTER_ROI = [(1466, 326), (1030, 128), (790, 240), (470, 416), (980, 760)]

def process_video(video_path, model_path, roi=DEFAULT_COUNTER_ROI, show=True, output_path=None):
    cap = cv2.VideoCapture(0 if video_path == "webcam" else video_path)
    detector = YOLODetector(model_path)
    tracker = SortTracker()
    counter = ROICounter(roi)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        counter.update(tracks)

        draw_roi(frame, roi)
        draw_tracks(frame, tracks)
        draw_count(frame, counter.count)

        if writer:
            writer.write(frame)
        if show:
            display_frame("Object Counter", frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection and Counting in ROI")
    parser.add_argument("--video", type=str, default="webcam", help="Path to input video file or 'webcam'")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model file")
    parser.add_argument("--output", type=str, help="Path to save output video")

    args = parser.parse_args()
    process_video(args.video, args.model, output_path=args.output)
