import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def draw_boxes(frame, boxes, classes, names, confidences=None, ids=None):
    h, w = frame.shape[:2]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(classes[i]) if classes is not None else -1
        label = names.get(cls_id, f"id:{cls_id}") if cls_id >= 0 else "object"
        if confidences is not None:
            label += f" {confidences[i]:.2f}"
        if ids is not None:
            label = f"#{int(ids[i])} " + label

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(20, y1 - 10)
        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 4, y_text + 4), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Real-time Object Detection")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="model weights path")
    parser.add_argument("--source", type=str, default="0", help="camera index or video path")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--save", type=str, default="", help="optional output video path (e.g., out.mp4)")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--track", action="store_true", help="assign simple track IDs per object")
    args = parser.parse_args()

    # Convert numeric strings "0", "1" to int for webcam index
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.weights)
    names = model.model.names if hasattr(model, "model") else model.names

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # Video writer (optional)
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    prev = time.time()
    avg_fps = 0.0
    alpha = 0.1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Inference
        results = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)

        if len(results) > 0:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
            classes = r.boxes.cls.cpu().numpy() if r.boxes is not None else []
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
            ids = None
            if args.track and hasattr(r, "boxes") and hasattr(r.boxes, "id") and r.boxes.id is not None:
                # Some YOLO versions set r.boxes.id when using model.track(). Here we emulate simple IDs per frame.
                ids = np.arange(len(boxes))

            draw_boxes(frame, boxes, classes, names, confs, ids)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        avg_fps = alpha * fps + (1 - alpha) * avg_fps
        cv2.putText(frame, f"FPS: {avg_fps:0.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2)

        cv2.imshow("YOLOv8 - Real-time", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()