import threading
import warnings

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

MODEL_NAME = "yolov8n.pt"
VIDEO_SOURCE = 1


class DeepDiveYOLO:
    RUN_UP_BREAK = 0.1
    WINDOW_NAME = "DeepDive"

    def __init__(self, model: YOLO, cap: cv2.VideoCapture):
        self.model = model
        self.cap = cap
        self.current_frame: np.ndarray | None = None
        self.current_results: list[Results] | None = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                warnings.warn("Could not read frame from video source")
                continue
            with self.frame_lock:
                self.current_frame = frame

    def process_model(self):
        while self.current_frame is None and not self.stop_event.is_set():
            threading.Event().wait(self.RUN_UP_BREAK)

        while not self.stop_event.is_set():
            threading.Event().wait(2)
            with self.frame_lock:
                frame = self.current_frame
            if frame is not None:
                self.current_results = self.model.track(frame, persist=True)

    def run(self):
        self.stop_event.clear()
        capture_thread = threading.Thread(target=self.capture_frames)
        process_thread = threading.Thread(target=self.process_model)

        capture_thread.start()
        process_thread.start()

        while self.current_frame is None or self.current_results is None:
            threading.Event().wait(self.RUN_UP_BREAK)

        self.current_frame: np.ndarray
        self.current_results: list[Results]

        while not self.stop_event.is_set():
            with self.frame_lock:
                display_frame = self.current_results[0].plot(img=self.current_frame)

            cv2.imshow(self.WINDOW_NAME, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop_event.set()

        capture_thread.join()
        process_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise Exception("Could not open video source {VIDEO_SOURCE}")
    try:
        DeepDiveYOLO(model, cap).run()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
