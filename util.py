from typing import List
from typing import Tuple

import cv2
import numpy as np


def draw_roi(frame: np.ndarray, roi: List[Tuple[int, int]]) -> np.ndarray:
    return cv2.polylines(frame, [np.array(roi, np.int32)], True, (255, 0, 0), 2)


def draw_tracks(
    frame: np.ndarray, tracks: np.ndarray, counted_ids: set[float]
) -> np.ndarray:
    for track in tracks:
        track_id, bbox = track[-1], track[:-1]
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) if track_id not in counted_ids else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return frame


def draw_count(frame: np.ndarray, count: int, ) -> np.ndarray:
    return cv2.putText(
        frame,
        f"Count: {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )


def display_frame(frame: np.ndarray) -> bool:
    cv2.imshow("Tracking and Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True

    return False
