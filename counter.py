from typing import List, Set, Tuple
import cv2
import numpy as np


class ROICounter:
    def __init__(self, roi: List[Tuple[int, int]]) -> None:
        self.roi = roi
        self._reset_counters()
        self._type_mapping = {"car": "CAR", "bus": "BUS", "truck": "TRUCK"}
        self._next_id = 1

    def _reset_counters(self) -> None:
        self._entered = self._exited = 0
        self._previous_states = {}
        self._entered_objects = set()
        self._exited_objects = set()
        self._previous_positions = {}
        self._counted_objects = set()
        self._systematic_ids = {}

    def _assign_type(self, track_id: int) -> str:
        object_type = ["car", "bus", "truck"][track_id % 3]
        if object_type not in self._type_mapping:
            self._type_mapping[object_type] = f"TYPE{self._next_id}"
            self._next_id += 1
        return self._type_mapping[object_type]

    def is_inside(self, bbox: np.ndarray) -> bool:
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        return cv2.pointPolygonTest(np.array(self.roi, np.int32), center, False) >= 0

    def _get_direction(prev_pos, curr_pos, threshold=1.0) -> str:
        dx, dy = curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]
        if abs(dx) < threshold and abs(dy) < threshold:
            return "Stationary"

    def _update_position_and_direction(self, track_id: int, bbox: np.ndarray, directions: dict) -> None:
        curr_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        if track_id in self._previous_positions:
            direction = self._get_direction(self._previous_positions[track_id], curr_pos)
            if direction in directions:
                directions[direction].append(self._systematic_ids[track_id])
        self._previous_positions[track_id] = curr_pos

    def update(self, tracks: np.ndarray) -> Set[int]:
        counted_ids, directions = set(), {"Right": [], "Left": [], "Up": [], "Down": []}
        for track in tracks:
            bbox, track_id = track[:-1], int(track[-1])
            self._systematic_ids.setdefault(track_id, self._assign_type(track_id))
            is_inside = self.is_inside(bbox)

            if is_inside and track_id not in self._entered_objects:
                self._entered += 1
                self._entered_objects.add(track_id)

            if not is_inside and self._previous_states.get(track_id, False):
                self._exited += 1
                self._exited_objects.add(track_id)

            self._update_position_and_direction(track_id, bbox, directions)
            self._previous_states[track_id] = is_inside
            if is_inside and track_id not in self._counted_objects:
                counted_ids.add(track_id)
                self._counted_objects.add(track_id)

        self._print_directions(directions)
        return counted_ids

    def _print_directions(self, directions: dict) -> None:
        for direction, objects in directions.items():
            if objects:
                print(f"Objects {', '.join(map(str, objects))} moving {direction}")
            else:
                print(f"No objects moving {direction}")
        print(f"Entered: {self._entered} | Exited: {self._exited}")

    def reset(self) -> None:
        self._reset_counters()

    @property
    def entered(self) -> int:
        return self._entered

    @property
    def exited(self) -> int:
        return self._exited

    @property
    def count(self) -> int:
        return self._entered + self._exited
