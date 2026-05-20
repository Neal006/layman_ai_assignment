import cv2
import numpy as np
from ultralytics import YOLO

COURT_POLYGON = np.array([
    [660,  30],   # top-left  (far end)
    [1233, 30],   # top-right (far end)
    [1908, 942], # bottom-right (near end)
    [12, 916], # bottom-left  (near end)
], dtype=np.int32)

MAX_PLAYERS = 4 

class IDStabilizer:
    def __init__(self, max_players: int = MAX_PLAYERS):
        self.max_players = max_players
        self._raw_to_stable: dict[int, int] = {}
        self._stable_pos: dict[int, tuple[int, int]] = {}
        self._next = 1

    def update(self, raw_id: int, cx: int, cy: int) -> int:
        if raw_id in self._raw_to_stable:
            sid = self._raw_to_stable[raw_id]
            self._stable_pos[sid] = (cx, cy)
            return sid

        if self._next <= self.max_players:
            sid = self._next
            self._next += 1
            self._raw_to_stable[raw_id] = sid
            self._stable_pos[sid] = (cx, cy)
            return sid

        best_sid, best_d = None, float("inf")
        for sid, (px, py) in self._stable_pos.items():
            d = (px - cx) ** 2 + (py - cy) ** 2
            if d < best_d:
                best_d, best_sid = d, sid
        self._raw_to_stable[raw_id] = best_sid
        self._stable_pos[best_sid] = (cx, cy)
        return best_sid

PLAYER_COLORS = {1: (255, 100, 0), 2: (0, 180, 255),
                 3: (50, 220, 50),  4: (200, 50, 255)}

def point_in_court(cx: int, cy: int) -> bool:
    return cv2.pointPolygonTest(COURT_POLYGON, (float(cx), float(cy)), False) >= 0

def draw_court_boundary(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    cv2.polylines(out, [COURT_POLYGON], isClosed=True, color=(0, 255, 0), thickness=2)
    return out

def run(video_path: str = "video.mp4",
        output_path: str = "output_players.mp4",
        show: bool = False):

    model = YOLO("yolov8n.pt")
    stabilizer = IDStabilizer()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, persist=True, classes=[0],
            tracker="custom_botsort.yaml", verbose=False
        )[0]

        annotated = draw_court_boundary(frame)

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids   = results.boxes.id.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, raw_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if not point_in_court(cx, cy):
                    continue

                stable_id = stabilizer.update(raw_id, cx, cy)
                color = PLAYER_COLORS.get(stable_id, (255, 255, 255))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"P{stable_id}  {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)

        writer.write(annotated)
        if show:
            cv2.imshow("Player Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"Done — saved to {output_path}")


if __name__ == "__main__":
    run("video.mp4", show=False)
