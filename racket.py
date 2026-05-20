import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from player import IDStabilizer, PLAYER_COLORS, point_in_court, draw_court_boundary

FLOW_MIN_MAG    = 3.0   # ignore motion below this (px/frame)
FLOW_PERCENTILE = 85    # keep top 15% highest-magnitude pixels
MIN_HOT_PIXELS  = 5     # trust a detection only if this many pixels qualify
MAX_BLOB_FRAC   = 0.40  # if >40% of bbox is hot, whole body is moving — skip
LINGER_FRAMES   = 5     # hold last known position after occlusion
RACKET_RADIUS   = 22    # visual circle radius (px)


class RacketTracker:
    def __init__(self, velocity_window: int = 4):
        self._vw        = velocity_window
        self._prev_gray = None
        self._hist:   dict[int, deque] = {}
        self._linger: dict[int, int]   = {}
        self._last:   dict[int, dict]  = {}

    def update(self, gray: np.ndarray, player_bboxes: dict[int, tuple]) -> dict:
        """
        gray          : full-frame grayscale image
        player_bboxes : {pid: (x1,y1,x2,y2)}
        returns       : {pid: {center, bbox, velocity, lingering}}
        """
        result: dict[int, dict] = {}

        if self._prev_gray is None:
            self._prev_gray = gray
            return result

        for pid, (px1, py1, px2, py2) in player_bboxes.items():
            # Clamp to frame
            px1 = max(0, px1); py1 = max(0, py1)
            px2 = min(gray.shape[1] - 1, px2)
            py2 = min(gray.shape[0] - 1, py2)

            prev_roi = self._prev_gray[py1:py2, px1:px2]
            curr_roi = gray[py1:py2, px1:px2]

            if prev_roi.shape[0] < 16 or prev_roi.shape[1] < 16:
                continue

            flow = cv2.calcOpticalFlowFarneback(
                prev_roi, curr_roi, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            thresh = max(FLOW_MIN_MAG, np.percentile(mag, FLOW_PERCENTILE))
            mask   = mag > thresh
            ys, xs = np.where(mask)
            n_hot  = len(xs)
            roi_area = (py2 - py1) * (px2 - px1)

            if n_hot < MIN_HOT_PIXELS or n_hot > roi_area * MAX_BLOB_FRAC:
                # Not a valid racket hit this frame — handled by linger below
                pass
            else:
                # Centroid of the high-motion cluster = racket position
                cx = px1 + int(np.mean(xs))
                cy = py1 + int(np.mean(ys))

                self._hist.setdefault(pid, deque(maxlen=self._vw)).append((cx, cy))
                hist = list(self._hist[pid])
                vx   = hist[-1][0] - hist[0][0] if len(hist) >= 2 else 0
                vy   = hist[-1][1] - hist[0][1] if len(hist) >= 2 else 0

                state: dict = {
                    "center":    (cx, cy),
                    "bbox":      (cx - RACKET_RADIUS, cy - RACKET_RADIUS,
                                  cx + RACKET_RADIUS, cy + RACKET_RADIUS),
                    "velocity":  (vx, vy),
                    "lingering": False,
                }
                result[pid]       = state
                self._last[pid]   = state
                self._linger[pid] = 0

        # Carry forward last known state for players with no fresh detection
        for pid, last in list(self._last.items()):
            if pid not in result:
                self._linger[pid] = self._linger.get(pid, 0) + 1
                if self._linger[pid] <= LINGER_FRAMES:
                    result[pid] = {**last, "lingering": True}
                else:
                    del self._last[pid]

        self._prev_gray = gray
        return result


def run(video_path: str = "video.mp4",
        output_path: str = "output_racket.mp4",
        show: bool = False):

    model      = YOLO("yolov8n.pt")
    stabilizer = IDStabilizer()
    tracker    = RacketTracker()

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        track_res = model.track(frame, persist=True, classes=[0],
                                tracker="custom_botsort.yaml", verbose=False)[0]

        annotated: np.ndarray    = draw_court_boundary(frame)
        player_bboxes: dict[int, tuple] = {}

        if track_res.boxes is not None and track_res.boxes.id is not None:
            for box, raw_id, conf in zip(
                track_res.boxes.xyxy.cpu().numpy(),
                track_res.boxes.id.cpu().numpy().astype(int),
                track_res.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if not point_in_court(cx, cy):
                    continue
                sid   = stabilizer.update(raw_id, cx, cy)
                color = PLAYER_COLORS.get(sid, (255, 255, 255))
                player_bboxes[sid] = (x1, y1, x2, y2)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"P{sid} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        racket_states = tracker.update(gray, player_bboxes)

        for pid, state in racket_states.items():
            cx, cy    = state["center"]
            vx, vy    = state["velocity"]
            color     = PLAYER_COLORS.get(pid, (255, 255, 255))
            thickness = 1 if state["lingering"] else 2
            tag       = f"R{pid}" + (" ~" if state["lingering"] else "")

            cv2.circle(annotated, (cx, cy), RACKET_RADIUS, color, thickness)
            cv2.putText(annotated, tag, (cx - 12, cy - RACKET_RADIUS - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            speed = (vx ** 2 + vy ** 2) ** 0.5
            if speed > 2:
                scale = 40 / speed
                cv2.arrowedLine(annotated, (cx, cy),
                                (int(cx + vx * scale), int(cy + vy * scale)),
                                color, 2, tipLength=0.4)

        writer.write(annotated)
        if show:
            cv2.imshow("Racket Tracking", annotated)
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
