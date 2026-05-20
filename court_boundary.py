"""
Interactive court boundary adjuster.

Run:  python court_boundary.py

Controls:
  Drag red handles  — move a corner
  ← / →            — jump 10 frames back / forward
  S                 — print final coordinates (copy into player.py)
  R                 — reset to starting coordinates
  Q                 — quit
"""

import cv2
import numpy as np

VIDEO_PATH = "video.mp4"

# -----------------------------------------------------------------------
# Edit these starting coordinates, then drag to fine-tune in the window.
# -----------------------------------------------------------------------
_DEFAULT = np.array([
    [310,  42],   # 0  top-left
    [1595, 42],   # 1  top-right
    [1870, 1000], # 2  bottom-right
    [55,   1000], # 3  bottom-left
], dtype=np.int32)
# -----------------------------------------------------------------------

HANDLE_R   = 14          # click/drag radius for corner handles
FILL_ALPHA = 0.12        # transparency of the polygon fill

LABEL = ["top-left", "top-right", "bottom-right", "bottom-left"]


# -----------------------------------------------------------------------
# Shared state (avoids passing mutable refs through OpenCV callbacks)
# -----------------------------------------------------------------------
_pts      = _DEFAULT.tolist()   # [[x,y], ...]
_selected = None                # index of the currently dragged corner


def _on_mouse(event, x, y, _flags, _param):  # noqa: ARG001
    global _selected
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (px, py) in enumerate(_pts):
            if abs(px - x) <= HANDLE_R * 2 and abs(py - y) <= HANDLE_R * 2:
                _selected = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and _selected is not None:
        _pts[_selected] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        _selected = None


def _draw(frame: np.ndarray, frame_idx: int, total: int) -> np.ndarray:
    out  = frame.copy()
    pts  = np.array(_pts, dtype=np.int32)

    # Semi-transparent fill
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], (0, 200, 50))
    cv2.addWeighted(overlay, FILL_ALPHA, out, 1 - FILL_ALPHA, 0, out)

    # Outline
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Corner handles
    for i, (x, y) in enumerate(_pts):
        is_sel = (i == _selected)
        fill   = (0, 0, 255) if is_sel else (0, 220, 255)
        cv2.circle(out, (x, y), HANDLE_R, fill, -1)
        cv2.circle(out, (x, y), HANDLE_R, (255, 255, 255), 2)
        cv2.putText(out, str(i), (x - 5, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Instructions (top-left)
    instructions = [
        "Drag corners to set boundary",
        "< >  prev / next 10 frames",
        "S    save & print coordinates",
        "R    reset to defaults",
        "Q    quit",
        f"Frame {frame_idx} / {total}",
    ]
    for i, line in enumerate(instructions):
        y = 28 + i * 26
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1)

    # Live coordinates (bottom-left)
    by = out.shape[0] - 115
    cv2.putText(out, "COURT_POLYGON coordinates:", (12, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(out, "COURT_POLYGON coordinates:", (12, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)
    for i, ((x, y_c), lbl) in enumerate(zip(_pts, LABEL)):
        text = f"  {i} {lbl:14s} ({x:4d}, {y_c:4d})"
        row  = by + 24 + i * 22
        cv2.putText(out, text, (12, row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 4)
        cv2.putText(out, text, (12, row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 255, 200), 1)

    return out


def _print_coords():
    print("\n--- Paste this into player.py ---")
    print("COURT_POLYGON = np.array([")
    for (x, y), lbl in zip(_pts, LABEL):
        print(f"    [{x:4d}, {y:4d}],   # {lbl}")
    print("], dtype=np.int32)")
    print("---------------------------------\n")


def main():
    global _pts

    cap   = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    idx   = 0

    def load(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        return f if ok else None

    frame = load(0)
    if frame is None:
        print(f"Cannot open {VIDEO_PATH}")
        return

    win = "Court Boundary  |  drag corners"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, _on_mouse)

    while True:
        cv2.imshow(win, _draw(frame, idx, total))
        key = cv2.waitKey(16) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            _print_coords()
            np.save("court_polygon.npy",
                    np.array(_pts, dtype=np.int32))
            print("Also saved as court_polygon.npy")
        elif key == ord("r"):
            _pts = _DEFAULT.tolist()
            print("Reset to defaults.")
        elif key in (81, 2, ord(","), ord("<")):   # left arrow or <
            idx   = max(0, idx - 10)
            frame = load(idx)
        elif key in (83, 3, ord("."), ord(">")):   # right arrow or >
            idx   = min(total, idx + 10)
            frame = load(idx)

    cap.release()
    cv2.destroyAllWindows()
    _print_coords()


if __name__ == "__main__":
    main()
