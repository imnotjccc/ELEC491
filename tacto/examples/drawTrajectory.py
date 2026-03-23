import cv2
import numpy as np
from collections import deque

class ContactTrajectoryGUI_XY:
    def __init__(self, x_range_m=0.123, y_range_m=0.123,
                 img_size=700, margin=60, max_points=2000, window_name="Contact trajectory (x-y)"):
        self.Xr = float(x_range_m)  # full range (meters) shown on x axis
        self.Yr = float(y_range_m)  # full range (meters) shown on y axis
        self.S = int(img_size)
        self.M = int(margin)
        self.win = window_name
        self.pts = deque(maxlen=max_points)

        self.scale = min((self.S - 2*self.M) / self.Xr,
                         (self.S - 2*self.M) / self.Yr)

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

    def meter_to_pixel(self, x, y):
        cx = self.S // 2
        cy = self.S // 2
        px = int(round(cx + x * self.scale))
        py = int(round(cy - y * self.scale))
        return px, py

    def draw_axes_and_box(self, img):
        cx = self.S // 2
        cy = self.S // 2

        half_x_px = int(round((self.Xr/2) * self.scale))
        half_y_px = int(round((self.Yr/2) * self.scale))

        x1, y1 = cx - half_x_px, cy - half_y_px
        x2, y2 = cx + half_x_px, cy + half_y_px
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2)

        cv2.line(img, (cx - half_x_px, cy), (cx + half_x_px, cy), (180, 180, 180), 1)
        cv2.line(img, (cx, cy - half_y_px), (cx, cy + half_y_px), (180, 180, 180), 1)

        cv2.putText(img, "+x", (cx + half_x_px - 30, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        cv2.putText(img, "+y", (cx + 8, cy - half_y_px + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        cv2.circle(img, (cx, cy), 3, (180, 180, 180), -1)

    def update(self, x, y, t=None):
        self.pts.append((float(x), float(y), None if t is None else float(t)))

        img = np.zeros((self.S, self.S, 3), dtype=np.uint8)
        self.draw_axes_and_box(img)

        if len(self.pts) >= 2:
            for i in range(1, len(self.pts)):
                x0, y0, _ = self.pts[i-1]
                x1, y1, _ = self.pts[i]
                p0 = self.meter_to_pixel(x0, y0)
                p1 = self.meter_to_pixel(x1, y1)
                cv2.line(img, p0, p1, (0, 255, 255), 2)

        x_last, y_last, t_last = self.pts[-1]
        p_last = self.meter_to_pixel(x_last, y_last)
        cv2.circle(img, p_last, 5, (0, 0, 255), -1)

        txt = f"x={x_last:+.3f} m, y={y_last:+.3f} m"
        if t_last is not None:
            txt += f", t={t_last:.3f}s"
        cv2.putText(img, txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

        cv2.imshow(self.win, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            self.pts.clear()
        if key == ord('q') or key == 27:
            return False
        return True