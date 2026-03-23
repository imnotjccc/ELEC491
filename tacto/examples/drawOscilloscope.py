import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class LiveOscilloscope:
    def __init__(self, N=600, title="TACTO scope", ylabels=("F (N)", "dmax (m)", "dmean (m)")):
        plt.ion()  # 交互模式，允许循环中刷新 [web:638]

        self.N = int(N)
        self.t0 = time.time()

        self.tbuf = deque(maxlen=self.N)
        self.ybuf = [deque(maxlen=self.N), deque(maxlen=self.N), deque(maxlen=self.N)]

        self.fig, self.axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
        self.fig.canvas.manager.set_window_title(title)

        self.lines = []
        for ax, yl in zip(self.axes, ylabels):
            (line,) = ax.plot([], [], lw=1)
            ax.set_ylabel(yl)
            ax.grid(True)
            self.lines.append(line)

        self.axes[-1].set_xlabel("time (s)")

        # 让窗口先显示出来
        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", lambda evt: setattr(self, "closed", True))

    def update(self, F, dmax, dmean):
        """追加一个采样点并刷新图；如果窗口已关闭则什么也不做。"""
        if self.closed:
            return

        t = time.time() - self.t0
        self.tbuf.append(t)
        self.ybuf[0].append(float(F))
        self.ybuf[1].append(float(dmax))
        self.ybuf[2].append(float(dmean))

        tt = np.fromiter(self.tbuf, dtype=np.float32)
        ys = [np.fromiter(b, dtype=np.float32) for b in self.ybuf]

        for line, y in zip(self.lines, ys):
            line.set_data(tt, y)

        if len(tt) >= 2:
            self.axes[0].set_xlim(float(tt.min()), float(tt.max()))

        # 自适应 y 轴范围（避免全挤一条线）
        for ax, y in zip(self.axes, ys):
            if len(y) >= 2:
                ymin, ymax = float(y.min()), float(y.max())
                if abs(ymax - ymin) < 1e-12:
                    ymax = ymin + 1e-6
                pad = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  # 处理GUI事件并刷新 [web:640]

    def update_every_k(self, k, F, dmax, dmean, frame_id):
        """可选：降低刷新频率，避免太卡。"""
        if frame_id % k == 0:
            self.update(F, dmax, dmean)
