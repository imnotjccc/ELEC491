import matplotlib.pyplot as plt
import numpy as np

class ForceDisplacementPlotter:
    def __init__(self, k_theoretical=1283.5):
        self.k_theoretical = k_theoretical
        
        # 存储历史数据
        self.z_data = []
        self.f_data = []
        self.delta_z_data = []  # 记录位移量
        self.z_start = None     # 记录接触时的初始 z 高度
        
        # 记录最近几次的F，用于判断是否停止变化
        self.f_history = []
        self.stop_plotting = False
        
        # 初始化 Matplotlib 交互模式
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlabel('Displacement (m) [Initial Z - Current Z]')
        self.ax.set_ylabel('Force (N)')
        self.ax.set_title('Force vs Displacement')
        self.ax.grid(True)

        self.ax.axhline(y=2.5, color='g', linestyle='-.', 
                        label=f'Target Force F={2.5}N', alpha=0.8)
        
        # 实际数据折线
        self.line_actual, = self.ax.plot([], [], 'b.-', label='Measured Data', markersize=4)
        
        # 理论数据折线 (F = k * \Delta z)
        self.line_theory, = self.ax.plot([], [], 'r--', label=f'Theoretical k={self.k_theoretical}', alpha=0.7)
        
        self.ax.legend()

    def update_plot(self, z_current, f_current):
        # 如果已经判定停止，则不再更新
        if self.stop_plotting:
            return False
            
        # 记录初始的 z 位置 (作为位移0点)
        if self.z_start is None:
            self.z_start = z_current
            
        # 计算当前的位移量 (下降的距离，转为正数)
        displacement = self.z_start - z_current
        
        # 存入数据
        self.z_data.append(z_current)
        self.f_data.append(f_current)
        self.delta_z_data.append(displacement)
        
        # 判断力是否停止变化 (连续3次F变化量极小)
        self.f_history.append(f_current)
        if len(self.f_history) > 3:
            self.f_history.pop(0)
            # 如果最大最小的力差值小于 0.001，认为到底停止了
            if max(self.f_history) - min(self.f_history) < 0.001 and f_current > 0.1:
                print("力已不再变化，停止绘制！")
                self.stop_plotting = True
        
        # 理论力的计算 F = k * displacement
        # 截取从 0 到当前最大位移的数组，计算理论线
        max_disp = max(self.delta_z_data) if self.delta_z_data else 0
        theory_x = np.linspace(0, max_disp + 0.001, 100)
        theory_y = self.k_theoretical * theory_x

        # 更新图表数据
        self.line_actual.set_data(self.delta_z_data, self.f_data)
        self.line_theory.set_data(theory_x, theory_y)
        
        # 动态调整坐标轴范围
        self.ax.relim()
        self.ax.autoscale_view()
        
        # 刷新画布
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        return not self.stop_plotting
        
    def keep_window_open(self):
        plt.ioff()
        plt.show()
