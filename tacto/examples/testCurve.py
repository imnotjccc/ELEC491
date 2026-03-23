import trimesh
import pyrender
import numpy as np

# ==========================================
# 1. 替换为你的自定义 Gel Mesh 路径
# ==========================================
MESH_PATH = "../meshes/case_meshes/SoftPCB_mesh_curve.STL"  # 强烈建议用 .obj，或者 .stl
mesh_data = trimesh.load(MESH_PATH)

# 如果导出的是 Scene (多个物体)，则取第一个几何体
if isinstance(mesh_data, trimesh.Scene):
    mesh_data = list(mesh_data.geometry.values())[0]

# 打印一下 Mesh 的基础信息，看看尺寸和中心对不对
print("=== Mesh 诊断信息 ===")
print("顶点数:", mesh_data.vertices.shape)
print("面数:", mesh_data.faces.shape)
print("包围盒边界 (min, max):", mesh_data.bounds)
print("包围盒尺寸 (m):", mesh_data.extents)
print("=====================")

# ==========================================
# 2. 强制赋予一个纯白的材质，排除材质导致死黑的问题
# ==========================================
test_material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[1.0, 1.0, 1.0, 1.0],  # 纯白
    metallicFactor=0.0,                    # 非金属
    roughnessFactor=1.0,                   # 完全粗糙
    doubleSided=True                       # 关闭背面剔除！双面都可见
)

# 关闭 smooth shading，确保不会因为顶点法线插值导致曲面变黑
mesh = pyrender.Mesh.from_trimesh(mesh_data, material=test_material, smooth=False)

# ==========================================
# 3. 创建 Pyrender 场景并添加 Mesh
# ==========================================
scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0])  # 深灰色背景，防止和黑色混淆
scene.add(mesh, pose=np.eye(4))

# ==========================================
# 4. 添加相机 (模拟 Tacto 里的视角)
# ==========================================
# 这里的 yfov (垂直视角) 设为 60 度，znear 设得很小
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, znear=0.001)

# 将相机放在 Z 轴上往下看，你可以根据之前打印的 bounds 调整这里的高度 (z值)
camera_pose = np.array([
    [1.0,  0.0,  0.0,  0.0],
    [0.0,  1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0, -0.05], # 相机位于 z = -0.05米 处（请根据你的模型大小修改这里）
    [0.0,  0.0,  0.0,  1.0],
])
scene.add(camera, pose=camera_pose)

# ==========================================
# 5. 添加一个极强的点光源 (模拟 RGB 灯珠)
# ==========================================
# 我们给它一个很强的白光，确保只要没遮挡就一定能照亮
light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=500.0)

# 把灯光放在相机旁边稍微偏一点的位置
light_pose = np.array([
    [1.0, 0.0, 0.0,  0.01],  # X 轴稍微偏移一点
    [0.0, 1.0, 0.0,  0.00],  
    [0.0, 0.0, 1.0, -0.04],  # Z 轴跟相机差不多高
    [0.0, 0.0, 0.0,  1.0],
])
scene.add(light, pose=light_pose)

# ==========================================
# 6. 弹窗可视化 (按 ESC 退出，按 W 切换线框模式)
# ==========================================
print("弹窗中... 如果画面全黑，请用鼠标滚动(缩放)或拖拽(旋转)看看是不是视角不对。")
pyrender.Viewer(scene, use_raymond_lighting=False)

