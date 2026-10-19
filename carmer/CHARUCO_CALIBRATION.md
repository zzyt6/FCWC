# ChArUco 拍照、内参标定与相机到 torso 外参

程序：`carmer/charuco_calibrate.py`；配置：`carmer/charuco_config.json`。
仅打开 RGB 相机并保存测量结果，不订阅机器人反馈、不发布运动指令。

## 目前可以标定什么

机器人和相机固定、标定板自由移动，足以估计 **相机内参 K、畸变 D，以及每张照片中板相对于相机的位姿**。
这些照片中没有独立的 torso 信息，因此不能单独确定相机相对于 torso 的固定外参。
程序不会把某张照片的“相机到板”矩阵或 URDF 矩阵冒充实测外参。

当前先完成内参。以后在一个独立测得的、相对于 torso 位姿已知的位置固定标定板，
单独运行 `--reference`，即可求出相机到 torso 的固定变换，不必重拍内参照片。
反过来用 URDF 相机位置算出板的 torso 位姿，再把它当作参考，会形成循环，无法验证 URDF 的偏差。

## 1. 修改板参数

配置已按你提供的实物参数设置：横向 5 格、纵向 5 格、方格 39 mm、标记黑框 27.3 mm、
字典 `DICT_4X4_50`。棋盘区域为 195×195 mm，共有 16 个 ChArUco 内角点。
仍需确认实物的标记 ID 排列和黑白格布局与生成图一致；仅有尺寸和字典相同并不足够。

| 字段 | 含义 |
|---|---|
| `board.squares_x / squares_y` | 横向、纵向的完整棋盘方格数，不是内角点数 |
| `board.square_length_m` | 相邻棋盘格角点距离，单位米 |
| `board.marker_length_m` | ArUco 标记黑色外框边长，单位米，不含白边 |
| `board.dictionary` | 与实际标记一致的 OpenCV 字典 |
| `board.legacy_pattern` | 老版本 OpenCV 印刷板兼容开关；偶数行旧板尤其需要核对 |
| `capture.sample_count` | 按空格保存多少张有效照片后自动开始计算，默认 25 |
| `capture.min_charuco_corners` | 每张图至少包含多少个匹配成功的棋盘角点，默认 12 |
| `calibration.min_views` | 剔除高误差图像后最少保留多少张，默认 12 |

至少使用 8 个不同观测姿态；推荐 20～30 张。数量增加不能弥补始终正对相机、姿态几乎相同。
画面固定为当前相机的原生 640×480 RGB。不要裁剪、缩放或预先去畸变后再输入标定。

如果还没有板，可按当前配置生成 300 DPI 的 PNG：

```bash
cd /home/unitree/CWC
conda activate CWC
export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
python -B carmer/charuco_calibrate.py --generate-board carmer/log/charuco_board.png
```

以 100% 原尺寸打印，关闭“适合页面”，贴在平整刚性板上，**用尺再次测量方格边长**。
PNG 四周有额外 5 mm 白边，整体约 205×205 mm，可按原尺寸放入 A4 纸；
白边不属于棋盘尺寸或板坐标原点。
如已有板，请先确认配置生成的图案与实物的字典、ID 排列、黑白格布局一致。

## 2. 拍照与自动计算

先退出其他占用同一 RGB 相机的预览或 AprilTag 程序。保持相机固定；吊挂状态下尽量减少晃动。

```bash
python -B carmer/charuco_calibrate.py
# 或加载另一份板配置：
python -B carmer/charuco_calibrate.py --config /absolute/path/to/charuco_config.json
```

每次调整板的位置、距离和倾斜角度，停稳后按 **空格**。窗口显示棋盘角点、保存数量和拒绝原因。
只有角点足够、不共线、图像占比合适且图像未过期时才保存；保存的是按键时显示的那张原始图像。
板应覆盖画面中心、四角和边缘；包含近处、远处，以及绕两个方向的明显倾斜，避免模糊、反光和弯曲。

有效照片达到 `sample_count` 后自动关闭采集窗口并开始标定，输出 K、D、总体/逐图重投影误差。
默认剔除 RMS 超过 1.5 像素的图像，并要求总体 RMS 不超过 1 像素、观测法向有足够变化。
这些是质量筛查阈值，不是毫米级精度保证；若失败，照片与错误报告仍保留。

Q/Esc 可提前结束并保留已拍照片；稍后可离线处理：

```bash
python -B carmer/charuco_calibrate.py --images carmer/log/charuco/实际会话目录/raw
```

离线处理应使用拍摄时相同的板配置；每个会话都保存了 `config.json`，可以通过 `--config` 指定该副本。

## 3. 保存结果与检查

所有会话数据位于 `carmer/log/charuco/时间戳_intrinsics/`，该目录已被 Git 忽略：

| 文件 | 内容 |
|---|---|
| `raw/001.png ...` | 原始照片 |
| `observations.json` | 与照片对应的角点、ID、时间等信息 |
| `config.json` | 本次配置副本 |
| `intrinsics_report.json` | RMS、剔除图像、参数标准差，以及逐图 `T_camera_optical_board` |
| `camera_intrinsics.json` | 可直接供现有内参加载器读取的 K、D |
| `undistorted/` | 成功参与标定的图像去畸变预览，仍使用 K、640×480 |
| `extrinsics_status.json` | URDF 相机到 torso 矩阵；实测矩阵未标定时为 `null` |
| `failure.json` | 若失败，记录原因 |

成功内参还会写到固定入口 `carmer/calibrate_camera/charuco_intrinsics.json`。
当前出厂内参文件不会被覆盖；现有测量工具可显式使用新内参：

```bash
python -B carmer/cart_pose_measure.py --intrinsics carmer/calibrate_camera/charuco_intrinsics.json
```

内参重投影误差很小，仍不代表相机到 torso 外参正确。比较去畸变图中的直线与原图；
随后重复已知距离和角度实验。畸变校正预览图不应与原始 K、D 混用而再次去畸变。

## 4. 已知参考位姿接口：以后标定固定外参

板坐标系使用 **OpenCV ChArUco 定义**：原点在棋盘区域左上方外角（不是第一个内角点、不是纸张白边角），
面对印刷面，X 向右、Y 向下，Z 指向板内。**它与本工程 AprilTag 的 Z 朝外约定不同。**
torso 坐标系使用 URDF 的 `torso_link`；位置和角度都必须相对于这个坐标系独立测量。
机器人吊起后，不能自动把重力水平系当作 torso 坐标系。

补充配置中的两个字段，默认 `null` 表示尚无这项测量：

```text
reference.position_in_torso_m = [板原点的 x, y, z]，单位米
reference.rpy_in_torso_deg    = [roll, pitch, yaw]，单位度
```

旋转按 `R_torso_board = Rz(yaw) Ry(pitch) Rx(roll)` 定义，即
`p_torso = R_torso_board @ p_board + position_in_torso_m`。
之后把板固定在这个已知位置，保持板和 torso 的相对关系不变，运行：

```bash
python -B carmer/charuco_calibrate.py --reference
# 也可显式指定之前标定的内参：
python -B carmer/charuco_calibrate.py --reference --intrinsics carmer/calibrate_camera/charuco_intrinsics.json
```

默认空格拍摄 8 张该固定参考位姿的照片，用于检查重复性。
此步骤不要继续自由移动板；若更换参考位置，必须同时更新已知参考位姿。
公式为：

```text
T_torso_camera_optical = T_torso_board @ inverse(T_camera_optical_board)
```

程序输出实测矩阵和 URDF 矩阵、平移差、旋转差；保存到会话目录及
`carmer/calibrate_camera/camera_to_torso.json`。URDF 矩阵也包含光学坐标轴换算，因此两者源坐标系一致。
光学系为 X 图像右、Y 图像下、Z 镜头前方。平移单位米，矩阵使用齐次列向量。
没有有效参考位姿或质量不合格时，不会生成/覆盖这个固定入口的有效外参。

## 5. 其他程序读取与实时腰角换算

直接读取相机光学坐标系到 torso 的矩阵：

```python
from carmer.camera_extrinsics import load_camera_to_torso
T_torso_camera = load_camera_to_torso("carmer/calibrate_camera/camera_to_torso.json")
```

加载器会拒绝未标定状态、不正确的坐标系和无效矩阵。
需要得到相机相对于 pelvis 的实时位姿时：

```python
from carmer.pelvis_attitude import G1Kinematics
from carmer.camera_extrinsics import CalibratedCameraExtrinsics

model = G1Kinematics()
camera = CalibratedCameraExtrinsics(model, "carmer/calibrate_camera/camera_to_torso.json")
# 每次使用与图像时间匹配的实测关节角 q，索引与现有反馈一致：
T_pelvis_camera = camera.in_pelvis(q)
```

其计算是 `T_pelvis_torso(q) @ T_torso_camera`，实时使用腰 yaw。
遵循当前工程的 URDF 假设：腰 roll/pitch 固定；如果硬件上解锁这两个关节，必须更新运动学接口。
固定文件保存 torso 外参，不能把某一时刻的 pelvis 矩阵当作所有腰角通用的常量。
当前标定工具提供此接口；现有导航和控制不会自动切换到新外参。

离线测试：

```bash
python -B -m unittest -v carmer.test_charuco_calibrate
```

这些测试不打开相机、不读取或控制机器人。实际标定精度仍需独立测量验证。
