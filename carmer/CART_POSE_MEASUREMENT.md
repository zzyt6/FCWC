# 实际小车位姿测量与验证

`cart_pose_measure.py` 是独立的只读测量工具。复用当前相机采集、AprilTag 检测、
相机内参与机器人反馈读取；不发布运动指令、不启用控制、不更改现有导航估计器。
机器人运动只能由操作者通过原有流程控制。

## 测量定义

小车原点是 **90×50 cm 顶层载物板中心在地面的投影**；X 向前（远离推车机器人）、
Y 向左、Z 向上。前后把手各外伸 2.5 cm，所以包含把手的前后长度是 95 cm，
中心不变。标签平面高 77.5 cm，黑色方框边长 10 cm。50 cm 按载物板两边沿的间距解释。

| 标签 | X/m | Y/m | Z/m | 安装旋转 |
|---|---:|---:|---:|---|
| 左侧 ID 0 | -0.103 | +0.149 | 0.775 | 单位旋转 |
| 右侧 ID 1 | -0.103 | -0.141 | 0.775 | 单位旋转 |

两标签中心距应为 29.0 cm。X 使用 `-0.90/2 + 0.347 = -0.103 m`；
34.7 cm 是从板边量起，不能直接从 95 cm 整车后外缘减去。
上述参数集中在代码顶部，不读取旧 URDF 的小车尺寸。

从两个标签的八个原始角点联合做平面 PnP；只看见一个时使用四个角点。
不做时间平滑，便于观察真实噪声和偏差。程序检查正深度、标签正面朝向、平面解歧义、
每个标签的重投影误差。默认某个标签的 RMS 超过 2 像素则整帧无效，不自动选择更合意的一个。
每帧另算两个独立单标签结果，以记录它们之间的小车原点差异和实测标签中心距。

输出坐标系：

| 名称 | 含义 |
|---|---|
| `camera` | 光心原点，X 图像右、Y 图像下、Z 镜头前方 |
| `robot` | 机器人骨盆 `base_link`，含身体倾斜 |
| `level` | 原点仍在骨盆，Z 按 IMU 重力校平，X 为身体前向的水平投影 |
| `map` | 现有 ROS 地图坐标系，供地面导航主要使用的是 X、Y、yaw |

**`level` 的 Z 不是距地高度。** 小车原点在地面，所以它通常在骨盆下方，Z 为负数。
相机坐标系的 yaw 也不是地面导航航向；方向测试优先看 `level` 或 `map`。
CSV 长度单位是米，角度是度，RPY 顺序为 `Rz(yaw) Ry(pitch) Rx(roll)`。

## 启动

在工程根目录的终端执行一次环境设置：

```bash
cd /home/unitree/CWC
conda activate CWC
export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
```

最后一行用于本机 ARM/OpenCV 可能出现的 `cannot allocate memory in static TLS block`。
先退出占用同一 RGB 相机的其他预览/AprilTag 进程。此工具需要独占相机；不会自动停止其他进程。

先验证相机和图案，不读取机器人：

```bash
python -B carmer/cart_pose_measure.py --camera-only --duration 20
```

相机、机器人相对位姿（读取 eth0 上的低层反馈）：

```bash
python -B carmer/cart_pose_measure.py --duration 20
```

测量地图位姿：先通过原有导航界面完成地图加载和机器人定位，使
`map → odom → base_nav → base_link` 的 TF 持续可用，再执行：

```bash
python -B carmer/cart_pose_measure.py --map --duration 20
```

`--map` 使用系统 Python 启动本文件的 ROS Foxy 只读 TF 子进程，因此 CWC 环境无需安装 rclpy。
默认加载 `/opt/ros/foxy/setup.bash`；可用 `--ros-setup` 指定其他设置文件。
终端应使用与导航相同的 `ROS_DOMAIN_ID`、RMW 和网络设置。此工具不会启动导航或 AMCL，
不会让机器人进入运动状态。

地图变换按 V4L2 驱动提供的图像帧时间减 `--camera-latency-ms` 查询，**不退回最新 TF 冒充同时刻数据**。帧时间必须属于主机 monotonic 时钟且严格递增；不可用时报告错误，不用 `cap.read()` 完成时间替代。该参数现在只用于标定后的剩余时间偏差，不应再重复扣除读图延迟。驱动帧时间并不等同于相机、机器人和雷达之间的硬件同步。
仅支持当前实机的系统时间，不支持 `/clock` 仿真时间。
无地图 TF、无有效机器人反馈、无标签、图像过期时，对应位姿字段留空，并记录原因。
初始几帧可能需要等待 TF 缓冲建立。地图未定位时仍可测试 `camera/robot/level`。

无桌面时加 `--no-display`。不设 `--duration` 则持续记录，Q/Esc 或 Ctrl-C 结束。
默认自动创建脚本旁的 `carmer/log/` 目录，以本机时间戳命名，例如
`cart_pose_20261018_183012_123456.csv`（年月日、时分秒、微秒）。从其他工作目录启动时，
默认日志目录也保持不变；终端启动信息会打印本次文件的完整路径。
CSV 旁生成同名 `.meta.json`（参数、坐标定义）与 `.summary.json`（各坐标系统计）。
整个 `carmer/log/` 已加入 Git 忽略规则。
通常不需要指定 `--output`；显式指定时会覆盖默认路径和文件名，程序仍拒绝覆盖已有 CSV。
相机预热帧不计入记录时长。画面中的青色四边形是估计位姿下的 **90×50 cm 载物板边沿**。

## 对比给定矩阵与 URDF 外参

你提供的矩阵保存在 `carmer/cart_camera_extrinsics.json`，定义为**腰部 yaw 为 0** 时，
从相机光学坐标系到物理 `pelvis` 坐标系的变换，平移单位米。
它不是重力校平后的 `level` 变换，也不要求骨盆的 IMU roll/pitch 为 0。
可编辑该 JSON，或通过 `--extrinsics-file /path/to/file.json` 指定同格式文件。

```bash
# 使用 URDF（默认行为）
python -B carmer/cart_pose_measure.py --extrinsics urdf --duration 20

# 使用你提供的矩阵
python -B carmer/cart_pose_measure.py --extrinsics provided --duration 20

# 推荐用于对比：对同一帧、同一份反馈同时计算两组结果
python -B carmer/cart_pose_measure.py --extrinsics compare --duration 20

# 只打印两份零腰角矩阵及差异，不访问相机或机器人
python -B carmer/cart_pose_measure.py --show-extrinsics
```

`provided` 和 `compare` 模式需要机器人反馈，因此不能同时使用 `--camera-only`。
腰角变化时使用以下公式，腰 roll/pitch 仍遵循本工程 URDF 的固定关节约束：

```text
T_provided(q) = T_urdf(q) @ inverse(T_urdf(0)) @ T_provided(0)
T_pelvis_cart = T_provided(q) @ T_camera_cart
T_level_cart  = T_level_pelvis(IMU) @ T_pelvis_cart
```

`q` 来自与该图像时间匹配的反馈。骨盆相对世界的倾斜只在后续 `level`/`map` 变换中应用，
不会重复加到相机安装外参上。零腰角时恰好使用给定矩阵。

对比模式的终端、预览和 CSV 同时提供：

| 字段前缀 | 含义 |
|---|---|
| `robot_urdf` / `robot_provided` | 两种外参得到的小车物理 pelvis 位姿 |
| `level_urdf` / `level_provided` | 两组经过同一次 IMU 校平的位姿 |
| `map_urdf` / `map_provided` | 启用 `--map` 且有有效 TF 时的两组地图位姿 |
| `provided_minus_urdf_level_x_m` 等 | 给定矩阵结果减去 URDF 结果的各轴位置差 |
| `provided_vs_urdf_level_rotation_deg` 等 | 两个旋转矩阵的夹角，非 RPY 分量直接相减 |
| `waist_yaw_deg`、`pelvis_roll_deg`、`pelvis_pitch_deg` | 本帧用于换算的腰角与 IMU 倾斜 |

原有无后缀 `robot/level/map` 字段在 `provided` 模式下使用给定矩阵，在 `urdf` 和 `compare`
模式下使用 URDF。对比时优先看带后缀的字段。`camera` 位姿和重投影误差来自共同的 PnP，
因此**不会因为切换外参而改变，不能用它们判断哪份外参更准**。
`.meta.json` 保存本次实际使用的两份零腰角矩阵和来源文件；`.summary.json` 包含各组统计。

分别查看同一日志的两组结果：

```bash
python -B carmer/cart_pose_measure.py --summarize carmer/log/实际文件名.csv --reference level_urdf
python -B carmer/cart_pose_measure.py --summarize carmer/log/实际文件名.csv --reference level_provided
```

当前两份零腰角外参的光心位置差（给定 − URDF）约为 `[-6.01, 0, -11.19] mm`，
旋转相差约 `0.4°`。这不是所有距离下小车估计位置的固定差值，旋转差的影响取决于观测位置。
验证时固定机器人，在多个距离和左右位置停放小车，独立测量小车原点相对骨盆原点的
水平投影位置及朝向，再对照 `level_urdf` 和 `level_provided` 的均值。
只测试“前进 20 cm”这类相对位移，不能验证相机平移外参的常量偏差；需要绝对位置参考。

## 按顺序验证精度

以下数字是可调整的初始验收目标，并非已测得的设备精度。每组静态记录只包含停稳后的数据；
运动过程另存文件，不能用整段运动的标准差判断静止噪声。

### 1. 静止重复性与几何一致性

固定机器人、相机、小车，双标签无遮挡，连续记录 20 秒。检查青色板框能否贴合真实板边。
查看 CSV 的 `rmse_px`、`baseline_m`、`single_position_difference_m`。
其中 `baseline_m` 来自两个**独立**单标签估计，而非被联合求解固定为 0.29 m。

```bash
# 将示例文件名替换为本次终端输出的实际日志文件名：
python -B carmer/cart_pose_measure.py --summarize carmer/log/cart_pose_20261018_183012_123456.csv --reference level
```

先关注：有效帧比例是否稳定接近 100%；位置每轴标准差是否小于 0.01 m；yaw 标准差是否小于 1°；
独立测得的标签间距是否接近 0.29 m。`p95_position_deviation_m` 是相对均值的抖动，**不是绝对误差**。
重投影误差小也不能证明相机外参、标签尺寸或地图定位正确。

### 2. 已知平移：20 cm 前移、10 cm 左移

机器人保持静止。在地面标出小车原点：用铅垂线从载物板中心落到地面。
再标出初始小车 +X/+Y 方向。保持小车朝向不变，将原点沿初始 +X 移动 20 cm。
每次移动前后均停稳，再分别记录 10 秒：

```bash
python -B carmer/cart_pose_measure.py --map --duration 10
# 操作者移动小车并停稳后：
python -B carmer/cart_pose_measure.py --map --duration 10
# 将两个示例文件名分别替换为移动前、移动后的实际日志：
python -B carmer/cart_pose_measure.py --compare carmer/log/cart_pose_20261018_183012_123456.csv carmer/log/cart_pose_20261018_183100_654321.csv --reference map
```

`delta_xy_in_initial_cart_heading_m` 应接近 `[+0.20, 0]`，`delta_yaw_deg` 接近 0。
恢复到起点，再向初始小车左侧移动 10 cm，结果应接近 `[0, +0.10]`。
先以位移误差不超过 1～2 cm、角度变化不超过 1～2°为检查目标。
没有地图时可去掉 `--map`，比较时使用 `--reference level`，但机器人位置和姿态必须保持不变。
该比较依赖两次记录使用同一参考系；地图重定位/重载后不能直接比较旧记录。

### 3. 已知旋转：左转 15°、右转 15°

利用地面角度线，让小车绕已标出的**小车原点**旋转并停稳。前后各记录 10 秒，使用相同的
`--compare` 命令。左转 yaw 应增加约 15°，右转减少约 15°。
如果实际上绕轮子或把手转，小车原点会平移，这不是测量错误。
程序对 ±180° 的航向差进行了环绕处理。

### 4. 单标签、遮挡与重新检测

在同一静止位置分别运行 `--tag-ids 0`、`--tag-ids 1` 和默认双标签，比较小车原点与航向均值。
再在持续记录中遮挡一个标签，应切换为单标签；全部遮挡时位姿应留空，不能维持旧值假装有效；
移开遮挡后重新输出当前测量。单标签的倾角和地面原点通常更容易抖动。
若两标签结果长期偏向不同方向，检查 ID、贴装轴向、边长和安装位置测量。

### 5. 地图绝对位置与运动补偿

选择地图里可独立测量的位置，以已知墙面/地图轴测出小车原点的地图 X、Y 和方向，
与 `map` 均值比较。只在网页上观察图标，不能作为厘米级精度的独立依据。
把小车保持在原地，操作者在原有控制流程允许的情况下改变机器人的观测位置或身体姿态；
此时 `camera/robot` 位姿会变，而 `map` 小车位姿应基本不变。
本工具本身不执行上述机器人动作。

地图位置明显变动而相对测量稳定：检查 AMCL/FAST-LIO 定位。
站姿变化时位置系统偏移：检查相机光心外参；可用 `--extrinsics compare` 对比名义 URDF 与给定参数。
静止准确、运动时误差大：优先检查图像/反馈/TF 时间对齐和实测相机延迟。
接收时间并非曝光硬件时间，默认延迟 0 ms 表示未标定，不保证已同步。
不要单纯增大 `--max-age` 来掩盖延迟。

## 离线代码验证

```bash
python -B -m unittest -v carmer.test_cart_pose_measure
```

覆盖已知位姿投影恢复、真实 OpenCV 标签检测角点顺序、单标签、像素噪声、错误/丢失标签、
含身体倾斜和相机偏移的地图变换、零腰角给定外参及非零腰角更新、同帧外参比较，
以及 CSV 统计与 ±180° 角度比较。
离线测试通过只说明算法与坐标处理符合测试条件，实物精度应以上面的测量结果为准。
