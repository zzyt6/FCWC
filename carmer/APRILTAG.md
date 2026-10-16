# AprilTag 实时识别

在 Ubuntu 桌面或已启用 X11 转发的终端运行：

```bash
conda activate CWC
cd /home/unitree/CWC
python carmer/apriltag_detector.py
```

程序默认读取脚本同目录的 **`apriltag_config.json`**；从工程根目录或 `carmer` 目录
启动都会读取同一文件。全部日常运行参数已写入配置，键名与命令行选项一致，去掉前面的 `--`。
当前关键配置为：

```json
{
  "quad-decimate": 2,
  "cart-to-tag-rpy-deg": [0, 0, 0],
  "cart-tag-id": null,
  "cart-tag-ids": [0, 1],
  "pelvis-attitude-source": "imu",
  "robot-urdf": "../G1_URDF/g1.urdf",
  "network-interface": "eth0"
}
```

上面只展示关键项，完整配置还包含设备、显示、计时、运行帧数、输出目录、内参、Tag 尺寸、
腰角和滤波参数。`no-display`、`profile`、`no-smoothing` 使用 JSON 布尔值 `true`/`false`；
`frames: 0` 表示持续运行；`axis-length: null` 表示轴长度等于 Tag 边长；
本机 `cart-tag-ids: [0, 1]` 只融合已确认的小车标签 0、1；背景 ID 2 等不参与。
`--cart-tag-id 0` 可临时覆盖分组，只检查一个标签；单个 ID 与 ID 分组不能同时配置。
`intrinsics`、`snapshot-dir` 和 `robot-urdf` 的相对路径按配置文件所在目录解析。
可通过 `--config 路径` 指定其它 JSON，显式命令行选项优先于文件中的值；其它配置文件中
省略的字段使用程序内置默认值。未知键名、错误类型或无效数值会在打开相机前报错。
`--display`、`--smoothing`、`--no-profile` 可以临时覆盖文件中相反的开关设置。

现在默认输出**融合后的小车相对 pelvis 的 yaw**，仍采用 URDF 名义相机外参。
当前配置 `[0, 0, 0]` 表示两个 Tag 的轴均与小车轴一致：**红色 +X 朝小车前方、
绿色 +Y 朝小车左方、蓝色 +Z 朝上**。两个 Tag 平贴小车顶面，腰 yaw 使用真机反馈；
腰 roll/pitch 按当前真机及 URDF 的固定关节处理，不使用 13/14 号电机槽位。
显式的 `cart-to-tag-rpy-deg` 优先于 `tag-mount`，所以启动日志会显示 `Mount custom`。
需要使用 `top`/`rear` 预设时，将配置中的 `cart-to-tag-rpy-deg` 改为 `null`。
其中 `top` 预设仍表示绿色 +Y 朝小车前方，与当前零旋转配置代表不同的贴装方向。

```bash
# 加载默认配置，显示融合后的小车 yaw 和每枚 Tag 的结果
python carmer/apriltag_detector.py

# 加载指定配置
python carmer/apriltag_detector.py --config carmer/apriltag_config.json

# 单独检查 ID 0 时才筛选单个标签（将 0 改成实际 ID）
python carmer/apriltag_detector.py --cart-tag-id 0

# 如果之后改为车尾竖直贴装：先将配置中的 cart-to-tag-rpy-deg 改为 null
python carmer/apriltag_detector.py --tag-mount rear

# 临时使用关节+IMU融合（也可直接改配置）
python carmer/apriltag_detector.py --pelvis-attitude-source fused

# 仅在对比旧算法时：关闭实时补偿，使用固定 +10° 腰角
python carmer/apriltag_detector.py --pelvis-attitude-source fixed --waist-yaw-deg 10
```

右侧信息栏主结果 `Cart yaw (pelvis)`、终端 `fused_cart_yaw_pelvis_deg` 为融合角度，
`fused_cart_yaw_pelvis_rad` 为弧度；0 表示小车前向与 pelvis +X 的**水平投影**一致，
正值表示向左偏转，负值表示向右偏转，范围 [-180°, 180°]。
实时补偿模式的日志 `yaw_reference=pelvis_level` 和 JSON `reference_frame=pelvis_level`
表示已经按重力校平；保留原日志角度字段名以便读取。`fixed` 模式仍为未校平的 pelvis XY。
当前配置只对 ID 0、1 按同一贴装关系计算，再融合其中本帧有效的航向。
两者安装朝向一致时，得到的 yaw 应接近；它们沿左右方向的位置不同不影响旋转换算，
此方法不需要测量两个 Tag 的间距。两枚或更多有效 Tag 使用等权圆周平均：
`yaw = atan2(sum(sin(yaw_i)), sum(cos(yaw_i)))`，公式中的角度使用弧度。
例如 +179° 与 -179° 的结果在 ±180° 附近；不会被错误平均为 0°。
只有一枚有效 Tag 时直接使用它的 yaw；没有有效航向时结果为不可用，角度为 `null`。
一枚 Tag 位姿无效时只使用其余有效 Tag。方向相反、圆周平均无定义时也明确输出不可用。
原来的逐 ID 结果仍保留，便于比较两枚 Tag；融合使用平滑后的航向，另保存原始航向的融合结果。
两个 Tag 应使用不同 ID，因为位姿滤波按 ID 独立维护；贴同一 ID 会让左右标签共用滤波状态。
两个图案朝向不一致时，需要统一贴装朝向或另行配置逐 ID 的安装旋转，不能直接共用当前变换。
一侧 Tag 被遮挡时，融合输出自动切换为单 Tag；两侧都消失时，提示未检测到 Tag。
主循环看到标签全部消失或重新出现时立即打印状态，持续无 Tag 时约每秒重复提示。
多个无关标签同时出现时，它们只保留检测信息，不参与小车航向；可用 `--cart-tag-id` 单独检查一个小车标签。目标标签消失时，显示 yaw 不可用，
不把上次检测结果作为本帧有效角度。三种实时模式都使用实际腰 yaw，
`waist-yaw-deg` 仅在 `fixed` 模式生效；它始终不是机器人运动指令。

## 骨盆倾斜补偿与三种姿态来源

在 `apriltag_config.json` 中选择 `pelvis-attitude-source`：

| 值 | 输入及算法 | 使用条件 |
|---|---|---|
| `imu`（默认） | `rt/lowstate.imu_state.quaternion`，按 wxyz、IMU 到世界的旋转解释；结合 URDF 的骨盆 IMU 安装旋转得到 pelvis 中的向上方向 | 不依赖支撑脚判断 |
| `kinematics` | 真实腿关节角 + URDF，假设支撑脚底与水平地面贴合，脚底 +Z 即世界向上方向 | 水平地面、至少一只脚平贴支撑；不适用于悬空、明显脚尖/脚跟支撑、坡地 |
| `fused` | 对 IMU 与运动学的单位向上向量加权，再归一化 | 同样采用水平支撑脚假设；运动学不明确或偏差过大时仅使用 IMU |

三种模式都由 URDF 和 12 号电机实测角计算 `R_camera_to_pelvis`。腿关节用于
运动学倾角估计；头部与 torso 之间没有活动关节。纯上下平移不直接改变航向。
只估计所需的 roll/pitch（重力方向），不尝试用关节角推算不可观测的绝对世界 yaw。
融合是带差异门限的向量加权，不是 EKF，也不保证比单独 IMU 更准确。

相关配置（默认值已写入文件）：

| 配置 | 默认值 | 含义 |
|---|---|---|
| `network-interface` | `eth0` | 接收只读 DDS `rt/lowstate` 的网卡 |
| `kinematics-support-foot` | `auto` | 可选 `auto` / `left` / `right` / `both`；后面三项表示用户明确假设的支撑脚 |
| `support-height-tolerance-m` | 0.015 | 自动支撑判断及双脚支撑的脚底高度容差 |
| `support-normal-tolerance-deg` | 8 | 双脚支撑时两只脚底法向的最大差异 |
| `fusion-imu-weight` | 0.9 | 运动学通过检查后，融合中 IMU 向上向量的权重；运动学占 0.1 |
| `fusion-max-disagreement-deg` | 15 | IMU 与运动学向上方向夹角超过此值时拒绝运动学 |
| `robot-state-timeout-ms` | 100 | 最新状态接收时间超过此值，yaw 不可用 |
| `state-sync-tolerance-ms` | 30 | 图像目标时间与最近状态样本的最大允许差异 |
| `camera-latency-ms` | 0 | 从图像接收时间中减去的实测相机延迟；0 表示尚未测量补偿 |

`auto` 使用 URDF 的 `left/right_foot_mid_link`。分别假设左/右脚平贴地面，检查
另一只脚的相对高度；双脚近似等高且法向一致时合并两者，否则仅在高度假设明确时选择
较低的脚。不明确时运动学结果无效。这是**几何支撑假设，没有足底力或接触传感器确认**；
双脚同时腾空等状态可能无法仅凭关节角识别。指定 `left` / `right` 时始终假设该脚平贴
支撑，不会自动随步态换脚，因此行走调试应保留 `auto`。

默认与上述配置对应的计算顺序：

1. 图像接收时间减去 `camera-latency-ms`，匹配同一主机单调时钟上的状态缓冲。
   有满足容差的前后两帧时，关节线性插值、IMU 四元数球面插值；否则使用容差内最近帧。
2. 用实时腰 yaw、URDF 固定链和光学轴假设，将原始 Tag 位姿转换到 pelvis。
3. 用所选来源的重力方向建立水平 pelvis 坐标，求小车航向。
4. 各 Tag 航向做圆周低通，再对当前可见有效 Tag 做圆周融合。

相机显示用的位姿滤波与步骤 4 独立，**不会将已经滞后的相机旋转与新姿态拼接**。
`--no-smoothing` 同时关闭显示位姿和航向平滑。Tag 丢失、补偿无效会清除相应航向滤波状态。
IMU 无效时 `imu` / `fused` 模式输出不可用；反馈缺失、过期、与图像时间不匹配或腰 yaw
无效时，所有实时模式均输出不可用，图像与 Tag 检测继续运行。
融合退回 IMU 时日志明确显示 `attitude=fused->imu` 和原因；不会悄悄沿用旧补偿角。

首次运行自动编译 `carmer/.build/robot_state_bridge`，使用工程已附带的 C++ Unitree SDK，
需要 `cmake`、`g++`；无需在 CWC 环境安装 Python DDS 包。子进程仅订阅反馈，无 LowCmd
发布器、模式切换或运动控制调用。退出检测程序时一并关闭子进程。DDS 校验 CRC，并拒绝
重复/倒退 tick；机器人重启导致 tick 重置后需要重启检测程序。
状态接收时间记录在 DDS 回调处，图像时间记录在 `cap.read()` 返回后，二者都不是硬件
曝光同步时间；30 ms 容差只检查接收时间匹配，不能证明曝光时刻已对齐。
快速运动时还应实测相机延迟、检查误差及名义外参，不能将仿真或离线结果当作真机精度保证。

右侧增加 `Attitude`（配置来源→实际来源）、`Waist`（实际腰角）、`Pelvis R/P`（补偿倾角）
和 `Sync`（最近状态与图像目标时间的差异）。`Pelvis R/P` 与 `Cam R/P/Y` 属于不同坐标定义。

预览左侧保留原生 640×480 图像，图像内绘制边框、ID 和坐标轴；右侧 420 像素宽的
独立信息栏优先显示融合后的 yaw 和来源 ID，再按 ID 显示单枚 Tag 的 yaw、相机 R/P/Y，
同时显示帧率和耗时。角度文字不再跟随左右 Tag
挤在图像内。所有文字只绘制一次，使用深色底板保证对比度，避免 OpenCV 5 中粗细字体
字宽不同造成的描边重影。预览宽度为 1060，多标签或较多状态提示时信息栏按需增高。
图像内的坐标仍以左上角为原点，采集分辨率、内参和检测结果的像素坐标不变。
保存的 `*_raw.png` 为原始图像；`*_detected.png` 带有相同的信息栏，JSON 中
`image_size` 表示原图尺寸，`annotated_image_size` 表示带信息栏的标注图尺寸。

怀疑帧率不足时，可以启用分阶段计时。先退出旧的检测进程，避免同时占用相机：

```bash
python carmer/apriltag_detector.py --profile
# 对比关闭窗口后的检测速度
python carmer/apriltag_detector.py --profile --no-display --frames 600
```

终端每秒额外输出 `Profile detect` 和（启用窗口时）`Profile view`。
每项 `a/b` 分别表示最近最多 120 帧的平均耗时和第 95 百分位耗时，单位 ms。
检测统计包含所有处理完成的帧，即使预览跳过了其中一些帧；窗口统计只包含实际绘制的帧。

| 字段 | 计时范围 |
|---|---|
| `read_wait` | 检测线程等待下一幅可用图像；不属于算法计算，也不是相机底层读取耗时 |
| `gray` | BGR 转灰度 |
| `tag` | OpenCV `detectMarkers`：四边形搜索、角点处理和 ID 解码 |
| `pose` | 所有 Tag 的原始位姿：IPPE、两个候选解的 LM 优化、重投影检查、RPY 和坐标轴投影 |
| `tag_output` | 检测结果整理、中心坐标计算等，排除 `pose` |
| `smooth` | 所有 Tag 的位姿滤波及滤波后 RPY/投影 |
| `attitude` | 图像/状态匹配、URDF 运动学、骨盆倾角估计与融合 |
| `yaw` | 原始和滤波后的小车 pelvis yaw 换算及多 Tag 圆周融合 |
| `process` | 完整检测处理耗时，与已有 `Process` 同一口径，已包含上述算法阶段 |
| `draw` | 图像拷贝、边框/坐标轴/文字和右侧信息栏绘制 |
| `imshow` | 调用 OpenCV 显示接口的耗时 |
| `events` | `waitKey(1)`、窗口事件处理和可见性检查；可能包含 GUI/X11 等待 |

这些都是墙钟耗时，可能包含线程调度等待。采集、检测、预览用不同线程和最新帧缓存，
不直接等待对方显示完成，但仍可能受 CPU/GIL 竞争影响。检测与预览耗时不能直接相加推算 FPS；
GUI 调用耗时也不等于图像到达 Windows 屏幕的端到端延迟。
`--profile` 不增加图像上的文字；按 S 保存时，JSON 额外保存当前帧的 `stage_timings_ms`。

先关闭其他占用相机的预览程序。默认设备为 `auto`，程序会探测 `/dev/video*`，
自动选择 RealSense USB 接口 `1.3` 中具有 `VIDEO_CAPTURE` 能力的 RGB 彩色节点
（节点编号可能变化），跳过同一接口的 UVC 元数据节点，并在启动时
打印 BGR 三通道统计；不会把接口
`1.0` 的深度/红外 UYVY 帧误判为彩色。使用原生
640×480、60 Hz 彩色流，识别 `tag36h11` 家族，可同时识别多个标签。

当前 640×480 彩色模式的出厂视场约为：水平 **55.62°**、垂直 **43.20°**；
主点为 `(cx, cy) = (323.67, 246.51)`，相对图像中心 `(320, 240)` 只偏移约
`(+3.67, +6.51)` 像素。程序不做裁剪、缩放或画面平移。
之前保存的 1280×720、30 Hz 彩色模式水平视场约 **70.23°**、垂直 **43.20°**，
所以如果你拿它和当前 640×480 比，当前画面横向会窄约 14.6°，这是分辨率/画幅模式
造成的视场差异。

之前自动节点逻辑只检查“是否返回三通道”，曾把接口 1.0 的 UYVY 红外流误选成彩色；
这会同时造成灰度画面、不同视角和不同光心。现在已改为只探测 USB 接口 1.3 的 RGB
节点。当前启动实测选择 `/dev/video5`，BGR 通道标准差为 `(34.0, 39.6, 37.4)`，
平均通道差为 `39.5`，确认返回的是彩色数据。

如果程序提示 RGB 接口被占用或打不开，先按错误列出的节点检查占用者：

```bash
sudo fuser -v /dev/video4 /dev/video5
```

节点号以本机当前枚举和程序日志为准。普通用户的 `fuser` 可能看不到其他用户的进程，
没有输出不能单独证明相机空闲。`Device or resource busy` 表示采集设备/缓冲区被占用；
先确认并退出对应相机程序，再重新运行检测。`EACCES`/`EPERM` 是权限或设备访问限制，
`ENOENT` 表示当前环境看不到设备节点，`ENODEV` 表示设备断开。

若占用进程是 `root ... videohub_pc4`，它是宇树原厂视频服务，由 `master_service` 管理。
本机 `/unitree/etc/master_service/service/video_hub_pc4` 和模块安装配置给出了对应管理命令。
在机器人 SSH 终端临时停止该服务后，再运行检测程序：

```bash
sudo /unitree/sbin/mscli stopservice video_hub_pc4
sudo fuser -v /dev/video4 /dev/video5
python carmer/apriltag_detector.py
```

第二条命令应不再列出 `videohub_pc4` 的占用；若还有其他进程，需继续确认其用途。
停止期间原厂视频图传暂停。检测结束、相机释放后可恢复：

```bash
sudo /unitree/sbin/mscli startservice video_hub_pc4
```

使用服务管理命令停止，而不是直接 `kill` 被监护的进程；无需修改开机配置。
节点能打开但 OpenCV 初始化失败时，可查看底层日志：

```bash
OPENCV_VIDEOIO_DEBUG=1 OPENCV_LOG_LEVEL=DEBUG python carmer/apriltag_detector.py
```

不要把 `backend is generally available but can't be used to capture by name` 单独理解成
必须改用设备编号；后端初始化失败（包括设备忙）也会产生这条提示。
不要手动把 `/dev/video2` 当作彩色节点，它通常属于接口 1.0 的非彩色流。
下文设备编号、序列号和实测数值是此前采集记录；更换相机后应重新读取该设备的出厂内参。

这台 D435i 的硬件组成是 **1 个 RGB 彩色相机 + 2 个红外立体相机**，不是两个
RGB 彩色相机：

| 传感器 | RealSense 名称 | 当前 V4L2 特征 | 用途 |
|---|---|---|---|
| 彩色 | Color / RGB Camera | USB 接口 1.3，`Camera 6`，YUYV | 当前 AprilTag 程序使用 |
| 左红外 | Infrared 1 | Stereo Module / 接口 1.0 | 深度立体匹配 |
| 右红外 | Infrared 2 | Stereo Module / 接口 1.0 | 深度立体匹配 |
| 深度 | Depth | Stereo Module / Z16 | 深度图 |

当前设备是 `Intel RealSense D435I`，序列号 `342222072884`，固件 `5.15.1.55`，
USB 3.2。最近一次运行打印 `Selected RGB color node: /dev/video5`，并验证了
640×480、60 Hz 的 BGR 通道差异，确认程序没有使用左红外图像。
默认内部检测降采样倍数为 **2**，在配置中修改 `quad-decimate`，或用 `--quad-decimate` 临时覆盖。
默认对每个标签的位姿启用自适应低通平滑，减少坐标轴轻微抖动；原始位姿仍保存在
每个标签的 `raw_pose` 中，平滑后的位姿在 `pose` 中用于绘图和终端输出。

- 绿色边框和文字：标签边界与 ID。
- 标签中心绘制三维坐标轴：X 红色、Y 绿色、Z 蓝色，并标注 X/Y/Z。
- 默认标签黑色方框边长 **10 cm**（不含白边），每根坐标轴长 **10 cm**。
- 终端 `center=(x,y)`：中心像素坐标，图像左上角为原点，x 向右、y 向下。
- 终端 `camera_xyz_cm`：标签中心在相机坐标系中的位置，单位厘米。
- 终端 `camera_rpy_deg=(roll,pitch,yaw)`、信息栏的 `Cam R/P/Y`：标签相对于相机光学坐标系
  的欧拉角，单位度；默认来自平滑后的旋转矩阵。它们不是小车在机器人或世界坐标系中的航向角。
- 终端 `fused_cart_yaw_pelvis_deg` / `fused_cart_yaw_pelvis_rad`、信息栏主结果 `Cart yaw (pelvis)`：
  使用名义外参和标签贴装关系计算，实时补偿模式先校平，再融合当前有效 Tag 的水平相对航向。
  `yaw_source=fused` 表示多 Tag 融合，`single` 表示单 Tag，`yaw_tag_ids` 列出参与计算的 ID。
  每个 ID 后的 `cart_yaw_pelvis_deg` / `cart_yaw_pelvis_rad` 仍表示这枚 Tag 单独的结果。
- `Capture / Detect / Preview`：采集、识别和窗口更新的实际帧率。
- `Detect+Pose`：当前帧灰度转换、所有标签识别和原始位姿/RPY 计算的总耗时，单位 ms。
  `Process`：加上位姿平滑、状态匹配、骨盆姿态计算、小车 yaw 计算和融合的处理耗时。两者均不包含等待相机帧、
  绘图、图片保存、窗口刷新或 X11/网络传输；无标签时也会输出耗时。
- 未识别到标签时继续显示实时图像，画面和终端提示 `No AprilTag detected - searching...`。
  标签重新进入画面后自动恢复 ID 和坐标轴显示；无标签本身不会导致退出。
- `pose.tvec_m`、`pose.R_tag_to_camera` 和 `pose.axes_px` 默认是平滑后的结果。
- `Q` 或 `Esc`：退出。`S`：保存原图、识别标注图和坐标 JSON 到 `carmer/apriltag_snapshots/`。

采集、识别、显示独立运行。识别优先处理最新画面，显示使用对应帧的识别结果。
因此，远程窗口刷新慢时，后台仍会继续采集和识别；识别速度由实际处理耗时决定。
标签应完整出现在画面中，保留周围的白色边缘，避免反光和运动模糊。

平滑使用按标签 ID 独立工作的 **One-Euro 风格自适应低通**：位置按相机 X/Y/Z
三个方向分别滤波，旋转用四元数球面插值。默认针对画面中的左右、上下平移调节：
**X/Y 运动响应 beta 为 40，Z 和旋转 beta 为 0.35，静止截止频率均为 2 Hz**。
X/Y 使用米为单位的位置变化估计运动，因此它的 beta 数值与旋转的参数不能直接比较。
X/Y 根据当前位置与上次滤波位置之差的绝对值估计运动强度，再以 3 Hz 平滑该强度。
先取绝对值可避免左右掉头时正负抵消，从而避免突然加强位置平滑、产生拖后。
运动强度估计扣除 2 mm 噪声余量，减少静止抖动对响应的影响；这不是位置死区，
小于 2 mm 的真实位移仍会逐帧更新并收敛到新位置，也不进行位置预测。
运动时自动提高对应方向的截止频率；X 运动不会放松 Y、Z 或旋转的滤波。
这里的 X/Y 是相机坐标系，不是随标签旋转的局部坐标系；如果标签平面大幅倾斜，
沿标签平面移动也可能包含相机 Z 方向的位移。
标签消失超过 0.5 秒后重新初始化，避免重新出现时从旧位置慢慢拖过去。
2 Hz 是滤波截止频率，输出仍然逐个检测结果更新。

```bash
python carmer/apriltag_detector.py                              # 默认优先跟随 X/Y 平移
python carmer/apriltag_detector.py --no-smoothing               # 查看原始姿态
python carmer/apriltag_detector.py --smooth-xy-beta 25           # X/Y 更保守、更稳
python carmer/apriltag_detector.py --smooth-xy-beta 60           # X/Y 运动时跟随更快
python carmer/apriltag_detector.py --smooth-xy-noise-mm 3        # 较大噪声余量，小幅运动加速响应会减弱
python carmer/apriltag_detector.py --smooth-min-cutoff-hz 1.0    # 所有方向更稳、更慢
python carmer/apriltag_detector.py --smooth-beta 0.6             # 只提高 Z 和旋转的运动响应
```

平滑不会改变标签 ID、角点检测、相机采集帧率或像素坐标；它只改变三维姿态和坐标轴
的绘制与输出。保存的 JSON 同时包含 `raw_pose` 和平滑后的 `pose`，并记录
`smoothing_settings`、每帧按相机 X/Y/Z 顺序的 `position_alpha_xyz` 和 `rotation_alpha`。
位置滤波改为逐方向后，原来的单值 `position_alpha` 已替换为 `position_alpha_xyz`。

左右往返检查可在 CWC 中重现：`python -m carmer.check_xy_tracking`。
轨迹以 30 cm/s 在相机 X=-15 cm 和 X=+15 cm 之间往返三次，然后停止；
在 43.25 FPS 无噪声模拟中，新旧代码对比为：

| 指标 | 调整前：X/Y beta 12、带方向速度平均 | 当前默认值 |
|---|---:|---:|
| 匀速段平均位置滞后 | 5.77 mm | 3.03 mm |
| 匀速段等效延迟 | 约 19 ms | 约 10 ms |
| 掉头后 300 ms 内最大位置误差 | 13.25 mm | 3.50 mm |
| 停止后误差进入 1 mm 并保持的时间 | 66 ms | 43 ms |
| 静止白噪声输入标准差 1 mm：输出标准差 | 0.364 mm | 0.361 mm |
| 静止白噪声输入标准差 2 mm：输出标准差 | 0.753 mm | 0.877 mm |

当前首次启动时仍有约 4.85 mm 的短暂峰值误差；上表的 3.50 mm 专指掉头阶段。
较大测量噪声下，响应更快也会多通过一些噪声。无噪声停止模拟中没有越过停止位置。
这些均为滤波器模拟，不是实物运动测量，不包含采集、检测、丢帧或显示延迟；
稳定匀速时的等效延迟也不能代表掉头和启动时的响应。
对比结果保存在 `xy_tracking_check/comparison.json`，当前报告、轨迹 CSV 和图保存在
`xy_tracking_check/`，调整前的轨迹和结果保存在其 `before_fast_xy/` 子目录。
更早的 `apriltag_xy_smoothing_results.json` 是历史参数结果，不代表当前默认设置。
CWC 未安装 matplotlib 时，
可用系统已有环境绘图：`/usr/bin/python3 -m carmer.check_xy_tracking --plot-only`。
检查脚本还包含 1 mm 模拟测量噪声及 35/50 FPS 情形；所有结果均为滤波器模拟。

`--quad-decimate` 对应 OpenCV 的 `aprilTagQuadDecimate`，仅控制寻找标签边框时
的内部降采样：1.0 使用完整 640×480，1.5 约为 427×320，2.0 为 320×240。
相机采集分辨率、预览分辨率、内参和返回的像素坐标均为原始 640×480。
数值越大通常越快，但小标签更容易漏检，角点和姿态也可能有偏差。

```bash
python carmer/apriltag_detector.py --quad-decimate 1.5  # 兼顾速度与小标签
python carmer/apriltag_detector.py --quad-decimate 2.0  # 较近、较大标签：更快
python carmer/apriltag_detector.py --quad-decimate 1.0  # 较远、较小标签：完整分辨率
```

2026-10-11 在 CWC 环境对同一组 30 张实拍图（ID 0、1）各运行两轮，
检测加姿态计算平均耗时为：1.0 约 52.4 ms，1.5 约 23.1 ms，2.0 约 13.2 ms。
这三档在本组原始图上检测到的 ID 集合一致。2.5 和 3.0 已出现漏检。
将相同画面缩小至 75% 的离线检查中，1.5 仍与全分辨率检测结果一致，
2.0 开始漏检，因此当时选择 1.5。缩图检查不等同于真实远距离实测。
这些处理耗时的倒数代表处理能力，实际实时识别帧率还受相机帧率、调度和现场画面影响。
完整对比结果见 `apriltag_tuning_results.json`；记录的是相对 1.0 的差异，不能当作绝对测量精度。
选定 1.5 后，完整程序无窗口实测：采集 59.53 FPS，检测加姿态平均 44.83 FPS；
共处理 542 帧，538 帧至少有一个标签，检测到 ID 0、1。窗口刷新率需在实际显示环境中观察。

2026-10-15 在添加多 Tag 融合前的双 Tag 实景对照中，1.5 倍无窗口检测约 29 FPS，
2 倍为平均 50.09 FPS（采集 59.52 FPS，605 个处理帧均至少检测到一个 Tag）。
当前配置据此使用 2 倍；该历史结果不代表所有距离、场景或新增融合后的实时帧率保证。

不打开窗口，仅在终端查看结果：

```bash
python carmer/apriltag_detector.py --no-display --frames 600
```

该命令在采集 120 帧预热数据和 600 帧测量数据后退出。识别可能跳过旧帧，
因此处理帧数不一定等于采集帧数。连续运行可省略 `--frames`。

三维姿态使用现有 `calibrate_camera/camera_intrinsics.json` 中的 640×480
出厂内参和 10 cm 标签边长计算。在
[OpenCV ArUco 的标签中心定义](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
基础上，将局部坐标系绕 Z 轴旋转 180°，使 X、Y 的方向与用户的标签朝向一致。
原点在标签中心；以 AprilRobotics 印刷标签的正放方向为准，X 向右、Y 向上，
Z 垂直标签平面、从印刷正面向外。标签旋转时，坐标轴随标签旋转。
直接调用 `cv2.drawFrameAxes` 绘制 X 红、Y 绿、Z 蓝三根轴，再加字母标记。
Z 的“向外/向上”是相对于标签平面而言，在屏幕上投影的方向取决于标签姿态。
正对镜头时，Z 轴投影可能很短或接近一个点，这是正常的透视效果。

程序先用 OpenCV 返回的解码角点顺序求解姿态，再进行坐标变换：
`R_tag_to_camera = R_opencv_to_camera @ diag(-1, -1, 1)`。
AprilRobotics 的 tag36h11 示例图与 OpenCV 生成图相差 180°；
这次变换将原预览的 X、Y 同时反向，并保留向外的 Z 轴和右手坐标系。
旋转后的姿态同时用于绘图和 JSON 输出。标签中心仍以相机坐标表示。
测试覆盖两种来源的标签图片及四个旋转方向。
平面标签正面、远距离或角点噪声较大时，原始姿态可能抖动；默认平滑会降低这类抖动。

2026-10-11 用默认平滑参数在相机画面上采集 600 帧：采集 59.79 FPS。
ID 0 的 Z 轴末端相邻帧 95% 步长从原始 0.071 px 降到 0.009 px，
ID 1 从 0.180 px 降到 0.025 px；旋转相邻帧 95% 步长分别从 0.071° 降到 0.010°、
从 0.264° 降到 0.040°。这组数据是在标签和相机基本静止时测得，运动时滤波会自适应提高响应速度。

标签尺寸和坐标轴长度也可用米指定：

```bash
python carmer/apriltag_detector.py --tag-size 0.10 --axis-length 0.10
```

保存结果中的 `pose_convention.name` 为 `opencv_aruco_centered_z180`；
`camera_matrix` 和 `distortion_coefficients` 记录投影使用的内参。
`pose.tvec_m` 单位是米，`pose.rvec` 为弧度旋转向量，
`pose.R_tag_to_camera` 为从标签坐标系到相机坐标系的旋转矩阵：
`p_camera = R_tag_to_camera @ p_tag + tvec_m`。
相机坐标原点为光心，X 向图像右侧、Y 向图像下方、Z 从镜头朝前。
`pose.axes_px` 依次记录原点、+X、+Y、+Z 端点的图像坐标。
无法得到有效姿态时保留 ID/边框，省略该标签的三维轴。

`pose.rpy_rad` 和 `pose.rpy_deg` 分别保存弧度和角度形式的 `[roll, pitch, yaw]`，
满足 `R_tag_to_camera = Rz(yaw) @ Ry(pitch) @ Rx(roll)`。
roll/yaw 范围为 [-180°, 180°]，pitch 为 [-90°, 90°]；越过 ±180° 时数值会换号。
原始位姿 `raw_pose` 也有同样字段。程序先平滑旋转矩阵/四元数，再转换为欧拉角，
不会直接对欧拉角求平均。pitch 达到 ±90° 时 roll/yaw 无法唯一分解，程序设置
`rpy_singular=true`、按约定取 yaw=0，并保留等价旋转；此时应使用旋转矩阵。
JSON 顶层的 `pose_convention` 记录角度顺序和坐标定义，`detection_ms`、`processing_ms`
记录本帧耗时，`timing_convention` 记录计时范围。

本项目高层策略需要小车相对于机器人骨盆基座 `pelvis` 的水平航向角。
相机装在机器人头部、标签贴在小车上时，小车 yaw 必须在该参考系中计算。
相机光学 Z 轴朝前，通常不与机器人或世界竖直轴一致，因此不能直接把 `camera_rpy_deg`
中的第三个数作为小车航向。若策略需要小车相对机器人机身的姿态，应先计算：

`R_cart_to_robot = R_camera_to_robot @ R_tag_to_camera @ R_cart_to_tag`

再从该矩阵按策略约定提取 yaw。`R_camera_to_robot` 是相机安装姿态；`R_cart_to_tag`
由标签在小车上的贴装方向确定。若相机与所用机器人参考系之间存在活动关节，需使用当前
关节姿态更新外参；若策略需要世界系航向，还需组合机器人在世界系中的姿态。
`G1_URDF/g1.urdf` 已提供 `torso_link -> d435_link` 的名义安装外参：
`xyz = [0.0576235, 0.01753, 0.42987]` m，`rpy = [0, 0.8307767239493009, 0]` rad
（pitch 为 47.6°）。从 `pelvis` 到 `torso_link` 还经过 `waist_yaw_joint`，
计算相机相对 `pelvis` 的姿态时须使用与图像时刻对应的实际腰 yaw 角。
这份 URDF 的腰 roll/pitch 为 fixed。
URDF 没有 RGB `color_optical_frame`。当前按用户选择暂不标定，假设 `d435_link`
使用 X 前/Y 左/Z 上、相机实际安装符合 URDF，并补入标准光学轴变换。
默认 URDF 对应以下名义旋转（显示值已舍入；`fixed` 模式保留此常数）：

```text
R_camera_to_torso =
[[ 0, -0.738455341,  0.674302388],
 [-1,  0,            0          ],
 [ 0, -0.674302388, -0.738455341]]
```

实时模式在启动时读取配置的 URDF、缓存运动链，每帧使用实测角更新。
当前固定腰 roll/pitch 的模型等价于 `R_camera_to_pelvis = Rz(waist_yaw) @ R_camera_to_torso`。
仅在旧 `fixed` 模式使用 `--waist-yaw-deg` 的固定角；程序不发送机器人控制命令。
车尾正贴时，小车 +X（前）对应 tag -Z，小车 +Z（上）对应 tag +Y；
`top` 预设表示小车 +X 对应 tag +Y，小车 +Z 对应 tag +Z。
当前配置则显式指定 `[0, 0, 0]`，使 `R_cart_to_tag` 为单位阵，表示 Tag +X 与小车 +X 一致。
对于其它贴装方向，可以通过
`--cart-to-tag-rpy-deg ROLL PITCH YAW` 覆盖预设，单位为度，定义为
`R_cart_to_tag = Rz(YAW) @ Ry(PITCH) @ Rx(ROLL)`，表示把小车坐标转换到标签坐标，
不是相机的安装角。标签轴始终沿用本程序的印刷图案约定。

程序先计算完整的 `R_cart_to_pelvis`，取第一列作为小车前向 `f`。
令 pelvis 中的单位向上向量为 `u`，水平参考前向为
`x = normalize([1,0,0] - u[0]*u)`，水平左向为 `y = cross(u,x)`，
最终 `yaw = atan2(dot(f,y), dot(f,x))`。它不需要世界绝对 yaw 或位置。
只有 `fixed` 模式仍直接使用 `atan2(f[1], f[0])`。
小车前向或 pelvis 前向的水平投影退化时，输出 `valid=false`、角度为 `null`，
而不是把无定义的航向当作 0。仅求航向不需要相机光心的平移外参；本次没有新增小车位置输出。

保存的 JSON 中，每个符合 ID 筛选的标签增加：

- `cart_heading`：实时补偿后的平滑航向，包含 `valid`、`yaw_rad`、`yaw_deg`、
  `reference_frame`、`reason`、`R_cart_to_pelvis` 和小车前向向量。
  其中矩阵/向量保留本帧原始补偿值，仅 yaw 字段做航向低通；不会反改显示用 `pose`。
- `raw_cart_heading`：从未经平滑的位姿计算，用于比较滤波前后结果。
- 顶层 `cart_heading_settings`：记录名义外参、贴装旋转、姿态来源、融合/同步配置和坐标约定。
- 顶层 `pelvis_compensation`：记录本帧 `valid`、`reason`、配置/实际姿态来源、实测腰角、
  补偿 roll/pitch、向上向量、相机到 pelvis 的旋转、状态年龄、同步误差和匹配方式。
  运动学/融合模式还记录假设的支撑脚及可用的方向差异。
- 顶层 `fused_cart_heading`：融合后的最终结果，包含 `valid`、`yaw_rad`、`yaw_deg`、
  `reference_frame`、`mode`（`fused`/`single`/`unavailable`）、`tag_ids`、`tag_count`、`reason`。
  `tag_count` 是参与融合的有效航向数；`resultant_length` 是圆周平均向量长度，用于描述方向一致程度，
  不是经过标定的置信概率。`method` 为 `equal_weight_circular_mean`。
- 顶层 `raw_fused_cart_heading`：使用各 Tag 经实时骨盆补偿、尚未低通的航向进行同样融合。
  `heading_udp.py` 将此值送入 V8.1 策略，与训练的相机延迟/采样保持模型对齐；
  `fused_cart_heading` 的低通结果继续用于相机显示。

标签存在但位姿无效时，该标签航向 `valid=false`；标签消失时，本帧 `tags` 不再包含它。
使用融合结果的程序应检查所选字段的 `valid`（策略使用 `raw_fused_cart_heading.valid`）和帧时间戳，不能复用旧 JSON
文件来判断当前仍检测到小车。`Cam RPY` 仍只表示标签相对于相机的角度。

离线验证识别、印刷标签坐标方向、透视、尺度、内参和保存结果：

```bash
python -m unittest -v carmer.test_pelvis_compensation carmer.test_apriltag_config_fusion carmer.test_apriltag_detector carmer.test_camera_open
```
# 向推车高层发布航向

从原 Viser 网页点击“连接 / 重试小车相机”，复用本文的相机和姿态补偿，
将有效融合航向发布到本机 UDP 9870。诊断时也可给检测器传 `--heading-udp-port 9870`。
默认端口为 0（不发布），因此原有相机用法保持不变。
协议实现在 `heading_udp.py`：保留图像单调时间戳和相机延迟，必须是有效的
`pelvis_level` 航向；无标签/补偿失效会发布无效标记，不伪造零航向。
本机高层用法见 `../G1_high_policy_deploy/README.md`。

