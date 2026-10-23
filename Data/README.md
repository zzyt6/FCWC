# 推车数据记录说明

本目录保存推车过程的分析数据。**每次“启用推车策略”到“关闭推车策略”自动生成一个子文件夹**，网页控制模式与导航模式都会记录，无需额外操作。

当前格式为 `pushcart_data_v2`：**每次模型推理完成后，将本次推理使用的电机反馈与策略输出各保存一行**，频率跟随策略（通常 50 Hz）。电机和手部反馈直接复用现有控制桥接，不再启动独立 DDS 采集进程；CSV 格式化和写盘在控制进程的后台线程中完成。无推理的控制拍不写入这组 CSV。

导航的里程计、实测速度和 `/cmd_vel` 由现有 `navigation_web_bridge` 进程记录：复用机身里程计回调，另订阅 `/Odometry` 和 `/cmd_vel`，回调只将消息及接收时间放入有界队列，后台线程处理记录窗口、格式化和写盘。不再启动独立 ROS 记录进程。这三路数据仍按实际消息频率保存，不补齐到 50 Hz。

相机仅记录标签检测结果和时间戳；不录制视频，不保存逐帧图像，也不缓存用于录像的原始帧，以减少编码、内存和磁盘开销。相机仍正常运行 AprilTag 检测，供推车策略使用。

```text
Data/
├── 2026-10-24_15-30-12_web/    # Viser 网页滑条控制（bash Viser/run.sh）
└── 2026-10-24_16-02-45_nav/    # Nav2 导航控制（bash nav/run_navigation.sh）
```

- 文件夹名是推车启用时的**本机本地时间**，后缀 `_web` / `_nav` 区分控制方式；同一秒内重复启用会追加 `_2`、`_3`。
- 每个文件夹只保存时间窗口 **[启用时刻 − 1 s, 关闭时刻 + 1 s]** 内的样本（前后各多记 1 秒）。程序退出时若推车仍在运行，以退出时刻为关闭时刻，不再多记。
- 推车中途因相机/抓握失效被系统暂停，也算一次“关闭”；再次启用会生成新的文件夹。
- 本目录不会自动删除旧数据，请自行整理磁盘。仅本说明文件纳入 git。

## 1. 开关与运行

| 模式 | 默认 | 关闭方法 |
|---|---|---|
| 网页控制 | `--mode control` 时自动记录 | `bash Viser/run.sh --pushcart-data off` |
| 导航 | 自动记录 | `bash nav/run_navigation.sh pushcart_data:=off` |

- 只读模式（observe）不能启用推车，不产生记录；离线模式（sim）默认不记录，需要时用 `--pushcart-data-dir <CWC/Data 以外的目录>`。
- Viser 网页“④ 可选：推车策略”状态区显示“数据记录：正在记录 Data/…”或“上次记录已保存…”。终端/launch 日志里以 `PUSHCART DATA` 开头的行记录开始、结束、保存状态和警告。
- 每次记录结束后先看 `meta.json` 的 `status`：`complete` 表示所有数据流都确认已关闭文件；`incomplete` 表示有数据流没有响应，见 `meta.json` 的 `streams`。

## 2. 文件清单

| 文件 | 频率 | 内容 | 来源 |
|---|---|---|---|
| `meta.json` | — | 参数、时间窗口、时钟对照、关节顺序、各数据流状态 | 控制进程 |
| `policy.csv` | 50 Hz | 推车高层的输入/输出：任务速度 vx/vy/wz、小车 yaw、航向保持、相机帧时间 | 控制进程，每个策略拍一行 |
| `joint_command.csv` | 50 Hz | 关节指令：策略目标、桥接实际目标、Kp/Kd、最近一次 DDS 下发值 | 控制进程 |
| `motor_state.csv` | 与推理同频，通常 50 Hz | 29 个机身电机 q/dq/ddq/tau_est/vol/mode 与骨盆 IMU | 本拍推理使用的控制桥接反馈 |
| `hand_left_state.csv` / `hand_right_state.csv` | 与推理同频，通常 50 Hz | Dex3 每只手 7 个电机 q/dq/ddq/tau_est、供电电压/电流、9×12 压力 | 同一桥接快照中的左右手反馈 |
| `camera_detections.csv` | 检测频率 | 每个被检测帧识别到的标签和小车相对骨盆 yaw | 相机进程 |
| `odometry_fastlio.csv` | 雷达频率 | 仅导航：FAST-LIO `/Odometry` 位姿 | 导航网页桥接进程中的 ROS 订阅 |
| `odometry_base.csv` | 雷达频率 | 仅导航：`/Odometry_base` 位姿与**机身系平面速度（实测）** | 复用导航网页桥接的机身里程计回调 |
| `cmd_vel.csv` | Nav2 频率 | 仅导航：Nav2 原始 `/cmd_vel`（限幅前） | 导航网页桥接进程中的 ROS 订阅 |
| `motor_summary.json` 等 `*_summary.json` | — | 各数据流自己的行数、首末时间、关闭原因、丢失计数 | 各记录程序 |

**没有记录的内容：** 电机电流（SDK 的 `MotorState_` 没有电流字段，`tau_est` 是电机估计力矩，不等于电流）；电机温度和错误码（按要求不记录）。Dex3 的 `power_a` 是整只手的供电电流，不是单个电机电流。网页模式没有机身实测速度来源（无 `odometry_*`）；Homie 估计器的速度只是网络估计，不是测量。

单位：角度 rad，角速度 rad/s，线速度 m/s，力矩 N·m，电压 V，电流 A，时间 s。空单元格表示该拍没有这个值（例如推车高层未推理）。

开启采集时，控制桥接通过原有 UDP 状态包的可选只读扩展发送加速度、力矩、电压和手部压力。未开启采集时不发送此扩展。仿真或旧桥接未提供的扩展字段留空；手部尚无反馈时，其角度、速度和接收时间也留空。保留过期反馈时保留原接收时间，并通过 `healthy` 标记手部状态，不把旧数据标成新采样。

## 3. 字段说明

### 3.1 所有 CSV 共有的时间列

| 列 | 含义 |
|---|---|
| `t` | **对齐用的统一时间轴**：本机 `CLOCK_MONOTONIC`，单位 s，精确到 µs。各文件 `t` 的具体含义见下方各表 |
| `t_rel` | `t − meta.episode.start_monotonic`，即相对推车启用时刻的秒数；前 1 s 为负 |

### 3.2 `policy.csv`（50 Hz，每个策略拍一行）

| 列 | 含义 |
|---|---|
| `t` | 本拍关节指令发给 C++ 桥接后立即取的时间（≈ 指令发出时刻） |
| `state_received_t` | 本拍使用的机身反馈被 C++ 桥接收到的时间 |
| `inference_done_t` | 本拍 Balance/Homie 推理完成时间 |
| `camera_captured_t` | 本拍推车高层所用相机帧的采集时间（与 `camera_detections.csv.t` 同一时钟） |
| `state_session`, `state_sequence` | 桥接反馈包编号 |
| `stage` | 控制阶段（`POLICY` 为策略运行） |
| `pushcart_enabled` | 1 = 推车策略启用；前后 1 s 的预留部分为 0 |
| `motion_blocked` | 1 = 推车被系统暂停（相机/抓握失效） |
| `inference_role` | `active` = 高层输出被执行；`shadow` = 未启用时的影子推理 |
| `selected_model` | 下层实际执行的模型（`balance` / `homie`） |
| `command_source` | 下层速度命令来源；推车时为 `PUSHCART`，原始输入来自哪里看文件夹后缀 / `meta.input_mode` |
| `decelerating` | 1 = 正在按 0.4 s 滤波平滑减速 |
| `raw_task_vx/vy/wz` | **原始任务速度**：网页滑条或导航给出、经任务限幅后的 vx/vy/wz |
| `task_vx/vy/wz` | **送入推车高层策略的任务速度**：原始任务速度经小车航向保持修正 wz 后 |
| `task_ref_vx/vy/wz` | 高层内部的一阶滤波任务参考 |
| `homie_cmd_vx/vy/wz` | 推车高层输出、送给下层 Homie 的速度命令 |
| `policy_cmd_vx/vy/wz` | 下层实际使用的速度命令（选 Balance 时为 0） |
| `filtered_cmd_vx/vy/wz` | 置零前的下层命令（模型选择依据） |
| `camera_yaw_rad` | **推车高层实际输入的小车 yaw**：小车相对骨盆的 yaw（已减去航向零位偏置） |
| `camera_raw_yaw_rad`, `camera_yaw_offset_rad` | 相机原始相对 yaw、零位偏置（`camera_yaw_rad = raw − offset`） |
| `camera_fresh`, `camera_status`, `camera_age_s`, `camera_tag_ids`, `camera_pose_source` | 相机 yaw 是否有效、状态、数据年龄、所用标签（`;` 分隔）、位姿来源 |
| `cart_yaw_rad` | 小车绝对 yaw = 相机采集时刻的骨盆 IMU yaw + 相对 yaw |
| `imu_yaw_rad`, `imu_yaw_at_camera_rad` | 当前骨盆 IMU yaw；插值到相机采集时刻的骨盆 yaw |
| `hold_ref_rad`, `hold_err_rad`, `hold_corr_rad_s` | 小车航向保持的参考、误差、加到 wz 上的修正 |
| `inference_ms`, `high_policy_inference_ms`, `policy_hz`, `state_age_ms` | 推理耗时、策略频率、反馈年龄 |

### 3.3 `joint_command.csv`（50 Hz，与 `policy.csv` 逐行同一拍）

| 列 | 含义 |
|---|---|
| `t`, `t_rel`, `state_received_t` | 同 `policy.csv` |
| `pd_profile`, `pd_transition_blend`, `upper_pd_mode` | 下层 PD 组、增益过渡进度、上肢 PD 模式（`pushcart` / `hold`） |
| `target_req_<关节>` | Python 本拍请求的关节目标（推车上肢为插值终点） |
| `target_<关节>` | C++ 桥接回报的实际目标 |
| `kp_<关节>`, `kd_<关节>` | 实际 Kp / Kd |
| `sent_<关节>` | C++ 最近一次成功写入 DDS 的电机指令（27 个使能机身关节 + 14 个手指） |
| `sent_body_t`, `sent_hand_left_t`, `sent_hand_right_t` | 上述三组指令的写入时间 |

关节名按 SDK 编号 0–28 排列，见 `meta.json` 的 `joint_names.body_sdk29`。外层采用单腰结构，SDK 13/14（`waist_roll`/`waist_pitch`）不使能。

### 3.4 `motor_state.csv`（与策略推理同拍）

| 列 | 含义 |
|---|---|
| `t`, `t_rel`, `inference_done_t` | 与 `policy.csv` 对应行相同的指令时刻、相对时间和推理完成时间 |
| `state_received_t` | 本拍推理实际使用的 LowState 被控制桥接接收的时间，同 `policy.csv.state_received_t` |
| `state_session`, `state_sequence` | 本拍使用的桥接反馈标识，同 `policy.csv` |
| `tick` | LowState 自带的机器人侧计数；按推理采样会自然跳过中间的 LowState，不再作为逐帧 DDS 丢包统计 |
| `mode_pr`, `mode_machine` | SDK 模式字段 |
| `q_<关节>`, `dq_<关节>`, `ddq_<关节>` | 位置、速度、加速度 |
| `tau_est_<关节>` | 电机估计力矩 |
| `vol_<关节>` | 电机电压 |
| `mode_<关节>` | 电机模式 |
| `imu_qw, imu_qx, imu_qy, imu_qz` | 骨盆 IMU 四元数（SDK 顺序 w, x, y, z） |
| `imu_gyro_x/y/z`, `imu_acc_x/y/z`, `imu_roll/pitch/yaw` | 角速度、加速度、欧拉角 |

按列名前缀筛选时注意 `dq_` 也含 `q_`，用正则：`df.filter(regex='^q_')`。沿用控制桥接的 CRC 和重复/倒退 tick 检查。若相邻推理使用同一条反馈，`state_received_t` 和 `tick` 保持不变，记录的是每次推理使用的数据。

### 3.5 `hand_left_state.csv` / `hand_right_state.csv`

| 列 | 含义 |
|---|---|
| `t`, `t_rel`, `inference_done_t` | 与 `policy.csv` 对应行相同 |
| `hand_received_t` | 对应手部原始反馈的接收时间；左右手异步发布，可能不同 |
| `healthy` | 控制桥接报告的该手部反馈健康状态（包含新鲜度） |
| `topic` | 控制桥接实际选用的反馈话题，沿用优先 `rt/lf/dex3/<side>/state`、失效时回退 `rt/dex3/<side>/state` 的逻辑；每拍只记录选中的一路 |
| `q_/dq_/ddq_/tau_est_<side>_hand_<关节>_joint` | 7 个手指电机：`thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1` |
| `power_v`, `power_a`, `system_v`, `device_v` | 整只手供电电压/电流、系统电压、设备电压 |
| `press_sensor_count` | 实际收到的压力传感器数量（正常为 9） |
| `press_<k>_<i>` | 第 k 个压力传感器（0–8）的第 i 个测点（0–11）；缺失传感器为 `nan` |

### 3.6 相机标签结果：`camera_detections.csv`

每个实际检测帧一行，只包含数值和文本：

| 列 | 含义 |
|---|---|
| `camera_frame_count` | 相机进程的采集帧计数；检测会跳过来不及处理的帧，因此不保证连续 |
| `t` | 相机驱动（V4L2）给该帧的主机采集时间 |
| `tag_ids` | 识别到的标签 ID，用 `;` 分隔 |
| `cart_yaw_valid` | 小车航向及骨盆姿态补偿是否有效 |
| `cart_yaw_pelvis_rad` | 融合后的小车相对骨盆 yaw（未减零位偏置） |
| `reason` | 航向结果的状态说明 |
| `detection_ms`, `processing_ms` | 检测和处理耗时 |

相机数据流在 `meta.json` 中记为 `camera`，结束统计为 `camera_summary.json`。当前记录不再生成 `video.mp4`、`video_frames.csv` 或 `video_summary.json`，已有历史文件不受影响。

### 3.7 导航数据（仅 `_nav`）

`odometry_fastlio.csv`、`odometry_base.csv`：

| 列 | 含义 |
|---|---|
| `t` | 导航网页桥接的 ROS 回调收到消息的时间（monotonic，与其它文件对齐），不是后台写盘时间 |
| `wall_t` | 收到时的系统墙上时间 |
| `stamp_t` | 消息头时间戳（ROS/雷达时钟，**不能直接与 `t` 比较**，仅供参考） |
| `frame_id`, `child_frame_id` | 坐标系 |
| `x, y, z, qw, qx, qy, qz, yaw` | 位姿 |
| `vx, vy, vz, wx, wy, wz` | twist |

`/Odometry`（FAST-LIO）的 twist 恒为 0；**实测机身速度用 `odometry_base.csv` 的 `vx, vy, wz`**（`base_nav` 机身系，由相邻位姿差分并滤波得到）。`cmd_vel.csv` 是 Nav2 原始输出，之后还要经过 `g1_cmd_vel_udp_node` 限幅；实际进入推车高层的是 `policy.csv` 的 `raw_task_*`。

`pushcart_data:=on/off` 同时控制这三路记录。后台队列最多保留 512 条消息，满时丢弃新记录并累计到 `ros_summary.json.dropped_before_write`；磁盘错误记录在 `writer_error`。关闭推车时通过原有记录窗口协议结束文件，退出导航网页桥接时排空队列并关闭。记录模块不增加 ROS 节点、不发布速度指令，也不改变导航里程计有效性检查。

### 3.8 `meta.json`

主要字段：`schema`、`status`、`input_mode`（web/nav）、`episode`（`start_monotonic`、`stop_monotonic`、`stop_reason`、`pre_s`、`post_s`）、`clock.pairs`（monotonic 与墙上时间对照，开始和结束各一组）、`motor_sampling`（反馈来源、采样时机、缺失字段约定）、`model_set`、`high_policy`（模型版本、学生模型 SHA256、任务限幅、航向零位）、`joint_names`、`units`、`streams`、`files`、`git`（代码版本、是否有未提交修改）。`streams.motor` 在本进程关闭文件后确认完成，不再等待独立电机记录器回复。

## 4. 时间对齐

### 4.1 为什么可以对齐

所有记录程序都运行在机器人这台 Jetson 上，使用同一个 `CLOCK_MONOTONIC`。策略、关节指令、电机和双手 CSV 的每一行来自同次推理，共用 `t`，可直接按行或按 `t` 连接。电机/手部实际采样时间保存在独立的接收时间列中；相机与 ROS 数据仍按各自 `t` 对齐。这个时钟从开机开始计数，不受联网对时跳变影响。

换算成墙上时间：`wall = t + (pair.realtime_unix − pair.monotonic)`，`pair` 取 `meta.json` 的 `clock.pairs[0]`。机器人重启后 monotonic 会归零，不同文件夹之间请用文件夹名或墙上时间比较先后。

### 4.2 各时间戳的物理含义与误差

| 时间戳 | 代表的时刻 | 误差来源 |
|---|---|---|
| `policy.t` / `motor_state.t` / `hand_*.t` | 同一拍关节指令发出 | 完全相同的记录时刻，不能代替传感器采样时间 |
| `policy.state_received_t` / `motor_state.state_received_t` | 控制桥接收到本次推理使用的 LowState | 两列完全相同；与传感器实际测量时刻仍有 DDS 传输延迟 |
| `hand_*.hand_received_t` | 控制桥接收到选中的手部反馈 | 左右手异步；结合 `healthy` 判断新鲜度 |
| `policy.camera_captured_t`、`camera_detections.t` | 内核收到该图像帧 | **比真实曝光时刻晚一个大致固定的延迟**（曝光 + 读出 + USB 传输，通常几到二十几 ms，尚未标定） |
| `odometry_*.t`、`cmd_vel.t` | 主机收到 ROS 消息 | 雷达处理和 ROS 传输延迟，几 ms 到几十 ms |

### 4.3 常用对齐方法（pandas）

```python
import json
import pandas as pd

folder = 'Data/2026-10-24_15-30-12_web'
meta = json.load(open(f'{folder}/meta.json'))
policy = pd.read_csv(f'{folder}/policy.csv')
motor = pd.read_csv(f'{folder}/motor_state.csv')
detections = pd.read_csv(f'{folder}/camera_detections.csv')

# 1) 只保留推车运行区间（去掉前后各 1 s 预留）
t0, t1 = meta['episode']['start_monotonic'], meta['episode']['stop_monotonic']
policy_run = policy[(policy.t >= t0) & (policy.t <= t1)]

# 2) 策略和电机反馈是同一拍，无需最近邻或插值
merged = policy.merge(motor, on='t', validate='one_to_one', suffixes=('', '_motor'))
assert (merged.state_received_t == merged.state_received_t_motor).all()

# 3) 本拍实际使用的反馈年龄
feedback_age_ms = (motor.t - motor.state_received_t) * 1000

# 4) 推车高层这一拍使用的标签结果（carmer/apriltag_config.json 的 camera-latency-ms 为 0 时，
#    camera_captured_t 与该帧的 camera_detections.t 完全相同；若改为非 0，需先加回该值）
policy_detections = pd.merge_asof(policy.sort_values('camera_captured_t').dropna(subset=['camera_captured_t']),
                                  detections.sort_values('t'), left_on='camera_captured_t', right_on='t',
                                  direction='nearest', tolerance=0.002, suffixes=('', '_detection'))
```

导航数据同理：`odometry_base.csv` 按 `t` 用 `merge_asof` 或插值对到 `policy.t`，就能逐拍比较指令速度（`raw_task_*` / `homie_cmd_*`）与实测速度（`vx, vy, wz`）。比较时注意里程计本身有处理延迟，可用互相关估计并补偿这段滞后。

### 4.4 更高精度（可选）

- **传感器时刻：** 如需按电机/手部接收时刻分析，使用 `state_received_t` / `hand_received_t`；相邻推理可能复用同一采样，不要把重复的接收时间当成更高采样频率。
- **对齐区间：** 各数据流都在同一个窗口 `[start − 1 s, stop + 1 s]` 内，但首末样本取决于各自频率；跨文件分析前按 `meta.json` 的时间裁剪到同一区间。

## 5. 数据完整性检查

- `meta.json`：`status` 为 `complete`；`streams.<name>.files_closed` 均为 true；`streams.policy.dropped_ticks_total` 为 0。
- `motor_summary.json`：`source` 为 `policy_feedback`；三个电机/手部 CSV 的行数应与 `policy.csv`、`joint_command.csv` 一致，`dropped_before_write` 为 0。`tick` 跳号是按推理频率取样的正常结果。
- `camera_summary.json`：`rows` 等于 `camera_detections.csv` 数据行数；`dropped_before_write` 为 0。检测帧编号不连续是正常情况，不代表记录队列丢失。
- 程序被 `kill -9` 或断电时，CSV 可能缺少尚未刷入磁盘的数据；`meta.json` 可能停在 `recording` 状态。

## 6. 数据量

电机与双手从按 DDS 发布频率记录改为通常 50 Hz，每次推理各写一行；数据量随实际推理次数增长，不再随高频 DDS 消息数量增长。实际大小取决于字段内容与运行时间，以生成文件为准。

## 7. 实现位置

| 部分 | 文件 |
|---|---|
| 记录协调及策略/电机/手部 CSV | `Viser/core/pushcart_data.py`（由 `Viser/core/controller.py` 传入本拍反馈，后台线程写盘） |
| 相机与 ROS 时间窗口控制（本机 UDP 39011、39012） | `Viser/core/data_episode.py` |
| 现有桥接的只读反馈扩展 | `G1Deploy_BalanceHomie/cpp/motor_telemetry.hpp` 与 `python/hybrid_deploy/protocol.py` |
| 相机标签文本结果 | `carmer/episode_detections.py`（在相机进程内运行，不接收或保存图像） |
| 导航 ROS 数据 | `nav/src/unitree_nav2_web/unitree_nav2_web/episode_data.py`（由现有 `nav_web_bridge.py` 创建后台线程） |
| 测试 | `Viser/test/test_pushcart_data.py`、`Viser/test/test_motor_telemetry.py`、`carmer/test_episode_detections.py`、`nav/src/unitree_nav2_web/test/test_episode_data.py` |

记录逻辑只读取控制桥接反馈，不增加机器人指令。独立电机进程、UDP 39010 和 `G1_DATA_CPUS` 设置已移除。更新后完整重启 Viser 或导航，启动脚本会重新构建桥接。历史 `pushcart_data_v1` 目录保留原格式，其电机 CSV 的 `t` 含义与当前版本不同，分析前先检查 `meta.json.schema`。
