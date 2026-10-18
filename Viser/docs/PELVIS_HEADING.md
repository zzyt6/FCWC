# 骨盆 IMU 与实时 wz 修正

Viser 手动模式以 IMU 处理、实时命令接入为主：前进、后退、侧移、斜走均可保持骨盆朝向，主动旋转与边走边转时释放旧航向。`nav/` 复用本控制器时采用随导航转弯推进的参考航向，详见下文。没有增加 XY 闭环。尚未做真机验证。

## 1. 实际链路与审计

| 环节 | 代码位置 / 含义 |
| --- | --- |
| 接收 | `G1Deploy_BalanceHomie/cpp/g1_control.cpp::Robot::state`，订阅 `rt/lowstate`，CRC 校验，完整 LowState 的 quaternion/gyroscope/rpy 在同一互斥锁中复制 |
| 缓存 | C++ `StatePacket` → 本机 UDP → `Viser/backends.py::Bridge.read`；最新完整包，session/sequence 检查，不混拼姿态与 gyro |
| 来源依据 | 本机 SDK 示例 `/home/unitree/code/unitree_sdk2/example/g1/low_level/g1_ankle_swing_example.cpp` 将 LowState 标为 pelvis、secondary_imu 标为 torso；实际编译使用部署包 vendor SDK，字段为 HG LowState.imu_state |
| 安装/根系 | `G1_URDF/g1.urdf::imu_in_pelvis_joint` 父节点 pelvis，固定 rpy=0；Homie deployment_contract 指定 imu_in_pelvis。没有再次减腰 yaw 或做 torso→pelvis 变换 |
| 输入/仲裁 | `Viser/Vender/app.py` 的滑条/按住事件 → `hybrid_deploy/web_control.py::WebControl`；或 `hybrid_deploy/policy.py::Commands` 的遥控摇杆 |
| 命令入口 | `Viser/core/controller.py::Controller.step` 在模型选择、`Observations.push` 和推理前调用唯一的 `HeadingControl.update` |
| 新 IMU 分支 | `Viser/core/pelvis_imu.py::PelvisImu.update`，独立校验、ZYX 姿态/角速度和低通；不写入原状态数组 |
| 原观测 | `balance_observations.py`、`homie_observations.py` 保持原样：原 gyro ×0.5、原四元数生成投影重力，历史保持六帧 |
| 后端执行 | 两套 ONNX → 原动作 ×0.25 与关节目标限制 → `Bridge.send` → 原 C++ safety/control core、200 Hz 伺服与 PD 切换 |

旧实现已确认的问题：转弯时积分用户 wz 到航向目标，旧航向误差仍参与修正；启动便捕获目标而非进入平移保持时捕获；没有新样本角速度处理、姿态跳变恢复与主动转弯释放；命令前级按固定 0.02 s 斜坡而后级又做 XY 斜坡；默认自动纠偏。现在替换旧入口，移除旧修正低通和重复 XY 斜坡。历史 world XY 旋转错误在前一次修改已修复，此次不恢复。

这些是实现差异及确定的时间/模式问题，不能据此认定它们解释全部真机误差。实际 RL 转向延迟、骨盆周期摆动、陀螺零偏、硬件姿态滤波及物理正负响应仍需测量。旧实机 Kp=1 的结果不能直接证明新状态机的最佳增益。

## 2. 坐标、单位与实时处理

- 四元数 `w,x,y,z`，沿用原重力处理的 `R_WB=Rz(yaw)Ry(pitch)Rx(roll)` 约定；绕 +Z 逆时针 yaw 为正。世界参考由 SDK 提供，不能声称它是无漂移北向参考。
- 检查有限值及 0.8 < norm < 1.2，再归一化副本。q/-q 等价。yaw 不积分 gyro、不做额外低通；误差通过 atan2(sin,cos) 取最短角。通过相邻 yaw 增量与同期 gyro×dt 的差检测参考跳变，±π 环绕不触发跳变。
- gyro 是桥接直接复制的骨盆轴系 rad/s，没有部署侧减偏置或缩放；硬件内部偏置补偿细节无法从接口确认。本次不学习零偏，现有包也缺少加速度统计，不能把“命令零”视为可信静止。
- 分别记录 body-z 分量 r、world-z 投影、Euler yaw rate：`(sin(roll)*q + cos(roll)*r)/cos(pitch)`。航向阻尼只使用后者的低通值。三者在倾斜时不同。
- 自定义训练奖励源文件未随部署导出物提供，不能严格核实 `wz` 跟踪哪一种角速度。主路径在 |roll|、|pitch|≤0.35 rad 内使用集中适配函数 `heading_to_policy_wz` 的小倾角近似。超界仅退出新增补偿，保留原倾斜安全阈值；临近奇异姿态不放大反馈。不是任意姿态的严格映射。
- `rate_execution_target` 保留前馈原值；独立角速度跟踪默认关闭，配置试图启用会拒绝并说明训练定义未核实。因此主动转弯保留策略执行，不能宣称转速已被新闭环精确跟踪。
- 网页直接给 m/s、rad/s；实体摇杆先死区、再乘各轴正/负物理上限。新控制器在物理量阶段修正 wz，之后两个观测构造仅各乘一次 `[2,2,0.5]`，没有重复乘速度上限。gyro 反馈不读取乘过 0.5 的策略观测。

**时间与频率：**配置策略 50 Hz、桥接伺服 200 Hz，实际频率仍看日志。`received_ns` 是 C++ 同机 steady_clock 主机接收时刻；Python 使用 monotonic，测试可注入控制时钟。硬件采样时间未提供，`imu_sample_time=null`，`imu_time_basis=host_receive_proxy`。`imu_age` 仅是接收到使用的年龄，不是传感器端到端延迟。`consumed_sample_hz` 是 Python 消费的新状态间隔倒数，不是原生 IMU 频率。

滤波按相邻**新包接收时间差**，控制积分与斜坡按**控制时刻差**。相同 session/sequence 缓存不重复更新；主机时间倒退、序号倒退、长缺帧/停更走共享反馈故障。C++ 不再让重复或倒退的 LowState tick 刷新接收时间，正常 uint32 环绕允许；底层 tick 重置会超时停步，需重启会话。硬件采样时间映射仍不可用。

角速度低通为 `a=exp(-2*pi*f*dt_imu)`，首个有效样本直接初始化。默认 f=3 Hz（单独低通时间常数约 53 ms）；逐样本检查 f < 采样 Nyquist，采样不足则退出修正并等待恢复。不要再叠加旧纠偏低通。降低截止频率会增加反馈延迟；P 使用当前姿态，不受这个角速度低通拖慢。滤波不会消除 IMU 自身的长期 yaw 漂移。

## 3. 控制与配置

| 状态 | 行为 |
| --- | --- |
| STOPPED | 无附加纠偏，释放参考/积分；保留原停车规则 |
| HOLD | 手动有效平移，无主动转弯：仅进入时捕获 yaw_ref；前后左右、斜走一致 |
| TURNING | 使用补偿前原命令判定转弯，立即释放旧参考及 I/D；原地旋转也属于该状态 |
| REACQUIRE | 转向回零去抖，角速度减小后重锁；最大等待 0.6 s，慢漂不会导致无限等待；异常残余转速请求原安全停止 |
| TRACKING（NAVIGATION） | 非零导航命令进入；参考 yaw 按导航角速度推进，不使用手动转向死区；来源改变重新捕获参考 |
| PUSHCART | 学生之前已完成小车任务航向修正；下层原样传递 HOMIE 指令，继续提供 IMU 检查和停车减速 |
| EXTERNAL / UNKNOWN | 不启用航向补偿，wz=0 也按来源保留；返回 MANUAL 捕获当前朝向 |
| RECOVERING | 释放旧参考，按 wz 变化率退出附加修正；健康窗口完成并回零后重新使能 |

`e_eff=sign(e)*max(abs(e)-deadband,0)`，`request=Kp*e_eff + I - Kd*filtered_yaw_rate`。只有 HOLD 使用该控制律；修正以 0.3 s 增益接入、限幅 ±0.15 rad/s，然后加原用户 wz，最终限幅和变化率限制。I 采用连续死区、输出限幅以及基于**实际最终应用值**的条件积分，考虑修正饱和、最终饱和和斜坡。shadow 冻结 I；切换配置清除旧状态，不能把 shadow 积分带入 active。

TRACKING 先以当前 yaw 建立参考，之后 `yaw_ref += previous_wz_nav * dt`，其中 `wz_nav` 是前级斜坡后的导航角速度。最短角误差限于 ±`navigation_max_heading_error_rad`（默认 0.20 rad），避免不可执行的参考持续累积。补偿为 `Kp*e_eff + I + Kd*(wz_nav-filtered_yaw_rate)`，加回 `wz_nav` 后限幅并约束最终变化率；沿用 HOLD 的接入过渡和抗积分饱和。跟上导航转弯时误差接近零，不用固定的旧航向抵消 DWB。小倾角近似和训练定义限制仍适用，不等同于精确角速度执行环。停车和来源切换清除参考与积分。

PUSHCART 的独立 `CartHeadingHold` 位于 `G1_high_policy_deploy/robot/pushcart_v81/viser_policy.py`，在任务参考滤波和学生之前执行。
测量为骨盆 IMU yaw 加已标零的相机相对 yaw，短暂丢帧使用保持值，超时继续走原暂停流程。
起步捕获当前小车朝向，参考按限幅后的原始小车任务 wz 乘实际 monotonic 控制间隔推进。
最短角误差限幅 ±0.3 rad，达到限幅时参考跟随测量拖动；Kp=1.0，修正限幅 ±0.15 rad/s，修正后的任务 wz 限于 ±0.6。
不加积分或额外低通；零任务/暂停复位。普通行走的网页/启动参数只控制骨盆环，不改变这个任务环。
正常推车 Homie 执行时 `filtered_velocity` 和 `velocity` 等于学生的 `homie_command`，下层修正为零。
实际选择的命令回写高层和低层历史；停止沿原 0.4 秒滤波减速，Balance 接管后当前命令严格为零。

速度命令 vx/vy/wz 均沿用前级 0.4 秒滤波，不重复过滤。active 航向分支仅对附加纠偏使用 0.4 rad/s² 变化率限制，再与已滤波的 wz 相加；disabled/shadow 原样返回前级命令。安全停止直接覆盖。

普通行走使用 `config/motion_config.json`，默认 active；导航骨盆环独立使用 `nav/locomotion/motion_config.json`，默认 active，Kp=1、Ki=Kd=0。范围在 `heading.validate_settings` 校验：

| 参数 | 初始配置 | 单位/目的 |
| --- | ---: | --- |
| run_mode | active | disabled 关闭；shadow 仅建议；active 才应用 |
| heading_kp / ki / kd | 1.0 / 0 / 0 | s⁻¹ / s⁻² / 无量纲；非关节 PD |
| gyro_cutoff_hz | 3 | Hz，仅独立角速度反馈 |
| max_correction_rad_s / integral_limit_rad_s | 0.15 / 0.04 | rad/s |
| heading_deadband_rad | 0.00873 | 0.5°，连续死区 |
| imu_timeout_s / imu_recovery_s | 0.1 / 0.25 | s；共享失效停止，恢复不带旧参考 |
| translation_enter/exit_m_s | 0.025 / 0.015 | 原用户 XY 范数滞回 |
| turn_enter/exit_rad_s | 0.035 / 0.015 | 原用户转向意图滞回 |
| turn_release_s / settle_max_s | 0.12 / 0.6 | 回零去抖/最大等待 |
| settle_rate / residual_stop_rad_s | 0.08 / 0.8 | rad/s，正常重锁与异常残余旋转 |
| engage_ramp_s | 0.3 | s，接入修正过渡 |
| rate_control_enabled | false | 训练语义未确认，拒绝启用 |
| navigation_max_heading_error_rad | 0.20 | rad，仅导航移动参考使用 |

命令范围和滤波时间常数读取部署 `deploy.json`：vx [-0.8,1.2]，vy ±0.5，wz ±0.8，`command_filter_tau_s=0.4`。`command_slew_per_s[2]=0.4` 只约束附加航向纠偏，不再代替命令滤波。Kp=1.0、纠偏上限 0.15 沿用当前参数，实机效果仍需验证。

手动归零、网页松手和导航停车从最后机身命令按 `v *= max(0, 1-dt/0.4)` 衰减，范数低于 0.05 时置零并立即交给 Balance，无零命令等待或旧目标混合。网页松手/指令超时先检测 250 ms，UDP 重复转发不能续期。减速中不反向、不增加幅值、不产生新纠偏，排队运动不能打断。B、L2、持有者断线、EXTERNAL 超时、共享 IMU 非法/过期和控制周期异常仍立即停止。切换不以关节速度或骨盆角速度为门槛。仅航向分支异常仍按原逻辑退出附加补偿。

命令来源接口在内部事件上使用 `command_source`。网页/实体遥控明确为 MANUAL；导航执行器在持有者、会话、反馈新鲜度和心跳约束下调用 `Controller.emit('move', client_id, [vx,vy,wz], command_source='NAVIGATION', command_time=original_monotonic_time)`。时间戳来自同机 UDP v2 报文中的原始命令接收时间；每条原始命令仅转交一次。未知字符串视为 UNKNOWN，不擅自锁航向。Nav2/DWB 保留上层路径跟随，不增加第二个路径 PID。

## 4. 离线验证与记录

在仓库根目录运行（不触碰硬件）：

```bash
PYTHON=/home/unitree/miniconda3/envs/CWC/bin/python
$PYTHON -m unittest discover -s Viser/test -p 'test_*.py' -v
$PYTHON -m unittest discover -s G1Deploy_BalanceHomie/tests -p 'test_*.py' -v
HYBRID_DEPLOY_PYTHON=$PYTHON bash G1Deploy_BalanceHomie/build.sh
$PYTHON Viser/test/test_web.py
/usr/bin/python3 Viser/tools/replay_heading.py --synthetic --heading-mode active
/usr/bin/python3 Viser/tools/replay_heading.py --session Viser/logs/sessions/某次会话 --heading-mode shadow
```

`test_web.py` 只启动 `--mode sim` → C++ `--simulate`，不创建 DDS。合成信号与离线回放只能验证逻辑、符号、时间处理和输入输出，不能验证新命令作用后的真实运动或闭环稳定性。旧日志缺少原始意图时，回放明确计数 `inferred_raw_command_samples`；不能据此精确重建旧操作阈值。

回放输出到独立 系统临时目录 `/tmp/viser_replay_时间/`（可用 `--output` 指定项目外目录）：`heading.jsonl`、`summary.json`、`heading.png`，图含连续展开的航向、误差、三种角速度、建议/实际 wz、状态和时间/限制事件。`--no-plot` 只需 NumPy；本机系统 Python 已有 matplotlib。文件脚本不导入 Bridge、不创建通信。

每次 observe/control 仍自动独立存档，只保留最近 10 次有本体反馈的会话；sim 需指定项目外 `--record-sim --record-dir /tmp/viser-sim`。telemetry.heading 保存处理量；raw state 保留原四元数/gyro。null 表示未提供或不适用，不能当成零测量。`wz_final_applied` 是航向环输出、送去选模的物理指令；最终选中运控的命令看同帧 `velocity`（Balance 会置零），`wz_suggested` 是建议。`correction_rad_s` 是主动模式的附加修正请求，受变化率约束时不等于瞬时实际输出差值。推车 `command_source=PUSHCART`、`pushcart_tracking=false`、`correction_rad_s=0`；任务环数据见 `high_policy.heading_hold` 的 `ref/err/corr/dt_s/imu_yaw_rad/cart_yaw_rad`，修正前后任务见 `raw_task_command` / `task_command`。`inspect_session.py 会话目录` 可读旧/新日志。

## 5. 真机步骤（仅供操作者执行，本次未执行）

先只读采集，确认传感器数据及正负号：

```bash
bash Viser/run.sh --mode observe --interface eth0 --port 8086 --heading-mode shadow
```

observe 不接收运动指令，不能从这里测得运动命令的 shadow 对比。需要在原无补偿策略下比较运动时，由操作者使用：

```bash
bash Viser/run.sh --mode control --interface eth0 --port 8086 --heading-mode shadow
```

按原网页接管/准备/启动流程，在低速、急停和防跌保护就绪后确认时间、左右转符号、XY 和 wz 缩放。准备验证主动纠偏时先归零，在网页将模式改为 active；或下次启动显式使用：

```bash
bash Viser/run.sh --mode control --interface eth0 --port 8086 --heading-mode active
# 完整关闭附加控制（单独 Kp=0 不能代替关闭未来 I/D）
bash Viser/run.sh --mode control --interface eth0 --port 8086 --heading-mode disabled
```

初始 Ki/Kd 和速率环保持关闭；依次验证前进/后退/左右侧移/斜走、原地左右转、边走边转、转向回零、停车再起步。每次只变一个因素，对比相同运动段的航向偏差、慢偏航速率、最终命令变化与步态；辅以地面方向线/视频，不能只看同一 IMU 的误差。需要阻尼时再依据角速度噪声和响应考虑小 Kd，不采用误差差分，也不让 D 压制主动转弯。振荡、步态不稳、饱和或延迟明显时停止试验，按原安全流程处理，不继续加增益寻找极限。
