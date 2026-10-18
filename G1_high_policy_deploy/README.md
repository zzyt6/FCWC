# G1 推车高层：由原 Viser 统一使用

Viser 开始时可在“模型组合”选择原 Homie `model_60000.pt`＋V8.1（默认），
或 Homie `model_68000.pt`＋V8.2。两套均提前加载和预热，准备请求发出后锁定。
每套成对加载对应的学生、导出配置和微调 Homie：`v81_student.onnx`／`v81_config.json`／`homie_v81.onnx`，
或 `v82_student.onnx`／`v82_config.json`／`homie_v82.onnx`。V8.1 配 estimator 6000，V8.2 配 estimator 7000。
共享适配器入口仍为 `pushcart_v81`，本机相机与任务限幅配置仍为 `viser_v81.json`；模型文件更新后须完整重启。

唯一机器人网页入口：

```bash
cd /home/unitree/CWC
bash Viser/run.sh
```

默认仍是 `--mode control --interface eth0 --port 8086`。可显式覆盖模式、网卡和端口。
高层目录不再提供 `run.sh`、相机启动包装或另一套网页 / tmux 启动链。
Viser 启动时加载 Balance/Homie、推车学生和配套微调 HIM 分支并完成首次推理预热。推车执行开关默认关闭，
准备阶段即可积累有效历史并进行影子推理，不下发高层动作，不改变普通行走速度范围。
高层模型/配置不可用时，网页显示原因，普通混合策略仍可运行。

## 实际操作顺序

1. 接管网页 → 接管底层 → 选择准备姿态 → 移到准备姿态。
2. 准备完成后启动混合策略，机器人先以零速度站立；此时可照常使用普通行走。
3. 要推车时选择姿态 1，把小车移到手边，明确使能 Dex3 并抓握。
4. 在现有抓握面板调节角度；继续使用 `G1_Upper_Initpose/deploy_upper_hold.yaml` 和网页当前设定。
5. 准备阶段会后台提前连接小车相机；连接失败可点击“连接 / 重试小车相机”。
   它只启动现有 carmer 检测及只读姿态反馈，不启动其他运控。
6. 速度归零，将小车前进方向与骨盆正前方对齐并保持静止，点击“标定航向零位（车身已对正）”。
   等待标定完成、抓握动作完成、航向有效且推车历史重新达到 32/32 帧，再点击“启用推车策略”。
7. 不想推车时，先速度归零，再点击“返回普通行走 / 继续调抓握”。混合策略持续运行，手指不自动松开。

不点启用就始终是普通混合策略；没有相机或未抓握也可普通行走。
推车期间锁定抓握编辑、加载/保存和松手按钮，避免两处同时调整手部；返回普通行走后恢复。
本次集成不修改同事正在调的抓握文件，不从训练仓库覆盖手指角度。

普通模式速度滑条为机器人机身坐标；推车模式明确改标“小车”任务速度，采用训练定义。
切换时清零旧速度，防止同一数值被带到另一坐标含义下继续执行。
Viser 推车限幅与V8.1/V8.2 模型导出的训练范围对齐：vx `[-0.6, 0.9]` m/s、vy `±0.25` m/s、wz `±0.6` rad/s。
普通机器人指令范围为 vx `[-0.8, 1.2]` m/s、vy `±0.5` m/s、wz `±0.8` rad/s，也与模型导出配置一致。
推车滑条与高层执行端共用 `robot/config/viser_v81.json` 的 `task_limits`；修改后完整重启 Viser 生效。
Walk/Homie 与 Balance 的选择、减速、PD 和 DDS 下发均由原 Viser 管理。
相机或抓握中断后，推车减速并锁住旧指令；归零后可重新启用，也可在没有相机时返回普通行走。

推车学生的 30 维动作包含 3 维机器人速度残差、15 维腰 yaw/双臂修正和 12 维腿部残差。
小车任务参考与速度残差经过高层自身滤波后生成机器人侧速度指令，两者通常不相等。
这条指令在 Viser 中标为 `PUSHCART`，直接交给 Walk/Balance，下层不再叠加骨盆航向修正。
小车航向保持放在任务参考滤波和学生之前：限幅后的原始任务 wz 按实际 monotonic 控制间隔积分为参考，
测量取骨盆 IMU yaw 加相机已标零的相对 yaw（短暂丢帧使用保持值）。起步捕获当前小车航向，零指令/暂停复位。
使用 Kp=1.0、修正上限 ±0.15 rad/s、误差限幅 ±0.3 rad 并拖动参考；修正加到任务 wz 后限于 ±0.6 rad/s。
vx/vy 保持原值，任务参考仍用原一阶滤波，当前不加积分或额外低通。
普通行走的骨盆航向参数独立控制原行走环。正常 Homie 执行时 `velocity` 和 `filtered_velocity` 等于学生的 `homie_command`，
实际选中的命令写入下一拍低层与高层历史。归零减速采用原 `command_filter_tau_s=0.4` 滤波，期间不追加航向修正。

持续发送严格零小车指令、点击速度归零或松手超时，都会截断高层内部残留的速度输出，
让 Viser 从上一拍机器人指令按 `v += min(1, dt/0.4) * (0-v)` 减速。
停止期间不应用推车腿部残差，腰臂保持最后下发目标，不再追加新的高层修正。
机器人侧三维指令原单位范数 `<0.05` 时立即选 Balance，当前 Balance 速度观测严格为零；
这不代表机器人实测速度已为零，历史帧仍保留真实执行记录。
小车侧当前没有额外的非零死区；高层的 `gate_speed=0.05` 只是残差渐入尺度，
不能等同于机器人侧的 Balance 切换阈值，也不能据此把小车指令 `<0.05` 一律判零。

## 预热与历史帧

模型文件和 ONNX 首次运行工作在 Viser 启动时完成，不放在“启动混合策略”或“启用推车”按钮回调里。
准备过程中，低层持续读取真实反馈并维护 6 帧历史；进入准备完成和策略运行时不清空。
推车模型在相机 ID 0/1 航向有效时维护 5 帧短历史及 32 帧长历史，按 50 Hz 约需 0.64 秒。
没有有效航向时仍可普通行走，推车历史不以虚构的零航向填满。

未启用推车时的输出是影子推理，日志 `inference_role=shadow`；执行端仍仅接受已选中的低层动作。
影子预测不会回写成下一帧的已执行高层动作或腿部残差。启用时只重置任务指令/滤波状态，
保留已积累的真实历史；只在新会话、重新准备、反馈中断/相机失效等情况下丢弃失效历史。
网页显示低层 6 帧及推车 32 帧准备进度。推车启用后的 0.5 秒上肢增益渐变仍按原流程执行，
这是有意的增益切换，不是模型现场加载。影子推理失败只标记推车未就绪，普通混合策略继续运行。

## 小车与手指显示

- 使用训练仓库原始 `xiaotuiche.obj` 和 `cart_v7.urdf`，见 [资产来源](assets/README.md)。
- 启用推车自动显示小车；摆车时可先勾选“预览参考小车”。
- 机器人机身仍跟随实时反馈；抓握时的手指参考模型使用当前网页抓握设定及选中的补偿。
  调节或加载新角度会同步显示，实测手指角度继续在左侧关节表中显示。
- 小车航向来自 carmer。标签安装位置尚未填写，车体位置默认按双手抓握点作参考，明确标为“未标定”。
- 在 `Viser/config/cart_display.json` 的 `tag_positions_in_cart_m` 填入各标签中心相对车体 base_link
  的 `[前, 左, 上]` 米制坐标后，位置改用对应标签测量。
- 相机过期时冻结最后观测并标明过期，不把旧数据当新测量，不用指令积分虚构世界位置。

## 配置和数据约定

- `robot/config/viser_v81.json`：本机高层配置、相机端口/超时、航向偏置、任务限幅。
- `robot/models/v81_student.onnx`、`v82_student.onnx`：所选组合的学生，630/4032 维输入，30 维动作。
- `robot/models/homie_v81.onnx`、`homie_v82.onnx`：分别配套 Homie 60000、68000 actor、原 HIM 和推车微调速度估计器。
- `robot/models/v81_config.json`、`v82_config.json`：各自导出的观测、滤波、动作缩放、关节约定和模型来源。
- `robot/pushcart_v81/viser_policy.py`：与 Viser 同拍执行的适配器。
- `carmer/heading_udp.py`：只发往本机的 `carmer_cart_yaw_v1` JSON。

yaw 来自 `raw_fused_cart_heading.yaw_rad`，必须有有效骨盆补偿和 `pelvis_level` 参考系，
单位弧度、左转为正。保留图像原始单调时钟，并扣除配置相机延迟。
它等价于 `wrap(小车世界 yaw − 骨盆世界 yaw)`，位于每帧观测下标 94。
原始航向已完成实时腰角、骨盆重力方向补偿和双标签圆周融合；策略不使用显示用的
`fused_cart_heading` 低通结果，避免在训练已有的相机延迟/采样保持模型之外增加滤波滞后。
小车标签已确认是 ID 0 和 1；检测端只融合这两个 ID，接收端也拒绝包含其他 ID 的航向。
只出现背景 ID 2 时不会产生有效小车航向。任意一个小车标签可见即可使用，不要求两个同时可见。
无标签帧不刷新最后有效测量；过期、乱序、重复、非有限数值不能续期。
相机由 Viser 网页启动时自动使用这里的接收端口。
`yaw_offset_rad` 从测量值中减去。Viser 的“标定航向零位（车身已对正）”采集点击后的 30 个新有效帧，
用原始航向的圆周平均替换本次运行的偏置（不累加、不写配置文件、不修改骨盆 IMU 或图像时间戳）。
样本偏离平均值超过 3°、3 秒内有效新帧不足、航向过期、运动或控制状态改变时取消并保留原偏置。
采集只允许在普通混合策略站立、速度归零时由网页持有者启动；对正需由操作者确认，软件只检查航向稳定性。
成功后清空推车历史并按新零位重建 32 帧，完成前不能启用推车。相机延迟配置保持原值。

相机已经是独立 Python 进程，运行同一个 `carmer/apriltag_detector.py`；网页启动时关闭相机绘图，
用本机 UDP 发送航向，骨盆反馈仍由只读 DDS 订阅获取。要保留独立相机窗口，可在 `carmer/` 中先运行
`python apriltag_detector.py --heading-udp-port 9870`。Viser 收到有效航向后会复用它，不再自动启动第二个相机进程。
窗口 `Detect FPS` 是检测完成频率；接收报文频率、每个策略周期实际采用的新图像频率和策略执行频率应分别统计。
策略每拍使用最新有效航向：同一拍收到多帧时取最新值，没有新帧时短暂保持上一值（受 0.3 秒超时限制），不排队回放旧图像。

126 维帧：HOMIE 76 + 上一任务参考 3 + 上肢 PD 估计力矩 15 + 小车 yaw 1
+ 上次高层动作 18 + 实际腿部残差 12 + 尺度 1；短/长历史为 5/32 帧。
先推理学生，再由 Viser 选模；Balance 不加腿部残差。
两套底层下一拍都使用选中动作（Homie 加实际残差，再按训练裁剪），不使用未执行输出。
原始网络输出和高层状态写入现有 Viser 会话日志，不复制模型或完整输入历史。

`Viser/logs/sessions/` 保留最近 10 段有本体反馈的会话。新版 `telemetry.jsonl.gz`（schema 20）逐帧记录：

| 字段 | 用途 |
|---|---|
| `model_set` / `high_policy.model_release` | 当前整套模型选择（v81/v82）及高层版本；对应元数据 model_sets 中的文件与哈希 |
| `high_policy.raw_task_command` / `task_command` / `task_reference` | 原始限幅任务 / 航向修正后的任务 / 本拍更新后的滤波参考，顺序 vx/vy/wz |
| `high_policy.heading_hold` | `ref`、`err` 为 rad，`corr` 为 rad/s；`dt_s` 为实际控制间隔，`imu_yaw_rad`、`cart_yaw_rad` 为该拍测量；零任务复位 |
| `high_policy.homie_command` | 高层生成的机器人候选速度，尚未经过 Viser 的零速拦截和滤波 |
| `input_velocity` → `filtered_velocity` → `velocity` | 拦截后输入 → 归零滤波结果 → 当前选中运控的命令；正常推车 Homie 执行时三者等于 `homie_command`，Balance 时最后一项为零 |
| `heading` | 推车 `command_source=PUSHCART`，保留 IMU 信息和最终 wz；`correction_rad_s=0`、`pushcart_tracking=false`，下层不做推车航向修正 |
| `high_policy.gate` / `raw_actions` / `estimator` | 任务门控、30 维原始动作、学生 19 维估计输出 |
| `high_policy.leg_residual` / `applied_leg_residual` | 高层候选 / 选中叠加到 Homie 的 12 维腿部残差，单位为网络动作值，后续还有动作裁剪和关节/PD 限制 |
| `high_policy.upper_delta_rad` / `upper_target_selected` | 相对训练保持姿态的 15 维腰臂候选偏移 / 是否合入 Python 请求目标，不代表电机已执行 |
| `high_policy.inference_inputs` | 实际采用的反馈 tick/时间、相机时间/航向、上一拍运控命令和任务参考、上肢 PD 力矩估计及其来源反馈时间 |
| `high_policy.camera` / `zero_calibration` | 原始和归零后的航向、偏置、有效性、样本年龄、标零状态 |

学生推理开始/结束时间保存在 `high_policy.inference_started_monotonic` / `inference_finished_monotonic`，
与同帧 `state.tick`、低层 `policy_inference_monotonic` 对齐。`inference_role=shadow` 表示未启用的影子推理；
没有新推理时上述推理数据为 `null`。`metadata.high_policy.telemetry` 记录关节顺序、保持姿态及字段定义。
反馈 q/dq、IMU、两套底层动作、HIM 输出、Python 请求目标、C++ 实际下发目标/增益和时序仍保存在同一行。
零速后可直接对照高层候选尾速与实际送入运控的命令；命令和估计值都不能当作机器人实测线速度。
重启 Viser 后的新会话才包含新增字段，旧会话不补写。

## 与训练发布管线的差异

参考 [CJME_Push_car](https://github.com/zzyt6/CJME_Push_car) 提交
`1b9574d682a2acfe34eebd21654e792c781aa3f7` 的
[MuJoCo V8 runner](https://github.com/zzyt6/CJME_Push_car/blob/1b9574d682a2acfe34eebd21654e792c781aa3f7/HomieRL/homie_pushcart_v8/mujoco_runner.py)。

按本机要求运行现有 Viser 控制链路。加载时核对每套组合中普通行走与配套推车 Homie 的 actor 和原始 encoder 全部 14 个权重张量，
要求逐值完全一致；V8.1 对应 `model_60000.pt`，V8.2 对应 `model_68000.pt`。跨版本配对会被拒绝。
推车启用时，Viser 用所选 `homie_v81.onnx` 或 `homie_v82.onnx` 的微调 `hv` 三维速度估计和原 HIM 的 32 维归一化隐变量
共同输入同一 Homie actor，与参考 runner 的 `use_finetuned=1` 一致。配套估计器来自
V8.1 的 `2026-10-12_20-52-19_v8_1_roa_teacher_8k/homie_velocity_estimator_6000.pt`，
或 V8.2 的 `2026-10-16_22-28-14_v8_2_roa_teacher_20k/homie_velocity_estimator_7000.pt`，不是高层学生自己的 19 维显式估计头。
两者作用不同：学生估计头服务高层动作，微调 HIM 速度服务下层 Homie 腿部 actor。
训练环境以仿真真值比较两种速度估计误差并选择；参考部署在高层启用时使用配套微调估计，Viser 采用此部署规则。
普通行走恢复原 Homie 估计器；Balance 不变。推车减速期间仍使用微调 Homie，归零后按原规则切入 Balance。
相机或抓握中断造成的推车暂停也保留微调 Homie 分支，直到明确返回普通行走，避免在减速中途切换估计器。
所有分支都在启动时加载并预热，每拍只执行当前 Homie 分支，仍使用 Viser 的同一份 6 帧执行历史。
日志 `homie_estimator_source` 标明 `homie_original` / `pushcart_finetuned`；
`homie_estimator.velocity_raw` 是实际输入 Homie actor 的估计，`velocity_original_raw` 保留同帧原估计作对比，
`latent_raw` / `latent_normalized` 始终来自原 HIM。估计值不是速度实测；推车 `velocity_raw / 2` 为骨盆机身系估计 m/s。
上肢 PD 已按用户确认的推车参数接入 `G1Deploy_BalanceHomie/config/pushcart_upper_pd.json`。
启动准备及普通混合策略使用原两份 Balance/Homie 配置，启用推车时只切换腰 yaw 和双臂 15 个关节：

| 关节 | 普通模式 Kp / Kd | 推车模式 Kp / Kd |
|---|---|---|
| 肩 pitch / roll | 200 / 4 | 100 / 2 |
| 肩 yaw | 200 / 4 | 50 / 2 |
| 肘 | 100 / 1 | 50 / 2 |
| 腕 | 20 / 0.5 | 20 / 1 |
| 腰 yaw | 300 / 5 | 300 / 3 |

切入和退出均由 200 Hz C++ 桥接执行 0.5 秒上肢增益渐变；期间暂停新的运动指令。
这个渐变独立于腿部 Balance/Homie 的即时切换。推车模式同时允许高层腰臂目标下发；
关闭后保持最后执行的上肢姿态，恢复原增益，不重置抓握。固定腰 roll/pitch 及 Dex3 不使用此配置。
日志包含请求/执行的上肢模式、渐变进度及实际下发 Kp/Kd；高层力矩观测继续读取实际反馈增益。
本机通信协议现为 v9（状态 1120 字节、命令 248 字节），新增上肢目标执行标志，需完整重启旧 Viser。
按用户要求，推车腰臂在 C++ 使用 20 ms 线性插值，将约 50 Hz 终点分到 200 Hz 下发；
提前换目标时从最后实际下发值接续，目标迟到时保持终点。归零/暂停清除未完成插值并保持当前目标，
实体 B 冻结腰臂，L2/显式停止/故障优先。腿部和 Dex3 执行逻辑不增加此插值。
`pushcart_upper_pd.json` 的 `target_interpolation_s=0.02` 启用，0 恢复直接目标，重启生效。
这是部署侧新增平滑，会改变训练的目标保持时序；日志请求终点与实际插值输出必须分开解释。

因此原仓库的仿真成绩不能直接代表本组合。高层滤波和动作映射保持导出定义，
停止过程和静止时 Balance 接管采用 Viser 语义。
上肢力矩用 `applied_kp*(applied_q-q)-applied_kd*dq` 并按训练限幅。
优先使用上一条 200 Hz 桥接反馈（允许 2–8 ms 间隔），缺失时退回当前反馈；
日志的 `torque_sample_age_ms` 标明实际间隔，0 表示退回当前帧。它是 PD 估计而非力矩实测。

若相机报 `EBUSY`（设备忙），先释放其他相机程序的占用，再在网页点击连接 / 重试。
相机故障不会阻止普通混合策略；推车中丢失相机时会减速并等待人工处理。

## 离线检查

```bash
/home/unitree/miniconda3/envs/CWC/bin/python -B -m unittest discover -s /home/unitree/CWC/G1_high_policy_deploy/test -v
/home/unitree/miniconda3/envs/CWC/bin/python -B /home/unitree/CWC/G1_high_policy_deploy/test/test_web_pushcart.py
```

使用真实 ONNX、合成机器人反馈和独立 UDP 端口，检查两份 HOMIE 基座导出的一致性、航向新鲜度、
抓握/控制权限、动作历史、Balance 接管、掉线锁止和急停，不代表真机闭环稳定性。
`Viser/test/test_pushcart_pd.py` 使用真实 C++ 控制核心及管道合成反馈，验证上肢目标、增益切换及抓握保持。
本机检查不依赖安装时的 `deploy.zip`。`pc/export_v81.py` 保留为 V8.1 导出参考；
本次回切直接使用原导出文件。`pc/` 的仿真脚本需要训练仓库和其环境，
原始说明见 [训练仓库部署文档](https://github.com/zzyt6/CJME_Push_car/blob/main/deploy/README.md)。
