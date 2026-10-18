# G1 Balance + Homie 混合部署包

零速／小速度命令时使用上传的 `models/balance.zip` 内 Balance（`model_68600.pt` 导出的 ONNX）。行走默认使用 `models/homie/` 的 Homie 60000 轮；Viser 准备前也可选 `models/homie_68000/` 的 68000 轮，分别与高层 V8.1、V8.2 配套。两套完整模型、契约、来源和校验向量独立保留，清单分别为 `models/manifest.json` 和 `models/manifest_v82.json`；ONNX 保留原始字节。Balance 的原始导出配置在 `models/balance/config.json`，两套组合共用同一 Balance、机身配置和 PD。

## 启动与网页操作

网页统一使用相邻 **`../Viser`** 目录，原内置 `visualizer.py` 已删除。完整的页面操作、单腰映射与自动存档见 [外层 Viser 说明](../Viser/README.md)。

```bash
cd /home/unitree/CWC
bash Viser/run.sh --mode sim --port 8086                    # 合成反馈联调，不是物理仿真
bash Viser/run.sh --mode observe --interface eth0 --port 8086 # 真机只读
bash Viser/run.sh --mode control --interface eth0 --port 8086 # 真机控制，页面显式接管
```

旧 `bash run.sh shadow/control eth0 --viser --viser-port 8085` 和 `bash run.sh simulate-web --viser --viser-port 8085` 也会转到这个外层网页。网页启动先只读，点击“① 接管底层”后才释放原厂控制器；依次准备姿态、启动混合策略。外层维持单腰 27 轴，禁用 SDK 13/14，不压缩后续索引，observe/control 自动生成独立会话存档。

本包仍可独立运行无网页部署，沿用原 29 轴控制配置：

```bash
conda activate CWC
cd /home/unitree/CWC/G1Deploy_BalanceHomie
bash run.sh shadow eth0
bash run.sh control eth0 --log reports/control_hybrid.jsonl
bash run.sh simulate
```

无网页 `control` 启动时即释放原厂控制器；R2 准备、R2 启动策略、L1+摇杆运动、B 归零、L2 停止。请勿同时运行本包和外层 Viser，它们独占相同的 UDP 27001/27002。停止使用 L2 或 Ctrl+C；Kp=0、Kd=2 退出阻尼不维持站立，需要支撑。

安装无网页依赖使用 `bash setup.sh`，网页依赖使用 `bash ../Viser/setup.sh`。迁移网页时须一起复制 `Viser`、本包、`G1_URDF` 和 `G1_Upper_Initpose`。

## 切换规则及控制范围

部署命令范围为 vx **[-0.8, 1.2] m/s**、vy **[-0.5, 0.5] m/s**、wz **[-0.8, 0.8] rad/s**。网页和遥控器使用 `config/deploy.json` 的 `command_filter_tau_s=0.4` 一阶滤波：`v += min(1, dt/tau)*(target-v)`。这是命令而非实测速度。公共配置修改后必须重建桥接以匹配配置 token，Viser 启动会自动构建。

`config/deploy.json` 中 `model_switch_command_norm=0.05`：

- `norm([vx,vy,wz]) >= 0.05` 使用 Homie；低于阈值的同一拍使用 Balance 并将当前速度命令置零。
- 选择依据是限幅、滤波和可选航向修正后的原单位速度**命令**，在模型输入缩放前计算，包含转向分量。
- 无额外零命令等待或身体静止判定。50 Hz 下，0.6 m/s 纯前进命令经过 49 拍（0.98 秒）减速后交接；网页松手另计 250 ms 检测。故障与过期反馈继续按独立保护停止。
- `--policy-mode auto` 是默认值；可用 `--policy-mode balance` 或 `--policy-mode homie` 做单模型对比。此选项只控制模型选择，准备阶段始终使用 Balance PD。重启时恢复 auto 即可。

| 项目 | Balance | Homie |
| --- | --- | --- |
| ONNX | `models/balance/policy.onnx` | `models/homie/policy.onnx` |
| 输入/动作维度 | 456 / 12 | 456 / 12 |
| 观测历史 | 6×76，继承选中策略执行历史 | 同一执行历史 |
| 关节观测 | SDK 0–12、15–28，跳过腰 roll/pitch | SDK 0–12、15–28，跳过腰 roll/pitch |
| 速度／高度输入 | 当前帧零命令、0.74；旧帧保留历史命令 | 当前滤波命令、0.74 |
| previous action | 上一拍被选中的 12 维输出，按训练裁剪 | 同左，不从 applied_q 反算 |
| 策略控制范围 | 双腿；腰保持准备目标 | 双腿；腰保持同一准备目标 |
| 双臂 | 保持第一次准备时捕获的目标 | 保持同一捕获目标，使用 Homie 的臂部 PD |
| 动作缩放 | 0.25 | 0.25 |

外层 Viser 使用可选 `--prepare-full-body`，准备目标来自 Viser 读取的上半身 YAML，双臂随后保持该目标；`--dex3-grasp` 提供双手 3 秒平滑切换能力；接管和机身准备不使能手部，Viser 首次点击手部按钮时才使能并抓握。上述表格中的捕获姿态行为仍适用于本包默认入口。桥接本机 UDP 协议已升级至 v8（状态 1120 字节、命令 244 字节），包含 Dex3 姿态和最近成功 DDS Write 的关节角、按机身／左手／右手分组的写入时间、手部反馈接收时间及 Kp>0 位掩码；发送成功不代表电机确认接收。Python 与 C++ 必须一起更新并重新构建。

Viser 从 `Viser/config/dex3_pd.json` 加载双手正常控制增益（默认 kp=1.5、kd=0.2），通过 `--dex3-pd <kp,kd>` 传入桥接，全部 14 个手部关节共用；修改配置后需重启网页程序。直接运行本包桥接且省略该参数时仍使用旧默认 kp=0.5、kd=0.1。退出阻尼固定 kp=0、kd=0.1。`DEX3 PD` 日志在每侧手成功写入第一条指令及切换至阻尼时输出实际增益；本次参数配置不改变 UDP 协议。

准备阶段采用 Balance PD。无网页默认入口对 12 个腿部关节做至少 3 秒准备，腰和双臂保持接管时姿态；Viser 则同时按上半身 YAML 准备腰 yaw 和双臂，并检查全部可动关节到位。进入 POLICY 后才执行模型动作。

两套候选输入从同一实测反馈构造，保留此前选中策略的命令和动作历史。每拍只替换各候选的最新命令：Balance 为零，Homie 为滤波后速度。发送 RUN 后把选中历史及按训练裁剪的动作提交给下一拍；后台输出仍每拍记录，但不写入执行历史。WAITING/HOLDING 只推理，RAMPING 暂停推理；准备阶段按原规则清空历史，模型切换不清空。ONNX 和原始导出 `config.json` 保留不动，`deployment_contract.json` 按训练源码补全交接语义。依据：[CJME_Push_car 1b9574d 的历史恢复](https://github.com/zzyt6/CJME_Push_car/blob/1b9574d682a2acfe34eebd21654e792c781aa3f7/HomieRL/homie_isaaclab_v2/env.py#L268)及同提交 `task_core.py`、`balance_bank.py`、`view_homie_v2_balance.py`。

C++ 对两种策略都只接受前 12 个腿部目标。腰和双臂在 Balance↔Homie 切换时继续保持准备目标，不接收策略包中的上半身目标；Viser 的 SDK 13/14 仍禁用。

## 两套独立 Kp/Kd

- **Balance：`config/balance_pd.json`**，按新导出 `models/balance/config.json` 的 27 个关节名映射到 SDK 29 槽位。
- **Homie：`config/homie_pd.json`**，复制自 Homie60000 已恢复的原始配置。
- 两个文件都含 SDK 29 关节名称、`kp_sdk29`、`kd_sdk29`，包括腿、腰和双臂。修改一个文件不改变另一个。

以下左右对称的部位按单侧顺序列出：

| 部位/顺序 | Balance Kp | Balance Kd | Homie Kp | Homie Kd |
| --- | --- | --- | --- | --- |
| 髋 pitch/roll/yaw | 100 / 100 / 100 | 2 / 2 / 2 | 100 / 100 / 100 | 2 / 2 / 2 |
| 膝 | 150 | 4 | 150 | 4 |
| 踝 pitch/roll | 40 / 40 | 2 / 2 | 40 / 40 | 2 / 2 |
| 腰 yaw/roll/pitch | 300 / 250 / 250 | 5 / 5 / 5 | 300 / 300 / 300 | 5 / 5 / 5 |
| 肩 pitch/roll/yaw | 200 / 200 / 200 | 4 / 4 / 4 | 200 / 200 / 200 | 4 / 4 / 4 |
| 肘 | 100 | 1 | 100 | 1 |
| 腕 roll/pitch/yaw | 20 / 20 / 20 | 0.5 / 0.5 / 0.5 | 20 / 20 / 20 | 0.5 / 0.5 / 0.5 |

两套 PD 在 Viser 实际启用的 27 个关节上相同，文件仍独立。Balance 导出未包含的 SDK 13/14 保留原增益，Viser 不启用这两个槽位。共享 effort 继续采用原有较小限制：髋 roll 为 88 Nm，低于新导出的 139 Nm。

切换请求通过本机协议携带 profile 编号（Balance=0，Homie=1）。C++ 在 200 Hz 循环中选择配置。`profile_transition_s=0` 关闭旧目标/增益混合，`policy_target_speed_rad_s=0` 关闭训练中没有的正常策略目标变化率限制，新动作直接进入 PD。准备过程的渐变、关节范围、名义 PD 力矩限制及健康检查继续生效。

两套 PD 文件仍独立，且始终只有一个底层发布器。公共关节范围、effort、准备时序和健康阈值保留在 `config/deploy.json`，effort 保留原有较小限制；这些保护与训练物理仍存在差异，离线检查不等于实机停稳验证。

所有关节实测角度超出配置参考范围（含 0.02 rad 余量）仅为位置警告，不再影响健康标志、接管、发起准备或继续策略。`BODY POSITION warning/recovered` 记录机身逐关节越界及恢复。有限值的机身请求目标超界时限幅并记录 `BODY TARGET`，不触发全局故障；非有限值仍拒绝。目标位置范围与 PD 力矩范围无交集时，仅按比例减小对应关节 Kp/Kd，使名义 PD 力矩不超过 effort，记录 `BODY EFFORT`，其余关节继续输出；恢复后使用正常增益。URDF/配置范围不是实机硬件故障边界。机身 SDK 故障、通信/指令超时、无效反馈、倾角保护及显式停止仍独立处理。

准备过程超过 `calibration_timeout_s=10` 秒尚未实测到位时，继续保持准备目标和正常 Kp/Kd，仅输出 `CALIBRATION warning holding_position_control`。这条超时不进入 Fault、不启动阻尼退出，也不修改健康标志。状态包 `flags` 的 `0x08` 表示准备超时警告；Viser 显示未到位关节并记录警告开始及解除。实测位置误差不超过 `0.1 rad`、速度不超过 `0.5 rad/s`，且下发目标完成过渡后，进入 Holding 并解除警告；策略仍需手动启动。

**修改任何配置后停止并重启程序**，`run.sh` 自动重新构建。公共配置和两个 PD 文件一起计算配置摘要，Python 与 C++ 不一致时拒绝运行；不支持运行中热更新。构建脚本也可显式运行 `bash build.sh`。

## 日志与复核

`--log reports/control_hybrid.jsonl` 保存约 50 Hz 的：29 关节 q/dq、IMU、速度命令、被选模型、模型原始动作、完整的两套观测历史、两套影子动作、Python 请求目标、C++ 反馈目标、实际下发 Kp/Kd、profile 和过渡进度，以及单调时间戳。无网页入口文件名固定时再次运行会覆盖；留存对比时换新名字。外层 Viser 自动会话及显式导出不覆盖旧数据。

- `selected_model`：本次 Python 选中的模型。
- `pd_profile`：已收到的反馈中 C++ 接受的模型编号。反馈采样早于本次请求，切换附近可能落后一个控制周期。
- `pd_transition_blend`：0 表示刚开始交接，1 表示已完成；等待阶段为 1，不代表正在发电机命令。
- `applied_kp_sdk29` / `applied_kd_sdk29`：反馈包内对应的输出增益，准备阶段反映增益渐变。
- `model_observations.balance` / `.homie`：各自完整 456 维历史；RAMPING 为 null。
- 同名 `.metadata.json` 保存本次公共配置、两套 PD、模型契约及模型 SHA256。

只有 POLICY 阶段执行选中模型动作，WAITING/HOLDING 中的模型动作属于影子推理。日志不是全部 200 Hz LowCmd 抓包，也没有机身真实位置/线速度，不能靠模型输出幅值判断漂移。

```bash
python tools/verify_model.py
python tools/test_offline.py
python tools/test_viser_offline.py
python tools/check_bundle.py
# 用当前模型/配置重放日志中的两套输入，并核对已完成的 PD 切换
python tools/verify_telemetry.py reports/control_hybrid.jsonl
```

模型验证使用新 Balance 的 64 组输入／输出和 Homie 的 128 组固定向量；观测与导出配置及 Homie 原适配器对比，并测试 C++ 状态机和合成反馈通信，不证明机器人上闭环稳定。`tools/verify_model.py` 默认只打印结果，可用 `--output /tmp/model_verification.json` 保存。历史 `reports/` 不代表当前模型的验证结果。`tools/package.py` 可生成带 SHA256 的无网页部署压缩包（不含本机 build/运行日志、相邻 Viser/G1_URDF）；新电脑解包后需要重新编译。

## Viser Dex3 recovery and protocol v6

Viser may request one Dex3 finger to return inside its software limits while the body is still waiting for preparation. The request uses the existing official HandCmd position mode, with a 0.15 rad maximum initial excursion, Kp ramping from 0.5 to 2 over 1 s (Kd=0.1), a 3 s timeout, measured arrival checks, and feedback/command watchdogs. It does not reset motor errors or change the joint limits. Dex3 position excursions are warnings and do not block preparation or stop body control. Recovery failure cancels only that hand attempt and emits a DEX3 RECOVERY warning; it does not fault the body core. Hand SDK/feedback faults and hand-only DDS channel initialization or write failures are local warnings: pause both hand trajectories, attempt Kp=0/Kd=0.1 hand damping, and keep publishing body commands. Body preparation is independent of hand availability and cancels any ongoing finger recovery. It never enables or zeroes Dex3. In the normal startup flow, no hand motor commands are published until the first explicit pose-button click after body preparation; that click enables both hands and grasps from fresh measured angles, respecting the compensation checkbox. Subsequent clicks alternate open/grasp. Explicit single-finger recovery remains a separate manual action. Reconnection alone cannot resume hand motion: the client acknowledges suspension with hand_pose=0, then explicitly requests a pose using fresh feedback as the interpolation start. Viser exposes this as a 3-second re-enable-and-open action respecting the compensation checkbox. Body health/communication checks and explicit stops remain required. `cpp/hand_core.hpp` contains the recovery bounds; `tests/test_hand_recovery.cpp` and `tests/test_hand_dds.cpp` cover recovery and the generated motor messages.

Local protocol v6 appends a selected recovery joint to commands and recovery availability/status to telemetry. Rebuild the bridge and use the accompanying Python protocol together. The shipped runtime manifest omits the absent, unused training archive `models/homie/params.zip`; ONNX and deployment contract hashes remain checked.

Dex3 DDS samples with a motor count other than seven are discarded without refreshing the last complete sample's timestamp. A single empty sample therefore does not release grasp PD; repeated malformed samples still expire the existing 200 ms watchdog. Explicit hand/motor errors and overheating invalidate feedback immediately, including errors carried by incomplete samples. The bridge logs discarded samples per hand/topic, at most once per second. Feedback recovery after an actual suspension never rearms the hands automatically.

## 推车上肢模式

普通 Balance/Homie 的模型编号仍为 0/1。独立的 `upper_mode` 只在策略运行时选择推车腰臂目标及增益，
参数来自 `config/pushcart_upper_pd.json`，其文件也纳入 Python/C++ 配置校验 token。
推车上肢增益使用 0.5 秒渐变，腿部切换参数不变。退出后保持最后上肢姿态并恢复对应普通增益。
只有 SDK 12、15–28 被此模式覆盖；13/14 和 Dex3 不受影响。
