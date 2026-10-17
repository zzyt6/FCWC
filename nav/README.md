# G1 · FAST-LIO 建图 + Nav2 常规导航（自研运控）

本目录支持建图、普通导航和推车导航。推车导航复用已经在 Viser 测试的高层策略、Balance/Homie 下层策略和 Dex3 控制，不自动对接小车。

小车任务速度范围统一读取 [`config/pushcart_navigation.json`](config/pushcart_navigation.json)：
vx `[-0.2, 0.6]` m/s、vy `[-0.15, 0.15]` m/s、wz `[-0.3, 0.3]` rad/s，采用小车自身坐标系。
DWB 候选速度、UDP 转发和推车高层输入共同遵守这些范围，保留横向速度。
这些限幅独立于 Viser 手动推车范围，也不改变高层生成的机器人速度范围；网页的巡航速度另限制平移合速度。

- **建图**仍用宇树本体运控：用遥控器/官方应用让 G1 行走，FAST-LIO 只读雷达和 IMU。
- **导航**不再调用宇树 `LocoClient`。Nav2/DWB 做上层路径跟随，普通模式的 `/cmd_vel` 交给 Balance/Homie，推车模式先交给现有推车高层，再由其输出驱动下层策略，复用 `Viser/core` 控制器及骨盆 IMU 航向外环。导航使用独立参数和随转弯推进的参考航向；Viser 网页仍用于本体运控测试，手动保向行为不变。

```text
Nav2/DWB /cmd_vel ─ g1_cmd_vel_udp（限速 + 运动门租约 + 250 ms 超时）
   └─ UDP v3 127.0.0.1:39002（含模式代次和原始命令接收时间）─ locomotion/g1_locomotion_executor.py
         └─ Viser/core Controller：0.4 s 时间常数滤波、航向 PID、命令范数低于 0.05 即切 Balance
               └─ G1Deploy_BalanceHomie/build/g1_control（C++ 200 Hz）─ rt/lowcmd
   运控状态 ─ UDP 127.0.0.1:39005 ─ 网页桥（运控未进入策略运行时运动门不会打开）
```

## 推车流程与坐标

1. 按原流程接管、准备、启动 Balance/Homie 站立。选择能接近小车把手的准备姿态。
2. 人工摆入小车，在“推车策略”面板明确点击抓握；可调整、加载、保存双手抓握参数及切换补偿，沿用 Viser 的限制和配置文件。
3. 启动航向相机，摆正小车并按需归零。等待 ID 0/1 的新鲜航向和高层观测历史就绪，点击“启动推车策略”。
4. 页面变为“小车中心导航”后，等待过滤后的扫描与导航参考点就绪，再打开运动门、设置小车中心目标。地图同时显示机器人、小车估计中心和包含两者的碰撞轮廓。
5. 停止导航并归零后可退出推车策略；松手、移开小车后点击“恢复普通导航”。退出高层不会自行把仍抓着的小车解释为普通步行。模式切换关闭运动门并丢弃旧速度，需要重新开门。

`map -> odom -> base_nav` 的 FAST-LIO/AMCL 定位链保持机器人基准。导航使用 `nav_control`：普通模式与 `base_nav` 重合；推车模式位于小车中心，其朝向为机器人世界 yaw 加相机实测相对 yaw。小车位置沿用 Viser 的双手抓握几何估计，不要求先标定 ID 0/1 的安装位置，也不把位置估计标成实测。把手、手掌偏置与小车模型读取 `../Viser/config/cart_display.json`。

推车局部控制器 `CartPush` 使用 DWB 的三轴候选速度和完整 footprint 碰撞检查，同时增加随速度调整距离的前视目标评分；参考训练仓库巡线的前视思路，保留 costmap 避障。推车行为树只做等待、重新规划，不自动执行普通模式的后退/旋转脱困动作。正常规划仍允许经碰撞检查的倒退、侧移及转动。到点范围仍为 15 cm / 10°，推车在末段 0.80 m 开始减速。轮廓覆盖小车、双手与机器人，并增加位置跟踪和相对航向余量，窄通道可能被保守地判为不可通行。

推车模式的原始点云先按小车模型的板、杆、轮等几何剔除自体点，再进入 FAST-LIO，继而用于定位和避障。保留点的全部字段和原始传感器时间戳不变；两路扫描中被车体遮挡的远端射线设为无效，不能用空回波清除后方障碍。小车附近的外部点不按整块包围盒删除。雷达安装变换直接来自当前关节反馈，不依赖过滤后的里程计，航向恢复后可继续过滤。

`pushcart_navigation.json` 提供 footprint、跟踪、航向及自滤波余量；自滤波默认仅额外扩张 4 cm，不与导航的大轮廓共用。几何估计依赖正确抓握和模型尺寸，未包含小车额外装载物的形状；推车导航的实际跟踪误差和遮挡效果仍需现场验证。航向/反馈或过滤扫描过期时关闭运动门、平滑归零，机身保持原有策略支撑。

## 构建与启动

```bash
cd /home/unitree/CWC/nav
./build.sh            # colcon 构建 + 生成默认 cf 二维地图
./run_mapping.sh      # 建图（Viser 0.0.0.0:8080）
./run_navigation.sh   # 导航（Canvas http://<Wi-Fi IP>:8080，rosbridge 9090）
```

导航与 `Viser/run.sh` **不能同时运行**：两者都要占用部署包的底层桥接（UDP 27001/27002）和机器人低层控制。

入口脚本会在加载 ROS 环境后优先使用本机 `~/cyclonedds/install/lib` 中的 CycloneDDS 0.10.2（可用 `UNITREE_CYCLONEDDS_LIB` 指定目录）。Foxy 自带的 0.7 在当前机器人网络的 DDS 发现阶段会崩溃，表现为多个节点同时退出、8080/9090 没有监听。SDK 本体反馈桥与运控执行器独立加载各自 SDK 的 DDS 库，避免混用；ROS 网页节点使用系统 Python，不受已激活的 Viser 虚拟环境影响。

常用参数：

```bash
./run_navigation.sh locomotion_mode:=sim        # 运控使用 C++ 合成反馈；雷达/定位节点仍启动
./run_navigation.sh heading_mode:=active        # 启用导航航向外环（独立配置默认 active）
./run_navigation.sh heading_mode:=disabled      # 关闭附加航向修正，保留滤波和模型切换
./run_navigation.sh locomotion_policy:=homie    # 固定单模型（默认 auto）
./run_navigation.sh start_locomotion:=false     # 不启动运控（只看定位/规划）
```

## 网页操作顺序

1. **PCD 地图**：选择 `maps/` 中的 PCD，生成并装载二维地图。
2. **定位**：拖拽设置初始位姿，等待 AMCL 显示 READY。
3. **运控**：在“上半身准备姿态”选择“姿态 1 · 原准备姿态”或“姿态 2 · 双臂下垂”，等待页面确认；吊装或扶稳后依次点 ① 接管底层 → ② 准备姿态 → ③ 启动策略。也可在①之后、②之前选择。选择本身不动作，开始准备后锁定，后续策略保持所选上半身姿态。这和 Viser 使用同一套配置和控制流程；机身准备不使能 Dex3，也不依赖手部就绪。
4. **导航**：选择速度模式和 0.20–0.60 m/s 的速度，点击“应用并保存速度”，看到“已应用”后启用 G1 运动安全门，再在地图上拖拽 Nav2 Goal。也可在导航中调整速度。
5. 停止：“停止 / 取消导航”会关闭运动门，按与 Viser 相同的滤波减速；自动模式在命令范数低于 0.05 的同一拍切回 Balance，并将当前策略命令置零。“阻尼停止”立即进入阻尼，会卸掉支撑力。实体遥控器 L2 随时生效。阻尼后点“复位（只读）”，再重新执行 ①②③。

运动门打开需要同时满足：地图 + AMCL 收敛 + 传感器新鲜 + **运控处于策略运行** + 速度设置已确认 + 当前模式的导航参考点就绪。运控离开策略运行（L2、阻尼、故障、进程退出）0.6 s 后，导航会自动取消、运动门关闭。

网页“运控提示”沿用 Viser 的关节参考范围检查，显示机身、双手超限及准备超时等提示；恢复后的提示保留约 5 秒并标记“已恢复”。关节角度超出参考范围不影响机身健康状态、不关闭导航运动门，也不触发全身阻尼。有限值请求目标仍限幅；位置范围与名义 PD 力矩约束冲突时只降低相关关节增益。准备超过 10 秒未到位会继续保持位置控制，等待实测到位后再手动启动策略。SDK 机身故障、非有限数值、机身反馈/底层策略心跳超时、倾角保护和显式停止仍按原规则处理；Nav2 断指令或定位丢失则取消导航并平滑归零。

网页速度作用于平移命令的合速度 `hypot(vx, vy)`，不是实际机身速度的保证值。**固定巡航**在普通路段优先选择设定速度，保留较慢轨迹供转弯和避障；**自动调速**由 DWB 自行选择不超过设定上限的速度。两种模式都从距最终目标 0.60 m 起，把平移速度上限逐步降至 0.15 m 处的 0.10 m/s；固定巡航偏好同时逐步释放。减速约束在候选速度生成时生效，DWB 继续用当前反馈速度预测制动距离并做碰撞检查，底层仍使用 0.4 s 滤波。

到点范围为 **15 cm、朝向 10°**，目标检查器和 DWB 的位置范围一致。Nav2 确认完成后，网页桥关闭导航速度门并发送平滑归零请求；机身继续运行策略，过滤后命令范数低于 0.05 时置零并切回 Balance。旧速度不能继续微调或重新起步；等待归零结束、重新启用运动门后再下发下一目标。这里的 0.05 是 `[vx, vy, wz]` 原单位命令的数值范数，不是纯平移速度阈值。位置判定范围也不是机身最终停止误差的保证。

速度通过 DWB 候选轨迹生成器和巡航评分生效，不在碰撞检查后放大 `/cmd_vel`。模式与速度作为一个参数事务提交给控制器，确认成功后保存到 `config/navigation_speed.json`；重启后重新确认再允许导航。首次默认是自动调速、上限 0.35 m/s。接口为 `/api/navigation/speed?mode=auto|fixed&speed=0.20..0.60`，当前确认状态在 `/api/status` 的 `navigation_speed` 中。

导航直接使用当前 Viser 的 `Controller`、Balance/Homie 模型、观测历史和底层 PD，不维护另一套停车控制器。命令按 `v += min(1, dt / 0.4) * (target - v)` 滤波；停车时从最后实际下发的命令（含已有航向补偿）向零衰减，范数低于 0.05 即置零并切 Balance。没有固定 1 秒截止，也没有归零后再等 0.5 秒的阶段；0.4 秒是时间常数，交接时间取决于起始命令大小。

正常减速仍属于健康的策略运行，不触发上述故障取消；减速期间禁止开启运动门或提交新目标。导航零速和关门直接触发减速；上游断指令或 UDP 断流从最后一次原始 `/cmd_vel` 接收时间判断 250 ms 超时，不因缓存转发续期，这与 Viser 松手检测后减速的流程一致。减速中不反向、不增加命令幅值、不产生新纠偏，不接收排队运动。交接后新的有效非零命令可重新起步。命令归零不代表实际机身已停稳。显式阻尼、实体 B/L2 和反馈故障仍立即覆盖普通减速。

定位故障、取消导航和异常退出导航任务会先关闭速度运动门，再尝试取消 Nav2 action；取消接口报错不会跳过关门。迟到的开门回执、旧的续租、正在等待轮廓确认的开门请求都不能覆盖更新的停止请求。运动门关闭只撤销导航速度授权，运控执行器继续保持策略运行，按上述滤波减速并切回 Balance。网页桥进程失联由 1.5 秒授权租约触发关门，速度转发进程退出由 250 ms 原始命令看门狗触发减速。

关门状态与发布在同一临界区完成，并发但校验失败的开门请求不能取消关门。每次重新开门会清掉速度缓存，等待新的 `/cmd_vel`；执行器也会丢弃关门前已经在途的旧速度。AMCL 或导航里程计包含非有限值、无效四元数时，按定位故障关门并平滑停车，不向底层发送阻尼停止。

**退出整个导航启动终端（Ctrl+C）、结束运控执行器或底层控制进程，会结束机身支撑，不属于“关闭运动门”。** 正常停车用网页“停止 / 取消导航”，保留运控执行器运行；要退出整套程序，应先完成停车并吊装或扶稳。底层进程崩溃、机身反馈故障不能靠上层导航保证平滑交接。

## DWB 与骨盆航向控制

DWB 通过采样、预测和评分候选速度轨迹完成局部路径跟随，不是 PID。保留它作为上层，骨盆航向 PID 作为策略输入前的执行补偿，不再增加第二个路径 PID。

普通导航命令使用 `NAVIGATION`。进入运动时捕获当前骨盆 yaw，此后按经过滤波的导航角速度推进参考：`yaw_ref += wz_nav * dt`，误差取最短角。最终 `wz = wz_nav + bounded_PID(yaw_ref - yaw)`；D 项使用参考角速度与测量 yaw rate 的差，因此正常转弯不再被手动锁向抵消。参考偏差上限 0.20 rad，修正上限 0.15 rad/s；XY 始终使用机身坐标，不旋转、不加闭环。停车、来源切换和反馈异常会释放旧参考与积分。这样消除了两层争用固定航向的逻辑冲突，真实延迟、漂移和闭环稳定性仍需实机验证。

导航默认 `Kp=1.0, Ki=Kd=0`，目前实际只启用 P；I/D 保留可配置能力。Viser 使用自身配置。推车由同一 Controller 自动切换为 `PUSHCART` 来源，参考按高层给出的机器人 wz 推进，不能用小车任务 wz 积分。

## 换模型 / 调参数在哪里

| 要改的 | 位置 | 说明 |
| --- | --- | --- |
| Balance / Homie ONNX 模型 | `../G1Deploy_BalanceHomie/models/{balance,homie}/policy.onnx` | 同时更新 `models/manifest.json` 中的 SHA-256 和对应 `deployment_contract.json`，否则启动时报 `Artifact hash mismatch`。当前两套模型输入/输出维度均为 456/12。 |
| 关节 PD | `../G1Deploy_BalanceHomie/config/{balance,homie}_pd.json` | 修改后重启，C++ 桥接会自动重新编译。 |
| 速度上限 / 命令滤波 / 模型切换阈值 | `../G1Deploy_BalanceHomie/config/deploy.json` 的 `command_limits`、`command_filter_tau_s`、`model_switch_command_norm` | 与 Viser 共用；当前滤波时间常数 0.4 s、切换阈值 0.05，不设额外零速等待。 |
| 导航骨盆航向 PID | `locomotion/motion_config.json` | `heading_kp/ki/kd`、`run_mode`、`max_correction_rad_s`、`navigation_max_heading_error_rad`；独立于 Viser 手动航向参数，修改后重启。停车滤波使用上一行的共同部署配置。 |
| 导航巡航 / 自动调速 | 网页导航面板；保存到 `config/navigation_speed.json` | 0.20–0.60 m/s；控制器确认后生效。 |
| 导航硬限幅 | `src/unitree_nav2_web/launch/g1_navigation.launch.py` 中 `g1_cmd_vel_udp` 的 `max_vx/max_vy/max_vyaw` | 当前为 0.60 / 0.20 / 0.50，DWB 的侧向与倒退约束仍单独保留。 |
| 路径跟随（DWB）速度/加速度 | `src/unitree_nav2_web/config/nav2_foxy_g1.yaml` 的 `FollowPath` | 当前规划加速度约束为 vx 0.3、vy 0.2、wz 0.4；它们与底层一阶滤波是不同参数，需结合实机响应验证。 |
| 推车高层与配套 Homie | `../G1_high_policy_deploy/robot/config/viser_v81.json` 及 `robot/models/` | 与 Viser 共用加载、预热、抓握及航向检查；导航运行时只覆盖小车任务限幅。 |
| 推车速度、轮廓、自体过滤余量 | `config/pushcart_navigation.json` | 重启后同时影响候选速度、转发与高层输入；前视评分参数在 Nav2 YAML 的 `CartPush` 下。 |
| 执行进程 | `locomotion/g1_locomotion_executor.py` | 它直接 import `Viser/core`，复用同一套 Controller，不复制代码。 |

## 测试

```bash
/home/unitree/miniconda3/envs/CWC/bin/python -B -m unittest discover -s locomotion -p 'test_executor*.py'
python3 locomotion/test_executor_sim.py   # 离线端到端：网页桥链路 → 执行进程 → Controller → C++ 合成反馈
source /opt/ros/foxy/setup.bash && source install/setup.bash
export LD_LIBRARY_PATH="$HOME/cyclonedds/install/lib:${LD_LIBRARY_PATH:-}"
colcon test --packages-select nav2_pose_progress_checker nav2_rotation_shim_controller unitree_nav2_core unitree_nav2_web
colcon test-result --all

# 网页按钮、错误处理与建图回调（不会启动机器人）
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q -p no:cacheprovider src/unitree_nav2_web/test src/unitree_fastlio_baseline/test
# 实际 Firefox 页面点击；需外部安装 puppeteer-core（本机验证版本 23.0.0）
PUPPETEER_MODULE=/path/to/node_modules/puppeteer-core node src/unitree_nav2_web/test/test_browser_flow.cjs
```

离线测试只能验证通信、状态机和命令链路，**不能**代替真机闭环验证。第一次真机导航请在围挡区域吊装测试，只发 0.5–1.0 m 的短目标，并有人手持遥控器准备 L2。

速度 UDP 已升级为 v3（仍为 40 字节，原填充区存导航模式代次），必须重新构建 `unitree_g1_kinematics` 后与新版执行进程一起使用；旧版 v1/v2 报文会被拒绝；状态端口 39005，导航几何状态端口 39006。

## 与原项目的差异

- 运控改为当前 Viser 的共享实现，不再调用原 `LocoClient`、旧抓握播放器或旧推车控制器。
- 普通导航保留 `FollowPath`；推车使用独立的 `CartPush`、小车中心参考、动态组合轮廓和模型自体过滤，不强制 `vy=0` 或固定最小转弯半径。
- 网页包含运控和推车/抓握面板；导航停机继续复用原有滤波和模型交接。

安装依赖、建图、PCD 转换、AMCL 与故障排查见 [`docs/算法安装与使用手册.md`](docs/算法安装与使用手册.md)。FAST-LIO 参数、G1 腰部坐标链、时间戳边界等原理说明沿用原项目 `/home/unitree/uinitree_fast_lio_nav2/README.md`。
