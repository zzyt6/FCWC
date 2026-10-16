# 来源与许可

本包组合本机两个既有部署包：`G1Deploy_DecoupledWBC` 的 Balance ONNX，以及 `G1Deploy_Homie60000` 当前的 60000 轮 ONNX。模型字节不变，来源摘要、原模型契约和原导出校验向量保存在 `models/`；模型总清单为 `models/manifest.json`。子目录中的原 manifest 为追溯材料，其中可能列出不用于本包运行的其他原模型。

Balance 来自 NVlabs/GR00T-WholeBodyControl，源码版本 b042411fae38ee4d1af9aac82a37a1f8d14d6dd0。上游源码采用 Apache 2.0，模型使用 NVIDIA Open Model License；完整许可在 `licenses/` 和 `reference_upstream/`。Homie 模型来自用户训练的 model_60000.pt 导出，其原始来源摘要随模型保留。

运行框架沿用原部署包的 SDK2/C++/Python/Viser 实现。HOMIE 原 LICENSE 为 CC BY-NC 4.0，原 README 另有 CC BY-NC-SA 4.0 表述；完整 HOMIE、SDK2 和第三方许可文件随包保留。不将单个模型许可视为对整个派生运行框架重新授权。

本包新增混合模型选择、独立 PD 文件、组合配置摘要、模型编号通信协议、底层目标/增益平滑切换和遥测。两个原策略的观测代码保存在 reference_adapters/，对应观测类保持原计算方式。原 WBC Walk 模型不参与本包运行；不包括 ROS2、Pico、双臂 IK 或手指控制。
