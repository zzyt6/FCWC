# 小车显示资产

来源为训练仓库 `zzyt6/CJME_Push_car`，提交 `1b9574d682a2acfe34eebd21654e792c781aa3f7`：

- [xiaotuiche.obj](https://github.com/zzyt6/CJME_Push_car/blob/1b9574d682a2acfe34eebd21654e792c781aa3f7/HomieRL/legged_gym/resources/object/xiaotuiche.obj)，SHA256 `3e6ba99daa0852bb2775f501325ebf91772fac1c5530371698eea8be99b7ee00`。
- [cart_v7.urdf](https://github.com/zzyt6/CJME_Push_car/blob/1b9574d682a2acfe34eebd21654e792c781aa3f7/HomieRL/legged_gym/resources/object/cart_v7/cart_v7.urdf)，SHA256 `8c9c758adfe85516a996b3a0fdfd9a6471986e2711c853262110fe8ed00f6cb7`。

文件保持原始字节。OBJ 毫米单位由原 URDF 的 `0.001` 缩放转换成米，网格原点偏移也保留。
轮子未接入编码器，显示其参考姿态，不生成虚假的滚动测量。

抓握参考点来自同一提交的 `assets/g1_pushcart_v7/asset_report.json`，只用于参考车体位置。
`Viser/config/cart_display.json` 的 `tag_positions_in_cart_m` 填入真实标签中心相对小车
`base_link` 的米制坐标后，显示会使用标签测得的位置；空配置时明确标为抓握参考位置。

机器人继续使用本机 `G1_URDF/g1.urdf`。手指参考角来自本机抓握配置和网页实时调节，
不加载训练仓库中烘焙了固定手指角度的机器人模型。
