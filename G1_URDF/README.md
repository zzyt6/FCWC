# G1 URDF 独立模型包

本目录包含 `g1.urdf` 和它引用的全部 49 个 STL mesh 文件，可整体复制或打包到其他工程。

```text
G1_URDF/
├── g1.urdf
├── meshes/      # URDF 引用的 STL 文件
└── README.md
```

mesh 路径采用相对于 `g1.urdf` 所在目录的形式，例如 `meshes/pelvis.STL`。迁移时请保持目录结构，并使用支持以 URDF 所在目录解析相对资源路径的加载器。

来源：`HomieRL/legged_gym/resources/robots/g1_description/g1.urdf`。URDF 内容保持原样，原工程文件保留。

本模型包含 27 个旋转关节和 31 个固定关节；内部机器人名称虽然包含 `29dof`，实际可动关节数为 27。
