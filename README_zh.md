# RoomGen.Agent

[English](README.md) | 简体中文

基于 Blender 的自主室内场景生成 agent——一键产出**多空间连通、观感真实、数值精确**的室内场景。

一个种子（或一份 3D-FRONT 户型 JSON）经过校验、渲染，输出完整的场景包：全景图、欧氏深度、相机位姿与房间邻接关系。全链路**没有任何"2D 生成 → 3D 重建"环节**。

```bash
# 单套场景：生成 → 校验 → 构建 → 渲染 → 导出 → 验证
bash scripts/layout_gen_run.sh 7 outputs/demo_seed7

# 连续批量生成：N 套户型，每套独立目录
PANOWORLD_BATCH_RENDER=1 bash scripts/batch_generate.sh 0 100 outputs/stream

# 高质量出图
PANOWORLD_RES=2048x1024 PANOWORLD_SAMPLES=128 \
  bash scripts/blender_agent_run.sh <spec.json> <outdir>
```

渲染成品与校验报告见 [`demo/`](demo/README.md)。

## 为什么先在 3D 里把场景造对

传统管线用扩散模型生成全景图再重建 3D，所有重建误差（深度漂移、位姿漂移、未观测区域）都需要额外机制补偿。本方案直接在 Blender 中构建场景，其余一切都从场景**精确渲染**：

| 产物 | 来源 | 精度 |
| --- | --- | --- |
| 等距柱状全景图 | Cycles 路径追踪 | 无生成伪影 |
| 米制深度 | Position pass → 射线长度 | 毫米量化，对网格中位误差 0.25 mm |
| 相机位姿 | 相机矩阵 | 精确 |
| 实例/语义分割 | Cryptomatte | 像素级精确 |
| 多空间连通 | Boolean 门洞 | 构造保证 |

## 架构

```
种子 / 需求 / 3D-FRONT JSON
        │
        ▼
layout_gen.py ──► scene_spec.json ◄── front3d.py（数据集回放）
        │              │
        ▼              │
   checks.py           │   碰撞 / 净空 / 连通性 BFS
        │              ▼
        └──► run_build.py（headless Blender）
                shell.py    地板/墙体/天花 + Boolean 门窗洞
                stage.py    PBR 材质、日光+天空、窗口补光
                assets.py   Poly Haven / MIDI-3D 模型库
                details.py  踢脚线、门窗套、倒角、内置道具
                panorama.py 等距柱状 Cycles + Position-pass 深度
                     │
                     ▼
        export_scene.py  viewpoints/<v>/{panoImage_2048.png,
        place_depth.png, place_depth_scale.txt, extrinsics.txt} + map.json
                     │
                     ▼
        tools/validate_blender_agent_p0.py  逐像素投射数值校验
```

`scene_spec.json` 是唯一契约：规划层只输出 JSON，构建层是唯一接触 Blender 的代码，校验层闭环反馈。

## 目录结构

```
blender_agent/    spec、构建、渲染、导出、布局生成、约束校验
scripts/          一键入口（含 fetch_assets.sh 资产下载）
tools/            资产索引/合并、数值校验
configs/          demo 规格（手写与生成）
tests/            单元测试 + 端到端测试（含 3D-FRONT 回放 fixture）
demo/             渲染成品 + 校验报告
third_party/      本地工具链与资产数据（不入库）
  blender-*/      Blender 4.5 LTS 可执行文件
  assets/         模型/贴图缓存与索引，由 fetch_assets.sh 生成
```

## 安装

```bash
# 1. Blender 4.5 LTS（headless）装到 third_party/
mkdir -p third_party && cd third_party
curl -L -o blender.tar.xz \
  https://mirrors.tuna.tsinghua.edu.cn/blender/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz
tar xf blender.tar.xz && cd ..

# 2. 资产库（下载 + 建索引，不入库）
bash scripts/fetch_assets.sh                 # Poly Haven + MIDI-3D（约 13.3 GB）
bash scripts/fetch_assets.sh --skip-midi3d   # 快速路径（约 200 MB）

# 3. 主机 Python 需带 numpy/PIL（校验与建索引用）
```

`third_party/assets/` **有意不入库**：它是批量数据，由下载脚本从源头重建（含索引）。如路径特殊，用 `PANOWORLD_BLENDER` / `PANOWORLD_HOSTPY` 覆盖工具路径。

## 资产与许可

* Poly Haven 模型/贴图 —— **CC0**，通过其 API 按需下载。
* MIDI-3D 家具 glb（3D-FRONT/3D-FUTURE 处理版）—— **CC-BY-NC-4.0**：仅限研究/评估，**不可商用**。商用请替换为官方 3D-FRONT + 3D-FUTURE（向数据集作者邮件申请）；`blender_agent/front3d.py` 已支持官方场景 JSON 格式回放。
* 本仓库代码：Apache-2.0。

## 测试

```bash
python tests/test_front3d_math.py                       # 变换顺序单元测试
bash  tests/test_front3d_replay.sh                      # 回放端到端
python tests/test_layout_gen.py                         # 布局生成单元测试
bash  scripts/blender_agent_run.sh configs/blender_agent_p0_demo.json outputs/p0
```

随附 demo 的数值验收：深度对网格中位误差 0.25 mm（16-bit 量化下限）、100% 有效像素、ERP 朝向偏差 < 0.1 像素。
