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

标准流水线在导出前还会对 **settle 后的最终世界几何**复核一次
（`final_geometry_report.json`）：家具不得穿过真实楼板/顶板、站进楼板洞口，
吊装物不得脱离支撑面。仅在明确需要时用 `PANOWORLD_FINAL_GEOMETRY=0` 跳过。
成功产物的多样性（请求/生成/成功三套分布、户型回退率与偏差）用
`python tools/diversity_report.py success --root <batch> [--scenes]` 分析。

本系列新增：`--shape open_plan` 支持凹多边形房间与 L 形开放客餐厨（物理房间
与语义功能区分离）；最终几何同时复核柜门开启、椅子抽出、床侧通行、淋浴进入
等使用空间（`checks.py --functional-usage` 可独立运行）；浏览器 Walk 模式按
GLB 支撑面上下楼梯并处理楼板洞口与碰撞；资产库带内容指纹，同路径内容替换会
触发重跑，仅改 mtime 不会导致全量重读。跨层布局新增 `--shape void`（挑空客厅
+ 回廊栏杆）、`mezzanine`（局部夹层，未覆盖区保留顶板）与 `split_level`（不同
标高房间由坡道连接），洞口、栏杆与通行轨迹共用支撑/净空校验；settle 的同名
代理与多地毯已改为按实例匹配，glTF 资产在归一化时与注册表轴向对齐。

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
                settle.py   刚体沉降（drop-to-rest），沉降结果烘焙回位姿
                panorama.py 等距柱状 Cycles + Position-pass 深度
                     │
                     ▼
        export_scene.py  viewpoints/<v>/{panoImage_2048.png,
        place_depth.png, place_depth_scale.txt, extrinsics.txt} + map.json
                     │
                     ▼
        tools/validate_blender_agent_p0.py  逐像素投射数值校验
                     │
                     ▼
        blender_agent/review.py  空房、功能缺失、悬浮家具、相机贴墙、
        过曝全景 → review_report.json（附可落地的修复建议）
        tools/diversity_report.py  结构多样性指标与批量基准
```

`scene_spec.json` 是唯一契约：规划层只输出 JSON，构建层是唯一接触 Blender 的代码，校验层闭环反馈。

## 布局生成（种子 → spec）

`blender_agent/layout_gen.py --seed N --out spec.json` 把种子变成一份能通过
`checks.py` 与渲染验证的家具化 spec：2–5 个房间串成一条链——默认一字排开，`--shape block`
（或 `PANOWORLD_LAYOUT_SHAPE=block`）把同一条链折成两排（L 型 / 块状户型）——每面共享墙
一扇门、门在墙上自由取位，窗只开在外墙段，家具按房型模板从资产索引里选取，**相机在房间的
可走网格上采样**——每房间至少一台，按"从这个点能看到什么"打分（可见可走面积占比、
可见家具数、离墙与离家具净空、离门洞距离、门洞是否在通视线上）。小户型客厅加第二台相机：
至少相距 1.2 m，且观察房间的方位角相差 ≥ 45°；相机离家具 ≥ 0.6 m。

门和相机都不再被钉在一条线上。旧校验用 `atan2(dy, dx)` 定位门所在的采样列，而在 `R0`
约定下它是 ERP 列的**镜像**（东北方向会采到东南），生成器只能让所有门与相机共用一条 Y 线，
校验才落在真正的视线上。现在投影由相机矩阵求逆得到（`tools/validate_blender_agent_p0.py`
的 `world_to_erp`，以及 `layout_gen.py` 里等价的纯 Python 版本），门洞验收也改为局部射线：
每扇门从墙两侧各打一组水平射线，检查是否真的通透、门旁是否真有墙、开口宽高是否与 spec 一致；
相机能看到几扇门只作为独立的覆盖指标上报，合理遮挡不会判场景失败。

```bash
bash scripts/layout_gen_run.sh 7 outputs/layout_gen/seed7   # 生成 → 校验 → 渲染 → 验证
bash scripts/batch_generate.sh 0 20 outputs/layout_gen/sweep20
```

### 批次分级、退出码与断点续跑

每个场景的 `run_summary.json` 与 `batch_summary.json` 都按四个层级记账，不再用一个
"通过"数把只跑布局和内存跳过渲染混在一起：

| 层级 | 含义 |
|---|---|
| `layout_ok` | spec 已生成且 `checks.py` 通过 |
| `render_ok` | 构建 + 渲染 + 导出产出了完整视点集 |
| `validation_ok` | `validation_report.json` 在请求分辨率下通过（低于标准 2048x1024 时只允许 `canonical_resolution` 失败） |
| `review_ok` | `PANOWORLD_REVIEW=1` 时，`blender_agent/review.py` 没有发现 *fail* 级问题（见下节） |
| `bundle_ok` | 本次请求的产物完整：只跑布局时为 `layout_ok`；`PANOWORLD_BATCH_RENDER=1` 时要求上面三层全过；请求了评审时再加 `review_ok` |

请求了渲染但没跑完的场景记为 `pending_retry`（内存不足）或 `failed`，**不计入
passed**；`failure_reasons` / `failed_checks` 给出失败原因分布。`layout_gen_run.sh`
按失败阶段退出（`0` 成功、`2` 布局、`3` 检查、`4` 渲染、`5` 验证、`6` 评审），并把
`run_id` 写进 `run_summary.json`，批次据此拒绝复用旧报告；内存失败会降参重跑一次
（分辨率减半、`min(samples, 16)`，可用 `PANOWORLD_RETRY_RES` /
`PANOWORLD_RETRY_SAMPLES` 覆盖，`run_summary` 同时保留 requested/effective 两档，
降质成功不会冒充原质量完成）。`batch_generate.sh` 只有在所有场景都 `bundle_ok`
时才退出 0；断点续跑按 `resume_manifest.json` 的**配置指纹**判定复用：请求参数
（分辨率/采样/楼层数/拓扑族）、规范化 spec 哈希、代码与资产哈希都匹配，且逐相机的
全景/深度/位姿文件都存在可解码、尺寸正确，才会跳过；任一不符就重跑并打印原因：

```bash
PANOWORLD_BATCH_RENDER=1 bash scripts/batch_generate.sh 0 100 outputs/stream
PANOWORLD_BATCH_RENDER=1 PANOWORLD_BATCH_RESUME=1 \
  bash scripts/batch_generate.sh 0 100 outputs/stream     # 按 manifest 指纹复用
PANOWORLD_BATCH_SUMMARY_ONLY=1 \
  bash scripts/batch_generate.sh 0 100 outputs/stream     # 只重建汇总表
PANOWORLD_BATCH_RENDER=1 PANOWORLD_REVIEW=1 \
  bash scripts/batch_generate.sh 0 20 outputs/stream      # 附带视觉/规则评审
```

### 视觉评审与修复闭环

数值校验回答的是*几何*问题；一个场景可以全部通过却依然不适合入库：空厨房、沙发和
自己重叠、沉降后悬浮的家具、贴进墙里的相机、被窗户冲成一整片白斑的全景。
`blender_agent/review.py` 读取场景包（`spec.json`、`build/build_manifest.json`、
`build/settle_report.json`、导出的全景图与深度图），在无模型、无 GPU、无 Blender
的前提下把这些算出来（单场景几百毫秒）：

| 检查 | 读什么 | 抓什么 |
|---|---|---|
| `rooms_furnished` | spec | 可居住房间被留空（灯 + 盆栽不算家具） |
| `functional_coverage` | spec | 厨房无橱柜/灶台/水槽、卫生间无马桶、卧室无床、客厅无座位 |
| `no_duplicate_furniture` | spec | 同一资产叠在同一位置；一屋子四件同款 |
| `camera_clearance` | spec | 相机在墙里、贴在家具上、没有站位 |
| `furniture_grounding` | manifest + settle | 沉降后悬浮 / 沉地 / 互穿 |
| `build_matches_spec` | manifest | 构建产物用的不是当前 spec（陈旧构建） |
| `exposure` | 全景图 + 深度 | 过曝像素比例与最大亮斑、极暗视点 |
| `camera_framing` | spec | *提示级*：站位背对本房间家具 |
| `camera_coverage` | spec | 除走廊外的房间没有相机 |

问题分 **fail**（场景不可用）与 **warn**（质量余量）两级，每条都带机器可读的修复
建议；`review_report.json` 记录逐项细节、支撑数值与建议。外部评审通过 `REVIEW_CMD`
接入：它会被 `shlex.split` 拆分并追加全景图路径，需要向 stdout 打印一个 JSON 对象
（`{"pass": ..., "issues": [...]}`）；崩溃或输出非 JSON 只记为
`external.status: "error"`，不会单独判死场景。

```bash
python blender_agent/review.py outputs/stream/scene_0007            # 退出 0 通过 / 1 失败
python blender_agent/review.py outputs/stream/scene_0007 --strict   # warn 也算失败
python blender_agent/review.py outputs/stream/scene_0007 --apply-repair   # 微调 spec.json
python blender_agent/review.py outputs/stream/scene_0007 --fix-exposure   # 重做色彩管理
REVIEW_CMD="mymodel --json" python blender_agent/review.py <outdir>  # 可插拔外部评审
```

`--apply-repair` 的 spec 微调只对*下一次*渲染有效，因此 `--fix-exposure` 直接重做
磁盘上已有渲染的色彩管理：对保存的线性 EXR 走 OpenImageDenoise + 显示变换，按线性
数据算出的曝光档位（只统计室内过曝、按 ERP 权重、整组视点共用一个值）重写全景图，
不建模、不重打光线，分钟级而非重渲。旧图留在 `build/pano_pre_exposure/`，
`review_exposure.json` 记录改动。每个相机都按它那张全景图**实际交付时**的显示设置
重写：优先 `build/build_manifest.json` 中降噪流程记录的
`render.post_denoise.cameras[]`，其次是没有被降噪流程接管的相机的
`render.view_transform`/`render.exposure`，只有完全没有元数据的旧产物才按其真实输出
回退到 `Standard`/0——不会把 AgX 全景悄悄改写成 Standard；没有 Blender 时宁可拒绝
（`unsupported_transform`）。`--exposure N` 可直接指定档位（`--exposure 0` 即按原曝光
重做一遍，像素应保持不变）。

闭环已接进脚本入口：`PANOWORLD_REVIEW=1` 时，渲染前先做一次仅 spec 的评审（很便宜，
能在花渲染时间之前拦住空房），渲染后做完整评审。评审失败且
`PANOWORLD_REVIEW_REPAIR=1` 时，`review.py --apply-repair` 把建议落地为 spec 微调
——用资产索引或 `builtin:` 功能家具补齐空房、把相机移到房间内最优站位——**每个
候选都会先用 `checks.py` 校验再写入 spec**，随后重新检查、重新渲染、重新评审，只做
一次。原始 spec 另存为 `spec.pre_review.json`，动作记录在 `review_repair.json`。

### 多样性：这是新户型还是同一条长廊？

`tools/diversity_report.py` 度量一个批次真正覆盖了什么：房间数与类型直方图、
**带属性的房间图**（每个房间实例一个节点、带功能与楼层，每条门/通道/楼梯/坡道连接
一条边、带类型，环路数按 `E-V+C` 计算，规范指纹忽略房间名、列表顺序、家具与尺寸）、
占地并集与分层楼面面积、功能覆盖、各类型房间的家具组合、相机布点（3D/层内/导航三
种间距）与光照。拓扑重复按规范指纹分组，因此"同一条长廊换一套家具"只算一种拓扑，
而同样四个房间的链式、星形、环形是三种。

```bash
python tools/diversity_report.py specs outputs/stream/scene_*/spec.json
python tools/diversity_report.py seeds --seeds 0-99 --shape auto --out /tmp/div.json
```

它还带批量基准：先跑满 100 个种子的布局扫描并出多样性统计，再**分层抽样**（每个拓扑
族至少 1 景，其余优先选能带来新结构哈希的种子）按 1024x512 渲染并评审，汇总成一份
报告：

```bash
python tools/diversity_report.py benchmark --seeds 0-99 --sample 10 \
  --root outputs/benchmark100 --shape auto --resolution 1024x512 --samples 24 --repair
# -> outputs/benchmark100/{layout,sample}/、diversity.json、report.json
```

`report.json` 含分级成功率（layout / render / validation / review / bundle）、各阶段
与单场景耗时、多样性摘要与重复率、抽样名单与抽样原因，以及失败与评审问题分布。
渲染前若发现另有 `blender` 进程在跑会拒绝启动（`--force` 可覆盖），`--resume`
会复用已完成场景。

## 目录结构

```
blender_agent/    spec、构建、渲染、导出、布局生成、约束校验、视觉评审
scripts/          一键入口（含 fetch_assets.sh 资产下载）
tools/            资产索引/合并、数值校验、多样性指标与批量基准
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

## 生成式资产（`gen:`）

Poly Haven + MIDI-3D 之外，类目缺口（尤其 sofa / bed / chair）用生成式资产补齐：
`tools/gen_assets_trellis.py` 把生成的 glb 写进 `third_party/assets/generated/`，用与
`ph:` / `midi3d:` **完全一样**的 index + library 机制登记（`blender_agent/assets.py`），
场景 spec 只写 `gen:<id>`：

```jsonc
"furniture": [{"asset": "gen:sofa_01", "room": "living", "pos": [3.7, 3.6], "rot_z_deg": 0}]
```

```bash
# 一键：补库 → 生成 demo spec → 标准 6 步流水线（低分辨率快跑）
bash scripts/gen_assets_trellis.sh outputs/gen_assets
bash scripts/gen_assets_trellis.sh outputs/gen_assets proc        # 强制程序化后端

# 分步
python tools/gen_assets_trellis.py generate --backend proc        # 程序化兜底（trimesh）
python tools/gen_assets_trellis.py list                           # 看缓存里有什么
python tools/gen_assets_trellis.py emit-spec --out outputs/gen_assets/scene_spec.json
```

两个 backend（`--backend auto` 默认：找到可用 GPU 容器就走 trellis，否则 proc）：

| backend | 产出 | 依赖 |
|---|---|---|
| `proc` | trimesh 盒体拼装（sofa/armchair/chair/stool/bed/table/cabinet/lamp） | 主机 python + trimesh，无 GPU、无网络 |
| `trellis` | 真 TRELLIS text-to-3D（`microsoft/TRELLIS-text-xlarge`） | GPU 容器 + TRELLIS checkout + 权重 |

宿主没有 GPU，`trellis` backend 用 `docker exec` 在 GPU 容器里跑，容器需把本仓库与
TRELLIS checkout 挂在**相同的绝对路径**下：

```bash
docker run -d --name roomgen-trellis --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 \
  --user $(id -u):$(id -g) -v /srv/TRELLIS:/srv/TRELLIS:ro \
  -v /srv/conda-envs/trellis:/srv/conda-envs/trellis -v "$PWD:$PWD" -w /srv/TRELLIS \
  <cuda-image> sleep infinity

python tools/gen_assets_trellis.py fetch-weights --container roomgen-trellis   # 下载权重（可在容器内）
python tools/gen_assets_trellis.py generate --backend trellis \
  --container roomgen-trellis --classes sofa bed chair --variants 2
```

路径可用 `PANOWORLD_TRELLIS_REPO` / `PANOWORLD_TRELLIS_PYTHON` /
`PANOWORLD_TRELLIS_CONTAINER` / `PANOWORLD_TRELLIS_HF` 覆盖（权重默认缓存在
`third_party/assets/trellis_hf/`，官方源 `HF_ENDPOINT=https://hf-mirror.com` 可指定）。
每个生成资产都会做与 Poly Haven 资产相同的归一化（等比缩放到目标尺寸、水平主轴对齐、
包围盒底面中心归零、正面朝 -Y），因此 spec 里不需要任何额外参数；`emit-spec` 同时写出
每个 `gen:` 资产的出处（backend / prompt / seed / 实测尺寸）到 `<spec>.assets.json`，
`validation_report.json` 旁边还有 `checks_report.json`（`blender_agent/checks.py` 的摆放评分）。

已知限制：生成资产的朝向来自启发式（TRELLIS 文生 3D 本身没有统一正面），可在
`CATEGORIES[...]["yaw_deg"]` 或 `--yaw` 里微调；demo 户型里床与卧室相机只隔 0.25 m
（随附的 P1 spec 是 0.30 m，同一个几何约束）。

## 在浏览器里查看场景

两个查看器回答两个不同的问题：

| 查看器 | 看什么 | 启动方式 |
|---|---|---|
| 全景漫游（PanoWorld，8003） | *渲染结果*：Cycles 光照/天空/降噪后的 ERP 全景，点位之间有热点跳转 | `panoworld_viewer_server.py --viewpoints <outdir>/scene/viewpoints` |
| 3D 场景查看器（本仓库，8004） | *几何本体*：墙体、家具、PBR 材质，1:1 米制尺度，任意角度 | `bash scripts/scene3d_view.sh <outdir>` |

3D 查看器是 `tools/scene3d_viewer/` 下的静态页面（three.js r180 已本地 vendor，
无需打包、不依赖 CDN），加载 `run_build.py --export-glb` 写出的 `scene.glb`：

```bash
# 完整流水线顺带出 GLB；或只建模的快速路径（随附 demo 场景即此路径）
PANOWORLD_EXPORT_GLB=1 bash scripts/blender_agent_run.sh configs/blender_agent_p1_furnished.json outputs/p1
third_party/blender-4.5.14-linux-x64/blender --background --factory-startup \
  --python blender_agent/run_build.py -- --spec configs/blender_agent_p1_furnished.json \
  --out outputs/blender_agent/p1_settled/build --skip-render --export-glb

# 起服务，浏览器打开 http://<host>:8004/
bash scripts/scene3d_view.sh outputs/blender_agent/p1_settled   # Poly Haven 家具之家
bash scripts/scene3d_view.sh outputs/gen_assets                 # TRELLIS 生成家具

# 不依赖 Blender 复读校验 GLB（网格/材质/包围盒/相机是否在房间内）
/opt/conda/envs/diffaero/bin/python tools/check_scene_glb.py outputs/blender_agent/p1_settled
```

页面能力：Orbit（拖拽旋转 / 滚轮缩放 / 右键平移）、Fly 与 Walk（指针锁定后
WASD + 鼠标视角，Shift 加速；Walk 固定 1.5 m 眼高，Fly 用 Space/C 升降）、
视角下拉框可直接跳到每个全景相机位姿与房间、加载进度条，以及
半球光 + 平行光 + IBL 补光叠加 GLB 自带材质。

GLB 内容：全景相机能看到的一切，减去仅用于渲染的天空球，并去掉相机与灯光
（页面自带光照）。P1 demo 之家为 53 个网格 / 61 个节点、743,627 三角面、
27 个材质、61 张贴图，52.5 MB，glTF 惯例 +Y 向上、单位米。
`build/scene3d_manifest.json` 里已把相机位姿与房间范围换算到 glTF 坐标系，
页面无需猜测；`tools/check_scene_glb.py` 会用几何反查这些数值（包围盒毫米级一致、
每个相机都落在房间内部）。

## 资产与许可

* Poly Haven 模型/贴图 —— **CC0**，通过其 API 按需下载。
* MIDI-3D 家具 glb（3D-FRONT/3D-FUTURE 处理版）—— **CC-BY-NC-4.0**：仅限研究/评估，**不可商用**。商用请替换为官方 3D-FRONT + 3D-FUTURE（向数据集作者邮件申请）；`blender_agent/front3d.py` 已支持官方场景 JSON 格式回放。
* `gen:` 资产在本机生成，不附带第三方资产许可：`proc` 后端是本仓库代码（Apache-2.0），`trellis` 后端使用 MIT 许可的 TRELLIS 代码与 `microsoft/TRELLIS-text-xlarge` 权重（同为 MIT）。
* 本仓库代码：Apache-2.0。

## 测试

```bash
python tests/test_front3d_math.py                       # 变换顺序单元测试
bash  tests/test_front3d_replay.sh                      # 回放端到端
python tests/test_layout_gen.py                         # 布局生成单元测试
python tests/test_settle.py                             # 家具物理沉降：单元测试 + 盒子自检
python tests/test_batch_stats.py                        # 批次分级、退出码、重试与断点续跑
python tests/test_review.py                             # 评审检查、修复闭环与评审钩子
bash  scripts/blender_agent_run.sh configs/blender_agent_p0_demo.json outputs/p0
```

随附 demo 的数值验收：深度对网格中位误差 0.25 mm（16-bit 量化下限）、100% 有效像素、ERP 朝向偏差 < 0.1 像素。
