<div align="center">

# 🩻 medscope

**医生端胸片阅片副驾** —— 异质双读一裁（CNN × VLM，失效模式正交）· 危急值旁路抢报 · 每句结论挂证据(硬门) · PHI 0 泄漏(硬门) · 诊断口径红线(硬门) · 五套件 eval 门禁 · 零构建工作台悬停即溯源

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#-许可证)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TorchXRayVision](https://img.shields.io/badge/TorchXRayVision-densenet121--res224--all-orange.svg)](https://github.com/mlmed/torchxrayvision)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1c3c3c.svg)](https://langchain-ai.github.io/langgraph/)
[![Qwen3-VL](https://img.shields.io/badge/reader__b-Qwen3--VL--32B-6b46c1.svg)](https://github.com/QwenLM/Qwen3-VL)
[![tests](https://img.shields.io/badge/tests-352%20passed-brightgreen.svg)](#-评测)
[![gate](https://img.shields.io/badge/make%20eval-G1%20FAIL-red.svg)](#-评测)
[![PHI leaks](https://img.shields.io/badge/PHI泄漏-0-brightgreen.svg)](#-评测)

「武道AI / AI Engineering Dojo」**以阵制胜**系列 · 阵 05 · 高风险域里，把「测不准」写在门上而不是藏进指标

</div>

---

> **教育演示项目，非医疗器械。** 不用于临床诊断，不用于任何真实诊疗决策。
> 数据全部来自公开去标识数据集（NLM Open-i / Indiana University 胸片集），不接触任何真实患者数据。
> 系统只产出**报告草稿**，签发权永远在执业医师。

`medscope` 取自放射科真实术语 **second reader（第二读片者）**——既是架构隐喻，也是定位声明：AI 是第二个读片的人，不是主诊医生。

输入一张胸片 + 既往病历文本 + 检查申请单，输出一份**每句结论都挂着证据**的结构化报告草稿，外加一条独立的危急值告警通道。

**这个仓库现在最值钱的部分不是那些绿灯，是 [🔍 诚实的局限](#-诚实的局限)。** `make eval` 当前是 **FAIL**，G1 危急值召回 0.815。这个红字是刻意留着的：把门调绿有的是办法，但每一种都是在同一条 ROC 曲线上滑动工作点，而不是让系统真的看得更准。详见 [G1 为什么是红的](#g1-为什么是红的不是阈值问题也不是调参能解决的)。

## 📑 目录

- [✨ 特性](#-特性)
- [🏗️ 架构](#️-架构)
- [🧱 技术栈](#-技术栈)
- [🚀 快速开始](#-快速开始)
- [💬 使用示例](#-使用示例)
- [📊 评测](#-评测)
- [🔍 诚实的局限](#-诚实的局限)
- [🔒 安全](#-安全)
- [🔭 可观测](#-可观测)
- [📁 项目结构](#-项目结构)
- [🧩 配置](#-配置)
- [🗺️ 路线图](#️-路线图)
- [📄 许可证](#-许可证)

## ✨ 特性

- 🔬 **异质双读一裁（招牌）** —— 直接搬自放射科真实的双读制 SOP。`reader_a` 是判别式 CNN（TorchXRayVision DenseNet-121），出**定量概率 + Grad-CAM 定位**但读不懂临床语境；`reader_b` 是生成式 VLM **看原图独立读片**，能结合病史推理、能描述标签集之外的征象但不给校准概率。两者**失效模式正交**，所以分歧不是要抹平的噪声，而是「这张片上有一处需要人看」的精确指针。这也是与同质 Jury（多个同构模型投票）的本质区别。
- 🚨 **危急值旁路抢报** —— `triage()` 与报告撰写**并行**跑，不等报告写完就告警，照搬医院的 critical results communication 制度。用 `critical_threshold=0.3` 而非报告阈值 0.5：**漏一例气胸可能致命，多报一例只花医生三十秒**，这个非对称是刻意的，且召回与误报率**永不合并成 F1**。
- 📎 **无证据不出句（硬门）** —— 报告里每一句话都必须引用一个带 `source` / `prob` / `locus` 的 `Finding`；零引用句在图里由 `evidence_check` 处理（重试一次再剥离），不是在写作层偷偷丢掉。工作台上悬停任一句话，图上对应区域与证据卡同时高亮。
- ⚖️ **仲裁器自己也看图** —— 只看两个读者的结论 + 指南散文，等于拿先验重新掂量断言，那不是仲裁而是复读。`arbiter` 走 `chat_with_image`，prompt 明确要求「根据图上所见判断」；且**只在分歧项上花 LLM 预算**（`max_llm_judgments` 封顶）。
- 🔒 **PHI 0 泄漏（硬门）** —— 先写**合成 PHI 注入器**再写脱敏器，顺序反了 G4 就是恒真断言（Open-i 本身已去标识，拿它验脱敏是重言式）。注入器沿命名维度做组合生成，2000 种子扫描零泄漏。
- 📐 **诊断口径红线（硬门）** —— 输出层拦截确诊式表述、强制免责声明，并在 14 例夹具上同时量**拦截率**与**误伤率**——只量拦截率的话，一个把所有句子都拦下的护栏能拿满分。
- 🧪 **每道硬门都有一个已被演示过的失败方式** —— 五道门各配一个「故意打断实现 → 门必须变红」的非重言性测试。写这些破坏测试的过程本身就抓出过一个指标测错对象的 bug（见[评测](#-评测)）。
- 🔌 **离线优先** —— 仓内含 3 例样本切片，**克隆即可零 key 零网络跑通全流程**：`OfflineVLMClient` 替身 + 本地哈希 embedder + 内存 store。真实后端（SiliconFlow VLM / Langfuse / SQLite）config-gated 懒加载。
- 🖥️ **零构建工作台** —— 单文件 HTML，五块可解释面板（双读对照 / 分歧与仲裁 / 证据溯源 / 危急值时间线 / 门禁状态）+ SSE 逐节点流式。

## 🏗️ 架构

```text
                    浏览器工作台（五块面板 · SSE 逐节点流式）
                                  │
┌─────────────────────────────────┴──────────────────────────────────┐
│  intake ──→ deid ──→ qc ─┬─(分辨率/无信号)→ NEEDS_REPEAT 终止        │
│                          │                                          │
│                          ├──→ reader_a (CNN)  ─┐                    │
│                          ├──→ reader_b (VLM)  ─┤  独立读片           │
│                          │    prompt 不含对方任何输出                 │
│                          ↓                     ↓                    │
│                        merge（分歧检测 + Cohen's kappa）             │
│                          │                                          │
│            ┌─────旁路────┤                                          │
│            ↓             ↓                                          │
│   critical_triage    arbiter（只处理分歧项 · 看图 + 指南 RAG）        │
│      抢报告先告警         ↓                                          │
│                    report_writer                                    │
│                          ↓                                          │
│                    evidence_check ──(零引用句)──┐                   │
│                          ↓                     │ 重试一次            │
│                    language_guard  ←───────────┘                    │
│                          ↓                                          │
│                    review_queue（人在环签发）                        │
└────────────────────────────────────────────────────────────────────┘
        guardrails: input(注入筛查) / process(预算) / output(口径+PHI)
```

**双读的独立性是被测试守护的**，不是靠自觉：`tests/test_reader_independence.py` 断言 `reader_a` 的任何输出都不得出现在 `reader_b` 的 prompt 里，`describer` 降级模式下依然独立。

**降级分支是正式设计而非风险备注**：`kappa < 0.4 或 分歧率 > 40%` 时 `reader_b` 从「独立读者」降为「描述生成器」（只描述征象不下判定，判定权归 CNN，仲裁语义改为「描述是否支持 CNN 判定」），由 `READER_B_MODE` 控制，两种模式都有测试覆盖。

## 🧱 技术栈

| 层 | 选型 | 说明 |
| --- | --- | --- |
| 视觉 reader_a | **TorchXRayVision** `densenet121-res224-all` | 18 类概率 + Grad-CAM；类目数从 `model.pathologies` **运行时读取**，禁止硬编码 |
| 视觉 reader_b | **Qwen3-VL-32B-Instruct**（SiliconFlow，OpenAI 兼容） | 看图独立读片；离线时降级为 `OfflineVLMClient` 替身 |
| 编排 | **LangGraph** | 条件路由 + 危急值并行旁路 |
| 服务 / 工作台 | **FastAPI** + 单文件 HTML + SSE | 零构建，无打包步骤 |
| 配置 | **pydantic-settings** | 裸变量名（无前缀），`tests/conftest.py` autouse 隔离真 `.env` |
| 持久化 | 内存 / **SQLite** | `RUN_STORE` 切换；审计层**不存** `indication`/`history_text` 原文 |
| 可观测 | **Langfuse**（可选） | 一次阅片 = 一条 trace |
| 数据 | **NLM Open-i / IU 胸片集** | 7470 图 + 3955 份配对报告，公开直下 |
| 运行时 | Python 3.12 · torch 2.2.2 (CPU) | 3.14 无可靠 torch wheel；容器内钉死 `numpy==1.26.4 + torch==2.2.2` |

> **刻意避开 MIMIC-CXR**：PhysioNet 协议禁止把数据传第三方 API，选它会直接堵死云端 VLM 这条路线。

## 🚀 快速开始

```bash
# 1. 建虚拟环境并安装
python3.12 -m venv .venv                  # torch 在 3.14 上无可靠 wheel
.venv/bin/pip install -e ".[cv,llm]"

# 2. 跑测试（离线、hermetic、零 key）
PYTHONPATH=src .venv/bin/pytest -q        # 352 passed, 2 deselected in ~83s
PYTHONPATH=src .venv/bin/pytest -m slow   # 2 个真权重用例，约 10 分钟

# 3. 跑评测门禁
make eval                                 # 当前 GATE: FAIL（G1，见「诚实的局限」）

# 4. 启动工作台
PYTHONPATH=src .venv/bin/uvicorn medscope.app:app --reload   # → /workbench
```

仓内已含 3 例样本切片，**克隆即可离线跑通全流程**，无需 key、无需网络。
Docker：`docker compose up`（CNN 权重在**构建时预取**，运行时零下载）。

**接入真实 VLM**（可选，`reader_b` 从替身换成真模型）：

```bash
cp .env.example .env
# VLM_BASE_URL=https://api.siliconflow.com/v1
# VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct     ← 填前先核对目录，不要凭记忆猜
# VLM_API_KEY=sk-...
# USE_REAL_VLM=true
curl -s -H "Authorization: Bearer $VLM_API_KEY" \
  "$VLM_BASE_URL/models?type=text&sub_type=chat" | jq -r '.data[].id'
```

**完整数据集**：

```bash
PYTHONPATH=src .venv/bin/python scripts/fetch_openi.py --image-bytes 20000000
```

图像包 1.36 GB、实测带宽约 379 KB/s，需一小时以上；`--image-bytes` 用 HTTP Range 只取前缀，20 MB 即可解出约 108 张完整 PNG。

## 💬 使用示例

```bash
# 仓内样本研究（3 例，克隆即有）
curl -s localhost:8000/studies | jq -r '.studies[].study_id'    # → 38 / 797 / 1187

# 单份研究穿过全流程，返回五块面板（双读 / 分歧仲裁 / 证据 / 危急值 / 门禁）
# 面板自带 base64 原图供工作台渲染，命令行看时剔掉更清爽
curl -s -X POST localhost:8000/workbench/run \
  -H 'Content-Type: application/json' \
  -d '{"study_id":"1187"}' | jq 'del(.image.data_url)'

# 已跑过的研究直接取快照
curl -s 'localhost:8000/workbench/dashboard?study_id=1187' | jq 'del(.image.data_url)'

# 逐节点 SSE 流式
curl -N 'localhost:8000/workbench/stream?study_id=1187'

# 历史运行审计
curl -s localhost:8000/runs | jq
```

**VLM 基线校准**（决定 `reader_b` 当读者还是描述器）：

```bash
USE_REAL_VLM=true PYTHONPATH=src .venv/bin/python scripts/calibrate_vlm.py --limit 30
```

离线模式下该脚本**拒绝给出结论并以 exit 2 退出**——唯一可用的离线 `reader_b` 从每份研究**自己的配对报告**反推 findings，拿它算 kappa 等于让报告和自己比对，kappa 会无意义地高。用假数字支撑真架构决策，比不做决策更糟。

## 📊 评测

```bash
make eval        # 或 EVAL_ARGS="--suite critical" make eval
```

**6 套件 / 五硬一软**，任一硬门不过即 **exit 2**：

| 套件 | 类型 | 覆盖 | 当前 |
| --- | --- | --- | --- |
| `critical` **G1 危急值零漏判** | **硬门** | recall = 1.0（FPR 仅软警告，上限 0.35） | ❌ **FAIL** — recall 0.815 (22/27)，FPR 0.808 |
| `evidence` **G2 无证据不出句** | **硬门** | 覆盖率 = 1.0，裸断言 = 0 | ✅ PASS (n=3) |
| `language` **G3 诊断口径红线** | **硬门** | 拦截率 = 1.0，免责声明率 = 1.0，误伤率 = 0 | ✅ PASS (n=14) |
| `phi` **G4 PHI 零泄漏** | **硬门** | 泄漏数 = 0 | ✅ PASS (n=20) |
| `robustness` **G5 鲁棒性** | **硬门** | 注入拦截率 = 1.0，不变性 = 1.0 | ✅ PASS (n=9) |
| `golden` | soft | 端到端产出完整度 | ✅ PASS (n=3) |

单测：**352 passed, 2 deselected**（`slow` 标记的真权重用例默认不跑）。

### 一条原则

> **任何能让交付失败的门，都必须有一个已被演示过的失败方式。**

一道接了 exit 2 的门是有实权的，它能拦住发布。如果从来没人见过它变红，我们并不知道它是真有这个能力，还是一盏焊死的绿灯。五道门各配一个「故意打断实现 → 门必须变红」的测试，`golden` 保持软指标。

**写破坏测试的过程本身就抓出过一个 bug**：`injection_block_rate` 原先是对「已被 `neutralize_untrusted` 净化过的 prompt」跑 `detect_injection`——于是**阉割检测器反而让指标升到虚假的 1.0**，破坏它指标反而变好看。已改建到真实决策路径 `guardrails.input.screen_intake` 上。证伪测试不只验证门有效，还会校验「你到底在测什么」。

## 🔍 诚实的局限

这一节记录**已知不成立的部分**。它不是待办清单的委婉说法，而是判断本项目结论可信到什么程度的**唯一依据**。上面那张表必须连同这一节一起读。

### G1 为什么是红的（不是阈值问题，也不是调参能解决的）

在完整数据集上、用真实 CNN 跑真实像素，把「判别力」和「工作点」拆开量：

| 变体 | AUC 气胸 | AUC 积液 | recall@0.3 | FPR@0.3 |
| --- | --- | --- | --- | --- |
| **当前实现**（center-crop，取一张图） | 0.646 | 0.923 | 22/27 = .815 | .808 |
| 不裁剪 + 读全研究所有图 | 0.659 | 0.917 | **27/27 = 1.000** | .808 |
| crop+squash TTA 平均 + 读全图 | **0.681** | 0.914 | 26/27 = .963 | .808 |
| 5 个 xrv 权重集集成 | 0.681 | 0.932 | **27/27 = 1.000** | .923 |
| 只用 `mimic_ch` 单模型 | **0.451** | 0.861 | **27/27 = 1.000** | .962 |

最后一行是决定性的：**`mimic_ch` 的气胸 AUC 只有 0.451（比抛硬币还差），却单枪匹马把 G1 打到满分**——因为它对 96% 的阴性片也报警。集成之所以变绿，主要就是它在拉。**「集成」在这里是穿了外套的降阈值。**

**气胸 AUC ≈ 0.65 是天花板。** 在这条曲线上，recall=1.0 的代价必然是 FPR 0.8~0.96。一道靠着对 80% 正常片报警来通过的门，和之前那版「合成 Finding 直接喂给 triage」的绿灯是同一种绿。所以红字留着。

### 三个已定位、尚未修的读图缺陷

- **选图不确定**：`next(root.rglob(pattern))` 取文件系统顺序的第一个。51 份研究里 **19 份首选的是侧位片**，而侧位喂进正位模型时 47 张里有 40 张气胸告警（85% 乱报）。同一份代码同一批数据，分数取决于 OS 返回顺序。
- **一份研究只读一张**：放射科医生读的是 study 不是 file。多视图全部保留在 `image_paths`，但只读第一张，侧位印证未实现。
- **center-crop 裁掉了病灶所在**：Open-i 的 PNG 是 512×624 这类竖版，`XRayCenterCrop` 上下各切约 56px——正好是**肺尖（气胸）与肋膈角（积液）**。

### `prob` 不是概率

`densenet121-res224-all` 的 `op_threshs[Pneumothorax] = 0.0098`，`forward()` 内的 `op_norm` 把它映射到 0.5。所以系统里流通的那个数是**工作点相对坐标**，不是校准概率：报告阈值 0.5 = 原始 sigmoid 0.0098，`critical_threshold=0.3` = 原始 sigmoid 0.0059。这解释了为什么气胸概率全挤在 0.50±0.02（原始值 0.01~0.05 被压进 [0.5, 0.52]），也解释了 FPR 0.808。这个数还一路流进 `magnitude_gap`、仲裁器和工作台展示。

### G1 的实际覆盖窄于它的名字

三个危急标签的证据强度差异悬殊，所以 eval **按标签分别输出召回率**，不给一个混合数字——否则两个覆盖良好的标签会抬着第三个走过关口。

- **纵隔气肿**：`densenet121-res224-all` 的 18 类输出里**根本没有这个标签**，`reader_a` 结构上无法产出它。
- **气胸** 15 例、**大量胸腔积液** 14 例；「大量」这一严重程度由概率阈值近似，模型输出的 `Effusion` 并不分级。

保留纵隔气肿标签是刻意的：它临床上就是危急值，**为了让指标好看而删掉它，是用重新定义标准来消灭局限**。

### 纵隔气肿永远无法端到端验证

这一条不是「数据还没取到」，而是**数据不存在**。全库 3955 份报告里唯一一例非否定式的纵隔气肿是 study 895，而**它的图像不在 Open-i 归档中**（7470 张图，3955 份报告里有 104 份无图可配，895 是其中之一）。叠加模型侧没有这个输出——**G1 对这个标签的绿灯，在现有数据与模型下不可能变成真实的端到端证据**。

### 双读的真实质量：形态修好了，结论还没有

配上真实 VLM（`Qwen/Qwen3-VL-32B-Instruct`）跑校准，第一轮暴露的是**合并层的结构缺陷而非模型质量**：三份研究 71 条分歧**全部是 `unique`**，`presence`/`magnitude` 为 0——两个读者从头到尾没发生过一次正面比对。根因是 `reader_a` 对全部 18 个标签都产出 Finding（含自信为阴性的），而 `reader_b` 只报它看见的，于是每个未提及的阴性标签都被记成分歧。修复后：

| | 修复前 | 修复后 |
| --- | --- | --- |
| 分歧 / 研究 | 22.4 | 6.6 |
| mean kappa | 0.200 | 0.335 |
| 两读者共同标签 | 0 | 有 |

**但 n=3~5 太小，`describer` 这个降级结论只是「形态对了」，不是结论。** 要下判断得跑几十份。

### `critical_fpr` 是真测量，但不是校准过的假警报率

它现在确实在量「reader_a 读真实像素时，多少阴性研究被误报」（0.808）。但金标准的 26 例阴性是按报告文本挑的，不是按分布抽的，所以这个数不能当生产环境的假警报率读。它仍只是软警告——recall 是唯一硬要求。

### G2 存在循环性

eval 里评的草稿**已经过运行时证据门处理**（`graph._apply_evidence_gate`，重试一次再剥离），所以绿灯主要在复验那道运行时门，而非检验一条无保护的路径。

### PHI 脱敏不完备，且这个不完备是结构性的

`deid.py` 的规则由**注入器的形态清单**驱动，而手工枚举的清单永远是现实的真子集——**测试集的对抗性上限就封顶在那份清单上**。开发中先后发现并修复八类漏检（连写手机号、CJK 紧邻标识符、三字名与复姓、`+86` 前缀、间隔号音译名、点分日期、小写前缀……），**每一次发现之前测试都是全绿的**。后来把注入器重构成沿命名维度（分隔符 × 前缀 × 大小写 × 贴合方式 × 字符集）组合生成，2000 种子扫描零泄漏。已知仍不支持的形态列在 `deid.py` 的模块 docstring 里。**请当作「对已发现的形态有效」，而不是「PHI-proof」。**

### 其他

- **未接真实 DICOM**：Open-i 的 DICOM 分发不可达，仓库只有 PNG。DICOM 标签白名单逻辑由**内存中合成的 `pydicom.Dataset`** 覆盖，未在真实 DICOM 文件上验证过。
- 正位/侧位判据是镜像对称性启发式（实测正位 0.833–0.851、侧位 0.151/−0.049，阈值 0.5），**不是经过验证的视图分类器**，故仅作软告警。
- 环境中 torch 2.2.2 与 numpy 2.x 存在 ABI 冲突，`readers/cnn.py` 以 `torch.frombuffer` 绕开（容器内已钉死兼容版本组合）。
- 指南语料是**自行合成的教学材料**，非真实指南摘录，不含任何虚构出处。

## 🔒 安全

面向「患者数据 × 不可信临床散文 × 多模态 LLM」的纵深防御，全部**规则化、可复现**。

**威胁模型** —— ①病历文本与检查申请单是第三方散文，会被拼进 VLM / 仲裁器的 prompt；②影像与报告一旦外发即不可撤回；③审计层是最容易在「可追溯」这个正当理由下积累 PHI 的地方。

| 防线 | 措施 | 位置 |
| --- | --- | --- |
| 脱敏 | DICOM 标签白名单 + 文本 PHI 规则；G4 硬门量 0 泄漏 | `deid.py` |
| 数据→LLM | 每个 prompt 边界用 `neutralize_untrusted()` 包裹 + 归一化 | `readers/vlm.py` · `arbiter.py` |
| 输入 | `screen_intake` 注入筛查（**记录不阻断**） | `guardrails/input.py` |
| 过程 | `max_llm_judgments` / `max_findings` 预算封顶 | `guardrails/process.py` |
| 输出 | 诊断口径红线 + 免责声明强制 + 输出侧 PHI 复查 | `guardrails/output.py` · `language.py` |
| 存储 | 审计层对**所有状态**一律不存 `indication`/`history_text` | `store.py` |

**注入不阻断阅片，是一个临床判断而非疏忽**：文本在每个 prompt 边界已净化，且真实临床散文极易被误报为注入；**拒绝阅片的临床代价远大于带标记阅片**。只有结构性问题（无图像）才终止流程。

**审计层为什么一刀切**：`GUARDRAIL_BLOCKED` 的研究**不经过 deid 节点**（intake 直接路由到终止），那种状态下 `indication`/`history_text` 仍是未脱敏原文，而系统里**没有字段标记「这一实例上 deid 跑没跑」**。既然分不清，就对所有状态都不存。存的是 `deid_report` 摘要（只含规则名与哈希）。

**密钥**：`.env` 全程 gitignore，仅 `.env.example` 入库；`tests/conftest.py` 的 autouse fixture 枚举 `Settings.model_fields` 做隔离——**手工清单会漏掉新增字段，而失效表现是测试悄悄读到了真 key**。

## 🔭 可观测

一次阅片 = 一条 Langfuse trace，逐节点可下钻（qc / reader_a / reader_b / merge / arbiter / report / gates）。可选开启：

```bash
LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_HOST=https://us.cloud.langfuse.com
```

未配 key 时 `Settings.tracing_enabled` 为 False，全链路静默跳过，不影响任何测试与门禁。

**工作台本身就是可观测面**：五块面板把双读对照、分歧与仲裁裁决、证据溯源、危急值时间线、门禁状态摊开；悬停报告里任一句，图上对应区域与证据卡同时高亮。**仲裁裁决如实存储在 `StudyState.arbitration_records`**，不从「finding 在不在最终集合 + needs_human」反推——反推与 `arbiter.py` 当下行为一致，但隐式耦合其内部，一旦 REJECT 语义变化就会**自信地报出错误裁决**。审计视图误报比不报更糟。

## 📁 项目结构

```text
src/medscope/
  data/openi.py      数据集加载与图文配对
  data/synth_phi.py  合成 PHI 注入器（G4 的验证前提，必须先于 deid.py 写）
  deid.py            DICOM 白名单 + 文本 PHI 规则 + scan_payload（G4 评分函数）
  qc.py              图像质控 + 正位/侧位筛查
  ontology.py        标签本体；CRITICAL_LABELS 是 G1 的单一真源（不进 config）
  readers/cnn.py     reader_a：概率 + Grad-CAM
  readers/vlm.py     reader_b + 独立性守护 + describer 降级
  merge.py           合并、分歧检测、Cohen's kappa
  arbiter.py         看图仲裁 + 指南 RAG
  critical.py        危急值分诊（与报告并行）
  report.py          四段草稿撰写      evidence.py  G2 证据校验
  language.py        G3 口径红线       guardrails/  input / process / output 三层
  graph.py           LangGraph 编排    runner.py    执行入口
  workbench.py       五块 Dashboard + SSE          app.py  FastAPI
  eval.py            五道硬门          store.py     审计持久化
scripts/
  fetch_openi.py           数据集拉取（HTTP Range 前缀下载）
  build_critical_goldset.py G1 金标准候选生成（候选须人工核对后才入库）
  calibrate_vlm.py         reader_b 基线校准 / 降级决策
  gen_phi_fixture.py       G4 夹具生成
data/evals/                critical.json（53 例人工核对）· phi.json
```

## 🧩 配置

`.env`（见 `.env.example`）关键项 —— **变量名无前缀**：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `OPENI_ROOT` | `data/openi` | 数据集根目录 |
| `CNN_WEIGHTS` | `densenet121-res224-all` | reader_a 权重集；类目数运行时读取，**不要硬编码 14 类** |
| `CNN_PROB_THRESHOLD` | `0.5` | 进报告的阈值（= 模型工作点，见「`prob` 不是概率」） |
| `CRITICAL_THRESHOLD` | `0.3` | 危急值告警阈值，**刻意低于**报告阈值 |
| `READER_B_MODE` | `reader` | `reader`（同侪）\| `describer`（降级为描述器） |
| `VLM_MODEL` / `VLM_BASE_URL` / `VLM_API_KEY` | 空 | reader_b 的 OpenAI 兼容端点；**填前用 `/models` 核对 id，不要凭记忆猜** |
| `USE_REAL_VLM` | `false` | 关时 reader_b 走 `OfflineVLMClient` 替身 |
| `KAPPA_FLOOR` / `DISAGREEMENT_CEILING` | `0.4` / `0.4` | `calibrate_vlm.py` 的降级判据 |
| `MAX_LLM_JUDGMENTS` | `12` | 单次阅片的仲裁预算上限 |
| `RUN_STORE` / `SQLITE_PATH` | `memory` | `memory`（进程内）\| `sqlite`（跨重启持久） |
| `LANGFUSE_*` | 空 | 两个 key 都填才启用 tracing |

> 危急值标签清单**不在这里**——单一真源是 `ontology.CRITICAL_LABELS`。放两处必然拼写漂移，而漂移的表现是 G1 静默失效。

## 🗺️ 路线图

- [x] **阶段①** 确定性地基：数据 / 脱敏 / 质控 / CNN 读片 / 合并 / 危急值 / 口径 / 证据
- [x] **阶段②** 智能层：VLM 双读 / 看图仲裁 / 报告撰写 / LangGraph 编排 / 三层护栏
- [x] **阶段③** 交付层：工作台 / 五道硬门 / 审计持久化 / 容器 / CI
- [x] **接入真实 VLM** —— Qwen3-VL-32B；并修掉暴露出的合并层比对语义与解析层否定式缺陷
- [ ] **让 G1 的红字有意义地变绿** —— 按 study 读全部视图（含确定性选图）+ 停止裁掉肺尖与肋膈角；预期能到 27/27，但**必须同时把 AUC 0.65 这个事实写在报告里**，否则又是一盏焊死的绿灯
- [ ] **reader_b 规模化校准** —— 跑满几十份研究，让 `reader_b_mode` 的决策有统计意义
- [ ] **P4 生成轨** —— SD 合成稀有阳性 + **反事实对照图**（生成「移除病灶后的同一张片」，可解释性强于 CAM）
- [ ] **P4 时序轨** —— SD 渐进 inpainting 造病灶演进序列（Open-i 无纵向随访配对，合成的好处是变化幅度已知、时序对比能真做 eval）
- [ ] 真实 DICOM 端到端 —— 现有白名单逻辑仅由合成 `Dataset` 覆盖

## 📄 许可证

[MIT](LICENSE) © Kevin Tu · 武道AI 工程修炼系列。

medscope 是**教学 Demo，不是医疗器械**，未经任何监管审批，不得用于临床诊断或诊疗决策。它只产出报告草稿，签发权永远在执业医师。所有数据来自公开去标识数据集，不接触真实患者数据。
