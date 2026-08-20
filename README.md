# medscope — 胸片阅片副驾

> **教育演示项目，非医疗器械。** 不用于临床诊断，不用于任何真实诊疗决策。
> 数据全部来自公开去标识数据集（NLM Open-i / Indiana University 胸片集），不接触任何真实患者数据。
> 系统只产出**报告草稿**，签发权永远在执业医师。

`medscope` 取自放射科真实术语 **second reader（第二读片者）**——既是架构隐喻，也是定位声明：AI 是第二个读片的人，不是主诊医生。

---

## 这是什么

输入一张胸片 + 既往病历文本 + 检查申请单，输出一份**每句结论都挂着证据**的结构化报告草稿，外加一条独立的危急值告警通道。

**招牌是异质双读一裁**，直接搬自放射科真实的双读制 SOP：

- **`reader_a`** — 判别式 CNN（TorchXRayVision DenseNet-121），出**定量概率 + Grad-CAM 定位**，但读不懂临床语境
- **`reader_b`** — 生成式 VLM 看原图独立读片，能结合病史与检查目的推理、能描述标签集之外的征象，但不给校准概率、不给定位
- 两者**失效模式正交**，所以分歧不是要抹平的噪声，而是「这张片上有一处需要人看」的精确指针
- **仲裁只在分歧项上花 LLM 预算**，且仲裁器自己也看图——只看两个读者的结论等于拿先验掂量断言，那不是仲裁
- **危急值走旁路**，不等报告写完就告警，照搬医院的 critical results communication 制度

```
intake → deid → qc ─┬─(不合格)→ NEEDS_REPEAT 终止
                    ├→ reader_a ─┐ 独立读片
                    ├→ reader_b ─┤ (prompt 中不含对方任何输出)
                    ↓            ↓
                  merge (分歧检测 + Cohen's kappa)
                    ├──旁路──→ critical_triage → 抢报
                    ↓
                 arbiter (只处理分歧项)
                    ↓
        report_writer → evidence_check → language_guard → review_queue
```

## 五道硬门

`make eval` 决定这套系统能不能交付。任一硬门不过即 **exit 2**。

| 门 | 指标 | 当前 |
|---|---|---|
| **G1 危急值零漏判** | recall = 1.0；FPR 仅软警告 | PASS (n=53) |
| **G2 无证据不出句** | 覆盖率 = 1.0，裸断言 = 0 | PASS (n=3) |
| **G3 诊断口径红线** | 拦截率 = 1.0，免责声明率 = 1.0 | PASS (n=14) |
| **G4 PHI 零泄漏** | 泄漏数 = 0 | PASS (n=20) |
| **G5 鲁棒性** | 注入拦截率 = 1.0，不变性 = 1.0 | PASS (n=9) |

**G1 的非对称是刻意的**：漏一例气胸可能致命，多报一例只花医生三十秒。召回是硬门、误报率只是软指标，**两者不合并成 F1**——一个平衡后的数字会让召回退化藏在精度提升背后，而这在医疗域正好是反的。

**每一道硬门都配了一个「故意打断实现 → 门必须变红」的测试。** 遵循一条原则：

> **任何能让交付失败的门，都必须有一个已被演示过的失败方式。**

一道接了 exit 2 的门是有实权的，它能拦住发布。如果从来没人见过它变红，我们并不知道它是真有这个能力，还是一盏焊死的绿灯。

## 快速开始

```bash
python3.12 -m venv .venv                  # torch 在 3.14 上无可靠 wheel
.venv/bin/pip install -e ".[cv,llm]"
PYTHONPATH=src .venv/bin/pytest -q        # 346 passed, 1 deselected
make eval                                 # GATE: PASS
PYTHONPATH=src .venv/bin/uvicorn medscope.app:app --reload   # → /workbench
```

仓内已含 3 例样本切片，**克隆即可离线跑通全流程**，无需 key、无需网络。

Docker：`docker compose up`（CNN 权重在构建时预取，运行时零下载）。

完整数据集：

```bash
PYTHONPATH=src .venv/bin/python scripts/fetch_openi.py --image-bytes 20000000
```

图像包 1.36 GB、实测带宽约 379 KB/s，需一小时以上；`--image-bytes` 用 HTTP Range 只取前缀，20 MB 即可解出约 108 张完整 PNG。

---

## 诚实的局限

这一节记录**已知不成立的部分**。它不是待办清单的委婉说法，而是判断本项目结论可信到什么程度的**唯一依据**。上面那张全 PASS 的表，必须连同这一节一起读。

### G1 的实际覆盖窄于它的名字

三个危急标签的证据强度差异悬殊：

- **纵隔气肿**：`densenet121-res224-all` 的 18 类输出里**根本没有这个标签**，`reader_a` 结构上无法产出它；金标准里**确证阳性仅 1 例**（全库 3955 份报告只有 7 份提及，5 份是否定式、1 份是既往史）。其召回依赖尚未接入的 VLM。
- **气胸** 15 例、**大量胸腔积液** 14 例。
- 「大量」这一严重程度判断由概率阈值近似，模型输出的 `Effusion` 并不分级。

所以 eval 报告**按标签分别输出召回率**，不给一个混合数字——否则两个覆盖良好的标签会抬着第三个走过关口，然后为它打印一个绿色的 G1。

保留纵隔气肿标签是刻意的：它在临床上就是危急值，**为了让指标好看而删掉它，是用重新定义标准来消灭局限**。

### 金标准没有一例阳性能端到端跑

53 例全部依据报告文本 + MeSH 人工核对，但**确证阳性病例的图像都不在本地那 108 张里**。G1 今天验证的是 `triage()` 的阈值、去重与本体过滤逻辑——**在 finding 已被正确识别的前提下**。它没有验证 reader_a/reader_b 能否从像素中真的检出这些征象。

### `critical_fpr` 名不副实

这个指标结构上恒为 0：负例探针合成的是固定的非危急标签，`triage()` 的标签过滤保证它永远不会告警。它测的是「标签过滤没漏」，**不是任何意义上的假警报率**。带着说明保留，是因为拿到完整数据集后它可能变得有意义；但现在不能当校准结果读。

### G2 存在循环性

eval 里评的草稿**已经过运行时证据门处理**（`graph._apply_evidence_gate`，重试一次再剥离），所以绿灯主要在复验那道运行时门，而非检验一条无保护的路径。

### PHI 脱敏不完备，且这个不完备是结构性的

`deid.py` 的规则由**注入器的形态清单**驱动，而手工枚举的清单永远是现实的真子集——**测试集的对抗性上限就封顶在那份清单上**。开发中先后发现并修复八类漏检（连写手机号、CJK 紧邻标识符、三字名与复姓、`+86` 前缀、间隔号音译名、点分日期、小写前缀……），**每一次发现之前测试都是全绿的**。

后来把注入器重构成沿命名维度（分隔符 × 前缀 × 大小写 × 贴合方式 × 字符集）做组合生成，2000 种子扫描零泄漏。已知仍不支持的形态列在 `deid.py` 的模块 docstring 里。**请当作「对已发现的形态有效」，而不是「PHI-proof」。**

### 双读的真实质量未经验证

没有 VLM key，`vlm_model` 为空。唯一可用的 `reader_b` 是 `OfflineVLMClient`——它的 findings 从每份研究**自己的配对报告**反推，拿它算 kappa 等于让报告和自己比对。

因此 `scripts/calibrate_vlm.py` 在离线模式下**拒绝给出结论并以 exit 2 退出**。`reader_b_mode` 保持默认的 peer 模式，但**没有任何证据支持这个选择**。工作台上显示的 kappa 与分歧数反映的是替身的行为，不是模型的能力。

### 未接真实 DICOM

Open-i 的 DICOM 分发不可达，仓库只有 PNG。DICOM 标签白名单逻辑由**内存中合成的 `pydicom.Dataset`** 覆盖，未在真实 DICOM 文件上验证过。

### 审计层不存原始文本

`GUARDRAIL_BLOCKED` 的研究**不经过 deid 节点**（intake 直接路由到终止），所以那种状态下 `indication`/`history_text` 仍是未脱敏原文；而系统里没有字段标记「这一实例上 deid 跑没跑」。因此持久化层对**所有状态**一律不存这两个字段，`read_a`/`read_b` 的原始生成文本同理。存的是 `deid_report` 摘要（只含规则名与哈希）。

### 其他

- 正位/侧位判据是镜像对称性启发式（实测正位 0.833–0.851、侧位 0.151/−0.049，阈值 0.5），**不是经过验证的视图分类器**，故仅作软告警。
- 多视图全部保留在 `image_paths`，但只读第一张；侧位印证未实现。
- 环境中 torch 2.2.2 与 numpy 2.x 存在 ABI 冲突，`readers/cnn.py` 以 `torch.frombuffer` 绕开（容器内已钉死兼容版本组合）。
- 指南语料是**自行合成的教学材料**，非真实指南摘录，不含任何虚构出处。

---

## 项目结构

```
src/medscope/
  data/openi.py      数据集加载与图文配对
  data/synth_phi.py  合成 PHI 注入器（G4 的验证前提）
  deid.py            DICOM 白名单 + 文本 PHI 规则 + scan_payload（G4 评分函数）
  qc.py              图像质控 + 正位/侧位筛查
  ontology.py        标签本体，CRITICAL_LABELS 是 G1 的单一真源
  readers/cnn.py     reader_a：概率 + Grad-CAM
  readers/vlm.py     reader_b + 独立性守护 + describer 降级
  merge.py           合并、分歧检测、Cohen's kappa
  arbiter.py         看图仲裁 + 指南 RAG
  critical.py        危急值分诊
  report.py          四段草稿撰写
  language.py        G3 口径红线    evidence.py  G2 证据校验
  guardrails/        input / process / output 三层
  graph.py           LangGraph 编排    runner.py  执行入口
  workbench.py       五块 Dashboard + SSE
  eval.py            五道硬门        store.py   审计持久化
```

## 许可

MIT — Kevin Tu
