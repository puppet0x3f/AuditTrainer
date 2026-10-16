# AuditTrainer：责任分解、契约履行与有界修复
## v3.0 从零实现版：框架优先型去工程化研究路线与实现规格

- **文档版本**：v3.0-R1，作为 v3.0 的从零实现修订版。
- **修订日期**：2026-09-06。
- **执行起点**：空项目目录。全部项目模块、数据集、适配器与实验从本路线重新建立；不存在默认已完成的阶段。
- **允许参考**：共享递归 policy、结构化 action、契约、显式消费、ownership/lineage、冻结补集等设计思想。
- **不作为前提**：任何旧项目代码、旧训练数据、旧实验结论或旧微调 checkpoint。
- **研究主线**：责任分解学习 → 契约条件下的责任履行与组合学习 → 失败定位、修复边界确定与有界修复。
- **默认实现范围**：Python、一个公开基础模型、一种递归 runtime、一套 SFT；正式主域先采用 APPS 的可支持子集。
- **状态说明**：本文是待实现的技术规格和研究路线，不是已完成代码、已执行训练或有效性证明。文中的命令是新仓库需要实现的入口。

> **最重要的执行规则**：先新建一个能真实完成“规划—递归—代码实现—显式消费—验证—失败定位—有界修复—动作数据导出—SFT 更新—checkpoint 回接”的最小系统。每完成一个模块就做接口与正确性测试；模型效果不好只记录，不阻止继续补齐后续模块。框架完成后才比较机制、扩大训练、优化效果和执行正式评测。

“从零”指重新实现项目，而不是从随机参数预训练大模型。默认从公开的 `Qwen/Qwen3-8B` 权重启动，训练新的 LoRA adapter；不加载任何本项目此前微调权重。

---

## 0. 本次修订改变什么

保留 v3.0 的三主线、A0–A10 阶段骨架、框架优先、局部效果不阻塞、有限开发、二因素训练对照和固定修复器实验。删除“盘点已有成果”“复用现有 runtime/SFT”“沿用已训练模型”等实施前提。

本版新增并冻结一份可直接用于开发的初始规格：

| 原先偏抽象的事项 | 本版的确定起点 |
| --- | --- |
| 递归 policy | `Policy.act(observation) -> PolicyOutput`；根与所有子节点共享权重，每次输出一条 JSON action |
| action 集合 | 10 类：`plan`、`write_code`、`verify`、`consume`、`reject`、`request_child_repair`、`localize`、`repair`、`finish`、`abstain` |
| 递归实现 | 单线程、深度优先、顺序执行；`plan(SPLIT)` 显式授权创建子节点；runtime 不替模型消费 |
| 责任载体 | 节点契约 + 一节点一源码模块 + 不可变版本；函数级边界起步 |
| child artifact | 可调用代码模块及其依赖清单，不是已经算出的题目答案；运行结果另存 observation/evidence |
| policy observation | 固定字段、只含当时可见状态；训练与推理共用同一个构造函数 |
| SFT 单位 | 一个决策时刻：`决策前 observation → 一条正确 action`；不是整条 trace 全 token 学习 |
| SFT 首批来源 | 从新建 fixture 的 ScriptedPolicy 真实执行得到，不等待未训练模型自然产生正例 |
| SFT 实现 | Transformers `Trainer` + PEFT QLoRA；明确 prompt/target mask，不依赖模板自动识别 assistant |
| 框架完成 | CPU 脚本链、真实 LLM 调用、真实参数更新、checkpoint 回接、统一报告均跑通；不要求效果提升 |
| 可选扩展 | DPO、RL、专用定位训练、复杂切片在核心框架之后按证据加入 |

这些 action/payload、状态、文件路径、预算和训练参数是**本次提出的初始实现设计**，不是对旧项目实现状态的转述。外部软件 API 依据见文末 [E1]–[E6]；研究设计参数是建议起点，不代表已验证最优值。

---

## 1. 研究目标与完成层次

### 1.1 统一研究问题

> 能否训练一个共享递归代码策略，学会形成可验证、可组合的责任结构，在代码实现和父节点组合中保持这种结构，并在出现失败时利用同一结构实施有证据支持、范围受控的修复？

三部分不是三个独立平台：

```text
责任分解：plan(ATOMIC / SPLIT)
    ↓ 冻结子契约、分配直接实现职责
责任履行：write_code + verify + consume / reject / request_child_repair
    ↓ 形成带版本和证据的程序
责任干预：localize + repair，必要时 plan 的显式修订
    ↓ 冻结补集、可见重放、重新提交
```

前两部分是主要学习贡献。第三部分先采用固定模型与有限候选算法，检验前两部分的下游价值；不以专用 Locator 训练成功为开工条件。

### 1.2 三种完成状态

| 状态 | 必须回答的问题 | 不代表什么 |
| --- | --- | --- |
| 工程闭环完成 | 所有接口、证据、数据、参数更新与回接是否真实工作？ | 不代表模型学会责任或论文假设成立 |
| 核心学习验证完成 | 用对照检验了分解与履约能力吗？ | 不要求每个假设都是正结果 |
| 科学闭环完成 | 修复价值、成本、泛化、局限是否有对应证据？ | 不等于仅凭流程完整即可支持投稿竞争力 |

### 阅读与执行导航

先读第 2 节创建仓库，再按第 11 节 F1–F9 顺序开发。第 3–9 节是这些阶段共同遵守的协议；第 12–15 节专门落实数据与 SFT；第 16–24 节对应框架之后的 A2–A10 研究实验；第 25 节给出贯穿示例，第 26 节提供交付清单。

**不要先完成所有实验脚本才开始训练接线，也不要先写完全文的可选模块才宣布 A1 完成。**

### 1.3 开发不追求“节点越细越好”

叶节点应有清楚的主要目标、可实施接口和可以说明的验证方式。允许合理的原子实现、私有 helper 与必要局部计算。父节点的整体义务仍由父子共同完成，但父节点**直接实现的责任**必须与子节点的直接实现责任区分。

不把增加节点、递归深度、调用次数或契约条数当奖励。功能正确也不自动说明责任保持：parent 可以绕过 child 后重新计算出同一答案。

---

## 2. A0：从空目录创建新项目

### 2.1 首版固定技术选择

| 项目 | 起步选择 | 不在 A1 实现 |
| --- | --- | --- |
| 语言 | Python，开发环境选一个 Python 3.11 环境并锁定依赖 | 多语言、编译器适配平台 |
| 策略模型 | `Qwen/Qwen3-8B`，冻结公开 revision | 模型排行榜、多供应商路由 |
| 推理后端 | 本地 Transformers，一次只加载一个活动策略实例 | vLLM 服务、多后端抽象层 |
| 训练 | 单 GPU QLoRA + 一套监督训练脚本 | 分布式训练、多 adapter 编排 |
| 调度 | 顺序深度优先 | 并行子智能体、异步队列 |
| 代码单元 | 每节点一个模块，含一个公开函数及私有 helper | 任意跨文件重构与完整程序依赖图 |
| 存储 | JSON/JSONL、源码文件、版本 hash | 图数据库、向量库、Web 平台 |
| 执行 | 独立、受限的代码执行环境 | 在训练进程中直接 exec 模型代码 |
| 数据 | 先自建 fixture，随后接 APPS | 框架阶段同时接多个外部域 |

Qwen3-8B 的模型标识及 non-thinking 配置依据其官方模型卡 [E1]。这里选择它是为了固定一个可执行起点，不声称它是当前最优模型。

### 2.2 新仓库目录与文件职责

```text
audittrainer/
├── pyproject.toml
├── requirements.lock             # 实际验证后记录依赖版本，不预填猜测版本
├── README.md
├── STATUS.md                     # 所有阶段初始均为 TODO
├── configs/
│   ├── framework.yaml
│   ├── sft_smoke.yaml
│   ├── sft_main.yaml
│   └── evaluation.yaml
├── src/audittrainer/
│   ├── __init__.py
│   ├── cli.py                    # 薄命令入口，不做通用任务平台
│   ├── schema.py                 # 类型、JSON 校验、枚举
│   ├── policy.py                 # ScriptedPolicy / LLMPolicy
│   ├── prompts.py                # 固定系统提示与观测序列化
│   ├── runtime.py                # step、递归栈、状态与预算
│   ├── artifacts.py              # 模块版本、bundle、ownership、消费记录
│   ├── verifier.py               # suite 绑定、可见执行、三态与错误状态
│   ├── sandbox.py                # 隔离执行与资源上限
│   ├── repair.py                 # 候选、ticket、事务式补丁、回滚
│   ├── trace.py                  # 原始 action、前后状态、证据与成本
│   ├── dataset.py                # root 导入、分区、样本导出
│   ├── tokenize_sft.py           # chat prefix、target token、label mask
│   ├── train_sft.py              # 真实参数更新、保存与恢复
│   └── evaluate.py               # root 级评价、异常分母、成本
├── tests/
│   ├── test_schema.py
│   ├── test_runtime.py
│   ├── test_artifacts.py
│   ├── test_verifier.py
│   ├── test_repair.py
│   └── test_sft_data.py
├── data/
│   ├── fixtures/
│   ├── manifests/
│   ├── raw_traces/
│   └── sft/
└── runs/                         # 每次运行一个目录；默认不提交生成产物到 Git
```

不要一次生成大量空类。文件在对应 F 阶段需要时创建；某些文件开始只有几十行是正常状态。

### 2.3 A0 当次交付

创建仓库和配置；安装一个经过实际 import 检查的训练环境；保存模型 revision、tokenizer/chat-template hash、依赖版本、GPU/精度条件；定义数据分区；写 `STATUS.md`，第一项为“F1：实现 schema 与脚本原子任务”。

不设置历史盘点、旧模型迁移、旧结果复现等任务。若新选入的公开题目已经被开发者用于设计或检查，依照**本次使用事实**归入开发范围；重新下载不使它成为未见题。

### 2.4 建议起始配置

```yaml
protocol_version: at-v3r1
model:
  name_or_path: Qwen/Qwen3-8B
  revision: REQUIRED_RESOLVED_COMMIT
  enable_thinking: false
  backend: transformers
runtime:
  schedule: depth_first_sequential
  max_depth: 2                 # root=0；能够真实运行到孙节点
  max_nodes: 7
  max_children: 3
  max_policy_calls: 64         # 全 root 共用，不为 child 重置
  max_total_tokens: 196608     # 所有输入+输出，共享累计上限
  max_verify_calls: 32
  max_child_repair_requests: 1 # 每条 edge
  max_patch_attempts: 3        # 全 root，包括失败与越界补丁
  max_plan_revisions: 1        # 每节点；修订生成的新子节点仍计总预算
  max_protocol_retries: 1      # 每次非法 action 的额外反馈机会
  context_tokens: 8192
  generation_tokens:
    plan: 2048
    write_code: 3072
    repair: 3072
    control: 512
    localize: 768
execution:
  network: disabled
  per_case_timeout_seconds: 2
  memory_mb: 1024
  max_output_bytes: 1048576
repair:
  candidate_limit: 3
  parent_expansions: 1
  automatic_commit_rule: visible_all_pass_no_regression
```

上述数值只用于开始接线；不是能力门禁，也不是论文的默认公平预算。A3/A8 根据开发样本的长度和资源确定最终值。保存输入、输出、验证、修复的实际用量；不可只记配置上限。

---

## 3. 先实现的数据结构：不要先写 prompt

### 3.1 最小对象清单

| 对象 | 必需字段 | 身份由谁生成 |
| --- | --- | --- |
| `RootTask` | root_id、family_id、split、spec、entrypoint、visible_suite_ref、private_suite_ref | 数据导入器 |
| `Contract` | contract_id、version、signature、requires、ensures、owns、uses、must_not、frame | 内容由 policy/数据构造给出；正式 ID、版本由 runtime 分配 |
| `Plan` | node_id、version、mode、children、obligation_map、retained_responsibilities、dataflow | policy 内容 + runtime 身份 |
| `NodeState` | node_id、parent_id、phase、incoming_contract、plan_ref、artifact_ref、pending_edges、last_evidence | runtime |
| `Artifact` | node_id、version、module_path、public_symbol、source_hash、contract_version、dependency_bindings | artifact store |
| `Evidence` | evidence_id、check_kind、target_ref、suite_id、suite_hash、status、coverage、visibility、details | host verifier |
| `Consumption` | consumer_node、child_node、artifact_version/hash、alias、decision、evidence_refs | runtime 记录模型 action |
| `RepairTicket` | snapshot_id、target、editable_files、editable_plan_nodes、rebind_edges、revalidate_nodes、frozen_hashes | runtime 审核模型边界请求后生成 |
| `TraceEvent` | event_id、observation_before、raw_output、parsed_action、host_result、state_after_ref、usage | trace writer |

首版用 dataclass/Pydantic 等一种类型工具即可；无需设计通用 RAAT 服务或数据库。`schema.py` 区分模型提交的 `ContractSpec`（签名与责任内容）和 host 登记的 `Contract`（再加 contract_id/version）。plan.children 不要求模型预知或生成 host 尚未分配的身份。

### 3.2 契约示例

```json
{
  "contract_id": "C_N0_1",
  "version": 1,
  "signature": {"name": "prefix_sums", "parameters": [{"name": "nums", "type": "list[int]"}], "returns": "list[int]"},
  "requires": ["nums 是有限整数列表"],
  "ensures": [
    {"id": "O_PREFIX_LENGTH", "text": "返回长度为 len(nums)+1"},
    {"id": "O_PREFIX_ZERO", "text": "返回列表第 0 项为 0"},
    {"id": "O_PREFIX_VALUES", "text": "第 i 项等于 nums[:i] 的和"}
  ],
  "owns": ["prefix_sum_construction"],
  "uses": [],
  "must_not": ["回答区间查询", "读取标准输入", "修改 nums"],
  "frame": {"public_symbol": "prefix_sums", "private_helper_prefix": "_", "mutate_inputs": false, "external_io": false}
}
```

`type` 起步是受支持类型的规格和轻量检查依据，不承诺形式类型证明。代码签名、参数数量可以硬检查；自然语言前置条件的蕴含关系不能靠字段相等冒充证明。

### 3.2.1 root 契约从哪里来

root 的 `spec` 原文由数据导入器保持不变，`solve(text: str) -> str` 接口由适配器规定。外部义务清单在离线建样时依据规格提取并审查，写入 manifest；运行中的 policy 不能删除它。

fixture 和高置信 TRAIN pilot 使用审查过的外部义务。对尚未细化审查的自然任务，保留完整规格和“满足根规格”的功能义务，但将详细语义覆盖记为 UNKNOWN；不能用 policy 自己列出的较少要求作为评价分母。无需先开发一个能形式化所有自然语言题目的系统。

### 3.3 义务分配必须保留父节点义务

`obligation_map` 表示父义务怎样被子义务和父保留职责共同细化。允许多个子节点共同完成一个父义务；不强行一一对应。

检查器可以发现“父义务完全未被引用”，但不能仅凭所有 ID 出现就判定语义完整。结构覆盖、语义覆盖证据、根任务验证分开存储。

子契约不携带模型可随意指定的可信 oracle 路径。suite 绑定由 host 根据接口、契约版本和已审查注册项决定；不存在匹配 oracle 时返回 UNKNOWN。

### 3.4 artifact 与运行结果的区别

本版默认 `artifact` 是代码：例如 `nodes/N0_1.py` 中的 `prefix_sums`，加其依赖版本清单。

```text
消费 artifact = 明确选择并在最终程序中使用这一版本的代码模块。
执行 artifact = 在某组输入上调用该模块，生成返回值与运行证据。
```

不能把某个测试上的前缀数组当作通用 child artifact，更不能把验证用例的正确答案直接传给 parent 作为求解捷径。跨节点业务数据由 parent 生成的代码在运行时传递。

---

## 4. 首版 action：固定为 10 类

### 4.1 一次 policy 调用只产生一个动作

统一外壳：

```json
{"action": "consume", "args": {"child_id": "N0_1", "artifact_version": 1, "artifact_hash": "h_child_v1", "alias": "prefix_sums", "evidence_ids": ["E_LOCAL_1"], "accept_unverified": false}}
```

`root_id/node_id/state_id` 来自调用时 observation，写入 trace 时由 host 绑定，不让模型自行认领其他节点身份。示例中的 hash 是演示值；实际运行必须是内容计算出的 hash。

### 4.2 action 规格总表

| action | 谁决定、做什么 | 核心 args | runtime 做什么，不做什么 |
| --- | --- | --- | --- |
| `plan` | 当前节点选择原子或分解，必要时显式修订自己的内部计划 | base_plan_version、mode、children、obligation_map、retained_responsibilities、dataflow、ticket_id | 校验并登记；按显式 SPLIT 创建子节点；不替模型补齐遗漏语义 |
| `write_code` | 为当前节点第一次生成实现或 parent glue | contract_version、source、declared_dependencies | 只写当前节点模块、记录版本；不拼接出模型未写的业务调用 |
| `verify` | 请求对指定候选做可见检查 | target_ref、suite_id | host 执行已登记 suite；不接受模型自报 pass |
| `consume` | 选择一个确切 child 版本并承诺使用 | child_id、artifact_version/hash、alias、evidence_ids、accept_unverified | 登记消费与允许依赖；不自动 import、不自动调用、不自动补 glue |
| `reject` | 拒绝某个 child 返回结果 | child_id、artifact_version、reason_code、evidence_ids | 关闭该次接收决定，但保留“被委托职责尚未完成”；不默许 parent 重做 |
| `request_child_repair` | 要求 child 在现有契约下修复 | child_id、artifact_version、evidence_ids、instruction | 重新激活 child、沿用总预算和契约；不由 parent 直接修改 child |
| `localize` | 指定责任候选、类型和所需边界 | target_type、target_id、evidence_ids、boundary_kind、rationale | 基于允许对象生成/拒绝 ticket；这不是因果认证 |
| `repair` | 按 ticket 提交代码替换和必要版本重绑定 | ticket_id、replacements、dependency_updates | 隔离应用、检查冻结补集、可见重放、接受或回滚 |
| `finish` | 结束当前节点并提交其候选 artifact | artifact_version | host 附加真实 PASS/FAIL/UNKNOWN 状态后返回；不信任“我已成功” |
| `abstain` | 当前节点无法继续或证据不足 | reason_code、evidence_ids、suggested_change | 终止/向上报告；不伪造产物与成功 |

**`ATOMIC` 和 `SPLIT` 是 `plan.args.mode` 的两个值，不是两个另外的 action。**

**PLAN/IMPLEMENT/COMPOSE/LOCATE/REPAIR 是 observation 的任务模式，不是五个不同的模型。**

### 4.2.1 args 的字段类型与缺省规则

| action | 必填字段 | 可选字段 / 缺省 |
| --- | --- | --- |
| plan | `base_plan_version:int`、`mode:ATOMIC或SPLIT`、`children:list`、`obligation_map:dict[str,list[str]]`、`retained_responsibilities:list[str]`、`dataflow:list` | `ticket_id:null或str`；修订时必须非空 |
| write_code | `contract_version:int`、`source:str`、`declared_dependencies:list` | 无；零依赖显式传空列表 |
| verify | `target_ref:str`、`suite_id:str` | 无；suite_id 必须来自当前可见注册项 |
| consume | `child_id:str`、`artifact_version:int`、`artifact_hash:str`、`alias:str`、`evidence_ids:list[str]` | `accept_unverified:bool=false` |
| reject | `child_id:str`、`artifact_version:int`、`reason_code:str`、`evidence_ids:list[str]` | 无 |
| request_child_repair | `child_id:str`、`artifact_version:int`、`evidence_ids:list[str]`、`instruction:str` | 无 |
| localize | `target_type:str`、`target_id:str`、`evidence_ids:list[str]`、`boundary_kind:str`、`rationale:str` | 无；无证据可用 abstain，不编造 ID |
| repair | `ticket_id:str`、`replacements:list`、`dependency_updates:list` | 无；不需更新依赖时显式空列表 |
| finish | `artifact_version:int` | 无；不接收成功 verdict 字段 |
| abstain | `reason_code:str`、`evidence_ids:list[str]` | `suggested_change:str或null=null`，仅建议、不直接改契约 |

`replacements` 的每项为 `{file, base_hash, source}`。`dependency_updates` 的每项为 `{consumer_node, child_id, artifact_ref, alias}`：artifact_ref 可以是 `{kind: EXISTING, version, hash}`，或 `{kind: CANDIDATE, node_id}`。CANDIDATE 明确选择本次补丁生成/重建的候选模块，实际版本与 hash 由 host 计算后记录，不能要求模型预知尚未生成的 hash。两类更新都只允许 ticket 中的 rebind_edges，且相应候选节点必须属于本次受影响集合。

`children` 的每项为 `{local_key, contract}`；`dataflow` 的每项为 `{from_child, from_output, to_child, to_parameter}`。dataflow 描述父组合应实现的数据关系，不能让 runtime 自动生成调用。

`declared_dependencies` 的每项为 `{child_id, artifact_version, artifact_hash, alias}`，必须与模型已明确选择的 binding 一致。模块实际 import 是否一致由代码检查器另行检查。

`target_type` 首版限定 NODE、HANDOFF、OMISSION、CONTRACT；`boundary_kind` 限定 OWNER_MODULE、CONSUMER_MODULE、PLAN_SUBTREE。`reason_code` 首版采用 CONTRACT_INSUFFICIENT、LOCAL_CHECK_FAILED、INSUFFICIENT_EVIDENCE、UNFULFILLED_OBLIGATION、BUDGET_LIMIT、OTHER，并保留短说明到 event；枚举扩展必须版本化。

### 4.3 不加入的动作

首版没有独立的 `launch_subagent`、`wait_children`、`aggregate_children`、`read_file`、`shell`、`search`、`browser`。它们分别由已提交的计划触发系统调度，或由观测构造器提供必要只读内容。

parent 的业务组合通过 `write_code` 完成，不能由名为 aggregate 的 runtime 函数替它编写。

### 4.4 `plan` 的原子与分解 payload

原子决策：

```json
{"action":"plan","args":{"base_plan_version":0,"mode":"ATOMIC","children":[],"obligation_map":{"O_ROOT":["SELF"]},"retained_responsibilities":["complete_root_task"],"dataflow":[],"ticket_id":null}}
```

分解决策的完整语义如下；实际 JSON 中 `children[*].contract` 必须填入第 3 节的真实字段，不能使用省略占位符。

```text
plan.args = {
  base_plan_version: 0,
  mode: SPLIT,
  children: [
    {local_key: prefix, contract: 前缀和契约},
    {local_key: query, contract: 使用给定前缀和回答区间查询的契约}
  ],
  obligation_map: {
    O_ROOT: [child:prefix, child:query, SELF:parse_and_format]
  },
  retained_responsibilities: [输入解析, 子函数调用, 输出格式化],
  dataflow: [prefix.return -> query.parameter:prefix],
  ticket_id: null
}
```

子节点 ID 由 runtime 根据 parent、plan version、local_key 稳定分配，随后在 observation 中反馈。子节点不能修改自己的 incoming contract；它只能在该契约内部选择分解方式。

### 4.5 `write_code` 必须是可编译的完整当前模块

首版每次初始实现输出一个模块，包含公开函数和下划线前缀私有 helper。代码字符串就是监督目标的一部分。

`declared_dependencies` 引用已消费 artifact；代码中的 import 和业务调用仍必须由模型写出。host 只检查、保存、组装目录和生成固定 I/O 包装器。

已提交模块后不再用 `write_code` 悄悄覆盖；修改走 `localize → repair`。语法错误而未成功存成候选的首次 action 可以在预算内重试，原始错误照常记录。

### 4.6 `consume/reject/request_child_repair` 的边界

`consume` 是选择行为，不是功能正确性证明。允许模型在 UNKNOWN 下显式 `accept_unverified=true` 接受，并将不确定性保留；不得自动给它可信成功标签。

首版采用 audit-only 的语义责任检查：已知 FAIL 的返回可以在 `accept_unverified=true` 时被模型接受，并记录 `known_failed_consumed`，继续观察其组合后果；不自动给它合规标签。所有方法采用同一规则。非法版本、错误 hash、未授权文件访问和源码越权始终拒绝；语义责任审计不等于关闭运行安全。

`reject` 不能抹掉原分配义务。需要 child 修复时，应在关闭接收决定前选择 `request_child_repair`；已经 reject 后，首版通过显式内部计划修订重新分配，或提交一个被审计为不完整的产物，不能把“拒绝了，所以这项责任不存在”作为成功。

`request_child_repair` 保留子契约，将已经返回的 child 从终态重新激活到 REVIEW，并携带当前失败证据；它需要通过 localize/repair 完成修改，再提交新版本。此动作仅用于实现/组合修复。若 child 指出契约本身不足，用 `abstain(reason_code=CONTRACT_INSUFFICIENT, suggested_change=...)` 返回；parent 决定是否修订其内部计划。

### 4.7 首版契约修订：复用 `plan`，不再增 action

初始计划 `base_plan_version=0`；修订必须带当前版本和 `PLAN_SUBTREE` ticket。修订规则：

1. 当前节点的 incoming contract 不可减少或改写；root 外部规格永远不可由模型修改。
2. 允许改变当前节点怎样细分该契约、子契约和保留职责。
3. 首版保守地废弃该节点的旧内部计划与受影响子树，建立新 plan version，重新执行。
4. 旧 artifact 和证据保留在日志，但不再作为当前有效结果；外部无关分支源码不动。
5. 新 child 需要重新验证和显式消费，不能把旧 PASS 自动继承过来。

这条路径允许处理规划/合同下发错误，同时避免首版建设复杂的细粒增量重规划系统。代码替换 ticket 不允许修改契约；需要改契约时必须走本路径。

---

## 5. policy 到底是什么，怎样输入输出

### 5.1 数学和程序接口

共享策略为：

\[
a_t \sim \pi_\theta(a\mid o_t),\quad o_t=\operatorname{build\_observation}(s_t).
\]

```python
from typing import Protocol

class Policy(Protocol):
    def act(self, observation: dict) -> dict:
        """返回 raw_text、parsed_action(可为空)、token_usage、model_identity。"""
        ...
```

`PolicyOutput` 不仅有 parsed action，还必须保留原始文本、生成截断标记、模型/adapter 身份、输入输出 token 数。解析失败不是 exception 后丢掉这个任务，而是一个可记录的 protocol failure。

### 5.2 三种运行实例，不是三个学习模型

| 实例 | 怎么实现 | 用途 |
| --- | --- | --- |
| `ScriptedPolicy` | 根据 fixture 中的 node/phase/step 查表返回 action | 确定性工程测试、首批示范数据 |
| `LLMPolicy(base)` | 公开基础模型，序列化 observation 后生成 JSON | 真实模型 smoke、未训练基线 |
| `LLMPolicy(base + new_adapter)` | 相同类，加载本轮新 SFT adapter | 训练回接、后续实验 |

ScriptedPolicy 能访问的答案只用于 fixture 预先编写动作。其隐藏答案和脚本查表 key 不进入 LLM observation。不能让 runtime 在 LLM 失败时静默切换 ScriptedPolicy。

### 5.3 observation 必需字段

```json
{
  "protocol_version": "at-v3r1",
  "root_id": "fixture_prefix_queries",
  "node_id": "N0",
  "phase": "RESOLVE_CHILD",
  "mode": "COMPOSE",
  "root_spec": "回答整数数组的多个闭区间求和查询",
  "incoming_contract": {"contract_id": "C_N0", "version": 1},
  "current_plan": {"version": 1, "mode": "SPLIT"},
  "own_code": null,
  "returned_child": {
    "node_id": "N0_1",
    "artifact_version": 1,
    "artifact_hash": "h_child_v1",
    "module_path": "nodes.N0_1",
    "public_symbol": "prefix_sums",
    "signature": "prefix_sums(nums: list[int]) -> list[int]",
    "visible_functional_status": "PASS"
  },
  "approved_dependencies": [],
  "visible_evidence": [{"id":"E_LOCAL_1","status":"PASS","covered_obligations":["O_PREFIX_LENGTH","O_PREFIX_ZERO","O_PREFIX_VALUES"]}],
  "pending_obligations": ["resolve_child:N0_1"],
  "repair_ticket": null,
  "last_action_result": null,
  "budget_remaining": {"policy_calls": 51, "tokens": 160000, "patch_attempts": 3},
  "allowed_actions": ["consume", "reject", "request_child_repair", "localize", "abstain"]
}
```

上例只展示关键字段形态。真实 observation 必须嵌入完整 incoming contract、需要的计划接口及当时可见证据，不能让模型只看到 contract ID 后猜内容。

### 5.4 observation 不允许包含的内容

最终隐藏测试输入/输出、参考代码、故障注入类型、正确责任标签、未来 root 结果、尚未生成的 child 返回、teacher 的私有答案、policy 决策之后才出现的验证结果。

训练和推理必须调用同一个 `build_observation` 和同一个 `render_messages`。离线导出时读取 `observation_before`，禁止用最终完整 trace 倒灌未来信息。

### 5.5 mode 如何分配

- 尚未决定当前节点分解方式：`PLAN`。
- 叶节点写当前模块：`IMPLEMENT`。
- parent 处理 child 或写调用组合：`COMPOSE`。
- 可见失败之后提出候选：`LOCATE`。
- 有已授权 ticket 时写补丁：`REPAIR`。
- 请求检查与提交时：`CONTROL`。

mode 是任务描述，由状态和当前阶段确定；`allowed_actions` 只限定可执行协议，不替模型选“正确消费还是拒绝”。不能根据 gold label 在同一状态只保留正确 action。

### 5.6 固定系统提示的最小内容

```text
你是一个共享递归代码策略。当前节点、契约、可见证据和预算由 observation 给出。
每次只返回一条满足当前 action schema 的 JSON，不输出 Markdown 代码围栏。
plan 选择 ATOMIC 或 SPLIT；保留 incoming contract 的所有外部义务。
代码只能实现当前分配职责；使用子模块前必须显式 consume 对应版本。
consume 不等于自动生成调用，你仍需在自己的代码中实现正确参数传递与组合。
不得自行更改 host verdict、隐藏测试、ownership 或版本身份。
失败后依据可见证据提出边界；只有授权范围内的代码和声明元数据可改变。
无法继续时明确 abstain，不伪造成功。
```

提示词用来声明协议，不被当作模型已经学会责任的证据。首版不反复优化角色扮演或长推理提示。

Qwen3-8B 使用 `apply_chat_template(..., enable_thinking=False)` 构造统一前缀 [E1]。不新增特殊 mode token，也不要求训练不可核验的长自然语言思维链。

### 5.7 长上下文处理

优先保留完整当前契约、当前待处理 child、必要依赖接口、当前代码、失败证据和 ticket；移除已归档的无关历史，用确定性字段摘要替代，不用第二个 LLM 做摘要。

在动作尚未生成时，按当前 phase 可能输出动作中的最大配额保留生成空间，而不是先知道 action 再决定窗口。prompt 必须给 action 留出生成预算。超出上限时先按固定规则压缩历史，仍不够则记录 `CONTEXT_OVERFLOW`。不截断 contract、action JSON 尾部或待学习代码后继续当合格样本。

---

## 6. runtime 状态机与递归语义

### 6.1 最小状态转移

| phase | 正常可执行动作 | 后续 |
| --- | --- | --- |
| `NEW` | plan、abstain | ATOMIC→IMPLEMENT；SPLIT→WAIT_CHILD |
| `WAIT_CHILD` | 无 policy 调用；系统按已授权计划调度 | child 终止→RESOLVE_CHILD |
| `RESOLVE_CHILD` | consume、reject、request_child_repair、localize、abstain | 已处理则继续下一 child；全部处理→IMPLEMENT |
| `IMPLEMENT` | write_code、abstain；必要时 localize 请求内部计划修订 | 有候选→READY_VERIFY |
| `READY_VERIFY` | verify、abstain | 检查完成→REVIEW |
| `REVIEW` | finish、localize、verify、abstain；有失效依赖时 consume/reject | 提交、修复或终止 |
| `REPAIR_AUTHORIZED` | repair；PLAN_SUBTREE ticket 下允许 plan；abstain | 接受后重新验证/处理依赖；失败回到 REVIEW |
| `TERMINAL` | 无 | 将 NodeResult 返回上层或结束 root |

表中不是穷举所有 protocol error 后的反馈循环；实现中任何非法动作均不改变业务状态，但计入调用与失败日志，并按统一重试预算决定是否继续。

### 6.2 必须实现的递归顺序

```text
N0: plan SPLIT
  系统按计划执行 N0_1
    N0_1: plan ATOMIC → write_code → verify → finish
  N0: consume / reject / request_child_repair       # 必须再次调用 policy
  系统执行 N0_2
    N0_2: plan SPLIT → 孙节点 ... → 显式消费 → write_code → verify → finish
  N0: consume / reject / request_child_repair
N0: write_code(业务组合) → verify → finish 或 localize/repair
```

`plan(SPLIT)` 已授权顺序执行计划中的 child，因此不再训练“等待线程完成”之类无研究含义的动作。调度自动化不等于组合自动化。

首版 child 之间不直接 import sibling；它们通过输入输出契约描述数据依赖，运行时由 parent 的代码传值。生成时可以读取 sibling 已承诺接口，但不能获得未生成的正确返回。

### 6.3 核心循环伪代码

```python
def solve_node(node_id, session, policy):
    while not session.is_terminal(node_id):
        if session.budget.exhausted():
            return session.terminate(node_id, "BUDGET_EXHAUSTED")
        if session.phase(node_id) == "WAIT_CHILD":
            child_id = session.next_authorized_child(node_id)
            result = solve_node(child_id, session, policy)
            session.return_child(node_id, result)  # 只返回，不消费
            continue
        obs = build_observation(session, node_id)
        output = policy.act(obs)
        session.charge_actual_usage(output)
        event = session.trace.begin(obs, output)
        result = validate_and_dispatch(session, node_id, output)
        session.trace.finish(event, result)
    return session.node_result(node_id)
```

`validate_and_dispatch` 完成真实状态转移；不能捕获所有 exception 后统一返回 success。代码错误、policy 错误和基础设施错误分别编码。

### 6.4 `finish` 不等于成功

节点有候选、且针对当前版本至少执行过一次可见检查，才允许 finish。检查可为 FAIL 或 UNKNOWN：这时仍将候选和真实状态返回给 parent，parent 才有机会学习正确处理失败。

没有代码用 abstain 返回。root 可提交可见失败的最终候选，评价如实失败；不能因为模型说“完成”就提升 verdict。child 的本地正确性也不能替代 root 的组合正确性。

### 6.5 预算与协议错误

所有节点、重试、请求修复、局部重放共用 root budget。达到深度/节点上限时，从 NEW 的可用计划方式中移除 SPLIT，仍可 ATOMIC 或 abstain；记录结构被预算限制的事实。

JSON 解析失败、未知动作、错误字段、版本不匹配均保留 raw text。最多一次结构化错误反馈后重试，不使用另一个模型自动把非法输出修成成功 action。若存在固定的空白/终止 token 清理，训练与推理必须一致且 raw text 不被覆盖。

---
## 7. 代码、artifact、ownership 与 parent consumption 的具体实现

### 7.1 一节点一源码模块

每个候选工作区包含：

```text
workspace/
├── nodes/
│   ├── __init__.py
│   ├── N0.py           # root / parent，拥有输入解析与组合代码
│   ├── N0_1.py         # child 1，只拥有该契约的实现
│   └── N0_2.py         # child 2
└── entry.py            # 固定、受信任的输入输出包装，不写业务算法
```

源码版本存入 artifact store，工作区只是一次 manifest 的物化副本。`nodes/N0_1.py` 可有多个版本，但每个历史版本不可覆盖。

`ownership` 首版按模块/函数识别；公开函数签名固定，私有 helper 归同一节点。模块路径由 runtime 指定，action 不得任意指定绝对路径、`..` 或符号链接。

### 7.2 组装不是替模型完成组合

允许 runtime 做：复制已授权源码到工作区、生成 `__init__.py`、固定 I/O wrapper、设置可见模块路径。

不允许 runtime 做：猜参数、添加 child 调用、将 child 返回自动接到 parent 变量、替 parent 完成业务循环、根据参考解修改代码。

root 默认接口为 `solve(text: str) -> str`。对于 stdin 型任务，wrapper 只执行：读取 stdin → 调用 solve → 写回返回字符串。解析与算法由 root 代码承担。

### 7.3 消费记录分四级

| 级别 | 如何实现 | 能说明什么 |
| --- | --- | --- |
| `selected` | consume 指向正确版本和 hash | 模型显式选择了 artifact |
| `linked` | AST/import 与 bundle manifest 一致 | 程序引用该模块；不证明调用发生 |
| `called` | 受控执行记录到该函数被调用 | 该测试上执行过；不证明返回值影响结果 |
| `dependency_supported` | 已知数据流样例、局部受限 def-use 或合同兼容替换/受控干预支持 | 在说明的检查范围内存在使用证据 |

首版必须实现 selected、linked、called；`dependency_supported` 先用于可判定 fixture 和审查过的训练样例。未知的自然程序不得仅凭一次调用标成“实际结果被消费”。

一种可以确定的违规是：consume 指向某个版本，但代码 import 的路径/manifest 实际绑定另一个版本。另一种需要更强证据的违规是：调用后丢弃结果并重算。二者不要混用同一个检测指标。

### 7.4 版本失效与再验证

每个功能证据绑定完整 fingerprint：

```text
contract_version + source_hash + dependency_bundle_hash
+ suite_hash + runner_version + relevant_execution_config
```

任一项变化使旧证据变为 stale，不删除历史记录。child 更新后，受影响 consumer 的 binding 待重新确认；必要时 parent 再次输出 consume。不能自动把旧 `consume(v1)` 改写成 `consume(v2)`。

重放时区分：**哪些源码可改、哪些绑定元数据需要确认、哪些节点只需重新验证**。依赖失效不等于允许改动所有依赖者源码。

---

## 8. verifier：先保证证据可信，再追求覆盖率

### 8.1 功能结果与责任状态分开

功能 suite 的执行状态：`PASS / FAIL / UNKNOWN / ERROR`。

- PASS：非空、有效 suite 在指定环境及版本上全部通过。
- FAIL：程序错误、断言不满足、资源上限内的超时等可归于候选的失败。
- UNKNOWN：没有适用 suite、必要义务未覆盖或无法作出相应判断。
- ERROR：运行器、数据读取、可信 oracle 等基础设施出错；不能冒充模型失败或通过。

责任检查另存 `SUPPORTED / VIOLATION / UNKNOWN`，同时列出覆盖的 obligation 和证据。功能 suite PASS 不会自动使所有责任义务 SUPPORTED。

### 8.2 首版三种 suite

| suite | 输入/执行 | 来源与用途 |
| --- | --- | --- |
| fixture root suite | 调用 `solve(text)`，比较确定性结果 | 新建的小规模、人工审查测试 |
| fixture/local suite | 对公开子函数传 JSON 可表达参数，比较返回或性质 | 独立朴素参考、边界用例、已审查断言 |
| APPS root suite | 通过统一 wrapper 运行题目输入输出 | 数据导入器适配；保留原始测试与比较规则 |

本版不假设任意 LLM 生成的 child contract 都能自动得到可信测试。初期只为有限责任家族注册 local suite；新契约不能匹配时明确 UNKNOWN，仍可继续框架流程。

### 8.3 测试可信度

使用宿主执行，不等于测试 oracle 正确。测试期望结果需要独立依据，例如人工确认、朴素枚举或已审查算法。模型自写 assertion 可作为候选测试，不能自动成为高置信责任标签。

注册器核对签名、端点语义、输入前置条件、契约版本和 suite 适用范围。签名碰巧相同不够；输入域不同的 suite 不得绑定。

### 8.4 visible 与 private 的隔离

- policy、localizer、repair selector 只收到 visible 结果。
- TRAIN 中私有检查可用于离线筛选标签，但不写回当时 observation。
- VALIDATION 可用于事前规定的 checkpoint 选择，不能混入训练。
- SEALED 的私有检查只评价已选定的输出；不得触发修复、挑补丁或改配置。

若 visible 已通过而 private 失败，记录“当前在线系统未发现失败”，不能事后用 private 反例启动同一条自动修复轨迹。

### 8.5 最小执行隔离

模型代码不得在训练主进程或携带服务凭据的进程里直接执行。`sandbox.py` 使用独立受限进程/容器，网络关闭，源码只读，工作目录临时可写，限制进程数、时间、内存与输出。

预期答案、私有测试文件、训练数据目录、GPU 与宿主凭据不挂载给候选代码。可信宿主传入测试输入，在外部比较结果；记录异常并清理整个进程组。

AST 黑名单只是额外检查，不能被称为安全沙箱。隔离方案未经验证时只允许运行人工审查 fixture，不执行自由生成代码。此项是实际安全门禁，不是可略过的论文装饰。

---

## 9. 第三部分的最小实现：localize → ticket → repair → replay

### 9.1 责任候选索引

首版不建复杂图：使用三个映射和少量边即可。

```text
module/function -> owner_node
artifact_version -> contract/version + consumer_edges
evidence -> target artifact / failed obligation
parent_of、issued_contract、consumed_by
```

从 traceback 的模块、失败义务或版本/消费异常获得起点，加入当前 owner、直接 parent 和相关消费边，最多保留 3 个候选。没有直接定位信息时允许提出当前节点/父边界或弃权，不能根据隐藏答案选候选。

### 9.2 localize 输出例子

```json
{"action":"localize","args":{"target_type":"HANDOFF","target_id":"N0<-N0_1","evidence_ids":["E_ROOT_FAIL","E_CHILD_PASS","E_BINDING_UNUSED"],"boundary_kind":"CONSUMER_MODULE","rationale":"child 局部检查通过；parent 没有使用声明的前缀和依赖"}}
```

模型给的是候选与理由，不是已证实根因。runtime 验证 evidence ID 确实可见且存在，然后生成 ticket。

### 9.3 ticket 内容

```json
{
  "ticket_id": "T1",
  "snapshot_id": "S_failure_1",
  "target": {"type":"HANDOFF","id":"N0<-N0_1"},
  "editable_files": ["nodes/N0.py"],
  "editable_plan_nodes": [],
  "rebind_edges": ["N0<-N0_1"],
  "revalidate_nodes": ["N0"],
  "frozen_hashes": {"nodes/N0_1.py":"h_child_v1","nodes/N0_2.py":"h_query_v1"},
  "remaining_attempts": 1
}
```

示例 hash 不是实际产物。ticket 中的 revalidate 集合不是 editable 集合；重新运行某模块不授予修改它的权限。

### 9.4 两种边界足以启动

**代码边界：** 一节点模块及其私有 helper。若目标是消费/组合问题，优先限定 consumer 的模块，冻结正确 child。允许必要绑定更新，但必须有显式 consume 或 repair 中声明、host 记录的 binding 更新；后者仍是模型决定，不是自动接线。

**计划子树边界：** 当前节点内部计划、该节点组合代码及重建子树。incoming contract 和外部无关分支不变。使用 `plan` 修订，不允许 `repair` 顺手删义务。

这里的 repair payload 中 `dependency_updates` 是显式消费更新决策，解析到确切版本后生成同类 Consumption 记录；普通新建依赖仍使用 consume。首版只在已授权的 rebind_edges 上接受这种原子更新，避免补丁源码与 binding 暂时不一致。

### 9.5 repair 的事务步骤

```text
冻结失败快照
→ 验证 ticket、base hash 和预算
→ 将 replacements 写入隔离的候选副本
→ 检查实际改动是否在 editable_files 内
→ 检查 frozen_hashes 完全保持
→ 应用被授权的 dependency_updates
→ 重新绑定/使过期证据失效
→ 执行受影响 local suite 和全部固定 visible root suite
→ 按冻结规则接受候选，或回滚到原快照
→ 原失败、补丁、重放与回滚结果全部记录
```

首版接受规则为：没有越界或版本错误、修复目标的固定可见检查通过、原已通过的可见检查不回归；若完整 root 程序已形成，还要求其固定可见检查全部通过。child 构造阶段尚无完整 root 时，只能标记 local repair 通过，不能宣称 root 修复成功。没有有效检查时不能宣称重放成功；可保存补丁，但自动选择不将 UNKNOWN 当 PASS。

自动重放产生的证据同样绑定新版本。接受补丁后回到 REVIEW，可直接 finish，或在证据失效时再 verify；不为了状态接线重复执行同一 suite 并漏记成本。回滚后恢复原 REVIEW 状态与失败证据。

如果所有候选失败，最多扩大到一次父边界，并仍受总计 3 次补丁预算限制；不是 3 个候选各自重新获得 3 次预算。

### 9.6 checkpoint 的最小含义

首版 checkpoint 是完整源码/契约/binding manifest、可见 suite 与环境配置的快照，不是 Python 进程或 GPU KV cache 快照。

代码修复只恢复失败程序后重新执行；计划修订恢复到相关计划决策状态并顺序重跑子树。不宣称完整物理过程级重放。先跑一次无修改重放确认失败可复现；不稳定任务单独标记。

### 9.7 必须保留的结论边界

单个补丁通过只说明“在这些检查下，该边界容纳了一个有效干预”，不等于唯一历史原因，也不等于全局最小边界。自动模型补丁、撤销人工故障、参考补丁辅助必须分开报告。

---

## 10. 数据分区：从本次新采集开始建立

| split | 用途 | 能否进入正式主结果 |
| --- | --- | --- |
| `SYN_FIXTURE` | 新建确定性样例、单测、trainer smoke | 否 |
| `DEV_ENGINEERING` | 实际 LLM 接线和错误调试 | 否 |
| `DEV_MECHANISM` | P1–P3 的有限机制选择 | 开发性证据，不冒充独立测试 |
| `TRAIN` | 示范、候选、故障变换、SFT/可选偏好 | 否 |
| `VALIDATION` | 模型选择和一次有限修订 | 否 |
| `MECHANISM_HOLDOUT` | 未见责任配置/故障类别与共同失败集 | 是 |
| `APPS_SEALED` | 主域最终冻结批次 | 是 |
| `EXTERNAL_SEALED` | 一个外部域；A1 不实现适配 | 是 |

划分以 root/family 为单位；同题候选、子任务、重构、故障注入、修复和反向推导出的局部样本全部跟随原 root。不能先切 action 行，再做 train/test split。

APPS 导入时保留原始 split、题目 ID、来源 hash、输入输出类型；优先从官方训练范围划分本项目开发/训练/验证。正式测试范围保持隔离，并记录已知污染风险，不声称公开基准对基础模型绝对未见。

root 隔离和用例隔离是两件事：同一个训练 root 的未展示用例仍是训练任务资源，不构成独立测试任务。

---

## 11. A1：按 F1–F9 新建完整框架

每个 F 阶段结束后写一个小报告：新增文件、实际执行命令、预期与观测、工程 bug、下一阶段。报告标记 `SMOKE_ONLY`。不设置 Pass@1、有效委托率或训练增益门槛。

### F1：schema、脚本 policy、原子状态机

**新建**：`schema.py`、`policy.py` 的接口和 ScriptedPolicy、`runtime.py` 原子路径、`trace.py` 基本 append、`artifacts.py` 初始版本保存。

**具体实现**：10 个 action schema；Contract/Node/Artifact/Event 类型；NEW→IMPLEMENT→READY_VERIFY 的转移；全局预算；非法 JSON、错误 action、版本错误的记录；原子模块保存。

**输入/输出**：人工编写的“整数求和”fixture 和 2–3 个 action → 保存的源码、状态事件和 READY_VERIFY 状态。

**测试**：plan ATOMIC 不创建 child；write_code 只写当前模块；非法 action 不修改状态；达到预算结束。F1 尚未实现 verifier 时停止在 READY_VERIFY，不能用伪 PASS 跑通 finish。

**建议入口**：`python -m audittrainer.cli smoke --stage f1`。

### F2：真实代码执行、根/局部 verifier 与原子完成

**新建**：`sandbox.py`、`verifier.py`；补 `finish/abstain` 与证据 fingerprint。

**具体实现**：受限执行、超时、输出限制；callable 和 solve(text) 适配；期望结果留在宿主；非空 suite；PASS/FAIL/UNKNOWN/ERROR；当前候选检查后提交。

**输入/输出**：正确、错误、死循环、没有适用 oracle 的模块 → 真实 Evidence 和 NodeResult。

**测试**：正确 sum 通过、故意减法失败、死循环超时、空 suite 不算 PASS、runner 出错为 ERROR、finish 不篡改 verdict。

**完成标准**：ScriptedPolicy 的 `plan→write_code→verify→finish` 在真实执行器中闭合，而不是 mock verifier。

**建议入口**：`python -m audittrainer.cli smoke --stage f2`。

### F3：SPLIT、递归栈、artifact 与显式消费

**新增/扩展**：`runtime.py` 子栈、`artifacts.py` bundle/binding、observation 构造、consume/reject/request_child_repair。

**具体实现**：plan 注册子契约和 ID；顺序深度优先；child 返回后 parent 再次决策；将 source artifact 放到固定模块路径；生成固定 wrapper；检查版本、import、调用记录。

**测试**：两 child 组合；至少一个孙节点；错误 child 返回后请求修复；拒绝不取消责任；旧版本消费被拒；仅 consume 但无调用不计实际使用。

**完成标准**：脚本产生 `child真实验证→parent显式消费→parent自行生成组合→root验证`；trace 能看到每个模型/脚本决策。

**建议入口**：`python -m audittrainer.cli smoke --stage f3`。

### F4：责任候选、ticket、补丁、回滚和内部计划修订

**新建**：`repair.py`；扩充 localize/repair/plan revision 和证据失效。

**具体实现**：三个责任索引；单模块与计划子树两种 ticket；修复前快照；代码替换；冻结 hash；绑定更新；无修改重放；失败回滚；总修复预算。

**测试**：只修 parent 恢复结果；错误定位修复失败；试图改 frozen child 被拒；无依据扩大 ticket 被拒；计划修订不允许修改 incoming contract；child 新版本导致 consumer 证据 stale。

**完成标准**：一条确定性失败链可以被脚本在合法边界内修复，越界补丁确实不能进入最终工作区；并保留正确弃权路径。

**建议入口**：`python -m audittrainer.cli smoke --stage f4`。

### F5：接入真实 LLMPolicy

**新增/扩展**：`LLMPolicy`、`prompts.py`、模型/adapter identity、上下文裁剪和 token 计数。

**具体实现**：公开模型加载；统一 no-thinking chat prefix；一次一条 action；仅解析规定 JSON；错误反馈一次；保存输入与原始输出；同一模型服务全部递归节点。

**测试**：2–5 个短 fixture 或 DEV_ENGINEERING roots。至少一次生成确实来自模型；读取的不是脚本答案或缓存 chosen。

**完成标准**：成功和失败都能走到可保存的终态；不要求真实模型立即完成递归或正确修复。若它在初始步骤就失败，另外运行预注册局部 observation 的模型调用以检查各接口，标记为 state-level smoke，不能把这些拼成自然完整成功轨迹。

**建议入口**：`python -m audittrainer.cli rollout --policy base --split DEV_ENGINEERING --limit 5`。

### F6：完整 trace 与 SFT 数据导出

**新增/扩展**：`trace.py`、`dataset.py`、root manifest、事件级监督证据。

**具体实现**：决策前 observation 固化；未来 evidence 单独用于筛选标签；raw traces 全保留；导出一步 action 样本；错误代码不作为正确 target；mode、split、来源、版本随样本保存。

**测试**：同一失败 root 中“正确 child 实现”可在局部证据充分时作为局部正例，但错误 parent 不进入合规正例；对未来 token/私有字段做泄漏测试；同 root 派生全部同 split。

**完成标准**：新建 fixture 可导出约 32 条演示样本，覆盖 PLAN、IMPLEMENT、COMPOSE、CONTROL。数量只检查流水线，不是数据质量指标。

**建议入口**：`python -m audittrainer.cli export-sft --trace-dir runs/fixture_suite --out data/sft/smoke.jsonl`。

### F7：tokenizer、mask、collator 与真实 SFT

**新建**：`tokenize_sft.py`、`train_sft.py`、smoke 配置。

**具体实现**：第 14–15 节的一步 SFT；无 prompt loss；不截断 target；真实 QLoRA 参数更新；保存 adapter/tokenizer/model revision；mask 单测和权重变化检查。

**测试**：32 条左右短样本、5–20 个 optimizer steps；finite loss、非零梯度、至少一个 trainable tensor 确实改变、保存文件存在且可重新加载。

**完成标准**：参数更新是真的。短程 loss 不下降或模型效果不好，不阻塞 F8；labels 全 -100、无梯度、NaN、checkpoint 身份错误必须修复。

**建议入口**：`python -m audittrainer.train_sft --config configs/sft_smoke.yaml`。

### F8：checkpoint 回接并重新跑同一 runtime

**扩展**：LLMPolicy adapter 加载和身份日志。

**具体实现**：卸载训练实例，重新加载同一公开 base revision + 新 adapter；使用同一 tokenizer/template、action schema 和 evaluator。

**测试**：固定 observation 的 logits 或目标 log-prob 在 adapter 开关之间有可检测差异，并确认实际使用新 adapter；再次生成 action 并运行 2–5 个开发任务。文本不变不自动等于没有加载，要看权重/forward 检查。

**完成标准**：不是只保存文件，而是新权重真实参加了后续推理。仍不要求短程训练有能力提升。

**建议入口**：`python -m audittrainer.cli rollout --policy sft --adapter runs/sft_smoke/final --split DEV_ENGINEERING --limit 5`。

### F9：统一 evaluator 与 APPS 接口

**新建/扩展**：`evaluate.py`、APPS loader、root/test manifest、首份根级汇总。

**具体实现**：第一候选快照、预算内最终候选、hidden 只评分；对每 root 保留失败原因和成本；从 APPS 标准输入型记录开始，不支持的类型显式列为适配范围外。

APPS 官方代码通过 `fn_name` 区分标准输入与 call-based 任务 [E6]。首版只支持其中一种是范围选择，不得把子集结果写成整个 APPS 的结果。后续是否增加另一类型在 A3 冻结。

**测试**：5 个固定 DEV_ENGINEERING 任务 + 全部 fixture。脚本、base 和 SFT 报告字段相同；private 失败不得反馈给在线修复；0 成功成本记 N/A；异常不丢分母。

**完成标准**：可以从新数据导入、完整 rollout、监督训练一直运行到统一报告。正式数据量、基线复现和论文效果不属于 F9 门槛。

**建议入口**：`python -m audittrainer.cli evaluate --config configs/evaluation.yaml --split DEV_ENGINEERING`。

### 11.1 12 个首版 fixture

| ID | 场景 | 预期工程行为 |
| --- | --- | --- |
| FX01 | 原子整数求和 | 完整原子链 |
| FX02 | 前缀和 + 区间查询 | 两 child、显式消费、业务组合 |
| FX03 | 区间调度的嵌套分解 | 真正运行到孙节点，不只是平铺多个函数 |
| FX04 | 缺少一个父义务的计划 | 标记结构遗漏，不伪造语义覆盖 |
| FX05 | 子接口输入不匹配 | 检查失败或证据不足，不绑定错误 oracle |
| FX06 | 错误 child 返回 | parent 请求修复或拒绝，不能改 verdict |
| FX07 | child 正确但 parent 绕过 | functional 与 scope 分别记录 |
| FX08 | stale artifact | 版本检查和旧证据失效 |
| FX09 | parent 单模块补丁 | child 源码 hash 保持、可见重放真实执行 |
| FX10 | 越权补丁 | 事务拒绝并回滚 |
| FX11 | 不可判定契约 | UNKNOWN 与 abstain 都能结束 |
| FX12 | 契约下发错误 | 通过内部 plan 修订重建子树，外部义务不变 |

一个 fixture 可以有正常与异常两条预先编写轨迹。fixture 是工程样例，不是论文责任定位 gold benchmark。

### 11.2 A1 整体通过条件

F1–F9 全部产生实际产物；脚本闭环、真实模型调用、真实 SFT 更新、adapter 回接、failure/unknown/cost 分母齐全；没有占位 verifier、自动消费或私有反馈泄漏。

**不需要** DPO、RL、专用 Locator、复杂因果图、多域适配或某个通过率阈值。

---
## 12. SFT 到底需要什么数据

### 12.1 不是“题目—最终答案”，而是“状态—下一条动作”

最小训练样本：

```text
输入：某节点作出决定之前，在线 policy 实际能够看到的 observation。
输出：该状态下一个经证据支持的完整 action JSON。
```

代码生成样本的 target 是 `write_code` action，代码位于 `args.source`。分解样本的 target 是 `plan` action，包含实际子契约。组合样本既包括 consume 等选择，也包括 parent 自己的 write_code。

一条成功轨迹可以产生多个样本，但不能因为 root 成功，就把其中所有动作都判为责任正确。

### 12.2 四类必备数据与一类后续数据

| 数据类别 | 输入 | 正确 target | 最低标签依据 |
| --- | --- | --- | --- |
| `D_PLAN` | 父契约、当前计划状态、预算 | plan ATOMIC/SPLIT、子契约和义务分配 | 结构无明确遗漏、接口审查、局部/组合执行支持 |
| `D_IMPL` | 固定局部契约、输入规格、只读依赖接口 | write_code | 局部功能证据 + 对声称责任范围的相应证据 |
| `D_COMPOSE` | child card、真实状态、父契约、已有绑定 | consume/reject/request_child_repair、parent write_code | 接受/拒绝理由和版本证据；组合代码另需执行/依赖检查 |
| `D_CONTROL` | 当前候选、检查状态和预算 | verify、finish、适当 abstain | 协议语义正确；不以自然语言自报成功为真值 |
| `D_REPAIR` | 可见失败、候选和 ticket | localize/repair/plan revision | 来源标签或重放结果按类别区分；A7/A8 再决定是否训练 |

F7 先训练前四类。D_REPAIR 可以已经导出供框架测试，但核心生成期 SFT 不必立即学习它；第三部分先用冻结基础策略执行。

### 12.3 首批示范怎么从零得到

**步骤一：新建 12 个 fixture 的契约、模块参考和可见 suite。** 这些内容只用于 SYN_FIXTURE；每项 oracle 人工确认或由独立朴素算法支持。

**步骤二：编写 ScriptedPolicy 的动作表。** 动作真的经由同一个 runtime 执行，包含节点返回、验证证据、消费与提交。不要直接手写一段“成功日志”跳过执行。

**步骤三：导出当时的 observation 和真实目标 action。** 约 32 个短动作即可检查 trainer，示例分配为 PLAN 8、IMPLEMENT 12、COMPOSE 8、CONTROL 4。该分配是 smoke 输入，不要求所有正式数据按此数量。

**步骤四：训练一次 smoke adapter 并回接。** 这个 adapter 的作用是证明训练管道，不自动作为正式主模型起点。正式各组重新从同一个公开 theta0 出发。

### 12.4 正式 TRAIN 数据怎么产生

框架完成后，先选约 50–100 个新划分的 TRAIN roots 做 pilot。来源有两条：

**可审查示范路径。** 从 TRAIN 参考解或可信手写实现提出责任切分，生成契约和模块，重新验证局部与根功能，补充人工核验责任边界。参考代码只能给离线构造器，不进入 policy 观测。

**候选执行路径。** 对同一父状态提出 ATOMIC 和少量不同 SPLIT 候选，固定执行器与总预算实现，保留合法且有证据支持的多个结构。teacher 可用公开基础模型生成候选，不要求先接入某个外部付费模型；teacher 只是提议者，不是 oracle。

每 root 开始只采 2 条完整候选；规划比较每状态 2–3 个候选足够检查机制。所有失败保留 raw trace，只选择哪些动作进入哪类监督，不能修改采集分母。

### 12.5 怎样使用“功能正确但责任违规”的负例

负例至少包括：遗漏一个后置条件、parent 重做 child 算法、调用后不用返回值、错误版本、静默修改契约、只声明 consume 不在代码使用。

**不能把负例代码当 SFT 正确输出。** 有三种合法使用方式：

1. 同一 observation 下，验证过的合规实现作为 SFT target，违规实现保存为对照元数据。
2. 把已经发生违规的状态作为新输入，监督正确的边界内纠正动作；需要真实重放得到新轨迹。
3. A6 有合法 pair 时用于 DPO，chosen/rejected 共享完全相同的决策前信息。

简单在数据中写 `label=bad`，再把 bad code 的 token 也计算正向生成 loss，不会训练模型避免 bad code。

### 12.6 同一代码、不同责任配置

构造两份合法契约：A 允许当前节点完成前缀和与查询，B 已将前缀和委托，当前节点只允许查询。两份目标代码应适应各自契约。

这种样本训练条件化行为，而不是把“代码中出现前缀和算法”一律判为违规。不同契约不是同一 DPO 上下文；应分别构造监督或各自的同契约 pair。

### 12.7 失败轨迹的局部正例

若 child 的完整局部义务和边界有独立支持，而 root 因组合失误失败，可保留 child 的 D_IMPL。不能因为终局失败把所有节点统一标负。

相反，某次 root 通过但 child 的遗漏被 parent 补做掩盖，不能把整条轨迹全部收入责任正例。PLAN 的语义覆盖不足时，仅作为结构/协议样本或弱标签保存，不假称优质分解。

### 12.8 标签强度要能区分

```text
supervision_scope = PROTOCOL_ONLY / FUNCTION_ONLY / RESPONSIBILITY_SUPPORTED
label_origin = scripted_fixture / reviewed_reference / independent_oracle
             / controlled_transform / verified_model_candidate
```

FUNCTION_ONLY 可以进入普通功能对照数据，不自动进入声称“完整且不越界”的高置信目标。UNKNOWN 不变成合规标签；可以用来监督诚实报告或恰当 abstain。

---

## 13. SFT 文件格式、事件切片与数据检查

### 13.1 raw trace 与训练文件分开

```text
raw_traces/*.jsonl        全部 action、失败、证据、原始输出和后状态
sft/examples.jsonl       被选中的 observation/action 监督
sft/rejected.jsonl       未纳入原因，不删除原轨迹
sft/tokenized.jsonl      input_ids / attention_mask / labels
sft/manifest.json       root 分区、模型/template、数据版本和样本计数
```

### 13.2 一条可落地的训练记录

下面展示记录结构；真正生产时 messages 由完整 observation 和固定 system prompt 生成。

```json
{
  "sample_id": "fx02_n1_write_1",
  "root_id": "fixture_prefix_queries",
  "family_id": "fixture_prefix_queries",
  "split": "SYN_FIXTURE",
  "bucket": "D_IMPL",
  "messages": [
    {"role":"system","content":"你是共享递归代码策略。只返回一条 action JSON，完整履行当前契约，不修改其他节点。"},
    {"role":"user","content":"当前节点 N0_1，phase=IMPLEMENT。契约版本1：实现 prefix_sums(nums: list[int]) -> list[int]；返回长 n+1 的列表，第0项为0，第i项为nums[:i]之和；只负责前缀和，不回答查询、不修改输入、不使用外部I/O。当前无代码、无依赖。允许 write_code 或 abstain。剩余调用10。"}
  ],
  "target_action": {
    "action": "write_code",
    "args": {
      "contract_version": 1,
      "source": "def prefix_sums(nums):\n    result = [0]\n    for value in nums:\n        result.append(result[-1] + value)\n    return result\n",
      "declared_dependencies": []
    }
  },
  "evidence_refs": ["E_prefix_test_1", "E_scope_review_1"],
  "supervision_scope": "RESPONSIBILITY_SUPPORTED",
  "label_origin": "scripted_fixture"
}
```

这条记录用于说明格式，不是已经产生的研究数据。正式记录采用统一 render 函数，而不是为每条数据另写自然语言提示。

### 13.3 exporter 的固定步骤

```text
读取 event.observation_before
→ 检查 root/family/split 与来源
→ 根据该 action 的独立证据决定可监督范围
→ 对 chosen action 做 schema 校验
→ 使用当时冻结的 renderer 得到 messages
→ target = canonical JSON(action)
→ 私有标签与未来证据只放 meta，不进入 messages/target
→ 写出样本与 rejection reason
```

错误模型输出不能只修一下 JSON 后继承原来的成功标签；若想采用修正 target，需要在同一前置状态真实执行修正动作并验证，形成新的可追溯 event。

### 13.4 不把整条历史助手消息全部计算 loss

输入可以包含必要历史的只读摘要，但一次训练只预测下一个 action。前面的 assistant 内容也是输入，不参与当前样本目标 loss。

这种一步模仿训练本身不保证长程泛化；A5 后必须重新运行模型的自由递归 rollout，而不是只报告离线 next-action accuracy。

### 13.5 必做数据单测

root/family 不跨分区；private/ref/fault label 不在 messages；target 可 JSON 解析；目标 action 允许执行；证据与对应版本相符；所有 target token 完整；被拒样本有理由；同事件不能被不明重复导出改变训练权重。

加入一个“未来 canary”测试：只在后状态/私有 meta 中放唯一字符串，导出后的 prompt 不得出现它。该测试是防接线泄漏，不是基础模型预训练污染检测。

---

## 14. SFT 的 token、mask 和 loss：必须明确实现

### 14.1 为什么本版不用自动 assistant mask 作为默认路径

为了固定一步动作训练，首版直接预先构造 `input_ids/labels`，然后交给 Transformers Trainer。这样不依赖不同 chat template 是否支持 assistant span mask，也不把 prompt-completion API 版本差异变成主要开发问题。

这仍是标准监督微调；不是新训练算法。Transformers 支持自定义 data collator 与 loss [E2]。后续如切换 TRL，可使用 prompt-completion 的 completion-only 配置，但不在首版同时维护两条 SFT 实现 [E5]。

### 14.2 编码函数核心实现

```python
import json

def encode_action_example(record: dict, tokenizer, max_length: int) -> dict:
    """单步动作监督；不截断目标。超长样本由调用者记录并排除/重新构造。"""
    prefix_ids = tokenizer.apply_chat_template(
        record["messages"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    target_text = json.dumps(
        record["target_action"], ensure_ascii=False, separators=(",", ":")
    )
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer 必须具有明确的结束 token")
    target_ids = target_ids + [eos_id]
    if not target_ids or len(prefix_ids) + len(target_ids) > max_length:
        raise ValueError("EMPTY_OR_OVERLONG_TARGET：不得静默截断")
    input_ids = list(prefix_ids) + target_ids
    labels = [-100] * len(prefix_ids) + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
```

模板前缀可能包含模型规定的空 thinking 标记；以锁定 tokenizer 的实际输出为准。只把追加的 action JSON 与结束 token 作为 target，不手工重复拼接 assistant header 或 `<think>`。

训练与推理必须使用完全相同的 prefix token 序列。不要分别随意渲染，再靠查找某个字符串猜 assistant 边界。[E1][E4]

### 14.3 collator 核心实现

```python
import torch

class ActionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]) -> dict:
        if not examples:
            raise ValueError("Empty batch")
        width = max(len(x["input_ids"]) for x in examples)
        ids, masks, labels = [], [], []
        for x in examples:
            pad = width - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.pad_token_id] * pad)
            masks.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
```

padding 按真实长度追加，不能简单把所有 `token_id == pad_id` 的位置设为 -100：某些配置下 pad 与 EOS 可以相同，这会误删应学习的结束 token。

### 14.4 样本归一化目标

为了避免长代码样本在 loss 中压过短 plan/consume，本版采用每样本 target token 平均，再在 batch 中平均：

\[
\ell_i=-\frac{1}{|A_i|}\sum_{t\in A_i}\log p_\theta(y_{it}\mid x_i,y_{i,<t}),
\qquad L=\frac{1}{B}\sum_{i=1}^B\ell_i.
\]

这里 `A_i` 是完整 action 的 token 集合，包含结束 token，不包含 prompt。任务类型通过采样平衡；不声称仅使用默认全 token 平均 loss 就自动实现了同样归一化。

实现为一个很小的 Trainer 子类，核心代码如下：

```python
import torch
import torch.nn.functional as F
from transformers import Trainer

class ActionSFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 本 loss 自己完成每样本归一化，不使用 Trainer 的总有效 token 缩放。
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        labels = inputs["labels"]
        outputs = model(**model_inputs)
        shifted_logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        valid = shifted_labels.ne(-100)
        counts = valid.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("存在没有目标 token 的训练样本")
        # 只对目标位置计算 CE；不需要为全部 prompt logits 创建 float32 副本。
        token_losses = torch.zeros_like(shifted_labels, dtype=torch.float32)
        token_losses[valid] = F.cross_entropy(
            shifted_logits[valid].float(), shifted_labels[valid], reduction="none"
        )
        loss = (token_losses.sum(dim=1) / counts).mean()
        return (loss, outputs) if return_outputs else loss
```

这是待集成的核心片段，不是宣称训练已执行。F7 要测试梯度累积下的尺度、有效 target 数和 EOS 学习；锁定实际依赖版本后再冻结实现。使用不同 batch/累积配置时，不假设所有 Trainer 版本的缩放逻辑完全一致。

### 14.5 采样与课程

正式起步配置采用一个训练 run：前 20% 更新主要学习固定契约实现（D_IMPL 80%、D_CONTROL 20%）；其余更新混合 D_PLAN 30%、D_IMPL 40%、D_COMPOSE 25%、D_CONTROL 5%。

比例是初始设计，不是最优结论。D_COMPOSE 中同时保留接收决策和 parent 组合代码，不能全被短 consume 动作占满。各 bucket 先按 root 均匀取样，再取该 root 的动作，限制大任务因节点多占据全部样本。

一个样本的 loss 归一化与 mode/root 采样一起定义实际训练目标。A6 最多允许一次有依据的采样调整，不做大规模权重搜索。

---

## 15. QLoRA SFT：模型加载、参数与真实训练验收

### 15.1 加载顺序

```text
锁定公开 model revision 与 tokenizer
→ 加载 4-bit base
→ prepare_model_for_kbit_training
→ 通过 get_peft_model 添加一个新的 LoRA adapter
→ 关闭训练期 use_cache，启用 gradient checkpointing
→ 训练自定义 action loss
→ 保存 adapter、tokenizer、配置和 optimizer/trainer 状态
```

PEFT 官方量化训练文档提供 `prepare_model_for_kbit_training` 与量化底座上的 adapter 训练方式 [E3]。首版只训练 adapter，不将 4-bit 权重解冻当普通全参数训练。

### 15.2 默认参数

```yaml
model:
  base: Qwen/Qwen3-8B
  revision: REQUIRED_RESOLVED_COMMIT
  load_in_4bit: true
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16
  device_map: single_visible_gpu
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear
  bias: none
  task_type: CAUSAL_LM
training:
  max_length: 8192
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 0.0001
  weight_decay: 0.0
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  max_grad_norm: 1.0
  gradient_checkpointing: true
  packing: false
  loss_normalization: per_example_then_batch
  seed: 42
  report_to: none
  epoch_limit: 1
```

这些超参数是本版建议的单一启动配置，不是来自项目旧训练，也不是官方保证最优。正式更新数由实际采样清单与有效 batch 决定；不通过无限重复少数 fixture 凑训练步数。

若 GPU 不支持 bf16，A0 明确选择 fp16 并检查 scaler/数值；传给 TrainingArguments 的 bf16/fp16 必须与实际计算精度相符，不能同时启用。所有对照保持一致。显存不足先减序列长度或 microbatch，重新固定全部组；不承诺某显卡一定能容纳给定长度。

单卡训练使用显式当前设备，避免把推理用 `device_map="auto"` 当作分布式训练方案。[E3]

### 15.3 模型构造代码片段

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def build_trainable_model(model_id: str, revision: str):
    if not torch.cuda.is_available():
        raise RuntimeError("该 QLoRA 路径需要已配置的兼容 GPU")
    if not revision or revision == "REQUIRED_RESOLVED_COMMIT":
        raise ValueError("必须记录并传入实际模型 revision")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quant,
        dtype=dtype,
        device_map={"": torch.cuda.current_device()},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    ))
    return model
```

API 依据 [E2][E3]；安装后用固定依赖版本完成 smoke，再保存 lock。不要边开发边无记录地升级 Transformers/PEFT。此片段只负责模型构造，完整 train_sft.py 还须实现配置读取、数据加载、Trainer、保存和验收。

### 15.4 train_sft.py 应当按顺序做什么

1. 读取 config 与 dataset manifest，拒绝正式 TRAIN 混入 SEALED，smoke 只允许 SYN_FIXTURE/开发演示。
2. 固定 seed，加载 tokenizer，核对 template hash 和模型 revision。
3. 调用 encode_action_example；记录超长/非法样本及原因，不默默截断。
4. 按课程与 root/bucket 生成确定性采样清单，保存其 hash。
5. 构建 QLoRA 模型、collator、ActionSFTTrainer；打印并保存 trainable parameter 列表与数量。
6. 在训练前复制少量 trainable tensor 到 CPU，记录基线 checksum/范数。
7. 执行 Trainer.train；记录 optimizer_step、loss、grad_norm、target tokens、学习率。
8. 保存 final adapter 与 tokenizer、训练配置、数据 hash、trainer state；需要断点恢复时同时保存 optimizer/scheduler 状态。
9. 对比 trainable tensor：至少一个存在非零变化；检查训练 base 未被错误解冻。
10. 用重新加载的 adapter 运行固定前缀 forward，和保存前输出在允许数值误差内一致；然后交给 F8 完整推理。

只看 loss 曲线或保存目录不足以证明训练成功。LoRA 某些矩阵在第一个更新时梯度可不同，验收要求不是每个矩阵每一步都变化，而是整体存在真实可追溯更新。

### 15.5 smoke 配置

```yaml
extends: sft_main.yaml
training:
  max_length: 4096
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  max_steps: 10
  learning_rate: 0.0001
  logging_steps: 1
  save_strategy: final_only
  evaluation_strategy: no_effect_gate
  output_dir: runs/sft_smoke
```

`extends`、`final_only`、`no_effect_gate` 是本项目配置语义，配置读取器应翻译为实际 Trainer 参数，而不是直接把所有 YAML 字段传入 TrainingArguments。正式运行不用字面上的 `no_effect_gate` 作为库参数。

smoke 可以在小数据上过拟合检查动作输出，但不能把该训练集上的 improvement 当泛化证据。正式模型从统一公开底座重新训练。

### 15.6 checkpoint 的用途分开

- `sft_smoke`：验证更新与加载，不能当论文方法 checkpoint。
- `sft_F00/F10/F01/F11`：正式核心对照，每组独立从同一 theta0 开始。
- `E_ref`：冻结执行器，比较分解时共用。
- `R_ref`：冻结定位/修复模型，检验生成期责任结构的修复价值。

E_ref/R_ref 可以是同一公开 base 的只读实例，或事前冻结的本轮训练快照；必须明确来源，不能随着被评策略变化而暗中更换。

---
## 16. A2：框架完成后，做三个小规模机制比较

三项实验只比较影响后续学习的关键设计，不继续建设额外平台。使用同一小批 DEV_MECHANISM roots；建议 20–30 题启动，样本量只用于开发判断。

### 16.1 P1：责任表示与粒度

**实现内容**：为同一 runtime 增加三个配置，不增加三套执行器。

| 条件 | 具体生成方式 |
| --- | --- |
| 粗粒度 | root 原子实现，完整责任都在一个节点 |
| 函数级 | 拆成函数并保留签名和目标，不增加完整义务分配证据 |
| 层次契约 | 本版 plan + 责任细化 + 契约 + 显式消费 |

固定模型、总预算和可见测试。检查父义务遗漏、接口失败、局部可验证覆盖、职责重做、组合功能与成本。不要把函数多本身当好分解。

这里同时改变了表示和策略可用信息，属于机制比较，不自动等于严格的单因素训练消融；论文若声称特定字段的独立作用，需另作针对性控制。

**产物**：`mechanism_p1.json` 和简短选择理由。函数级载体若足够，就不扩 AST/PDG 平台。

### 16.2 P2：事后链接与生成期共同承诺

**新写的小型 baseline driver**：同一模型直接生成代码后补责任链接；先一次性静态计划再代码；本版逐节点承诺契约与实现。

对直接生成的多函数模块，用轻量解析收集函数，不根据人工正确责任重新切成对主方法有利的样本。分别测“来源链接完整”与“语义履约”，不合并。

不提前给主方法额外 SFT 后再与未训练 baseline 比机制。训练收益在 A5 单独比较。

**产物**：`mechanism_p2.json`、来源覆盖与语义检查的分列结果。

### 16.3 P3：日志定位、责任筛选与有限重放

同一失败快照、同一可见 evidence、同一冻结修复模型：

```text
R0：直接从日志提出候选边界。
R1：责任索引筛选后，由同一模型提出边界。
R2：R1 + ticket 冻结检查 + 在相同总补丁预算内重放选择。
```

所有组获得相同的失败反馈权限。R0/R1 的补丁也必须接受最低执行安全与测试，不通过拆掉安全检查制造弱基线；R2 的增量是责任边界与受控候选比较。

比较已知注入位置命中、可见/私有修复、弃权、边界大小、回归与总成本。一个模型只给了合法 JSON 不算定位正确。

**产物**：`mechanism_p3.json`。无清楚优势时保留最简单可信方式，不因无显著差异阻止 A4。

---

## 17. A3：冻结一种主机制，不保留待挑选菜单

新增 `configs/frozen_mechanism.yaml` 和一个数据/检查协议版本。冻结：

- action schema、observation 字段、PLAN 修订规则及模型模板；
- 主责任表示、源码所有权单位、artifact/消费格式；
- 局部 oracle 适用规则、责任 UNKNOWN 的处理；
- 候选与补丁预算、ticket 类型、可见补丁选择规则；
- SFT 正例证据标准、功能负例和责任负例的区别；
- 开发分区、采样与初始训练配置。

正常效果不佳后不得同时改 action、prompt、oracle、数据和预算。协议确有 bug 可以修改，但需变更版本并标记受影响样本；不是所有字段调整都必须变成新研究分支。

---

## 18. A4：从新数据构造可追踪的责任训练集

### 18.1 具体新增脚本功能

在 `dataset.py` 与薄 CLI 中实现：

```text
import-roots       导入任务、规范化题型、写 root manifest
build-demos        从人工/teacher 候选运行完整示范
collect            从固定 policy 每 root 采 K 条真实轨迹
make-contrasts     在 TRAIN 中构造有记录的职责迁移/绕过/遗漏
validate-labels    运行独立检查、记录 UNKNOWN 和抽查结论
export-sft         输出 D_PLAN/D_IMPL/D_COMPOSE/D_CONTROL
make-schedule      生成本次训练的 bucket/root 采样清单
```

它们可作为同一 CLI 的子命令，不需要为每个功能建设框架。

### 18.2 分解样本的产生算法

对同一个父状态：生成 2–3 个不同合法计划 → 结构检查 → 同一个 E_ref、同预算实现并组合 → 单独记录计划缺陷、实现失败与 unknown → 保留有证据的计划示范。

失败类型很明确（例如删去必须返回的输出）时可以产生强负标签。两个合法计划各采一次的成败差异不足以判定谁“本质更优”；作为经验信息保留，不能强造粒度偏好。

允许多个合格分解以及合适 ATOMIC。不要把一个 teacher 的任务树定为唯一 gold。

### 18.3 履约与组合的对照构造

从已核查的合规实现出发，分别只改变一个因素：

```text
遗漏一个要求
把相邻节点的算法移入当前节点
调用 child 后丢弃返回并自行重算
选择一个过期 artifact
把父组合中的参数映射改错
悄悄扩大 contract.owns 或删除 ensures
```

保存原始和变换后代码、变换操作、根/局部检查、责任标签强度。加入变量改名、合法 helper、不同正确算法等正例，避免模型只是识别变换风格。

能够保持根功能正确的违规变体优先用于证明责任监督的独立性；普通功能失败也保留，但单列。

### 18.4 数据规模与推进条件

先完成 50–100 个 TRAIN roots 的 pilot，产生多少可信动作就报告多少，不以必须达到某条数作为硬门禁。自然正例不足时增加可审查示范，不从 SEALED 寻找容易成功的题。

分层人工复核计划、履约、消费、unknown 和受控负例；分歧与证据不足样本不混入强正标签。扩量之前先检查同类错误是否来自导出、错误 oracle 或未来信息泄漏。

**阶段产物**：root manifest、raw traces、监督文件、负例/拒收文件、证据审核表、训练采样清单。A4 不需要产生一个新的训练模型。

---

## 19. A5：训练四组核心模型，并隔离两项能力

### 19.1 四组均从公开 theta0 新训

| 组 | 针对性分解 D | 针对性履约/组合 I | 数据含义 |
| --- | --- | --- | --- |
| F00 | 无 | 无 | 新采集的普通功能质量轨迹，同样支持合法 action 协议 |
| F10 | 有 | 无 | 分解槽位用责任分配示范；实现槽位用功能数据 |
| F01 | 无 | 有 | 实现/组合槽位用责任监督；计划槽位用基础规划数据 |
| F11 | 有 | 有 | 主方法，形成与保持联合训练 |

“不针对性训练”不意味着删除模型的计划或代码能力。F00 也要知道协议；不能让它因为完全不会 JSON 而天然处于劣势。

未开启的监督槽位用同域基础任务样本替换，不简单减少更新数。尽量匹配 root 池、采样步数、非 padding target token 和训练资源；不能同时精确匹配时报告实际差异，不声称严格相同。

### 19.2 推荐执行顺序

先各训练一次 F00/F11 pilot，排除 loss/数据错误；随后完成 F10/F01。正式比较的四组使用冻结后的统一配置，从同一 theta0 独立训练，不能让 F11 额外继承 pilot 的更新。

E_ref/R_ref 的来源在比较前固定。开发用 1 个 seed；正式核心比较建议 3 个 matched seeds，资源不足则限定稳定性结论，不挑 best seed。

### 19.3 分解隔离实验怎么实现

评价 driver 只向被评模型开放 plan 决策；然后将其计划交给同一 E_ref 完成实现与组合。若评估整棵递归分解树，必须由被评模型生成所有层的计划，不能只生成 root 计划后由 E_ref 替它继续分解而仍称“整树分解能力”。

首版可以先评价单个父状态的分解决策，明确 `evaluation_scope=local_plan`；随后再增加整树模式。固定所有组的 budget 与已知接口。

### 19.4 履约隔离实验怎么实现

使用同一批经过核查的固定契约与只读 artifact，禁用重新分解/修改 incoming contract，比较各组 write_code 和 consume/组合。保存其原始输出、runtime 拦截、功能、责任与 unknown。

在合法责任重分配条件下重复测量：同一整体任务，不同职责归属，模型是否相应改变实现。

### 19.5 自由递归评价

模型自行决定整条路线，不把 teacher plan 注入主结果。分别保存初始程序、预算内可见修复结果和最终私有评分。能力隔离结果不能替代自由运行，也不能用自由运行一个总分解释所有能力。

**阶段产物**：四组新 adapter、配置/数据/模型身份、隔离结果、自由运行开发结果与一次聚合诊断。

---

## 20. A6：有限优化；DPO 不是必经步骤

### 20.1 先区分问题来自哪里

将开发失败归为 D（义务分解/接口）、I（局部履约）、C（消费/组合）、R（定位/补丁）、SYS（框架/证据）。工程 bug 先修复并重算受影响结果；模型失败只允许一次主要数据/训练修订。

每次只改变一种主要因素。单题 persistent failure 进入案例，不触发一串新 prompt 版本。

### 20.2 条件满足时才新增一种 DPO

只有存在足够的同状态、证据明确的 chosen/rejected 时，才在新文件 `train_dpo.py` 中实现一种现成 DPO 训练路径。

输入必须相同：同一 observation、同一契约、同一可见信息和预算。response 就是完整的一次 plan、write_code 或 consume 决策；不任意截取一些有利 token 后还称标准 DPO。

从同一个 F11 checkpoint 分出：

```text
F11-SFTplus：使用同一新增数据源继续 SFT。
F11-DPO：使用同上下文偏好数据训练。
```

DPO 无 pair 或无收益，停止该分支，继续 A7。未实现 DPO 不代表核心项目未完成。

---

## 21. A7：正式检验“责任结构是否让修复更可靠”

### 21.1 轨道一：固定失败，比较定位/边界方法

新增 `evaluate --track common_failure`。输入是冻结的失败源码、契约、绑定、可见 evidence 和合法预算，所有方法读取相同数据并使用同一 R_ref 生成补丁。

故障池包含受控故障和自然失败，分别标记来源标签可信度。自然失败无可靠来源时只报告修复和边界，不制造定位 accuracy。

输出候选命中（有标签时）、弃权覆盖、自动补丁成功、私有测试成功、冻结违规、边界大小、回归和成本。

### 21.2 轨道二：固定修复器，比较生成期结构

新增 `evaluate --track structure_to_repair`。F00/F10/F01/F11 生成不同结构，然后共用同一 locator/R_ref、预算和检查权限。

受控故障尽量对应同一外部义务或同类局部错误；无法在不同结构上等价注入时，记录适用性与排除原因。不能只选择对主方法有利的节点。

同时报告全 root 的预算内成功。只比较各模型自己的失败修复率会受失败难度不同影响，不能单独当结构价值的因果证据。

### 21.3 四种最小修复条件

```text
无修复
整程序修复/重试
仅提示指定局部区域
责任证据 + ticket + 冻结补集 + 可见重放
```

相同失败、R_ref、总调用/token/测试预算；private 测试不参与选择。整程序对照由实验配置将 ticket 的 editable_files 设为全部候选源码，仍不允许修改根规格、私有 oracle 或环境；不增加第 11 个业务 action。边界过大虽可能提高成功，也必须报告改动与成本，不只展示修复率。

### 21.4 如需定位学习，先确认瓶颈

候选召回明显不足、且已有可信候选/来源标签时，A8 才加入小型 LOCATE SFT。输出的“已知错误来源”“成功干预对象”“证据不足”分开标注；不把任何修复成功自动回填成唯一根因。

定位训练前后使用同一 repair model 比较，不同时增强补丁模型。没有独立收益则保留固定定位方式。

---

## 22. A8：可选 RL、最终冻结与停止规则

### 22.1 RL 只在需要检验信用分配时增加

没有必要为了论文完整而同时实现 SFT、DPO、GRPO。只有在 SFT 形成稳定行为、真实分组采样存在有信息的奖励差异、且本轮要研究局部责任信用时，才增加一种 RL。

先在 TRAIN 上零更新采样，每 root 3–4 条轨迹；检查可见功能、责任证据和组内差异。不制造随机 reward 让算法看起来在更新。

比较同一出发模型、预算和采样的 RL-T（终局功能）与 RL-R（终局 + 有证据局部责任）。不得奖励节点数量或把未知当负例，也不能把根失败平均惩罚全部正确 child。没有奖励信息，允许一次课程调整后停止。

### 22.2 最终冻结清单

`final_manifest.json` 包含模型及 adapter hash、tokenizer/template、依赖 lock、action/observation、runtime、oracle、数据分区、选择规则、预算、seed、主指标和统计方法。

正式 frozen batch 可以包含预注册的多个 seed 和条件，不等于只运行一条命令。基础设施错误允许统一重跑受影响范围并留痕；不因结果不好改方法后仍叫原冻结实验。

### 22.3 两类门禁

| 必须阻塞并修复 | 框架阶段只记录，不阻塞 |
| --- | --- |
| 代码隔离不可靠、私有答案泄漏 | 模型不会分解、总选 ATOMIC |
| oracle 与契约绑定错误 | 新子契约无法获得局部 oracle |
| runtime 替模型消费或写组合逻辑 | 模型不消费或错误拒绝 child |
| stale 证据被当当前 PASS | 模型选择旧 artifact 且被正确拦截 |
| 无梯度、无参数更新、mask 错误 | loss 不改善或短训生成仍失败 |
| frozen 源码被未授权修改 | 模型提出越界补丁且被拒绝 |
| 失败被丢弃或改变分母 | 通过率、修复率、责任符合率为零 |

---

## 23. A9：正式评测的具体实现与指标

### 23.1 需要实现的评价入口

| track | 输入 | 固定项 | 输出 |
| --- | --- | --- | --- |
| `initial_generation` | root 与 visible 接口 | 模型预算与候选选择 | 第一完整候选的私有功能/责任评分 |
| `local_plan` / `recursive_plan` | 父状态或 root | E_ref 与预算 | 分解覆盖、接口、组合及成本 |
| `fixed_contract` | 固定契约和 artifact | 契约、证据、允许 scope | 履约、消费、越界及 unknown |
| `common_failure` | 相同失败快照 | R_ref、反馈、预算 | 定位与修复 |
| `structure_to_repair` | 各组结构及配对干预 | 相同 R_ref | 前两项学习的修复价值 |
| `budgeted_end_to_end` | root | 总资源和可见选择规则 | 最终成功、责任、全部成本 |
| `mechanism_holdout` | 新责任配置/故障类型 | 冻结全系统 | 结构条件化和故障泛化 |

不把所有模型×所有 locator×所有训练算法做全笛卡尔积。初始生成比较固定修复条件，定位比较固定失败集，增强算法只与其对应控制成对比较。

### 23.2 第一候选与修复后候选必须区分

第一完整可执行 root 候选在第一次 root 验证前/时固定快照；后续修改不得覆盖它。子节点在构造完整程序前的局部验证和修复是否允许，由初始生成协议事先统一声明，并计入成本。

`FirstProgramSuccess` 表示该第一完整程序的私有评分；它不冒充“只调用一次模型”的代码 Pass@1。`BudgetedSuccess` 表示固定预算内按可见反馈选出的最终产物。严格单次 direct baseline 的 Pass@1 另列。

### 23.3 核心指标

| 目标 | 指标与口径 |
| --- | --- |
| 功能 | 第一完整程序成功、预算内最终成功；全 root 为分母 |
| 责任 | 父义务遗漏、契约履行遗漏、明确语义越界、代码越权、依赖绕过；同时报告检查覆盖与 UNKNOWN |
| 联合质量 | 功能通过且预先指定责任检查全部有支持的符合；UNKNOWN 不计保守联合成功，但不称违规 |
| 消费 | selected、linked、called、dependency_supported 分列；无消费机会的 root 单列 |
| 修复 | 共同失败集自动修复率、尝试覆盖、可见与私有通过、回归、授权/实际改动范围 |
| 成本 | 所有输入/输出 token、调用、验证/重放、延迟、训练与标注离线成本 |

功能正确条件下的违规率为：`功能正确且明确违规数 / 功能正确数`。分母为 0 时记 N/A。责任状态 UNKNOWN 不应被悄悄算成无违规。

节点/边指标先在 root 内汇总再取 macro；不能靠增加简单节点提高主分数。全部失败、超预算、弃权、protocol error 和 infrastructure error 保留记录，基础设施问题单列并按冻结规则处理。

### 23.4 最小基线组

新实现 DIR-0（相同底座直接生成）、AUDIT-PROMPT（同底座、提示与审计协议）、F00/F10/F01/F11。AUDIT-PROMPT 需要明确复现范围，不能仅因设计相似就宣称完整复现某篇论文的全部实现。

外部竞争基线在 A8 冻结前按当时可获得版本核验。本版不要求 A1 先复现所有论文，也不自动声称优于未运行的方法。

### 23.5 泛化与外部效度

优先做责任重分配、未见组合、未见故障/合法重构等机制泛化。外部域只新增一个 ClassEval 或小型仓库适配器，A1 不建设。

外部域的 sealed 任务不能用于调契约或补训练。适配范围不足时如实限制论文结论，而不是横向加多个域拖延核心证据。

### 23.6 统计和预算

root/family 是主要独立单位，节点、action、同 root 多次重放不是独立样本。方法比较保持 root 与 seed 配对，报告置信区间及逐 seed 结果。

样本量根据开发方差、资源与希望辨别的效应确定，不把某个任意任务数当结项门槛。训练、验证与最终推理成本分开核算。每条 child 不能重新获得 root 预算；反事实执行不能从成本中消失。

---

## 24. A10：论文结项与证据对应

| 论文主张 | 必须对应的产物 |
| --- | --- |
| 学会责任分解 | F10/F11、固定 E_ref 的局部/整树分解结果，不只是树更深 |
| 学会契约条件履约与组合 | 固定契约、功能匹配责任对照、原始动作与 runtime 干预分列 |
| 两种学习相互支持 | F00/F10/F01/F11、训练量与数据来源说明 |
| 责任结构改善修复 | 固定 R_ref、共同失败与结构到修复两条轨道、全 root 结果 |
| 有界修复有效且更局部 | 自动补丁、冻结检查、私有评分、回归与实际改动 |
| 效率/泛化 | 相同预算、完整成本、机制 holdout 与明确外部范围 |

若某一环节没有获得支持，在该处缩小主张。不能通过只展示自定义责任得分、隐藏功能下降、使用参考补丁充当自动能力等方式维持完整故事。

A10 交付：完整配置与产物 manifest、全量结果、成本、主张—证据表、失败案例与局限；再据此完成论文。

---
## 25. 贯穿三部分的可实现示例：前缀和与区间查询

本节是本版新设计的 fixture，用于展示协议和数据，不是 APPS 实验结果。

### 25.1 根任务与合法职责

输入第一行为 n、q；随后 n 个整数及 q 对 1-based 闭区间端点。输出每个区间的元素和。

```text
输入：
4 2
2 4 1 3
1 3
2 4

预期输出：
7
8
```

合法分解：

```text
N0 solve(text): 解析、传值、调用、格式化
├── N0_1 prefix_sums(nums): 只建立前缀和
└── N0_2 query_sums(prefix, queries): 使用给定前缀和回答查询
```

N0_2 不重新生成 prefix，也不 import sibling；prefix 是调用方按契约提供的参数。两个代码 artifact 都由 N0 显式消费，N0 负责传递实际数组。

### 25.2 三个模块的合格源码

`nodes/N0_1.py`：

```python
def prefix_sums(nums):
    result = [0]
    for value in nums:
        result.append(result[-1] + value)
    return result
```

`nodes/N0_2.py`：

```python
def query_sums(prefix, queries):
    return [prefix[right] - prefix[left - 1] for left, right in queries]
```

`nodes/N0.py`：

```python
from nodes.N0_1 import prefix_sums
from nodes.N0_2 import query_sums

def solve(text):
    values = iter(map(int, text.split()))
    n, q = next(values), next(values)
    nums = [next(values) for _ in range(n)]
    queries = [(next(values), next(values)) for _ in range(q)]
    prefix = prefix_sums(nums)
    answers = query_sums(prefix, queries)
    return "\n".join(map(str, answers))
```

local suite 使用例如 nums=[]、单元素、负数、多个查询、左右端点相同等用例；query oracle 可以用原数组朴素求和独立检查。suite 必须只在满足契约的输入上判断实现。

### 25.3 预期 action 序列

```text
01 N0    plan(SPLIT，两个完整子契约)
02 N0_1  plan(ATOMIC)
03 N0_1  write_code(prefix_sums)
04 N0_1  verify(local)
05 N0_1  finish
06 N0    consume(N0_1@v1，host 当前 hash)
07 N0_2  plan(ATOMIC)
08 N0_2  write_code(query_sums)
09 N0_2  verify(local)
10 N0_2  finish
11 N0    consume(N0_2@v1，host 当前 hash)
12 N0    write_code(solve，自己写 import 与业务传值)
13 N0    verify(root-visible)
14 N0    finish
```

系统自动执行的是步骤间的调度和证据写入，不是第 06、11、12 步。

### 25.4 对应的 SFT 样本

| event | 输入 | target | bucket |
| --- | --- | --- | --- |
| 01 | 根任务、外部义务、NEW | 完整 SPLIT action | D_PLAN |
| 02/07 | 当前子契约、NEW | ATOMIC action | D_PLAN |
| 03/08 | 固定子契约、IMPLEMENT | 各自的完整 write_code JSON | D_IMPL |
| 06/11 | 当时返回的 child card、PASS evidence、未绑定状态 | consume | D_COMPOSE |
| 12 | 父职责、两个已选版本与接口 | 组合 write_code | D_COMPOSE |
| 04/09/13、提交动作 | 对应候选状态和可见反馈 | verify/finish | D_CONTROL |

计划样本不能提前看到第 13 步结果；第 06 步不能看到第 08 步尚未生成的 child 代码。后续检查可以决定是否选入样本，但不进入当时的输入。

### 25.5 功能正确但责任违规的变体

在 N0 内保留正确前缀和调用，却把查询计算改成：

```python
answers = [sum(nums[left - 1:right]) for left, right in queries]
```

根输出可以仍为 7、8，但 parent 重做了分配给 N0_2 的查询职责，N0_2 的结果没有参与求解。此变体用于责任对照，不作为合规 SFT target。

如果重新分配契约，让 N0 明确拥有直接查询求和职责，则上述行为可能合法。评价必须读取当前责任配置，而不是永远禁止 sum。

### 25.6 失败与有界修复变体

再将上述未经授权的重算改成：

```python
answers = [sum(nums[left:right]) for left, right in queries]
```

该输入会得到 5、4。child 两个模块仍正确，但 parent 的绕过兼有索引错误。

起始责任候选是 N0 的消费/组合；ticket 只允许修改 `nodes/N0.py`，冻结两个 child。修复恢复 `answers = query_sums(prefix, queries)`，重新验证 root-visible 与回归。

这一例支持的工程预期是：源码替换、冻结 hash、可见重放和数据导出能工作。撤销已知变体是机制测试；由模型自动找出并写出补丁才是自动修复实验。

---

## 26. A0–A10 的交付清单：每一步具体留下什么

| 阶段 | 新实现/执行任务 | 必须留下的产物 | 推进标准 |
| --- | --- | --- | --- |
| A0 | 空仓库、依赖、公开模型身份、初始配置和分区 | pyproject、lock、STATUS、configs、manifest | 新环境可 import，身份与任务边界清楚 |
| A1 | 按 F1–F9 写完首版 runtime、数据、SFT 与评价 | 12 类 fixture、真实日志、新 smoke adapter、回接报告 | 完整流程可信，不看效果是否好 |
| A2 | 三个小型机制 driver 与配对比较 | P1/P2/P3 开发报告 | 有足够信息选择合法起点 |
| A3 | 只保留一种主机制与检查/预算规则 | frozen_mechanism.yaml | 不留下事后选择菜单 |
| A4 | 示范运行、候选采集、责任负例、验证与导出 | 新 TRAIN 动作集、证据和拒收表 | 标签用途清楚且可追溯 |
| A5 | 新训 F00/F10/F01/F11；两项隔离 + 自由运行 | 四组 adapter、完整训练身份和结果 | 已按对照检验能力，不要求全正向 |
| A6 | 一次聚合修订；可选一种 DPO 及继续 SFT 控制 | 修订或分支停止报告 | 不为少数失败无限迭代 |
| A7 | common_failure 和 structure_to_repair 两轨 | 相同 R_ref 的定位/修复/成本结果 | 两项学习的下游价值可被检验 |
| A8 | 必要时小定位训练或一种 RL；最终冻结 | final_manifest、统计计划 | 隐藏结果不参与继续开发 |
| A9 | 冻结批次、机制 holdout、一个外部域 | 全 root 结果、成本、unknown/失败分母 | 无选择性删样或挑 seed |
| A10 | 分析、主张证据对照和论文 | 完整研究记录与稿件 | 结论严格对应实际证据 |

### 26.1 开发任务的统一交接格式

```text
当前阶段：F3
本次只实现：child 返回后的 parent 显式接收决策
修改文件：runtime.py / artifacts.py / test_artifacts.py
输入：FX02 的 child artifact 与真实验证记录
输出：consume / reject / request_child_repair 后的状态和 trace
应通过：版本一致、未自动消费、拒绝不抹掉义务
不要求：LLM 对真实任务的消费成功率提高
下一项：F4 的 ticket 与冻结补丁
```

将这个模板用于每个开发任务，避免一句“完成可审计 runtime”导致实现范围失控。

### 26.2 项目最先执行的五项任务

1. 建立空仓库、schema 和配置；写整数求和 fixture，不导入旧模块。
2. 实现 ScriptedPolicy 和原子状态转移，保存 raw action；停在待验证状态也如实记录。
3. 接真实 sandbox/verifier，完成原子链。
4. 实现 SPLIT 和显式消费，跑通前缀和示例及孙节点 fixture。
5. 加 ticket/repair 后再接真实 LLM、导出动作数据、做一套 SFT smoke 与回接。

这五项是执行顺序，不要求把每项先优化到论文水平。

### 26.3 首版禁止新增的工程负担

不建图数据库、微服务、远程调度、全套多智能体平台、前端门户、多后端统一接口、技能库检索、完整因果集合穷举或多算法训练菜单。日志、配置、类型检查和少量测试足够支持首版。

本文字段用于责任证据和训练接口；若实现者想新增字段，先说明它直接支撑哪条 action、标签或指标。仅为“以后可能用到”不新增。

---

## 27. 最后核对：不要在实现中偷换的 12 件事

| 不要偷换 | 正确实现 |
| --- | --- |
| 从零开发变成整理旧成果 | 全部新建，公开 theta0 是唯一模型起点 |
| task tree 等于代码调用图 | task tree 记录分工，调用与数据依赖由实际代码/执行检查 |
| consume 等于 auto-compose | 模型选择版本并自己写组合逻辑 |
| code artifact 等于测试答案 | artifact 是可调用模块，具体值只由运行产生 |
| host 跑了测试等于 oracle 可信 | 期望结果有独立来源，契约绑定正确 |
| root 通过等于每个节点履约 | 局部与语义责任独立检查，未知单列 |
| 来源 ownership 等于没有语义越界 | 谁写代码与代码做了哪些职责分开 |
| finish 等于 PASS | host verdict 才决定当前提交状态 |
| 整条成功 trace 都是 SFT 正例 | 按 action 的证据范围选择，失败局部也可有正例 |
| labels=bad 就会学会避开坏代码 | bad code 不做正向目标，使用纠正监督或合法偏好 |
| 保存了 adapter 就是训练完成 | 真实更新、重载、forward 与 runtime 回接都验证 |
| 修复成功等于唯一根因 | 区分来源、干预对象、修复边界与检查范围 |

**最终实施顺序：先从零写通协议与执行，再写通数据与一套 SFT；然后验证责任分解和履约学习，最后用固定修复器检验结构的价值。**

---

## 28. 依据与外部实现文档

### 28.1 本文继承与新增的边界

基础材料是本对话提供的 `AuditTrainer_三主线框架优先去工程化研究路线_v3.0.md` 及用户最新“从零重写、逐步落实 action/policy/SFT”的要求。

继承的是三主线、A0–A10、双层门禁、有限开发、能力隔离和固定修复器设计。完整 action 枚举与 payload、状态机、artifact 编译语义、文件职责、SFT 编码/损失实现、起始配置及新 fixture 是本次新增规格。它们尚需在新仓库实现和验证。

不继承旧实验数值、已完成阶段、旧 checkpoint 或旧代码可用性假设；也不将前版提到的文献名称自动作为已核验方法依据。

### 28.2 本次核对的官方技术来源

以下来源核对日期为 2026-09-06。它们支撑软件用法与数据格式，不为本研究方案的有效性背书。

**[E1] Qwen3-8B 官方模型卡**：模型标识、chat template 与 `enable_thinking=False`。

`https://huggingface.co/Qwen/Qwen3-8B`

**[E2] Transformers Trainer 官方文档**：自定义 collator、Trainer 与 compute_loss 接口；实际实现需锁定依赖版本。

`https://huggingface.co/docs/transformers/en/main_classes/trainer`

**[E3] PEFT Quantization 与 Transformers Bitsandbytes 官方文档**：量化底座上的 adapter 训练、prepare_model_for_kbit_training、NF4/计算精度与训练设备映射。

`https://huggingface.co/docs/peft/en/developer_guides/quantization`

`https://huggingface.co/docs/transformers/en/quantization/bitsandbytes`

**[E4] Transformers Chat Templates 官方文档**：模板渲染、生成前缀和特殊 token 的一致处理。

`https://huggingface.co/docs/transformers/en/chat_templating`

**[E5] TRL SFTTrainer 官方文档**：prompt-completion 数据与 completion-only loss。仅作为兼容参考，本版不要求同时实现第二套 trainer。

`https://huggingface.co/docs/trl/en/sft_trainer`

**[E6] APPS 官方仓库与 testing_util.py**：原始数据/评价入口以及通过 fn_name 区分 stdin 与 call-based 任务。

`https://github.com/hendrycks/apps`

`https://github.com/hendrycks/apps/blob/main/eval/testing_util.py`

文中的模型参数、数据规模、预算和课程比例均是执行起点。正式论文需另行核验当时的竞争方法与数据版本，不能由本实现路线替代相关工作审查。
