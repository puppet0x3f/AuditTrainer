# AuditTrainer 数据与检查协议 v1

版本：`at-data-check-v1`。首次冻结为主机制 `a3-main-v1`、运行协议 `at-v3r1-a3.1`，日期 2026-09-08。
当前运行协议为 `at-v3r1-a3.2`、主机制 `a3-main-v1.1`：A4 修复了未展示每节点 child 上限的
observation 缺陷，增加 `budget_remaining.child_slots` 和明确的拒绝反馈。原预算与本数据检查规则不变，
受影响旧轨迹记录于 `runs/a4/protocol_bug_impact.json`，重跑后再导出，不回填旧输入。
可执行配置为 `configs/frozen_mechanism.yaml`；核心 SFT 配置为 `configs/sft_main.yaml`。

## 1. 唯一主机制与冻结边界

主机制采用逐节点承诺的层次契约，每个节点拥有一个 Python 源码模块及一个契约公开函数，
私有 helper 以 `_` 开头。共享 Qwen3-8B 策略负责根、子与孙节点；PLAN、IMPLEMENT、COMPOSE、
LOCATE、REPAIR 是输入模式。`Policy.act(observation)` 每次返回一条动作、原始输出、用量与模型身份。

动作固定为 `plan / write_code / verify / consume / reject / request_child_repair / localize /
repair / finish / abstain`。`plan` 可选 ATOMIC 或 SPLIT；模型根据任务选择，不固定子节点数量。
SPLIT 由 host 深度优先顺序执行；最大深度 2、累计节点 7、每节点 child 3。
内部计划只能经 CONTRACT/OMISSION 定位与 PLAN_SUBTREE ticket 修订，每节点最多一次，
重建整个授权子树、保留 incoming contract、重新消费新 child。累计创建节点和失败尝试不返还预算。

`schema.py` 的动作结构、`prompts.py` 的 observation 字段白名单与模板代码保持 F6 版本。
准确字段列表、两文件摘要、tokenizer 与 chat template 摘要记录在 frozen 配置中；
推理和导出使用同一 `prepare_prompt / render_messages`，关闭 thinking。
不使用 A2 专用 prompt、固定动作控制器或固定两 child 模板。

artifact 身份是 node/version/source hash，源码版本不可变。parent 必须显式消费返回的精确版本，
再自行写 import、参数传递、调用和结果组合。`selected / linked / called` 分别记录；
调用 child 不是正确消费其结果的语义证明。`finish` 仅提交当前已验证版本，成功由 host 判定。

## 2. 验证与 UNKNOWN

局部 oracle 必须由 host 独立审查，按 root、完整 ContractSpec 和 contract version 精确适用。
每条证据绑定 artifact 源码、依赖闭包、suite、verifier 与 runner。对不匹配的自由分解，不把其他
契约的 oracle 挪来验证；无适用 oracle、空用例或证据不足按 verifier 结果记 UNKNOWN。
模型自称正确、根测试 PASS、只调用了 child，都不能替代局部责任证据。

在线 observation 仅允许决策时已经可见的 VISIBLE 证据，不包含未来结果、审查答案或 PRIVATE 测试。
TRAIN 的私有 oracle 可在离线构造标签时使用；不修改此前输入。评测的私有 oracle 仅评分在线已选定
的产物，不能选择补丁、候选或重试。APPS stdin 保留现有 `apps_stdio_legacy` 比较规则及标记，
自建 fixture 使用 exact JSON；两者不混称严格等价检查。

功能状态与责任状态分开记录。UNKNOWN 不作 PASS、不自动转为责任负例。
具有执行 PASS 但缺乏独立责任审查的实现只能是 FUNCTION_ONLY；若另有绑定到该动作与契约的独立
SUPPORTED 审查，可形成责任监督，但不能据此把 runtime 的 UNKNOWN 改写为自动证明的 PASS。
已知责任 VIOLATION 不得因功能 PASS 进入核心正例。

## 3. 定位、ticket 与补丁选择

定位采用 R1 责任筛选：从当前可见、版本仍匹配的 FAIL 证据找 owner，先列 child，再列当前节点，
最多 3 个；没有匹配 FAIL 时只保留当前节点供明确的 omission/plan 决策。policy 在候选内声明 NODE、
HANDOFF、CONTRACT 或 OMISSION 目标。缺少局部 oracle 不被伪装成 failed child。
父级扩大须已有局部失败，最多一次；不自动轮换候选，不做 R2 的候选比较重放，不训练独立 Locator。

保留 F4 事务：ticket 前对证据锚点做一次无修改稳定性重放；这是确认修复前提的检查，
不是对所有候选做额外诊断。ticket 有 OWNER_MODULE、CONSUMER_MODULE、PLAN_SUBTREE 三类。
前两类只改授权模块，后一类只重建授权计划子树；补集源码、外部义务和测试冻结。
依赖版本变化必须显式声明 rebind；失败候选保存在隔离目录并回滚，历史源码与证据保留。

根任务共享最多 3 次补丁尝试、32 次在线验证（含自动重放）、64 次策略调用、196608 个总 token。
每节点最多一次 child repair request；解析/动作错误最多一次反馈重试。
上下文 8192 token，plan 输出最多 2048、write_code/repair 3072、localize 768、控制动作 512。
执行禁网，每例 2 秒、1024 MiB、输出 1 MiB、进程上限 1。

按在线决策顺序接受首个满足 `visible_all_pass_no_regression` 的补丁：冻结检查通过，受影响节点、
消费者及已有 root 的固定可见 suite 全部 PASS，已通过用例不回归。UNKNOWN 不批准补丁。
局部修复发生时 root 尚未生成的，保留 root_checked=false，不能声称根任务已经修好。
私有测试与人工挑选不参与自动接受；失败、回滚、重放与弃权均计入成本和结果分母。

## 4. root 分区与数据来源

使用既有 `data/manifests/splits.yaml` 和 `roots.jsonl`，采集前按 root 与 family 分区。
同源 child、候选、重写、故障注入及修复继承 root 分区，不能按函数重新划分。

| 分区 | 固定用途 |
| --- | --- |
| SYN_FIXTURE | 框架与训练连通验收，独立导出，不混入正式 TRAIN |
| DEV_ENGINEERING | runtime、训练回接与统一评测调试 |
| DEV_MECHANISM | A2 机制选择；其代码、负例和修复不能进入正式训练 |
| TRAIN | A4 构造数据、A5 核心 SFT |
| VALIDATION | 后续 checkpoint 选择，不能梯度训练 |
| MECHANISM_HOLDOUT / APPS_SEALED / EXTERNAL_SEALED | 后续正式评价，不能训练或调协议 |

A3 时已登记 6 个 SYN_FIXTURE、5 个 DEV_ENGINEERING、20 个 DEV_MECHANISM。
TRAIN、VALIDATION 与封存评测集尚未构造；A3 固定规则，不把已看过的开发任务重新命名为未见数据。
公开底座的预训练暴露未知，不据此声称 benchmark 无污染。

## 5. 一步动作正例标准

最小样本为 `observation_before → 一条完整 action JSON`，附带 root/family/split、run/event/node、
契约/计划/artifact 版本、证据引用、label_origin、监督范围和协议版本。
只导出终止轨迹中真实产生且 host APPLIED 的完整策略动作；排除 host 事件、截断输出、越权、
绑定错误、未来输入与重复事件。整个根失败不自动淘汰其中证据充分的正确 child。

| 桶 | 进入核心 SFT 的必要证据 |
| --- | --- |
| D_IMPL | 本动作产生的精确 artifact/version 有匹配执行证据；匹配 suite 均 PASS、执行非空且覆盖契约义务；独立责任 SUPPORTED 审查 |
| D_PLAN | 义务分配无缺口、child 引用有效；独立审查分解与接口；该计划下同节点的实现已实际执行且满足上述功能证据 |
| D_COMPOSE：consume | 对返回 child 的精确版本/hash，引用决策前可见 PASS；独立审查支持此消费决策 |
| D_COMPOSE：write_code | 同 D_IMPL，且已选依赖均 linked/called；独立审查结果确实被使用、没有重复承担 child 算法 |
| D_COMPOSE：reject/request_child_repair | 版本绑定正确，决策前存在 FAIL/UNKNOWN 支持；标为 PROTOCOL_ONLY |
| D_CONTROL | verify 对当前版本与可用 suite；finish 据实提交；abstain 有可见失败/UNKNOWN、耗尽预算、拒绝反馈或独立审查依据；均只标 PROTOCOL_ONLY |

SUPPORTED 审查应来自独立源码/接口审查或可信构造规则，绑定 action hash 与 contract hash，
记录判据与 rationale；不把生成者自评当独立证据。所有匹配执行结果均须考虑，不能只挑 PASS suite。
修复后的 v2 PASS 不能回填 v1。`localize / repair / plan revision` 保留原轨迹，初始核心 SFT 不训练这些动作。

## 6. 功能负例与责任负例

功能负例：精确版本在适用 oracle 上 FAIL，有失败用例及版本证据。
基础设施 ERROR、无 oracle 的 UNKNOWN、格式无效动作分别记录，不冒充算法功能失败。

责任负例：有独立、绑定到具体义务与源码位置的责任 VIOLATION 证据。用于隔离责任学习效应的
对照必须与正例在同 root、契约、输入、可见接口和测试预算下功能匹配，保留双方执行结果；
例如 parent 调用了正确 child 却丢弃结果重算。双方功能不匹配时可作混合错误诊断，
不能计入“功能匹配责任对照”。仅缺少审查或语义 UNKNOWN 不是责任违反。

负例不以坏 action 目标进入普通 SFT。A4 将另存配对与排除原因，供分析或后续按证据启用的偏好训练。
FUNCTION_ONLY 功能正例只能单独导出作为控制；当前核心 loader 拒绝混入。

## 7. 初始共享 SFT

从固定 revision `b968826d9c46dd6066d109eabc6255188de91218` 的公开 Qwen3-8B 重新开始，
不用 SMOKE_ONLY adapter 作为正式训练起点。NF4 双重量化、BF16 计算，单可见 GPU；
LoRA all-linear，r=16、alpha=32、dropout=0.05，只更新 adapter。

正式数据路径 `data/sft/train.jsonl`，仅 TRAIN。最大长度 8192、每卡 batch 1、梯度累积 16、
学习率 1e-4、weight decay 0、warmup ratio 0.03、cosine、max grad norm 1、seed 42、梯度检查点。
完整 action JSON 与 EOS 监督，prompt/padding mask=-100，不 packing，不截断半条动作；
先按每条样本目标 token 平均 loss，再按样本平均。

前 20% optimizer step 使用 D_IMPL/D_CONTROL=80%/20%；随后使用
D_PLAN/D_IMPL/D_COMPOSE/D_CONTROL=30%/40%/25%/5%。先选桶，再均匀选 root，最后选 root 内动作，
有放回但总抽样量最多一个数据集大小，epoch_limit=1；保存实际抽样顺序与重复暴露次数。
D_COMPOSE 必须同时含 consume 与父 write_code，缺桶不能静默换配比。
初始仅共享 SFT；DPO、RL 与独立 Locator 训练关闭。A5 二因素训练对照使用同一数据检查协议。

A5 对照补充：F00/F10 的 FUNCTION_ONLY 整题实现来自同一 TRAIN root 池中新执行的参考轨迹，
只按功能验证选择，没有人为注入职责违规。仅在训练配置和数据 manifest 均明确
`allow_function_only=true` 时，正式 TRAIN loader 接受这些功能控制；F01/F11 与默认核心配置仍拒绝。
关闭 D 时，以该 root 的基础 ATOMIC 计划替换计划槽位；关闭 I 时，以整题功能实现替换
write_code 槽位。consume 和控制动作四组共享，避免协议能力差异造成天然劣势。
这是分解示范覆盖与模块履约覆盖的二因素干预，不声称仅改变标签、输入与代码逐字相同。

四组读取由 A4 共同采样表派生的有限顺序：每槽位 root、总 2,240 个槽位、140 次更新和 seed 42
相同；保留真实事件 ID，功能替换样本可重复使用，不伪造新事件。这个共同曝光预算代替按各组
去重后行数重新计算 epoch；实际目标 token、输入 token、不同样本数与重复曝光分别报告。
F00/F11 各做一次 5 步连通 pilot；F11 pilot 包含实际顺序中最长的完整样本。
正式四组从公开底座重新初始化，不继承 pilot；同用 CPU 暂存反向传播张量以适应共享 GPU 显存，
不截断动作、不改变 loss 或优化器。当前仅单种子开发比较，多种子稳定性留到正式评测。

A5 的能力隔离配置为 `configs/a5_evaluation.yaml`：8 个事先固定的 DEV_MECHANISM root，
不改原分区；局部分解只让被评模型作 root plan，固定公开 theta0 在关闭 adapter 后完成实现/消费，
子节点计划固定 ATOMIC，验证和提交由固定控制器执行。因此只称 local_plan，不称整树分解。
固定契约实验覆盖 transform、summarize、两 child 父组合和合法改为单 transform child、父承担
summarize 的责任重分配；禁止修订契约与修复 child，保留 host 拒绝、UNKNOWN、失败与成本。
E_ref 与后续 R_ref 的来源均在比较前固定为公开 theta0。自由递归使用既有全部 5 个
DEV_ENGINEERING root，被评模型自主决策，分别保存首次完整程序、预算内终止程序与终止后的私有评分。

在线推理固定 seed 42，do_sample=true、temperature=0.7、top_p=0.8、top_k=20；
所有正式对照使用一致配置和成本预算。A2 的 greedy 为历史机制探针配置，不作为 A3 性能结果。
A3 不选 checkpoint；先用一轮结束的 checkpoint 回接，后续多 checkpoint 选择只能依据 VALIDATION。

## 8. 版本变更与影响样本

导出 manifest 与逐样本 meta 同时保存运行协议和数据检查协议版本；正式 loader 要求匹配。
旧 F6/F7 使用 framework.yaml，仍可做 smoke，但不能直接重标记成 A3 正式训练数据。

能力暂时不佳时先依既定课程补数据与训练，不能同时改 action、prompt、oracle、数据和预算再挑结果。
协议有明确 bug 才修：说明修复原因、递增受影响协议版本，按已有 run_id/event_id/sample_id 与
证据引用列出受影响集合；在重新审查/导出前排除这些样本，保留旧原始轨迹与旧导出。
排版或不改变输入/标签/执行语义的注释不要求新研究分支。

## 9. A6 一次数据覆盖修订

沿用同一运行/数据检查协议，在既有 TRAIN root 上增加 prepare 委托给 child、aggregate
保留给 parent 的实际参考轨迹。仅选择 parent 的完整 observation → write_code 正例进入
修订池，强标签需同时绑定原根契约、新职责分配、实际 child 版本/hash/alias、已审查组件源码
和独立执行证据。不会把新源码放进仍将 aggregate 委托出去的旧输入状态。

A5 F11 每个 root 的父组合 write_code 槽位从首个开始交替替换，共 96/2,240 个；其他槽位
完全不变。保持公开底座、140 步训练和 A5 评测协议，记录实际 token/不同样本曝光差异。
这轮只修订父职责覆盖，不同时修改提示、oracle、训练算法或分解目标，也不新增 DPO。
