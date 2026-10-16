# A2 固定开发池

20 个自建整数列表组合任务，全部登记为 `DEV_MECHANISM`，共同 family 为 `a2_integer_pipeline_v1`。这是一个有 20 个不同操作组合的受控开发池，不是 20 个独立算法家族，也不是正式泛化基准。所有任务、可见/私有测试和故障方案在第一次模型推理前写入并固定。

每个任务先对整数列表执行 `transform(items)`，再对中间列表执行 `summarize(items)`；root 负责 JSON 解析和紧凑 JSON 输出。两种可选子契约都表达完整的函数目标：函数级仅保留签名、目标和接口约束；层次契约增加 owns/must_not 和父义务分配。局部 verifier 只绑定完全匹配的已审查契约，模型自行改变契约后不会自动获得同样的局部 oracle。

- `task.json`：公开任务、根契约、两种局部契约模板和可见 suite。
- `private_tests.json`：只用于选定产物的离线评分；不进入在线 verifier 或模型 prompt。
- `reference.json`：离线构造故障池使用的自建参考代码和故障位置，模型不读取。
- `selection.json`：固定的 20 个 root、来源和哈希。

每题 2 个 root 可见用例、每个组件 2 个局部用例，以及 5 个 root 私有用例。预期值来自独立于模型的自建封闭操作目录；参考实现与 oracle 共享该目录，不能据此声称形式验证。测试包含空输入、重复值、零、负数及顺序相关操作，另有手算用例检查目录输出。

故障方案按任务编号预定：5 个 transform 错误常量、5 个 summarize 错误常量、5 个 parent 绕过 transform 输出、5 个 parent 输出格式错误。正确参考和故障版本均通过真实 Bubblewrap 执行，分别要求 root 可见 PASS/FAIL。所有 R0/R1/R2 使用相同源码和可见反馈；位置标签仅在评分时读取。

本池不得导入 SFT，也不读取或代替 APPS_SEALED / EXTERNAL_SEALED。A4 需要另建 TRAIN 数据。
