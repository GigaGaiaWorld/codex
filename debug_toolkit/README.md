# Debug Toolkit

## 变更说明
- 新增 `debug_toolkit` 包，用于在独立目录中调试工具绑定逻辑。
- 在 `tool_card` 中修正参数填充行为，显式传入 `None` 时不会被类属性或默认值覆盖。
- 提供最小化的 `schema_from_docs` 实现，便于从函数签名生成参数模型。
- 增加可运行示例，展示 `CtxBinding` 与 `tool_card` 的参数填充流程。

## 功能特性
- **参数绑定**：支持从实例属性、默认值或外部调用填充参数。
- **None 处理**：显式传入 `None` 视为已提供参数，避免被覆盖。
- **简化 schema 生成**：根据函数签名自动构建 Pydantic 参数模型。
- **示例演示**：包含可直接运行的示例脚本验证行为。

## 测试方法
在仓库根目录执行：

```bash
PYTHONPATH=/workspace/codex python -m debug_toolkit.example
```

预期输出将展示两次调用：一次显式传入 `None`，一次传入字符串标签。
