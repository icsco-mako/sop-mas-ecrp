项目目录结构（已调整）：

- `src/llm_mas_ecrp/`：Python 包根
  - `core/`：核心业务与算法
    - `sop_mac.py`：SOP-MAC 核心编排入口
  - `config/`：项目配置模块（占位，可扩展）
  - `models/`：模型定义（占位，可扩展）
  - `utils/`：工具模块（占位，可扩展）
- `sop_mac/`：历史模块位置，仅保留兼容导出
  - `sop_mac.py`：从新位置导出 `sop_mac`
- `src/llm_mas_ecrp/services/workflow/`：智能体与消息池实现（迁移自 `sop_mac/workflow`）
- `baseline/`：基线方法（已设置只读保护）
- `model_p2p_compare/`：模型比较与数据（已设置只读保护）
- `tests/`：单元测试占位
- `pyproject.toml`：Poetry 配置

运行方式：`poetry run python main.py`（根据实际入口调整）
