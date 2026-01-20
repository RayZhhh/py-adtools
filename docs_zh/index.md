# 欢迎使用 Py-ADTools

**Py-ADTools** 是一个专为 **LLM 辅助算法设计 (LLM-AAD)** 和代码优化而设计的 Python 工具包。它弥合了生成代码（使用 LLM）和验证代码（使用安全执行和评估）之间的差距。

## 工作流

Py-ADTools 促进了一个闭环工作流：

1.  **解析**：将代码分解为结构化对象 (`py_code`)。
2.  **生成/修改**：使用 LLM 变异函数或生成新算法 (`lm`)。
3.  **沙盒**：在隔离的进程中执行可能不安全的代码 (`sandbox`)。
4.  **评估**：基于正确性或性能对代码进行评分 (`evaluator`)。
5.  **迭代**：将结果反馈给 LLM。

## 模块概览

- **[代码解析 (`adtools.py_code`)](./parsing.md)**：
    一个强大的解析器，将 Python 源代码转换为可操作的 `PyProgram` 对象。它允许你以编程方式重命名函数、交换主体或注入文档字符串，同时保留原始布局和注释。

- **[安全执行 (`adtools.sandbox`)](./sandbox.md)**：
    一个安全层，在单独的进程（通过 `multiprocessing`）或分布式 Actor（通过 `Ray`）中运行不受信任的代码。它提供超时强制、输出重定向和僵尸进程清理。

- **[评估框架 (`adtools.evaluator`)](./evaluator.md)**：
    一个基于模板的系统，用于定义应如何测试代码。它处理“字符串到函数”的转换和沙盒生命周期，让你专注于测试逻辑本身。

- **[自动评估服务器 (`adtools.evaluator.auto_server`)](./auto_server.md)**：
    只需一条命令即可将任何评估器转换为并发 HTTP 微服务。非常适合将繁重的评估任务与主生成循环解耦。

- **[LLM 接口 (`adtools.lm`)](./language_models.md)**：
    一个统一的 API，用于连接到 OpenAI，或管理本地高性能推理服务器，如 vLLM 和 SGLang。

## 快速开始

查看 [安装指南](./installation.md) 进行设置，或探索 README 中的 [教程](../README_zh.md#教程)。
