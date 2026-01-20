# 安装指南

`py-adtools` 兼容 **Python 3.10** 及更高版本。

## 选项 1：从 PyPI 安装

对于稳定版本（推荐）：

```bash
pip install py-adtools
```

## 选项 2：从源码安装

如需最新功能和开发更新：

```bash
pip install git+https://github.com/RayZhhh/py-adtools.git
```

## 依赖项

核心库是轻量级的。但是，特定模块有可选的依赖项：

- **Ray 沙盒**：需要 `ray` 和 `ray[default]`。
    ```bash
    pip install ray[default]
    ```
- **本地 LLM 服务器**：需要 `vllm` 或 `sglang`。
    ```bash
    pip install vllm
    # 或者
    pip install sglang
    ```
- **异步客户端**：异步提交客户端需要 `aiohttp`。
    ```bash
    pip install aiohttp
    ```
