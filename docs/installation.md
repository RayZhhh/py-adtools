# Installation

`py-adtools` is compatible with **Python 3.10** and higher.

## Option 1: Install from PyPI

For stable releases (recommended):

```bash
pip install py-adtools
```

## Option 2: Install from Source

For the latest features and development updates:

```bash
pip install git+https://github.com/RayZhhh/py-adtools.git
```

## Dependencies

The core library is lightweight. However, specific modules have optional dependencies:

- **Ray Sandbox**: Requires `ray` and `ray[default]`.
    ```bash
    pip install ray[default]
    ```
- **Local LLM Servers**: Require `vllm` or `sglang`.
    ```bash
    pip install vllm
    # or
    pip install sglang
    ```
- **Async Client**: The async submission client requires `aiohttp`.
    ```bash
    pip install aiohttp
    ```