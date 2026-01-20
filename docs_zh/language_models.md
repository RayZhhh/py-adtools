# 语言模型实现分析

`adtools.lm` 模块标准化了与远程 API 和本地推理引擎的交互。

## 1. `VLLMServer`：管理器模式

`VLLMServer` 包装了 `vllm` 库的 API 服务器。它不仅仅是一个客户端，它自己管理服务器进程。

### 1.1 启动服务器

```python
# adtools/lm/vllm_server.py

def _launch_vllm(self, detach: bool = False):
    # 1. 构建命令行
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", self._model_path,
        "--port", str(self._port),
        "--tensor-parallel-size", str(len(self._gpus)),
        # ... 其他参数
    ]
    
    # 2. 配置环境
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self._gpus))
    
    # 3. 生成进程
    proc = subprocess.Popen(
        cmd, 
        env=env, 
        stdout=subprocess.DEVNULL if self._silent_mode else None,
        # ...
    )
    return proc
```

**实现洞察**：
- 我们通过 `python -m vllm...` 调用 `vllm`，而不是在主进程中直接导入它。这至关重要，因为 `vllm` 会初始化 CUDA 上下文并分配大量 GPU 内存。通过将其保持在子进程中，我们确保：
    1. 我们可以彻底杀死它 (`kill()`) 以释放内存。
    2. 它不会与主进程中的其他 GPU 库冲突。

### 1.2 健康检查循环

启动后，我们不能立即发送请求。模型加载需要时间（数十秒）。

```python
# adtools/lm/vllm_server.py

def _wait_for_server(self):
    for _ in range(self._deploy_timeout_seconds):
        # 1. 检查进程是否死亡
        if self._vllm_server_process.poll() is not None:
             sys.exit(f"vLLM crashed with code {self._vllm_server_process.returncode}")

        # 2. 检查 HTTP 健康端点
        try:
            if requests.get(f"http://{host}:{port}/health").status_code == 200:
                return  # 就绪！
        except:
            pass
            
        time.sleep(1)
        
    # 超时
    self._kill_vllm_process()
    sys.exit("Failed to start")
```

### 1.3 动态 LoRA 加载

服务器持续运行，但我们可能想要更改 LoRA 适配器。我们为此使用 vLLM 的内部 API。

```python
# adtools/lm/vllm_server.py

def load_lora_adapter(self, lora_name, path):
    payload = {"lora_name": lora_name, "lora_path": path}
    url = f"http://{host}:{port}/v1/load_lora_adapter"
    
    # 发送请求到正在运行的子进程
    requests.post(url, json=payload)
```

**实现洞察**：
- 这利用了 `vllm` 暴露 LoRA 管理端点的事实。通过将这些包装在 Python 方法中，我们使其感觉像是一个本地函数调用，隐藏了 HTTP 的复杂性。

## 2. `SGLangServer`

`SGLangServer` 的实现结构与 `VLLMServer` 相同，但它调用 `sglang.launch_server` 并使用 SGLang 特定的参数（如 `--mem-fraction-static`）。

它作为一个直接替换品：你可以在代码中将 `VLLMServer` 替换为 `SGLangServer`，`chat_completion` 方法将以完全相同的方式工作。
