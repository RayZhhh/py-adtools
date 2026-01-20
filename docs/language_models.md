# Language Model Implementation Analysis

The `adtools.lm` module standardizes interactions with both remote APIs and local inference engines.

## 1. `VLLMServer`: The Manager Pattern

`VLLMServer` wraps the `vllm` library's API server. Instead of just being a client, it manages the server process itself.

### 1.1 Launching the Server

```python
# adtools/lm/vllm_server.py

def _launch_vllm(self, detach: bool = False):
    # 1. Construct Command Line
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", self._model_path,
        "--port", str(self._port),
        "--tensor-parallel-size", str(len(self._gpus)),
        # ... other args
    ]
    
    # 2. Configure Environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self._gpus))
    
    # 3. Spawn Process
    proc = subprocess.Popen(
        cmd, 
        env=env, 
        stdout=subprocess.DEVNULL if self._silent_mode else None,
        # ...
    )
    return proc
```

**Implementation Insight**:
- We invoke `vllm` via `python -m vllm...` rather than importing it directly in the main process. This is critical because `vllm` initializes CUDA contexts and allocates massive GPU memory. By keeping it in a subprocess, we ensure that:
    1.  We can cleanly kill it (`kill()`) to free memory.
    2.  It doesn't conflict with other GPU libraries in the main process.

### 1.2 Health Check Loop

After launching, we can't send requests immediately. The model takes time to load (tens of seconds).

```python
# adtools/lm/vllm_server.py

def _wait_for_server(self):
    for _ in range(self._deploy_timeout_seconds):
        # 1. Check if process died
        if self._vllm_server_process.poll() is not None:
             sys.exit(f"vLLM crashed with code {self._vllm_server_process.returncode}")

        # 2. Check HTTP Health Endpoint
        try:
            if requests.get(f"http://{host}:{port}/health").status_code == 200:
                return  # Ready!
        except:
            pass
            
        time.sleep(1)
        
    # Timeout
    self._kill_vllm_process()
    sys.exit("Failed to start")
```

### 1.3 Dynamic LoRA Loading

The server runs continuously, but we might want to change the LoRA adapter. We use vLLM's internal API for this.

```python
# adtools/lm/vllm_server.py

def load_lora_adapter(self, lora_name, path):
    payload = {"lora_name": lora_name, "lora_path": path}
    url = f"http://{host}:{port}/v1/load_lora_adapter"
    
    # Send request to the running subprocess
    requests.post(url, json=payload)
```

**Implementation Insight**:
- This leverages the fact that `vllm` exposes LoRA management endpoints. By wrapping these in a Python method, we make it feel like a local function call, hiding the HTTP complexity.

## 2. `SGLangServer`

The implementation of `SGLangServer` is structurally identical to `VLLMServer`, but it invokes `sglang.launch_server` and uses SGLang-specific arguments (like `--mem-fraction-static`).

It serves as a drop-in replacement: you can swap `VLLMServer` with `SGLangServer` in your code, and the `chat_completion` methods will work exactly the same way.
