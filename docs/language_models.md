# Language Model Interface

The `adtools.lm` module provides a unified abstraction for interacting with Large Language Models (LLMs). It handles the complexity of connecting to remote APIs (OpenAI) or managing the lifecycle of local high-performance inference servers (vLLM, SGLang).

## 1. Unified Abstraction (`LanguageModel`)

**Implementation Idea**:
To switch between different backends (e.g., prototyping with GPT-4, production with local Llama-3) without changing your application logic, `LanguageModel` enforces a standard interface.

```python
class LanguageModel:
    def chat_completion(self, message, max_tokens, ...): ...
    def embedding(self, text, ...): ...
```

This polymorphism allows your main algorithm to just accept a `LanguageModel` instance and work regardless of the backend.

## 2. Remote APIs (`OpenAIAPI`)

**Implementation Idea**:
A lightweight wrapper around the official `openai` Python client. It handles environment variable resolution (`OPENAI_API_KEY`, `OPENAI_BASE_URL`) and standardizes the input/output formats to match the `LanguageModel` interface.

```python
from adtools.lm import OpenAIAPI
llm = OpenAIAPI(model="gpt-4", api_key="sk-...")
```

## 3. Local Inference Servers (`VLLMServer`, `SGLangServer`)

**Implementation Idea: The "Manager" Pattern**

Unlike `OpenAIAPI` which connects to an existing server, `VLLMServer` and `SGLangServer` are designed to **manage** the server process itself. They turn a Python script into a self-contained deployment unit.

### Key Implementation Details:

1.  **Subprocess Launch**:
    When you initialize `VLLMServer`, it constructs a command-line string (invoking `vllm.entrypoints.openai.api_server`) and launches it using `subprocess.Popen`. This runs the heavy inference engine in a separate process, isolated from your main Python logic.

2.  **Health Check & Waiting**:
    The constructor doesn't return immediately. It enters a `_wait_for_server()` loop, polling the server's `/health` endpoint via HTTP. This ensures that when your code proceeds, the model is fully loaded and ready to accept requests.

3.  **Lifecycle Management**:
    These classes implement `close()` (and `__del__`) to robustly terminate the server subprocess. They use `psutil` to kill the entire process tree, ensuring no GPU memory is leaked if your script crashes.

4.  **Dynamic LoRA Support**:
    Both classes provide pythonic wrappers (`load_lora_adapter`, `unload_lora_adapter`) around the underlying server's HTTP management endpoints. This allows you to hot-swap fine-tuned adapters at runtime without restarting the heavy base model.

### Usage Example (vLLM)

```python
from adtools.lm import VLLMServer

# 1. Launch Server (This blocks until the model is loaded)
# It effectively runs: `python -m vllm... --model meta-llama...` in background
llm = VLLMServer(
    model_path="meta-llama/Meta-Llama-3-8B-Instruct",
    port=8000,
    gpus=[0]
)

# 2. Use it (Standard Interface)
print(llm.chat_completion("Hello!"))

# 3. Dynamic LoRA Loading
# Sends a POST request to the vLLM server to load weights
llm.load_lora_adapter("math_adapter", "/path/to/lora/weights")
print(llm.chat_completion("Solve this equation", lora_name="math_adapter"))

# 4. Cleanup
# Kills the background vLLM process and frees GPU memory
llm.close()
```