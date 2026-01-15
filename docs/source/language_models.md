# Language Model Connections

The `adtools.lm` module provides a unified interface for interacting with various Large Language Model (LLM) backends. Whether you're using the OpenAI API, a locally hosted vLLM, or an SGLang server, you can simplify your code with the `LanguageModel` base class and its implementations.

## The `LanguageModel` Base Class

`LanguageModel` is the abstract base class for all LLM interfaces, defining two core methods:

- **`chat_completion(...)`**: For generating chat-based responses.
- **`embedding(...)`**: For generating text embeddings.

This ensures a consistent calling convention across different LLM backends.

## Supported Backends

### 1. `OpenAIAPI`

This is a client for interacting with any server that conforms to the OpenAI API specification, including the official OpenAI service or third-party providers.

#### Usage Example

```python
from adtools.lm import OpenAIAPI

# Reads API Key and Base URL from environment variables
# The environment variables need to be set as:
# export OPENAI_API_KEY="your_api_key"
# export OPENAI_BASE_URL="https://api.openai.com/v1/"

llm = OpenAIAPI(model="gpt-4-turbo")

# Generate text
response = llm.chat_completion("Hello, world!", max_tokens=50)
print(response)

# Generate an embedding
embedding = llm.embedding("This is an example text.")
print(embedding)
```

### 2. `VLLMServer`

The `VLLMServer` class automatically launches and manages a local [vLLM](https://github.com/vllm-project/vllm) server. vLLM is a high-throughput library for LLM inference and serving.

- **Automatic Deployment**: Automatically starts a vLLM server on the specified GPUs upon initialization.
- **Dynamic LoRA Loading**: Supports dynamically loading and unloading LoRA adapters at runtime.
- **Resource Management**: Automatically shuts down the server process when the object is destroyed.

#### Usage Example

```python
from adtools.lm import VLLMServer

# Deploy a Llama-3-8B model on GPU 0
llm = VLLMServer(
    model_path='meta-llama/Meta-Llama-3-8B-Instruct',
    port=30000,
    gpus=0,  # Or [0, 1] for multi-GPU deployment
    max_model_len=8192,
    max_lora_rank=16  # Enable LoRA support
)

# Generate text using the base model
response = llm.chat_completion("Write a Hello World in Python.")
print(response)

# Dynamically load a LoRA adapter
llm.load_lora_adapter("my_adapter", "/path/to/my/lora_adapter")

# Generate text using the LoRA adapter
response_with_lora = llm.chat_completion(
    "Write a Hello World in Python.", 
    lora_name="my_adapter"
)
print(response_with_lora)

# Shut down the server
llm.close()
```

### 3. `SGLangServer`

Similar to `VLLMServer`, the `SGLangServer` class automatically starts and manages a local [SGLang](https://github.com/sgl-project/sglang) server. SGLang is a high-efficiency language designed for large language models.

#### Usage Example

```python
from adtools.lm import SGLangServer

# Deploy on GPU 0
llm = SGLangServer(
    model_path='meta-llama/Meta-Llama-3-8B-Instruct',
    port=30001,
    gpus=0,
    max_lora_rank=16  # Enable LoRA
)

# Generate text
response = llm.chat_completion("Tell me about SGLang.")
print(response)

# Shut down the server
llm.close()
```
