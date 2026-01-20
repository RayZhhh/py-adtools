# Auto-Evaluation Server

The `adtools.evaluator.auto_server` module transforms any local `PyEvaluator` script into a scalable, concurrent HTTP web service. This decouples the "Generation" (LLM) from the "Evaluation" (Execution), allowing them to run on different machines or scale independently.

## Architecture & Implementation

The server is built on Python's standard `http.server` but adds critical features for production use.

### 1. Dynamic Evaluator Loading

The server is **generic**. It does not know about your specific evaluator class at compile time. Instead, it uses dynamic inspection to load your code:

1.  **File Parsing**: It reads the Python file provided via `--dir`.
2.  **Class Discovery**: It uses `adtools.py_code.PyClass` to parse the file and find the **single public class** (a class not starting with `_`).
3.  **Validation**: It verifies that this class inherits from `PyEvaluator` or `PyEvaluatorRay`.
4.  **Instantiation**: It imports the module dynamically using `importlib` and creates an instance of your evaluator.

**Why this matters**: You don't need to write boilerplate server code. Just write your evaluator class, and `auto_server` wraps it.

### 2. Concurrency Control (Semaphore)

Since evaluation often involves spawning heavy subprocesses, an unchecked HTTP server could easily crash the machine if too many requests arrive simultaneously.

**Implementation**:
- The server initializes a `threading.Semaphore` with a value equal to `--max-workers`.
- Inside the `do_POST` handler, the thread must **acquire** this semaphore before it can call `evaluator.secure_evaluate`.
- If all slots are busy, new requests wait (blocking the thread) until a slot opens.

### 3. Request Handling Flow

1.  **POST Request**: Client sends a JSON payload `{"code": "...", "timeout": 10}`.
2.  **Handler**: `EvaluationHandler.do_POST` parses the JSON.
3.  **Execution**: It calls `server.evaluator.secure_evaluate(code, ...)` (protected by the semaphore).
4.  **Response**: The `ExecutionResults` dict is serialized to JSON and returned.

## Usage Guide

### 1. Preparing Your Evaluator

Your evaluator file (e.g., `my_eval.py`) must contain exactly **one** public class inheriting from `PyEvaluator`.

```python
# my_eval.py
from adtools.evaluator import PyEvaluator

class MyEvaluator(PyEvaluator):  # <--- This will be loaded
    def evaluate_program(self, ...):
        ...

class _Helper: # <--- This will be ignored (starts with _)
    ...
```

### 2. Launching the Server

```bash
python -m adtools.evaluator.auto_server \
    --dir ./my_eval.py \
    --port 8000 \
    --max-workers 4
```

### 3. Client Submission

We provide helper functions to simplify the HTTP interaction.

```python
from adtools.evaluator.auto_server import submit_code

# Synchronous call
response = submit_code(
    host="localhost",
    port=8000,
    code="def solution(): return 42",
    timeout=5.0
)
print(response)
# Output: {'result': ..., 'evaluate_time': ..., 'error_msg': ...}
```

For async applications (like FastAPI or discord bots), use `submit_code_async`:

```python
from adtools.evaluator.auto_server import submit_code_async

await response = submit_code_async(..., code="...")
```