# Auto-Evaluation Server

The `adtools.evaluator.auto_server` module provides an out-of-the-box HTTP server to deploy your custom evaluator as a web service. This allows you to decouple your evaluation tasks and submit code for evaluation via network requests.

## Core Features

- **HTTP Interface**: Starts an HTTP server that accepts code via `POST` requests.
- **Concurrency Control**: Uses a semaphore to limit the number of concurrent evaluation tasks, preventing the server from being overloaded.
- **Dynamic Loading**: Automatically loads your `PyEvaluator` or `PyEvaluatorRay` subclass from a specified Python file at startup.
- **Client Functions**: Provides `submit_code` and `submit_code_async` functions for easy client-side code submission.

## How to Use

The workflow is divided into two parts: starting the server and submitting code from a client.

### 1. Start the Evaluation Server

You can start the server using the `launch_auto_eval_server.sh` script or directly with the `python -m` command. You need to specify the path to the file containing your evaluator class.

#### Important Constraint for Evaluator File

The Python file provided via the `-d` or `--dir` argument **must contain exactly one public class** (a class whose name does not start with an underscore `_`) that inherits from either `PyEvaluator` or `PyEvaluatorRay`.

Any other classes within the file intended for internal use or not meant to be loaded as the primary evaluator **must be made private** by prefixing their names with an underscore (e.g., `_MyHelperClass`). This is because the `auto_server` identifies and loads the single public class as the main evaluator.

#### Example Launch Script

Assume your evaluator, `SortAlgorithmEvaluator`, is defined in `usage/example_evaluator.py`.

`usage/launch_auto_eval_server.sh`:
```bash
cd ../
python -m adtools.evaluator.auto_server \
    -d usage/example_evaluator.py \
    --host 0.0.0.0 \
    --port 8000 \
    -t 10 \
    --max-workers 4
```

**Argument Descriptions**:
- `-d` or `--dir`: Path to the Python file containing your evaluator class.
- `--host`: Host address for the server to bind to.
- `--port`: Port for the server to listen on.
- `-t` or `--timeout`: Default timeout in seconds for each evaluation.
- `--max-workers`: Maximum number of concurrent evaluation tasks allowed.

Once started, the server will automatically find the file's single public `PyEvaluator` subclass, instantiate it, and prepare to receive evaluation requests.

### 2. Submit Code from a Client

The `auto_server` module also provides the `submit_code` function, allowing you to easily send code to the server for evaluation from anywhere.

#### Client Example

`usage/example_eval_client.py`:
```python
from adtools.evaluator.auto_server import submit_code

# Correct code
code_correct = """
def merge_sort(arr):
    # ... (implementation omitted)
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid]); right = merge_sort(arr[mid:])
    # ... (merge logic omitted)
"""

print("Sending request with correct code...")
response = submit_code(host="0.0.0.0", port=8000, code=code_correct, timeout=10)
print("Server Response:", response)
# Expected Response: {'result': 0.0001, 'evaluate_time': 0.01, 'error_msg': ''}

# Code with an infinite loop
code_loop = """
def merge_sort(arr):
    while True: pass
"""
print("\nSending request with infinite loop (should time out)...")
response = submit_code(host="0.0.0.0", port=8000, code=code_loop, timeout=5)
print("Server Response:", response)
# Expected Response: {'result': None, 'evaluate_time': 5.0, 'error_msg': 'Evaluation timeout.'}
```

The `submit_code` function sends a POST request to the server's `/` endpoint with a JSON body containing `code` and `timeout` fields. The server completes the evaluation and returns a JSON response with `result`, `evaluate_time`, and `error_msg`.

An asynchronous version, `submit_code_async`, is also available for use in `asyncio` environments.
