# Safe Execution (Sandbox)

The `adtools.sandbox` module provides the mechanism for safely executing untrusted code. It isolates the execution environment to prevent crashes, infinite loops, or malicious interference from affecting the main application.

## 1. Process-Based Sandbox (`SandboxExecutor`)

**Implementation Idea**:
The `SandboxExecutor` relies on Python's `multiprocessing` module to run code in a separate operating system process. The challenge in multiprocess execution is robustly handling return values (especially large ones) and timeouts.

### Mechanism: Shared Memory & Queues

Instead of passing large results back through a simple `Queue` (which involves slow pickling/unpickling and pipe size limits), `SandboxExecutor` uses a hybrid approach:

1.  **Shared Memory**: The child process allocates a `multiprocessing.shared_memory.SharedMemory` block. It serializes the result using `pickle` and writes the raw bytes directly into this memory block.
2.  **Metadata Queue**: The child process sends a small tuple `(success, size_or_error)` to the parent via a `multiprocessing.Queue`.
3.  **Zero-Copy Retrieval**: The parent reads the metadata, attaches to the specified shared memory block, and reconstructs the object.

This design minimizes overhead for large data structures (like matrices or large lists).

### Timeout & Process Management

- **Timeout**: The parent process waits on the `Queue.get()` with a `timeout` argument. If it times out, the parent immediately proceeds to terminate the child.
- **Cleanup**: To ensure no zombie processes are left behind (e.g., if the user code spawns its own subprocesses), the executor uses `psutil` to find and kill the entire process tree of the child process.

```python
from adtools.sandbox import SandboxExecutor

# Wrap your worker object
executor = SandboxExecutor(my_worker_instance)

# Execute 'my_method' in a separate process with a 5s timeout
result = executor.secure_execute(
    "my_method",
    method_args=(10, 20),
    timeout_seconds=5.0
)
```

## 2. Ray-Based Sandbox (`SandboxExecutorRay`)

**Implementation Idea**:
For distributed or cluster-based environments, `SandboxExecutorRay` leverages the [Ray](https://github.com/ray-project/ray) framework. Ray Actors provide a more natural isolation boundary than raw processes and offer advanced features like GPU assignment.

### Mechanism: Actors & Object Store

1.  **Actor Creation**: The executor creates a Ray Actor (`@ray.remote` class) that wraps your worker object. This actor runs in its own process (potentially on a different node).
2.  **Execution**: The method call is dispatched asynchronously using `actor.method.remote()`.
3.  **Result Retrieval**: The parent waits for the result using `ray.get(future, timeout=...)`. Ray's Plasma Object Store handles the data transfer, offering **zero-copy** reads for large NumPy arrays and Tensors.

### Resource Management

Ray automatically manages the lifecycle of actors. If `timeout_seconds` is exceeded during `ray.get`, the `SandboxExecutorRay` catches the `GetTimeoutError` and forcibly kills the actor using `ray.kill(worker, no_restart=True)`.

```python
from adtools.sandbox import SandboxExecutorRay

# Automatically initializes Ray if needed
executor = SandboxExecutorRay(my_worker_instance, init_ray=True)

# Execute remotely
result = executor.secure_execute("my_method", timeout_seconds=5.0)
```

## 3. Decorators (`@sandbox_run`)

**Implementation Idea**:
The `@sandbox_run` decorator is a syntactic sugar that automates the creation of an executor. It inspects the decorated function:
- **For Methods**: It detects `self`. It creates a shallow copy of the instance to pass to the sandbox (preventing race conditions on the original object state).
- **For Functions**: It wraps the function in a temporary worker class.

It allows you to seamlessly "offload" a specific function to a sandbox without refactoring your entire class structure.

```python
from adtools.sandbox import sandbox_run

@sandbox_run(timeout=2.0)
def risky_calculation(x):
    import time
    time.sleep(1)
    return x * x

# Calling this automatically spawns a process, runs the code, and returns the result
res = risky_calculation(5)
print(res['result'])  # 25
```
