# Sandbox Implementation Analysis

The `adtools.sandbox` module ensures safe execution. This document analyzes the low-level implementation of process management and data transfer.

## 1. `SandboxExecutor`: Process Isolation

The core challenge is executing a method in a separate process and getting the result back efficiently.

### 1.1 The Secure Execution Flow

```python
# adtools/sandbox/sandbox_executor.py

def secure_execute(self, ...):
    # 1. Create a communication channel (Queue)
    meta_queue = multiprocessing.Queue()
    
    # 2. Create a unique name for Shared Memory
    unique_shm_name = f"psm_{uuid.uuid4().hex[:8]}"

    # 3. Launch the child process
    process = multiprocessing.Process(
        target=self._execute_and_put_res_in_shared_memory,
        args=(..., meta_queue, unique_shm_name),
    )
    process.start()

    # 4. Wait for result with timeout
    try:
        # Blocks here until timeout
        meta = meta_queue.get(timeout=timeout_seconds)
    except Empty:
        # Timeout occurred!
        return ExecutionResults(result=None, error_msg="Evaluation timeout.")
    finally:
        # 5. Cleanup regardless of outcome
        self._kill_process_and_its_children(process)
        # Unlink shared memory
        # ...
```

**Implementation Insight**:
- We pass `unique_shm_name` to the child. The child *creates* the shared memory, but the parent dictates the name. This allows the parent to clean it up (unlink) even if the child crashes or is killed before it can clean up itself.

### 1.2 The Child Process Logic

This method runs inside the isolated process.

```python
# adtools/sandbox/sandbox_executor.py

def _execute_and_put_res_in_shared_memory(self, ..., shm_name_id):
    # 1. Output Redirection
    if redirect_to_devnull:
        _redirect_to_devnull()  # os.dup2(devnull, sys.stdout)

    try:
        # 2. Execute User Code
        res = method_to_call(*args, **kwargs)

        # 3. Serialize
        data = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)

        # 4. Write to Shared Memory
        # create=True tells OS to allocate memory
        shm = shared_memory.SharedMemory(create=True, name=shm_name_id, size=len(data))
        
        # Unregister from resource_tracker to prevent Python from complaining 
        # when this process exits but memory is still held by parent.
        resource_tracker.unregister(name=shm._name, rtype="shared_memory")

        # Copy bytes
        shm.buf[:len(data)] = data
        
        # 5. Notify Parent
        # We send (True, size) so parent knows how many bytes to read
        meta_queue.put((True, len(data)))
        
    except:
        # Send exception string back
        meta_queue.put((False, str(traceback.format_exc())))
```

**Implementation Insight**:
- **Resource Tracker Hack**: Python's `multiprocessing` has a `resource_tracker` that cleans up leaked shared memory. However, since we want the *parent* to read the memory after the *child* exits, we must explicitly `unregister` it in the child; otherwise, Python might delete the memory block as soon as the child process ends, causing a segfault or FileNotFoundError in the parent.

### 1.3 Process Cleanup

Simply calling `process.terminate()` isn't enough if the user code spawned its own subprocesses (zombies).

```python
# adtools/sandbox/sandbox_executor.py

def _kill_process_and_its_children(self, process):
    if self.recur_kill_eval_proc:
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []

    process.terminate()
    # ... join ...

    for child in children:
        child.terminate()
```

**Implementation Insight**:
- We use `psutil` to inspect the process tree and kill all descendants recursively. This prevents "orphan" processes from accumulating on the server.

## 2. `SandboxExecutorRay`: Distributed Execution

This implementation swaps `multiprocessing` for Ray Actors.

### 2.1 Worker Actor

```python
# adtools/sandbox/sandbox_executor_ray.py

class _RayWorker:
    def __init__(self, evaluate_worker):
        self.evaluate_worker = evaluate_worker

    def execute(self, method_name, args, kwargs, ...):
        # Simply calls the method.
        # Ray handles the serialization and return values automatically.
        return getattr(self.evaluate_worker, method_name)(*args, **kwargs)
```

### 2.2 Execution with Timeout

```python
# adtools/sandbox/sandbox_executor_ray.py

def secure_execute(self, ...):
    # Create the remote worker
    worker = self._RemoteWorkerClass.remote(self.evaluate_worker)

    try:
        # Schedule execution
        future = worker.execute.remote(...)
        
        # Wait for result with timeout
        result = ray.get(future, timeout=timeout_seconds)
        
    except GetTimeoutError:
        # Handle timeout
        return ExecutionResults(..., error_msg="Evaluation timeout.")
        
    finally:
        # Crucial: Kill the actor to free resources (GPU/Memory)
        ray.kill(worker, no_restart=True)
```

**Implementation Insight**:
- Unlike the process-based executor, we don't need manual shared memory management. Ray's Object Store ("Plasma") automatically handles zero-copy transfer for supported objects.
- `ray.kill` is the equivalent of `process.terminate()`.

## 3. Decorator Implementation (`@sandbox_run`)

The decorator is a syntax sugar that decides which Executor to instantiate.

```python
# adtools/sandbox/decorators.py

def sandbox_run(sandbox_type="process", ...):
    def decorator(func):
        # 1. Detect if it's a method or function
        is_class_method = "." in func.__qualname__ 

        @wraps(func)
        def wrapper(*args, **kwargs):
            if is_class_method:
                self_instance = args[0]
                # 2. Prevent infinite recursion
                # If we are already inside the sandbox (bypassed), just run the function
                if getattr(self_instance, "_bypass_sandbox", False):
                    return func(*args, **kwargs)
                
                # 3. Create a copy for the worker
                evaluate_worker = copy.copy(self_instance)
                evaluate_worker._bypass_sandbox = True
            else:
                # Wrap standalone function
                evaluate_worker = _FunctionWorker(func)

            # 4. Instantiate Executor and Run
            executor = SandboxExecutor(evaluate_worker, ...)
            return executor.secure_execute(...)
            
        return wrapper
    return decorator
```

**Implementation Insight**:
- **Recursion Prevention**: When `secure_execute` runs, it deserializes the `evaluate_worker` in the new process and calls the method again. We set `_bypass_sandbox = True` on the copy so that inside the child process, the decorator sees the flag and executes the *real* code instead of trying to spawn *another* sandbox (which would lead to an infinite loop of process creation).