# 沙盒实现分析

`adtools.sandbox` 模块确保安全执行。本文档分析了进程管理和数据传输的底层实现。

## 1. `SandboxExecutor`：进程隔离

核心挑战是在单独的进程中执行方法并高效地取回结果。

### 1.1 安全执行流程

```python
# adtools/sandbox/sandbox_executor.py

def secure_execute(self, ...):
    # 1. 创建通信通道 (Queue)
    meta_queue = multiprocessing.Queue()
    
    # 2. 为共享内存创建唯一名称
    unique_shm_name = f"psm_{uuid.uuid4().hex[:8]}"

    # 3. 启动子进程
    process = multiprocessing.Process(
        target=self._execute_and_put_res_in_shared_memory,
        args=(..., meta_queue, unique_shm_name),
    )
    process.start()

    # 4. 带超时的等待结果
    try:
        # 阻塞直到超时
        meta = meta_queue.get(timeout=timeout_seconds)
    except Empty:
        # 发生超时！
        return ExecutionResults(result=None, error_msg="Evaluation timeout.")
    finally:
        # 5. 无论结果如何都进行清理
        self._kill_process_and_its_children(process)
        # 解除链接共享内存
        # ...
```

**实现洞察**：
- 我们将 `unique_shm_name` 传递给子进程。子进程*创建*共享内存，但父进程指定名称。这允许父进程在子进程崩溃或在清理自身之前被杀死时，能够清理（unlink）共享内存。

### 1.2 子进程逻辑

此方法在隔离的进程中运行。

```python
# adtools/sandbox/sandbox_executor.py

def _execute_and_put_res_in_shared_memory(self, ..., shm_name_id):
    # 1. 输出重定向
    if redirect_to_devnull:
        _redirect_to_devnull()  # os.dup2(devnull, sys.stdout)

    try:
        # 2. 执行用户代码
        res = method_to_call(*args, **kwargs)

        # 3. 序列化
        data = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)

        # 4. 写入共享内存
        # create=True 告诉操作系统分配内存
        shm = shared_memory.SharedMemory(create=True, name=shm_name_id, size=len(data))
        
        # 从 resource_tracker 注销以防止 Python 抱怨
        # 当此进程退出但内存仍由父进程持有时。
        resource_tracker.unregister(name=shm._name, rtype="shared_memory")

        # 复制字节
        shm.buf[:len(data)] = data
        
        # 5. 通知父进程
        # 我们发送 (True, size) 以便父进程知道要读取多少字节
        meta_queue.put((True, len(data)))
        
    except:
        # 发回异常字符串
        meta_queue.put((False, str(traceback.format_exc())))
```

**实现洞察**：
- **资源追踪器（Resource Tracker）Hack**：Python 的 `multiprocessing` 有一个 `resource_tracker`，用于清理泄漏的共享内存。但是，由于我们希望*父进程*在*子进程*退出后读取内存，我们必须在子进程中显式 `unregister` 它；否则，Python 可能会在子进程结束时立即删除内存块，导致父进程中出现段错误或 FileNotFoundError。

### 1.3 进程清理

如果用户代码生成了自己的子进程（僵尸进程），仅仅调用 `process.terminate()` 是不够的。

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

**实现洞察**：
- 我们使用 `psutil` 检查进程树并递归杀死所有后代。这防止了“孤儿”进程在服务器上堆积。

## 2. `SandboxExecutorRay`：分布式执行

此实现用 Ray Actors 替换了 `multiprocessing`。

### 2.1 Worker Actor

```python
# adtools/sandbox/sandbox_executor_ray.py

class _RayWorker:
    def __init__(self, evaluate_worker):
        self.evaluate_worker = evaluate_worker

    def execute(self, method_name, args, kwargs, ...):
        # 简单地调用方法。
        # Ray 自动处理序列化和返回值。
        return getattr(self.evaluate_worker, method_name)(*args, **kwargs)
```

### 2.2 带超时的执行

```python
# adtools/sandbox/sandbox_executor_ray.py

def secure_execute(self, ...):
    # 创建远程 worker
    worker = self._RemoteWorkerClass.remote(self.evaluate_worker)

    try:
        # 调度执行
        future = worker.execute.remote(...)
        
        # 带超时的等待结果
        result = ray.get(future, timeout=timeout_seconds)
        
    except GetTimeoutError:
        # 处理超时
        return ExecutionResults(..., error_msg="Evaluation timeout.")
        
    finally:
        # 关键：杀死 actor 以释放资源（GPU/内存）
        ray.kill(worker, no_restart=True)
```

**实现洞察**：
- 与基于进程的执行器不同，我们不需要手动管理共享内存。Ray 的对象存储（"Plasma"）自动处理支持对象的零拷贝传输。
- `ray.kill` 相当于 `process.terminate()`。

## 3. 装饰器实现 (`@sandbox_run`)

装饰器是一个语法糖，用于决定实例化哪个执行器。

```python
# adtools/sandbox/decorators.py

def sandbox_run(sandbox_type="process", ...):
    def decorator(func):
        # 1. 检测它是方法还是函数
        is_class_method = "." in func.__qualname__ 

        @wraps(func)
        def wrapper(*args, **kwargs):
            if is_class_method:
                self_instance = args[0]
                # 2. 防止无限递归
                # 如果我们已经在沙盒内（被绕过），则只运行函数
                if getattr(self_instance, "_bypass_sandbox", False):
                    return func(*args, **kwargs)
                
                # 3. 为 worker 创建副本
                evaluate_worker = copy.copy(self_instance)
                evaluate_worker._bypass_sandbox = True
            else:
                # 包装独立函数
                evaluate_worker = _FunctionWorker(func)

            # 4. 实例化执行器并运行
            executor = SandboxExecutor(evaluate_worker, ...)
            return executor.secure_execute(...)
            
        return wrapper
    return decorator
```

**实现洞察**：
- **递归预防**：当 `secure_execute` 运行时，它在新进程中反序列化 `evaluate_worker` 并再次调用该方法。我们在副本上设置 `_bypass_sandbox = True`，以便在子进程内，装饰器看到该标志并执行*真实*代码，而不是尝试生成*另一个*沙盒（这将导致无限的进程创建循环）。
