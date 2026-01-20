# 评估器实现分析

`adtools.evaluator` 模块充当原始字符串代码和沙盒执行环境之间的桥梁。

## 1. `PyEvaluator`：编译管道

基类 `PyEvaluator` 定义了将字符串代码转换为可执行 Python 对象的模板。

### 1.1 `_exec_and_get_res`

这是在**沙盒内部**运行的内部方法。

```python
# adtools/evaluator/py_evaluator.py

def _exec_and_get_res(self, program: str | PyProgram, **kwargs):
    # 1. 解析程序以查找名称
    if isinstance(program, str):
        program = PyProgram.from_text(program)
    function_names = [f.name for f in program.functions]
    
    # 2. 执行代码（编译）
    if self.exec_code:
        all_globals_namespace = {}
        # 这将执行脚本，在字典中定义函数/类
        exec(str(program), all_globals_namespace)
        
        # 3. 提取可调用对象
        callable_funcs_dict = {
            name: all_globals_namespace[name] 
            for name in function_names
        }
    else:
        callable_funcs_dict = None

    # 4. 委托给用户逻辑
    res = self.evaluate_program(
        str(program),
        callable_functions_dict=callable_funcs_dict,
        # ... 其他参数
        **kwargs
    )
    return res
```

**实现洞察**：
- **为什么要先解析？** 我们解析代码是为了确切知道其中定义了*哪些*函数和类（`function_names`）。这允许我们在 `exec()` 后有选择地从 `all_globals_namespace` 中提取它们。
- **依赖注入**：`evaluate_program` 方法（由用户实现）接收现成的 `callable_functions_dict`。这意味着用户的代码不需要执行 `exec` 或解析；它只需检索 `func = dict['my_func']` 并调用 `func()`。

### 1.2 `secure_evaluate`

这是主应用程序调用的公共方法。

```python
# adtools/evaluator/py_evaluator.py

def secure_evaluate(self, program, timeout_seconds=None, ...):
    return self.sandbox_executor.secure_execute(
        worker_execute_method_name="_exec_and_get_res",
        method_args=[program],
        method_kwargs=kwargs,
        timeout_seconds=timeout_seconds,
        # ...
    )
```

**实现洞察**：
- 它不直接运行 `_exec_and_get_res`。
- 相反，它告诉 `sandbox_executor` 在子进程中运行 `_exec_and_get_res`。
- `program` 字符串被 pickle 并发送给子进程。
- 子进程运行 `exec()`，然后运行 `evaluate_program()`，然后将结果 pickle 返回。

## 2. `PyEvaluatorRay`：分布式变体

继承自 `PyEvaluator` 但初始化 `SandboxExecutorRay` 而不是标准的执行器。

```python
# adtools/evaluator/py_evaluator_ray.py

class PyEvaluatorRay(PyEvaluator):
    def __init__(self, ...):
        # ...
        self.sandbox_executor = SandboxExecutorRay(...)
```

**实现洞察**：
- 由于它共享父类的相同 `_exec_and_get_res` 方法，因此编译逻辑完全相同。唯一的区别是该方法运行的*位置*（Ray Actor 与子进程）。
