# Evaluator Implementation Analysis

The `adtools.evaluator` module acts as the bridge between raw string code and the sandbox execution environment.

## 1. `PyEvaluator`: The Compilation Pipeline

The base class `PyEvaluator` defines the template for converting string code into executable Python objects.

### 1.1 `_exec_and_get_res`

This is the internal method that runs **inside the sandbox**.

```python
# adtools/evaluator/py_evaluator.py

def _exec_and_get_res(self, program: str | PyProgram, **kwargs):
    # 1. Parse program to find names
    if isinstance(program, str):
        program = PyProgram.from_text(program)
    function_names = [f.name for f in program.functions]
    
    # 2. Execute Code (Compilation)
    if self.exec_code:
        all_globals_namespace = {}
        # This executes the script, defining functions/classes in the dict
        exec(str(program), all_globals_namespace)
        
        # 3. Extract Callables
        callable_funcs_dict = {
            name: all_globals_namespace[name] 
            for name in function_names
        }
    else:
        callable_funcs_dict = None

    # 4. Delegate to User Logic
    res = self.evaluate_program(
        str(program),
        callable_functions_dict=callable_funcs_dict,
        # ... other args
        **kwargs
    )
    return res
```

**Implementation Insight**:
- **Why parse first?** We parse the code to know exactly *which* functions and classes are defined in it (`function_names`). This allows us to selectively extract them from the `all_globals_namespace` after `exec()`.
- **Dependency Injection**: The `evaluate_program` method (implemented by the user) receives the ready-to-use `callable_functions_dict`. This means the user's code doesn't need to do `exec` or parsing; it simply retrieves `func = dict['my_func']` and calls `func()`.

### 1.2 `secure_evaluate`

This is the public method called by the main application.

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

**Implementation Insight**:
- It doesn't run `_exec_and_get_res` directly.
- Instead, it tells the `sandbox_executor` to run `_exec_and_get_res` in the child process.
- The `program` string is pickled and sent to the child.
- The child runs `exec()`, then runs `evaluate_program()`, then pickles the result back.

## 2. `PyEvaluatorRay`: Distributed Variant

Inherits from `PyEvaluator` but initializes a `SandboxExecutorRay` instead of the standard one.

```python
# adtools/evaluator/py_evaluator_ray.py

class PyEvaluatorRay(PyEvaluator):
    def __init__(self, ...):
        # ...
        self.sandbox_executor = SandboxExecutorRay(...)
```

**Implementation Insight**:
- Since it shares the same `_exec_and_get_res` method from the parent class, the compilation logic is identical. The only difference is *where* that method runs (Ray Actor vs. Subprocess).
