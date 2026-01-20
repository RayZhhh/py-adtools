# Code Evaluation Framework

The `adtools.evaluator` module builds upon the sandbox layer to provide a structured workflow for testing and scoring code. It abstracts away the complexity of "compiling" string code into executable functions and managing the sandbox lifecycle.

## The Architecture: `PyEvaluator`

The core class is `PyEvaluator`. It implements the **Template Method** design pattern to separate the *execution mechanics* from the *evaluation logic*.

### 1. The Execution Flow

When you call `evaluator.secure_evaluate(code_str)`, the following pipeline executes:

1.  **Sandbox Entry**: A new process (or Ray actor) is spawned.
2.  **Compilation (`_exec_and_get_res`)**:
    *   The raw `code_str` is parsed using `PyProgram`.
    *   The code is executed via Python's `exec()` command within a restricted local namespace.
    *   The framework extracts all `callable` objects (functions, classes) from this namespace.
3.  **User Logic (`evaluate_program`)**:
    *   The framework calls your custom implementation of `evaluate_program`.
    *   **Crucially**, it passes the *executable function objects*, not just the source text. You don't need to manually `exec` or parse strings; you just call `func(input)`.
4.  **Result Return**: The result from your method is serialized and returned to the main process.

### 2. Why this Design?

- **Safety**: The potentially dangerous `exec()` happens strictly inside the sandbox.
- **Convenience**: Your evaluation logic deals with standard Python objects. You can write `my_algo(data)` instead of messing with `subprocess.run` or string parsing.
- **Flexibility**: You can inject dependencies (datasets, helper functions) into the `evaluate_program` method.

## Implementing a Custom Evaluator

You must inherit from `PyEvaluator` (or `PyEvaluatorRay`) and implement `evaluate_program`.

```python
from adtools.evaluator import PyEvaluator

class MathEvaluator(PyEvaluator):
    def evaluate_program(
        self,
        program_str: str,
        callable_functions_dict: dict,
        **kwargs
    ):
        # 1. Retrieve the function to test
        # The framework has already 'exec'ed the code and populated this dict
        func = callable_functions_dict.get("square")
        if not func:
            return 0.0 # Fail if function missing

        # 2. Run your test logic
        try:
            result = func(10)
            return 1.0 if result == 100 else 0.0
        except Exception:
            return 0.0
```

## Evaluator Variants

### `PyEvaluator` (Process-Based)
Uses `SandboxExecutor`. It is the default choice for most tasks.
- **Pros**: Lightweight, no external dependencies (like Ray).
- **Cons**: Data transfer (pickling) has some overhead for massive objects.

### `PyEvaluatorRay` (Ray-Based)
Uses `SandboxExecutorRay`.
- **Pros**: Zero-copy data transfer, distributed execution, easy GPU access.
- **Cons**: Requires installing and initializing Ray.

## Secure Evaluation

Always use `secure_evaluate` in production or when handling LLM code.

```python
evaluator = MathEvaluator()

# secure_evaluate handles the process creation, timeout, and cleanup
results = evaluator.secure_evaluate(
    "def square(x): return x*x",
    timeout_seconds=2.0
)

print(f"Result: {results['result']}")
print(f"Time: {results['evaluate_time']}")
```