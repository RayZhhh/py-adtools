# Useful tools for parsing and evaluating Python programs for algorithm design/code optimization

> This repo aims to help develop more
> powerful [Large Language Models for Algorithm Design (LLM4AD)](https://github.com/Optima-CityU/llm4ad) applications.
>
> More tools will be provided soon.

------

The figure demonstrates how a Python program is parsed
into [PyCodeBlock](./adtools/py_code.py#L19-L30), [PyFunction](./adtools/py_code.py#L34-L96), [PyClass](./adtools/py_code.py#L100-L172),
and [PyProgram](./adtools/py_code.py#L176-L203) via `adtools`.

![pycode](./assets/pycode.png)

------

## Installation

> [!TIP]
>
> It is recommended to use Python >= 3.10.

Run the following instructions to install adtools.

```shell
pip install git+https://github.com/RayZhhh/py-adtools.git
```

Or install via pip:

```shell
pip install py-adtools
```

## 1 Code Parsing with [py_code](./algolm/py-adtools/adtools/py_code.py#L0-L518)

[adtools.py_code](./adtools/py_code.py#L0-L518) provides robust parsing of Python programs into structured components
that can be easily manipulated, modified, and analyzed.

### Core Components

The parser decomposes Python code into four main data structures:

| **Component**   | **Description**                                                                                                              | **Key Attributes**                                              |
|-----------------|------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| **PyProgram**   | Represents the entire file. It maintains the exact sequence of scripts, functions, and classes.                              | `functions`, `classes`, `scripts`, `elements`                   |
| **PyFunction**  | Represents a top-level function or a class method. You can modify its signature, decorators, docstring, or body dynamically. | `name`, `args`, `body`, `docstring`, `decorator`, `return_type` |
| **PyClass**     | Represents a class definition. It serves as a container for methods and class-level statements.                              | `name`, `bases`, `functions` (methods), `body`                  |
| **PyCodeBlock** | Represents raw code segments, such as imports, global variables, or specific logic blocks inside classes.                    | `code`                                                          |

### Basic Usage

```python
from adtools import PyProgram

code = r"""
import ast, numba                 # This part will be parsed into PyCodeBlock
import numpy as np

@numba.jit()                      # This part will be parsed into PyFunction
def function(arg1, arg2=True):     
    if arg2:
    	return arg1 * 2
    else:
    	return arg1 * 4

@some.decorators()                # This part will be parsed into PyClass
class PythonClass(BaseClass):
    
    class_var1 = 1                # This part will be parsed into PyCodeBlock
    class_var2 = 2                # and placed in PyClass.class_vars_and_code

    def __init__(self, x):        # This part will be parsed into PyFunction
        self.x = x                # and placed in PyClass.functions

    def method1(self):
        return self.x * 10

    @some.decorators()
    def method2(self, x, y):
    	return x + y + self.method1(x)

    class InnerClass:             # This part will be parsed into PyCodeBlock
    	def __init__(self):       # and placed in PyClass.class_vars_and_code
    		...

if __name__ == '__main__':        # This part will be parsed into PyCodeBlock
	res = function(1)
	print(res)
	res = PythonClass().method2(1, 2)
"""

p = PyProgram.from_text(code)
print(p)
print(f"-------------------------------------")
print(p.classes[0].functions[2].decorator)
print(f"-------------------------------------")
print(p.functions[0].name)

```

### Key Features

- **Preserves Code Structure**: Maintains original indentation and formatting
- **Handles Multiline Strings**: Properly preserves multiline string content without incorrect indentation
- **Access to Components**: Easily access functions, classes, and code blocks
- **Modify Code Elements**: Change function names, docstrings, or body content programmatically
- **Complete Program Representation**: [PyProgram](./adtools/py_code.py#L176-L203) maintains the exact sequence of
  elements as they appear in the source code

## 2 Code Evaluation with `evaluator`

`adtools.evaluator` provides multiple secure evaluation options for running and testing Python code.

### Basic Usage

```python
import time
from typing import Dict, Callable, List, Any

from adtools import PyEvaluator


class SortAlgorithmEvaluator(PyEvaluator):
    def evaluate_program(
        self,
        program_str: str,
        callable_functions_dict: Dict[str, Callable] | None,
        callable_functions_list: List[Callable] | None,
        callable_classes_dict: Dict[str, Callable] | None,
        callable_classes_list: List[Callable] | None,
        **kwargs,
    ) -> Any | None:
        """Evaluate a given sort algorithm program.
        Args:
            program_str            : The raw program text.
            callable_functions_dict: A dict maps function name to callable function.
            callable_functions_list: A list of callable functions.
            callable_classes_dict  : A dict maps class name to callable class.
            callable_classes_list  : A list of callable classes.
        Return:
            Returns the evaluation result.
        """
        # Get the sort algorithm
        sort_algo: Callable = callable_functions_dict["merge_sort"]
        # Test data
        input = [10, 2, 4, 76, 19, 29, 3, 5, 1]
        # Compute execution time
        start = time.time()
        res = sort_algo(input)
        duration = time.time() - start
        if res == sorted(input):  # If the result is correct
            return duration  # Return the execution time as the score of the algorithm
        else:
            return None  # Return None as the algorithm is incorrect


code_generated_by_llm = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2              
    left = merge_sort(arr[:mid])     
    right = merge_sort(arr[mid:])   

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result
"""

harmful_code_generated_by_llm = """
def merge_sort(arr):
    print('I am harmful')  # There will be no output since we redirect STDOUT to /dev/null by default.
    while True:
        pass
"""

if __name__ == "__main__":
    evaluator = SortAlgorithmEvaluator()

    # Evaluate
    score = evaluator._exec_and_get_res(code_generated_by_llm)
    print(f"Score: {score}")

    # Secure evaluate (the evaluation is executed in a sandbox process)
    score = evaluator.secure_evaluate(code_generated_by_llm, timeout_seconds=10)
    print(f"Score: {score}")

    # Evaluate a harmful code, the evaluation will be terminated within 10 seconds
    # We will obtain a score of `None` due to the violation of time restriction
    score = evaluator.secure_evaluate(harmful_code_generated_by_llm, timeout_seconds=10)
    print(f"Score: {score}")

```

### Evaluator Types and Their Characteristics

`adtools` provides four different evaluator implementations, each optimized for different scenarios:

- **[PyEvaluator](./adtools/evaluator/py_evaluator.py#L47-L270) (Recommend)**
    - *Basic evaluator* that executes code directly in the current process
    - *Provides process isolation* with timeout capabilities
    - *Best for trusted code* with samll return objects (e.g., int, float)
    - *Use case*: Evaluating heuristics with small return objects

- **[PyEvaluatorReturnInManagerDict](./adtools/evaluator/py_evaluator.py#L273-L431)**
    - *Uses Manager().dict()* to handle large return objects
    - *Provides process isolation* with timeout capabilities
    - *Ideal for medium-sized results* where pickle serialization is acceptable
    - *Use case*: Evaluating code that returns moderately large data structures

- **[PyEvaluatorReturnInSharedMemory](./adtools/evaluator/py_evaluator.py#L434-L632) (Recommend)**
    - *Uses shared memory* for extremely large return objects (e.g., large tensors)
    - *Avoids pickle serialization overhead* for massive data
    - *Best for high-performance scenarios* with very large result objects
    - *Use case*: Evaluating ML algorithms that produce large tensors or arrays

- **[PyEvaluatorRay](./adtools/evaluator/py_evaluator_ray.py#L15-L136) (Recommend)**
    - *Leverages Ray* for distributed, secure evaluation
    - *Supports zero-copy return* of large objects
    - *Ideal for cluster environments* and when maximum isolation is required
    - *Use case*: Large-scale evaluation across multiple machines or when using GPU resources

All evaluators share the same interface through the abstract [PyEvaluator](./adtools/evaluator/py_evaluator.py#L47-L270)
class, making it easy to switch between implementations based on your specific needs.

## 3 Practical Applications

### 3.1 Parser for Code Manipulation

The parser is designed to handle complex scenarios, including **multiline strings**, **decorators**, and **indentation
management**.

```python
from adtools import PyProgram

# A complex piece of code with imports, decorators, and a class
code = r'''
import numpy as np

@jit(nopython=True)
def heuristics(x):
    """Calculates the heuristic value."""
    return x * 0.5

class EvolutionStrategy:
    population_size = 100
    
    def __init__(self, mu, lambda_):
        self.mu = mu
        self.lambda_ = lambda_
        
    def mutate(self, individual):
        # Apply mutation
        return individual + np.random.normal(0, 1)
'''

# 1. Parse the program
program = PyProgram.from_text(code)

# 2. Access and Modify Functions
func = program.functions[0]
print(f"Function detected: {func.name}")
# Output: Function detected: heuristics

# Modify the function programmatically
func.name = "fast_heuristics"
func.decorator = None  # Remove decorator
func.docstring = "Optimized heuristic calculation."

# 3. Access Class Methods
cls_obj = program.classes[0]
init_method = cls_obj.functions[0]
mutate_method = cls_obj.functions[1]

print(f"Class: {cls_obj.name}, Method: {mutate_method.name}")
# Output: Class: EvolutionStrategy, Method: mutate

# 4. Generate the modified code
# The PyProgram object reconstructs the code preserving the original order
print("\n--- Reconstructed Code ---")
print(program)

```

### 3.2 Parser for Prompt Construction

`adtools` is particularly powerful for LLM-based algorithm design, where you need to manage populations of generated
code, standardize formats for prompts, or inject generated logic into existing templates.

In LLM-based Automated Algorithm Design (LLM-AAD), you often maintain a population of algorithms. You may need to rename
them (e.g., `v1`, `v2`), standardize their docstrings for the context, or remove docstrings to save token costs before
feeding them back into the LLM.

```python
from adtools import PyFunction

# Assume LLM generated two variants of a crossover algorithm
llm_output_1 = '''
def crossover(p1, p2):
    """Single point crossover."""
    point = len(p1) // 2
    return p1[:point] + p2[point:], p2[:point] + p1[point:]
'''

llm_output_2 = """
def crossover_op(parent_a, parent_b):
    # This is a uniform crossover
    mask = [True, False] * (len(parent_a) // 2)
    return [a if m else b for a, b, m in zip(parent_a, parent_b, mask)]
"""

# Parse the functions
func_v1 = PyFunction.extract_first_function_from_text(llm_output_1)
func_v2 = PyFunction.extract_first_function_from_text(llm_output_2)

# --- Modification Logic ---

# 1. Standardize Naming: Rename to v1 and v2
func_v1.name = "crossover_v1"
func_v2.name = "crossover_v2"

# 2. Docstring Management:
# For v1: Enforce a specific docstring format for the prompt
func_v1.docstring = "Variant 1: Implementation of Single Point Crossover."

# For v2: Remove docstring entirely (e.g., to reduce context window usage)
func_v2.docstring = None

# --- Construct Prompt ---

prompt = "Here are the two crossover algorithms currently in the population:\n\n"
prompt += str(func_v1) + "\n"
prompt += str(func_v2) + "\n"
prompt += "Please generate a v3 that combines the best features of both."

print(prompt)

```

**Output:**

```text
Here are the two crossover algorithms currently in the population:

def crossover_v1(p1, p2):
    """Variant 1: Implementation of Single Point Crossover."""
    point = len(p1) // 2
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def crossover_v2(parent_a, parent_b):
    # This is a uniform crossover
    mask = [True, False] * (len(parent_a) // 2)
    return [a if m else b for a, b, m in zip(parent_a, parent_b, mask)]

Please generate a v3 that combines the best features of both.
```

### 3.3 Secure Code Evaluation using Evaluators

When evaluating code generated by LLMs, safety and reliability are critical:

```python
import time
from adtools import PyEvaluatorReturnInSharedMemory
from typing import Dict, Callable, List


class AlgorithmValidator(PyEvaluatorReturnInSharedMemory):
    def evaluate_program(
        self,
        program_str: str,
        callable_functions_dict: Dict[str, Callable] | None,
        callable_functions_list: List[Callable] | None,
        callable_classes_dict: Dict[str, Callable] | None,
        callable_classes_list: List[Callable] | None,
        **kwargs
    ) -> dict:
        results = {"correct": 0, "total": 0, "time": 0}

        try:
            # Get the sorting function
            sort_func = callable_functions_dict.get("sort_algorithm")
            if not sort_func:
                return {**results, "error": "Missing required function"}

            # Test with multiple inputs
            test_cases = [
                [5, 3, 1, 4, 2],
                [1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1],
                list(range(100)),  # Large test case
                [],
            ]

            for case in test_cases:
                start = time.time()
                result = sort_func(
                    case[:]
                )  # Pass a copy to avoid in-place modification
                duration = time.time() - start

                results["total"] += 1
                if result == sorted(case):
                    results["correct"] += 1
                results["time"] += duration

        except Exception as e:
            results["error"] = str(e)

        return results


# Example usage with potentially problematic code
problematic_code = """
def sort_algorithm(arr):
    # This implementation has a bug for empty arrays
    if not arr:
        return []  # Missing this case would cause failure
        
    # Implementation with potential infinite loop
    i = 0
    while i < len(arr) - 1:
        if arr[i] > arr[i+1]:
            arr[i], arr[i+1] = arr[i+1], arr[i]
            i = 0  # Reset to beginning after swap
        else:
            i += 1
    return arr
"""

malicious_code = """
def sort_algorithm(arr):
    import time
    time.sleep(15)  # Exceeds timeout
    return sorted(arr)
"""

validator = AlgorithmValidator()
print(validator.secure_evaluate(problematic_code, timeout_seconds=5))
print(validator.secure_evaluate(malicious_code, timeout_seconds=5))

```

This demonstrates how `adtools` handles:

- **Timeout protection**: Malicious code with infinite loops is terminated
- **Error isolation**: Exceptions in evaluated code don't crash your main process
- **Output redirection**: Prevents unwanted print statements from cluttering your console
- **Resource management**: Proper cleanup of processes and shared resources

The evaluation framework ensures that even if the code contains errors, infinite loops, or attempts to access system
resources, your main application remains safe and responsive.

## 4 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

## 5 Contact & Feedback

If you have any questions, encounter bugs, or have suggestions for improvement, please feel free to [open an issue](https://github.com/RayZhhh/py-adtools/issues) or contact us. Your contributions and feedback are highly appreciated!