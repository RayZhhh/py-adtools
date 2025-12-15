# Useful tools for parsing and evaluating Python programs for algorithm design/code optimization

> This repo aims to help develop more powerful [Large Language Models for Algorithm Design (LLM4AD)](https://github.com/Optima-CityU/llm4ad) applications. 
>
> More tools will be provided soon.

------

The figure demonstrates how a Python program is parsed into [PyCodeBlock](./adtools/py_code.py#L19-L30), [PyFunction](./adtools/py_code.py#L34-L96), [PyClass](./adtools/py_code.py#L100-L172), and [PyProgram](./adtools/py_code.py#L176-L203) via `adtools`.

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

## 1. Code Parsing with [py_code](./algolm/py-adtools/adtools/py_code.py#L0-L518)

[adtools.py_code](./adtools/py_code.py#L0-L518) provides robust parsing of Python programs into structured components that can be easily manipulated, modified, and analyzed.

### Basic Usage

```python
from adtools import PyProgram

code = r'''
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
'''

p = PyProgram.from_text(code)
print(p)
print(f'-------------------------------------')
print(p.classes[0].functions[2].decorator)
print(f'-------------------------------------')
print(p.functions[0].name)
```

### Key Features

- **Preserves Code Structure**: Maintains original indentation and formatting
- **Handles Multiline Strings**: Properly preserves multiline string content without incorrect indentation
- **Access to Components**: Easily access functions, classes, and code blocks
- **Modify Code Elements**: Change function names, docstrings, or body content programmatically
- **Complete Program Representation**: [PyProgram](./adtools/py_code.py#L176-L203) maintains the exact sequence of elements as they appear in the source code

## 2. Code Evaluation with `evaluator`

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
            **kwargs
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
        sort_algo: Callable = callable_functions_dict['merge_sort']
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


code_generated_by_llm = '''
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
'''

harmful_code_generated_by_llm = '''
def merge_sort(arr):
    print('I am harmful')  # There will be no output since we redirect STDOUT to /dev/null by default.
    while True:
        pass
'''

if __name__ == '__main__':
    evaluator = SortAlgorithmEvaluator()

    # Evaluate
    score = evaluator._exec_and_get_res(code_generated_by_llm)
    print(f'Score: {score}')

    # Secure evaluate (the evaluation is executed in a sandbox process)
    score = evaluator.secure_evaluate(code_generated_by_llm, timeout_seconds=10)
    print(f'Score: {score}')

    # Evaluate a harmful code, the evaluation will be terminated within 10 seconds
    # We will obtain a score of `None` due to the violation of time restriction
    score = evaluator.secure_evaluate(harmful_code_generated_by_llm, timeout_seconds=10)
    print(f'Score: {score}')
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

All evaluators share the same interface through the abstract [PyEvaluator](./adtools/evaluator/py_evaluator.py#L47-L270) class, making it easy to switch between implementations based on your specific needs.

## 3. Practical Applications

### 1. Automatic Algorithm Design

When working with LLMs for algorithm design, you often need to modify generated code to fit specific requirements:

```python
from adtools import PyProgram, PyFunction

# Assume we have code generated by an LLM
llm_generated_code = """
def sort_algorithm(arr):
    '''Sorts an array using a custom algorithm'''
    # Implementation here...
    return arr
"""

# Parse the code
program = PyProgram.from_text(llm_generated_code)
function = program.functions[0]

# Modify function name and docstring to meet requirements
function.name = "merge_sort"
function.docstring = "Efficiently sorts an array using the merge sort algorithm."

# Add proper implementation
function.body = """
if len(arr) <= 1:
    return arr

mid = len(arr) // 2
left = merge_sort(arr[:mid])
right = merge_sort(arr[mid:])

return merge(left, right)
"""

# Create a new function for the merge helper
merge_func = PyFunction(
    name="merge",
    args="left, right",
    body="""
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
""",
    docstring="Merges two sorted arrays into one sorted array."
)

# Add the new function to our program
program.functions.append(merge_func)

# Convert back to string for further processing
modified_code = str(program)
print(modified_code)
```

This approach allows you to:
- Systematically modify function names to match expected interfaces
- Enhance or correct docstrings for better documentation
- Reconstruct code structure while preserving algorithmic content
- Prepare properly formatted code for inclusion in prompts or evaluation

### 2. Secure Code Evaluation

When evaluating code generated by LLMs, safety and reliability are critical:

```python
from adtools import PyEvaluatorReturnInSharedMemory

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
            sort_func = callable_functions_dict.get('sort_algorithm')
            if not sort_func:
                return {**results, "error": "Missing required function"}
                
            # Test with multiple inputs
            test_cases = [
                [5, 3, 1, 4, 2],
                [1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1],
                list(range(100)),  # Large test case
                []
            ]
            
            for case in test_cases:
                start = time.time()
                result = sort_func(case[:])  # Pass a copy to avoid in-place modification
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

The evaluation framework ensures that even if the code contains errors, infinite loops, or attempts to access system resources, your main application remains safe and responsive.