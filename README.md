# Useful tools for parsing and evaluating Python programs for algorithm design

------

> This repo aims to help develop more powerful [Large Language Models for Algorithm Design (LLM4AD)](https://github.com/Optima-CityU/llm4ad) applications. 
>
> More tools will be provided soon.

------

The figure demonstrates how a Python program is parsed into `PyCodeBlock`, `PyFunction`, `PyClass,` and `PyProgram` via `adtools`.

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

## Usage

### Parser for a Python program

Parse your code (in string) into Python code instances, so that you can check each component and modify it.

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

### Evaluate Python programs

Evaluate Python programs in a secure process to avoid the abortation of the main process. Two steps:

- Extend the `PyEvaluator` class and override the `evaluate_program` method.
- Evaluate the program (in str) by calling the `evaluate` (directly evaluate without executing in a sandbox process) or the `secure_evaluate` (evaluate in a sandbox process) methods.

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
    score = evaluator.evaluate(code_generated_by_llm)
    print(f'Score: {score}')

    # Secure evaluate (the evaluation is executed in a sandbox process)
    score = evaluator.secure_evaluate(code_generated_by_llm, timeout_seconds=10)
    print(f'Score: {score}')

    # Evaluate a harmful code, the evaluation will be terminated within 10 seconds
    # We will obtain a score of `None` due to the violation of time restriction
    score = evaluator.secure_evaluate(harmful_code_generated_by_llm, timeout_seconds=10)
    print(f'Score: {score}')
```

