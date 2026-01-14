import sys

from adtools.sandbox.sandbox_executor import SandboxExecutor

sys.path.append("../")
import time
from typing import Dict, Callable, List, Any

from adtools.evaluator import PyEvaluator, PyEvaluatorRay


class SortAlgorithmEvaluator:
    def evaluate_program(self, program: str) -> Any | None:
        g = {}
        exec(program, g)
        # Get the sort algorithm
        sort_algo: Callable = g["merge_sort"]
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
    sandbox = SandboxExecutor(SortAlgorithmEvaluator(), debug_mode=True)
    score = sandbox.secure_execute(
        "evaluate_program",
        method_args=(code_generated_by_llm,),
    )
    print(f"Score: {score}")

    # Secure evaluate (the evaluation is executed in a sandbox process)
    score = sandbox.secure_execute(
        "evaluate_program",
        method_args=(code_generated_by_llm,),
        timeout_seconds=10,
    )
    print(f"Score: {score}")

    # Evaluate a harmful code, the evaluation will be terminated within 10 seconds
    # We will obtain a score of `None` due to the violation of time restriction
    score = sandbox.secure_execute(
        "evaluate_program",
        method_args=(harmful_code_generated_by_llm,),
        timeout_seconds=10,
    )
    print(f"Score: {score}")
