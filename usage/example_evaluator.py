import time
from typing import Dict, Callable, List, Any

from adtools import PyEvaluator
from adtools.evaluator import PyEvaluatorReturnInSharedMemory


class SortAlgorithmEvaluator(PyEvaluatorReturnInSharedMemory):
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
    evaluator = SortAlgorithmEvaluator(debug_mode=True)

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
