import time
from typing import Dict, List, Callable, Any
from adtools.evaluator import PyEvaluator, PyEvaluatorReturnInManagerDict, PyEvaluatorReturnInSharedMemory

# Try importing Ray Evaluator Base
from adtools.evaluator import PyEvaluatorRay


# =============================================================================
# 1. Base Evaluator (Queue-based)
# =============================================================================
class ConcreteEvaluatorBase(PyEvaluator):
    """
    Uses standard multiprocessing.Queue.
    """
    def evaluate_program(self, program_str, callable_functions_dict, *args, **kwargs):
        target_func_name = "solver"
        if callable_functions_dict and target_func_name in callable_functions_dict:
            return callable_functions_dict[target_func_name]()
        else:
            local_scope = {}
            exec(program_str, {}, local_scope)
            if target_func_name in local_scope:
                return local_scope[target_func_name]()
            raise RuntimeError(f"Entry point '{target_func_name}' not found.")

# =============================================================================
# 2. Manager Dict Evaluator
# =============================================================================
class ConcreteEvaluatorDict(PyEvaluatorReturnInManagerDict):
    """
    Uses multiprocessing.Manager().dict().
    """
    def evaluate_program(self, program_str, callable_functions_dict, *args, **kwargs):
        target_func_name = "solver"
        if callable_functions_dict and target_func_name in callable_functions_dict:
            return callable_functions_dict[target_func_name]()
        raise RuntimeError(f"Entry point '{target_func_name}' not found.")

# =============================================================================
# 3. Shared Memory Evaluator
# =============================================================================
class ConcreteEvaluatorShm(PyEvaluatorReturnInSharedMemory):
    """
    Uses SharedMemory for zero-copy (simulated) data transfer.
    """
    def evaluate_program(self, program_str, callable_functions_dict, *args, **kwargs):
        target_func_name = "solver"
        if callable_functions_dict and target_func_name in callable_functions_dict:
            return callable_functions_dict[target_func_name]()
        raise RuntimeError(f"Entry point '{target_func_name}' not found.")

# =============================================================================
# 4. Ray Evaluator
# =============================================================================
class ConcreteEvaluatorRay(PyEvaluatorRay):
    """
    Uses Ray Actors for execution.
    """
    def evaluate_program(self, program_str, callable_functions_dict, *args, **kwargs):
        target_func_name = "solver"
        if callable_functions_dict and target_func_name in callable_functions_dict:
            return callable_functions_dict[target_func_name]()
        raise RuntimeError(f"Entry point '{target_func_name}' not found.")
