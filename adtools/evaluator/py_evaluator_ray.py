import logging
import os
import time
import traceback
from abc import abstractmethod
from typing import Any, Tuple, Dict, List, Callable

from adtools.py_code import PyProgram
from adtools.evaluator.py_evaluator import PyEvaluator, EvaluationResults
from adtools.evaluator.utils import _redirect_to_devnull


__all__ = ["PyEvaluatorRay"]


class PyEvaluatorRay(PyEvaluator):
    def __init__(
        self,
        exec_code: bool = True,
        debug_mode: bool = False,
    ):
        """Evaluator using Ray for secure, isolated execution.
        It supports efficient zero-copy return of large objects (e.g., Tensors).

        Args:
            exec_code: Whether to execute the code using 'exec()'.
            debug_mode: Enable debug print statements.
        """
        super().__init__(
            exec_code=exec_code,
            debug_mode=debug_mode,
        )

        # Lazy Import Start
        import ray

        # Set environment variable before Ray initialization (moved from top-level)
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level=logging.ERROR,
                log_to_driver=False,
            )

    def secure_evaluate(
        self,
        program: str | PyProgram,
        timeout_seconds: int | float = None,
        redirect_to_devnull: bool = False,
        get_evaluate_time: bool = False,
        ray_worker_options: dict[str, Any] = None,
        **kwargs,
    ) -> EvaluationResults:
        """Evaluates the program in a separate Ray Actor (process)."""
        # Lazy Import for Execution
        import ray
        from ray.exceptions import GetTimeoutError  # fmt:skip

        # Convert PyProgram to string if necessary
        program_str = str(program)

        # Create a new Ray Actor (Sandbox)
        # Since we cannot use @ray.remote at the top level (ray is not imported yet),
        # we dynamically convert the class to a remote actor here.
        RemoteWorkerClass = ray.remote(max_concurrency=1)(_RayWorker)

        # Create the worker instance
        worker = RemoteWorkerClass.options(**(ray_worker_options or {})).remote()

        start_time = time.time()
        try:
            # Execute asynchronously
            # Pass 'self' to the remote worker. Ray pickles this instance
            # The actual execution logic (evaluate_program) runs inside the worker process
            future = worker.run_evaluation.remote(
                self, program_str, redirect_to_devnull, **kwargs
            )
            # Wait for result with timeout
            result = ray.get(future, timeout=timeout_seconds)
            return EvaluationResults(
                result=result,
                evaluate_time=time.time() - start_time,
                error_msg="",
            )
        except GetTimeoutError:
            if self.debug_mode:
                print(f"DEBUG: Ray evaluation timed out after {timeout_seconds}s.")
            return EvaluationResults(
                result=None,
                evaluate_time=time.time() - start_time,
                error_msg="Evaluation timeout.",
            )
        except:
            # Handle other runtime exceptions (syntax errors, runtime errors in code)
            if self.debug_mode:
                print(f"DEBUG: Ray evaluation exception:\n{traceback.format_exc()}")
            return EvaluationResults(
                result=None,
                evaluate_time=time.time() - start_time,
                error_msg=str(traceback.format_exc()),
            )
        finally:
            # Cleanup: Force kill the actor
            # 'no_restart=True' ensures Ray does not try to respawn this worker
            # This releases the resources (CPUs/GPUs) immediately
            ray.kill(worker, no_restart=True)

    @abstractmethod
    def evaluate_program(
        self,
        program_str: str,
        callable_functions_dict: Dict[str, Callable] | None,
        callable_functions_list: List[Callable] | None,
        callable_classes_dict: Dict[str, Callable] | None,
        callable_classes_list: List[Callable] | None,
        **kwargs,
    ) -> Any:
        """Evaluate a given program.

        Args:
            program_str: The raw program text.
            callable_functions_dict: A dict maps function name to callable function.
            callable_functions_list: A list of callable functions.
            callable_classes_dict: A dict maps class name to callable class.
            callable_classes_list: A list of callable classes.

        Returns:
            Returns the evaluation result.
        """
        raise NotImplementedError(
            "Must provide an evaluator for a python program. "
            "Override this method in a subclass."
        )


class _RayWorker:
    """A standalone Ray Actor used to execute the evaluation logic in a separate process."""

    def run_evaluation(
        self,
        evaluator_instance: "PyEvaluator",
        program_str: str,
        redirect_to_devnull: bool,
        **kwargs,
    ) -> Any:
        """Executes the evaluation inside the remote Ray process."""
        if redirect_to_devnull:
            _redirect_to_devnull()

        return evaluator_instance._exec_and_get_res(program_str, **kwargs)


if __name__ == "__main__":

    class SortAlgorithmEvaluator(PyEvaluatorRay):
        def evaluate_program(
            self,
            program_str: str,
            callable_functions_dict: Dict[str, Callable] | None,
            callable_functions_list: List[Callable] | None,
            callable_classes_dict: Dict[str, Callable] | None,
            callable_classes_list: List[Callable] | None,
            **kwargs,
        ) -> Any | None:
            """Evaluate a given sort algorithm program."""
            # Get the sort algorithm
            sort_algo: Callable = callable_functions_dict["merge_sort"]
            # Test data
            input = [10, 2, 4, 76, 19, 29, 3, 5, 1]
            # Compute execution time
            start = time.time()
            res = sort_algo(input)
            duration = time.time() - start
            if res == sorted(input):  # If the result is correct
                return (
                    duration  # Return the execution time as the score of the algorithm
                )
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
