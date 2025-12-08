try:  # fmt:no
    import ray
    from ray.exceptions import GetTimeoutError
    import os
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # fmt:yes
except ImportError:
    raise ImportError('Python package "ray" is not installed.')

from abc import abstractmethod
import time
import traceback

from typing import Any, Tuple, Dict, List, Callable

from adtools.py_code import PyProgram
from adtools.evaluator.py_evaluator import PyEvaluator
from adtools.evaluator.utils import _redirect_to_devnull


class PyEvaluatorRay(PyEvaluator):
    def __init__(
        self,
        exec_code: bool = True,
        debug_mode: bool = False,
        *,
        join_timeout_seconds: int = 10,
    ):
        """Evaluator using Ray for secure, isolated execution.
        It supports efficient zero-copy return of large objects (e.g., Tensors).

        Args:
            exec_code: Whether to execute the code using 'exec()'.
            debug_mode: Enable debug print statements.
            join_timeout_seconds: (Not primarily used in Ray logic, but kept for compatibility).
        """
        # We set find_and_kill_children_evaluation_process to False because Ray
        # manages the process tree, and we use ray.kill() to clean up
        super().__init__(
            exec_code,
            find_and_kill_children_evaluation_process=False,
            debug_mode=debug_mode,
            join_timeout_seconds=join_timeout_seconds,
        )
        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def secure_evaluate(
        self,
        program: str | PyProgram,
        timeout_seconds: int | float = None,
        redirect_to_devnull: bool = False,
        get_evaluate_time: bool = False,
        ray_worker_options: dict[str, Any] = None,
        **kwargs,
    ) -> Any | Tuple[Any, float]:
        """Evaluates the program in a separate Ray Actor (process).

        Args:
            program: the program to be evaluated.
            timeout_seconds: return 'None' if the execution time exceeds 'timeout_seconds'.
            redirect_to_devnull: redirect any output to '/dev/null'.
            get_evaluate_time: get evaluation time for this program.
            ray_worker_options: options to pass to ray.option(), e.g., dict(num_cpus1, num_gpus=4)
            **kwargs: additional keyword arguments to pass to 'evaluate_program'.

        Mechanism:
            1. Spawns a new Ray Actor (Worker).
            2. Sends 'self' (the evaluator) and the code to the Worker.
            3. Waits for the result with a timeout.
            4. Kills the Worker immediately to ensure a clean slate and resource release.
        """
        # Convert PyProgram to string if necessary
        program_str = str(program)
        # Create a new Ray Actor (Sandbox)
        # Create a fresh actor for every evaluation to ensure total isolation
        worker = _RayWorker.options(**(ray_worker_options or {})).remote()
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

        except GetTimeoutError:
            # Handle Timeout
            if self.debug_mode:
                print(f"DEBUG: Ray evaluation timed out after {timeout_seconds}s.")
            result = None
        except Exception as e:
            # Handle other runtime exceptions (syntax errors, runtime errors in code)
            if self.debug_mode:
                print(f"DEBUG: Ray evaluation exception:\n{traceback.format_exc()}")
            result = None
        finally:
            # Cleanup: Force kill the actor
            # 'no_restart=True' ensures Ray does not try to respawn this worker
            # This releases the resources (CPUs/GPUs) immediately
            ray.kill(worker, no_restart=True)
            eval_time = time.time() - start_time

        return (result, eval_time) if get_evaluate_time else result

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


@ray.remote(max_concurrency=1)  # noqa
class _RayWorker:
    """A standalone Ray Actor used to execute the evaluation logic in a separate process."""

    def run_evaluation(
        self,
        evaluator_instance: "PyEvaluator",
        program_str: str,
        redirect_to_devnull: bool,
        **kwargs,
    ) -> Any:
        """Executes the evaluation inside the remote Ray process.

        Args:
            evaluator_instance: The evaluator object (pickled and sent to this worker).
            program_str: The code to evaluate.
            redirect_to_devnull: Whether to silence stdout/stderr.
            **kwargs: Arguments passed to evaluate_program.
        """
        if redirect_to_devnull:
            _redirect_to_devnull()

        try:
            # Invoke the parent class's evaluate method
            return evaluator_instance.evaluate(program_str, **kwargs)
        except Exception:
            # Re-raise to let Ray handle the exception
            raise


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
    score = evaluator.evaluate(code_generated_by_llm)
    print(f"Score: {score}")

    # Secure evaluate (the evaluation is executed in a sandbox process)
    score = evaluator.secure_evaluate(code_generated_by_llm, timeout_seconds=10)
    print(f"Score: {score}")

    # Evaluate a harmful code, the evaluation will be terminated within 10 seconds
    # We will obtain a score of `None` due to the violation of time restriction
    score = evaluator.secure_evaluate(harmful_code_generated_by_llm, timeout_seconds=10)
    print(f"Score: {score}")
