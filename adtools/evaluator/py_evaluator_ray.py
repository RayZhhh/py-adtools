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
        *,
        ray_worker_options: dict[str, Any] = None,
        **kwargs,
    ) -> EvaluationResults:
        """Evaluates the program in a separate Ray Actor (process)."""
        import ray
        import sys
        from ray.exceptions import GetTimeoutError  # fmt:skip

        # Convert PyProgram to string if necessary
        program_str = str(program)

        # We synchronize the current 'sys.path' to the Ray Worker's PYTHONPATH.
        # This ensures that any package importable in the driver (this process)
        # is also importable in the worker, including 'adtools' itself and
        # any other local or custom-installed packages.
        if ray_worker_options is None:
            ray_worker_options = {}
        else:
            ray_worker_options = ray_worker_options.copy()

        runtime_env = ray_worker_options.get("runtime_env", {})
        env_vars = runtime_env.get("env_vars", {})

        # Collect current sys.path, filtering out empty or invalid paths
        current_paths = [p for p in sys.path if p and os.path.exists(p)]

        # Merge with existing PYTHONPATH if provided by user
        existing_pythonpath = env_vars.get("PYTHONPATH", "")
        if existing_pythonpath:
            current_paths.insert(0, existing_pythonpath)

        # Deduplicate while preserving order
        unique_paths = []
        seen = set()
        for p in current_paths:
            if p not in seen:
                unique_paths.append(p)
                seen.add(p)

        # Update environment variables for the worker
        env_vars["PYTHONPATH"] = os.pathsep.join(unique_paths)
        runtime_env["env_vars"] = env_vars
        ray_worker_options["runtime_env"] = runtime_env

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
