"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Callable

from adtools.py_code import PyProgram
from adtools.evaluator.py_evaluator import PyEvaluator
from adtools.sandbox.sandbox_executor_ray import SandboxExecutorRay, ExecutionResults


__all__ = ["PyEvaluatorRay"]


class PyEvaluatorRay(PyEvaluator):

    def __init__(
        self,
        init_ray: bool = True,
        exec_code: bool = True,
        debug_mode: bool = False,
        *,
        ray_rotation_max_bytes: int = 50 * 1024 * 1024,  # 50 MB
        ray_rotation_backup_count: int = 1,
    ):
        """Evaluator using Ray for secure, isolated execution.
        It supports efficient zero-copy return of large objects (e.g., Tensors).

        Args:
            init_ray: Whether to initialize the ray.
            exec_code: Whether to execute the code using 'exec()'.
            debug_mode: Enable debug print statements.
        """
        super().__init__(
            exec_code=exec_code,
            debug_mode=debug_mode,
        )

        self.sandbox_executor = SandboxExecutorRay(
            evaluate_worker=self,
            init_ray=init_ray,
            debug_mode=debug_mode,
            ray_rotation_max_bytes=ray_rotation_max_bytes,
            ray_rotation_backup_count=ray_rotation_backup_count,
        )

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

    def secure_evaluate(
        self,
        program: str | PyProgram,
        timeout_seconds: int | float = None,
        redirect_to_devnull: bool = False,
        *,
        ray_actor_options: dict[str, Any] = None,
        **kwargs,
    ) -> ExecutionResults:
        """Evaluates the program in a separate Ray Actor (process).

        Args:
            program: the program to be evaluated.
            timeout_seconds: return 'None' if the execution time exceeds 'timeout_seconds'.
            redirect_to_devnull: redirect any output to '/dev/null'.
            ray_actor_options: kwargs pass to RayWorkerClass.options(...).
            **kwargs: additional keyword arguments to pass to 'evaluate_program'.

        Returns:
            Returns the evaluation results. If the 'get_evaluate_time' is True,
            the return value will be (Results, Time).
        """
        return self.sandbox_executor.secure_execute(
            worker_execute_method_name="_exec_and_get_res",
            method_args=[program],
            method_kwargs=kwargs,
            timeout_seconds=timeout_seconds,
            redirect_to_devnull=redirect_to_devnull,
            ray_actor_options=ray_actor_options,
        )
