"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import copy
import inspect
from functools import wraps
from typing import Literal, Optional, Any, Callable, Dict

from adtools.sandbox.sandbox_executor import SandboxExecutor, ExecutionResults
from adtools.sandbox.sandbox_executor_ray import SandboxExecutorRay

__all__ = ["sandbox_run"]


def _is_class_method(func) -> bool:
    return "." in func.__qualname__ and "<locals>" not in func.__qualname__


def _is_top_level_function(func) -> bool:
    return func.__qualname__ == func.__name__


class _FunctionWorker:
    """Helper class to wrap standalone functions for SandboxExecutor."""

    def __init__(self, func):
        self.func = func

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def sandbox_run(
    sandbox_type: Literal["ray", "process"] = "process",
    timeout: Optional[float] = None,
    redirect_to_devnull: bool = False,
    ray_actor_options: Optional[Dict[str, Any]] = None,
    **executor_init_kwargs,
):
    """Decorator to execute a class method or standalone function in a sandbox (Process or Ray).

    When a method/function decorated with @sandbox_run is called, it will be executed
    in a separate process (or Ray actor).

    Args:
        sandbox_type: The type of sandbox to use. 'ray' for SandboxExecutorRay,
                      'process' for SandboxExecutor. Defaults to 'process'.
        timeout: Timeout in seconds for the execution.
        redirect_to_devnull: Whether to redirect stdout/stderr to /dev/null
                             inside the sandbox.
        ray_actor_options: Options for the Ray actor (only used if sandbox_type='ray').
        **executor_init_kwargs: Additional keyword arguments passed to the
                                Executor's constructor (e.g., debug_mode, init_ray).

    Returns:
        A decorator that returns ExecutionResults when the decorated method/function is called.
    """

    def decorator(func: Callable) -> Callable:
        is_method_likely = _is_class_method(func)  # noqa

        @wraps(func)
        def wrapper(*args, **kwargs) -> ExecutionResults:
            if is_method_likely:
                # Treated as a method call: args[0] is 'self'
                if not args:
                    raise RuntimeError("Method call expected 'self' as first argument.")
                self_instance = args[0]
                method_args = args[1:]

                # Check bypass flag to prevent recursion
                if getattr(self_instance, "_bypass_sandbox", False):
                    return func(self_instance, *method_args, **kwargs)

                # Create worker copy
                evaluate_worker = copy.copy(self_instance)
                evaluate_worker._bypass_sandbox = True
                method_name = func.__name__
            else:
                # Treated as a standalone function
                worker = _FunctionWorker(func)
                evaluate_worker = worker
                method_name = "run"
                method_args = args

            # Prepare Executor
            if sandbox_type == "ray":
                import ray

                executor = SandboxExecutorRay(
                    evaluate_worker,
                    init_ray=ray.is_initialized(),
                    **executor_init_kwargs,
                )
                ray_options = ray_actor_options
            else:
                executor = SandboxExecutor(evaluate_worker, **executor_init_kwargs)
                ray_options = None  # Not used for process

            # Execute
            if sandbox_type == "ray":
                result = executor.secure_execute(
                    worker_execute_method_name=method_name,
                    method_args=method_args,
                    method_kwargs=kwargs,
                    timeout_seconds=timeout,
                    redirect_to_devnull=redirect_to_devnull,
                    ray_actor_options=ray_options,
                )
            else:
                result = executor.secure_execute(
                    worker_execute_method_name=method_name,
                    method_args=method_args,
                    method_kwargs=kwargs,
                    timeout_seconds=timeout,
                    redirect_to_devnull=redirect_to_devnull,
                )

            return result

        return wrapper

    return decorator


# class Worker:
#     @sandbox_run(sandbox_type="ray", timeout=None, redirect_to_devnull=False)
#     def double(self, a):
#         return a * 2
#
#
# @sandbox_run(sandbox_type="ray")
# def square(a):
#     return a * a
#
#
# if __name__ == "__main__":
#     # Test Class Method
#     worker = Worker()
#     res_method = worker.double(5)
#     print(f"Method Result: {res_method}")
#
#     # Test Standalone Function
#     res_func = square(4)
#     print(f"Function Result: {res_func}")
#     print(square.__name__)
