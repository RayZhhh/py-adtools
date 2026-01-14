"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from adtools.sandbox.sandbox_executor import SandboxExecutor, ExecutionResults
from adtools.sandbox.utils import _redirect_to_devnull

__all__ = ["SandboxExecutorRay"]


class SandboxExecutorRay(SandboxExecutor):

    def __init__(
        self,
        evaluate_worker: Any,
        init_ray: bool = True,
        debug_mode: bool = False,
        *,
        ray_rotation_max_bytes: int = 50 * 1024 * 1024,  # 50 MB
        ray_rotation_backup_count: int = 1,
    ):
        """Evaluator using Ray for secure, isolated execution.

        Args:
            evaluate_worker: The worker object to be executed.
            init_ray: Whether to initialize ray.
            debug_mode: Enable debug print statements.
            ray_rotation_max_bytes: Max bytes for ray log rotation.
            ray_rotation_backup_count: Backup count for ray log rotation.
        """
        super().__init__(
            evaluate_worker=evaluate_worker,
            debug_mode=debug_mode,
        )

        import ray

        if init_ray:
            if ray.is_initialized():
                logging.warning(
                    f"Ray is already initialized. "
                    f"If you want to disable reinit, "
                    f"please set '{self.__class__.__name__}(..., init_ray=False)'."
                )
            # Set environment variable before Ray initialization
            os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
            os.environ["RAY_ROTATION_MAX_BYTES"] = str(ray_rotation_max_bytes)
            os.environ["RAY_ROTATION_BACKUP_COUNT"] = str(ray_rotation_backup_count)

            # Initialize Ray
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level=logging.ERROR,
                log_to_driver=True,
            )
        elif not ray.is_initialized():
            raise RuntimeError(
                f"Ray is not initialized. "
                f"Please set '{self.__class__.__name__}(..., init_ray=True)'."
            )

    def secure_execute(
        self,
        worker_execute_method_name: str,
        method_args: Optional[List | Tuple] = None,
        method_kwargs: Optional[Dict] = None,
        timeout_seconds: int | float = None,
        redirect_to_devnull: bool = False,
        *,
        ray_actor_options: dict[str, Any] = None,
        **kwargs,
    ) -> ExecutionResults:
        """Evaluates the program in a separate Ray Actor (process).
        This enables timeout restriction and output redirection.

        Args:
            worker_execute_method_name: Name of the worker execute method.
            method_args: Arguments of the worker execute method.
            method_kwargs: Keyword arguments of the worker execute method.
            timeout_seconds: return 'None' if the execution time exceeds 'timeout_seconds'.
            redirect_to_devnull: redirect any output to '/dev/null'.
            ray_actor_options: Ray actor options.

        Returns:
            Returns the evaluation results. If the 'get_evaluate_time' is True,
            the return value will be (Results, Time).
        """
        import ray
        from ray.exceptions import GetTimeoutError

        if ray_actor_options is None:
            ray_actor_options = {}
        else:
            ray_actor_options = ray_actor_options.copy()

        # Propagate sys.path and PYTHONPATH
        runtime_env = ray_actor_options.get("runtime_env", {})
        env_vars = runtime_env.get("env_vars", {})

        current_paths = [p for p in sys.path if p and os.path.exists(p)]
        existing_pythonpath = env_vars.get("PYTHONPATH", "")
        if existing_pythonpath:
            current_paths.insert(0, existing_pythonpath)

        # Deduplicate preserving order
        unique_paths = []
        seen = set()
        for p in current_paths:
            if p not in seen:
                unique_paths.append(p)
                seen.add(p)

        env_vars["PYTHONPATH"] = os.pathsep.join(unique_paths)
        runtime_env["env_vars"] = env_vars
        ray_actor_options["runtime_env"] = runtime_env

        # Create Remote Worker Class
        RemoteWorkerClass = ray.remote(max_concurrency=1)(_RayWorker)

        # Create worker
        worker = RemoteWorkerClass.options(**ray_actor_options).remote(
            self.evaluate_worker
        )

        start_time = time.time()
        try:
            future = worker.execute.remote(
                worker_execute_method_name,
                method_args,
                method_kwargs,
                redirect_to_devnull,
            )
            result = ray.get(future, timeout=timeout_seconds)
            return ExecutionResults(
                result=result,
                evaluate_time=time.time() - start_time,
                error_msg="",
            )
        except GetTimeoutError:
            if self.debug_mode:
                print(f"DEBUG: Ray evaluation timed out after {timeout_seconds}s.")
            return ExecutionResults(
                result=None,
                evaluate_time=time.time() - start_time,
                error_msg="Evaluation timeout.",
            )
        except Exception:
            if self.debug_mode:
                print(f"DEBUG: Ray evaluation exception:\n{traceback.format_exc()}")
            return ExecutionResults(
                result=None,
                evaluate_time=time.time() - start_time,
                error_msg=str(traceback.format_exc()),
            )
        finally:
            ray.kill(worker, no_restart=True)


class _RayWorker:
    """A standalone Ray Actor used to execute the evaluation logic in a separate process."""

    def __init__(self, evaluate_worker: Any):
        self.evaluate_worker = evaluate_worker

    def execute(
        self,
        worker_execute_method_name: str,
        method_args: Optional[List | Tuple],
        method_kwargs: Optional[Dict],
        redirect_to_devnull: bool,
    ) -> Any:
        if redirect_to_devnull:
            _redirect_to_devnull()

        if hasattr(self.evaluate_worker, worker_execute_method_name):
            method_to_call = getattr(self.evaluate_worker, worker_execute_method_name)
            args = method_args or []
            kwargs = method_kwargs or {}
            return method_to_call(*args, **kwargs)
        else:
            raise RuntimeError(
                f"Method named '{worker_execute_method_name}' not found in worker."
            )
