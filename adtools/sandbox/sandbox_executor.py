"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import multiprocessing
import pickle
import time
import uuid
from multiprocessing import shared_memory, resource_tracker
from queue import Empty
from typing import Any, Dict, List, TypedDict, Optional, Tuple
import multiprocessing.managers
import traceback

import psutil

from adtools.sandbox.utils import _redirect_to_devnull

__all__ = ["ExecutionResults", "SandboxExecutor"]


class ExecutionResults(TypedDict):
    result: Any
    evaluate_time: float
    error_msg: str


class SandboxExecutor:

    def __init__(
        self,
        evaluate_worker: Any,
        find_and_kill_children_evaluation_process: bool = False,
        debug_mode: bool = False,
        *,
        join_timeout_seconds: int = 10,
    ):
        """Evaluator interface for evaluating the Python algorithm program. Override this class and implement
        'evaluate_program' method, then invoke 'self.evaluate()' or 'self.secure_evaluate()' for evaluation.

        Args:
            exec_code: Using 'exec()' to execute the program code and obtain the callable functions and classes,
                which will be passed to 'self.evaluate_program()'. Set this parameter to 'False' if you are going to
                evaluate a Python scripy. Note that if the parameter is set to 'False', the arguments 'callable_...'
                in 'self.evaluate_program()' will no longer be affective.
            find_and_kill_children_evaluation_process: If using 'self.secure_evaluate', kill children processes
                when they are terminated. Note that it is suggested to set to 'False' if the evaluation process
                does not start new processes.
            debug_mode: Debug mode.
            join_timeout_seconds: Timeout in seconds to wait for the process to finish. Kill the process if timeout.
        """
        self.evaluate_worker = evaluate_worker
        self.debug_mode = debug_mode
        self.find_and_kill_children_evaluation_process = (
            find_and_kill_children_evaluation_process
        )
        self.join_timeout_seconds = join_timeout_seconds

    def _kill_process_and_its_children(self, process: multiprocessing.Process):
        if self.find_and_kill_children_evaluation_process:
            # Find all children processes
            try:
                parent = psutil.Process(process.pid)
                children_processes = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                children_processes = []
        else:
            children_processes = []
        # Terminate parent process
        process.terminate()
        process.join(timeout=self.join_timeout_seconds)
        if process.is_alive():
            process.kill()
            process.join()
        # Kill all children processes
        for child in children_processes:
            if self.debug_mode:
                print(f"Killing process {process.pid}'s children process {child.pid}")
            child.terminate()

    def _execute_and_put_res_in_shared_memory(
        self,
        worker_execute_method_name: str,
        method_args: Optional[List | Tuple],
        method_kwargs: Optional[Dict],
        meta_queue: multiprocessing.Queue,
        redirect_to_devnull: bool,
        shm_name_id: str,
    ):
        """Evaluate and store result in shared memory (for large results)."""
        # Redirect STDOUT and STDERR to '/dev/null'
        if redirect_to_devnull:
            _redirect_to_devnull()

        if hasattr(self.evaluate_worker, worker_execute_method_name):  # todo
            method_to_call = getattr(self.evaluate_worker, worker_execute_method_name)
        else:
            raise RuntimeError(
                f"Method named '{worker_execute_method_name}' not found."
            )

        # Execute and get results
        # noinspection PyBroadException
        try:
            # Execute the target method and get result
            args = method_args or []
            kwargs = method_kwargs or {}
            res = method_to_call(*args, **kwargs)

            # Dump the results to data
            data = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
            # Create shared memory using the ID provided by the parent
            # We must use create=True here as the child is responsible for allocation
            shm = shared_memory.SharedMemory(
                create=True, name=shm_name_id, size=len(data)
            )
            # Unregister the shared memory block from the resource tracker in this child process
            # The shared memory will be managed in the parent process
            # noinspection PyProtectedMember, PyUnresolvedReferences
            resource_tracker.unregister(name=shm._name, rtype="shared_memory")

            # Write data
            shm.buf[: len(data)] = data
            # We only need to send back the size, as the parent already knows the name.
            # Sending (True, size) to indicate success.
            meta_queue.put((True, len(data)))
            # Child closes its handle
            shm.close()
        except:
            if self.debug_mode:
                traceback.print_exc()
            # Put the exception message to the queue
            # Sending (False, error_message) to indicate failure.
            meta_queue.put((False, str(traceback.format_exc())))

    def secure_execute(
        self,
        worker_execute_method_name: str,
        method_args: Optional[List | Tuple] = None,
        method_kwargs: Optional[Dict] = None,
        timeout_seconds: int | float = None,
        redirect_to_devnull: bool = False,
        **kwargs,
    ) -> ExecutionResults:
        """Evaluate program in a new process.
        This enables timeout restriction and output redirection.

        Args:
            worker_execute_method_name: Name of the worker execute method.
            method_args: Arguments of the worker execute method.
            method_kwargs: Keyword arguments of the worker execute method.
            timeout_seconds: return 'None' if the execution time exceeds 'timeout_seconds'.
            redirect_to_devnull: redirect any output to '/dev/null'.

        Returns:
            Returns the evaluation results. If the 'get_evaluate_time' is True,
            the return value will be (Results, Time).
        """
        # Evaluate and get results
        # noinspection PyBroadException
        try:
            # Create a meta queue to get meta information from the evaluation process
            meta_queue = multiprocessing.Queue()
            # Generate a unique name for the shared memory block in the PARENT process.
            # This allows the parent to clean it up even if the child is killed.
            unique_shm_name = f"psm_{uuid.uuid4().hex[:8]}"

            process = multiprocessing.Process(
                target=self._execute_and_put_res_in_shared_memory,
                args=(
                    worker_execute_method_name,
                    method_args,
                    method_kwargs,
                    meta_queue,
                    redirect_to_devnull,
                    unique_shm_name,
                ),
            )
            evaluate_start_time = time.time()
            process.start()

            try:
                # Try to get the metadata before timeout
                meta = meta_queue.get(timeout=timeout_seconds)
                # Calculate evaluation time
                eval_time = time.time() - evaluate_start_time
            except Empty:
                if self.debug_mode:
                    print(f"DEBUG: evaluation time exceeds {timeout_seconds}s.")

                # Evaluation timeout happens, we return 'None' as well as the actual evaluate time
                return ExecutionResults(
                    result=None,
                    evaluate_time=time.time() - evaluate_start_time,
                    error_msg="Evaluation timeout.",
                )

            # The 'meta' is now (Success_Flag, Data_Size_or_Error_Msg)
            success, payload = meta

            if not success:
                # Payload is the error message
                error_msg = payload
                result = None
            else:
                error_msg = ""
                # Payload is the size of the data
                size = payload
                # Attach to the existing shared memory by name
                shm = shared_memory.SharedMemory(name=unique_shm_name)
                buf = bytes(shm.buf[:size])
                # Load results from buffer
                result = pickle.loads(buf)
                shm.close()

            return ExecutionResults(
                result=result, evaluate_time=eval_time, error_msg=error_msg
            )
        except:
            if self.debug_mode:
                print(f"DEBUG: exception in shared evaluate:\n{traceback.format_exc()}")

            return ExecutionResults(
                result=None,
                evaluate_time=time.time() - evaluate_start_time,
                error_msg=str(traceback.format_exc()),
            )
        finally:
            self._kill_process_and_its_children(process)
            # Critical Cleanup: Ensure the shared memory is unlinked from the OS
            # This runs whether the process finished, timed out, or crashed
            try:
                # Attempt to attach to the shared memory block
                shm_cleanup = shared_memory.SharedMemory(name=unique_shm_name)
                shm_cleanup.close()
                # Unlink (delete) it from the system, and close the shared memory
                shm_cleanup.unlink()
            except FileNotFoundError:
                # This is normal if the child process never reached the creation step
                # (e.g. crashed during calculation before creating SHM)
                pass
