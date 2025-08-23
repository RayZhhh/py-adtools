"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import multiprocessing
import os
import sys
from abc import ABC, abstractmethod
from queue import Empty
from typing import Any, Literal, Dict, Callable, List
import psutil
import traceback

from .py_code import PyProgram


class PyEvaluator(ABC):

    def __init__(
            self,
            exec_code: bool = True,
            debug_mode: bool = False,
            *,
            join_timeout_seconds: int = 10
    ):
        """Evaluator interface for evaluating the Python algorithm program. Override this class and implement
        'evaluate_program' method, then invoke 'self.evaluate()' or 'self.secure_evaluate()' for evaluation.
        Args:
            exec_code: Using 'exec()' to execute the program code and obtain the callable functions and classes,
                which will be passed to 'self.evaluate_program()'. Set this parameter to 'False' if you are going to
                evaluate a Python scripy. Note that if the parameter is set to 'False', the arguments 'callable_...'
                in 'self.evaluate_program()' will no longer be affective.
            debug_mode: Debug mode.
            join_timeout_seconds: Timeout in seconds to wait for the process to finish. Kill the process if timeout.
        """
        self.debug_mode = debug_mode
        self.exec_code = exec_code
        self.join_timeout_seconds = join_timeout_seconds

    @abstractmethod
    def evaluate_program(
            self,
            program_str: str,
            callable_functions_dict: Dict[str, Callable] | None,
            callable_functions_list: List[Callable] | None,
            callable_classes_dict: Dict[str, Callable] | None,
            callable_classes_list: List[Callable] | None,
            **kwargs
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
            'Must provide an evaluator for a python program. '
            'Override this method in a subclass.'
        )

    def _kill_process_and_its_children(self, process: multiprocessing.Process):
        # Find all children processes
        try:
            parent = psutil.Process(process.pid)
            children_processes = parent.children(recursive=True)
        except psutil.NoSuchProcess:
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

    def evaluate(self, program: str | PyProgram, **kwargs):
        """Evaluate a program.
        Args:
            program: the program to be evaluated.
            **kwargs: additional keyword arguments to pass to 'evaluate_program'.
        """
        try:
            # Parse to program instance
            if isinstance(program, str):
                program = PyProgram.from_text(program)
            function_names = [f.name for f in program.functions]
            class_names = [c.name for c in program.classes]

            # Execute the code and get callable instances
            if self.exec_code:
                all_globals_namespace = {}
                # Execute the program, map func/var/class to global namespace
                exec(str(program), all_globals_namespace)
                # Get callable functions
                callable_funcs_list = [all_globals_namespace[f_name] for f_name in function_names]
                callable_funcs_dict = dict(zip(function_names, callable_funcs_list))
                # Get callable classes
                callable_cls_list = [all_globals_namespace[c_name] for c_name in class_names]
                callable_cls_dict = dict(zip(class_names, callable_cls_list))
            else:
                callable_funcs_list, callable_funcs_dict, callable_cls_list, callable_cls_dict = (
                    None, None, None, None
                )

            # Get evaluate result
            res = self.evaluate_program(
                str(program),
                callable_funcs_dict,
                callable_funcs_list,
                callable_cls_dict,
                callable_cls_list,
                **kwargs
            )
            return res
        except Exception as e:
            if self.debug_mode:
                print(traceback.format_exc())
            return None

    def _evaluate_in_safe_process(
            self,
            program_str: str,
            result_queue: multiprocessing.Queue,
            redirect_to_devnull: bool,
            **kwargs
    ):
        # Redirect STDOUT and STDERR to '/dev/null'
        if redirect_to_devnull:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), sys.stdout.fileno())
                os.dup2(devnull.fileno(), sys.stderr.fileno())

        # Evaluate and put the results to the queue
        res = self.evaluate(program_str, **kwargs)
        result_queue.put(res)

    def secure_evaluate(
            self,
            program: str | PyProgram,
            timeout_seconds: int | float = None,
            redirect_to_devnull: bool = False,
            multiprocessing_start_method: Literal['default', 'auto', 'fork', 'spawn'] = 'auto',
            **kwargs
    ):
        """Evaluate program in a new process. This enables timeout restriction and output redirection.
        Args:
            program: the program to be evaluated.
            timeout_seconds: return 'None' if the execution time exceeds 'timeout_seconds'.
            redirect_to_devnull: redirect any output to '/dev/null'.
            multiprocessing_start_method: start a process using 'fork' or 'spawn'. If set to 'auto',
                the process will be started using 'fork' with Linux/macOS and 'spawn' with Windows.
                If set to 'default', there will be no changes to system default.
            **kwargs: additional keyword arguments to pass to 'evaluate_program'.
        """
        if multiprocessing_start_method == 'auto':
            # Force macOS and Linux use 'fork' to generate new process
            if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
                multiprocessing.set_start_method('fork', force=True)
        elif multiprocessing_start_method == 'fork':
            multiprocessing.set_start_method('fork', force=True)
        elif multiprocessing_start_method == 'spawn':
            multiprocessing.set_start_method('spawn', force=True)

        try:
            # Start evaluation process
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=self._evaluate_in_safe_process,
                args=(str(program), result_queue, redirect_to_devnull),
                kwargs=kwargs,
            )
            process.start()

            if timeout_seconds is not None:
                try:
                    # Get the result in timeout seconds
                    result = result_queue.get(timeout=timeout_seconds)
                    # After getting the result, terminate/kill the process
                    self._kill_process_and_its_children(process)
                except Empty:  # The queue is empty indicates a timeout
                    if self.debug_mode:
                        print(f'DEBUG: the evaluation time exceeds {timeout_seconds}s.')
                    # Terminate/kill all processes if timeout happens
                    self._kill_process_and_its_children(process)
                    result = None
                except Exception as e:
                    if self.debug_mode:
                        print(f'DEBUG: evaluation failed with exception:\n{traceback.format_exc()}')
                    # Terminate/kill all processes if meet exceptions
                    self._kill_process_and_its_children(process)
                    result = None
            else:
                # If there is no timeout limit, wait execution to finish
                result = result_queue.get()
                # Terminate/kill all processes after evaluation
                self._kill_process_and_its_children(process)
            return result
        except Exception as e:
            if self.debug_mode:
                print(traceback.format_exc())
            return None
