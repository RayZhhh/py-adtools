import multiprocessing
import os
import sys
import time
from abc import ABC, abstractmethod
from queue import Empty
from typing import Any, Literal, Dict, Callable, List
import psutil

from .py_code import PyProgram


class PyEvaluator(ABC):

    def __init__(self, debug_mode: bool = False, *, exec_code: bool = True):
        """Evaluator interface for evaluating the Python algorithm program.
        Args:
            debug_mode: Debug mode.
            exec_code : Using 'exec()' to compile the code and provide the callable function.
        """
        self._debug_mode = debug_mode
        self._exec_code = exec_code
        self._JOIN_TIMEOUT_SECONDS = 5

    @abstractmethod
    def evaluate_program(
            self,
            program_str: str,
            callable_functions_dict: Dict[str, Callable] | None,
            callable_functions_list: List[Callable] | None,
            callable_classes_dict: Dict[str, Callable] | None,
            callable_classes_list: List[Callable] | None,
            **kwargs
    ) -> Any | None:
        """Evaluate a given program.
        Args:
            program_str            : The raw program text.
            callable_functions_dict: A dict maps function name to callable function.
            callable_functions_list: A list of callable functions.
            callable_classes_dict  : A dict maps class name to callable class.
            callable_classes_list  : A list of callable classes.
        Return:
            Returns the evaluation result.
        """
        raise NotImplementedError('Must provide an evaluator for a python program. '
                                  'Override this method in a subclass.')

    def _kill_process_and_its_children(self, process: multiprocessing.Process):
        # Find all children processes
        try:
            parent = psutil.Process(process.pid)
            children_processes = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            children_processes = []
        # Terminate parent process
        process.terminate()
        process.join(timeout=self._JOIN_TIMEOUT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join()
        # Kill all children processes
        for child in children_processes:
            if self._debug_mode:
                print(f"Killing process {process.pid}'s children process {child.pid}")
            child.terminate()

    def evaluate(self, program_str: str, **kwargs):
        try:
            # Parse to program instance
            program = PyProgram.from_text(program_str)
            function_names = [f.name for f in program.functions]
            class_names = [c.name for c in program.classes]
            if self._exec_code:
                # Compile the program, and maps the global func/var/class name to its address
                all_globals_namespace = {}
                # Execute the program, map func/var/class to global namespace
                exec(program_str, all_globals_namespace)
                # Get callable functions
                callable_functions_list = [all_globals_namespace[f_name] for f_name in function_names]
                callable_functions_dict = dict(zip(function_names, callable_functions_list))
                # Get callable classes
                callable_classes_list = [all_globals_namespace[c_name] for c_name in class_names]
                callable_classes_dict = dict(zip(class_names, callable_classes_list))
            else:
                callable_functions_list = None
                callable_functions_dict = None
                callable_classes_list = None
                callable_classes_dict = None

            # Get evaluate result
            res = self.evaluate_program(
                program_str,
                callable_functions_dict,
                callable_functions_list,
                callable_classes_dict,
                callable_classes_list,
                **kwargs
            )
            return res
        except Exception as e:
            if self._debug_mode:
                print(e)
            return None

    def _evaluate_in_safe_process(
            self,
            program_str: str,
            result_queue: multiprocessing.Queue,
            redirect_to_devnull: bool,
            **kwargs
    ):
        if redirect_to_devnull:
            with open('/dev/null', 'w') as devnull:
                os.dup2(devnull.fileno(), sys.stdout.fileno())
                os.dup2(devnull.fileno(), sys.stderr.fileno())
        res = self.evaluate(program_str, **kwargs)
        result_queue.put(res)

    def secure_evaluate(
            self,
            program: str | PyProgram,
            timeout_seconds: int | float = None,
            redirect_to_devnull: bool = True,
            multiprocessing_start_method=Literal['auto', 'fork', 'spawn'],
            **kwargs
    ):
        """
        Args:
            program: the program to be evaluated.
            timeout_seconds: return 'None' if the execution time exceeds 'timeout_seconds'.
            redirect_to_devnull: redirect any output to '/dev/null'.
            multiprocessing_start_method: start a process using 'fork' or 'spawn'.
        """
        if multiprocessing_start_method == 'auto':
            # Force MacOS and Linux use 'fork' to generate new process
            if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
                multiprocessing.set_start_method('fork', force=True)
        elif multiprocessing_start_method == 'fork':
            multiprocessing.set_start_method('fork', force=True)
        else:
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
                except Empty:
                    # Timeout
                    if self._debug_mode:
                        print(f'DEBUG: the evaluation time exceeds {timeout_seconds}s.')
                    self._kill_process_and_its_children(process)
                    result = None
                except Exception as e:
                    if self._debug_mode:
                        print(f'DEBUG: evaluation failed with exception:\n{e}')
                    self._kill_process_and_its_children(process)
                    result = None
            else:
                result = result_queue.get()
                self._kill_process_and_its_children(process)
            return result
        except Exception as e:
            if self._debug_mode:
                print(e)
            return None
