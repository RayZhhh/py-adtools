"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import multiprocessing
import os
import pickle
import sys
import time
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from queue import Empty
from typing import Any, Literal, Dict, Callable, List, Tuple
import multiprocessing.managers
import psutil
import traceback

from .py_code import PyProgram

__all__ = ['PyEvaluator', 'PyEvaluatorForBigReturnedObject']


def _set_mp_start_method(multiprocessing_start_method: Literal['default', 'auto', 'fork', 'spawn']):
    if multiprocessing_start_method == 'auto':
        # Force macOS and Linux use 'fork' to generate new process
        if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
            multiprocessing.set_start_method('fork', force=True)
    elif multiprocessing_start_method == 'fork':
        multiprocessing.set_start_method('fork', force=True)
    elif multiprocessing_start_method == 'spawn':
        multiprocessing.set_start_method('spawn', force=True)


class PyEvaluator(ABC):

    def __init__(
            self,
            exec_code: bool = True,
            find_and_kill_children_evaluation_process: bool = False,
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
            find_and_kill_children_evaluation_process: If using 'self.secure_evaluate', kill children processes
                when they are terminated. Note that it is suggested to set to 'False' if the evaluation process
                does not start new processes.
            debug_mode: Debug mode.
            join_timeout_seconds: Timeout in seconds to wait for the process to finish. Kill the process if timeout.
        """
        self.debug_mode = debug_mode
        self.exec_code = exec_code
        self.find_and_kill_children_evaluation_process = find_and_kill_children_evaluation_process
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

    def evaluate(self, program: str | PyProgram, **kwargs):
        """Evaluate a program.
        Args:
            program: the program to be evaluated.
            **kwargs: additional keyword arguments to pass to 'evaluate_program'.
        """
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
            get_evaluate_time=False,
            **kwargs
    ) -> Any | Tuple[Any, float]:
        """Evaluate program in a new process. This enables timeout restriction and output redirection.
        Args:
            program: the program to be evaluated.
            timeout_seconds: return 'None' if the execution time exceeds 'timeout_seconds'.
            redirect_to_devnull: redirect any output to '/dev/null'.
            multiprocessing_start_method: start a process using 'fork' or 'spawn'. If set to 'auto',
                the process will be started using 'fork' with Linux/macOS and 'spawn' with Windows.
                If set to 'default', there will be no changes to system default.
            get_evaluate_time: get evaluation time for this program.
            **kwargs: additional keyword arguments to pass to 'evaluate_program'.
        Returns:
            Returns the evaluation results. If the 'get_evaluate_time' is True,
            the return value will be (Results, Time).
        """
        _set_mp_start_method(multiprocessing_start_method)

        try:
            # Start evaluation process
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=self._evaluate_in_safe_process,
                args=(str(program), result_queue, redirect_to_devnull),
                kwargs=kwargs,
            )
            evaluate_start_time = time.time()
            process.start()

            if timeout_seconds is not None:
                try:
                    # Get the result in timeout seconds
                    result = result_queue.get(timeout=timeout_seconds)
                    # Calculate the evaluate time
                    eval_time = time.time() - evaluate_start_time
                    # After getting the result, terminate and kill the process
                    self._kill_process_and_its_children(process)
                except Empty:  # The queue is empty indicates a timeout
                    # Calculate the evaluate time
                    eval_time = time.time() - evaluate_start_time
                    if self.debug_mode:
                        print(f'DEBUG: the evaluation time exceeds {timeout_seconds}s.')
                    # Terminate and kill all processes if timeout happens
                    self._kill_process_and_its_children(process)
                    result = None
                except Exception as e:
                    # Calculate the evaluate time
                    eval_time = time.time() - evaluate_start_time
                    if self.debug_mode:
                        print(f'DEBUG: evaluation failed with exception:\n{traceback.format_exc()}')
                    # Terminate and kill all processes if meet exceptions
                    self._kill_process_and_its_children(process)
                    result = None
            else:
                # If there is no timeout limit, wait execution to finish
                result = result_queue.get()
                # Calculate the evaluate time
                eval_time = time.time() - evaluate_start_time
                # Terminate and kill all processes after evaluation
                self._kill_process_and_its_children(process)

            return (result, eval_time) if get_evaluate_time else result
        except Exception as e:
            if self.debug_mode:
                traceback.print_exc()
            return None


class PyEvaluatorForBigReturnedObject(PyEvaluator):
    def __init__(
            self,
            exec_code: bool = True,
            find_and_kill_children_evaluation_process: bool = False,
            debug_mode: bool = False,
            *,
            join_timeout_seconds: int = 10
    ):
        """Evaluator interface for evaluating the Python algorithm program. Override this class and implement
         'evaluate_program' method, then invoke 'self.evaluate()' or 'self.secure_evaluate()' for evaluation.

         **Note:** This class supports the secure_evaluate to handle very big return object, e.g., Tensors.

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
        super().__init__(
            exec_code,
            find_and_kill_children_evaluation_process,
            debug_mode,
            join_timeout_seconds=join_timeout_seconds
        )

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
        raise NotImplementedError(
            'Must provide an evaluator for a python program. '
            'Override this method in a subclass.'
        )

    def _evaluate_and_put_res_in_manager_dict(
            self,
            program_str: str,
            result_dict: multiprocessing.managers.DictProxy,
            signal_queue: multiprocessing.Queue,
            redirect_to_devnull: bool,
            **kwargs
    ):
        """Evaluate and store result in Manager().dict() (for large results)."""
        if redirect_to_devnull:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), sys.stdout.fileno())
                os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            # Evaluate and get results
            res = self.evaluate(program_str, **kwargs)
            # Write results into dict
            result_dict['result'] = res
            # Put a signal to queue to inform the parent process the evaluation has done
            signal_queue.put(('ok', None))
        except Exception as e:
            if self.debug_mode:
                traceback.print_exc()
            # Write results into dict
            result_dict['result'] = None
            # Put a signal to queue to inform the parent process the evaluation has terminated
            signal_queue.put(('error', str(e)))

    def secure_evaluate(
            self,
            program: str | PyProgram,
            timeout_seconds: int | float = None,
            redirect_to_devnull: bool = False,
            multiprocessing_start_method: Literal['default', 'auto', 'fork', 'spawn'] = 'auto',
            get_evaluate_time: bool = False,
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
            get_evaluate_time: get evaluation time for this program.
            **kwargs: additional keyword arguments to pass to 'evaluate_program'.
        Returns:
            Returns the evaluation results. If the 'get_evaluate_time' is True,
            the return value will be (Results, Time).
        """
        _set_mp_start_method(multiprocessing_start_method)

        with multiprocessing.Manager() as manager:
            # Path a dictionary to the evaluation process to get maybe very big return objects
            result_dict = manager.dict()
            # Pass a queue to the evaluation process to get signals whether the evaluation terminates
            signal_queue = multiprocessing.Queue()
            # Start evaluation process
            process = multiprocessing.Process(
                target=self._evaluate_and_put_res_in_manager_dict,
                args=(str(program), result_dict, signal_queue, redirect_to_devnull),
                kwargs=kwargs,
            )
            evaluate_start_time = time.time()
            process.start()

            try:
                if timeout_seconds is not None:
                    try:
                        # If there is timeout restriction, we try to get results before timeout
                        signal = signal_queue.get(timeout=timeout_seconds)
                    except Empty:
                        # Evaluation timeout happens, we return 'None' as well as the actual evaluate time
                        eval_time = time.time() - evaluate_start_time
                        if self.debug_mode:
                            print(f'DEBUG: evaluation time exceeds {timeout_seconds}s.')
                        # Terminate and kill all processes after evaluation
                        self._kill_process_and_its_children(process)
                        return (None, eval_time) if get_evaluate_time else None
                else:
                    # If there is no timeout restriction, we wait until the evaluation terminates
                    signal = signal_queue.get()

                # Calculate evaluation time and kill children processes
                eval_time = time.time() - evaluate_start_time
                # Terminate and kill all processes after evaluation
                self._kill_process_and_its_children(process)

                # The first element is 'ok' indicates that the evaluation terminate without exceptions
                if signal[0] == 'ok':
                    # We get the evaluation results from 'manager.dict'
                    result = result_dict.get('result', None)
                else:
                    # The evaluation failed for some reason, so we set the result to 'None'
                    if self.debug_mode:
                        print(f'DEBUG: child process error: {signal[1]}')
                    result = None
            except:
                # If there is any exception during above procedure, we set the result to None
                eval_time = time.time() - evaluate_start_time
                if self.debug_mode:
                    print(f'DEBUG: exception in manager evaluate:\n{traceback.format_exc()}')
                # Terminate and kill all processes after evaluation
                self._kill_process_and_its_children(process)
                result = None

        return (result, eval_time) if get_evaluate_time else result


class PyEvaluatorForBigReturnedObjectV2(PyEvaluator):

    def __init__(
            self,
            exec_code: bool = True,
            find_and_kill_children_evaluation_process: bool = False,
            debug_mode: bool = False,
            *,
            join_timeout_seconds: int = 10
    ):
        """Evaluator interface for evaluating the Python algorithm program. Override this class and implement
        'evaluate_program' method, then invoke 'self.evaluate()' or 'self.secure_evaluate()' for evaluation.
        Note: This class supports the secure_evaluate to handle very big return object, e.g., Tensors.

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
        super().__init__(
            exec_code,
            find_and_kill_children_evaluation_process,
            debug_mode,
            join_timeout_seconds=join_timeout_seconds
        )

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

    def _evaluate_and_put_res_in_shared_memory(
            self,
            program_str: str,
            meta_queue: multiprocessing.Queue,
            redirect_to_devnull: bool,
            **kwargs
    ):
        """Evaluate and store result in shared memory (for large results)."""
        # Redirect STDOUT and STDERR to '/dev/null'
        if redirect_to_devnull:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), sys.stdout.fileno())
                os.dup2(devnull.fileno(), sys.stderr.fileno())

        # Evaluate and get results
        res = self.evaluate(program_str, **kwargs)

        try:
            # Dump the results to data
            data = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
            # Create shared memory with the size of data
            shm = shared_memory.SharedMemory(create=True, size=len(data))
            # Write data
            shm.buf[:len(data)] = data
            # Send back shm metadata (shared_mem_name, shared_mem_size) and put them into the queue
            meta_queue.put((shm.name, len(data)))
            # Child closes its handle
            shm.close()
        except Exception as data_pickle_error:
            # Put the exception message to the queue
            meta_queue.put((None, str(data_pickle_error)))

    def secure_evaluate(
            self,
            program: str | PyProgram,
            timeout_seconds: int | float = None,
            redirect_to_devnull: bool = False,
            multiprocessing_start_method: str = 'auto',
            get_evaluate_time: bool = False,
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
            get_evaluate_time: get evaluation time for this program.
            **kwargs: additional keyword arguments to pass to 'evaluate_program'.
        Returns:
            Returns the evaluation results. If the 'get_evaluate_time' is True,
            the return value will be (Results, Time).
        """
        if multiprocessing_start_method == 'auto':
            if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
                multiprocessing.set_start_method('fork', force=True)
        elif multiprocessing_start_method == 'fork':
            multiprocessing.set_start_method('fork', force=True)
        elif multiprocessing_start_method == 'spawn':
            multiprocessing.set_start_method('spawn', force=True)

        meta_queue = multiprocessing.Queue()

        process = multiprocessing.Process(
            target=self._evaluate_and_put_res_in_shared_memory,
            args=(str(program), meta_queue, redirect_to_devnull),
            kwargs=kwargs,
        )

        evaluate_start_time = time.time()
        process.start()

        try:
            if timeout_seconds is not None:
                try:
                    # Try to get the metadata before timeout
                    meta = meta_queue.get(timeout=timeout_seconds)
                except Empty:
                    # Evaluate timeout
                    eval_time = time.time() - evaluate_start_time
                    if self.debug_mode:
                        print(f'DEBUG: evaluation time exceeds {timeout_seconds}s.')
                    self._kill_process_and_its_children(process)
                    return (None, eval_time) if get_evaluate_time else None
            else:
                meta = meta_queue.get()

            # Calculate evaluation time
            eval_time = time.time() - evaluate_start_time
            self._kill_process_and_its_children(process)

            # If the first element in the queue is None,
            # it means that the shared memory raises exceptions
            if meta[0] is None:
                if self.debug_mode:
                    print(f'DEBUG: shared memory failed with exception: {meta[1]}')
                result = None
            else:
                # Read results from metadata
                shm_name, size = meta
                shm = shared_memory.SharedMemory(name=shm_name)
                buf = bytes(shm.buf[:size])
                # Load results from buffer
                result = pickle.loads(buf)
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        except Exception:
            eval_time = time.time() - evaluate_start_time
            if self.debug_mode:
                print(f'DEBUG: exception in shared evaluate:\n{traceback.format_exc()}')
            self._kill_process_and_its_children(process)
            result = None

        return (result, eval_time) if get_evaluate_time else result
