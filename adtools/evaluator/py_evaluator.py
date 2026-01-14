"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, List

from adtools.py_code import PyProgram
from adtools.sandbox import SandboxExecutor, ExecutionResults

__all__ = [
    "PyEvaluator",
]


class PyEvaluator(ABC):

    def __init__(
        self,
        exec_code: bool = True,
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
        self.debug_mode = debug_mode
        self.exec_code = exec_code
        self.sandbox_executor = SandboxExecutor(
            evaluate_worker=self,
            find_and_kill_children_evaluation_process=find_and_kill_children_evaluation_process,
            debug_mode=debug_mode,
            join_timeout_seconds=join_timeout_seconds,
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

    def _exec_and_get_res(self, program: str | PyProgram, **kwargs):
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
            callable_funcs_list = [
                all_globals_namespace[f_name] for f_name in function_names
            ]
            callable_funcs_dict = dict(zip(function_names, callable_funcs_list))
            # Get callable classes
            callable_cls_list = [
                all_globals_namespace[c_name] for c_name in class_names
            ]
            callable_cls_dict = dict(zip(class_names, callable_cls_list))
        else:
            (
                callable_funcs_list,
                callable_funcs_dict,
                callable_cls_list,
                callable_cls_dict,
            ) = (None, None, None, None)

        # Get evaluate result
        res = self.evaluate_program(
            str(program),
            callable_funcs_dict,
            callable_funcs_list,
            callable_cls_dict,
            callable_cls_list,
            **kwargs,
        )
        return res

    def evaluate(self, program: str | PyProgram, **kwargs) -> ExecutionResults:
        start_time = time.time()
        error_msg = ""
        # noinspection PyBroadException
        try:
            res = self._exec_and_get_res(program, **kwargs)
        except:
            res = None
            error_msg = str(traceback.format_exc())

        return ExecutionResults(
            result=res, evaluate_time=time.time() - start_time, error_msg=error_msg
        )

    def secure_evaluate(
        self,
        program: str | PyProgram,
        timeout_seconds: int | float = None,
        redirect_to_devnull: bool = False,
        **kwargs,
    ) -> ExecutionResults:
        """Evaluate program in a new process. This enables timeout restriction and output redirection.

        Args:
            program: the program to be evaluated.
            timeout_seconds: return 'None' if the execution time exceeds 'timeout_seconds'.
            redirect_to_devnull: redirect any output to '/dev/null'.
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
        )
