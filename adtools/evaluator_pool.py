"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Literal, Optional

from .evaluator import PyEvaluator
from .py_code import PyProgram


class EvaluatorExecutorPool:
    def __init__(
            self,
            evaluator: PyEvaluator,
            max_workers: int,
            pool_type: Literal['thread', 'process'] = 'thread'
    ):
        """Multi-thread/process executor pool for parallel evaluation.
        Args:
            evaluator: The PyEvaluator instance.
            max_workers: The maximum number of workers.
            pool_type: Type of the executor pool.
        """
        self.evaluator = evaluator
        self.max_workers = max_workers
        if pool_type == 'thread':
            self.pool = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.pool = ProcessPoolExecutor(max_workers=self.max_workers)

    def evaluate(self, program: str | PyProgram, return_time=True, **kwargs):
        """Evaluate program.
        Args:
            program: the program to be evaluated.
            **kwargs: additional keyword arguments to pass to 'evaluate_program'.
        """
        start_time = time.time()
        future = self.pool.submit(self.evaluator.evaluate, program, **kwargs)
        res = future.result()
        duration = time.time() - start_time
        if return_time:
            return res, duration
        else:
            return res

    def secure_evaluate(
            self,
            program: str | PyProgram,
            timeout_seconds: Optional[float],
            redirect_to_devnull: bool = False,
            multiprocessing_start_method: Literal['default', 'auto', 'fork', 'spawn'] = 'auto',
            return_time=True,
            **kwargs
    ):
        """Evaluate program in a new process. This enables timeout restriction and output redirection.
        Args:
            program: the program to be evaluated.
            timeout_seconds: return 'None' if the execution time exceeds 'timeout_seconds'.
            redirect_to_devnull: redirect any output to '/dev/null'.
            multiprocessing_start_method: start a process using 'fork' or 'spawn'.
            **kwargs: additional keyword arguments to pass to 'evaluate_program'.
        """
        start_time = time.time()
        future = self.pool.submit(
            self.evaluator.secure_evaluate,
            program,
            timeout_seconds,
            redirect_to_devnull,
            multiprocessing_start_method,
            **kwargs
        )
        res = future.result()
        duration = time.time() - start_time
        if return_time:
            return res, duration
        else:
            return res
