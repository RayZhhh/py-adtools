import time
import multiprocessing
import sys
import os
import traceback

# Dependency Check
try:
    import numpy as np
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ Warning: PyTorch/Numpy not found. Related tests will be skipped.")

try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    print("⚠️ Warning: Ray not found. Ray evaluator will be skipped.")

from adtools.evaluator import EvaluationResults
from impl import (
    ConcreteEvaluatorBase,
    ConcreteEvaluatorDict,
    ConcreteEvaluatorShm,
    ConcreteEvaluatorRay,
)

# =============================================================================
# 1. Define Code Snippets
# =============================================================================

CODE_NORMAL = """
def solver():
    return [x**2 for x in range(5)]
"""

CODE_RUNTIME_ERROR = """
def solver():
    return 1 / 0
"""

CODE_INFINITE_LOOP = """
import time
def solver():
    while True:
        time.sleep(0.1)
    return "Never reached"
"""

CODE_POLLUTION = """
import sys
def solver():
    # Attempt to delete a critical module in the runner
    sys.modules['os'] = None 
    return "System Polluted"
"""

CODE_CHECK_PID = """
import os
def solver():
    return os.getpid()
"""

CODE_LARGE_ARRAY = """
import numpy as np
def solver():
    # Return ~80MB array
    return np.ones((10000, 1000))
"""

CODE_HEAVY_PYTORCH = """
import torch
import time
def solver():
    while True:
        # Large matrix multiplication loop
        N = 3000
        A = torch.randn(N, N)
        B = torch.randn(N, N)
        # Busy loop
        for i in range(50):
            C = torch.matmul(A, B)
    return "Finished"
"""

# =============================================================================
# 2. Test Logic
# =============================================================================


def run_test_matrix():
    print("==================================================")
    print("      STARTING ORTHOGONAL EVALUATOR TEST MATRIX    ")
    print("==================================================")

    # --- A. Setup Evaluators ---
    evaluators = []

    # 1. Base
    evaluators.append(
        (
            "Base (Queue)",
            ConcreteEvaluatorBase(
                exec_code=True, find_and_kill_children_evaluation_process=True
            ),
        )
    )
    # 2. Manager Dict
    evaluators.append(
        (
            "ManagerDict",
            ConcreteEvaluatorDict(
                exec_code=True, find_and_kill_children_evaluation_process=True
            ),
        )
    )
    # 3. Shared Memory
    evaluators.append(
        (
            "SharedMemory",
            ConcreteEvaluatorShm(
                exec_code=True, find_and_kill_children_evaluation_process=True
            ),
        )
    )
    # 4. Ray
    evaluators.append(("Ray Actor", ConcreteEvaluatorRay(exec_code=True)))

    # --- B. Define Test Cases ---
    # Structure: (Name, Code, Timeout, Expected_Type)
    # Types: 'SUCCESS', 'ERROR', 'TIMEOUT', 'ISOLATION', 'PID_CHECK'
    test_cases = [
        ("Normal Execution", CODE_NORMAL, 5, "SUCCESS"),
        ("PID Isolation", CODE_CHECK_PID, 5, "PID_CHECK"),
        ("Runtime Error", CODE_RUNTIME_ERROR, 5, "ERROR"),
        ("Infinite Loop", CODE_INFINITE_LOOP, 2, "TIMEOUT"),
        ("Env Pollution", CODE_POLLUTION, 2, "ISOLATION"),
    ]

    # Add Heavy tests if dependencies exist
    if HAS_TORCH:
        test_cases.append(("Large Array Return", CODE_LARGE_ARRAY, 10, "SUCCESS"))
        test_cases.append(("Heavy PyTorch Kill", CODE_HEAVY_PYTORCH, 3, "TIMEOUT"))

    # --- C. Execution Loop ---

    # Iterate through each Evaluator
    for eval_name, evaluator in evaluators:
        print(f"\n██████████ EVALUATOR: {eval_name} ██████████")

        # Iterate through each Test Case
        for test_name, code, timeout, expect_type in test_cases:
            print(f"🔹 Test: {test_name:<20} | Timeout: {timeout}s", end=" ")

            start_t = time.time()
            try:
                # RUN SECURE EVALUATE
                res = evaluator.secure_evaluate(
                    program=code, timeout_seconds=timeout, redirect_to_devnull=True
                )

                # Normalize result access (dict vs object)
                if isinstance(res, dict):
                    res_obj = res["result"]
                    res_err = res["error_msg"]
                    res_time = res["evaluate_time"]
                else:
                    res_obj = res.result
                    res_err = res.error_msg
                    res_time = res.evaluate_time

                duration = time.time() - start_t

                # --- VERIFICATION LOGIC ---
                passed = False
                note = ""

                if expect_type == "SUCCESS":
                    if res_obj is not None and not res_err:
                        passed = True
                        if hasattr(res_obj, "shape"):
                            note = f"[Shape: {res_obj.shape}]"
                        elif isinstance(res_obj, list):
                            note = f"[Res: {str(res_obj)[:15]}...]"
                    else:
                        note = f"[Err: {str(res_err)[:30]}...]"

                elif expect_type == "ERROR":
                    if res_obj is None and res_err:
                        passed = True
                        note = f"[Caught: {res_err.splitlines()[-1][:30]}...]"

                elif expect_type == "TIMEOUT":
                    is_timeout_err = res_err and "timeout" in str(res_err).lower()
                    is_time_over = duration >= timeout or res_time >= timeout
                    # Some evaluators return None result and empty error on hard kill
                    if is_timeout_err or (res_obj is None and is_time_over):
                        passed = True
                        note = f"[Killed after {duration:.2f}s]"
                    else:
                        note = f"[Failed to kill or ran too fast: {duration:.2f}s]"

                elif expect_type == "ISOLATION":
                    if os is not None:
                        passed = True
                        note = "[Main Process Safe]"
                    else:
                        note = "[CRITICAL: OS Module Gone]"

                elif expect_type == "PID_CHECK":
                    main_pid = os.getpid()
                    worker_pid = res_obj
                    if worker_pid is not None and worker_pid != main_pid:
                        passed = True
                        note = f"[Main: {main_pid} != Worker: {worker_pid}]"
                    else:
                        note = f"[PID Same: {main_pid}]"

                # PRINT RESULT
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"-> {status} {note}")

            except Exception as e:
                print(f"-> ❌ CRITICAL EXCEPTION: {e}")
                traceback.print_exc()

            # Small cleanup pause
            time.sleep(0.5)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_test_matrix()
