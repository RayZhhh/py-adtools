import asyncio
import os

import zmq
import zmq.asyncio
import json
import concurrent.futures
from argparse import ArgumentParser
import importlib.util
import sys
from adtools import PyProgram, PyClass
from adtools.evaluator import PyEvaluator, PyEvaluatorRay

__all__ = ["submit_code"]


def submit_code(host, port, code, timeout):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{host}:{port}")
    req_data = {"code": code, "timeout": timeout}
    socket.send(json.dumps(req_data).encode("utf-8"))
    response = socket.recv()
    return json.loads(response.decode("utf-8"))


async def handle_request(
    socket, identity, message, evaluator, default_timeout, semaphore
):
    async with semaphore:
        try:
            # Parse message
            try:
                # message is bytes, decode it
                req_data = json.loads(message.decode("utf-8"))
            except Exception:
                # If parsing fails, create a minimal valid structure to trigger error response
                req_data = {}

            code_str = req_data.get("code")
            # Allow overriding timeout per request
            timeout = req_data.get("timeout", default_timeout)

            if not code_str:
                response = {
                    "result": None,
                    "evaluate_time": 0.0,
                    "error_msg": "No 'code' field in request or invalid JSON",
                }
            else:
                # Offload the blocking evaluation to a separate thread
                # This ensures the main asyncio loop stays responsive
                loop = asyncio.get_running_loop()

                # Create a wrapper to call the synchronous secure_evaluate
                def run_eval():
                    return evaluator.secure_evaluate(code_str, timeout_seconds=timeout)

                # run_in_executor(None, ...) uses the default ThreadPoolExecutor
                results = await loop.run_in_executor(None, run_eval)
                response = dict(results)

            # Ensure serialization
            try:
                json_response = json.dumps(response)
            except (TypeError, OverflowError):
                # If result is not serializable, convert it to string
                response["result"] = str(response["result"])
                json_response = json.dumps(response)

            # Send reply: [identity, empty_delimiter, response_bytes]
            await socket.send_multipart([identity, b"", json_response.encode("utf-8")])

        except Exception as e:
            error_msg = str(e)
            print(f"Server Error processing request from {identity}: {error_msg}")
            try:
                error_response = json.dumps(
                    {
                        "result": None,
                        "evaluate_time": 0.0,
                        "error_msg": f"Server internal error: {error_msg}",
                    }
                )
                await socket.send_multipart(
                    [identity, b"", error_response.encode("utf-8")]
                )
            except Exception:
                pass


async def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--dir", required=True, help="Directory (file path) of the evaluator."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host of the server.")
    parser.add_argument("--port", default=8000, type=int, help="Port of the server.")
    parser.add_argument(
        "-t", "--timeout", default=10, type=float, help="Default timeout in seconds."
    )
    parser.add_argument(
        "--max-workers", default=4, type=int, help="Max concurrent evaluations."
    )
    args = parser.parse_args()

    # Read file
    with open(args.dir) as f:
        program = f.read()

    # Extract all classes
    classes = PyClass.extract_all_classes_from_text(program)

    # Count the number of public classes
    count_public_classes = 0
    public_class_name = None
    for cls in classes:
        if not cls.name.startswith("_"):
            count_public_classes += 1
            public_class_name = cls.name

    if count_public_classes == 0:
        raise Exception("No public classes found.")
    if count_public_classes > 1:
        raise Exception(
            f"The file should only have one pubic class, "
            f"but found {count_public_classes}"
        )

    # Import evaluator from directory
    file_path = os.path.abspath(args.dir)
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    module_name = os.path.splitext(base_name)[0]

    # Add to sys.path for current process
    if dir_name not in sys.path:
        sys.path.insert(0, dir_name)

    # Add to PYTHONPATH for child processes (multiprocessing spawn)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if dir_name not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            (dir_name + os.pathsep + current_pythonpath)
            if current_pythonpath
            else dir_name
        )

    module = importlib.import_module(module_name)
    EvaluatorClass = getattr(module, public_class_name)

    # Assert the evaluator is either "PyEvaluator" or "PyEvaluatorRay"
    if not issubclass(EvaluatorClass, (PyEvaluator, PyEvaluatorRay)):
        raise TypeError(
            f"Class {public_class_name} must inherit from PyEvaluator or PyEvaluatorRay"
        )

    # Instantiate the evaluator
    evaluator = EvaluatorClass()

    # Start zmq server with asyncio
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.ROUTER)
    address = f"tcp://{args.host}:{args.port}"
    socket.bind(address)

    print(f"Evaluator '{public_class_name}' loaded from {args.dir}")
    print(f"Async ZMQ Server running at {address} with max_workers={args.max_workers}")

    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.max_workers)

    while True:
        # Wait for next request
        # ROUTER socket receives: [identity, empty_delimiter, message_bytes]
        try:
            msg_parts = await socket.recv_multipart()
            if len(msg_parts) < 3:
                continue

            identity = msg_parts[0]
            # msg_parts[1] is the delimiter (empty bytes)
            message = msg_parts[2]

            # Spawn a handler
            asyncio.create_task(
                handle_request(
                    socket, identity, message, evaluator, args.timeout, semaphore
                )
            )
        except Exception as e:
            print(f"Main loop error: {e}")


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = ArgumentParser()
