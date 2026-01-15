import json
import os
import sys
import argparse
import importlib.util
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Dict

import requests

from adtools import PyClass
from adtools.evaluator import PyEvaluator, PyEvaluatorRay

__all__ = ["submit_code", "submit_code_async"]


def submit_code(
    host: str,
    port: int | str,
    code: str,
    timeout: Optional[float] = None,
    *,
    post_timeout_seconds: float = 1800,
) -> Dict:
    """Submit code to the evaluation server.

    Args:
        host: Server host.
        port: Server port.
        code: Code to submit.
        timeout: evaluation timeout in seconds.
        post_timeout_seconds: Post request timeout in seconds.

    Returns:
        A dict containing the evaluation metadata.
    """
    url = f"http://{host}:{port}/"
    payload = {"code": code, "timeout": timeout}

    with requests.Session() as s:
        r = s.post(url, json=payload, timeout=post_timeout_seconds)
        r.raise_for_status()
        return r.json()


async def submit_code_async(
    host: str,
    port: int | str,
    code: str,
    timeout: Optional[float] = None,
    *,
    post_timeout_seconds: float = 1800,
) -> Dict:
    """Submit code to the evaluation server.

    Args:
        host: Server host.
        port: Server port.
        code: Code to submit.
        timeout: evaluation timeout in seconds.
        post_timeout_seconds: Post request timeout in seconds.

    Returns:
        A dict containing the evaluation metadata.
    """
    try:
        import aiohttp
    except ImportError:
        raise ImportError("Please install 'aiohttp'.")

    url = f"http://{host}:{port}/"
    payload = {"code": code, "timeout": timeout}
    timeout_cfg = aiohttp.ClientTimeout(total=post_timeout_seconds)

    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


class EvaluationHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            self.connection.settimeout(10)
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                req_data = json.loads(post_data.decode("utf-8"))
            except Exception:
                req_data = {}

            code_str = req_data.get("code")
            # Use server-level default timeout if not provided
            timeout = req_data.get("timeout", self.server.default_timeout)

            if not code_str:
                response = {
                    "result": None,
                    "evaluate_time": 0.0,
                    "error_msg": "No 'code' field in request or invalid JSON",
                }
            else:
                # Use the semaphore to limit concurrent evaluations
                # This blocks the thread handling this request until a slot is free
                with self.server.semaphore:
                    results = self.server.evaluator.secure_evaluate(
                        code_str, timeout_seconds=timeout
                    )
                    response = dict(results)

            # Ensure serialization
            try:
                json_response = json.dumps(response)
            except (TypeError, OverflowError):
                # If result is not serializable, convert it to string
                response["result"] = str(response["result"])
                json_response = json.dumps(response)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json_response.encode("utf-8"))

        except Exception as e:
            error_msg = str(e)
            print(f"Server Error: {error_msg}")
            error_response = json.dumps(
                {
                    "result": None,
                    "evaluate_time": 0.0,
                    "error_msg": f"Server internal error: {error_msg}",
                }
            )
            # We still return 200 OK because the *HTTP* request succeeded,
            # but the *application* (evaluation) had an error.
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(error_response.encode("utf-8"))

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(ThreadingHTTPServer):
    # Allow passing custom attributes to the server instance
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", required=True, help="Directory (file path) of the evaluator."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host of the server.")
    parser.add_argument("--port", default=8000, type=int, help="Port of the server.")
    parser.add_argument(
        "-t", "--timeout", default=None, type=float, help="Default timeout in seconds."
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

    # Check whether timeout is set
    timeout_defined_in_class = None
    for field in ["timeout_seconds, timeout, _timeout_seconds, _timeout"]:
        if hasattr(evaluator, field):
            timeout_defined_in_class = getattr(evaluator, field)

    # Initialize Threaded HTTP Server
    # We use ThreadedHTTPServer to handle requests in separate threads
    server = ThreadedHTTPServer((args.host, args.port), EvaluationHandler)

    # Attach shared resources to the server instance so handlers can access them
    server.evaluator = evaluator
    server.default_timeout = args.timeout or timeout_defined_in_class
    server.semaphore = threading.Semaphore(args.max_workers)

    print(f"Evaluator '{public_class_name}' loaded from {args.dir}")
    print(
        f"HTTP Server running at http://{args.host}:{args.port} with max_workers={args.max_workers}"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
