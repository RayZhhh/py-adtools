# Auto-Server Implementation Analysis

The `adtools.evaluator.auto_server` module implements a generic HTTP server capable of serving *any* `PyEvaluator`.

## 1. Dynamic Loading (`main`)

The server needs to load a user-defined class from a file path without knowing its name in advance.

```python
# adtools/evaluator/auto_server.py

def main():
    # ... parse args ...

    # 1. Read and Parse the user's file
    with open(args.dir) as f:
        program = f.read()
    
    # Use PyClass parser to find class names
    classes = PyClass.extract_all_classes_from_text(program)

    # 2. Find the "Public" Class
    # We look for the single class that doesn't start with "_"
    public_class_name = None
    for cls in classes:
        if not cls.name.startswith("_"):
            public_class_name = cls.name
            break
            
    # 3. Import the Module
    # We modify sys.path to allow importing from that directory
    sys.path.insert(0, dir_name)
    module = importlib.import_module(module_name)
    
    # 4. Get the Class and Instantiate
    EvaluatorClass = getattr(module, public_class_name)
    evaluator = EvaluatorClass()
```

**Implementation Insight**:
- **Parsing vs Importing**: We parse the code *as text* first to find the class name. This allows us to apply logic (like "must have exactly one public class") *before* we even try to import the module.
- **`sys.path` Manipulation**: To import a file like `./usage/my_eval.py`, we must add `./usage` to `sys.path`.

## 2. Request Handling (`EvaluationHandler`)

The server handles concurrent requests using a `ThreadingHTTPServer`.

```python
# adtools/evaluator/auto_server.py

class EvaluationHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # ... parse JSON body ...
        
        # 1. Concurrency Control
        # self.server.semaphore is initialized with max_workers
        with self.server.semaphore:
            # This block is thread-safe and limited to N threads
            results = self.server.evaluator.secure_evaluate(
                code_str, 
                timeout_seconds=timeout
            )
            
        # 2. Send Response
        self.wfile.write(json.dumps(results).encode("utf-8"))
```

**Implementation Insight**:
- **Semaphore**: Since `secure_evaluate` spawns a process (which is resource-intensive), we cannot allow unlimited concurrent requests. The semaphore ensures that if 100 requests come in but `max_workers=4`, 96 requests will wait in the queue (blocking their threads) until a worker is free.
