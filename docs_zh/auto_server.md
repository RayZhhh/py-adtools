# 自动服务器实现分析

`adtools.evaluator.auto_server` 模块实现了一个通用的 HTTP 服务器，能够服务于*任何* `PyEvaluator`。

## 1. 动态加载 (`main`)

服务器需要从文件路径加载用户定义的类，而不需要提前知道它的名字。

```python
# adtools/evaluator/auto_server.py

def main():
    # ... 解析参数 ...

    # 1. 读取并解析用户文件
    with open(args.dir) as f:
        program = f.read()
    
    # 使用 PyClass 解析器查找类名
    classes = PyClass.extract_all_classes_from_text(program)

    # 2. 查找“公共”类
    # 我们查找不以 "_" 开头的单个类
    public_class_name = None
    for cls in classes:
        if not cls.name.startswith("_"):
            public_class_name = cls.name
            break
            
    # 3. 导入模块
    # 我们修改 sys.path 以允许从该目录导入
    sys.path.insert(0, dir_name)
    module = importlib.import_module(module_name)
    
    # 4. 获取类并实例化
    EvaluatorClass = getattr(module, public_class_name)
    evaluator = EvaluatorClass()
```

**实现洞察**：
- **解析与导入**：我们首先将代码作为*文本*解析以查找类名。这允许我们在尝试导入模块*之前*应用逻辑（如“必须只有一个公共类”）。
- **`sys.path` 操作**：为了导入像 `./usage/my_eval.py` 这样的文件，我们必须将 `./usage` 添加到 `sys.path`。

## 2. 请求处理 (`EvaluationHandler`)

服务器使用 `ThreadingHTTPServer` 处理并发请求。

```python
# adtools/evaluator/auto_server.py

class EvaluationHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # ... 解析 JSON 主体 ...
        
        # 1. 并发控制
        # self.server.semaphore 使用 max_workers 初始化
        with self.server.semaphore:
            # 此块是线程安全的，并且限制为 N 个线程
            results = self.server.evaluator.secure_evaluate(
                code_str, 
                timeout_seconds=timeout
            )
            
        # 2. 发送响应
        self.wfile.write(json.dumps(results).encode("utf-8"))
```

**实现洞察**：
- **信号量（Semaphore）**：由于 `secure_evaluate` 会产生一个进程（资源密集型），我们不能允许无限的并发请求。信号量确保如果 100 个请求进来但 `max_workers=4`，则 96 个请求将在队列中等待（阻塞其线程），直到有一个 worker 空闲。
