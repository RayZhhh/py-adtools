# Py-ADTools: 用于 LLM 辅助算法设计与代码优化的代码解析、沙盒及评估工具

<div align="center">
<a href="https://github.com/RayZhhh/py-adtools"><img src="https://img.shields.io/github/stars/RayZhhh/py-adtools?style=social" alt="GitHub stars"></a>
<a href="https://github.com/RayZhhh/py-adtools/blob/main/LICENSE"><img src="https://img.shields.io/github/license/RayZhhh/py-adtools" alt="License"></a>
<a href="https://deepwiki.com/RayZhhh/py-adtools"><img src="./assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
</div>
<br>

下图展示了 Python 程序是如何通过 `adtools` 被解析为 [PyCodeBlock](./adtools/py_code.py#L18-L33)、[PyFunction](./adtools/py_code.py#L37-L126)、[PyClass](./adtools/py_code.py#L129-L206) 和 [PyProgram](./adtools/py_code.py#L209-L256) 的。

![pycode](./assets/pycode.png)

------

## 安装

> [!TIP]
> 
> 建议使用 Python >= 3.10。

运行以下命令安装 adtools：

```shell
pip install git+https://github.com/RayZhhh/py-adtools.git
```

或者通过 pip 安装：

```shell
pip install py-adtools
```

## 教程

| 教程 | Colab |
| :--- | :--- |
| **01. 代码解析** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayZhhh/py-adtools/blob/main/tutorial/01_py_code.ipynb) |
| **02. 安全执行** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayZhhh/py-adtools/blob/main/tutorial/02_sandbox.ipynb) |
| **03. 装饰器** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayZhhh/py-adtools/blob/main/tutorial/03_decorators.ipynb) |
| **04. 评估器** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayZhhh/py-adtools/blob/main/tutorial/04_evaluator.ipynb) |

## 代码解析工具 [py_code](./adtools/py_code.py)

[adtools.py_code](./adtools/py_code.py) 提供了强大的功能，可以将 Python 程序解析为结构化组件，便于操作、修改和分析。

### 核心组件

解析器将 Python 代码分解为四个主要数据结构：

| **组件** | **描述** | **关键属性** |
|---|---|---|
| **PyProgram** | 代表整个文件。它维护脚本、函数和类的确切顺序。 | `functions`, `classes`, `scripts`, `elements` |
| **PyFunction** | 代表顶级函数或类方法。你可以动态修改其签名、装饰器、文档字符串或函数体。 | `name`, `args`, `body`, `docstring`, `decorator`, `return_type` |
| **PyClass** | 代表类定义。它是方法和类级语句的容器。 | `name`, `bases`, `functions` (methods), `body` |
| **PyCodeBlock** | 代表原始代码段，如导入、全局变量或类内部的特定逻辑块。 | `code` |

### 基础用法

```python
from adtools import PyProgram

code = r"""
import ast, numba                 # 这部分将被解析为 PyCodeBlock
import numpy as np

@numba.jit()                      # 这部分将被解析为 PyFunction
def function(arg1, arg2=True):     
    '''Docstring.
    This is a function.
    '''
    if arg2:
    	return arg1 * 2
    else:
    	return arg1 * 4

@some.decorators()                # 这部分将被解析为 PyClass
class PythonClass(BaseClass):
    '''Docstring.'''
    # Comments
    class_var1 = 1                # 这部分将被解析为 PyCodeBlock
    class_var2 = 2                # 并放置在 PyClass.body 中

    def __init__(self, x):        # 这部分将被解析为 PyFunction
        self.x = x                # 并放置在 PyClass.functions 中

    def method1(self):
        '''Docstring.
        This is a class method.
        '''
        return self.x * 10

    @some.decorators()
    def method2(self, x, y):
    	return x + y + self.method1(x)
    
    @some.decorators(100)  
    class InnerClass:             # 这部分将被解析为 PyCodeBlock
        '''Docstring.'''
        def __init__(self):       # 并放置在 PyClass.body 中
            ...

if __name__ == '__main__':        # 这部分将被解析为 PyCodeBlock
	res = function(1)
	print(res)
	res = PythonClass().method2(1, 2)
"""

p = PyProgram.from_text(code, debug=True)
print(p)
print(f"-------------------------------------")
print(p.classes[0].functions[1])
print(f"-------------------------------------")
print(p.classes[0].functions[2].decorator)
print(f"-------------------------------------")
print(p.functions[0].name)

```

### 关键特性

- **保留代码结构**：保持原始缩进和格式。
- **处理多行字符串**：正确保留多行字符串内容，不会破坏缩进。
- **访问组件**：轻松访问函数、类和代码块。
- **修改代码元素**：以编程方式更改函数名称、文档字符串或主体内容。
- **完整的程序表示**：[PyProgram](./adtools/py_code.py#L209-L256) 保持元素在源代码中出现的各种确切顺序。

## 安全执行工具 `sandbox`

`adtools.sandbox` 为运行不受信任的代码提供了安全执行环境。它将执行隔离在单独的进程中，支持超时管理、资源保护和输出重定向。

### 基础用法

你可以用 `SandboxExecutor` 包装任何类或对象，以便在单独的进程中执行其方法。

```python
import time
from typing import Any
from adtools.sandbox.sandbox_executor import SandboxExecutor


class SortAlgorithmEvaluator:
    def evaluate_program(self, program: str) -> Any | None:
        g = {}
        exec(program, g)
        sort_algo = g.get("merge_sort")
        if not sort_algo: return None

        input_data = [10, 2, 4, 76, 19, 29, 3, 5, 1]
        start = time.time()
        res = sort_algo(input_data)
        duration = time.time() - start

        return duration if res == sorted(input_data) else None


code_generated_by_llm = """
def merge_sort(arr):
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    return merge(left, right)
"""

if __name__ == "__main__":
    # 使用 worker 实例初始化 SandboxExecutor
    sandbox = SandboxExecutor(SortAlgorithmEvaluator(), debug_mode=True)

    # 安全地执行方法
    score = sandbox.secure_execute(
        "evaluate_program",
        method_args=(code_generated_by_llm,),
        timeout_seconds=10
    )
    print(f"Score: {score}")
```

### 沙盒执行器

`adtools` 提供了两种沙盒实现：

- **[SandboxExecutor](./adtools/sandbox/sandbox_executor.py)**
    - 基于标准 `multiprocessing` 的沙盒。
    - 通过共享内存捕获返回值。
    - 支持超时和输出重定向。

- **[SandboxExecutorRay](./adtools/sandbox/sandbox_executor_ray.py)**
    - 基于 Ray 的沙盒，用于分布式执行。
    - 适合需要更强隔离或基于集群的评估场景。

### 装饰器用法

对于更简单的用例，你可以使用 `@sandbox_run` 装饰器自动在沙盒中执行函数或方法。

```python
from adtools.sandbox import sandbox_run

@sandbox_run(timeout=5.0)
def calculate(x):
    return x ** 2

# 在单独的进程中执行
res = calculate(10)
print(f"Result: {res['result']}, Time: {res['evaluate_time']}")
```

## 代码评估工具 `evaluator`

`adtools.evaluator` 提供了多种安全评估选项，用于运行和测试 Python 代码。

### 基础用法

```python
import time
from typing import Dict, Callable, List, Any

from adtools.evaluator import PyEvaluator


class SortAlgorithmEvaluator(PyEvaluator):
    def evaluate_program(
            self,
            program_str: str,
            callable_functions_dict: Dict[str, Callable] | None,
            callable_functions_list: List[Callable] | None,
            callable_classes_dict: Dict[str, Callable] | None,
            callable_classes_list: List[Callable] | None,
            **kwargs,
    ) -> Any | None:
        """评估给定的排序算法程序。
        Args:
            program_str            : 原始程序文本。
            callable_functions_dict: 映射函数名到可调用函数的字典。
            callable_functions_list: 可调用函数的列表。
            callable_classes_dict  : 映射类名到可调用类的字典。
            callable_classes_list  : 可调用类的列表。
        Return:
            返回评估结果。
        """
        # 获取排序算法
        sort_algo: Callable = callable_functions_dict["merge_sort"]
        # 测试数据
        input = [10, 2, 4, 76, 19, 29, 3, 5, 1]
        # 计算执行时间
        start = time.time()
        res = sort_algo(input)
        duration = time.time() - start
        if res == sorted(input):  # 如果结果正确
            return duration  # 返回执行时间作为算法的分数
        else:
            return None  # 返回 None 表示算法不正确


code_generated_by_llm = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2              
    left = merge_sort(arr[:mid])     
    right = merge_sort(arr[mid:])   

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result
"""

harmful_code_generated_by_llm = """
def merge_sort(arr):
    print('I am harmful')  # 默认情况下 STDOUT 会被重定向到 /dev/null，因此不会有输出。
    while True:
        pass
"""

if __name__ == "__main__":
    evaluator = SortAlgorithmEvaluator()

    # 评估
    score = evaluator._exec_and_get_res(code_generated_by_llm)
    print(f"Score: {score}")

    # 安全评估（评估在沙盒进程中执行）
    score = evaluator.secure_evaluate(code_generated_by_llm, timeout_seconds=10)
    print(f"Score: {score}")

    # 评估有害代码，评估将在 10 秒内终止
    # 由于违反时间限制，我们将获得 `None` 分数
    score = evaluator.secure_evaluate(harmful_code_generated_by_llm, timeout_seconds=10)
    print(f"Score: {score}")

```

### 评估器类型及其特性

`adtools` 提供了两种不同的评估器实现，各自针对不同场景进行了优化：

- **[PyEvaluator](./adtools/evaluator/py_evaluator.py)**
    - *使用共享内存*处理极大的返回对象（例如大型张量）。
    - 针对海量数据*避免 pickle 序列化开销*。
    - *最适合高性能场景*，特别是涉及大型结果对象时。
    - *用例*：评估产生大型张量或数组的 ML 算法。

- **[PyEvaluatorRay](./adtools/evaluator/py_evaluator_ray.py)**
    - *利用 Ray* 进行分布式、安全评估。
    - *支持零拷贝返回*大型对象。
    - *适合集群环境*或需要最大隔离时。
    - *用例*：跨多台机器的大规模评估或使用 GPU 资源时。

所有评估器都通过抽象的 [PyEvaluator](./adtools/evaluator/py_evaluator.py) 类共享相同的接口，使得根据具体需求在不同实现之间切换变得容易。

## 实际应用

### 用于代码操作的解析器

解析器旨在处理复杂场景，包括**多行字符串**、**装饰器**和**缩进管理**。

```python
from adtools import PyProgram

# 一段包含导入、装饰器和类的复杂代码
code = r'''
import numpy as np

@jit(nopython=True)
def heuristics(x):
    """Calculates the heuristic value."""
    return x * 0.5

class EvolutionStrategy:
    population_size = 100
    
    def __init__(self, mu, lambda_):
        self.mu = mu
        self.lambda_ = lambda_
        
    def mutate(self, individual):
        # Apply mutation
        return individual + np.random.normal(0, 1)
'''

# 1. 解析程序
program = PyProgram.from_text(code)

# 2. 访问和修改函数
func = program.functions[0]
print(f"Function detected: {func.name}")
# Output: Function detected: heuristics

# 编程方式修改函数
func.name = "fast_heuristics"
func.decorator = None  # 移除装饰器
func.docstring = "Optimized heuristic calculation."

# 3. 访问类方法
cls_obj = program.classes[0]
init_method = cls_obj.functions[0]
mutate_method = cls_obj.functions[1]

print(f"Class: {cls_obj.name}, Method: {mutate_method.name}")
# Output: Class: EvolutionStrategy, Method: mutate

# 4. 生成修改后的代码
# PyProgram 对象会重建代码并保留原始顺序
print("\n--- Reconstructed Code ---")
print(program)

```

### 用于提示词（Prompt）构建的解析器

`adtools` 在基于 LLM 的算法设计中特别强大，此时你需要管理生成的代码种群，标准化提示词格式，或将生成的逻辑注入现有模板中。

在基于 LLM 的自动化算法设计（LLM-AAD）中，你通常维护一个算法种群。你可能需要重命名它们（例如 `v1`, `v2`），根据上下文标准化它们的文档字符串，或在将其反馈给 LLM 之前移除文档字符串以节省 Token 成本。

```python
from adtools import PyFunction

# 假设 LLM 生成了两个版本的交叉变异算法
llm_output_1 = '''
def crossover(p1, p2):
    """Single point crossover."""
    point = len(p1) // 2
    return p1[:point] + p2[point:], p2[:point] + p1[point:]
'''

llm_output_2 = """
def crossover_op(parent_a, parent_b):
    # This is a uniform crossover
    mask = [True, False] * (len(parent_a) // 2)
    return [a if m else b for a, b, m in zip(parent_a, parent_b, mask)]
"""

# 解析函数
func_v1 = PyFunction.extract_first_function_from_text(llm_output_1)
func_v2 = PyFunction.extract_first_function_from_text(llm_output_2)

# --- 修改逻辑 ---

# 1. 标准化命名：重命名为 v1 和 v2
func_v1.name = "crossover_v1"
func_v2.name = "crossover_v2"

# 2. 文档字符串管理：
# 对于 v1：强制使用特定的文档字符串格式用于提示词
func_v1.docstring = "Variant 1: Implementation of Single Point Crossover."

# 对于 v2：完全移除文档字符串（例如，为了减少上下文窗口使用）
func_v2.docstring = None

# --- 构建提示词 ---

prompt = "Here are the two crossover algorithms currently in the population:\n\n"
prompt += str(func_v1) + "\n"
prompt += str(func_v2) + "\n"
prompt += "Please generate a v3 that combines the best features of both."

print(prompt)

```

**输出：**

```text
Here are the two crossover algorithms currently in the population:

def crossover_v1(p1, p2):
    """Variant 1: Implementation of Single Point Crossover."""
    point = len(p1) // 2
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def crossover_v2(parent_a, parent_b):
    # This is a uniform crossover
    mask = [True, False] * (len(parent_a) // 2)
    return [a if m else b for a, b, m in zip(parent_a, parent_b, mask)]

Please generate a v3 that combines the best features of both.
```

### 使用评估器进行安全代码评估

在评估 LLM 生成的代码时，安全性和可靠性至关重要：

```python
import time
from adtools.evaluator import PyEvaluator
from typing import Dict, Callable, List


class AlgorithmValidator(PyEvaluator):
    def evaluate_program(
            self,
            program_str: str,
            callable_functions_dict: Dict[str, Callable] | None,
            callable_functions_list: List[Callable] | None,
            callable_classes_dict: Dict[str, Callable] | None,
            callable_classes_list: List[Callable] | None,
            **kwargs
    ) -> dict:
        results = {"correct": 0, "total": 0, "time": 0}

        try:
            # 获取排序函数
            sort_func = callable_functions_dict.get("sort_algorithm")
            if not sort_func:
                return {**results, "error": "Missing required function"}

            # 使用多个输入进行测试
            test_cases = [
                [5, 3, 1, 4, 2],
                [1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1],
                list(range(100)),  # 大型测试用例
                [],
            ]

            for case in test_cases:
                start = time.time()
                result = sort_func(
                    case[:] 
                )  # 传递副本以避免就地修改
                duration = time.time() - start

                results["total"] += 1
                if result == sorted(case):
                    results["correct"] += 1
                results["time"] += duration

        except Exception as e:
            results["error"] = str(e)

        return results


# 示例：使用可能存在问题的代码
problematic_code = """
def sort_algorithm(arr):
    # 此实现对于空数组存在 Bug
    if not arr:
        return []  # 遗漏此情况会导致失败
        
    # 具有潜在无限循环的实现
    i = 0
    while i < len(arr) - 1:
        if arr[i] > arr[i+1]:
            arr[i], arr[i+1] = arr[i+1], arr[i]
            i = 0  # 交换后重置到开始
        else:
            i += 1
    return arr
"""

malicious_code = """
def sort_algorithm(arr):
    import time
    time.sleep(15)  # 超过超时时间
    return sorted(arr)
"""

validator = AlgorithmValidator()
print(validator.secure_evaluate(problematic_code, timeout_seconds=5))
print(validator.secure_evaluate(malicious_code, timeout_seconds=5))

```

这演示了 `adtools` 如何处理：

- **超时保护**：具有无限循环的恶意代码会被终止。
- **错误隔离**：被评估代码中的异常不会导致主进程崩溃。
- **输出重定向**：防止不需要的打印语句干扰控制台。
- **资源管理**：正确清理进程和共享资源。

评估框架确保即使代码包含错误、无限循环或尝试访问系统资源，你的主应用程序也能保持安全和响应。

## 许可证

本项目采用 **MIT 许可证**。详情请参阅 [LICENSE](./LICENSE) 文件。

## 联系与反馈

如果你有任何问题，遇到 Bug 或有改进建议，请随时[提交 Issue](https://github.com/RayZhhh/py-adtools/issues) 或联系我们。非常感谢你的贡献和反馈！
