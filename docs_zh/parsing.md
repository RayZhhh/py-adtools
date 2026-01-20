# 代码解析实现分析

`adtools.py_code` 模块实现了一个**往返（Round-trip）**解析器：它可以将 Python 代码解析为对象，并将其重构回与原始代码在语义上等效的字符串（保留注释和格式）。

本文档分析了核心组件的源码实现。

## 1. `_ProgramVisitor`：解析引擎

繁重的工作由 `_ProgramVisitor` 完成，它继承自 Python 内置的 `ast.NodeVisitor`。它不仅仅遍历 AST，还利用 AST 节点来识别组件的**起始和结束位置**（行号），然后从源代码中提取原始文本。

### 1.1 多行字符串检测

解析 Python 代码的一个主要挑战是区分“缩进”和“多行字符串内部的内容”。

```python
# adtools/py_code.py

def _detect_multiline_strings(self, sourcecode: str) -> Set[int]:
    """Scans the source code using tokenize to identify line numbers..."""
    string_lines = set()
    tokens = tokenize.tokenize(BytesIO(sourcecode.encode("utf-8")).readline)
    for token in tokens:
        if token.type == tokenize.STRING:
            start_line, _ = token.start
            end_line, _ = token.end
            # 如果 start_line != end_line，则它是多行字符串
            if end_line > start_line:
                # 标记字符串内的行
                for i in range(start_line + 1, end_line + 1):
                    string_lines.add(i)
    return string_lines
```

**实现洞察**：
- 我们在这里使用 `tokenize` 库（词法分析器）而不是 AST 解析器，因为 AST 在某些 Python 版本中会丢失多行字符串的确切行映射，或者不容易将它们与代码块区分开来。
- 我们将行号存储在一个 `set` 中。之后，当我们从函数体中“去除缩进（dedent）”时，我们会检查这个集合。如果一行在多行字符串内，我们**不**去除其空白字符，从而保留字符串的内容。

### 1.2 代码提取与去缩进

当我们提取函数体时，我们需要移除相对于函数定义的缩进，使其看起来像一个顶级代码块。

```python
# adtools/py_code.py

def _get_code(self, start_line: int, end_line: int, remove_indent: int = 0) -> str:
    # ...
    for idx, line in enumerate(lines):
        current_lineno = start_line + idx + 1
        
        # 检查行是否在多行字符串内（受保护）
        if current_lineno in self._multiline_string_lines:
            dedented_lines.append(line)
        else:
            # 对于普通代码，去除缩进（col_offset）
            if len(line) >= remove_indent and line[:remove_indent].isspace():
                dedented_lines.append(line[remove_indent:])
            else:
                dedented_lines.append(line)
    return "\n".join(dedented_lines).rstrip()
```

**实现洞察**：
- `remove_indent` 来自 AST 提供的 `node.col_offset`。
- 如果我们不保护多行字符串，像这样的文档字符串：
    ```python
    def foo():
        """
        Line 1
        Line 2
        """
    ```
    可能会丢失 "Line 1" 和 "Line 2" 的缩进，从而破坏格式。

### 1.3 访问函数 (`visit_FunctionDef`)

当访问者遇到函数时：

```python
# adtools/py_code.py

def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    if node.col_offset == 0:  # 我们这里只关心顶级函数
        # 1. 捕获此函数之前的“间隙”（脚本）
        start_line = node.lineno - 1
        if hasattr(node, "decorator_list") and node.decorator_list:
            start_line = min(d.lineno for d in node.decorator_list) - 1
        
        self._add_script_segment(self._last_script_end, start_line)
        self._last_script_end = node.end_lineno

        # 2. 提取函数信息
        func = self._extract_function_info(node)
        self._functions.append(func)
```

**实现洞察**：
- `_last_script_end` 跟踪上一个元素结束的位置。
- 上一个元素和当前函数之间的所有内容（例如导入、注释）都通过 `_add_script_segment` 被捕获为 `PyCodeBlock`。
- 这确保了对源文件的 100% 覆盖。

## 2. `PyFunction` 和 `PyClass` 重构

这些类充当容器。它们的复杂性在于**重构** (`__str__`)。

### 2.1 重建函数

```python
# adtools/py_code.py within PyFunction

def _to_str(self, indent_str=""):
    # ...
    # 重建签名
    function_def += f"{prefix} {self.name}({self.args}){return_type}:"
    
    # 重建文档字符串
    if self.docstring:
         function_def += textwrap.indent(f'"""{self.docstring}"""', indent_str)
    
    # 重建主体
    # 主体以“去缩进”状态存储。我们必须将其缩进回去。
    function_def += _indent_code_skip_multi_line_str(self.body, indent_str)
    return function_def
```

**实现洞察**：
- 我们在重构过程中再次使用 `_indent_code_skip_multi_line_str`。这是提取逻辑的逆过程。它确保我们在给代码行添加缩进时，不会对多行字符串的内容进行双重缩进。

## 3. `PyProgram` 包装器

```python
# adtools/py_code.py

@classmethod
def from_text(cls, text: str, debug=False) -> Optional["PyProgram"]:
    try:
        tree = ast.parse(text)
        visitor = _ProgramVisitor(text)
        visitor.visit(tree)
        return visitor.return_program()
    except:
        # ...
```

**实现洞察**：
- 它充当入口点。它实例化访问者，运行遍历，并将脚本、函数和类的列表收集到最终对象中。

```
