# Code Parsing Implementation Analysis

The `adtools.py_code` module implements a **round-trip** parser: it can parse Python code into objects and reconstruct it back to a string that is semantically equivalent to the original (preserving comments and formatting).

This document analyzes the source code implementation of the core components.

## 1. `_ProgramVisitor`: The Parsing Engine

The heavy lifting is done by `_ProgramVisitor`, which inherits from Python's built-in `ast.NodeVisitor`. Instead of just traversing the AST, it uses the AST nodes to identify **where** components start and end (line numbers), and then extracts the raw text from the source code.

### 1.1 Multiline String Detection

A major challenge in parsing Python code is distinguishing between "indentation" and "content inside a multiline string".

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
            # If start_line != end_line, it is a multiline string
            if end_line > start_line:
                # Mark lines within the string
                for i in range(start_line + 1, end_line + 1):
                    string_lines.add(i)
    return string_lines
```

**Implementation Insight**:
- We use the `tokenize` library (lexer) rather than the AST parser here because AST loses exact line mapping of multiline strings in some python versions or doesn't easily distinguish them from code blocks.
- We store the line numbers in a `set`. Later, when we "dedent" (remove indentation) from a function body, we check this set. If a line is inside a multiline string, we **do not** strip its whitespace, preserving the string's content.

### 1.2 Code Extraction and Dedenting

When we extract a function body, we need to remove the indentation relative to the function definition so it looks like a top-level block.

```python
# adtools/py_code.py

def _get_code(self, start_line: int, end_line: int, remove_indent: int = 0) -> str:
    # ...
    for idx, line in enumerate(lines):
        current_lineno = start_line + idx + 1
        
        # Check if line is inside a multiline string (protected)
        if current_lineno in self._multiline_string_lines:
            dedented_lines.append(line)
        else:
            # For normal code, strip the indentation (col_offset)
            if len(line) >= remove_indent and line[:remove_indent].isspace():
                dedented_lines.append(line[remove_indent:])
            else:
                dedented_lines.append(line)
    return "\n".join(dedented_lines).rstrip()
```

**Implementation Insight**:
- `remove_indent` comes from `node.col_offset` provided by the AST.
- If we didn't protect multiline strings, a docstring like this:
    ```python
    def foo():
        """
        Line 1
        Line 2
        """
    ```
    Might lose the indentation of "Line 1" and "Line 2", breaking formatting.

### 1.3 Visiting Functions (`visit_FunctionDef`)

When the visitor encounters a function:

```python
# adtools/py_code.py

def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    if node.col_offset == 0:  # We only care about top-level functions here
        # 1. Capture the "Gap" (Script) before this function
        start_line = node.lineno - 1
        if hasattr(node, "decorator_list") and node.decorator_list:
            start_line = min(d.lineno for d in node.decorator_list) - 1
        
        self._add_script_segment(self._last_script_end, start_line)
        self._last_script_end = node.end_lineno

        # 2. Extract function info
        func = self._extract_function_info(node)
        self._functions.append(func)
```

**Implementation Insight**:
- `_last_script_end` tracks where the previous element ended.
- Everything between the last element and the current function (e.g., imports, comments) is captured as a `PyCodeBlock` via `_add_script_segment`.
- This ensures 100% coverage of the source file.

## 2. `PyFunction` and `PyClass` Reconstruction

These classes act as containers. Their complexity lies in **reconstruction** (`__str__`).

### 2.1 Rebuilding a Function

```python
# adtools/py_code.py within PyFunction

def _to_str(self, indent_str=""):
    # ...
    # Reconstruct signature
    function_def += f"{prefix} {self.name}({self.args}){return_type}:"
    
    # Reconstruct Docstring
    if self.docstring:
         function_def += textwrap.indent(f'"""{self.docstring}"""', indent_str)
    
    # Reconstruct Body
    # The body is stored 'dedented'. We must indent it back.
    function_def += _indent_code_skip_multi_line_str(self.body, indent_str)
    return function_def
```

**Implementation Insight**:
- We use `_indent_code_skip_multi_line_str` again during reconstruction. This is the inverse of the logic in extraction. It ensures that while we add indentation to code lines, we don't double-indent the contents of multiline strings.

## 3. `PyProgram` Wrapper

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

**Implementation Insight**:
- It acts as the entry point. It instantiates the visitor, runs the pass, and collects the lists of scripts, functions, and classes into the final object.
