# Code Parsing

The `adtools.py_code` module is the foundation of the library, designed to transform raw Python source code into structured, manipulatable objects. Unlike simple string manipulation, it leverages Python's Abstract Syntax Tree (AST) to ensure syntactical correctness while preserving formatting, comments, and structure.

## Core Design Philosophy

The parsing logic is built around the `ast` module and `tokenize` library. The primary goal is to **decompose** a Python file into high-level logical blocks—Functions, Classes, and arbitrary Code Blocks (imports, global variables)—so they can be individually modified (e.g., renaming a function, changing a docstring) and then **reconstructed** back into valid Python code.

### The Parsing Process

1.  **AST Traversal**: The code is parsed into an AST. A custom `_ProgramVisitor` (inheriting from `ast.NodeVisitor`) traverses this tree.
2.  **Segment Identification**: The visitor identifies top-level nodes:
    *   `FunctionDef` and `AsyncFunctionDef` become `PyFunction`.
    *   `ClassDef` becomes `PyClass`.
    *   Gaps between these definitions (imports, assignments, comments) are captured as `PyCodeBlock`.
3.  **Indentation Handling**: One of the most complex aspects is handling indentation. The parser calculates the column offset of nodes and strips strictly necessary indentation when extracting function/class bodies, ensuring that the extracted code is valid (dedented). When reconstructing, it re-applies indentation based on the hierarchy.
4.  **Preservation**: It uses `tokenize` to detect multiline strings to avoid incorrectly stripping indentation from within string literals (e.g., SQL queries or large text blocks inside code).

## Class Reference & Implementation Details

### 1. `PyProgram`

**Implementation Idea**: Acts as the container for the entire file. It maintains a list of `elements` which preserves the exact order of `PyFunction`, `PyClass`, and `PyCodeBlock` objects as they appeared in the source.

```python
@dataclasses.dataclass
class PyProgram:
    scripts: List[PyCodeBlock]   # Code blocks outside functions/classes
    functions: List[PyFunction]  # Top-level functions
    classes: List[PyClass]       # Top-level classes
    elements: List[Union[PyFunction, PyClass, PyCodeBlock]] # Ordered list of all elements

    def __str__(self) -> str:
        # Reconstruction: Concatenates the string representation of all elements
        program = ""
        for item in self.elements:
            program += str(item) + "\n\n"
        return program.strip()
```

### 2. `PyFunction`

**Implementation Idea**: Encapsulates a function definition. It separates the signature (name, args, decorators, return type) from the body and docstring. This allows you to modify the body (e.g., injecting new logic) without worrying about the function header, or rename the function without regex.

**Key Attributes**:
- `body`: The function body text, **dedented** (indentation removed) so it looks like top-level code.
- `docstring`: Extracted separately to allow easy modification or removal (e.g., to save context window in LLMs).

```python
func = program.functions[0]
func.name = "new_name"           # Modify name
func.docstring = None            # Remove docstring
func.decorator = None            # Remove decorators
print(func)                      # Reconstructs the function with changes
```

### 3. `PyClass`

**Implementation Idea**: Similar to `PyFunction`, but for classes. It acts as a recursive container. A `PyClass` contains its own list of `functions` (methods) and `body` (which can contain nested `PyCodeBlock` for class attributes or inner classes).

**Indentation Management**:
When a method is extracted from a class, the parser removes the class-level indentation (usually 4 spaces) from the method's definition. When the `PyClass` is converted back to a string, it re-indents all its children.

### 4. `PyCodeBlock`

**Implementation Idea**: Represents everything that isn't a function or a class. This includes:
- Import statements (`import os`)
- Global variable assignments (`x = 1`)
- `if __name__ == "__main__":` blocks

This ensures that `PyProgram` covers 100% of the source code, not just the functions and classes.

## Example: Parsing and Modifying

```python
from adtools import PyProgram

source_code = """
import math

def calculate_circle_area(radius):
    """Returns the area of a circle."""
    return math.pi * radius ** 2
"""

# 1. Parse
program = PyProgram.from_text(source_code)

# 2. Inspect
func = program.functions[0]
print(f"Original Name: {func.name}")

# 3. Modify
func.name = "get_area"
func.docstring = "Calculates area."

# 4. Reconstruct
# The __str__ method automatically reassembles the code
print(program)
# Output:
# import math
#
# def get_area(radius):
#     """Calculates area."""
#     return math.pi * radius ** 2
```