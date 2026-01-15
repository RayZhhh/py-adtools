# Code Parsing

The `adtools.py_code` module provides powerful functionality to parse raw Python code into structured, manipulatable objects. This makes it easy to programmatically analyze, modify, and generate code.

## Core Components

The parser decomposes Python code into four primary data structures:

| Component       | Description                                                                                                   | Key Attributes                                              |
|:----------------|:--------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|
| **`PyProgram`**   | Represents an entire Python file, maintaining the original sequence of scripts, functions, and classes.       | `scripts`, `functions`, `classes`, `elements`               |
| **`PyFunction`**  | Represents a top-level function or a class method. You can dynamically modify its name, args, decorators, docstring, or body. | `name`, `args`, `body`, `docstring`, `decorator`, `return_type` |
| **`PyClass`**     | Represents a class definition. It serves as a container for methods and class-level statements.               | `name`, `bases`, `functions` (methods), `body`              |
| **`PyCodeBlock`** | Represents raw segments of code, such as import statements, global variables, or logic blocks inside a class. | `code`                                                      |

## Usage Example

The following example demonstrates how to parse a complex piece of Python code into a `PyProgram` object and access its internal components.

```python
from adtools import PyProgram

code = r"""
import ast, numba
import numpy as np

@numba.jit()
def function(arg1, arg2=True):
    '''This is a function.'''
    if arg2:
        return arg1 * 2
    else:
        return arg1 * 4

class PythonClass(BaseClass):
    '''This is a class.'''
    class_var = 1

    def __init__(self, x):
        self.x = x

    def method1(self):
        return self.x * 10

if __name__ == '__main__':
    res = function(1)
    print(res)
"""

# Parse the program from text
p = PyProgram.from_text(code)

# Print the entire program
print(p)

# Access a specific class and function
print("-------------------------------------")
py_class = p.classes[0]
print(f"Class Name: {py_class.name}")
print(f"Class Bases: {py_class.bases}")

print("-------------------------------------")
py_function = p.functions[0]
print(f"Function Name: {py_function.name}")
print(f"Function Decorator: {py_function.decorator}")

# Modify function attributes
py_function.name = "new_function_name"
py_function.docstring = "This is the modified docstring."
print("\n--- Modified Code ---\
")
print(p)
```

## Key Features

- **Structure Preservation**: Accurately maintains original indentation and formatting.
- **Multiline String Handling**: Correctly preserves multiline string content without breaking its format.
- **Easy Component Access**: Conveniently access and manipulate functions, classes, and code blocks via list indices.
- **Dynamic Code Modification**: Programmatically change function names, docstrings, or body content.
- **Complete Program Representation**: `PyProgram` maintains the exact sequence of all elements from the source file.
- **Robust Indentation Handling**: Intelligently handles different levels of indentation during parsing and reconstruction, preserving correctness even in complex nested structures (like methods within a class).
