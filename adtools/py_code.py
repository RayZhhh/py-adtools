"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import ast
import dataclasses
import textwrap
from typing import List, Optional

__all__ = ['PyScript', 'PyFunction', 'PyClass', 'PyProgram']


@dataclasses.dataclass
class PyScript:
    """A parsed Python script block (top-level code that's not in classes/functions).
    """
    code: str

    def __str__(self) -> str:
        return self.code + '\n'


@dataclasses.dataclass
class PyFunction:
    """A parsed Python function.
    > Part of this class is referenced from:
    > https://github.com/google-deepmind/funsearch/blob/main/implementation/code_manipulation.py
    """
    decorator: str
    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None

    def __str__(self) -> str:
        return_type = f' -> {self.return_type}' if self.return_type else ''
        function = f'{self.decorator}\ndef {self.name}({self.args}){return_type}:\n'
        if self.docstring:
            # The self.docstring is already indented on every line except the first one.
            # Here, we assume the indentation is always four spaces.
            new_line = '\n' if self.body else ''
            function += f'    """{self.docstring}"""{new_line}'
        # self.body is already indented.
        function += self.body + '\n\n'
        return function

    def __setattr__(self, name: str, value: str) -> None:
        # Ensure there aren't leading & trailing new lines in `body`
        if name == 'body':
            value = value.strip('\n')
        # ensure there aren't leading & trailing quotes in `docstring`
        if name == 'docstring' and value is not None:
            if '"""' in value:
                value = value.strip()
                value = value.replace('"""', '')
        super().__setattr__(name, value)

    @classmethod
    def extract_first_function_from_text(cls, text: str) -> 'PyFunction':
        tree = ast.parse(text)
        visitor = _ProgramVisitor(text)
        visitor.visit(tree)
        program = visitor.return_program()
        return program.functions[0]

    @classmethod
    def extract_all_functions_from_text(cls, text: str) -> List['PyFunction']:
        tree = ast.parse(text)
        visitor = _ProgramVisitor(text)
        visitor.visit(tree)
        program = visitor.return_program()
        return program.functions


@dataclasses.dataclass
class PyClass:
    """A parsed Python class."""
    decorator: str
    name: str
    bases: str
    class_vars_and_code: List[PyScript] = None
    docstring: str | None = None
    functions: list[PyFunction] = dataclasses.field(default_factory=list)
    functions_class_vars_and_code: List[PyScript | PyFunction] | None = None

    def __str__(self) -> str:
        class_def = f'{self.decorator}\nclass {self.name}'
        if self.bases:
            class_def += f'({self.bases})'
        class_def += ':\n'

        if self.docstring:
            class_def += f'    """{self.docstring}"""\n'

        for item in self.functions_class_vars_and_code:
            if isinstance(item, PyScript):
                class_def += f'{str(item)}\n'
            else:
                # Add functions with an extra level of indentation
                class_def += textwrap.indent(str(item).strip(), '    ')
                class_def += '\n\n'
        return class_def

    def __setattr__(self, name: str, value: str) -> None:
        # Ensure there aren't leading & trailing new lines in `body`
        if name == 'body':
            value = value.strip('\n')
        # ensure there aren't leading & trailing quotes in `docstring`
        if name == 'docstring' and value is not None:
            if '"""' in value:
                value = value.strip()
                value = value.replace('"""', '')
        super().__setattr__(name, value)

    @classmethod
    def extract_first_class_from_text(cls, text: str) -> 'PyClass':
        tree = ast.parse(text)
        visitor = _ProgramVisitor(text)
        visitor.visit(tree)
        program = visitor.return_program()
        return program.classes[0]

    @classmethod
    def extract_all_classes_from_text(cls, text: str) -> List['PyClass']:
        tree = ast.parse(text)
        visitor = _ProgramVisitor(text)
        visitor.visit(tree)
        program = visitor.return_program()
        return program.classes


@dataclasses.dataclass
class PyProgram:
    """A parsed Python program."""

    scripts: list[PyScript]  # Top-level code that's not in classes/functions
    functions: list[PyFunction]  # Top-level functions in the code
    classes: list[PyClass]  # Top-level classes in the code
    classes_functions_scripts: list[PyFunction | PyClass | PyScript]

    def __str__(self) -> str:
        program = ''
        for class_or_func_or_script in self.classes_functions_scripts:
            program += str(class_or_func_or_script) + '\n'
        return program

    @classmethod
    def from_text(cls, text: str) -> Optional['PyProgram']:
        tree = ast.parse(text)
        visitor = _ProgramVisitor(text)
        visitor.visit(tree)
        return visitor.return_program()


class _ProgramVisitor(ast.NodeVisitor):
    """Parses code to collect all required information to produce a `Program`.
    Now handles scripts, functions, and classes.
    """

    def __init__(self, sourcecode: str):
        self._codelines: list[str] = sourcecode.splitlines()
        self._scripts: list[PyScript] = []
        self._functions: list[PyFunction] = []
        self._classes: list[PyClass] = []
        self._classes_functions_scripts: list[PyFunction | PyClass | PyScript] = []
        self._last_script_end = 0

    def _add_script(self, start_line: int, end_line: int):
        """Add a script segment from the code."""
        if start_line >= end_line:
            return
        script_code = '\n'.join(self._codelines[start_line:end_line]).strip()
        if script_code:
            script = PyScript(code=script_code)
            self._scripts.append(script)
            self._classes_functions_scripts.append(script)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collects all information about the function being parsed.
        """
        # We only consider top-level function
        if node.col_offset == 0:
            # Extract decorator first
            has_decorators = bool(node.decorator_list)
            if has_decorators:
                # Find the minimum line number and retain the code above
                decorator_start_line = min(decorator.lineno for decorator in node.decorator_list)
                decorator = '\n'.join(self._codelines[decorator_start_line - 1: node.lineno - 1])
                # Update script end line
                script_end_line = decorator_start_line - 1
            else:
                decorator = ''
                script_end_line = node.lineno - 1

            # Add any script code before this function
            self._add_script(self._last_script_end, script_end_line)
            self._last_script_end = node.end_lineno

            function_end_line = node.end_lineno
            body_start_line = node.body[0].lineno - 1
            docstring = None

            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                docstring = ast.literal_eval(ast.unparse(node.body[0])).strip()
                if len(node.body) > 1:
                    body_start_line = node.body[0].end_lineno
                else:
                    body_start_line = function_end_line

            func = PyFunction(
                decorator=decorator,
                name=node.name,
                args=ast.unparse(node.args),
                return_type=ast.unparse(node.returns) if node.returns else None,
                docstring=docstring,
                body='\n'.join(self._codelines[body_start_line: function_end_line]),
            )

            self._functions.append(func)
            self._classes_functions_scripts.append(func)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Collects all information about the class being parsed.
        """
        # We only care about top-level classes
        if node.col_offset == 0:
            # Extract decorator first
            has_decorators = bool(node.decorator_list)
            if has_decorators:
                # Find the minimum line number and retain the code above
                decorator_start_line = min(decorator.lineno for decorator in node.decorator_list)
                class_decorator = '\n'.join(self._codelines[decorator_start_line - 1: node.lineno - 1])
                # Update script end line
                script_end_line = decorator_start_line - 1
            else:
                class_decorator = ''
                script_end_line = node.lineno - 1

            # Add any script code before this class
            self._add_script(self._last_script_end, script_end_line)
            self._last_script_end = node.end_lineno

            # Extract class docstring
            docstring = None
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                docstring = ast.literal_eval(ast.unparse(node.body[0]))

            methods = []
            class_vars_and_code = []
            function_class_vars_and_code = []  # record the order of function and class vars and code

            # Traverse each body, if there is a docstring, skip body[0]
            for item in node.body if docstring is None else node.body[1:]:
                if isinstance(item, ast.FunctionDef):
                    # Extract decorators
                    has_decorators = bool(item.decorator_list)
                    if has_decorators:
                        # Find the minimum line number and retain the code above
                        decorator_start_line = min(decorator.lineno for decorator in item.decorator_list)
                        # Dedent decorator code for 4 spaces
                        decorator = []
                        for line in range(decorator_start_line - 1, item.lineno - 1):
                            dedented_decorator = self._codelines[line][4:]
                            decorator.append(dedented_decorator)
                        decorator = '\n'.join(decorator)
                    else:
                        decorator = ''

                    method_end_line = item.end_lineno
                    method_body_start_line = item.body[0].lineno - 1
                    method_docstring = None

                    # Extract doc-string if there exists
                    if isinstance(item.body[0], ast.Expr) and isinstance(item.body[0].value, ast.Constant):
                        method_docstring = ast.literal_eval(ast.unparse(item.body[0])).strip()
                        if len(item.body) > 1:
                            method_body_start_line = item.body[0].end_lineno
                        else:
                            method_body_start_line = method_end_line

                    # Extract function body and dedent for 4 spaces
                    body = []
                    for line in range(method_body_start_line, method_end_line):
                        dedented_body = self._codelines[line][4:]
                        body.append(dedented_body)
                    body = '\n'.join(body)

                    py_func = PyFunction(
                        decorator=decorator,
                        name=item.name,
                        args=ast.unparse(item.args),
                        return_type=ast.unparse(item.returns) if item.returns else None,
                        docstring=method_docstring,
                        body=body,
                    )
                    methods.append(py_func)
                    function_class_vars_and_code.append(py_func)
                else:  # If the item is not a function definition,add to class variables and code
                    code = []
                    for i in range(item.lineno - 1, item.end_lineno):
                        code.append(self._codelines[i])
                    py_script = PyScript(code='\n'.join(code))
                    class_vars_and_code.append(py_script)
                    function_class_vars_and_code.append(py_script)

            # Get base classes
            bases = ', '.join([ast.unparse(base) for base in node.bases]) if node.bases else ''

            class_ = PyClass(
                decorator=class_decorator,
                name=node.name,
                bases=bases,
                docstring=docstring,
                class_vars_and_code=class_vars_and_code if class_vars_and_code else None,
                functions=methods,
                functions_class_vars_and_code=function_class_vars_and_code if function_class_vars_and_code else None
            )
            self._classes.append(class_)
            self._classes_functions_scripts.append(class_)
        self.generic_visit(node)

    def return_program(self) -> PyProgram:
        # Add any remaining script code after the last class/function
        self._add_script(self._last_script_end, len(self._codelines))
        return PyProgram(
            scripts=self._scripts,
            functions=self._functions,
            classes=self._classes,
            classes_functions_scripts=self._classes_functions_scripts,
        )
