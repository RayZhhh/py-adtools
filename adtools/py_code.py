"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import ast
import dataclasses
import textwrap
import traceback
from typing import List, Optional, Union

__all__ = ["PyCodeBlock", "PyFunction", "PyClass", "PyProgram"]


@dataclasses.dataclass
class PyCodeBlock:
    """A parsed Python code block (e.g., top-level code that is not in classes/functions,
    or miscellaneous statements inside a class).
    """

    code: str

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return self.__str__() + "\n"


@dataclasses.dataclass
class PyFunction:
    """A parsed Python function.
    Part of this class is referenced from:
    https://github.com/google-deepmind/funsearch/blob/main/implementation/code_manipulation.py
    """

    decorator: str
    name: str
    args: str
    body: str
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False

    def __str__(self) -> str:
        return_type = f" -> {self.return_type}" if self.return_type else ""
        function_def = f"{self.decorator}\n" if self.decorator else ""
        prefix = "async def" if self.is_async else "def"
        function_def += f"{prefix} {self.name}({self.args}){return_type}:\n"

        if self.docstring:
            # We indent the docstring. Assumes 4-space standard indentation for generation.
            new_line = "\n" if self.body else ""
            function_def += f'    """{self.docstring}"""{new_line}'

        # The body is expected to be already indented (if parsed correctly).
        # We ensure it is indented relative to the function definition.
        function_def += self.body
        return function_def

    def __repr__(self) -> str:
        return self.__str__() + "\n\n"

    def __setattr__(self, name: str, value: str) -> None:
        # Ensure there aren't leading & trailing new lines in `body`
        if name == "body" and isinstance(value, str):
            value = value.strip("\n")
        # Ensure there aren't leading & trailing quotes in `docstring`
        if name == "docstring" and value is not None:
            if '"""' in value:
                value = value.strip()
                value = value.replace('"""', "")
        super().__setattr__(name, value)

    @classmethod
    def extract_first_function_from_text(cls, text: str) -> "PyFunction":
        """Parses text and returns the first function found."""
        tree = ast.parse(text)
        visitor = _ProgramVisitor(text)
        visitor.visit(tree)
        program = visitor.return_program()
        if not program.functions:
            raise ValueError("No functions found in the provided text.")
        return program.functions[0]

    @classmethod
    def extract_all_functions_from_text(cls, text: str) -> List["PyFunction"]:
        """Parses text and returns all top-level functions found."""
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

    # Holds raw code blocks (variables, assignments) found in the class body.
    statements: Optional[List[PyCodeBlock]] = None
    docstring: Optional[str] = None
    functions: List[PyFunction] = dataclasses.field(default_factory=list)
    # Holds everything in order (Methods + Statements + Gaps).
    body: Optional[List[Union[PyCodeBlock, PyFunction]]] = None

    def __str__(self) -> str:
        class_def = f"{self.decorator}\n" if self.decorator else ""
        class_def += f"class {self.name}"
        if self.bases:
            class_def += f"({self.bases})"
        class_def += ":\n"

        if self.docstring:
            class_def += f'    """{self.docstring}"""\n'

        if self.body:
            last_item = None
            for i, item in enumerate(self.body):
                if last_item is not None:
                    # If there are not two consecutive PyCodeBlock intances, we add an addtional new line
                    if not (
                        isinstance(last_item, PyCodeBlock)
                        and isinstance(item, PyCodeBlock)
                    ):
                        class_def += "\n"
                # Indent everything inside the class by 4 spaces
                class_def += textwrap.indent(str(item), "    ")
                class_def += "\n" if i != len(self.body) - 1 else ""
                last_item = item
        else:
            class_def += "    pass"

        return class_def

    def __repr__(self):
        return self.__str__() + "\n\n"

    def __setattr__(self, name: str, value: str) -> None:
        if name == "body" and isinstance(value, str):
            value = value.strip("\n")
        if name == "docstring" and value is not None:
            if '"""' in value:
                value = value.strip()
                value = value.replace('"""', "")
        super().__setattr__(name, value)

    @classmethod
    def extract_first_class_from_text(cls, text: str) -> "PyClass":
        tree = ast.parse(text)
        visitor = _ProgramVisitor(text)
        visitor.visit(tree)
        program = visitor.return_program()
        if not program.classes:
            raise ValueError("No classes found in the provided text.")
        return program.classes[0]

    @classmethod
    def extract_all_classes_from_text(cls, text: str) -> List["PyClass"]:
        tree = ast.parse(text)
        visitor = _ProgramVisitor(text)
        visitor.visit(tree)
        program = visitor.return_program()
        return program.classes


@dataclasses.dataclass
class PyProgram:
    """A parsed Python program containing scripts, functions, and classes."""

    scripts: List[PyCodeBlock]  # Top-level code not in classes/functions
    functions: List[PyFunction]  # Top-level functions
    classes: List[PyClass]  # Top-level classes
    elements: List[
        Union[PyFunction, PyClass, PyCodeBlock]
    ]  # Complete sequence of the file elements.

    def __str__(self) -> str:
        program = ""
        for item in self.elements:
            program += str(item) + "\n\n"
        return program.strip()

    @classmethod
    def from_text(cls, text: str, debug=False) -> Optional["PyProgram"]:
        """Parses text into a PyProgram object. Returns None on syntax errors."""
        try:
            tree = ast.parse(text)
            visitor = _ProgramVisitor(text)
            visitor.visit(tree)
            return visitor.return_program()
        except:
            if debug:
                raise
            return None


class _ProgramVisitor(ast.NodeVisitor):
    """Parses code to collect all required information to produce a `PyProgram`.
    Handles scripts, functions, and classes with robust indentation handling.
    """

    def __init__(self, sourcecode: str):
        self._codelines: List[str] = sourcecode.splitlines()
        self._scripts: List[PyCodeBlock] = []
        self._functions: List[PyFunction] = []
        self._classes: List[PyClass] = []
        self._elements: List[Union[PyFunction, PyClass, PyCodeBlock]] = []
        self._last_script_end = 0

    def _get_code(self, start_line: int, end_line: int, remove_indent: int = 0) -> str:
        """Get code between start_line and end_line.

        Args:
            remove_indent: The number of spaces to strip from the beginning of each line.
                his corresponds to the column offset of the function definition.
        """
        if start_line >= end_line:
            return ""

        lines = self._codelines[start_line:end_line]

        if remove_indent > 0:
            dedented_lines = []
            for line in lines:
                # We only strip indentation if the line is long enough.
                # If a line inside a multiline string is shorter/unindented (e.g., column 0),
                # we keep it as is.
                if len(line) >= remove_indent and line[:remove_indent].isspace():
                    dedented_lines.append(line[remove_indent:])
                else:
                    dedented_lines.append(line)
            return "\n".join(dedented_lines).rstrip()
        else:
            # For top-level functions (remove_indent=0), we return the raw code.
            # This perfectly preserves the structure of multiline strings with 0 indentation.
            return "\n".join(lines).rstrip()

    def _add_script_segment(self, start_line: int, end_line: int):
        """Add a script segment (gap between functions/classes) from the code."""
        if start_line >= end_line:
            return
        script_code = self._get_code(start_line, end_line).strip()
        if script_code:
            script = PyCodeBlock(code=script_code)
            self._scripts.append(script)
            self._elements.append(script)

    def _extract_function_info(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> PyFunction:
        """Shared logic to extract information from FunctionDef or AsyncFunctionDef."""
        # Extract decorators
        if hasattr(node, "decorator_list") and node.decorator_list:
            dec_start = min(d.lineno for d in node.decorator_list)
            decorator = self._get_code(
                dec_start - 1, node.lineno - 1, remove_indent=node.col_offset
            )
        else:
            decorator = None

        # Extract docstring safely
        docstring = ast.get_docstring(node)

        # Determine where the actual code body starts
        body_start_line_node = node.body[0]
        if docstring and len(node.body) > 1:
            body_start_line = node.body[1].lineno - 1
        elif docstring:
            body_start_line = node.end_lineno
        else:
            body_start_line = node.body[0].lineno - 1

        # Extract body, and APPLY CRITICAL INDENTATION:
        #   - Top-level functions (col_offset=0) -> Body is NOT modified
        #   - Class methods (col_offset=4) -> Body loses exactly 4 spaces (the class indent)
        body = self._get_code(
            body_start_line, node.end_lineno, remove_indent=node.col_offset
        )
        is_async = isinstance(node, ast.AsyncFunctionDef)

        return PyFunction(
            decorator=decorator,
            name=node.name,
            args=ast.unparse(node.args),
            return_type=ast.unparse(node.returns) if node.returns else None,
            docstring=docstring,
            body=body,
            is_async=is_async,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handles top-level synchronous functions."""
        if node.col_offset == 0:
            start_line = node.lineno - 1
            if hasattr(node, "decorator_list") and node.decorator_list:
                start_line = min(d.lineno for d in node.decorator_list) - 1

            self._add_script_segment(self._last_script_end, start_line)
            self._last_script_end = node.end_lineno

            func = self._extract_function_info(node)
            self._functions.append(func)
            self._elements.append(func)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handles top-level asynchronous functions."""
        if node.col_offset == 0:
            start_line = node.lineno - 1
            if hasattr(node, "decorator_list") and node.decorator_list:
                start_line = min(d.lineno for d in node.decorator_list) - 1

            self._add_script_segment(self._last_script_end, start_line)
            self._last_script_end = node.end_lineno

            func = self._extract_function_info(node)
            self._functions.append(func)
            self._elements.append(func)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handles top-level classes."""
        if node.col_offset == 0:
            # Handle decorators and preceding script
            start_line = node.lineno - 1
            if hasattr(node, "decorator_list") and node.decorator_list:
                start_line = min(d.lineno for d in node.decorator_list) - 1
                decorator_code = self._get_code(start_line, node.lineno - 1)
            else:
                decorator_code = None

            self._add_script_segment(self._last_script_end, start_line)
            self._last_script_end = node.end_lineno

            # Extract class basic info
            docstring = ast.get_docstring(node)
            bases = (
                ", ".join([ast.unparse(base) for base in node.bases])
                if node.bases
                else None
            )

            # Process class body contents
            methods = []
            statements = []
            class_body = []
            last_inner_end = node.lineno
            body_nodes = node.body

            if docstring:
                if len(body_nodes) > 0:
                    last_inner_end = body_nodes[0].end_lineno
                body_nodes = body_nodes[1:]

            for item in body_nodes:
                # Default start is the definition line
                item_start_line = item.lineno

                # If the item has decorators (Function or Class), the visual start is the first decorator
                if hasattr(item, "decorator_list") and item.decorator_list:
                    item_start_line = min(d.lineno for d in item.decorator_list)

                # Capture Gaps (Use item_start_line instead of item.lineno)
                gap_code = self._get_code(
                    last_inner_end, item_start_line - 1, remove_indent=item.col_offset
                ).strip()
                if gap_code:
                    gap_block = PyCodeBlock(code=gap_code)
                    statements.append(gap_block)
                    class_body.append(gap_block)

                # Process the Item
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_func = self._extract_function_info(item)
                    methods.append(method_func)
                    class_body.append(method_func)
                else:
                    code_text = self._get_code(
                        item.lineno - 1, item.end_lineno, remove_indent=item.col_offset
                    )
                    block = PyCodeBlock(code=code_text)
                    statements.append(block)
                    class_body.append(block)

                last_inner_end = item.end_lineno

            class_obj = PyClass(
                decorator=decorator_code,
                name=node.name,
                bases=bases,
                docstring=docstring,
                statements=statements if statements else None,
                functions=methods,
                body=class_body if class_body else None,
            )
            self._classes.append(class_obj)
            self._elements.append(class_obj)

        self.generic_visit(node)

    def return_program(self) -> PyProgram:
        """Finalizes parsing and returns the PyProgram object."""
        self._add_script_segment(self._last_script_end, len(self._codelines))

        return PyProgram(
            scripts=self._scripts,
            functions=self._functions,
            classes=self._classes,
            elements=self._elements,
        )
