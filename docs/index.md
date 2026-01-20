# Welcome to Py-ADTools

**Py-ADTools** is a specialized Python toolkit designed for **LLM-Aided Algorithm Design (LLM-AAD)** and code optimization. It bridges the gap between generating code (with LLMs) and verifying it (with secure execution and evaluation).

## The Workflow

Py-ADTools facilitates a closed-loop workflow:

1.  **Parse**: Decompose code into structured objects (`py_code`).
2.  **Generate/Modify**: Use LLMs to mutate functions or generate new algorithms (`lm`).
3.  **Sandboxing**: Execute the potentially unsafe code in isolated processes (`sandbox`).
4.  **Evaluate**: Score the code based on correctness or performance (`evaluator`).
5.  **Iterate**: Feed the results back to the LLM.

## Module Overview

- **[Code Parsing (`adtools.py_code`)](./parsing.md)**:
    A robust parser that turns Python source code into manipulatable `PyProgram` objects. It allows you to programmatically rename functions, swap bodies, or inject docstrings while preserving the original layout and comments.

- **[Safe Execution (`adtools.sandbox`)](./sandbox.md)**:
    A security layer that runs untrusted code in separate processes (via `multiprocessing`) or distributed actors (via `Ray`). It provides timeout enforcement, output redirection, and zombie process cleanup.

- **[Evaluation Framework (`adtools.evaluator`)](./evaluator.md)**:
    A template-based system for defining how code should be tested. It handles the "string-to-function" conversion and sandboxing lifecycle, letting you focus on the test logic itself.

- **[Auto-Evaluation Server (`adtools.evaluator.auto_server`)](./auto_server.md)**:
    Turn any evaluator into a concurrent HTTP microservice with a single command. Ideal for decoupling heavy evaluation tasks from your main generation loop.

- **[LLM Interface (`adtools.lm`)](./language_models.md)**:
    A unified API for connecting to OpenAI, or managing local high-performance inference servers like vLLM and SGLang.

## Quick Start

Check the [Installation](./installation.md) guide to get set up, or explore the [Tutorials](../README.md#tutorials) in the README.