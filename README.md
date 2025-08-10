# Useful tools for parsing Python programs for algorithm design

------

> This repo aims to help develop more powerful [Large Language Models for Algorithm Design (LLM4AD)](https://github.com/Optima-CityU/llm4ad) applications. 
>
> More tools will be provided soon.

------

The figure demonstrates how a Python program is parsed into `PyScript`, `PyFunction`, `PyClass,` and `PyProgram` via `adtools`.

![pycode](./assets/pycode.png)

------

## Installation

> [!TIP]
>
> It is recommended to use Python >= 3.10.

Run the following instructions to install adtools.

```shell
pip install git+https://github.com/RayZhhh/py-adtools.git
```

## Usage

Parse your code (in string) into Python code instances.

```python
from adtools import PyProgram


code = r'''
import ast
import numpy as np

def func():
    a = 5
    return a + a

class A(B):
    a=1
    
    @yes()
    @deco()
    def __init__(self):
        pass

    def method(self):
        pass
    
    b=2
'''

p = PyProgram.from_text(code)
print(p)
print(f'-------------------------------------')
print(p.classes[0].functions[0].decorator)
print(f'-------------------------------------')
print(p.functions[0].name)
```

