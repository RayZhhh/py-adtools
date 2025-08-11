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
pip install git+https://github.com/RayZhhh/adtool.git
```

Or install via pip:

```shell
pip install py-adtools
```

## Usage

### Parser for a Python program

Parse your code (in string) into Python code instances.

```python
from adtools import PyProgram

code = r'''
import ast, numba                 # This part will be parsed into PyScript
import numpy as np

@numba.jit()                      # This part will be parsed into PyFunction
def funcion(arg1, arg2=True):     
    if arg2:
    	return arg1 * 2
    else:
    	return arg1 * 4

@some.decorators()                # This part will be parsed into PyClass
class PythonClass(BaseClass):
    class_var1 = 1                # This part will be parsed into PyScript
    class_varb = 2                # and placed in PyClass.class_vars_and_code
 
    def __init__(self, x):        # This part will be parsed into PyFunction
        self.x = x                # and placed in PyClass.functions
	
    def method1(self):
        return self.x * 10
    
    @some.decorators()
    def method2(self, x, y):
    	return x + y + self.method1(x)
    
    class InnerClass:             # This part will be parsed into PyScript
    	def __init__(self):       # and placed in PyClass.class_vars_and_code
    		...

if __name__ == '__main__':        # This part will be parsed into PyScript
	res = function(1)
	print(res)
	class = PythonClass()
'''

p = PyProgram.from_text(code)
print(p)
print(f'-------------------------------------')
print(p.classes[0].functions[0].decorator)
print(f'-------------------------------------')
print(p.functions[0].name)
```

