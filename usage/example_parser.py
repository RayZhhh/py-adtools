from adtools import PyProgram

code = r"""
import ast, numba                 # This part will be parsed into PyCodeBlock
import numpy as np

@numba.jit()                      # This part will be parsed into PyFunction
def function(arg1, arg2=True):     
    if arg2:
    	return arg1 * 2
    else:
    	return arg1 * 4

@some.decorators()                # This part will be parsed into PyClass
class PythonClass(BaseClass):
    '''Docstring.'''
    # Comments
    class_var1 = 1                # This part will be parsed into PyCodeBlock
    class_var2 = 2                # and placed in PyClass.body

    def __init__(self, x):        # This part will be parsed into PyFunction
        self.x = x                # and placed in PyClass.functions

    def method1(self):
        return self.x * 10

    @some.decorators()
    def method2(self, x, y):
    	return x + y + self.method1(x)
    
    @some.decorators(100)  
    class InnerClass:             # This part will be parsed into PyCodeBlock
        '''Docstring.'''
        def __init__(self):       # and placed in PyClass.body
            ...

if __name__ == '__main__':        # This part will be parsed into PyCodeBlock
	res = function(1)
	print(res)
	res = PythonClass().method2(1, 2)
"""

p = PyProgram.from_text(code, debug=True)
print(p)
print(f"-------------------------------------")
print(p.classes[0].functions[2].decorator)
print(f"-------------------------------------")
print(p.functions[0].name)
