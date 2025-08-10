from adtools import PyProgram

if __name__ == '__main__':
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
