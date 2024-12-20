from enum import Enum

class Mode(Enum): 
    GENETIC = 1
    IMAGE = 2 
    MULTIMODAL = 3 

x = Mode.GENETIC 
print(x) 
print(type(x))
y = Mode(1) 

print(y)

