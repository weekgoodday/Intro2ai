import random

class RandomVariables:
    _int_max = 2147483647
    seed = 0
    random.seed(seed)
    
    def __init__(self):
        raise RuntimeError("RandomVariables can not be instantiated.")
    
    @classmethod
    def uniform_int(cls):
        return random.randint(0, cls._int_max)
    
    @classmethod
    def uniform_real(cls):
        return random.random()
    
    @classmethod
    def uniform_permutation(cls, n:int):
        p = list(range(n))
        
        for i in range(n):
            j = random.randint(0, n-1)
            p[i], p[j] = p[j], p[i]

        return p