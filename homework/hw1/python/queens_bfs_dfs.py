from time import time

from algorithm.breadth_first_search import BreadthFirstSearch
from algorithm.depth_first_search import DepthFirstSearch
from problem.queens import QueensState

if __name__ == '__main__':    
    
    for i in range(8,14):
        t0 = time()
        
        s = QueensState(i)
        
        bfs = BreadthFirstSearch(s)
        
        bfs.search(True, False)
        
        t1=time()

        print(f"bfs: time = {t1 - t0}s")

        dfs = DepthFirstSearch(s)
        
        dfs.search(True, False)
        
        print(f"dfs: time = {time() - t1}s")