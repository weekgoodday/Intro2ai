from typing import Dict
from queue import LifoQueue, Queue

from interface.state import StateBase

def show_reversed_path(last_state_of:Dict[StateBase, StateBase], state:StateBase) -> None:
    
    s = state
    path = LifoQueue()
    
    while s in last_state_of.keys():
        path.put(s)
        s = last_state_of[s]
    path.put(s)
    
    print("<begin>")
    while not path.empty():
        s = path.get()
        s.show()
    print("<end>")

def show_path(next_state_of:Dict[StateBase, StateBase], state:StateBase) -> None:
    print("<begin>")
    s = state
    while s in next_state_of.keys():
        s.show()
        s = next_state_of[s]
    s.show()
    print("<end>")
        
