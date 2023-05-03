import numpy as np
from typing import List, Tuple, Dict
from Grove.Indices.index import RootIndex
from Grove.Entry.baseentry import BaseEntry

class FlatRootIndex(RootIndex):

    name: str 
    children: Dict
    max_children: int

    def __init__(self, name: str, max_children: int) -> None:
        self.name = name
        self.children = dict()
        self.max_children = max_children
    
    def search(self, query: np.array, k: int = 5) -> Tuple[str, List[BaseEntry], np.array]:
        """Returns a list of k nearest neighbors and their distances to the query point"""

        if len(self.children) == 0:
            raise ValueError(f"Index is empty, add children first")
        
        # KNN Search
        query = np.float32(query)
        y = np.array([v.key for k, v in self.children.items()], dtype=object)
        similarities = y.dot(query)
        sorted_ix = np.argpartition(-similarities, kth=0)[:1] 
        best_index = sorted_ix[0] 
        best_child = list(self.children.keys())[best_index]
        recursive_res =  self.children[best_child].search(query, k)
        path = best_child + "-" + recursive_res[0]
        return path, recursive_res[1], recursive_res[2]
    
    def __str__(self) -> str:
        return f"FlatRootIndex(name = {self.name}, num_children = {len(self)})"
    
