from typing import Dict, List, Tuple
import numpy as np
from sklearn import svm
from Grove.Indices.index import Index, InnerIndex, SearchableIndex
from Grove.Entry.baseentry import BaseEntry

class FlatInnerIndex(InnerIndex, SearchableIndex):

    """
    FlatInnerIndex is an inner node of the collection. It's children is a list of Indices. 
    It is recommended to use this index if the number of children is more than 1000
    It uses flat KNN to search for the nearest neighbors.
    """

    name: str
    key: np.array
    children: Dict[str, Index]
    max_children: int

    def __init__(self, name: str, max_children: int, key: np.array = None) -> None:
        self.name = name
        self.key = key
        self.searchable = self.key is not None
        self.max_children = max_children
        self.children = dict()

    def search(self, query: np.array, k: int = 5) -> Tuple[str, List[BaseEntry], np.array]:
        """Returns a list of k nearest neighbors and their distances to the query point"""
        
        if not self.is_searchable():
            raise ValueError(f"Index is not searchable, add key first")

        if len(self.children) == 0:
            raise ValueError(f"Index is empty, add children first")
        
        query = np.float32(query)
        # KNN Search
        y = np.array([v.key for k, v in self.children.items()], dtype=object)
        similarities = y.dot(query)
        sorted_ix = np.argpartition(-similarities, kth=0)[:1] 
        best_index = sorted_ix[0]
        best_child = list(self.children.keys())[best_index]
        recursive_res = self.children[best_child].search(query, k)
        path = best_child + "-" + recursive_res[0]
        return path, recursive_res[1], recursive_res[2]
    
    def __str__(self) -> str:
        return f"FlatInnerIndex(name={self.name}, num_children={len(self)}, searchable = {self.is_searchable()})"

