from typing import List, Tuple
from Grove.Indices.index import LeafIndex
import numpy as np
from Grove.Entry.baseentry import BaseEntry

class FlatLeafIndex(LeafIndex):
    """
    FlatLeafIndex is the leaf node of the collection. 
    Its children are the actual data points and thus it uses a simple linear search to find the nearest neighbors.
    """

    name: str
    dim : int
    data: List[BaseEntry]

    def __init__(self, name: str, max_children: int, dim: int, key: np.array = None) -> None:

        """
        Initializes a FlatLeafIndex object.

        Args:
            name: A string representing the name of the index.
            max_children: An integer representing the maximum number of children this index can have.
            dim: An integer representing the dimensionality of the vectors in this index.
            key: A numpy array representing the key vector for this index.
        """

        self.name = name
        self.key = key
        self.searchable = self.key is not None
        self.dim = dim
        self.max_children = max_children
        self.data_items = []

    def search(self, query: np.array, k: int = 5) -> Tuple[str, List[BaseEntry], np.array]:
        """Returns a list of k nearest neighbors and their distances to the query point"""
        if not self.is_searchable():
            raise ValueError(f"Index is not searchable, set key first")

        if query.shape[0] != self.dim:
            raise ValueError(f"Query has dimension {query.shape[0]} but expected {self.dim}")
        
        if k > len(self.data_items):
            raise ValueError(f"K is {k} but there are only {len(self.data_items)} vectors in the index")

        embeddings = np.array([v.data for v in self.data_items])
        similarities = embeddings.dot(query)
        sorted_ix = np.argsort(-similarities)
        return "", [self.data_items[i] for i in sorted_ix[:k]], similarities[sorted_ix[:k]]

    def insert(self, item: BaseEntry, loc: str) -> None:
        """Inserts a new vector into the index"""
        if loc != "": # Must be empty string since it is a leaf node
            raise ValueError(f"Location is {loc} but expected leaf")

        if item.data.shape[0] != self.dim:
            raise ValueError(f"Child has dimension {item.data.shape[0]} but expected {self.dim}")
        
        if len(self.data_items) >= self.max_children:
            raise ValueError(f"Index is full with count {self.max_children}. Cannot insert more elements")
        
        self.data_items.append(item)
    
    def insert_all(self, items: List[BaseEntry], loc: str = "") -> None:
        """Inserts a list of vectors into the index"""
        if loc != "":
            raise ValueError(f"Location is {loc} but expected leaf")
        
        for item in items:
            if item.data.shape[0] != self.dim:
                raise ValueError(f"Child has dimension {item.data.shape[0]} but expected {self.dim}")
            if len(self.data_items) >= self.max_children:
                raise ValueError(f"Index is full with count {self.max_children}. Cannot insert more elements")
            self.data_items.append(item)

    def delete(self, metadata: dict, loc: str) -> None:
        if loc != "":
            raise ValueError(f"Location is {loc} but expected leaf")
        if isinstance(metadata, BaseEntry):
            metadata = metadata.metadata
        for i, item in enumerate(self.data_items):
            if item.metadata == metadata:
                self.data_items.pop(i)
                return
    
    def delete_all(self) -> None:
        self.data_items = []
       
    def get_ids(self) -> List[dict]:
        return [item.metadata for item in self.data_items]

    def __len__(self) -> int:
        """Returns the number of vectors in the index"""
        return len(self.data_items)
    
    def __str__(self) -> str:
         return f"FlatLeafIndex(name = {self.name}, dim = {self.dim}, current_count = {len(self.data_items)}, searchable = {self.searchable})"
       

