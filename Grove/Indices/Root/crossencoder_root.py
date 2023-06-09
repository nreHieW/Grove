from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from Grove.Indices.index import RootIndex
from Grove.Entry.baseentry import BaseEntry


class CrossEncoderRootIndex(RootIndex):
    
    name: str
    children: Dict
    model: CrossEncoder

    def __init__(self, name: str, max_children: int) -> None:
        self.name = name
        self.children = dict()
        self.max_children = max_children

    def search(self, query: np.array, k: int = 5, encoder: CrossEncoder = None, query_str: str = None) -> Tuple[str, List[BaseEntry], np.array]:
        """
        Searches the index for the k nearest neighbors of the query vector.

        Args:
            query: A numpy array representing the query vector.
            k: An integer representing the number of nearest neighbors to return.
            encoder: A CrossEncoder model to use for determining the most relevant child.
            query_str: A string representing the query to use for determining the most relevant child.
        
        Returns:
            A tuple containing:
                - A string representing the location of the index that was searched.
                - A list of BaseEntry objects representing the k nearest neighbors.
                - A numpy array representing the similarities of the k nearest neighbors.
        """
        if encoder is None:
            raise ValueError(f"CrossEncoder model must be provided")
        if query_str is None:
            raise ValueError(f"Query string must be provided")

        if len(self.children) == 0:
            raise ValueError(f"Index is empty, add children first")

        # get the most relevant child using the cross encoder model
        scores = encoder.predict([(query_str, child) for child in self.children.keys()])
        best_idx = np.argmax(scores)
        best_child = list(self.children.keys())[best_idx]

        recursive_res = self.children[best_child].search(query, k)
        path = best_child + "-" + recursive_res[0]
        return path, recursive_res[1], recursive_res[2]
        
    
    def __str__(self) -> str:
        return f"CrossEncoderRootIndex(name={self.name}, num_children={len(self)}"
