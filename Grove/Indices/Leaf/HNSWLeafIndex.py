from typing import List, Tuple
from Grove.Indices.index import LeafIndex
import numpy as np
import hnswlib
from Grove.Entry.baseentry import BaseEntry


class HNSWLeafIndex(LeafIndex):
    """
    HNSWLeafIndex is the leaf node of the collection. 
    Its children are the actual data points and thus it uses the HNSW index to search for the nearest neighbors.
    """

    name: str
    key: np.array
    dim : int
    index: hnswlib.Index
    labels: dict
    curr_count: int

    def __init__(self, name: str, max_children: int, dim: int, key: np.array = None, **kwargs) -> None:

        """
        Initializes an HNSWLeafIndex object.

        Args:
            name: A string representing the name of the index.
            max_children: An integer representing the maximum number of children this index can have.
            dim: An integer representing the dimensionality of the vectors in this index.
            key: A numpy array representing the key vector for this index.
            kwargs: Additional arguments for hnswlib.Index. Refer to github.com/nmslib/hnswlib
        """

        self.name = name
        self.key = key
        self.searchable = self.key is not None
        self.dim = dim
        self.max_children = max_children
        self.labels_mapping = dict()
        self.index = hnswlib.Index(space = 'l2', dim = dim)
        self.curr_count = 0
        ef_construction = kwargs.get("ef_construction", 200)
        m = kwargs.get("M", 16)
        self.ef_construction = ef_construction
        self.m = m
        self.index.init_index(max_elements = max_children, ef_construction = ef_construction, M = m, allow_replace_deleted = True)

    def search(self, query: np.array, k: int = 5) -> Tuple[str, List[List[BaseEntry]], np.array]:
        """Returns a list of k nearest neighbors and their distances to the query point"""
        if not self.is_searchable():
            raise ValueError(f"Index is not searchable, set key first")

        if query.shape[0] != self.dim:
            raise ValueError(f"Query has dimension {query.shape[0]} but expected {self.dim}")
        
        if k > self.index.get_current_count():
            raise ValueError(f"K is {k} but there are only {self.index.get_current_count()} vectors in the index")
        query = np.float32(query)
        labels, distances = self.index.knn_query(query, k)

        res = []
        for label in labels:
            query_res = []
            items = self.index.get_items(label)
            for i in range(len(items)):
                content = self.labels_mapping.get(label[i])
                if "id" not in content:
                    content["id"] = label[i]
                query_res.append(BaseEntry(np.array(items[i]), metadata = content))
            res.append(query_res)

        return "", res, distances

    def insert(self, item: BaseEntry, loc: str) -> None:
        """Inserts a new vector into the index"""

        if loc != "": # Must be empty string since it is a leaf node
            raise ValueError(f"Location is {loc} but expected leaf")

        if item.data.shape[0] != self.dim:
            raise ValueError(f"Child has dimension {item.shape[0]} but expected {self.dim}")
        
        if self.index.get_current_count() >= self.max_children:
            raise ValueError(f"Index is full with count {self.max_children}. Cannot insert more elements")
        self.labels_mapping[self.curr_count] = item.metadata
        self.index.add_items(item.data, ids = self.curr_count, replace_deleted = True)
        self.curr_count += 1
    
    def insert_all(self, items: List[BaseEntry], loc: str = "") -> None:
        """Inserts a list of vectors into the index"""
        if loc != "":
            raise ValueError(f"Location is {loc} but expected leaf")
        
        int_labels = []
        for item in items:
            if item.data.shape[0] != self.dim:
                raise ValueError(f"Child has dimension {item.shape[0]} but expected {self.dim}")
            else:
                int_labels.append(self.curr_count)
                self.labels_mapping[self.curr_count] = item.metadata
                self.curr_count += 1
        self.index.add_items(np.array([item.data for item in items]), ids = np.asarray(int_labels), replace_deleted = True)

    def delete(self, metadata: dict) -> None:
        for label, m in self.labels_mapping.items():
            if m == metadata:
                del self.labels_mapping[label]
                self.index.mark_deleted(label)
                return
    
    def delete_all(self) -> None:
        self.index = hnswlib.Index(space = 'l2', dim = self.dim)
        self.curr_count = 0
        self.index.init_index(max_elements = self.max_children, ef_construction = self.ef_construction, M = self.m, allow_replace_deleted = True)
        self.labels_mapping = dict()


    def get_ids(self) -> dict:
        return self.labels_mapping

    def __len__(self) -> int:
        """Returns the number of vectors in the index"""
        return self.index.get_current_count()
    
    def __str__(self) -> str:
        return f"HNSWLeafIndex(name = {self.name}, dim = {self.dim}, current_count = {self.index.get_current_count()}, searchable = {self.searchable})"

