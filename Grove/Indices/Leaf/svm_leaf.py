from typing import List, Tuple
import numpy as np 
from sklearn import svm
from Grove.Indices.index import LeafIndex
from Grove.Entry.baseentry import BaseEntry

class SVMLeafIndex(LeafIndex):
    name: str
    dim : int
    data: List[BaseEntry]
    clf: svm.LinearSVC

    def __init__(self, name: str, max_children: int, dim: int, key: np.array = None, svm_params: dict = None) -> None:
        """
        Initializes a SVMLeafIndex object.

        Args:
            name: A string representing the name of the index.
            max_children: An integer representing the maximum number of children this index can have.
            dim: An integer representing the dimensionality of the vectors in this index.
            key: A numpy array representing the key vector for this index.
            svm_params: A dictionary containing the parameters to be used in training the SVM.
        """

        self.name = name
        self.key = key
        self.searchable = self.key is not None
        self.dim = dim
        self.max_children = max_children
        self.data_items = []

    def search(self, query: np.array, k: int = 5) -> Tuple[str, List[BaseEntry], np.array]:
        """
        Searches the index for the k nearest neighbors of the query vector.

        Args:
            query: A numpy array representing the query vector.
            k: An integer representing the number of nearest neighbors to return.
        
        Returns:
            A tuple containing:
                - A string representing the location of the index that was searched.
                - A list of BaseEntry objects representing the k nearest neighbors.
                - A numpy array representing the similarities of the k nearest neighbors.
        """
        if not self.is_searchable():
            raise ValueError(f"Index is not searchable, set key first")

        if query.shape[0] != self.dim:
            raise ValueError(f"Query has dimension {query.shape[0]} but expected {self.dim}")
        
        if k > len(self.data_items):
            raise ValueError(f"K is {k} but there are only {len(self.data_items)} vectors in the index ({self.name})")
        
        # Taken from https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb
        # Swap from https://github.com/hwchase17/langchain/blob/master/langchain/retrievers/svm.py#L17

        embeddings = np.array([v.data for v in self.data_items])
        x = np.concatenate([query[None, ...], embeddings])
        y = np.zeros(x.shape[0])
        y[0] = 1
        clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        clf.fit(x, y) 
        similarities = clf.decision_function(x)
        sorted_ix = np.argsort(-similarities)
        zero_index = np.where(sorted_ix == 0)[0][0]
        if zero_index != 0:
            sorted_ix[0], sorted_ix[zero_index] = sorted_ix[zero_index], sorted_ix[0]
        return "", [self.data_items[i - 1] for i in sorted_ix[1:k+1]], similarities[sorted_ix[1:k+1]]

    def insert(self, item: BaseEntry, loc: str) -> None:
        """
        Inserts a new vector into the index.

        Args:
            item: A BaseEntry object representing the vector to be inserted.
            loc: A string representing the location of the index to insert the vector.
        """
        if loc != "": # Must be empty string since it is a leaf node
            raise ValueError(f"Location is {loc} but expected leaf")

        if item.data.shape[0] != self.dim:
            raise ValueError(f"Child has dimension {item.data.shape[0]} but expected {self.dim}")
        
        if len(self.data_items) >= self.max_children:
            raise ValueError(f"Index is full with count {self.max_children}. Cannot insert more elements")
        
        self.data_items.append(item)

    def insert_all(self, items: List[BaseEntry], loc: str = "") -> None:
        """
        Inserts a list of vectors into the index.

        Args:
            items: A list of BaseEntry objects representing the vectors to be inserted.
            loc: A string representing the location of the index to insert the vectors.
        """
        if loc != "":
            raise ValueError(f"Location is {loc} but expected leaf")
        
        for item in items:
            if item.data.shape[0] != self.dim:
                raise ValueError(f"Child has dimension {item.data.shape[0]} but expected {self.dim}")
            if len(self.data_items) >= self.max_children:
                raise ValueError(f"Index is full with count {self.max_children}. Cannot insert more elements")
            self.data_items.append(item)

    def delete(self, metadata: dict, loc: str) -> None:
        """
        Deletes a vector from the index.

        Args:
            metadata: A dictionary representing the metadata of the vector to be deleted.
            loc: A string representing the location of the index to delete the vector.
        """
        if loc != "":
            raise ValueError(f"Location is {loc} but expected leaf")
        if isinstance(metadata, BaseEntry):
            metadata = metadata.metadata
        for i, item in enumerate(self.data_items):
            if item.metadata == metadata:
                self.data_items.pop(i)
                return
    
    def delete_all(self) -> None:
        """
        Deletes all vectors from the index.
        """
        self.data_items = []
       
    def get_ids(self) -> List[dict]:
        """
        Returns the metadata of all the vectors in the index.

        Returns:
            A list of dictionaries representing the metadata of the vectors in the index.
        """
        return [item.metadata for item in self.data_items]

    def __len__(self) -> int:
        """Returns the number of vectors in the index"""
        return len(self.data_items)
    
    def __str__(self) -> str:
         return f"SVMLeafIndex(name = {self.name}, dim = {self.dim}, current_count = {len(self.data_items)}, searchable = {self.searchable})"
       