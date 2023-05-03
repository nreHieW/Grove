import numpy as np
from typing import List, Tuple, Dict
from sklearn import svm
from Grove.Indices.index import RootIndex
from Grove.Entry.baseentry import BaseEntry

class SVMRootIndex(RootIndex):
    
    name: str
    children: Dict

    def __init__(self, name: str, max_children: int) -> None:
        self.name = name
        self.children = dict()
        self.max_children = max_children

    def search(self, query: np.array, k: int = 5) -> Tuple[str, List[BaseEntry], np.array]:
        """Returns a list of k nearest neighbors and their distances to the query point"""
        
        if len(self.children) == 0:
            raise ValueError(f"Index is empty, add children first")
        
        # Taken from https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb
        # Swap from https://github.com/hwchase17/langchain/blob/master/langchain/retrievers/svm.py#L17

        query = np.float32(query)
        keys = np.array([v.key for k, v in self.children.items()])
        x = np.vstack([query[None, ...], keys])
        y = np.zeros(x.shape[0])
        y[0] = 1
        clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        clf.fit(x, y) 
        similarities = clf.decision_function(x)
        sorted_ix = np.argsort(-similarities)
        zero_index = np.where(sorted_ix == 0)[0][0]
        if zero_index != 0:
            sorted_ix[0], sorted_ix[zero_index] = sorted_ix[zero_index], sorted_ix[0]
        best_index = sorted_ix[1] - 1
        best_child = list(self.children.keys())[best_index]
        recursive_res = self.children[best_child].search(query, k)
        path = best_child + "-" + recursive_res[0]
        return path, recursive_res[1], recursive_res[2]
        
    
    def __str__(self) -> str:
        return f"SVMRootIndex(name = {self.name}, num_children = {len(self)})"
    