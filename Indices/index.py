from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import numpy as np
import pickle
import gc
from Entry.baseentry import BaseEntry

class Index(ABC):
    
    name: str
    key: np.array

    @abstractmethod
    def search(self, query, k = 5) -> Tuple[List[BaseEntry], np.array]:
        pass

    @abstractmethod
    def insert(self, item: BaseEntry, loc: str = None) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
    
class SearchableIndex(Index):
    
    searchable: bool
    
    def set_key (self, key: np.array) -> bool:
        """Sets the key of the index"""
        if self.key is not None:
            return False
        else:
            self.key = key
            self.searchable = True
            return True

    def is_searchable(self) -> bool:
        """Returns whether the index is searchable or not"""
        return self.searchable


class InnerIndex(Index):
    """
    InnerIndex is an inner node of the collection. It's children is a list of Indices. 
    """

    name: str
    key: np.array
    children: Dict[str, Index]
    max_children: int

    def insert(self, item: BaseEntry, loc: str) -> None:
        """Inserts a new vector into the index"""
        if not loc or (len(loc) == 0):
            raise ValueError(f"Location must not be empty")
        else:
            loc_list = loc.split("-")
            child_name = loc_list[0].strip().lower()
            if len(loc_list) == 1:
                if child_name not in self.children:
                    raise ValueError(f"Child {child_name} does not exist")
                else:
                    self.children[child_name].insert(item, "")
            else:
                if child_name not in self.children:
                    raise ValueError(f"Child {child_name} does not exist")
                else:
                    self.children[child_name].insert(item, "-".join(loc_list[1:]))
    
    def insert_all(self, items: List[BaseEntry], loc: str) -> None:
        """Inserts a list of new vectors into the index"""
        for item in items:
            self.insert(item, loc)

        
    def create_child(self, child_name, t:Index, **kwargs) -> None: # TODO: if child_name already exists?
        """Inserts a new child index into the index"""
        if len(self.children) >= self.max_children:
            raise ValueError(f"Index is full with count {self.max_children}. Cannot insert more elements")
        
        if child_name in self.children:
            raise ValueError(f"Child {child_name} already exists")
        
        self.children[child_name] = t(name = child_name, **kwargs)

    def delete_child(self, child_name: str) -> None:
        """Deletes a child index from the index"""
        if child_name not in self.children:
            raise ValueError(f"Child {child_name} does not exist")
        else:
            referrers = gc.get_referrers(self.children[child_name])
            for referrer in referrers:
                print(type(referrer), referrer)
            del self.children[child_name]
        
    @abstractmethod
    def search(self, query: np.array, k: int = 5) -> Tuple[List[BaseEntry], np.array]:
        pass
    
    def create_child_level(self, names: List[str], t: Index, keys: List[np.array] = None, **kwargs) -> None:
        if keys is None:
            for name in names:
                self.create_child(name, t, **kwargs)
        else:
            if len(names) != len(keys):
                raise ValueError(f"Length of names and keys must be the same")
            else:
                for name, key in zip(names, keys):
                    self.create_child(name, t, key = key, **kwargs)

    def __len__(self) -> int:
        """Returns the number of children indices in the index"""
        return len(self.children)
    
class RootIndex(InnerIndex):

    name: str

    def get_schema(self) -> str:
        stack = [(self, 0)]
        schema = ""
        while len(stack) > 0:
            curr, level = stack.pop()
            schema += '    ' * level + f"{str(curr)}" + "\n"
            if isinstance(curr, LeafIndex): # Leaf Case
                continue
            for k, v in curr.children.items():
                stack.append((v, level+1))    

        return schema
    
    def save_to_disk(self) -> None:
        with open(f"{self.name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_disk(cls, name: str) -> "RootIndex":
        with open(f"{name}.pkl", "rb") as f:
            return pickle.load(f)

    
class LeafIndex(SearchableIndex):

    @abstractmethod
    def search(self, query: np.array, k: int = 5) -> Tuple[List[BaseEntry], np.array]:
        """Returns a list of k nearest neighbors and their distances to the query point"""
        pass

    @abstractmethod
    def insert(self, item: BaseEntry, loc: str = None) -> None:
        """Inserts a new vector into the index"""
        pass

    @abstractmethod
    def insert_all(self, items: List[BaseEntry], loc: str = "") -> None:
        """Inserts a list of new vectors into the index"""
        pass

    @abstractmethod
    def delete(self, item: int) -> None:
        """Deletes a vector from the index"""
        pass

    @abstractmethod
    def get_ids(self) -> List[str]:
        """Returns a list of ids of the vectors in the index"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of vectors in the index"""
        pass

    