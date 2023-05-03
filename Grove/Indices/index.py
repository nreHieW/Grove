from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import numpy as np
import pickle
from Grove.Entry.baseentry import BaseEntry

class Index(ABC):
    
    name: str
    key: np.array

    @abstractmethod
    def search(self, query, k = 5) -> Tuple[str, List[List[BaseEntry]], np.array]:
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
        if not loc:
            raise ValueError(f"Location must be provided. Provide empty string for current index")
        else:
            loc_list = loc.split("-")
            child_name = loc_list[0].strip()
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

        
    def create_child(self, new_child_name, t:Index, loc: str, **kwargs) -> None:
        """Inserts a new child index into the index"""
        if len(self.children) >= self.max_children:
            raise ValueError(f"Index is full with count {self.max_children}. Cannot insert more elements")
        
        if loc == "":
            if new_child_name in self.children:
                raise ValueError(f"Child {new_child_name} already exists")
            self.children[new_child_name] = t(name = new_child_name, **kwargs)
        else: 
            loc_list = loc.split("-")
            child_name = loc_list[0].strip()
            if child_name not in self.children:
                raise ValueError(f"Child {child_name} does not exist")
            else:
                self.children[child_name].create_child(new_child_name, t, "-".join(loc_list[1:]), **kwargs)

    def delete_child(self, loc: str) -> None:
        """Deletes a child index from the index"""
        child_name = loc.split("-")[0].strip()
        if loc == "":
            if child_name not in self.children:
                raise ValueError(f"Child {child_name} does not exist")
            else:
                del self.children[child_name]
        else:
            loc_list = loc.split("-")
            if child_name not in self.children:
                raise ValueError(f"Child {child_name} does not exist")
            else:
                self.children[child_name].delete_child("-".join(loc_list[1:]))
    
    def delete(self, metadata: BaseEntry, loc: str) -> None:
        if loc == "":
            raise ValueError(f"Can only delete data points at leaf level")
        else:
            loc_list = loc.split("-")
            child_name = loc_list[0].strip()
            if child_name not in self.children:
                raise ValueError(f"Child {child_name} does not exist")
            else:
                self.children[child_name].delete(metadata, "-".join(loc_list[1:]))
        
    @abstractmethod
    def search(self, query: np.array, k: int = 5) -> Tuple[str, List[List[BaseEntry]], np.array]:
        pass
    
    def create_child_level(self, names: List[str], t: Index, loc: str, keys: List[np.array] = None, **kwargs) -> None:
        if loc == "":
            if keys is None:
                for name in names:
                    self.create_child(name, t, **kwargs)
            else:
                if len(names) != len(keys):
                    raise ValueError(f"Length of names and keys must be the same")
                else:
                    for name, key in zip(names, keys):
                        self.create_child(name, t, key = key, **kwargs)
        else:
            loc_list = loc.split("-")
            child_name = loc_list[0].strip()
            if child_name not in self.children:
                raise ValueError(f"Child {child_name} does not exist")
            else:
                self.children[child_name].create_child_level(names, t, "-".join(loc_list[1:]), keys, **kwargs)

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
    def search(self, query: np.array, k: int = 5) -> Tuple[str, List[List[BaseEntry]], np.array]:
        """Returns a list of k nearest neighbors and their distances to the query point"""
        pass

    @abstractmethod
    def insert(self, item: BaseEntry, loc: str = None) -> None:
        """Inserts a new vector into the index"""
        pass

    @abstractmethod
    def insert_all(self, items: List[BaseEntry], loc: str) -> None:
        """Inserts a list of new vectors into the index"""
        pass

    @abstractmethod
    def delete(self, metadata: dict, loc: str) -> None:
        """Deletes a vector from the index"""
        pass

    @abstractmethod
    def get_ids(self) -> List[dict]:
        """Returns a list of ids of the vectors in the index"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of vectors in the index"""
        pass

    