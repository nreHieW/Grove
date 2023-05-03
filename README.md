# Grove
`Grove` is a tree-based hierarchical implementation of the Hierarchical Navigable Small World vector search algorithm (see here[https://github.com/nmslib/hnswlib]). Grove supports sub-indices that allows for searches across the sub-indices. 

Currently, it supports 3 different searching algorithms 
- Hierarchical Navigable Small World (HNSW) search for the leaf indices which contain the actual data points
- Flat Search conducts a KNN search to find the top k nearest neighbours. Only supported by sub indices.
- SVM Search is an implementation of Karparthy's notebook[https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb]. Only supported by sub indices.



------
## API Reference 
`Grove` is structured around indices, it contains 3 types of indices: `RootIndex`, `InnerIndex` and `LeafIndex`. Every database starts with a `RootIndex` and contains at least one `LeafIndex` which stores the actual data points.

`BaseEntry`: Every data point in `Grove` is wrapped by a `BaseEntry` which stores each vector and contains metadata for each data point in the form of a python dictionary.

Locations in `Grove` are specified in the following syntax: `A-B-C` where `B` is the child of `A` and `C` is the leaf of the structure,

To access a inner node directly, use the following syntax: `parent.children["child_name"]`.

### Root Indices 
There are 2 types of Root Indices - `SVMRootIndex` and `FlatRootIndex` and they support the following methods. These indices form the base of every database in `Grove`.
1. `insert(self, item: BaseEntry, loc: str) -> None:` inserts a `BaseEntry` into the specified location of the database.
2. `insert_all(self, items: List[BaseEntry], loc: str) -> None:` inserts a list of `BaseEntries` into the specified location. This is the same as calling `insert()` on each item of the list.
3. `create_child(self, new_child_name, t:Index, loc: str, **kwargs) -> None:` creates a child at the specified location with the given `child_name` and `Index` type with the provided `kwargs`. Provide empty string to create child at the current `Index`. 
4. `create_child_level(self, names: List[str], t: Index, loc: str, keys: List[np.array] = None, **kwargs) -> None:` creates `len(names)` number of children at the given location from the given `keys`. Other arguments are the same as `create_child`. Note that this method creates children with the same `kwargs`.
5. `delete_child(self, loc: str) -> None:` deletes the child at the given location.
6. `delete(self, metadata: dict, loc: str) -> None:` deletes the item given a metadata. If `BaseEntry` is provided for metadata, delete the associated entry in the database.
7. `search(self, query: np.array, k: int = 5) -> Tuple[str, List[BaseEntry], np.array]:` searches for the top k data points with the given `query`. Returns a tuple of (location of the leaf node the results are from, the top K results, and an array of distances). 
8. `get_schema(self) -> str:` returns a formatted string of the database structure. 
9. `save_to_disk(self) -> None:` pickles the database to disk
10. `load_from_disk(cls, name: str) -> "RootIndex":` loads the database to disk. Currently does not check if the loading `Index` is the same type as the calling `Index`.


### Inner Indices 
There are 2 types of Inner Indices - `FlatInnerIndex` and `SVMInnerIndex`. They support the same functionality as the Root Indices. However, all inner indices must contain a key to be searchable. As such, they also contain the following methods:
1. `set_key (self, key: np.array) -> bool:` sets a key at the given `Index`. Returns true if successful else if the key is already set, it returns false.
2. `is_searchable(self) -> bool:` checks if the key has been set. 

### Leaf Indices
Leaf Indices are where the actual searching takes place. There are currently 3 different variants: `FlatLeafIndex`, `HNSWLeafIndex` and `FlatLeafIndex`. They contain the following methods:
1. `search(self, query: np.array, k: int = 5) -> Tuple[str, List[List[BaseEntry]], np.array]: -> None` searches for the top k data points with the given `query`. Returns a tuple of (location of the leaf node the results are from, a `Q` x `K` list of results, a `Q` x `K` np.array of distances). 
2. `insert(self, item: BaseEntry, loc: str = None) -> None:` inserts a `BaseEntry` into the specified location of the database.
3. `insert_all(self, items: List[BaseEntry], loc: str) -> None:` inserts a list of `BaseEntries` into the specified location. This is the same as calling `insert()` on each item of the list.
4. `delete(self, metadata: dict, loc: str) -> None:` deletes the item given a metadata. If `BaseEntry` is provided for metadata, delete the associated entry in the database.
5. `get_ids(self) -> List[dict]:` returns all the metadata present in this leaf 

----
## Example Usage

```python
from Grove.Indices import *
from Grove.Entry import BaseEntry
import numpy as np

DIM = 768

# Create a new index
root = FlatRootIndex("root", max_children=10) # Create database

# Create SVM leaf index at current level with 100 max elements 
root.create_child("child1", SVMLeafIndex, max_children=100, loc="", key = np.float32(np.random.rand(DIM)), dim = DIM)

# Create flat inner index at current level with 10 max children leaf indices
root.create_child("child2", FlatInnerIndex, max_children=10, loc = "", key = np.float32(np.random.rand(DIM)))

# Create HNSW leaf index one level lower at child2 with 100 max elements
root.create_child("child2a", HNSWLeafIndex, max_children=100, loc = "child2", key = np.float32(np.random.rand(DIM)), dim = DIM)

root.insert_all([BaseEntry(np.float32(np.random.rand(DIM)), metadata= {}) for i in range(100)], "child1") # insert into child 1
root.insert_all([BaseEntry(np.float32(np.random.rand(DIM)), metadata= {}) for i in range(100)], "child2-child2a") # insert into child 2a

print(root.get_schema())
# FlatRootIndex(name = root, num_children = 2)
#     FlatInnerIndex(name=child2, num_children=1, searchable = True)
#         HNSWLeafIndex(name = child2a, dim = 768, current_count = 100, searchable = True)
#     SVMLeafIndex(name = child1, dim = 768, current_count = 100, searchable = True)

query = np.float32(np.random.rand(DIM)) # Example query in the form of numpy array
path, results, distances = root.search(query, k = 5) 

# print(path) # prints "child1"- -> searched inside child1
```
----
### To Do
- Support multiple queries at once 
- Performance metrics 

       

