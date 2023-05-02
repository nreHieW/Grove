from typing import List, Dict, Optional
import numpy as np

class BaseEntry:

    data: np.array
    metadata: Optional[Dict[str, object]] = None

    def __init__(self, data: np.array, metadata: Dict[str, object] = None) -> None:
        self.data = np.float32(data)
        self.metadata = metadata

    def __str__(self) -> str:
        return f"BaseEntry(metadata={self.metadata})"
    
    @classmethod
    def from_arr(cls, arr: List[np.array], metadata: List[Dict[str, object]] = None) -> 'List[BaseEntry]':
        """Creates a list of BaseEntries from a list of arrays"""
        if metadata:
            return [cls(arr[i], metadata[i]) for i in range(len(arr))]
        else:
            return [cls(arr[i]) for i in range(len(arr))]
