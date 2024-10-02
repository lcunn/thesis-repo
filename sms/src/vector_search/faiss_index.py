import faiss
import numpy as np
from typing import Dict, Any, List, Tuple

class CustomFAISSIndex:
    def __init__(self, index_type: str, index_args: List[Any] = [], index_kwargs: Dict[str, Any] = {}):
        self.index = getattr(faiss, index_type)(*index_args, **index_kwargs)
        self.id_to_index = {}  # Maps custom IDs to FAISS indices
        self.index_to_id = {}  # Maps FAISS indices to custom IDs
        self.id_to_data = {}   # Maps custom IDs to original data

    def add_with_id(self, id, vector, original_data=None):
        if id in self.id_to_index:
            raise ValueError(f"ID {id} already exists in the index")
        
        index = self.index.ntotal
        self.index.add(np.array([vector], dtype=np.float32))
        self.id_to_index[id] = index
        self.index_to_id[index] = id
        if original_data is not None:
            self.id_to_data[id] = original_data

    def remove(self, id):
        if id not in self.id_to_index:
            raise ValueError(f"ID {id} not found in the index")
        
        index_to_remove = self.id_to_index[id]
        self.index.remove_ids(np.array([index_to_remove]))
        
        # Update mappings
        del self.index_to_id[index_to_remove]
        del self.id_to_index[id]
        if id in self.id_to_data:
            del self.id_to_data[id]
        
        # # Update remaining indices
        # for i in range(index_to_remove, self.index.ntotal):
        #     old_id = self.index_to_id[i + 1]
        #     self.index_to_id[i] = old_id
        #     self.id_to_index[old_id] = i
        # del self.index_to_id[self.index.ntotal]

        # Update remaining indices
        for i in range(index_to_remove, self.index.ntotal):
            if i + 1 in self.index_to_id:
                old_id = self.index_to_id[i + 1]
                self.index_to_id[i] = old_id
                self.id_to_index[old_id] = i
        
        # Remove the last entry if it exists
        if self.index.ntotal in self.index_to_id:
            del self.index_to_id[self.index.ntotal]

    # def search(self, query_vector, k,):
    #     distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
    #     results = []
    #     for idx in indices[0]:
    #         if idx != -1 and idx in self.index_to_id:
    #             id = self.index_to_id[idx]
    #             results.append((id, self.id_to_data.get(id)))
    #     return results
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, Any, float]]:
        distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.index_to_id:
                id = self.index_to_id[idx]
                results.append((id, self.id_to_data.get(id), distances[0][i]))
        return results

    def radius_search(self, query_vector: np.ndarray, radius: float) -> List[Tuple[str, Any, float]]:
        lims, distances, indices = self.index.range_search(np.array([query_vector], dtype=np.float32), radius)
        results = []
        for i, idx in enumerate(indices):
            if idx != -1 and idx in self.index_to_id:
                id = self.index_to_id[idx]
                results.append((id, self.id_to_data.get(id), distances[i]))
        return results

    def get_vector(self, id):
        if id not in self.id_to_index:
            raise ValueError(f"ID {id} not found in the index")
        index = self.id_to_index[id]
        return self.index.reconstruct(index)

    def get_original_data(self, id):
        return self.id_to_data.get(id)
    
    def get_all_items(self, limit=3):
        items = []
        for id in list(self.id_to_data.keys())[:limit]:  # Limit the number of items
            vector = self.get_vector(id)
            original_data = self.get_original_data(id)
            items.append((id, vector, original_data))
        return items

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def __repr__(self):
        items = self.get_all_items(limit=3)  # Limit to 3 items
        total_items = self.index.ntotal
        repr_str = f"CustomFAISSIndex with {total_items} items:\n"
        for id, vector, original_data in items:
            repr_str += f"  ID: {id}\n"
            repr_str += f"    Vector: {vector}\n"
            repr_str += f"    Original Data: {original_data}\n"
        if total_items > 3:
            repr_str += f"  ... and {total_items - 3} more items\n"
        return repr_str
