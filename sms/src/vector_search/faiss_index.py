import faiss
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# class CustomFAISSIndex:
#     def __init__(self, index_type: str, index_args: List[Any] = [], index_kwargs: Dict[str, Any] = {}):
#         self.index = getattr(faiss, index_type)(*index_args, **index_kwargs)
#         self.id_to_index = {}  # Maps custom IDs to FAISS indices
#         self.index_to_id = {}  # Maps FAISS indices to custom IDs
#         self.id_to_data = {}   # Maps custom IDs to original data
#         self.supports_range_search = self._check_range_search_support()

#     def _check_range_search_support(self) -> bool:
#         if not hasattr(self.index, 'range_search'):
#             return False
#         try:
#             # Try a dummy range search
#             dummy_vector = np.zeros((1, self.index.d), dtype=np.float32)
#             self.index.range_search(dummy_vector, 1.0)
#             return True
#         except RuntimeError as e:
#             if "range search not implemented" in str(e):
#                 return False
#             raise  # Re-raise if it's a different error

#     def add_with_id(self, id, vector, original_data=None):
#         if id in self.id_to_index:
#             raise ValueError(f"ID {id} already exists in the index")
        
#         index = self.index.ntotal
#         self.index.add(np.array([vector], dtype=np.float32))
#         self.id_to_index[id] = index
#         self.index_to_id[index] = id
#         if original_data is not None:
#             self.id_to_data[id] = original_data

#     def remove(self, id):
#         if id not in self.id_to_index:
#             raise ValueError(f"ID {id} not found in the index")
        
#         index_to_remove = self.id_to_index[id]
#         self.index.remove_ids(np.array([index_to_remove]))
        
#         # Update mappings
#         del self.index_to_id[index_to_remove]
#         del self.id_to_index[id]
#         if id in self.id_to_data:
#             del self.id_to_data[id]

#         # Update remaining indices
#         for i in range(index_to_remove, self.index.ntotal):
#             if i + 1 in self.index_to_id:
#                 old_id = self.index_to_id[i + 1]
#                 self.index_to_id[i] = old_id
#                 self.id_to_index[old_id] = i
        
#         # Remove the last entry if it exists
#         if self.index.ntotal in self.index_to_id:
#             del self.index_to_id[self.index.ntotal]

#     # def search(self, query_vector, k,):
#     #     distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
#     #     results = []
#     #     for idx in indices[0]:
#     #         if idx != -1 and idx in self.index_to_id:
#     #             id = self.index_to_id[idx]
#     #             results.append((id, self.id_to_data.get(id)))
#     #     return results
    
#     def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, Any, float]]:
#         distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
#         results = []
#         for i, idx in enumerate(indices[0]):
#             if idx != -1 and idx in self.index_to_id:
#                 id = self.index_to_id[idx]
#                 results.append((id, self.id_to_data.get(id), distances[0][i]))
#         return results
    
#     def radius_search(self, query_vector: np.ndarray, target_vector: np.ndarray, target_id: str) -> List[Tuple[str, Any, float]]:
#         if self.supports_range_search:
#             radius = np.linalg.norm(query_vector - target_vector)
#             return self.exact_radius_search(query_vector, radius)
#         else:
#             return self.approximate_radius_search(query_vector, target_id, max_k = self.index.ntotal)

#     def exact_radius_search(self, query_vector: np.ndarray, radius: float) -> List[Tuple[str, Any, float]]:
#         lims, distances, indices = self.index.range_search(np.array([query_vector], dtype=np.float32), radius)
#         results = []
#         for i, idx in enumerate(indices):
#             if idx != -1 and idx in self.index_to_id:
#                 id = self.index_to_id[idx]
#                 results.append((id, self.id_to_data.get(id), distances[i]))
#         return results
    
#     def approximate_radius_search(self, query_vector: np.ndarray, target_id: str, max_k: Optional[int] = None) -> List[Tuple[str, Any, float]]:
#         """
#         Approximates range search using binary search to find the smallest k where target_id is found.
#         """
#         if max_k is None:
#             max_k = self.index.ntotal
#         left, right = 1, max_k
#         last_results = None

#         while left <= right:
#             k = (left + right) // 2
#             results = self.search(query_vector, k)
#             found_ids = set(result[0] for result in results)
            
#             if target_id in found_ids:
#                 right = k - 1  # Try to find a smaller k
#                 last_results = results
#             else:
#                 left = k + 1  # Need to search with a larger k

#         return last_results if last_results else self.search(query_vector, max_k)

#     def get_vector(self, id):
#         if id not in self.id_to_index:
#             raise ValueError(f"ID {id} not found in the index")
#         index = self.id_to_index[id]
#         return self.index.reconstruct(index)

#     def get_original_data(self, id):
#         return self.id_to_data.get(id)
    
#     def get_all_items(self, limit=3):
#         items = []
#         for id in list(self.id_to_data.keys())[:limit]:  # Limit the number of items
#             vector = self.get_vector(id)
#             original_data = self.get_original_data(id)
#             items.append((id, vector, original_data))
#         return items
    
#     def train(self, embeddings: np.ndarray):
#         """Allows training for IndexIVFFLat."""
#         self.index.train(embeddings)

#     def __repr__(self):
#         items = self.get_all_items(limit=3)  # Limit to 3 items
#         total_items = self.index.ntotal
#         repr_str = f"CustomFAISSIndex with {total_items} items:\n"
#         for id, vector, original_data in items:
#             repr_str += f"  ID: {id}\n"
#             repr_str += f"    Vector: {vector}\n"
#             repr_str += f"    Original Data: {original_data}\n"
#         if total_items > 3:
#             repr_str += f"  ... and {total_items - 3} more items\n"
#         return repr_str


class CustomFAISSIndex:
    def __init__(self, index_type: str, index_args: List[Any] = [], index_kwargs: Dict[str, Any] = {}):
        self.index = getattr(faiss, index_type)(*index_args, **index_kwargs)
        self.id_to_index = {}  # Maps custom IDs to FAISS indices
        self.index_to_id = {}  # Maps FAISS indices to custom IDs
        self.id_to_data = {}   # Maps custom IDs to original data
        self.deleted_ids = set()  # Add this line
        self.supports_remove = self._check_remove_support()
        self.supports_range_search = self._check_range_search_support()

    def _check_range_search_support(self) -> bool:
        if not hasattr(self.index, 'range_search'):
            return False
        try:
            # Try a dummy range search
            dummy_vector = np.zeros((1, self.index.d), dtype=np.float32)
            self.index.range_search(dummy_vector, 1.0)
            return True
        except RuntimeError as e:
            if "range search not implemented" in str(e):
                return False
            raise  # Re-raise if it's a different error
    
    def _check_remove_support(self) -> bool:
        try:
            # Try a dummy remove operation
            dummy_selector = faiss.IDSelectorBatch([])
            self.index.remove_ids(dummy_selector)
            return True
        except RuntimeError as e:
            if "not implemented" in str(e):
                return False
            raise  # Re-raise if it's a different error

    def add_with_id(self, id, vector, original_data=None):
        if id in self.id_to_index:
            if id not in self.deleted_ids:
                raise ValueError(f"ID {id} already exists in the index and is not deleted")
            else:
                # if it was deleted, just reactivate it
                self.deleted_ids.remove(id)
        else:
            # New ID, add it to the index
            index = self.index.ntotal
            self.index.add(np.array([vector], dtype=np.float32))
            self.id_to_index[id] = index
            self.index_to_id[index] = id

        # Update or add the original data
        if original_data is not None:
            self.id_to_data[id] = original_data

    def remove(self, id):
        if id not in self.id_to_index:
            raise ValueError(f"ID {id} not found in the index")
        
        if self.supports_remove:
            index_to_remove = self.id_to_index[id]
            self.index.remove_ids(np.array([index_to_remove]))
            
            # Update mappings
            del self.index_to_id[index_to_remove]
            del self.id_to_index[id]
            if id in self.id_to_data:
                del self.id_to_data[id]

            # Update remaining indices
            for i in range(index_to_remove, self.index.ntotal):
                if i + 1 in self.index_to_id:
                    old_id = self.index_to_id[i + 1]
                    self.index_to_id[i] = old_id
                    self.id_to_index[old_id] = i
            
            # Remove the last entry if it exists
            if self.index.ntotal in self.index_to_id:
                del self.index_to_id[self.index.ntotal]
        
        else:
            # Soft delete
            self.deleted_ids.add(id)

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
    
    def radius_search(self, query_vector: np.ndarray, target_vector: np.ndarray, target_id: str) -> List[Tuple[str, Any, float]]:
        if self.supports_range_search:
            radius = np.linalg.norm(query_vector - target_vector)
            return self.exact_radius_search(query_vector, radius)
        else:
            return self.approximate_radius_search(query_vector, target_id, max_k = self.index.ntotal)

    def exact_radius_search(self, query_vector: np.ndarray, radius: float) -> List[Tuple[str, Any, float]]:
        lims, distances, indices = self.index.range_search(np.array([query_vector], dtype=np.float32), radius)
        results = []
        for i, idx in enumerate(indices):
            if idx != -1 and idx in self.index_to_id:
                id = self.index_to_id[idx]
                if id not in self.deleted_ids:
                    results.append((id, self.id_to_data.get(id), distances[i]))
        return results
    
    def approximate_radius_search(self, query_vector: np.ndarray, target_id: str, max_k: Optional[int] = None) -> List[Tuple[str, Any, float]]:
        """
        Approximates range search using binary search to find the smallest k where target_id is found.
        """
        if max_k is None:
            max_k = self.index.ntotal
        left, right = 1, max_k
        last_results = None

        while left <= right:
            k = (left + right) // 2
            results = self.search(query_vector, k)
            found_ids = set(result[0] for result in results)
            
            if target_id in found_ids:
                right = k - 1  # Try to find a smaller k
                last_results = results
            else:
                left = k + 1  # Need to search with a larger k

        return last_results if last_results else self.search(query_vector, max_k)

    def get_vector(self, id):
        if id in self.deleted_ids:
            raise ValueError(f"ID {id} has been deleted")
        if id not in self.id_to_index:
            raise ValueError(f"ID {id} not found in the index")
        index = self.id_to_index[id]
        return self.index.reconstruct(index)

    def get_original_data(self, id):
        if id in self.deleted_ids:
            return None
        return self.id_to_data.get(id)
    
    def get_all_items(self, limit=3):
        items = []
        for id in list(self.id_to_data.keys())[:limit]:
            if id not in self.deleted_ids:
                vector = self.get_vector(id)
                original_data = self.get_original_data(id)
                items.append((id, vector, original_data))
        return items
    
    def train(self, embeddings: np.ndarray):
        """Allows training for IndexIVFFLat."""
        self.index.train(embeddings)

    def __repr__(self):
        active_items = self.index.ntotal - len(self.deleted_ids)
        repr_str = f"CustomFAISSIndex with {active_items} active items ({len(self.deleted_ids)} deleted):\n"
        items = self.get_all_items(limit=3)
        for id, vector, original_data in items:
            repr_str += f"  ID: {id}\n"
            repr_str += f"    Vector: {vector}\n"
            repr_str += f"    Original Data: {original_data}\n"
        if active_items > 3:
            repr_str += f"  ... and {active_items - 3} more active items\n"
        return repr_str