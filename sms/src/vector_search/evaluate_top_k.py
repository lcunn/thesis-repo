# produce vector embeddings
from uuid import uuid4
import torch
import time
import logging
import numpy as np
from typing import Callable, Optional, List, Dict, Any, Union
from sms.src.synthetic_data.formatter import InputFormatter
from sms.src.synthetic_data.note_arr_mod import NoteArrayModifier
from sms.src.vector_search.faiss_index import CustomFAISSIndex

from sms.exp1.run_training import build_encoder, build_projector
from sms.exp1.models.siamese import SiameseModel

logger = logging.getLogger(__name__)

def augment_chunk(chunk: np.ndarray, augmentation: str):
    """ 
    augmentation is one of the following:
        use_transposition
        use_shift_selected_notes_pitch
        use_change_note_durations
        use_delete_notes
        use_insert_notes
    """
    aug_dict = {
        "use_transposition": False,
        "use_shift_selected_notes_pitch": False,
        "use_change_note_durations": False,
        "use_delete_notes": False,
        "use_insert_notes": False
    }
    aug_dict[augmentation] = True
    modifier = NoteArrayModifier()
    return modifier(chunk, aug_dict)

def create_augmented_data(data_dict: Dict[str, np.ndarray], anchor_keys: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create the augmented data for the given anchor keys.
    Returns a dictionary of dictionaries, where the outer dictionary is keyed by the anchor keys, and the inner dictionary 
        is keyed by the type of augmentation and contains the augmented data.
    """
    augmented_data = {}
    for key in anchor_keys:
        chunk = data_dict[key]
        augmented_data[key] = {
            "chunk_transposed": augment_chunk(chunk, "use_transposition"),
            "chunk_one_pitch_shifted": augment_chunk(chunk, "use_shift_selected_notes_pitch"),
            "chunk_note_duration_changed": augment_chunk(chunk, "use_change_note_durations"),
            "chunk_note_deleted": augment_chunk(chunk, "use_delete_notes"),
            "chunk_note_inserted": augment_chunk(chunk, "use_insert_notes")
        }
    return augmented_data

def build_model(dumped_lp_config: Dict[str, Any], full_model_path: Optional[str] = None, encoder_path: Optional[str] = None, use_full_model: bool = False):
    """
    Only one of full_model_path or encoder_path should be provided. If both are provided, full_model_path is used.
    """
    encoder = build_encoder(dumped_lp_config)
    projector = build_projector(dumped_lp_config)
    model = SiameseModel(encoder, projector)
    if full_model_path is not None:
        model.load_state_dict(torch.load(full_model_path))
    elif encoder_path is not None:
        model = model.get_encoder()
        model.load_state_dict(torch.load(encoder_path))
    else:
        raise ValueError("Either full_model_path or encoder_path must be provided.")
    if not use_full_model and full_model_path is not None:
        model = model.get_encoder()
    return model

def create_embedding_dict(data_dict: Dict[str, np.ndarray], dumped_lp_config: Dict[str, Any], model: Callable) -> Dict[str, np.ndarray]:
    """
    Create the embedding dictionary for the given model. The dumped_lp_config is used to determine the input format of the model.
    Returns the data_dict, but with embeddings instead of the original data.
    """
    formatter = InputFormatter(**dumped_lp_config['input'])
    formatted_data_list = [torch.from_numpy(formatter(chunk).astype(np.float32).copy()) for chunk in data_dict.values()]
    formatted_data_stacked = torch.stack(formatted_data_list, dim=0) # shape [num_chunks, *input_shape]
    embeddings_stacked = model(formatted_data_stacked)
    embeddings_dict = {key: embeddings_stacked[i].detach().numpy() for i, key in enumerate(data_dict.keys())}
    return embeddings_dict

def embeddings_to_faiss_index(
        embeddings_dict: Dict[str, np.ndarray], 
        index_type: str, 
        index_args: List[Any] = [], 
        index_kwargs: Dict[str, Any] = {}
    ) -> CustomFAISSIndex:

    embedding_index = CustomFAISSIndex(index_type=index_type, index_args=index_args, index_kwargs=index_kwargs)
    for key, value in embeddings_dict.items():
        embedding_index.add_with_id(key, value)
    return embedding_index

    # For each embedding collection in embeddings_dicts, we perform the augmentation evaluation experiment num_loops times.
    # An augmentation evaluation experiment involves the following steps:
    # - Randomly select an anchor from data_dict
    # - Remove the anchor from data_dict
    # - Apply each of the five given augmentations to the anchor
    # - For each of the augmented melodies, add it to the database and perform a nearest neighbor search on the FAISS index
    # - Calculate the precision and recall of the search for each k in k_list

def evaluate_top_k(
        embedding_dict: Dict[str, Dict[str, np.ndarray]],
        augmented_embedding_dict: Dict[str, Dict[str, np.ndarray]], 
        k_list: List[int], 
        index: CustomFAISSIndex,
        time_queries: bool = False
    ) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    index is a CustomFAISSIndex object which has been initialized with the embeddings_dict.
    For each of the keys in augment_dict, we perform the following steps:
    - Remove the anchor (embedding_dict[key]) from the index
    - Add one of the augmentations from that key to the index
    - Perform a nearest neighbor search on the index using the anchor and record the position of the augmentation
    - Repeat for each augmentation
    
    Then we report the average precision and recall for each k in k_list.
    
    Args:
        embeddings_dict: dictionary of embeddings, keyed by data ids
        augmented_embedding_dict: dictionary keyed by a subset of the ids in embeddings_dict, containing dictionaries of augmented data
        k_list: list of k values to evaluate
        index: CustomFAISSIndex object which has been initialized with the embeddings_dict
        time_queries: whether to time the queries
    Returns:
        results: dictionary of precision and recall for each augmentation and k in k_list
    """
    results = {aug_type: {k: {'precision': [], 'recall': []} for k in k_list} for aug_type in augmented_embedding_dict[list(augmented_embedding_dict.keys())[0]].keys()}
    
    if time_queries:
        query_times = []
    
    for anchor_id, augmentations in augmented_embedding_dict.items():
        anchor_embedding = embedding_dict[anchor_id]
        
        # remove anchor from index
        index.remove(anchor_id)
        
        for aug_type, augmented_data in augmentations.items():
            # add augmented data to index
            aug_id = f"{anchor_id}_aug_{aug_type}"
            index.add_with_id(aug_id, augmented_data)
            
            # perform search
            if time_queries:
                start_time = time.time()
                search_results = index.search(anchor_embedding, max(k_list))
                end_time = time.time()
                query_times.append(end_time - start_time)
            else:
                search_results = index.search(anchor_embedding, max(k_list))
            
            # calculate precision and recall for each k
            for k in k_list:
                top_k_results = search_results[:k]
                true_positives = sum(1 for id, _ in top_k_results if id == aug_id)
                
                precision = true_positives / k
                recall = 1 if true_positives > 0 else 0  # Recall is 1 if found, 0 if not
                
                results[aug_type][k]['precision'].append(precision)
                results[aug_type][k]['recall'].append(recall)
            
            # remove augmented data from index
            index.remove(aug_id)
        
        # add anchor back to index
        index.add_with_id(anchor_id, anchor_embedding)
    
    # Calculate average precision and recall
    for aug_type in results:
        for k in k_list:
            results[aug_type][k]['avg_precision'] = np.mean(results[aug_type][k]['precision'])
            results[aug_type][k]['avg_recall'] = np.mean(results[aug_type][k]['recall'])
    
    if time_queries:
        results['query_times'] = {
            'average': np.mean(query_times),
            'minimum': np.min(query_times),
            'maximum': np.max(query_times),
            'median': np.median(query_times)
        }
    
    return results

def evaluate_search(
        embedding_dict: Dict[str, np.ndarray],
        augmented_embedding_dict: Dict[str, Dict[str, np.ndarray]], 
        k_list: List[int],
        index: CustomFAISSIndex,
        time_queries: bool = False
    ) -> Dict[str, Dict[str, Union[Dict[str, float], float]]]:
    """
    Evaluate the performance of both top-K and radius search for each augmentation type.
    
    Args:
        embedding_dict: dictionary of original embeddings, keyed by data ids
        augmented_embedding_dict: dictionary keyed by original ids, containing dictionaries of augmented data
        k_list: list of k values to evaluate for top-K search
        index: CustomFAISSIndex object initialized with the embedding_dict
        time_queries: whether to time the queries
    Returns:
        results: dictionary of metrics for each augmentation type and search method
    """
    results = {aug_type: {
        'top_k': {k: {'precision': [], 'recall': []} for k in k_list},
        'radius': {'recall': [], 'percentage_in_radius': []}
    } for aug_type in augmented_embedding_dict[list(augmented_embedding_dict.keys())[0]].keys()}
    
    if time_queries:
        query_times = {'top_k': [], 'radius': []}
    
    for anchor_id, augmentations in augmented_embedding_dict.items():
        anchor_embedding = embedding_dict[anchor_id]
        
        # remove anchor from index
        index.remove(anchor_id)
        
        for aug_type, augmented_data in augmentations.items():
            # add augmented data to index
            aug_id = f"{anchor_id}_aug_{aug_type}"
            index.add_with_id(aug_id, augmented_data)
            
            # Top-K search
            if time_queries:
                start_time = time.time()
                top_k_results = index.search(anchor_embedding, max(k_list))
                query_times['top_k'].append(time.time() - start_time)
            else:
                top_k_results = index.search(anchor_embedding, max(k_list))
            
            for k in k_list:
                top_k = top_k_results[:k]
                true_positives = sum(1 for id, _, _ in top_k if id == aug_id)
                precision = true_positives / k
                recall = 1 if true_positives > 0 else 0
                results[aug_type]['top_k'][k]['precision'].append(precision)
                results[aug_type]['top_k'][k]['recall'].append(recall)
            
            # Radius search
            radius = np.linalg.norm(anchor_embedding - augmented_data)
            if time_queries:
                start_time = time.time()
                radius_results = index.radius_search(anchor_embedding, radius)
                query_times['radius'].append(time.time() - start_time)
            else:
                radius_results = index.radius_search(anchor_embedding, radius)
            
            total_in_radius = len(radius_results)
            recall = 1 if aug_id in [id for id, _, _ in radius_results] else 0
            percentage_in_radius = total_in_radius / index.ntotal
            
            results[aug_type]['radius']['recall'].append(recall)
            results[aug_type]['radius']['percentage_in_radius'].append(percentage_in_radius)
            
            # remove augmented data from index
            index.remove(aug_id)
        
        # add anchor back to index
        index.add_with_id(anchor_id, anchor_embedding)
    
    # Calculate average metrics
    for aug_type in results:
        for k in k_list:
            results[aug_type]['top_k'][k]['avg_precision'] = np.mean(results[aug_type]['top_k'][k]['precision'])
            results[aug_type]['top_k'][k]['avg_recall'] = np.mean(results[aug_type]['top_k'][k]['recall'])
        results[aug_type]['radius']['avg_recall'] = np.mean(results[aug_type]['radius']['recall'])
        results[aug_type]['radius']['avg_percentage_in_radius'] = np.mean(results[aug_type]['radius']['percentage_in_radius'])
    
    if time_queries:
        results['query_times'] = {
            'top_k': {
                'average': np.mean(query_times['top_k']),
                'minimum': np.min(query_times['top_k']),
                'maximum': np.max(query_times['top_k']),
                'median': np.median(query_times['top_k'])
            },
            'radius': {
                'average': np.mean(query_times['radius']),
                'minimum': np.min(query_times['radius']),
                'maximum': np.max(query_times['radius']),
                'median': np.median(query_times['radius'])
            }
        }
    
    return results