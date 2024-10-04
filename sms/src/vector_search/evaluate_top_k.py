# produce vector embeddings
from uuid import uuid4
import torch
import time
import logging
import numpy as np
import psutil
from typing import Callable, Optional, List, Dict, Any, Union, Tuple
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

# def create_embedding_dict(data_dict: Dict[str, np.ndarray], dumped_lp_config: Dict[str, Any], model: Callable) -> Dict[str, np.ndarray]:
#     """
#     Create the embedding dictionary for the given model. The dumped_lp_config is used to determine the input format of the model.
#     Returns the data_dict, but with embeddings instead of the original data.
#     """
#     formatter = InputFormatter(**dumped_lp_config['input'])
#     formatted_data_list = [torch.from_numpy(formatter(chunk).astype(np.float32).copy()) for chunk in data_dict.values()]
#     formatted_data_stacked = torch.stack(formatted_data_list, dim=0) # shape [num_chunks, *input_shape]
#     embeddings_stacked = model(formatted_data_stacked)
#     embeddings_dict = {key: embeddings_stacked[i].detach().numpy() for i, key in enumerate(data_dict.keys())}
#     return embeddings_dict

def create_embedding_dict(data_dict: Dict[str, np.ndarray], dumped_lp_config: Dict[str, Any], model: Callable, batch_size: int = 256) -> Dict[str, np.ndarray]:
    """
    Create the embedding dictionary for the given model. The dumped_lp_config is used to determine the input format of the model.
    Returns the data_dict, but with embeddings instead of the original data.
    """
    formatter = InputFormatter(**dumped_lp_config['input'])
    embeddings_dict = {}
    
    keys = list(data_dict.keys())
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i+batch_size]
        batch_data = [data_dict[key] for key in batch_keys]
        formatted_data_list = [torch.from_numpy(formatter(chunk).astype(np.float32).copy()) for chunk in batch_data]
        formatted_data_stacked = torch.stack(formatted_data_list, dim=0)
        
        with torch.no_grad():
            embeddings_stacked = model(formatted_data_stacked)
        
        logger.info(f"Created embeddings for {len(embeddings_stacked)} keys, {i} of {len(keys)}")
        
        for j, key in enumerate(batch_keys):
            embeddings_dict[key] = embeddings_stacked[j].cpu().numpy()
    
    return embeddings_dict

def embeddings_to_faiss_index(
        embeddings_dict: Dict[str, np.ndarray], 
        index_type: str, 
        index_args: List[Any] = [], 
        index_kwargs: Dict[str, Any] = {}
    ) -> CustomFAISSIndex:

    if index_type == "IndexIVFFlat":
        # only input with quantizer and dims
        index_args.append(int(np.sqrt(len(embeddings_dict))))
        
    embedding_index = CustomFAISSIndex(index_type=index_type, index_args=index_args, index_kwargs=index_kwargs)
    embeddings = list(embeddings_dict.values())
    embeddings_array = np.vstack(embeddings)

    # train the index if it's a type that requires training
    if index_type in ["IndexIVFFlat", "IndexPQ", "IndexIVFPQ"]:
        embedding_index.train(embeddings_array)

    for key, value in embeddings_dict.items():
        embedding_index.add_with_id(key, value)
    return embedding_index

def embeddings_to_faiss_index_with_data(
        embeddings_dict: Dict[str, Tuple[np.ndarray, Any]], 
        index_type: str, 
        index_args: List[Any] = [], 
        index_kwargs: Dict[str, Any] = {}
    ) -> CustomFAISSIndex:

    if index_type == "IndexIVFFlat":
        # only input with quantizer and dims
        index_args.append(int(np.sqrt(len(embeddings_dict))))
        
    embedding_index = CustomFAISSIndex(index_type=index_type, index_args=index_args, index_kwargs=index_kwargs)

    # train the index if it's a type that requires training
    if index_type in ["IndexIVFFlat", "IndexPQ", "IndexIVFPQ"]:
        embeddings = [value[0] for value in embeddings_dict.values()]
        embeddings_array = np.vstack(embeddings)
        embedding_index.train(embeddings_array)

    for key, (vector, data) in embeddings_dict.items():
        embedding_index.add_with_id(key, vector, data)
    return embedding_index

def evaluate_search(
        embedding_dict: Dict[str, np.ndarray],
        augmented_embedding_dict: Dict[str, Dict[str, np.ndarray]], 
        k_list: List[int],
        index: CustomFAISSIndex,
        time_queries: bool = True,
        measure_memory: bool = False
    ) -> Dict[str, Dict[str, Union[Dict[str, float], float]]]:
    """
    Evaluate the performance of both top-K and radius search for each augmentation type.
    
    Args:
        embedding_dict: dictionary of original embeddings, keyed by data ids
        augmented_embedding_dict: dictionary keyed by original ids, containing dictionaries of augmented data
        k_list: list of k values to evaluate for top-K search
        index: CustomFAISSIndex object initialized with the embedding_dict
        time_queries: whether to time the queries
        measure_memory: whether to measure the memory usage of the index

    Returns:
        results: dictionary of metrics for each augmentation type and search method

    Output Structure:
    {
        augmentation_type: {
            'top_k': {
                k: {
                    'recall': [list of 0s and 1s],
                    'avg_recall': float
                }
                for k in k_list
            },
            'radius': {
                'dataset_proportion_in_radius': [list of floats],
                'avg_dataset_proportion_in_radius': float
            }
        }
        for each augmentation_type
    }

    If time_queries is True, an additional 'query_times' key is added with the following structure:
    'query_times': {
        'top_k': {
            k: {
                'average': float,
                'minimum': float,
                'maximum': float,
                'median': float
            }
            for k in k_list
        },
        'radius': {
            'radii': [list of floats],
            'times': [list of floats]
        }
    }
    """
    results = {aug_type: {
        'top_k': {k: {'recall': []} for k in k_list},
        'radius': {'dataset_proportion_in_radius': []}
    } for aug_type in augmented_embedding_dict[list(augmented_embedding_dict.keys())[0]].keys()}
    
    if time_queries:
        query_times = {
            'top_k': {k: [] for k in k_list},
            'radius': {'radii': [], 'times': []}
        }
    
    for anchor_id, augmentations in augmented_embedding_dict.items():
        anchor_embedding = embedding_dict[anchor_id]
        
        # remove anchor from index
        index.remove(anchor_id)
        
        for aug_type, augmented_embedding in augmentations.items():
            # add augmented data to index
            aug_id = f"{anchor_id}_aug_{aug_type}"
            index.add_with_id(aug_id, augmented_embedding)
            
            # Top-K search
            for k in k_list:
                if time_queries:
                    start_time = time.time()
                    top_k_results = index.search(anchor_embedding, k)
                    query_times['top_k'][k].append(time.time() - start_time)
                else:
                    top_k_results = index.search(anchor_embedding, k)
                
                recall = 1 if aug_id in [id for id, _, _ in top_k_results] else 0
                results[aug_type]['top_k'][k]['recall'].append(recall)
            
            # Radius search
            if time_queries:
                start_time = time.time()
                radius_results = index.radius_search(anchor_embedding, augmented_embedding, aug_id)
                query_time = time.time() - start_time
                radius = np.linalg.norm(anchor_embedding - augmented_embedding)
                query_times['radius']['radii'].append(radius)
                query_times['radius']['times'].append(query_time)
            else:
                radius_results = index.radius_search(anchor_embedding, augmented_embedding, aug_id)
            
            total_in_radius = len(radius_results)
            dataset_proportion_in_radius = total_in_radius / index.index.ntotal
            
            results[aug_type]['radius']['dataset_proportion_in_radius'].append(dataset_proportion_in_radius)
            
            # remove augmented data from index
            index.remove(aug_id)
        
        # add anchor back to index
        index.add_with_id(anchor_id, anchor_embedding)
    
    # Calculate average metrics
    for aug_type in results:
        for k in k_list:
            results[aug_type]['top_k'][k]['avg_recall'] = np.mean(results[aug_type]['top_k'][k]['recall'])
        results[aug_type]['radius']['avg_dataset_proportion_in_radius'] = np.mean(results[aug_type]['radius']['dataset_proportion_in_radius'])
    
    if time_queries:
        results['query_times'] = {
            'top_k': {
                k: {
                    'average': np.mean(query_times['top_k'][k]),
                    'minimum': np.min(query_times['top_k'][k]),
                    'maximum': np.max(query_times['top_k'][k]),
                    'median': np.median(query_times['top_k'][k])
                } for k in k_list
            },
            'radius': {
                'radii': query_times['radius']['radii'],
                'times': query_times['radius']['times']
            }
        }
    
    return results