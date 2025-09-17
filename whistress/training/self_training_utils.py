import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from typing import Dict, List, Any
import numpy as np
import os

os.environ["HF_HOME"] = "/sd1/jhansi/interns/abin/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/hub"

class SelfPredictionDataset(Dataset):
    """
    Dataset that incorporates model's own predictions as training labels for the next epoch.
    """
    def __init__(self, original_dataset, predicted_labels_head, column_name):
        """
        Initialize dataset with original features but replace labels with predicted ones.
        
        Args:
            original_dataset: The original HuggingFace dataset
            predicted_labels_head: Tensor of predicted labels from the model
            column_name: Name of the column to store predicted labels
        """
        self.dataset = original_dataset
        self.predicted_labels = predicted_labels_head
        self.column_name = column_name
        
        # Validate dimensions
        assert len(self.dataset) == len(self.predicted_labels), \
            f"Dataset length ({len(self.dataset)}) doesn't match predictions length ({len(self.predicted_labels)})"
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original item but replace the label
        item = self.dataset[idx].copy()
        if not isinstance(self.predicted_labels[idx], torch.Tensor):
            item[f"labels_head_{self.column_name}"] = torch.tensor(self.predicted_labels[idx])
        else:
            item[f"labels_head_{self.column_name}"] = self.predicted_labels[idx]
        return item
    
    @classmethod
    def create_from_model_predictions(cls, model, original_dataset, data_collator, device='cuda'):
        """
        Factory method to create dataset from model predictions in one pass.
        
        Args:
            model: WhiStress model
            original_dataset: Original training dataset
            data_collator: Data collator for batching
            device: Device to run inference on
            
        Returns:
            SelfPredictionDataset with model's predictions
        """
        model.eval()
        dataloader = DataLoader(
            original_dataset, 
            batch_size=16,
            collate_fn=data_collator
        )
        
        all_predictions = []
        all_indices = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # Get model predictions using the trained model
                outputs = model(
                    input_features=batch["input_features"],
                    whisper_labels=batch.get("whisper_labels", None)
                )
                
                # Store predictions and indices
                all_predictions.append(outputs['preds'].cpu())
                all_indices.append(batch["sentence_index"].cpu())
        
        # Create mapping from sentence index to prediction
        predictions_by_index = {}
        for preds_batch, indices_batch in zip(all_predictions, all_indices):
            for pred, idx in zip(preds_batch, indices_batch):
                predictions_by_index[idx.item()] = pred
        
        # Get column name from original dataset
        column_name = next(key.split('_')[-1] for key in original_dataset.features.keys() 
                          if key.startswith('labels_head_'))
        
        # Create ordered predictions list
        ordered_predictions = []
        for idx in range(len(original_dataset)):
            sent_idx = original_dataset[idx]["sentence_index"]
            ordered_predictions.append(predictions_by_index.get(sent_idx, 
                                      original_dataset[idx][f"labels_head_{column_name}"]))
        
        return cls(original_dataset, ordered_predictions, column_name)

def convert_to_hf_dataset(self_prediction_dataset):
    """
    Convert a SelfPredictionDataset to a HuggingFace Dataset.
    
    Args:
        self_prediction_dataset: SelfPredictionDataset instance
        
    Returns:
        HuggingFace Dataset with the same data
    """
    # Convert all data to dictionaries
    data_dicts = []
    for i in range(len(self_prediction_dataset)):
        item = self_prediction_dataset[i]
        # Ensure all tensors are converted to lists for HF Dataset creation
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                item[k] = v.tolist() if v.numel() > 0 else []
        data_dicts.append(item)
    
    # Create HuggingFace Dataset
    return HFDataset.from_dict({k: [d[k] for d in data_dicts] for k in data_dicts[0].keys()})
