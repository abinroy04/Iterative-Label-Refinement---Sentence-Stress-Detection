import torch
import evaluate
from tqdm import tqdm
from pathlib import Path
import sys
from huggingface_hub import login
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# HuggingFace dataset configuration
HF_DATASET_NAME = "abinroy04/ITA-word-stress"
HF_TOKEN = "hf_xxx"  # Replace with your actual token

os.environ["HF_HOME"] = "/sd1/jhansi/interns/abin/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/hub"

os.makedirs("/sd1/jhansi/interns/abin/hf_cache", exist_ok=True)
os.makedirs("/sd1/jhansi/interns/abin/hf_cache/transformers", exist_ok=True)
os.makedirs("/sd1/jhansi/interns/abin/hf_cache/datasets", exist_ok=True)
os.makedirs("/sd1/jhansi/interns/abin/tmp", exist_ok=True)


CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))
from whistress import WhiStressInferenceClient
from datasets import Dataset, load_dataset


precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

whistress_client = WhiStressInferenceClient(device="cuda" if torch.cuda.is_available() else "cpu")

def compute_prf_metrics(predictions, references, average="binary"):
    """
    Computes precision, recall, F1, and accuracy using Hugging Face's `evaluate`.
    Args:
        predictions (List[int]): Model's predicted labels.
        references  (List[int]): True labels.
        average     (str): "binary", "macro", "micro", or "weighted".
                          Use "binary" for two-class tasks.
    Returns:
        Dict[str, float]: e.g. {"precision": 0.8, "recall": 0.75, "f1": 0.77, "accuracy": 0.82}
    """
    p = precision_metric.compute(predictions=predictions, references=references, average=average)["precision"]
    r = recall_metric.compute(predictions=predictions, references=references, average=average)["recall"]
    f = f1_metric.compute(predictions=predictions, references=references, average=average)["f1"]
    a = accuracy_metric.compute(predictions=predictions, references=references)["accuracy"]

    return {"precision": p, "recall": r, "f1": f, "accuracy": a}

def calculate_metrics_on_dataset(dataset):
    predictions = []
    references = []
    skipped_samples = 0
    
    for i, sample in enumerate(tqdm(dataset)):
        if 'emphasis_indices' in sample and isinstance(sample['emphasis_indices'], dict) and 'binary' in sample['emphasis_indices']:
            gt_stresses = sample['emphasis_indices']['binary']
        elif 'emphasis_indices' in sample:
            gt_stresses = sample['emphasis_indices']
        else:
            print(f"Skipping sample {i}: Could not find stress information")
            skipped_samples += 1
            continue

        try:
            scored = whistress_client.predict(
                audio=sample['audio'], 
                transcription=sample['transcription'],
                # transcription=None,
                return_pairs=True
            )
            pred_words, pred_stresses = zip(*scored)
            
            # Print the predicted transcription
            predicted_transcription = " ".join(pred_words)
            # Handle length mismatch
            if len(pred_stresses) != len(gt_stresses):
                print(f"Sample {i}: Length mismatch - pred: {len(pred_stresses)}, gt: {len(gt_stresses)}")
                print(f"Transcription: {sample['transcription']}")
                print(f"Predicted Transcription: {predicted_transcription}")
                print(f"Pred stresses: {pred_stresses}")
                print(f"GT stresses: {gt_stresses}")
                
                # Align lengths by truncating to minimum length
                min_len = min(len(pred_stresses), len(gt_stresses))
                if min_len == 0:
                    print(f"Skipping sample {i} due to zero length")
                    skipped_samples += 1
                    continue
                    
                # Truncate to minimum length
                pred_stresses = pred_stresses[:min_len]
                gt_stresses = gt_stresses[:min_len]
                print(f"Aligned to length {min_len}")
                print("---")
            
            predictions.extend(pred_stresses)
            references.extend(gt_stresses)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            skipped_samples += 1
            continue
    
    print(f"Processed {len(dataset) - skipped_samples} samples, skipped {skipped_samples}")
    
    # Compute metrics
    metrics = compute_prf_metrics(predictions, references, average="binary")
    
    # Return both metrics and raw predictions/references for confusion matrix
    return metrics, predictions, references

def load_hf_dataset(dataset_name=HF_DATASET_NAME, token=HF_TOKEN, split="test", max_samples=None):
    """
    Load dataset from HuggingFace
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        token: HuggingFace authentication token
        split: Dataset split to use (default: test)
        max_samples: Maximum number of samples to load (optional)
    
    Returns:
        Dataset object
    """
    print(f"Loading {split} split from HuggingFace dataset: {dataset_name}...")
    
    login(token=token)
    
    dataset = load_dataset(dataset_name, split=split, token=token)
    
    def preprocess_sample(sample):
        if isinstance(sample['audio']['array'], list):
            import numpy as np
            sample['audio']['array'] = np.array(sample['audio']['array'], dtype=np.float32)
        return sample
    
    dataset = dataset.map(preprocess_sample)
    
    print("\nSample after preprocessing:")
    sample = dataset[0]
    for key, value in sample.items():
        if key != 'audio':  # Skip audio for brevity
            print(f"  {key}: {value}")
    
    # Limit to max_samples if specified
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited dataset to first {len(dataset)} samples")
    
    return dataset

if __name__ == "__main__":
    import json
    
    # Load dataset from HuggingFace
    split_name = 'test'
    max_samples = None
    
    print(f"Loading dataset from HuggingFace ({split_name} split)...")
    dataset = load_hf_dataset(split=split_name, max_samples=max_samples)
    
    print(f"Evaluating WhiStress on {len(dataset)} samples...")
    metrics, predictions, references = calculate_metrics_on_dataset(dataset=dataset)
    
    # Print individual metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    results = {
        "dataset": "huggingface",
        "split": split_name,
        "num_samples": len(dataset),
        "metrics": metrics
    }
    print(f"Results: {results}")
