import torch
import evaluate
from tqdm import tqdm
from pathlib import Path
import sys
from huggingface_hub import login
import os

CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))
from whistress import WhiStressInferenceClient
from datasets import Dataset, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig

os.environ["HF_HOME"] = "/sd1/jhansi/interns/abin/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/hub"

os.makedirs("/sd1/jhansi/interns/abin/hf_cache", exist_ok=True)
os.makedirs("/sd1/jhansi/interns/abin/hf_cache/transformers", exist_ok=True)
os.makedirs("/sd1/jhansi/interns/abin/hf_cache/datasets", exist_ok=True)
os.makedirs("/sd1/jhansi/interns/abin/tmp", exist_ok=True)

FINE_TUNED_MODEL_DIR = "/sd1/jhansi/interns/abin/hug-whisper-tune/output/checkpoint-387"
HF_DATASET_NAME = "abinroy04/ITA-word-stress"
HF_TOKEN = "hf_xxx"  # Replace with your actual 

precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

whistress_client = WhiStressInferenceClient(device="cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    print(f"Loading fine-tuned Whisper model from {FINE_TUNED_MODEL_DIR}...")
    fine_tuned_model = WhisperForConditionalGeneration.from_pretrained(FINE_TUNED_MODEL_DIR).to(device)
    fine_tuned_processor = WhisperProcessor.from_pretrained(FINE_TUNED_MODEL_DIR)
    fine_tuned_model.eval()
    fine_tuned_model.generation_config = GenerationConfig.from_pretrained("openai/whisper-small.en")
    print(f"Successfully loaded fine-tuned Whisper model on {device}")
    has_fine_tuned_model = True
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    has_fine_tuned_model = False

# Function to transcribe using fine-tuned model
def transcribe_with_fine_tuned_model(audio_array, sampling_rate=16000):
    if not has_fine_tuned_model:
        return None
    
    try:
        # Process audio
        input_features = fine_tuned_processor(
            audio_array, 
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            # Enable hidden states output for better compatibility
            fine_tuned_model.config.output_hidden_states = False
            fine_tuned_model.config.output_attentions = False
            fine_tuned_model.config.return_dict = True
            
            predicted_ids = fine_tuned_model.generate(
                input_features,
                do_sample=False,
                max_new_tokens=225
            )
            transcription = fine_tuned_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # Normalize capitalization
            transcription = transcription[0].upper() + transcription[1:].lower() if transcription else ""
        
        return transcription
    except Exception as e:
        print(f"Error in fine-tuned model transcription: {e}")
        return None


def compute_prf_metrics(predictions, references, average="binary"):
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
        gt_stresses = sample['emphasis_indices']
        try:
            # First get transcription from fine-tuned model
            fine_tuned_transcription = None
            if has_fine_tuned_model:
                fine_tuned_transcription = transcribe_with_fine_tuned_model(
                    sample['audio']['array'], 
                    sample['audio']['sampling_rate']
                )
            
            scored = whistress_client.predict(
                audio=sample['audio'],
                # transcription=fine_tuned_transcription,  # Use fine-tuned model's transcription
                transcription=sample['transcription'],  # Use ground truth transcription
                # transcription=None,                     # Use WhiStress's internal ASR
                return_pairs=True
            )
            pred_words, pred_stresses = zip(*scored)
            
            # Print the predicted transcription
            predicted_transcription = " ".join(pred_words)
            print("---")
            print(f"Fine-tuned model transcription: {fine_tuned_transcription}")
            print("Predicted transcription:", predicted_transcription)
            print("GT transcription:", sample['transcription'])
            
            # Handle length mismatch
            if len(pred_stresses) != len(gt_stresses):
                print(f"Sample {i}: Length mismatch - pred: {len(pred_stresses)}, gt: {len(gt_stresses)}")
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
    
    # Calculate metrics for fine-tuned model only
    metrics = compute_prf_metrics(predictions, references, average="binary")
    
    return metrics

def load_hf_dataset(dataset_name=HF_DATASET_NAME, token=HF_TOKEN, max_samples=None):
    """
    Load dataset from HuggingFace
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: 'train' or 'test' split
        token: HuggingFace authentication token
        max_samples: Maximum number of samples to load (optional)
    
    Returns:
        Dataset object
    """
    print(f"Loading test split from HuggingFace dataset: {dataset_name}...")
    
    # Login to authenticate with HuggingFace
    login(token=token)
    
    dataset = load_dataset(dataset_name, split="test", token=token)
    
    if 'emphasis_indices' in dataset.features:
        print("\nEmphasis indices structure:")
        for subfield in dataset.features['emphasis_indices'].keys():
            print(f"  {subfield}: {dataset.features['emphasis_indices'][subfield]}")
    
    def preprocess_sample(sample):
        if isinstance(sample['emphasis_indices'], dict) and 'binary' in sample['emphasis_indices']:
            sample['emphasis_indices_original'] = sample['emphasis_indices'].copy()
            sample['emphasis_indices'] = sample['emphasis_indices']['binary']
        
        if isinstance(sample['audio']['array'], list):
            import numpy as np
            sample['audio']['array'] = np.array(sample['audio']['array'], dtype=np.float32)
        return sample
    
    dataset = dataset.map(preprocess_sample)
    
    # Print a sample to verify structure after preprocessing
    print("\nSample after preprocessing:")
    sample = dataset[0]
    for key, value in sample.items():
        if key != 'audio' and key != 'emphasis_indices':
            print(f"  {key}: {value}")
        elif key == 'emphasis_indices':
            print(f"  {key}: {value[:5]}{'...' if len(value) > 5 else ''}")
    
    # Limit to max_samples if specified
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited dataset to first {len(dataset)} samples")
    
    return dataset

if __name__ == "__main__":
    import json
    
    max_samples = None  # Use all samples
    
    print(f"Loading HuggingFace dataset (test split)...")
    dataset = load_hf_dataset(max_samples=max_samples)
    
    print(f"Evaluating WhiStress with fine-tuned Whisper model on {len(dataset)} samples...")
    metrics = calculate_metrics_on_dataset(dataset=dataset)
    
    # Print individual metrics for fine-tuned model only
    print("\nFine-tuned model metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    # Save or log the metrics as needed
    results = {
        "dataset": "huggingface",
        "split": "test",
        "num_samples": len(dataset),
        "metrics": metrics
    }
    
    print(f"Results: {results}")
