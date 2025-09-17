import torch
import evaluate
from tqdm import tqdm
from pathlib import Path
import sys
from huggingface_hub import login

# Add parent directory to Python path
CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))
from whistress import WhiStressInferenceClient

# Add imports for the fine-tuned Whisper model
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig

FINE_TUNED_MODEL_DIR = "/sd1/jhansi/interns/abin/hug-whisper-tune/output/checkpoint-387"

# Load the original WhiStress client
whistress_client = WhiStressInferenceClient(device="cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned Whisper model
print(f"Loading fine-tuned Whisper model from {FINE_TUNED_MODEL_DIR}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
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

# Define normalize function at global scope so it can be used by all functions
def normalize(word):
    """
    Normalize words for comparison, handling both word and digit forms of numbers.
    """
    # Basic normalization first
    w = word.lower().strip(".,!?;:'\"")
    
    # Map number words to digits
    number_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20",
        # Add reverse mapping too
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
    }
    
    # If it's a simple word-to-digit mapping
    if w in number_map:
        return number_map[w]
    
    # Special case for numbers like "10" -> "ten"
    try:
        num_val = int(w)
        if 0 <= num_val <= 20:
            # Map digits back to words for numbers 0-20
            digit_to_word = {
                0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
                5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 
                14: "fourteen", 15: "fifteen", 16: "sixteen", 
                17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"
            }
            return digit_to_word[num_val]
    except ValueError:
        pass  # Not a simple number
    
    return w

# Helper function for improved string matching
def get_best_word_alignments(source_words, target_words):
    """
    Get the best alignment between source and target words using dynamic programming.
    Returns a list of (source_idx, target_idx) pairs.
    """
    import Levenshtein
    import re
    
    # Pre-process words to handle punctuation and case
    processed_source = []
    processed_target = []
    
    for word in source_words:
        # Remove punctuation and lowercase
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        processed_source.append(clean_word)
        
    for word in target_words:
        # Remove punctuation and lowercase
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        processed_target.append(clean_word)
    
    # Create a similarity matrix
    similarity = []
    for s_idx, s_word in enumerate(processed_source):
        row = []
        for t_idx, t_word in enumerate(processed_target):
            # Use normalized Levenshtein distance as similarity score
            s_norm = normalize(s_word)
            t_norm = normalize(t_word)
            
            # Direct match after normalization - give it a perfect score
            if s_norm == t_norm:
                row.append(1.0)
                continue
                
            # For non-matches, use Levenshtein distance
            max_len = max(len(s_norm), len(t_norm))
            if max_len == 0:  # Handle empty strings
                sim = 1.0
            else:
                # Higher score means more similar
                sim = 1.0 - (Levenshtein.distance(s_norm, t_norm) / max_len)
            row.append(sim)
        similarity.append(row)
    
    # Use dynamic programming to find optimal alignment
    # This approach tries to maximize global alignment quality
    alignments = []
    used_targets = set()
    
    # First pass: Assign high confidence matches
    for s_idx, s_scores in enumerate(similarity):
        best_score = -1
        best_t_idx = -1
        
        # Find the best available match for this source word
        for t_idx, score in enumerate(s_scores):
            if t_idx not in used_targets and score > best_score and score > 0.7:  # Higher threshold for first pass
                best_score = score
                best_t_idx = t_idx
        
        # If a good match was found
        if best_t_idx != -1:
            alignments.append((s_idx, best_t_idx))
            used_targets.add(best_t_idx)
    
    # Second pass: Try to align remaining words with lower threshold
    for s_idx, s_scores in enumerate(similarity):
        if any(src == s_idx for src, _ in alignments):
            continue  # Skip already aligned source words
            
        best_score = -1
        best_t_idx = -1
        
        # Find the best available match with lower threshold
        for t_idx, score in enumerate(s_scores):
            if t_idx not in used_targets and score > best_score and score > 0.4:  # Lower threshold for second pass
                best_score = score
                best_t_idx = t_idx
        
        # If a match was found
        if best_t_idx != -1:
            alignments.append((s_idx, best_t_idx))
            used_targets.add(best_t_idx)
    
    return sorted(alignments)  # Sort by source index

def calculate_metrics_on_dataset(dataset):
    # Try to import Levenshtein for better string matching
    import Levenshtein
    
    total_words = 0
    sentence_tables_printed = 0
    ft_sentence_accuracies = []
    all_gt_stresses = []
    all_ft_stresses = []
    
    for sample in tqdm(dataset):
        # Get stress pattern - check which field to use based on structure
        # After preprocessing, emphasis_indices should be directly accessible as a list
        if 'emphasis_binary' in sample:
            gt_stresses = sample['emphasis_binary']
        else:
            gt_stresses = sample['emphasis_indices']  # Now this is already the binary list
            
        gt_words = sample['transcription'].split()
        total_words += len(gt_words)
        
        # Get transcription from fine-tuned Whisper model only
        if has_fine_tuned_model:
            fine_tuned_transcription = transcribe_with_fine_tuned_model(
                sample['audio']['array'], 
                sample['audio']['sampling_rate']
            )
            # Use fine-tuned transcription to get stress patterns
            if fine_tuned_transcription:
                scored_ft = whistress_client.predict(
                    audio=sample['audio'],
                    transcription=fine_tuned_transcription,
                    # transcription=None,                   
                    return_pairs=True
                )
                ft_words, ft_stresses = zip(*scored_ft)
                
                # Calculate WER for fine-tuned transcription
                from jiwer import wer
                fine_tuned_wer = wer(sample['transcription'], fine_tuned_transcription)
        
        if sentence_tables_printed < 1689 and has_fine_tuned_model and fine_tuned_transcription:
            gt_row = []
            gt_stress_row = []
            ft_stress_row = []
            
            # Function to check if words match considering various forms
            def words_match(word1, word2):
                # Basic normalization - more aggressive cleaning
                norm1 = normalize(word1).lower().strip()
                norm2 = normalize(word2).lower().strip()
                
                # Remove all punctuation for comparison
                import re
                norm1 = re.sub(r'[^\w\s]', '', norm1)
                norm2 = re.sub(r'[^\w\s]', '', norm2)
                
                # Direct match after normalization
                if norm1 == norm2:
                    return True
                
                # Handle "we're" vs "we are" cases - split and check
                words1 = norm1.split()
                words2 = norm2.split()
                
                # If one is a contraction that got expanded
                if len(words1) > 1 and words2[0] == words1[0]:
                    return True
                if len(words2) > 1 and words1[0] == words2[0]:
                    return True
                
                # Additional similarity check for common variations
                # Check if one is contained in the other (e.g., "I" in "I'm")
                if (norm1 in norm2) or (norm2 in norm1):
                    # Only consider it a match if it's a significant part
                    shorter = min(len(norm1), len(norm2))
                    longer = max(len(norm1), len(norm2))
                    if shorter > 1 and shorter / longer > 0.5:  # More than half match
                        return True
                
                return False
            
            # Create visualization rows based on alignment
            for i, gt_word in enumerate(gt_words):
                gt_row.append(gt_word)
                
                # Find if this ground truth word is aligned with a fine-tuned model word
                aligned_ft_idx = None
                
                # Use improved alignment algorithm
                try:
                    # Align with fine-tuned model words
                    ft_alignments = get_best_word_alignments(gt_words, ft_words)
                    ft_alignment_dict = {src: tgt for src, tgt in ft_alignments}
                    if i in ft_alignment_dict:
                        aligned_ft_idx = ft_alignment_dict[i]
                except Exception as align_err:
                    print(f"Warning: Error in alignment algorithm: {align_err}")
                    # Fall back to the old method if there's an error
                    for j, ft_word in enumerate(ft_words):
                        if normalize(gt_word).lower() == normalize(ft_word).lower():
                            aligned_ft_idx = j
                            break
                
                # Process alignments for fine-tuned model
                gt_stress_row.append(str(gt_stresses[i]))
                if aligned_ft_idx is not None:
                    ft_stress_row.append(str(ft_stresses[aligned_ft_idx]))
                    
                    # Collect for metrics
                    all_gt_stresses.append(gt_stresses[i])
                    all_ft_stresses.append(ft_stresses[aligned_ft_idx])
                else:
                    # No alignment found
                    ft_stress_row.append("-")
            
            # Print results
            print(f"\nGT transcription: {' '.join(gt_words)}")
            print(f"Fine-tuned transcription: {fine_tuned_transcription}")
            print(" ".join(gt_row))
            print("GT stresses:    " + " ".join(gt_stress_row))
            print("FT stresses:    " + " ".join(ft_stress_row))
            
            # Calculate accuracy for the fine-tuned model's stress predictions
            ft_actual_matched = sum(
                1 for gt_s, ft_s in zip(gt_stress_row, ft_stress_row)
                if ft_s != "-" and gt_s == ft_s
            )
            ft_accuracy = ft_actual_matched / len(gt_words) if len(gt_words) > 0 else 0.0
            print(f"Fine-tuned model sentence accuracy: {ft_actual_matched}/{len(gt_words)} = {ft_accuracy:.2f}\n")
            
            ft_sentence_accuracies.append(ft_accuracy)
            
            sentence_tables_printed += 1
    
    # Print final average accuracy for fine-tuned model
    if ft_sentence_accuracies:
        ft_avg_accuracy = sum(ft_sentence_accuracies) / len(ft_sentence_accuracies)
        print(f"Fine-tuned model average sentence accuracy over {len(ft_sentence_accuracies)} sentences: {ft_avg_accuracy:.4f}")
    
    # Print precision, recall, f1 for fine-tuned model stress prediction
    if all_gt_stresses and all_ft_stresses:
        from sklearn.metrics import precision_score, recall_score, f1_score
        ft_precision = precision_score(all_gt_stresses, all_ft_stresses, zero_division=0)
        ft_recall = recall_score(all_gt_stresses, all_ft_stresses, zero_division=0)
        ft_f1 = f1_score(all_gt_stresses, all_ft_stresses, zero_division=0)
        print(f"\nFine-tuned model word-level metrics:")
        print(f"  Precision: {ft_precision:.4f}")
        print(f"  Recall: {ft_recall:.4f}")
        print(f"  F1 score: {ft_f1:.4f}")


def load_arrow_dataset(split="test", max_samples=None):
    """
    Load dataset from arrow files
    """
    # This function is kept for backward compatibility
    # but we're now using load_hf_dataset() instead
    raise NotImplementedError("This function is deprecated. Use load_hf_dataset() instead.")

def load_hf_dataset(dataset_name="abinroy04/ITA-word-stress", split="test", max_samples=None):
    """
    Load dataset from HuggingFace
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: 'train' or 'test' split
        max_samples: Maximum number of samples to load (optional)
    
    Returns:
        Dataset object
    """
    from datasets import load_dataset
    
    print(f"Loading {split} split from HuggingFace dataset: {dataset_name}...")
    
    # Load the specified split
    dataset = load_dataset(dataset_name, split=split)
    
    # Print info about the dataset
    print(f"Dataset format: {dataset.format}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Dataset features: {list(dataset.features.keys())}")
    
    # Display the nested structure of emphasis_indices
    if 'emphasis_indices' in dataset.features:
        print("\nEmphasis indices structure:")
        for subfield in dataset.features['emphasis_indices'].keys():
            print(f"  {subfield}: {dataset.features['emphasis_indices'][subfield]}")
    
    # Modify the dataset to make emphasis_indices.binary the default emphasis_indices
    # This helps compatibility with the existing evaluation code
    def preprocess_sample(sample):
        # Check if emphasis_indices.binary exists and move it up one level
        if 'emphasis_indices' in sample and 'binary' in sample['emphasis_indices']:
            sample['emphasis_binary'] = sample['emphasis_indices']['binary']
            # For compatibility with the existing code
            sample['emphasis_indices'] = sample['emphasis_indices']['binary']
        return sample
    
    dataset = dataset.map(preprocess_sample)
    
    # Print a sample to verify structure after preprocessing
    print("\nSample after preprocessing:")
    sample = dataset[0]
    for key, value in sample.items():
        if key != 'audio' and key != 'emphasis_indices' and key != 'emphasis_binary':
            print(f"  {key}: {value}")
        elif key == 'emphasis_indices' or key == 'emphasis_binary':
            print(f"  {key}: {value[:5]}{'...' if len(value) > 5 else ''}")
    
    # Limit to max_samples if specified
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited dataset to first {len(dataset)} samples")
    
    return dataset

if __name__ == "__main__":
    # Try to load dataset from HuggingFace
    try:
        print("Loading dataset from HuggingFace: abinroy04/ITA-word-stress...")
        
        # Get HuggingFace token from user or use stored token
        hf_token = "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"
        
        # Login to HuggingFace
        login(token=hf_token)
        
        # Load dataset from HuggingFace - specify the split you want
        dataset_name = "abinroy04/ITA-word-stress"
        split = "test"  # Change to "train" if you want the training split
        
        # Load the dataset
        dataset = load_hf_dataset(dataset_name=dataset_name, split=split)
            
        # Print a sample to verify structure
        print("\nSample from dataset:")
        sample = dataset[0]
        for key, value in sample.items():
            if key != 'audio':  # Skip audio for brevity
                print(f"  {key}: {value}")
            else:
                # Handle the audio data safely, checking its type
                print(f"  audio: [array type: {type(sample['audio']['array'])}, ", end="")
                if hasattr(sample['audio']['array'], 'shape'):
                    print(f"shape: {sample['audio']['array'].shape}, ", end="")
                else:
                    print(f"length: {len(sample['audio']['array'])}, ", end="")
                print(f"sampling rate: {sample['audio']['sampling_rate']}]")
                
                # Convert audio array to numpy array if it's a list
                if isinstance(sample['audio']['array'], list):
                    import numpy as np
                    print("  Converting audio array from list to numpy array...")
                    
                    # Function to convert audio arrays in the dataset
                    def convert_audio_to_numpy(example):
                        if isinstance(example['audio']['array'], list):
                            example['audio']['array'] = np.array(example['audio']['array'], dtype=np.float32)
                        return example
                    
                    # Apply the conversion to the entire dataset
                    dataset = dataset.map(convert_audio_to_numpy)
                    print("  Audio arrays converted to numpy arrays")
                
    except Exception as e:
        print(f"Error loading dataset from HuggingFace: {e}")
        print("Cannot continue without dataset. Exiting.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Ask user which split to evaluate
    print("\nDataset loaded successfully.")
    evaluate_dataset = True
    
    if evaluate_dataset:
        print(f"Evaluating with {len(dataset)} samples from {split} split...")
        calculate_metrics_on_dataset(dataset)
    else:
        print("Evaluation skipped.")