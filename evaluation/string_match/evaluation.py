import torch
import evaluate
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import login
import sys
import os

# Add parent directory to Python path
CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from whistress import WhiStressInferenceClient

# Load the original WhiStress client
whistress_client = WhiStressInferenceClient(device="cuda" if torch.cuda.is_available() else "cpu")

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
        if 0 <= num_val <= 20:  # Limit to reasonable range
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
    
    # Create a similarity matrix
    similarity = []
    for s_word in source_words:
        row = []
        for t_word in target_words:
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
    
    # Use greedy algorithm to find best matches
    alignments = []
    used_targets = set()
    
    for s_idx, s_scores in enumerate(similarity):
        best_score = -1
        best_t_idx = -1
        
        # Find the best available match for this source word
        for t_idx, score in enumerate(s_scores):
            if t_idx not in used_targets and score > best_score and score > 0.5:  # Threshold for matching
                best_score = score
                best_t_idx = t_idx
        
        # If a good match was found
        if best_t_idx != -1:
            alignments.append((s_idx, best_t_idx))
            used_targets.add(best_t_idx)
    
    return sorted(alignments)  # Sort by source index

def calculate_metrics_on_dataset(dataset):
    # Try to import Levenshtein for better string matching
    import Levenshtein
    
    total_words = 0
    sentence_tables_printed = 0
    sentence_accuracies = []
    all_gt_stresses = []
    all_pred_stresses = []
    
    for sample in tqdm(dataset):
        gt_stresses = sample['emphasis_indices']['binary']
        gt_words = sample['transcription'].split()
        total_words += len(gt_words)
        
        # Original transcription from WhiStress
        scored = whistress_client.predict(
            audio=sample['audio'],
            transcription=None,
            # transcription=sample['transcription'],
            return_pairs=True
        )
        pred_words, pred_stresses = zip(*scored)
        
        if sentence_tables_printed < 1689:
            gt_row = []
            gt_stress_row = []
            pred_stress_row = []
            
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
            
            # Improved alignment algorithm
            i, j = 0, 0
            aligned_pairs = []  # Will store (gt_idx, pred_idx) pairs of aligned words
            
            # Try to align words
            while i < len(gt_words) and j < len(pred_words):
                if words_match(gt_words[i], pred_words[j]):
                    # Words match, create alignment pair
                    aligned_pairs.append((i, j))
                    i += 1
                    j += 1
                else:
                    # Try to find a match by looking ahead in both sequences
                    match_found = False
                    
                    # Look ahead in predicted words
                    for k in range(j+1, min(j+3, len(pred_words))):
                        if words_match(gt_words[i], pred_words[k]):
                            # Skip j to k
                            j = k
                            aligned_pairs.append((i, j))
                            i += 1
                            j += 1
                            match_found = True
                            break
                    
                    if not match_found:
                        # Look ahead in ground truth words
                        for k in range(i+1, min(i+3, len(gt_words))):
                            if words_match(gt_words[k], pred_words[j]):
                                # Skip i to k
                                i = k
                                aligned_pairs.append((i, j))
                                i += 1
                                j += 1
                                match_found = True
                                break
                    
                    if not match_found:
                        # No match found, move both forward
                        i += 1
                        j += 1
            
            # Create visualization rows based on alignment
            for i, gt_word in enumerate(gt_words):
                gt_row.append(gt_word)
                
                # Find if this ground truth word is aligned with a predicted word
                aligned_pred_idx = None
                
                # Use improved alignment algorithm
                try:
                    # Use Levenshtein-based alignment for original model words
                    orig_alignments = get_best_word_alignments(gt_words, pred_words)
                    orig_alignment_dict = {src: tgt for src, tgt in orig_alignments}
                    if i in orig_alignment_dict:
                        aligned_pred_idx = orig_alignment_dict[i]
                except Exception as align_err:
                    print(f"Warning: Error in alignment algorithm: {align_err}")
                    # Fall back to the old method if there's an error
                    for j, pred_word in enumerate(pred_words):
                        if normalize(gt_word).lower() == normalize(pred_word).lower():
                            aligned_pred_idx = j
                            break
                
                # Process alignments for original model
                if aligned_pred_idx is not None:
                    # Word is aligned, compare stress
                    gt_stress_row.append(str(gt_stresses[i]))
                    pred_stress_row.append(str(pred_stresses[aligned_pred_idx]))
                    
                    # Collect for metrics
                    all_gt_stresses.append(gt_stresses[i])
                    all_pred_stresses.append(pred_stresses[aligned_pred_idx])
                else:
                    # No alignment found
                    gt_stress_row.append(str(gt_stresses[i]))
                    pred_stress_row.append("-")
            
            # Print results
            print(f"\nGT transcription: {' '.join(gt_words)}")
            print(f"Predicated transcription: {' '.join(pred_words)}")
            print(" ".join(gt_row))
            print("GT stresses:    " + " ".join(gt_stress_row))
            print("Orig stresses:  " + " ".join(pred_stress_row))
            
            # Calculate accuracy
            actual_matched = sum(
                1 for gt_s, pred_s in zip(gt_stress_row, pred_stress_row)
                if pred_s != "-" and gt_s == pred_s
            )
            accuracy = actual_matched / len(gt_words) if len(gt_words) > 0 else 0.0
            print(f"Original model sentence accuracy: {actual_matched}/{len(gt_words)} = {accuracy:.2f}")
            sentence_accuracies.append(accuracy)
            
            print()  # Add newline for consistent formatting
            
            sentence_tables_printed += 1
    
    # Print final average accuracy for original model
    if sentence_accuracies:
        avg_accuracy = sum(sentence_accuracies) / len(sentence_accuracies)
        print(f"Original model average sentence accuracy over {len(sentence_accuracies)} sentences: {avg_accuracy:.4f}")
    
    # Print precision, recall, f1 for original model stress prediction
    if all_gt_stresses and all_pred_stresses:
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(all_gt_stresses, all_pred_stresses, zero_division=0)
        recall = recall_score(all_gt_stresses, all_pred_stresses, zero_division=0)
        f1 = f1_score(all_gt_stresses, all_pred_stresses, zero_division=0)
        print(f"\nOriginal model word-level metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 score: {f1:.4f}")

if __name__ == "__main__":
    # Load dataset exclusively from Hugging Face
    from datasets import load_dataset
    import numpy as np
    
    print("Loading dataset from Hugging Face...")
    dataset_name = "abinroy04/ITA-word-stress"
    
    # Login to authenticate with private repository
    token = "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"
    login(token=token)
    
    # Load the dataset
    try:
        dataset = load_dataset(dataset_name, token=token)
        print(f"Successfully loaded dataset: {dataset}")
        
        # Select the test split
        split_name = 'test'
        subset = dataset[split_name]
        
        print(f"Using '{split_name}' split with {len(subset)} samples")
        
        # Check if audio arrays need conversion from list to numpy array
        sample = subset[0]
        if isinstance(sample['audio']['array'], list):
            print("Converting audio arrays from lists to numpy arrays...")
            
            def convert_audio_to_numpy(example):
                if isinstance(example['audio']['array'], list):
                    example['audio']['array'] = np.array(example['audio']['array'], dtype=np.float32)
                return example
            
            subset = subset.map(convert_audio_to_numpy)
            print("Audio arrays converted to numpy arrays")
        
        # Display sample information
        print("\nSample from dataset:")
        for key, value in sample.items():
            if key != 'audio':  # Skip audio for brevity
                print(f"  {key}: {value}")
            else:
                audio_type = type(sample['audio']['array']).__name__
                audio_length = len(sample['audio']['array']) if hasattr(sample['audio']['array'], '__len__') else "unknown"
                print(f"  audio: [array type: {audio_type}, length: {audio_length}, sampling rate: {sample['audio']['sampling_rate']}]")
        
        # Print emphasis information structure
        if 'emphasis_indices' in sample:
            print("\nEmphasis indices structure:")
            for key, value in sample['emphasis_indices'].items():
                print(f"  {key}: {type(value).__name__} - {value[:5]}{'...' if len(value) > 5 else ''}")
        
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        import traceback
        traceback.print_exc()
        print("Cannot continue without dataset. Exiting.")
        sys.exit(1)

    print(f"\nEvaluating with {len(subset)} samples...")
    calculate_metrics_on_dataset(subset)