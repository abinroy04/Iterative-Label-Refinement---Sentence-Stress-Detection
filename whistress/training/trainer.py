from transformers import Seq2SeqTrainer
from tqdm import tqdm
import torch
import numpy as np
import os
import json
import datetime

from whistress.training.self_training_utils import SelfPredictionDataset, convert_to_hf_dataset

os.environ["HF_HOME"] = "/sd1/jhansi/interns/abin/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/hub"

class WhiStressTrainer(Seq2SeqTrainer):
    """
    Custom trainer extending Seq2SeqTrainer for speech emphasis detection.
    
    Implements specialized training, evaluation, and model saving methods
    designed specifically for the emphasis detection model architecture.
    """

    def _pad_tensors_to_max_len(self, tensor, max_length):
        """
        Pad tensors to a specified maximum length using -100 as padding token.
        
        Args:
            tensor: Input tensor to pad
            max_length: Target length for padded tensor
            
        Returns:
            Padded tensor of shape (batch_size, max_length)
        """
        pad_token_id = -100

        # Create a padded tensor using the custom pad token
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )

        # Ensure that the tensor fits within the padded tensor up to the original tensor's length
        padded_tensor[:, : tensor.shape[-1]] = tensor

        return padded_tensor

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Execute a single training step with gradient clipping.
        
        Removes sentence indices from inputs before passing to parent class,
        then applies gradient clipping to prevent exploding gradients.
        
        Args:
            model: Model to train
            inputs: Dictionary of input tensors
            num_items_in_batch: Optional parameter specifying batch size
            
        Returns:
            Loss value for the training step
        """
        # Get number of samples in batch for metrics normalization
        num_items_in_batch = inputs["sentence_index"].shape[0]

        if self.state.global_step == 0:
            self.check_gradient_status(model)

        # Remove sentence_index from inputs as it's not needed for training
        sentence_index = inputs.pop("sentence_index")
        # Perform the default training step
        loss = super().training_step(model, inputs)

        # Clip gradients manually
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

        return loss

    def save_final_model(self, output_dir=None, training_args=None):
        """
        Save only the emphasis detection components of the model.
        
        Rather than saving the entire model, this method saves only:
        1. The classifier (head) used for emphasis detection
        2. The additional decoder block
        3. The selected layer passed to the head
        4. Training arguments for reproducibility
        
        Args:
            output_dir: Directory to save model components
            training_args: Training arguments to save
        """
        # save only the classifier and extra decoder layer
        classifier = (
            self.model.classifier if hasattr(self.model, "classifier") else None
        )
        additional_decoder_block = (
            self.model.additional_decoder_block
            if hasattr(self.model, "additional_decoder_block")
            else None
        )
        if output_dir is not None:
            torch.save(
                classifier.state_dict(), os.path.join(output_dir, "classifier.pt")
            )
            torch.save(
                additional_decoder_block.state_dict(),
                os.path.join(output_dir, "additional_decoder_block.pt"),
            )
            # save the layer passed to the head
            layer_for_head = self.model.layer_for_head
            with open(os.path.join(output_dir, "metadata.json"), "w") as file:
                json.dump({"layer_for_head": layer_for_head}, file)

            
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", dataset_name=''):
        """
        Evaluate model at token level on the evaluation dataset.
        
        Runs a forward pass through the model for each batch, collects predictions
        and labels, and calculates evaluation metrics. Operates at the token level,
        meaning each token's emphasis prediction is evaluated separately.
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in the model output
            metric_key_prefix: Prefix for metric keys in output
            dataset_name: Name of the dataset for logging purposes
            
        Returns:
            Dictionary of evaluation metrics
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(eval_dataloader):
            # Extract input features and labels
            input_features = batch["input_features"]
            labels_keys = [elem for i, elem in enumerate(batch.keys()) if "labels" in elem and not "labels_head" in elem][0]
            whisper_labels = batch[labels_keys]
            labels_head_keys = [elem for i, elem in enumerate(batch.keys()) if "labels_head" in elem][0]
            labels_head = batch[labels_head_keys]

            # Generate predictions by a forward pass through the model
            with torch.no_grad():
                # Run forward pass through the model to get predictions and loss
                outputs = self.model(
                    input_features=input_features,
                    labels_head=labels_head,
                    whisper_labels=whisper_labels
                )
                
                # Extract predictions and calculate loss
                generated_ids = outputs['preds']
                if outputs['loss'] is not None:
                    total_loss += outputs['loss'].item()
                    num_batches += 1
                
            # Pad predictions and labels to the same length for proper comparison
            # This ensures all tensors in the batch have consistent dimensions
            padded_preds = self._pad_tensors_to_max_len(
                generated_ids, max_length=self.args.generation_max_length
            )
            padded_labels = self._pad_tensors_to_max_len(
                labels_head, max_length=self.args.generation_max_length
            )
            
            all_preds.append(padded_preds.cpu().numpy())
            all_labels.append(padded_labels.cpu().numpy())

        # Concatenate all batches into single arrays for metric computation
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute evaluation metrics using the provided compute_metrics function
        outputs_metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(
                {"predictions": all_preds, "label_ids": all_labels}
            )
            for key, value in metrics.items():
                key = f"{metric_key_prefix}_{key}"
                if isinstance(value, np.ndarray):
                    outputs_metrics[key] = value.tolist()
                else:
                    outputs_metrics[key] = value
        
        # Add the average loss to metrics
        if num_batches > 0:
            outputs_metrics[f"{metric_key_prefix}_loss"] = total_loss / num_batches
                    
        with open(os.path.join(self.args.output_dir, "log_eval.txt"), "a") as file:
            json.dump(f'Evaluate at TOKEN LEVEL {dataset_name}:', file)
            json.dump(outputs_metrics, file)
        self.log(outputs_metrics)

        return outputs_metrics
    
    def evaluate_at_word_level(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", dataset_name=''):
        """
        Evaluate model at word level on the evaluation dataset.
        
        Similar to evaluate(), but aggregates token-level predictions to the word level
        before computing metrics. This provides a more meaningful evaluation for emphasis
        detection since emphasis typically applies to entire words, not individual tokens.
        
        A word is considered emphasized if any of its tokens are predicted as emphasized.
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in the model output
            metric_key_prefix: Prefix for metric keys in output
            dataset_name: Name of the dataset for logging purposes
            
        Returns:
            Dictionary of word-level evaluation metrics
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        all_preds_by_words = []
        all_labels_by_words = []

        for batch in tqdm(eval_dataloader):
            # Extract input features and labels
            input_features = batch["input_features"]
            labels_keys = [elem for i, elem in enumerate(batch.keys()) if "labels" in elem and not "labels_head" in elem][-1]
            whisper_labels = batch[labels_keys]
            labels_head_keys = [elem for i, elem in enumerate(batch.keys()) if "labels_head" in elem][-1]
            labels_head = batch[labels_head_keys]

            # Generate predictions by a forward pass through the model
            with torch.no_grad():
                generated_ids = self.model(
                    input_features=input_features,
                    labels_head=labels_head,
                    whisper_labels=whisper_labels,
                )['preds']
                
            all_labels_head_by_words = []
            all_generated_ids_by_words = []
            batch_samples = torch.where(torch.tensor(eval_dataset['sentence_index']).cpu() == batch['sentence_index'].unsqueeze(1).cpu())[1].numpy()
            map_dict_key = [elem for i, elem in enumerate(eval_dataset.column_names) if "map_dict" in elem][-1]
            for i in range(labels_head.shape[0]):
                j_start = 1
                labels_head_by_words = [-100]
                generated_ids_by_words = [-100]
                for val in eval_dataset[int(batch_samples[i])][map_dict_key]["values"]:
                    if len(val) == 0:
                        j_end += 1
                    else:
                        j_end = j_start + len(val)
                    while (not np.array_equal(whisper_labels[i][j_start:j_end].cpu().numpy(), val.numpy()) and \
                        not whisper_labels[i][j_end].item() == 50256 and not len(val) == 0):
                        # if we ran into tokens which aren't part of a word (like ',', '.', '\n'), we skip them/treat them as a word
                        # the second condition in the while loop is meant to prevent crossing the end of sequence
                        if 1 in labels_head[i][j_start]:
                            labels_head_by_words.append(1)
                        else:
                            labels_head_by_words.append(0)
                        if 1 in generated_ids[i][j_start]:
                            generated_ids_by_words.append(1)
                        else:
                            generated_ids_by_words.append(0)
                        j_start += 1
                        j_end += 1
                    if 1 in labels_head[i][j_start:j_end]:
                        labels_head_by_words.append(1)
                    else:
                        labels_head_by_words.append(0)
                    if 1 in generated_ids[i][j_start:j_end]:
                        generated_ids_by_words.append(1)
                    else:
                        generated_ids_by_words.append(0)
                    j_start = j_end
                # add the last punctuation mark if it's not the end of the sequence
                if whisper_labels[i][j_end].item() != 50256: # 50256 relates to the choice of the backbone as whisper's english model
                    if 1 in labels_head[i][j_end]:
                        labels_head_by_words.append(1)
                    else:
                        labels_head_by_words.append(0)
                    if 1 in generated_ids[i][j_end]:
                        generated_ids_by_words.append(1)
                    else:
                        generated_ids_by_words.append(0)
                    j_end += 1
                assert labels_head[i][j_end]==-100
                    
                labels_head_by_words_padded = self._pad_tensors_to_max_len(
                    torch.tensor(labels_head_by_words).unsqueeze(0), max_length=self.args.generation_max_length
                )
                generated_ids_by_words_padded = self._pad_tensors_to_max_len(
                    torch.tensor(generated_ids_by_words).unsqueeze(0), max_length=self.args.generation_max_length
                )                
                all_generated_ids_by_words.append(generated_ids_by_words_padded.squeeze(0))
                all_labels_head_by_words.append(labels_head_by_words_padded.squeeze(0))
                
            padded_labels_by_words = torch.stack(all_labels_head_by_words)
            padded_preds_by_words = torch.stack(all_generated_ids_by_words)
            
            all_preds_by_words.append(padded_preds_by_words.cpu().numpy())
            all_labels_by_words.append(padded_labels_by_words.cpu().numpy())

        # Flatten lists        
        all_preds_by_words = np.concatenate(all_preds_by_words, axis=0)
        all_labels_by_words = np.concatenate(all_labels_by_words, axis=0)

        # Compute metrics
        outputs_metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(
                {"predictions": all_preds_by_words, "label_ids": all_labels_by_words}
            )
            for key, value in metrics.items():
                key = f"{metric_key_prefix}_{key}"
                if isinstance(value, np.ndarray):
                    outputs_metrics[key] = value.tolist()
                else:
                    outputs_metrics[key] = value
                    
        with open(os.path.join(self.args.output_dir, "log_eval_word_level.txt"), "a") as file:
            json.dump(f'Evaluate at WORD LEVEL {dataset_name}:', file)
            json.dump(outputs_metrics, file)
        self.log(outputs_metrics)
        return outputs_metrics

    def save_metrics_to_file(self, metrics_data, filename, output_dir=None):
        """
        Save training and evaluation metrics to a file.
        
        Args:
            metrics_data: Dictionary or list of metrics to save
            filename: Name of the file to save metrics to
            output_dir: Directory to save the file in (defaults to self.args.output_dir)
        """
        if output_dir is None:
            output_dir = self.args.output_dir
            
        filepath = os.path.join(output_dir, filename)
        
        # Append to file if it exists, create it otherwise
        with open(filepath, 'a') as f:
            f.write(json.dumps(metrics_data, indent=2) + '\n')
            
        print(f"Metrics saved to {filepath}")

    def align_samples_aux(self, pred):
        """
        Identify samples where predictions and labels have mismatched lengths.
        
        Used to filter out problematic samples where the model's predictions
        cannot be directly compared to ground truth labels due to length mismatch.
        
        Args:
            pred: Dictionary containing 'predictions' and 'label_ids' arrays
            
        Returns:
            List of row indices to remove from evaluation
        """
        pred_ids = pred["predictions"]
        label_ids = pred["label_ids"]
        pad_token_id = -100

        rows_to_remove = []
        for i, (pred_id, label_id) in enumerate(zip(pred_ids, label_ids)):
            # Create a mask where pred_ids are not equal to pad_token_id
            mask_pred_ids = pred_id != pad_token_id
            # Create a mask where label_ids are not equal to pad_token_id
            mask_label_ids = label_id != pad_token_id
            if pred_id[mask_pred_ids].shape[0] != label_id[mask_label_ids].shape[0]:
                rows_to_remove.append(i)

        return rows_to_remove
    
    def aligned_whisper_transcriptions(self, example):
        """
        Generate Whisper transcriptions and check alignment with ground truth.
        
        Used during dataset preprocessing to identify samples where the Whisper model's
        transcription matches the ground truth transcription, ignoring formatting
        differences like capitalization and punctuation.
        
        Args:
            example: Dataset example containing audio and transcription
            
        Returns:
            Example with added 'aligned_whisper_transcriptions' field
        """
        # Filter out samples with '\n' in the transcription
        token_ids = self.model.whisper_model.generate(input_features=example['input_features'].to('cuda').unsqueeze(0), 
                                                    labels=example['whisper_labels'].to('cuda').unsqueeze(0))
        transcription = self.model.processor.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        example['aligned_whisper_transcriptions'] = ''
        if transcription.lstrip().lower().replace(',','').replace('.','') == example['transcription'].lower():
            example['aligned_whisper_transcriptions'] = transcription
        return example
    
    def filter_misaligned_samples(self, example):
        """
        Filter out examples where Whisper transcription doesn't align with ground truth.
        
        Args:
            example: Dataset example containing aligned_whisper_transcriptions
            
        Returns:
            Boolean indicating whether the example should be kept (True) or filtered out (False)
        """
        return example['aligned_whisper_transcriptions'] != ''

    def align_samples(self, dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Process dataset to identify and flag misaligned samples.
        
        Generates model predictions for each example in the dataset, and identifies
        examples where the prediction length doesn't match the label length, which
        would cause evaluation errors.
        
        Args:
            dataset: Dataset to check for alignment issues
            ignore_keys: Keys to ignore in the model output
            metric_key_prefix: Prefix for metric keys in output
            
        Returns:
            List of indices for samples that should be removed due to alignment issues
        """
        eval_dataloader = self.get_eval_dataloader(dataset)
        self.model.eval()

        all_preds = []
        all_labels = []

        for i, batch in enumerate(tqdm(eval_dataloader)):
            # Extract input features and labels
            input_features = batch["input_features"]
            whisper_labels = batch["whisper_labels"]
            labels_head = batch["labels_head"]

            # Generate predictions
            with torch.no_grad():
                # Adjust inputs according to your model's requirements
                generated_ids = self.model.generate(
                    input_features=input_features,
                    whisper_labels=whisper_labels,
                )

            # Pad or truncate predictions and labels to a fixed length
            padded_preds = self._pad_tensors_to_max_len(
                generated_ids, max_length=self.args.generation_max_length
            )
            padded_labels = self._pad_tensors_to_max_len(
                labels_head, max_length=self.args.generation_max_length
            )
            # Collect predictions and labels
            all_preds.append(padded_preds.cpu().numpy())
            all_labels.append(padded_labels.cpu().numpy())

        # Flatten lists
        for i in range(len(all_preds)):
            print(f"{all_preds[i].shape=}, {all_labels[i].shape=}")
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return self.align_samples_aux(
            {"predictions": all_preds, "label_ids": all_labels}
        )
    
    def check_gradient_status(self, model):
        """Debug method to check which parameters have gradients enabled."""
        
        print("\n--- Gradient Status ---")
        print("Encoder parameters:")
        enc_grads = [p.requires_grad for p in model.whisper_model.get_encoder().parameters()]
        print(f"  Trainable: {sum(enc_grads)}/{len(enc_grads)}")
    
        print("Decoder parameters:")
        dec_grads = [p.requires_grad for p in model.whisper_model.get_decoder().parameters()]
        print(f"  Trainable: {sum(dec_grads)}/{len(dec_grads)}")
    
        print("Additional decoder block:")
        add_dec_grads = [p.requires_grad for p in model.additional_decoder_block.parameters()]
        print(f"  Trainable: {sum(add_dec_grads)}/{len(add_dec_grads)}")
    
        print("Classifier:")
        cls_grads = [p.requires_grad for p in model.classifier.parameters()]
        print(f"  Trainable: {sum(cls_grads)}/{len(cls_grads)}")
        print("---------------------\n")
    
    def train_with_self_prediction(self, resume_from_checkpoint=None):
        """
        Custom training method that runs with cycles of 2 epochs each.
        
        Cycle 1 (epochs 1-2): Train on ground truth/noisy labels
        Subsequent cycles: Train for 2 epochs on model's predictions from the previous cycle
        
        Args:
            resume_from_checkpoint: Optional path to a checkpoint to resume training from
            
        Returns:
            TrainOutput with training metrics and state
        """
        args = self.args
        self._train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        
        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)
        
        self.model.train()
        
        original_epochs = self.args.num_train_epochs
        original_eval_strategy = self.args.eval_strategy

        # Save the original dataset for restoration at the end if needed
        original_dataset = self.train_dataset
        
        best_metrics = {}
        best_cycle = 0
        
        # Early stopping parameters
        patience = 3
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False
        
        # Track metrics for each cycle
        cycle_metrics = []
        self.cycle_metrics = cycle_metrics
        all_metrics = []
        
        # Create a metrics file with header
        self.save_metrics_to_file(
            {"timestamp": str(datetime.datetime.now()), "note": "Training started - metrics per cycle"}, 
            "training_metrics.jsonl"
        )
        
        # Iterative training
        total_epochs = int(original_epochs)
        num_cycles = total_epochs // 2  # Each cycle is 2 epochs
        print(f"\n*** Starting iterative self-training process for {total_epochs} epochs ({num_cycles} cycles) ***\n")
        
        # First cycle (epochs 1-2): Train on original noisy labels
        print(f"\n*** Cycle 1: Epochs 1-2 - Training on original noisy labels ***\n")
        
        # Set to train for just two epochs
        self.args.num_train_epochs = 2
        self.state.epoch = 0
        
        # Train for first cycle (2 epochs)
        train_output = self.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Get training loss from training output
        train_loss = train_output.training_loss
        print(f"Training Loss after cycle 1: {train_loss}")
        
        # Evaluate after first cycle if validation data is available
        if self.eval_dataset is not None:
            print("\n*** Performing validation after cycle 1 (epochs 1-2) ***\n")
            
            # Properly access validation loss - evaluate directly to get current loss
            eval_metrics = self.evaluate(eval_dataset=self.eval_dataset, metric_key_prefix="eval_cycle1")
            
            # Access loss directly from the returned metrics dictionary
            val_loss = eval_metrics.get("eval_cycle1_loss", "N/A")
            print(f"Validation Loss after cycle 1: {val_loss}")
            
            # Get F1 score from evaluation metrics
            f1_score = eval_metrics.get("eval_cycle1_f1", 0)
            print(f"Validation F1 Score after cycle 1: {f1_score:.5f}")
            
            # Save metrics for this cycle
            cycle_data = {
                "cycle": 1,
                "epochs": "1-2",
                "train_loss": train_loss,
                "val_loss": val_loss,
                "f1_score": f1_score
            }
            
            cycle_metrics.append(cycle_data)
            all_metrics.append(cycle_data)
            
            # Save metrics to file
            self.save_metrics_to_file(cycle_data, "training_metrics.jsonl")
            
            if not best_metrics or f1_score > best_metrics.get("f1", 0):
                best_metrics["f1"] = f1_score
                best_metrics["val_loss"] = val_loss
                best_metrics["train_loss"] = train_loss
                best_cycle = 1
        
        # Save model after first cycle
        cycle_dir = self.save_model_checkpoint(cycle_num=1)
        
        # Current dataset for generating predictions
        current_dataset = original_dataset
        
        for cycle in range(1, num_cycles):

            if early_stopped:
                print(f"🛑 Training stopped early after {cycle} cycles due to no improvement in validation loss.")
                break

            # Generate predictions for next cycle
            print(f"\n*** Generating predictions for cycle {cycle+1} (epochs {cycle*2+1}-{cycle*2+2}) ***\n")
            
            # Create a dataset with the model's own predictions
            self_prediction_dataset = SelfPredictionDataset.create_from_model_predictions(
                model=self.model,
                original_dataset=current_dataset,  # Use previous cycle's dataset structure
                data_collator=self.data_collator,
                device=self.args.device
            )
            
            self_prediction_hf_dataset = convert_to_hf_dataset(self_prediction_dataset)
            
            self.train_dataset = self_prediction_hf_dataset
            
            # Reset optimizer state for next cycle
            self.create_optimizer_and_scheduler(
                num_training_steps=int(len(self.train_dataset) / self._train_batch_size) * 2  # For 2 epochs
            )
            
            self.state.epoch = cycle * 2
            
            print(f"\n*** Cycle {cycle+1}: Epochs {cycle*2+1}-{cycle*2+2} - Training on predictions from cycle {cycle} ***\n")
            
            # Train for current cycle (2 epochs)
            train_output = self.train()
            
            train_loss = train_output.training_loss
            print(f"Training Loss after cycle {cycle+1}: {train_loss}")
            
            if self.eval_dataset is not None:
                print(f"\n*** Performing validation after cycle {cycle+1} (epochs {cycle*2+1}-{cycle*2+2}) ***\n")
                
                eval_metrics = self.evaluate(
                    eval_dataset=self.eval_dataset, 
                    metric_key_prefix=f"eval_cycle{cycle+1}"
                )
                
                val_loss = eval_metrics.get(f"eval_cycle{cycle+1}_loss", "N/A")
                print(f"Validation Loss after cycle {cycle+1}: {val_loss}")
                
                f1_score = eval_metrics.get(f"eval_cycle{cycle+1}_f1", 0)
                print(f"Validation F1 Score after cycle {cycle+1}: {f1_score:.5f}")
                
                cycle_data = {
                    "cycle": cycle+1,
                    "epochs": f"{cycle*2+1}-{cycle*2+2}",
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "f1_score": f1_score
                }
                
                cycle_metrics.append(cycle_data)
                all_metrics.append(cycle_data)
                
                # Save metrics to file
                self.save_metrics_to_file(cycle_data, "training_metrics.jsonl")
                
                # Early stopping check
                current_val_loss = val_loss if isinstance(val_loss, (int, float)) else float('inf')
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    print(f"✅ Validation loss improved to {current_val_loss:.6f}")
                else:
                    patience_counter += 1
                    print(f"⚠️ Validation loss did not improve. Patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        print(f"🛑 Early stopping triggered! No improvement for {patience} consecutive cycles.")
                        early_stopped = True

                if f1_score > best_metrics.get("f1", 0):
                    best_metrics["f1"] = f1_score
                    best_metrics["val_loss"] = val_loss
                    best_metrics["train_loss"] = train_loss
                    best_cycle = cycle + 1
            
            # Save model after current cycle
            cycle_dir = self.save_model_checkpoint(cycle_num=cycle+1)

            # Update current dataset for next iteration
            current_dataset = self.train_dataset
        
        # Print summary of training
        print("\n*** Iterative self-training completed ***")
        if early_stopped:
            print(f"🛑 Training stopped early due to no improvement in validation loss for {patience} consecutive cycles.")
        print(f"Best performance: F1={best_metrics.get('f1', 0)} at cycle {best_cycle}")
        if best_metrics.get('val_loss') != 'N/A':
            print(f"Best validation loss: {best_metrics.get('val_loss')}")
            
        # Print all validation metrics for each cycle
        print("\n*** Validation metrics for each cycle ***")
        print("-" * 80)
        print("| {:^5} | {:^8} | {:^20} | {:^20} | {:^20} |".format(
            "Cycle", "Epochs", "Train Loss", "Validation Loss", "F1 Score"
        ))
        print("-" * 80)
        for metrics in cycle_metrics:
            cycle_num = metrics["cycle"]
            epochs = metrics["epochs"]
            train_loss = metrics["train_loss"]
            val_loss = metrics["val_loss"]
            f1 = metrics["f1_score"]
            
            print("| {:^5} | {:^8} | {:^20} | {:^20} | {:^20.5f} |".format(
                cycle_num,
                epochs,
                str(train_loss),
                str(val_loss),
                f1
            ))
            # Mark the best cycle
            if cycle_num == best_cycle:
                print("| {:^5} | {:^8} | {:^20} | {:^20} | {:^20} |".format("", "", "", "", "^ BEST"))
        print("-" * 80)
        
        # Final summary 
        summary = {
            "timestamp": str(datetime.datetime.now()),
            "best_cycle": best_cycle,
            "best_f1": best_metrics.get("f1", 0),
            "best_val_loss": best_metrics.get("val_loss", "N/A"),
            "best_train_loss": best_metrics.get("train_loss", "N/A"),
            "all_metrics": all_metrics
        }
        
        self.save_metrics_to_file(summary, "training_summary.json")
                
        best_cycle_dir = os.path.join(self.args.output_dir, f"cycle_{best_cycle}")
        best_model_dir = os.path.join(self.args.output_dir, "best_model")
        
        if os.path.exists(best_model_dir):
            import shutil
            shutil.rmtree(best_model_dir)
        
        os.makedirs(best_model_dir, exist_ok=True)
        for file in ["classifier.pt", "additional_decoder_block.pt", "metadata.json", "metrics.json"]:
            src_file = os.path.join(best_cycle_dir, file)
            if os.path.exists(src_file):
                import shutil
                shutil.copy2(src_file, os.path.join(best_model_dir, file))
        
        print(f"\nBest model from cycle {best_cycle} copied to {best_model_dir}")
        
        # Restore original settings
        self.args.num_train_epochs = original_epochs
        self.args.eval_strategy = original_eval_strategy

        return self.state

            
    def save_model_checkpoint(self, cycle_num, output_dir=None):
        """
        Save model components for a specific training cycle.
        
        Args:
            cycle_num: Cycle number to include in the folder name
            output_dir: Base directory for saving checkpoints
        """
        if output_dir is None:
            output_dir = self.args.output_dir
            
        # Create cycle-specific directory
        cycle_dir = os.path.join(output_dir, f"cycle_{cycle_num}")
        os.makedirs(cycle_dir, exist_ok=True)
        
        # Save classifier state
        classifier = self.model.classifier if hasattr(self.model, "classifier") else None
        if classifier:
            torch.save(
                classifier.state_dict(), 
                os.path.join(cycle_dir, "classifier.pt")
            )
        
        # Save additional decoder block state
        additional_decoder_block = self.model.additional_decoder_block if hasattr(self.model, "additional_decoder_block") else None
        if additional_decoder_block:
            torch.save(
                additional_decoder_block.state_dict(),
                os.path.join(cycle_dir, "additional_decoder_block.pt")
            )
        
        # Save metadata
        if hasattr(self.model, "layer_for_head"):
            with open(os.path.join(cycle_dir, "metadata.json"), "w") as file:
                json.dump({"layer_for_head": self.model.layer_for_head}, file)
        
        # Save metrics for this cycle if available
        if hasattr(self, "cycle_metrics") and len(self.cycle_metrics) >= cycle_num:
            with open(os.path.join(cycle_dir, "metrics.json"), "w") as file:
                json.dump(self.cycle_metrics[cycle_num-1], file, indent=2)
                
        print(f"Model checkpoint saved for cycle {cycle_num} to {cycle_dir}")
        return cycle_dir