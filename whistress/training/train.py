import torch
import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from transformers import WhisperConfig, Seq2SeqTrainingArguments, TrainerCallback, set_seed

os.environ["HF_HOME"] = "/sd1/jhansi/interns/abin/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/hub"
os.environ["WAND_API_KEY"] = "" # Set Key

import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from whistress.model.model import WhiStress
from whistress.training.data_loader import load_data
from whistress.training.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from whistress.training.trainer import WhiStressTrainer
from whistress.training.metrics import WhiStressMetrics

CURRENT_DIR = Path(__file__).parent
WANDB_API_KEY = os.environ.get("WAND_API_KEY", None)

class CustomCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if hasattr(state, "metrics") and isinstance(state.metrics, np.ndarray):
            state.metrics = state.metrics.tolist()

def train_or_evaluate(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    whisper_backbone_name = f"openai/whisper-small.en"
    whisper_config = WhisperConfig()
    layer_for_head = 9
    
    if args["model_path"]:
        logger.info(f"Loading model from {args['model_path']}")
        whistress_model = WhiStress(
            whisper_config, layer_for_head=layer_for_head, whisper_backbone_name=whisper_backbone_name,
            use_auth_token="" # Token needed for private datasets
        ).to(device)
        whistress_model.load_model(args["model_path"], device=device)
        whistress_model.to(device)
        whistress_model.eval()
    else:
        logger.info("Training a new model from scratch")
        whistress_model = WhiStress(
            whisper_config, layer_for_head=layer_for_head, whisper_backbone_name=whisper_backbone_name,
            use_auth_token="" # Token needed for private datasets
        ).to(device)
    

    whistress_model.processor.tokenizer.model_input_names = [
        "input_ids",
        "attention_mask",
        "labels_head",
    ]
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=whistress_model.processor,
        decoder_start_token_id=whistress_model.whisper_model.config.decoder_start_token_id,
        forced_decoder_ids=whistress_model.whisper_model.config.forced_decoder_ids[
            0
        ][1],
        eos_token_id=whistress_model.whisper_model.config.eos_token_id,
        transcription_column_name=args["transcription_column_name"]
    )

    train, val = None, None
    if args["is_train"]:
        logger.info(f"Loading training dataset: {args['dataset_train']}")
        DatasetTrain = load_data(
            whistress_model, 
            args["transcription_column_name"], 
            dataset_name=args["dataset_train"], 
            save_path=args["dataset_path"],
            max_samples=args["max_samples"]
        )
        train, val, _ = DatasetTrain.split_train_val_test()
        if args["max_samples"]:
            logger.info(f"Using subset of data: max_samples={args['max_samples']}")
        logger.info(f"Training set size: {len(train)}, Validation set size: {len(val)}")

    logger.info(f"Loading evaluation dataset: {args['dataset_eval']}")
    DatasetEval = load_data(
        whistress_model, 
        args["transcription_column_name"], 
        dataset_name=args["dataset_eval"], 
        save_path=args["dataset_path"],
        max_samples=args["max_samples"] if args["is_train"] else None
    )
    _, _, test = DatasetEval.split_train_val_test()
    logger.info(f"Test set size: {len(test)}")

    print(f"Output path for the training run: {args['output_path']}")
    output_path = args['output_path']

    if WANDB_API_KEY and args["is_train"]:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project="whistress",
            name=f"{args['dataset_train']}_{args['dataset_eval']}",
            config={
                "dataset_train": args['dataset_train'],
                "dataset_eval": args['dataset_eval'],
                "transcription_column_name": args['transcription_column_name'],
                "validation_split": 0.2
            },
            dir=output_path,
        )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=24, # assuming 1 gpu. 
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=5e-4,
        warmup_ratio=0.05,
        num_train_epochs=4,
        seed=42,
        gradient_checkpointing=False,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        per_device_eval_batch_size=32,
        generation_max_length=96,
        save_steps=5000,
        eval_steps=100,
        logging_steps=50,
        weight_decay=0.01,
        push_to_hub=False,
        report_to=['wandb'] if WANDB_API_KEY is not None else None,
        label_names=[f"labels_head_{args['transcription_column_name']}", "sentence_index", f"labels_{args['transcription_column_name']}"],
        overwrite_output_dir=True,
    )
    # Set seed before initializing model.
    set_seed(training_args.seed)
    metrics = WhiStressMetrics()

    trainer_emphasis = None
    if args["is_train"]:
        trainer_emphasis = WhiStressTrainer(
        args=training_args,
        model=whistress_model,
        train_dataset=train,
        eval_dataset=val,
        data_collator=data_collator,
        compute_metrics=metrics.compute_metrics,
        )
        
        # Initial evaluation on test set
        logger.info("Performing initial evaluation on test set...")
        trainer_emphasis.evaluate_at_word_level(
            ignore_keys=["whisper_logits"],
            eval_dataset=test,
            dataset_name=f"{args['dataset_eval']}-initial-word_level",
        )
        trainer_emphasis.evaluate(
            ignore_keys=["whisper_logits"],
            eval_dataset=test,
            dataset_name=f"{args['dataset_eval']}-initial",
        )
        
        logger.info("Starting training with cycles of 2 epochs with iterative self-prediction...")
        trainer_emphasis.train_with_self_prediction()
            
        best_model_dir = os.path.join(args['output_path'], "best_model")
        
        if os.path.exists(best_model_dir):
            logger.info(f"Using best model from {best_model_dir} for final evaluation")
            # Load the best model
            whistress_model.load_model(best_model_dir, device=device)
            whistress_model.to(device)
            # Update trainer with best model
            trainer_emphasis.model = whistress_model
    else:
        # change the trainer for evaluation only
        trainer_emphasis = WhiStressTrainer(
            args=training_args,
            model=whistress_model,
            train_dataset=test, # we don't really use it, but the trainer requires it
            eval_dataset=test, # we don't really use it, but the trainer requires it
            data_collator=data_collator,
            compute_metrics=metrics.compute_metrics,
            tokenizer=whistress_model.processor.feature_extractor,
        )

    # Final evaluation on test set
    logger.info("Performing final evaluation on test set...")
    trainer_emphasis.evaluate_at_word_level(
        # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
        ignore_keys=["whisper_logits"],
        eval_dataset=test,
        dataset_name=f"{args['dataset_eval']}-final-word_level",
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# Main function to execute the training
if __name__ == "__main__":
    args = {
        "model_path": None,
        "dataset_path": "./preprocessed_ita_gmm_data",
        "output_path": "./Whistress_gmm",          
        "transcription_column_name": "transcription",
        "dataset_train": "abinroy04/ITA-GMM",         
        "dataset_eval": "abinroy04/ITA-GMM",
        "is_train": True,
        "max_samples": None
    }
    
    if not os.path.exists(args["output_path"]):
        print(f"Creating output directory: {args['output_path']}")
        os.makedirs(args["output_path"], exist_ok=True)
    
    if args["is_train"]:
        assert args["model_path"] is None, "If training, a model path must not be provided"
    else:
        assert args["model_path"] is not None, "If not training, a model path must be provided"
    
    # Run training or evaluation
    train_or_evaluate(args)