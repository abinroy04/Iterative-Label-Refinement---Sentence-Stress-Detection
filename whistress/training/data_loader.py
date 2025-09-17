from whistress.training.processor import DSProcessor
from datasets import load_from_disk
import os

os.environ["HF_HOME"] = "/sd1/jhansi/interns/abin/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/hub"

class PreprocessedDataLoader():
    """
    Generic data loading class for speech emphasis detection datasets.
    
    Handles dataset preprocessing, loading from disk or HuggingFace,
    adding necessary column indices, and preparing datasets for training.
    
    Attributes:
        preprocessed_dataset_path: Root directory for preprocessed dataset local storage
        model_with_emphasis_head: Model with emphasis detection capability
        hf_token: HuggingFace API token for accessing datasets
        ds_hf_train: HuggingFace dataset name for training (if applicable and the data is also used for training the emphasis detection head)
        ds_hf_eval: HuggingFace dataset name for evaluation (if applicable and the data is also used for evaluation of the emphasis detection head)
        emphasis_indices_column_name: Column name for emphasis labels
        columns_to_remove: Columns to exclude from the dataset
        split_train_val_percentage: Percentage of data to use for validation
        max_samples: Maximum number of samples to load (None for all samples)
    """
    
    def __init__(self, 
                preprocessed_dataset_path, 
                columns_to_remove, 
                model_with_emphasis_head, 
                hf_token=None,
                ds_hf_train=None, 
                ds_hf_eval=None,
                emphasis_indices_column_name="emphasis_indices", 
                transcription_column_name='transcription', 
                split_train_val_percentage=0.2,
                max_samples=None
            ):
        
        self.preprocessed_dataset_path = preprocessed_dataset_path
        self.model_with_emphasis_head = model_with_emphasis_head
        self.hf_token = hf_token
        self.ds_hf_train = ds_hf_train
        self.ds_hf_eval = ds_hf_eval
        self.emphasis_indices_column_name = emphasis_indices_column_name
        self.columns_to_remove = columns_to_remove
        self.transcription_column_name = transcription_column_name
        self.split_train_val_percentage = split_train_val_percentage
        self.max_samples = max_samples
        self.dataset = self.load_preproc_datasets(model_with_emphasis_head, 
                                                preprocessed_dataset_path, 
                                                columns_to_remove,
                                                emphasis_indices_column_name, 
                                                transcription_column_name, 
                                                ds_hf_train, 
                                                hf_token,
                                                max_samples)

    def load_preproc_datasets(self, 
                            model_with_emphasis_head, 
                            preprocessed_dataset_path, 
                            columns_to_remove,
                            emphasis_indices_column_name, 
                            transcription_column_name,
                            ds_name_hf, 
                            hf_token,
                            max_samples=None):
        """
        Load and preprocess datasets from disk or HuggingFace.
        
        If the dataset exists on disk, loads it directly. Otherwise, downloads and
        processes it using the DSProcessor, then saves it to disk for future use.
        Also adds sentence indices and performs necessary column transformations.
        
        Args:
            model_with_emphasis_head: Model with emphasis detection capability
            columns_to_remove: Columns to exclude from the dataset
            emphasis_indices_column_name: Column name for emphasis labels
            transcription_column_name: Column name for transcription text
            ds_name_hf: HuggingFace dataset name
            hf_token: HuggingFace API token
            max_samples: Maximum number of samples to load (None for all samples)
            
        Returns:
            Processed dataset with unnecessary columns removed
        """
        def change_input_features(example):
            example['input_features'] = example['input_features'][0]
            return example
        def add_sentence_index(row, index_container):
            curr_index = index_container['sentence_index']
            row['sentence_index'] = curr_index
            index_container["sentence_index"] += 1
            return row
        
        if os.path.exists(preprocessed_dataset_path):
            train_set = load_from_disk(preprocessed_dataset_path)
            # If max_samples is specified, select only a subset
            if max_samples is not None:
                print(f"Loading only the first {max_samples} samples from the dataset")
                for split in train_set:
                    train_set[split] = train_set[split].select(range(min(max_samples, len(train_set[split]))))
            return train_set.remove_columns(columns_to_remove)
        else:
            ds_preprocessor = DSProcessor(
                ds_name=ds_name_hf,
                processor=model_with_emphasis_head.processor,
                hyperparameters={"split_train_val_percentage": self.split_train_val_percentage},
                hf_token=hf_token,
                max_samples=max_samples
            )
            train_set = ds_preprocessor.get_train_dataset(emphasis_indices_column_name=emphasis_indices_column_name, 
                                                            transcription_column_name=transcription_column_name,
                                                            model=model_with_emphasis_head,
                                                            columns_to_remove=None)
            index_container = {"sentence_index": 0}
            train_set = train_set.map(add_sentence_index, num_proc=1, load_from_cache_file=False, fn_kwargs={'index_container': index_container})
            train_set = train_set.map(change_input_features, load_from_cache_file=False, num_proc=1)
            if "labels" in train_set['train'].column_names:
                train_set = train_set.rename_column("labels", f"labels_{transcription_column_name}")
            train_set.save_to_disk(os.path.join(preprocessed_dataset_path))            
            return train_set.remove_columns(columns_to_remove)
    
    def split_train_val(self):
        """
        Split dataset into training and validation sets.
        
        Args:
            rows_to_remove: Optional list of row indices to exclude
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.split_train_val_percentage == 0.0:
            return self.dataset, None
        dataset_split = self.dataset["train"].train_test_split(
            test_size=self.split_train_val_percentage,
            shuffle=True,
            seed=42,
        )
        return dataset_split["train"], dataset_split["test"]
 

class PreprocessedTinyStress15KLoader(PreprocessedDataLoader):
    """
    Data loader for synthetic GPT-generated data.
    
    This dataset contains TTS-generated speech from GPT-written stories with
    specifically marked emphasis. This synthetic data helps supplement real
    datasets for training emphasis detection models.
    """
    def __init__(self, model_with_emphasis_head, transcription_column_name, save_path, max_samples=None):
        # preprocessed_dataset_path = save_path # modify this path, if exists
        ds_hf_train = "slprl/TinyStress-15K" # train and val set
        columns_to_remove = ['id', 'original_sample_index', 'ssml', 'emphasis_indices', 'metadata', 'word_start_timestamps', 'audio']
        super().__init__(preprocessed_dataset_path=save_path, 
                        columns_to_remove=columns_to_remove,
                        model_with_emphasis_head=model_with_emphasis_head, 
                        emphasis_indices_column_name='emphasis_indices',
                        transcription_column_name=transcription_column_name, 
                        ds_hf_train=ds_hf_train,
                        max_samples=max_samples)

    def split_train_val_test(self):
        """
        Split synthetic GPT dataset into train, validation, and test sets.
        
        Uses predefined splits from the HuggingFace dataset, with an optional
        further split of the training set for validation.
        
        Returns:
            train_set, eval_set, test_set
        """
        if self.split_train_val_percentage == 0.0:
            return self.dataset["train"], None, self.dataset["test"]
        train_set, eval_set = super().split_train_val()
        return train_set, eval_set, self.dataset["test"]
    
class PreprocessedITAStressLoader(PreprocessedDataLoader):
    """
    Data loader for the ITA dataset (both word-stress and threshold variants).
    """
    def __init__(self, model_with_emphasis_head, transcription_column_name, save_path=None, max_samples=None, dataset_name=None):
        # Columns that are not needed for training or evaluation
        columns_to_remove = []
        
        dataset_specific_save_path = os.path.join(save_path, dataset_name.replace("/", "_"))
        
        # Initialize the parent class
        super().__init__(
            preprocessed_dataset_path=dataset_specific_save_path,
            columns_to_remove=columns_to_remove,
            model_with_emphasis_head=model_with_emphasis_head,
            emphasis_indices_column_name="emphasis_indices",
            transcription_column_name="transcription",
            ds_hf_train=dataset_name,
            hf_token="", # Set token for private datasets
            max_samples=max_samples
        )

    def split_train_val_test(self):
        """
        Split ITA-word-stress dataset into train, validation, and test sets.
        
        Returns:
            train_set, eval_set, test_set
        """
        if self.split_train_val_percentage == 0.0:
            return self.dataset["train"], None, self.dataset["test"]
        train_set, eval_set = self.split_train_val()
        return train_set, eval_set, self.dataset["test"]

def load_data(model_with_emphasis_head, transcription_column_name, dataset_name, save_path=None, max_samples=None):
    """
    Factory function to create the appropriate dataset loader.
    
    *Add here any new datasets you want to support.*
    
    Args:
        model_with_emphasis_head: Model with emphasis detection capability
        transcription_column_name: Column name for transcription text
        dataset_name: Name of the dataset to load (e.g., "tinyStress-15K")
        save_path: Path to save or load the preprocessed dataset
        max_samples: Maximum number of samples to load (None for all samples)
        
    Returns:
        Instantiated dataset loader for the specified dataset
        
    Raises:
        ValueError: If the requested dataset is not supported
    """
    dataset = None

    if dataset_name in [
            "abinroy04/ITA-word-stress", 
            "abinroy04/ITA-kmeans", 
            "abinroy04/ITA-tcssc-18", 
            "abinroy04/ITA-threshold", 
            "abinroy04/ITA-GMM"
        ]:
        dataset = PreprocessedITAStressLoader(
            model_with_emphasis_head, 
            transcription_column_name, 
            save_path=save_path, 
            max_samples=max_samples,
            dataset_name=dataset_name
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not defined in data_loader.py")

    return dataset
