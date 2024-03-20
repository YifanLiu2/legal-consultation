import argparse, os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


class PostsDataset(Dataset):
    """
    A PyTorch Dataset class that loads posts data from a CSV file.
    """
    def __init__(self, file_path, tokenizer, label_col):
        self.tokenizer = tokenizer
        dataset = pd.read_csv(file_path)

        posts = dataset["CleanedText"]
        if label_col not in dataset.columns:
            raise ValueError(f"'{label_col}' not found in dataset columns. Available columns are: {list(dataset.columns)}")
        labels = dataset[label_col]
        self.num_labels = len(labels.unique())

        self.encodings = self.tokenizer(posts.tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
        self.labels = torch.tensor(labels.values.astype(int).tolist(), dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
     

class QuestionDataset(Dataset):
    """
    A PyTorch Dataset class that loads text data grouped by questions from a CSV file.
    """
    def __init__(self, file_path, tokenizer, label_col):
        self.tokenizer = tokenizer
        dataset = pd.read_csv(file_path)

        if label_col not in dataset.columns:
            raise ValueError(f"'{label_col}' not found in dataset columns. Available columns are: {list(dataset.columns)}")
        labels = dataset[label_col]
        self.num_labels = len(labels.unique())
        
        # concatenate all posts for the given question with [SEP] tokens
        questions_data = dataset.groupby("QuestionUno")['CleanedText'].apply(lambda texts: tokenizer.sep_token.join(texts.astype(str))).reset_index()
        questions_data = questions_data.merge(dataset[["QuestionUno", label_col]].drop_duplicates(), on="QuestionUno", how='left')

        # convert categorical labels to integer
        unique_labels = questions_data[label_col].unique()
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        questions_data[label_col] = questions_data[label_col].map(label_mapping)

        questions = questions_data["CleanedText"]
        labels = questions_data[label_col]
        
        self.encodings = self.tokenizer(questions.to_list(), padding=True, truncation=True, max_length=512, return_tensors="pt")
        self.labels = torch.tensor(labels.values.astype(int).tolist(), dtype=torch.long)        

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class ModelFineTuner:
    """
    A class for fine-tuning pre-trained models for various sequence classification tasks.
    """
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
    
    def train(self, train_data):
        """
        Trains the model using the specified training data.

        Args:
            train_data (Dataset): The dataset for training the model.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,          
            num_train_epochs=3,              
            per_device_train_batch_size=8,  
            gradient_accumulation_steps=8,
            warmup_steps=500, 
            weight_decay=0.01,   
            save_strategy="epoch"         
        )

        trainer = Trainer(
            model=self.model,                         
            args=training_args,                  
            train_dataset=train_data    
        )

        trainer.train()


def main(args):
    model_path = args.model
    data_path = args.input
    output_dir = args.output

    if not os.path.exists(model_path):
        raise ValueError("Bad input path.")
    
    # make output dir
    os.makedirs(output_dir, exist_ok=True)

    # pretrain task1
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    post_dataset = PostsDataset(data_path, tokenizer, "WhosePost")
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=post_dataset.num_labels)
    fine_tunner = ModelFineTuner(model, output_dir)
    fine_tunner.train(post_dataset)

    # pretrain task2
    # load model from output_dir this time
    question_dataset = QuestionDataset(data_path, tokenizer, "Category")
    model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=question_dataset.num_labels)
    fine_tunner = ModelFineTuner(model, output_dir)
    fine_tunner.train(question_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on two sequential tasks.")
    parser.add_argument("-i", "--input", type=str, default="data/cleaned_posts.csv", help="Path to the input data CSV file.")
    parser.add_argument("-o", "--output", type=str, default="models/", help="Directory where outputs will be saved.")
    parser.add_argument("-m", "--model", type=str, default="nlpaueb/legal-bert-base-uncased", help="Model path or identifier.")

    args = parser.parse_args()

    main(args)
