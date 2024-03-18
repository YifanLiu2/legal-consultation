from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.nn import CrossEntropyLoss


class ConversationDataset(Dataset):
     def __init__(self, file_path, tokenizer, max_length=256):
         self.tokenizer = tokenizer
         self.max_length = max_length
         self.data = pd.read_csv(file_path)
         self.encodings = self.tokenizer(self.data['CleanedText'].tolist(), truncation=True, padding=True, 
                                         max_length=max_length, return_tensors="pt")
         self.labels = torch.tensor(self.data['WhosePost'].values.astype(int))
     def __getitem__(self, idx):
         item = {key: val[idx] for key, val in self.encodings.items()}
         item['labels'] = self.labels[idx]
         return item

     def __len__(self):
         return len(self.labels)
     
class QuestionDataset(Dataset):
     def __init__(self, file_path, tokenizer, max_length=256):
         self.tokenizer = tokenizer
         self.max_length = max_length
         
         df = pd.read_csv(file_path)
         grouped_df = df.groupby('QuestionUno')['CleanedText'].apply(lambda texts: ' '.join(texts)).reset_index()
         grouped_df = grouped_df.merge(df[['QuestionUno', 'Category']].drop_duplicates(), on='QuestionUno', how='left')
         
         self.label_mapping = {label: idx for idx, label in enumerate(grouped_df['Category'].unique())}
         grouped_df['Category'] = grouped_df['Category'].map(self.label_mapping)
         self.encodings = self.tokenizer(grouped_df['CleanedText'].tolist(), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
         self.labels = torch.tensor(grouped_df['Category'].values, dtype=torch.long)
     
     def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        return item

     def __len__(self):
         return len(self.labels)


class ModelFineTuner:
    """
    A class for fine-tuning pre-trained models for various sequence classification tasks.

    Attributes:
        model_name (str): The pre-trained model to use.
        num_labels (int): Number of labels for the classification task.
        max_length (int): Maximum length of the input sequences. Defaults to 512.
        tokenizer: Tokenizer associated with the pre-trained model.
        model: The pre-trained model loaded for sequence classification, adapted for `num_labels`.
        device: The device on which the model will be run.
    """
    def __init__(self, model_name, num_labels, max_length=512):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    

    def train(self, train_data):

        model = AutoModelForSequenceClassification.from_pretrained('nlpaueb/legal-bert-base-uncased', num_labels=2)

        training_args = TrainingArguments(
            output_dir='/Users/haobao/legal-consultation/data',          
            num_train_epochs=3,              
            per_device_train_batch_size=8,  
            logging_dir='/Users/haobao/legal-consultation/data',            
        )

        trainer = Trainer(
            model=model,                         
            args=training_args,                  
            train_dataset=train_data,       
        )

        trainer.train()

    def train2(self, train_data):
        df = pd.read_csv(dataset_path)
        grouped_df = df.groupby('QuestionUno')['CleanedText'].apply(lambda texts: ' '.join(texts)).reset_index()
        grouped_df = grouped_df.merge(df[['QuestionUno', 'Category']].drop_duplicates(), on='QuestionUno', how='left')
        model = BertForSequenceClassification.from_pretrained('nlpaueb/legal-bert-base-uncased', num_labels=grouped_df['Category'].nunique())
        
        training_args = TrainingArguments(
            output_dir="/Users/haobao/legal-consultation/data",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="/Users/haobao/legal-consultation/data",
            logging_steps=10,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
        )

        trainer.train()

    def save_model(self, path):
        self.model.save_pretrained('/Users/haobao/legal-consultation/data')
        self.tokenizer.save_pretrained('/Users/haobao/legal-consultation/data')


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    dataset_path = "/Users/haobao/Desktop/ECO482/Research Project/Code/Data/sample_dataset.csv"
    train_dataset = ConversationDataset(dataset_path, tokenizer, max_length=256)
    tunner = ModelFineTuner("nlpaueb/legal-bert-base-uncased", num_labels=2, max_length=512)
    tunner.train(train_dataset)

