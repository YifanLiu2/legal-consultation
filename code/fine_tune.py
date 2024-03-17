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
         self.labels = torch.tensor(self.data['WhosePost'].values.astype(float))
     def __getitem__(self, idx):
         item = {key: val[idx] for key, val in self.encodings.items()}
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
    

    #def preprocess_data(self, texts, labels):
    #    # Tokenize texts
    #    encodings = self.tokenizer(texts, truncation=True, padding=False, max_length=self.max_length, return_attention_mask=True)
    #    # Correctly create a dataset
    #    dataset = Dataset.from_dict({"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": labels})
    #    return dataset
    
    


    def train(self, train_data, epochs=3, batch_size=8, learning_rate=5e-5):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

        self.model.train()
        for _ in range(epochs):
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        

    def save_model(self, path):
        self.model.save_pretrained('/Users/haobao/legal-consultation/data')
        self.tokenizer.save_pretrained('/Users/haobao/legal-consultation/data')

#tokenizer中加入attention_mask参数    #将数据tokenize这一步骤放入get_item中
 # 用chat的方法来train  #在conversation中最好用文件路径当参数输入
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    dataset_path = "/Users/haobao/Desktop/ECO482/Research Project/Code/Data/sample_dataset.csv"
    train_dataset = ConversationDataset(dataset_path, tokenizer, max_length=256)
    tunner = ModelFineTuner("nlpaueb/legal-bert-base-uncased", num_labels=2, max_length=512)
    tunner.train(train_dataset)