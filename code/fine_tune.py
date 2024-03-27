import argparse, os
import pandas as pd
import torch
<<<<<<< HEAD
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel, BertTokenizer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
=======
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
>>>>>>> a882c046d806af9d835c0ab53d7206d044931c26


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
<<<<<<< HEAD
        labels = dataset["WhosePost"]
=======
        labels = dataset[label_col]
>>>>>>> a882c046d806af9d835c0ab53d7206d044931c26
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
<<<<<<< HEAD
        labels = dataset["WhosePost"]
=======
        labels = dataset[label_col]
>>>>>>> a882c046d806af9d835c0ab53d7206d044931c26
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

<<<<<<< HEAD

class MultiAspectClassifier(nn.Module):
    def __init__(self):
        super(MultiAspectClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        # Output layer now has 3 units, one for each aspect
        self.out = nn.Linear(self.bert.config.hidden_size, 3)  
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

class FeedbackDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.dataframe = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        text = str(self.dataframe.iloc[idx]['CleanedText'])
        labels = [
            self.dataframe.iloc[idx]['Validity'],
            self.dataframe.iloc[idx]['Civility'],
            self.dataframe.iloc[idx]['Relevance']
        ]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',  
            return_attention_mask=True,  
            return_tensors='pt',  
            truncation=True
        )
        
        # Prepare the final processed output
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)  # Convert labels to a tensor
        }
    

class Trainer:
    def __init__(self, model, dataset, batch_size=32, learning_rate=0.001):
        self.model = model
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, num_epochs):
        # Assuming you have defined self.device somewhere in your class, e.g.,
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # and you've moved your model to the device with self.model.to(self.device)

        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for i, batch in enumerate(self.dataloader):
                if i % 10 == 0:  # 每处理10个批次打印一次
                    print(f"  Processing batch {i+1}/{len(self.dataloader)}")
                inputs_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()  # Zero the parameter gradients

                # Adjust this call based on your model and input
                outputs = self.model(input_ids=inputs_ids, attention_mask=attention_mask)

                # Assuming outputs are the raw logits; adjust if your model structure is different
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 0:  # 每处理10个批次打印一次
                    print(f"    Current batch loss: {loss.item()}")
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.dataloader)}')


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)



def main(args):
    model_path = args.model
    data_path = args.input
    output_dir = args.output

=======

def main(args):
    model_path = args.model
    data_path = args.input
    output_dir = args.output

>>>>>>> a882c046d806af9d835c0ab53d7206d044931c26
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
<<<<<<< HEAD
    # parser = argparse.ArgumentParser(description="Fine-tune a model on two sequential tasks.")
    # parser.add_argument("-i", "--input", type=str, default="data/cleaned_posts.csv", help="Path to the input data CSV file.")
    # parser.add_argument("-o", "--output", type=str, default="models/", help="Directory where outputs will be saved.")
    # parser.add_argument("-m", "--model", type=str, default="nlpaueb/legal-bert-base-uncased", help="Model path or identifier.")

    # args = parser.parse_args()

    # main(args)


    learning_rate = 0.001
    batch_size = 32
    num_epochs = 1

    model = MultiAspectClassifier()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = FeedbackDataset('/Users/haobao/Desktop/ECO482/Research Project/Code/Data/updated_sample_dataset.csv', tokenizer)
    trainer = Trainer(model=model, dataset=dataset, batch_size=batch_size, learning_rate=learning_rate)
    trainer.train(num_epochs)

=======
    parser = argparse.ArgumentParser(description="Fine-tune a model on two sequential tasks.")
    parser.add_argument("-i", "--input", type=str, default="data/cleaned_posts.csv", help="Path to the input data CSV file.")
    parser.add_argument("-o", "--output", type=str, default="models/", help="Directory where outputs will be saved.")
    parser.add_argument("-m", "--model", type=str, default="nlpaueb/legal-bert-base-uncased", help="Model path or identifier.")

    args = parser.parse_args()
>>>>>>> a882c046d806af9d835c0ab53d7206d044931c26

    main(args)
