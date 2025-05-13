from django.conf import settings
from django.shortcuts import render
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from django.contrib import messages
from users.models import UserRegistrationModel
from .models import UserRegistrationModel

# Set up a non-interactive backend for matplotlib
matplotlib.use('Agg')

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm

# --- 1. Data Preparation ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def load_data(csv_file_path, text_col='lemmatized_text', label_col='Label', test_size=0.2, random_state=42):
    try:
        df = pd.read_csv(csv_file_path)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Columns '{text_col}' or '{label_col}' not found in the dataset.")

        texts = df[text_col].tolist()
        
        # Convert string labels to integers
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df[label_col].tolist())

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=test_size, random_state=random_state
        )

        num_labels = len(label_encoder.classes_)
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, num_labels, label_encoder

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# --- 2. BERT Feature Extraction ---
def get_bert_embeddings(texts, tokenizer, model, max_len, device):
    all_embeddings = []
    for text in texts:
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoding)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0, :].squeeze(0)
        all_embeddings.append(cls_embedding)
    return torch.stack(all_embeddings)


# --- 3. Classification (T5) ---
class T5Classifier(torch.nn.Module):
    def __init__(self, t5_model_name, num_labels):
        super(T5Classifier, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.linear = torch.nn.Linear(self.t5.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.linear(encoder_outputs.last_hidden_state[:, 0, :])
        
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return loss, logits
        
        return logits


def train_t5_model(t5_model, train_dataloader, val_dataloader, optimizer, num_epochs, device):
    t5_model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            loss, _ = t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dataloader)
        val_metrics = evaluate_model(t5_model, val_dataloader, device)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f} - Val Metrics: {val_metrics}")


def evaluate_model(model, dataloader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    return accuracy,precision,recall,f1

def predict_texts(t5_model, tokenizer, texts, device, max_len, label_encoder):
    t5_model.eval()
    predictions = []
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            logits = t5_model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
            pred_label_idx = torch.argmax(logits, dim=-1).cpu().numpy()[0]
            pred_label = label_encoder.inverse_transform([pred_label_idx])[0]  # Decode
            predictions.append(pred_label)  # Append decoded label
    return predictions


def trainingPage(request):
        return render(request, 'users/accuracy.html')


# --- Main Execution ---
def training(request):
    # Parameters
    DATA_CSV_PATH = r"C:\Users\DELL\projectone\Ai_Text_Detection_Using_Bert_and_T5_Models1\Ai_Text_Detection_Using_Bert_and_T5_Models\media\duplicates.csv"
    TEXT_COLUMN = "Lemmatized_text"
    LABEL_COLUMN = "Label"
    BERT_MODEL_NAME = "bert-base-uncased"
    T5_MODEL_NAME = "t5-small"
    MAX_LEN = 128
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    LEARNING_RATE = 2e-5
    MODEL_SAVE_PATH = "media/t5_classifier_model.pth"  # Path to save the model
    LABEL_ENCODER_SAVE_PATH = "media/label_encoder.pkl" #Path to save the label encoder

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    data = load_data(DATA_CSV_PATH, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN)
    if data is None:
        exit()

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, NUM_LABELS, label_encoder = data
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Initialize T5 model and optimizer
    t5_model = T5Classifier(T5_MODEL_NAME, NUM_LABELS).to(device)
    optimizer = AdamW(t5_model.parameters(), lr=LEARNING_RATE)

    # Train the T5 Model
    train_t5_model(t5_model, train_dataloader, val_dataloader, optimizer, NUM_EPOCHS, device)

    # Evaluate on Test Set
    
    accuracy,precision,recall,f1 = evaluate_model(t5_model, test_dataloader, device)
    print(f"Test Metrics: {accuracy,precision,recall,f1}")

    # Save the Model
    torch.save(t5_model.state_dict(), MODEL_SAVE_PATH)  # Save only the model's parameters
    print(f"T5 model saved to {MODEL_SAVE_PATH}")

    # Save the LabelEncoder


    
    return render(request, 'users/accuracy.html', {"accuracy":accuracy,'precision':precision,'recall':recall,'f1':f1})

from django.shortcuts import render
import torch
from transformers import AutoTokenizer
import joblib  # For loading the label encoder

# Load the trained model and label encoder
MODEL_SAVE_PATH = "media/t5_classifier_model.pth"
LABEL_ENCODER_SAVE_PATH = "media/label_encoder.pkl"
T5_MODEL_NAME = "t5-small"
MAX_LEN = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model():
    model = T5Classifier(T5_MODEL_NAME, num_labels=4)  # Adjust num_labels as needed
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load the label encoder
def load_label_encoder():
    return joblib.load(LABEL_ENCODER_SAVE_PATH)

# Prediction function
def predict_text(model, tokenizer, text, device, max_len, label_encoder):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
    pred_label_idx = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    pred_label = label_encoder.inverse_transform([pred_label_idx])[0]
    return pred_label

# View for handling user input
def prediction(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input", "").strip()
        if user_input:
            # Load model and tokenizer
            model = load_model()
            tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
            label_encoder = load_label_encoder()

            # Get prediction
            prediction = predict_text(model, tokenizer, user_input, device, MAX_LEN, label_encoder)
            print(prediction)
            return render(request, 'users/prediction.html', {
                "user_input": user_input,
                "prediction": prediction
            })
        
            
    else:
        return render(request,'users/prediction.html')
    
    






from django.shortcuts import render, redirect
from .models import UserRegistrationModel
from django.contrib import messages

def UserRegisterActions(request):

    if request.method == 'POST':

        try:

            user = UserRegistrationModel(
                name=request.POST['name'],
                loginid=request.POST['loginid'],
                password=request.POST['password'],
                mobile=request.POST['mobile'],
                email=request.POST['email'],
                locality=request.POST['locality'],
                address=request.POST['address'],
                state=request.POST['state'],
                status='waiting'
            )
            if user.full_clean:
                user.save()
                messages.error(request,'Registration Done Successfully')
                return render(request,'UserRegistrations.html')

            else:
                messages.error(request,'Invalid Details , Enter deatils carefully')
                return render(request,'UserRegistrations.html')
        except Exception as e:
            messages.error(request,e)
            return render(request,'UserRegistrations.html')
    return render(request, 'UserRegistrations.html') 


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                data = {'loginid': loginid}
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def index(request):
    return render(request,"index.html")