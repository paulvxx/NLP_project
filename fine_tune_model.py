import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

import torch.nn.functional as F

# Convert raw text to numerical data (tokens)
def text_2_num(texts, tokenizer, max_len=512, xl=False, padding=True):
    indexed_texts = []
    attention_masks = []
    max_tokens = max_len - 2

    for text in texts:
        tokens = tokenizer.tokenize(text)
        tokens = tokens[:max_tokens]

        if not xl:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        else:
            tokens = tokens + ['<sep>', '<cls>']

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(token_ids)

        if not padding:
            indexed_texts.append(token_ids)
            attention_masks.append(attention_mask)
            continue

        padded_length = max_len - len(token_ids)
        
        # Apply padding
        token_ids += [0] * padded_length
        attention_mask += [0] * padded_length

        indexed_texts.append(token_ids)
        attention_masks.append(attention_mask)

    return indexed_texts, attention_masks

class TextSummarizationModel(nn.Module):
    def __init__(self, hidden_dim, lstm_layers, vocab_size, pretrained):
        super(TextSummarizationModel, self).__init__()
        self.pretrained = pretrained #= BertModel.from_pretrained('bert-base-uncased')
        # Use Droupout and LSTM layers
        self.dropout = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.pretrained(input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs[0]
        sequence_output = self.dropout(sequence_output)
        decoder_output, _ = self.lstm(sequence_output)
        logits = self.fc(decoder_output)
        return logits

# more ideal loss function for summarization
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def train_model(dataload, model, epochs, lr=0.001, pad_token_id=0):
    # train on cuda if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    model.to(device)
    model.train()
    criterion = FocalLoss(gamma=2.0, alpha=0.25, ignore_index=pad_token_id, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  

    for epoch in range(epochs):
        # iterate through contents of batched data
        for input_ids, attention_mask, labels in tqdm(dataload):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()  # Decay the learning rate

            with torch.no_grad():
                predictions = outputs.argmax(dim=-1)
                unique, counts = torch.unique(predictions, return_counts=True)
                unique_tokens = dict(zip(unique.tolist(), counts.tolist()))
                print(f"Unique tokens this batch: {len(unique_tokens)}")
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


def predict_from_model(input_batch, model, tokenizer, max_length=512):
    model.eval()
    predictions = []
    device = next(model.parameters()).device
    eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids(['[SEP]'])[0]  # Fallback to '[SEP]' if EOS not set

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(input_batch):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = model(input_ids, attention_mask=attention_mask)
            #print(output)
            token_ids = output.argmax(dim=-1)
            
            for ids in token_ids:
            #    eos_mask = ids == eos_token_id
            #   if eos_mask.any():
            #        eos_index = eos_mask.nonzero(as_tuple=True)[0][0]
            #        summary = tokenizer.decode(ids[:eos_index + 1], skip_special_tokens=True)
            #    else:
            #       summary = tokenizer.decode(ids[:max_length], skip_special_tokens=True)
                summary = tokenizer.decode(ids[:max_length], skip_special_tokens=True)
                predictions.append(summary)

    return predictions
