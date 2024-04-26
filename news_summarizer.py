from datasets import load_dataset
from fine_tune_model import text_2_num, train_model, predict_from_model
from fine_tune_model import TextSummarizationModel
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
from transformers import BertModel, RobertaModel, XLNetModel
import torch
from torch.utils.data import DataLoader, TensorDataset

# Loads data (within function) from the news-qa-summarization dataset
# model_params are the specifications for the model
# tokenizer is a pretrained tokenizer for the model that is being fine tuned
# xl is a flag to indicate that the pretrained model is XLnet (instead of BERT or roBERTa)
# model_save_path specifies the location to save the model after training
# This function calls the functions defined in fine_tune_model.py
# for model setup, tokenizing, and training
# Train on max_samples
def fine_tune_from_pretrained(tokenizer, model_save_path, model_params, xl=False, max_samples=10388):
    # Load the dataset
    dataset = load_dataset('glnmario/news-qa-summarization')

    #train_pc = round(train_pc*max_samples) #number of training cases

    train_data = dataset['train']['story'][:max_samples]
    #test_data = story_data[train_test_split:]

    train_data_sum = dataset['train']['summary'][:max_samples]
    #test_data_sum = summary_data[train_test_split:]

    vocab_size = len(tokenizer.vocab)

    numerical, attention_masks = text_2_num(train_data, tokenizer, xl=xl)
    numerical_s, _ = text_2_num(train_data_sum, tokenizer, xl=xl)

    #numerical_test, attention_masks_test = text_2_num(test_data, tokenizer, xl=xl)
    #numerical_s_test, _ = text_2_num(train_pc, tokenizer, xl=xl)

    model_params['vocab_size'] = vocab_size

    model = TextSummarizationModel(hidden_dim=model_params['hidden_dim'], 
                                lstm_layers=model_params['lstm_layers'], 
                                vocab_size=model_params['vocab_size'], 
                                pretrained=model_params['pretrained'])


    # Creating TensorDataset and DataLoader
    dataset = TensorDataset(torch.tensor(numerical), torch.tensor(attention_masks), torch.tensor(numerical_s))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Creating TensorDataset and DataLoader
    #test_dataset = TensorDataset(torch.tensor(numerical_test), torch.tensor(attention_masks_test))
    #test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    train_model(dataloader, model, epochs=4, lr=0.0001)

    # Save the model
    torch.save(model.state_dict(), model_save_path)



models = {
    'bert-base-uncased': BertModel.from_pretrained('bert-base-uncased'),
    'roberta-base': RobertaModel.from_pretrained('roberta-base'),
    'xlnet-base-cased': XLNetModel.from_pretrained('xlnet-base-cased'),
}

tokenizers = {
    'tokenizer_bert' : BertTokenizer.from_pretrained('bert-base-uncased'),
    'tokenizer_roberta' : RobertaTokenizer.from_pretrained('roberta-base'),
    'tokenizer_xlnet' : XLNetTokenizer.from_pretrained('xlnet-base-cased'),
}

path_save = {
    'path_bert' : 'fine_tuned_bert.pth',
    'path_roberta' : 'fine_tuned_roberta.pth',
    'path_xlnet' : 'fine_tuned_xlnet.pth',
}

### Uncomment the lines corresponding to the model (BERT, roBERTa, or XLNet) that you want to train on

model_params = {'hidden_dim' : 256, 'lstm_layers' : 1, 'vocab_size' : 0,  'pretrained' : models['bert-base-uncased']}
#model_params = {'hidden_dim' : 256, 'lstm_layers' : 1, 'vocab_size' : 0,  'pretrained' : models['roberta-base']}
#model_params = {'hidden_dim' : 256, 'lstm_layers' : 1, 'vocab_size' : 0,  'pretrained' : models['xlnet-base-cased']}

# Test on 90 percent of the data

#fine_tune_from_pretrained(tokenizers['tokenizer_bert'], path_save['path_bert'], 
#                          model_params, max_samples=round(10388*0.9))
#fine_tune_from_pretrained(tokenizers['tokenizer_roberta'], path_save['path_roberta'], 
#                          model_params, max_samples=round(10388*0.9))
#fine_tune_from_pretrained(tokenizers['tokenizer_xlnet'], path_save['path_xlnet'], 
#                          model_params, xl=True, max_samples=round(10388*0.9))

# Example loading the model
new_model = TextSummarizationModel(hidden_dim=model_params['hidden_dim'], 
                                lstm_layers=model_params['lstm_layers'], 
                                vocab_size=model_params['vocab_size'], 
                                pretrained=model_params['pretrained'])

new_model.load_state_dict(torch.load(path_save['path_bert']))
#new_model.load_state_dict(torch.load(path_save['path_roberta']))
#new_model.load_state_dict(torch.load(path_save['path_xlnet']))

new_model.eval()  # Set the model to evaluation mode
print("model loaded!")
