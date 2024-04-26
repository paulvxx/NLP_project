from datasets import load_dataset
from fine_tune_model import text_2_num, train_model, predict_from_model
from fine_tune_model import TextSummarizationModel
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
from transformers import BertModel, RobertaModel, XLNetModel
import torch
from torch.utils.data import DataLoader, TensorDataset

# Function to call the predictions function in fine_tune_model.py
# start_testing_index indicates where to start testing in main dataset
# an index of 0 will use the entire dataset for testing
# model you want to test on
# tokenizer used to tokenize strings in the test dataset
def make_predictions(model, tokenizer, xl=False, start_testing_index=0):
    #index to start testing at start_testing_index until end of dataset
    dataset = load_dataset('glnmario/news-qa-summarization')

    test_data = dataset['train']['story'][start_testing_index:]

    numerical, attention_masks = text_2_num(test_data, tokenizer, xl=xl)

    dataset = TensorDataset(torch.tensor(numerical), torch.tensor(attention_masks))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    predictions = predict_from_model(dataloader, model, tokenizer)

    return predictions


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
    'bert_predictions' : 'bert_predictions.txt',
    'roberta_predictions' : 'roberta_predictions.txt',
    'xlnet_predictions' : 'xlnet_predictions.txt',
}

### Uncomment the lines corresponding to the model (BERT, roBERTa, or XLNet) that you want to train on

#model_params = {'hidden_dim' : 256, 'lstm_layers' : 1, 'vocab_size' : 30522,  'pretrained' : models['bert-base-uncased']}
#model_params = {'hidden_dim' : 256, 'lstm_layers' : 1, 'vocab_size' : 50265,  'pretrained' : models['roberta-base']}
model_params = {'hidden_dim' : 256, 'lstm_layers' : 1, 'vocab_size' : 32000,  'pretrained' : models['xlnet-base-cased']}

# Example loading the model
new_model = TextSummarizationModel(hidden_dim=model_params['hidden_dim'], 
                                lstm_layers=model_params['lstm_layers'], 
                                vocab_size=model_params['vocab_size'], 
                                pretrained=model_params['pretrained'])


#new_state_dict = torch.load(path_save['path_bert'], map_location=torch.device('cpu'))
#new_state_dict = torch.load(path_save['path_roberta'], map_location=torch.device('cpu'))
new_state_dict = torch.load(path_save['path_xlnet'], map_location=torch.device('cpu'))

new_model.load_state_dict(new_state_dict)

new_model.eval()  # Set the model to evaluation mode
print("model loaded!")

# effectively a 0.1 test-train split
start_testing = round(10388*0.9) # index to start testing at until end of dataset

#predictions = make_predictions(new_model, tokenizers['tokenizer_bert'], start_testing_index=start_testing)
#predictions = make_predictions(new_model, tokenizers['tokenizer_roberta'], start_testing_index=start_testing)
predictions = make_predictions(new_model, tokenizers['tokenizer_xlnet'], xl=True, start_testing_index=start_testing)

#f = open(path_save['bert_predictions'], 'w', encoding='utf-8')
#f = open(path_save['roberta_predictions'], 'w', encoding='utf-8')
f = open(path_save['xlnet_predictions'], 'w', encoding='utf-8')

for p in predictions:
    f.write(p)
    f.write('\n')

f.close()
