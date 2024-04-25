from datasets import load_dataset
from fine_tune_model import text_2_num
from fine_tune_model import TextSummarizationModel
from cosine import cosine_recall
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
from transformers import BertModel, RobertaModel, XLNetModel
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# Takes a list of tokens (ignores padding)
# for reference and prediction sentences 
# and determines the percentage of words 
# in the prediction that are present in the reference
def precision(reference, prediction):
    r_set = set(reference)
    found = 0
    total = len(prediction)
    for p in prediction:
        if p in r_set and p != 0:
            found += 1
    return found / total

# Takes a list of tokens (ignores padding)
# for reference and prediction sentences 
# and determines the percentage of words 
# in the reference that are present in the prediction
def recall(reference, prediction):
    p_set = set(prediction)
    found = 0
    total = len(reference)
    for r in reference:
        if r in p_set and r != 0:
            found += 1
    return found / total

# cosine normalized similarity for precision
def cos_precision(model, tokenizer, reference, prediction, k=1):
    # Swapped reference and prediction to get precision
    return cosine_recall(model, tokenizer, prediction, reference, const_k=k)

# cosine normalized similarity for recall
def cos_recall(model, tokenizer, reference, prediction, k=1):
    return cosine_recall(model, tokenizer, reference, prediction, const_k=k)


# Computes the BLEU metric based on invidiual words (unigrams)
# refs is a list of references
# preds is a list of predictions
# cos_sim is a parameter to indicate using cosine similarity integration
def bleu(refs, preds):
    total = 0
    points = 0

    samples = len(refs)

    for i in tqdm(range(samples)):
        total += 1
        points += precision(refs[i], preds[i])

    if total == 0:
        return 0
    else:
        return points / total


# Computes the ROGUE-1 metric based on invidiual words (unigrams)
# refs is a list of references
# preds is a list of predictions
# cos_sim is a parameter to indicate using cosine similarity integration
def rogue(refs, preds):
    total = 0
    points = 0
    total_prec = 0
    total_rec = 0

    samples = len(refs)

    for i in tqdm(range(samples)):
        total += 1
        points += precision(refs[i], preds[i])

    # get total precision
    if total == 0:
        total_prec = 0
    else:
        total_prec = points / total        

    total = 0
    points = 0
    for i in tqdm(range(samples)):
        total += 1
        points += recall(refs[i], preds[i])

    # get total recall
    if total == 0:
        total_rec = 0
    else:
        total_rec = points / total

    # metric
    if total_rec == 0 and total_prec == 0:
        return 0
    else:
        return (2 * total_prec * total_rec)/(total_prec + total_rec)


# Compute cosine similarity with BLEU metric
def bleu_cos_norm(model, tokenizer, refs, preds, k=1):
    total = 0
    points = 0

    samples = len(refs)

    for i in range(samples):
        total += 1
        points += cos_precision(model, tokenizer, refs[i], preds[i], k=k)

    if total == 0:
        return 0
    else:
        return points / total


# Compute cosine similarity with ROGUE metric
def rogue_cos_norm(model, tokenizer, refs, preds, k=1):
    total = 0
    points = 0
    total_prec = 0
    total_rec = 0

    samples = len(refs)

    for i in range(samples):
        total += 1
        points += cos_precision(model, tokenizer, refs[i], preds[i], k=k)

    # get total precision
    if total == 0:
        total_prec = 0
    else:
        total_prec = points / total        

    total = 0
    points = 0
    for i in range(samples):
        total += 1
        points += cos_recall(model, tokenizer, refs[i], preds[i], k=k)

    # get total recall
    if total == 0:
        total_rec = 0
    else:
        total_rec = points / total

    # metric
    if total_rec == 0 and total_prec == 0:
        return 0
    else:
        return (2 * total_prec * total_rec)/(total_prec + total_rec)


# Read lines of a file into list
def load_predictions(filename):
    predictions = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            predictions.append(line.strip())
    
    return predictions


# For a specific model (and its corresponding tokenizer)
# Graphs the k_normalized cosine similarity metric
# (exp(k*c + k) - 1) / (exp(2) - 1)
# Where c is the cosine value
# for different values of k, starting at delta
# and increasing the x value by delta up to 'step' times
# r is the reference sentence
# g is the predicted (or guessed) sentence
def k_norm_graph(model_Name, model, tokenizer, r, g, delta=0.04, step=50):

    # Avoid constant value of 0
    if delta < 0:
        print("delta cannot be 0")
        return

    x = np.arange(step, dtype=np.float64)*delta + delta
    print(x)
    y = np.zeros((2, step), dtype=np.float64)

    for i in tqdm(range(step)):
        y[0][i] = bleu_cos_norm(model, tokenizer, [r], [g], k=i*delta+delta)
        y[1][i] = rogue_cos_norm(model, tokenizer, [r], [g], k=i*delta+delta)

    plt.plot(x, y[0], label='BLEU k-normalized Cosine Similarity')
    plt.plot(x, y[1], label='ROGUE k-normalized Cosine Similarity')

    plt.legend()
    plt.title(model_Name + " k-normalized Cosine Similarity")
    plt.xlabel("k")
    plt.ylabel("k-normalized score")

    plt.show()


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


#model_name = "BERT"
#model = models['bert-base-uncased']
#tokenizer = tokenizers['tokenizer_bert']
#model_name = "roBERTa"
#model = models['roberta-base']
#tokenizer = tokenizers['tokenizer_roberta']
model_name = "xlnet"
model = models['xlnet-base-cased']
tokenizer = tokenizers['tokenizer_xlnet']

start_testing_index = round(10388*0.9)

dataset = load_dataset('glnmario/news-qa-summarization')
test_data = dataset['train']['summary'][start_testing_index:]

#predictions = load_predictions('bert_predictions.txt')
#predictions = load_predictions('roberta_predictions.txt')
predictions = load_predictions('xlnet_predictions.txt')

#ref_tokens, _ = text_2_num(test_data, tokenizers['tokenizer_bert'], padding=False)
#ref_tokens, _ = text_2_num(test_data, tokenizers['tokenizer_roberta'], padding=False)
ref_tokens, _ = text_2_num(test_data, tokenizers['tokenizer_xlnet'], xl=True, padding=False)

#prediction_tokens, _ = text_2_num(predictions, tokenizers['tokenizer_bert'], padding=False)
#prediction_tokens, _ = text_2_num(predictions, tokenizers['tokenizer_roberta'], padding=False)
prediction_tokens, _ = text_2_num(predictions, tokenizers['tokenizer_xlnet'], xl=True, padding=False)

#print("BLEU Score " + str(bleu(ref_tokens, prediction_tokens)))
#print("ROGUE Score " + str(rogue(ref_tokens, prediction_tokens)))

# Tokenizing is done within the cos similarity function

#print(test_data[69]) 
#print("--------------------------")
#print(predictions[69])
#print("--------------------------")

# Example sentences (ideally for a well-trained model)
r = "The fox jumped over the flowing river."
g = "The quick fox hopped over the stream."

#r_tokens, _ = text_2_num([r], tokenizers['tokenizer_bert'], padding=False)
#r_tokens, _ = text_2_num([r], tokenizers['tokenizer_roberta'], padding=False)
r_tokens, _ = text_2_num([r], tokenizers['tokenizer_xlnet'], xl=True, padding=False)

#g_tokens, _ = text_2_num([g], tokenizers['tokenizer_bert'], padding=False)
#g_tokens, _ = text_2_num([g], tokenizers['tokenizer_roberta'], padding=False)
g_tokens, _ = text_2_num([g], tokenizers['tokenizer_xlnet'], xl=True, padding=False)

print(r_tokens)
print(g_tokens)
print("BLEU Score " + str(bleu(r_tokens, g_tokens)))
print("ROGUE Score " + str(rogue(r_tokens, g_tokens)))
print("BLEU Cos Sim Normalized Score " + str(bleu_cos_norm(model, tokenizer, [r], [g])))
print("ROGUE Cos Sim Normalized Score " + str(rogue_cos_norm(model, tokenizer, [r], [g])))

# account for that we start at delta not 0
k_norm_graph(model_name, model, tokenizer, r, g, delta=0.01, step=400)
