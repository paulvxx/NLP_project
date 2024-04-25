from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from numpy import exp
from fine_tune_model import text_2_num


# This function takes two texts a reference and a prediction
# It outputs the cosine recall scores (with normalization)
# const_k (k) is a normalization constant  for  (exp(k*c+k)-1) / (exp(2*k)-1)
# The cosine similarity is taken for each pair of words in the reference and prediction
# For every word r in the reference, the argmax for the cosine similarity (across each word in the prediction)
# is taken as the "alternative" value compared to whether r is found in the prediction (so it is a value from 0 to 1)
# e.g. for the word "river" --> {"stream", "boat", "lake"},  "stream" may have the highest similarity score, 
# so the cosine similarity between "river" and "stream" is taken as the accuracy value for that word rather than 0 
# (as river is not present in the set {"stream", "boat", "lake"})
# NOTE : Precision can just be taken with reference and prediction having swapped arguments
def cosine_recall(model, tokenizer, reference, prediction, const_k=1):
    tokenized1, mask1 = text_2_num([reference], tokenizer, padding=False)
    encoded_input = {'input_ids':torch.tensor(tokenized1), 'attention_mask':torch.tensor(mask1)}

    tokenized2, mask2 = text_2_num([prediction], tokenizer, padding=False)
    encoded_input2 = {'input_ids':torch.tensor(tokenized2), 'attention_mask':torch.tensor(mask2)}

    # Extract embeddings
    with torch.no_grad():
        outputs = model(**encoded_input)
        # use last hidden state as contextual embeddings
        embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
        outputs = model(**encoded_input2)
        embeddings2 = outputs.last_hidden_state.squeeze(0)

    # number of tokens
    nt1 = len(tokenized1[0])
    # number of tokens
    nt2 = len(tokenized2[0])
    
    total_words = nt1
    points = 0

    for t in range(nt1):
        closest = -1 # cos is from -1 to 1
        for t2 in range(nt2):
            cos_sim = cosine_similarity(
                [embeddings[t].numpy()],  # Convert tensor to numpy array
                [embeddings2[t2].numpy()]
            )
            sim = cos_sim.flatten()[0]
            if sim > closest:
                closest = sim  #argmax
        # normalize the argmax
        points += (exp(const_k*closest+const_k)-1) / (exp(2*const_k)-1)

    if (total_words==0):
        return 0.0
    
    # output the total accuracy
    return points / total_words
