# NLP_project
CS436 Natural Language Processing Project

Repository used to store, files, results and graphs used turing training, testing, and evaluation
More information regarding the exact functions used can be found by navigating each file. The information
below gives a brief overview of each file and it's main purpose in the project.

BERT_k_cos.png 
roBERTa_k_cos.png
xlnet_k_cos.png - 



bert_predictions.txt
roberta_predictions.txt
xlnet_predictions.txt - 



cosine.py - 
Contains the function used to calculate the normalize cosine similarity metric
for recall
Precision reverses the roles of reference and prediction so no need for an almost
duplicate function for precision.

fine_tune_model.py - 
Main program used to define the tokenization function, Loss function, Model class, training function, and generative prediction function.

metrics.py - 
These define functions used to calculate BLEU, ROGUE-1, and their respective cosine similarity integrated metrics.

news_summarizer.py - 
Main "driver" program that calls the functions defined in fine_tune_model to tokenize data and fine-tune a model on that data.
Prepares and processes raw data for training, and converts it to a batched Dataloader to ready to be used by the training function 

summary_example.py - 
Main "driver" program to generate predictions for already fine-tuned models. Calls the prediction function in fine_tune_model
Prepares and proccess raw data for testing, and converts it to a batched Dataloader to ready to be used by the prediction function 

scores.txt - Used to keep track of (overal, random sample) Metric values (BLEU, ROGUE-1, Cosine, etc.) 
