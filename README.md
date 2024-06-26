# NLP_project
### By: Paul Vanderveen

## CS436 Natural Language Processing Project

Repository used to store, files, results and graphs used turing training, testing, and evaluation
More information regarding the exact functions used can be found by navigating each file. The information
below gives a brief overview of each file and it's main purpose in the project.

## Image files
BERT_k_cos.png 
roBERTa_k_cos.png
xlnet_k_cos.png - 
These are graphs generated with MatpltLib of the average values of the normalized cosine function

(exp(c*k+k)-1) / (exp(2*k)-1)

where c ranges across all (raw) cosine values corresponding to model-specific metrics

for different constant values of k
(discretized of course, but the interval difference or delta between consecuative k's is 0.01,
small enough where the graph can be quite accurate of the actual function).

## Prediction (model-generated) files
bert_predictions.txt
roberta_predictions.txt
xlnet_predictions.txt - 
These are the generated predictions corresponding from previously fine-tuned models used during this experiment.
(i.e. bert_predictions are the predicitons from fine tuning BERT)
However, you will need to re-train the models to generate new predictions.
(The model save files were too large to include in this repository)


## Cosine similarity calculator
cosine.py - 
Contains the function used to calculate the normalize cosine similarity metric
for recall
Precision reverses the roles of reference and prediction so no need for an almost
duplicate function for precision.

## Model setup and definitions
fine_tune_model.py - 
Main program used to define the tokenization function, Loss function, Model class, training function, and generative prediction function.

## Metric calculator
metrics.py - 
These define functions used to calculate BLEU, ROGUE-1, and their respective cosine similarity integrated metrics.

## Model trainer
news_summarizer.py - 
Main "driver" program that calls the functions defined in fine_tune_model to tokenize data and fine-tune a model on that data.
Prepares and processes raw data for training, and converts it to a batched Dataloader to ready to be used by the training function 

## Prediction Generator
summary_example.py - 
Main "driver" program to generate predictions for already fine-tuned models. Calls the prediction function in fine_tune_model
Prepares and proccess raw data for testing, and converts it to a batched Dataloader to ready to be used by the prediction function 

## Saved scores from experiment
scores.txt - Used to keep track of (overal, random sample) Metric values (BLEU, ROGUE-1, Cosine, etc.) 

---------------------------------------

## How would you get started:

You would need to train your model if you wish to generate machine-based predictions. 
You can do this by simply running the 'news_summarizer.py' program. Just be sure to uncomment the lines corresponding to the model you wish to train on!

Alternatively, to focus on computing metrics for any pair of sentences (i.e. "The cat sat on the mat", "The cat in the red hat" ) you can get these metrics
for any pretrained model (again, just uncomment the appropriate lines!)
In lines 270-271 of 'metrics.py' (or these lines):

r = "The fox jumped over the flowing river."
g = "The quick fox hopped over the stream."

You would replace the referring sentence with the current value of r, and the prediction (guessed) sentence with the current value of g.

Then just run 'metrics.py' to get the corresponding scores.
