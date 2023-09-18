# Fake News Detection by Learning Convolution Filters through Contextualized Attention

## Dataset
The [LIAR dataset](https://github.com/thiagorainmaker77/liar_dataset) consists of 12,836 short statements taken from POLITIFACT and labeled by humans for truthfulness, subject, context/venue, speaker, state, party, and prior history. For truthfulness, the LIAR dataset has six labels: pants-fire, false, mostly-false, half-true, mostly-true, and true. These six label sets are relatively balanced in size. The statements were collected from a variety of broadcasting mediums, like TV interviews, speeches, tweets, debates, and they cover a broad range of topics such as the economy, health care, taxes and election. The [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) dataset is an extension to the LIAR dataset by automatically extracting for each claim the justification that humans have provided in the fact-checking article associated with the claim.

LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

Description of the TSV format:

Column 1: the ID of the statement ([ID].json).
Column 2: the label.
Column 3: the statement.
Column 4: the subject(s).
Column 5: the speaker.
Column 6: the speaker's job title.
Column 7: the state info.
Column 8: the party affiliation.
Column 9-13: the total credit history count, including the current statement.
9: barely true counts.
10: false counts.
11: half true counts.
12: mostly true counts.
13: pants on fire counts.
Column 14: the context (venue / location of the speech or statement).

## Network Architecture
![Screenshot 1](https://github.com/ekagra-ranjan/fake-news-detection-LIAR-pytorch/blob/master/fake-net.png "Net")


## Methodology
Instead of directly extracting features from Statement, we employ an attention mechanism to use the given side information (subject, speaker, job, state, party, context and justification) to attend over the given statement to check its truthfulness. The attention mechanism makes the process of feature extraction from statement contextualized based on side information. See Fig. 1 for the graphical representation of
the architecture.

## How to Use

Following libraries need to be installed:
  1. Torch
  2. Numpy

To execute the code:
  1. Run `main.py` which is the driver of the experiments. All parameters and function call done from this script.
  2. To train a model change the variable `mode` in `main.py` to `train`.
  3. Model will be saved as per parameters defined in main.py script.
  4. For evaluating a saved model, change `mode` to `test` and put the name of the saved model in the variable `pathModel`.
  5. To run LIAR dataset, change the variable `dataset_name` to `LIAR` and
  6. To run LIAR-PLUS dataset then change `dataset_name` to `LIAR-PLUS`.
