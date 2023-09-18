# Fake News Detection by Learning Convolution Filters through Contextualized Attention

## Dataset
The [LIAR dataset](https://github.com/thiagorainmaker77/liar_dataset) consists of 12,836 short statements taken from POLITIFACT and labeled by humans for truthfulness, subject, context/venue, speaker, state, party, and prior history. For truthfulness, the LIAR dataset has six labels: pants-fire, false, mostly-false, half-true, mostly-true, and true. These six label sets are relatively balanced in size. The statements were collected from a variety of broadcasting mediums, like TV interviews, speeches, tweets, debates, and they cover a broad range of topics such as the economy, health care, taxes and election. The [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) dataset is an extension to the LIAR dataset by automatically extracting for each claim the justification that humans have provided in the fact-checking article associated with the claim.

## Network Architecture
![Screenshot 1](https://github.com/ekagra-ranjan/fake-news-detection-LIAR-pytorch/blob/master/fake-net.png "Net")


## Methodology
Instead of directly extracting features from Statement, we employ an attention mechanism to use the given side information (subject, speaker, job, state, party, context and justification) to attend over the given statement to check its truthfulness. The attention mechanism makes the process of feature extraction from statement contextualized based on side information. See Fig. 1 for the graphical representation of
the architecture. For more detailed explanation of the approach read the [paper](https://www.researchgate.net/publication/341378920_Fake_News_Detection_by_Learning_Convolution_Filters_through_Contextualized_Attention).

## How to Use

Following libraries need to be installed:
  1. Torch
  2. Numpy

To execute the code:
  1. Run `main.py` which is the driver of the experiments. All parameters and function call done from this script
  2. To train a model change the variable `mode` in `main.py` to `train`.
  3. Model will be saved as per parameters defined in main.py script
  4. For evaluating a saved model, change `mode` to `test` and put the name of the saved model in the variable `pathModel`.
  5. To run LIAR dataset, change the variable `dataset_name` to `LIAR` and
  6. To run LIAR-PLUS dataset then change `dataset_name` to `LIAR-PLUS`.
