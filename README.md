# Paraphraser 

This project providers users the ability to generate paraphrases for sentences through a clean and simple API.  A demo can be seen here: [http://pair-a-phrase.it](Pair-a-phrase)

### Prerequisites

* Python 3.5
* Tensorflow

Run the following command from within your virtual environment:
```
pip install git+git://github.com/vsuthichai/paraphraser.git
```

### Usage
```
from paraphraser import paraphrase
print(paraphrase("hello world"))
```

### Consulting Project

This project was developed for Praxis SW under the Insight Data Science Artificial Intelligence program.  Praxis SW goals are to generate paraphrases for applications.  Their immediate goal is to generate data to improve interaction between consumers and their voice activated devices.  Assume a user issued a command to their user by saying "Book me a flight to Hawaii.", it should be able to understand all the myriad number of ways that you could convey the same meaning using different words.  For example, "Schedule a flight to Hawaii" or "Purchase a planet ticket to Hawaii".  Alexa needs to be able to recognize these different paraphrases in order to recognize your task.

### Model

The underlying model used is a bidirectional LSTM encoder and LSTM decoder with attention trained using Tensorflow.

### Datasets

The dataset used to train this model is an aggregation of many different public datasets.  To name a few:
* para-nmt-5m
* Quora question pair
* SNLI
* Semeval
* And more!

I have not included the dataset as part of this repo.  If you're curious and would like to know more, contact me.

### Training

Training was done for 2 epochs on a Nvidia GTX 1080 and evaluted on the BLEU score. The Tensorboard training curves can be seen below

<img src="https://raw.githubusercontent.com/vsuthichai/paraphraser/master/images/20180128-035256-plot.png" align="center">


