# <h1 align="center">Welcome to our [<img src="https://kaggle.com/static/images/site-logo.png" height="50" style="margin-bottom:-15px" />](https://kaggle.com) repo 👋</h1>


# Table of content
  - [Description](#description)
    - [Data](#data)
    - [Evaluation](#evaluation)
  - [Results](#results)
  - [Computing Resources](#computing-resources)
  - [Reproductibility](#reproductibility)
  - [Usage](#usage)
  - [Run tests](#run-tests)
  - [Author](#author)
  - [Show your support](#show-your-support)

## Description
<h1 align="center"><img src="images/logos-defi-insa.svg" height="500"></h1>
This is our repository for the  5th edition of the so-called Défi IA.

This edition of the Défi IA pertains to NLP. The task is straightforward: assign the correct job category to a job description. This is thus a **multi-class classification** task with 28 classes to choose from.

### Data
The data has been retrieved from CommonCrawl. The latter has been famously used to train OpenAI's GPT-3 model. The data is therefore representative of what can be found on the English speaking part of the Internet, and thus contains a certain amount of bias. One of the goals of this competition is to design a solution that is both accurate as well as fair.
The **train set** contains **217,197 sample** of job descriptions  as well as their labels and genders.
The **test set** contains **54,300 sample** of job descriptions as well as their genders. This is the set used for submissions.


### Evaluation

The original aspect of this competition is that there will be 3 tracks on which solutions will be ranked. First of all, solutions are ranked according to the Macro F1 metric, which will be used to build the [Kaggle leaderboard](https://www.kaggle.com/c/defi-ia-insa-toulouse/leaderboard). The Macro F1 score is simply the arithmetic average of the F1 score for each class.


> carotte
> pomme de terre

##  Results
We ended at the `14th place` in the private leaderboard with a private score of `0.81026`.

Below are all of our submissions from latest to first submission.


| N° | Submissions                                   | Private Score | Public Score |
|----|-----------------------------------------------|---------------|--------------|
| 10 | Roberta Bert Xlnet 2 epochs voting lenght 256 | 0.81026       | 0.80776      |
| 9  | Roberta Bert Xlnet 2 epochs voting            | 0.80636       | 0.80179      |
| 8  | Roberta base 2 epochs first phrase 20 characs | 0.69292       | 0.68064      |
| 7  | Roberta base 2 epochs clean text              | 0.79886       | 0.79813      |
| 6  | Roberta Large 3 epochs                        | 0.77888       | 0.78147      |
| 5  | Roberta Base 5 epochs                         | 0.79442       | 0.79318      |
| 4  | xlnet-base-cased 2 epochs                     | 0.79953       | 0.79171      |
| 3  | Roberta-base 2 epochs                         | 0.79903       | 0.79789      |
| 2  | Bert-base-cased 2 epochs                      | 0.79475       | 0.79647      |
| 1  | ULMFIT                                        | 0.78005       | 0.77978      |



#### Runtime
Training runtime per epoch using Roberta base:



GPU | lenght=128 batch=16 | lenght=128 batch=32 | lenght=256 batch=16 | lenght=256 batch=32 |
|--------------|-----------|-----|---|---|
GTX 1060  | 1h 23min | Out of memory | Out of memory | Out of memory |
K80  | ...s | ...s | ...s | ...s |
V100  | ...s | ...s | ...s | ...s |



Preprocessing runtime by preprocessing task :
Nb Core | Cleaning | Language | Complexity | Distance |
|--------------|-----------|-----|---|---|
1| 30 s | 13 min 57 s | 2 min 19 s |1 h 56 min 21 s |
2  | 16 s | 6 min 57 s | 55 s |1h 34s |
4  | 9 s | 3 min 45 s| 26 s |29 min 54 s |
8  | 8 s | 3 min 16 s| 24 s |20 min 44 s |




## Computing Resources


Prototyping was done on our `local desktop` or using free cloud resources such as `Google Colab` or `Kaggle Kernel`.

Local desktop specs:
- CPU :  i7 4770K 8 threads
- GPU : Nvidia GTX 1060 6 Go
- Ram : 16 Go

Complete training was done on Google Cloud Platform VM:
- CPU : Xeon 8 threads
- GPU : Nvidia Tesla V100 16 Go
- Ram : 30 Go


## Reproductibility
#### Environment
In order to reproduce our result, you have to recreate our virtual environment. We recommend using `conda` to create an environment and to install the dependencies using the `requirements.txt` file.

```bash
conda create -n defi_ia python=3.7
conda activate defi_ia
pip install -r requirements.txt
```

#### Downloading data

#### Training
```python
import pickle
```

#### Prediction

```bash
python run
```
####

## Author
**Team 3TP** Mastère Spécialisé Valorisation des Données Massives (**VALDOM**)


<img src=images/logo_insa_toulouse.png height="30"/> <img src=images/logo_n7.png height="30"/>

👤 **Premchanok BAMRUNG**
👤 **Thibault DELMON**
👤 **Thibaut HERNANDEZ**
👤 **Thomas HUSTACHE**



## Show your support

Give a ⭐️ if this project helped you!
