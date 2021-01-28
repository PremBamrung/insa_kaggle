# <h1 align="center">Welcome to our [<img src="https://kaggle.com/static/images/site-logo.png" height="50" style="margin-bottom:-15px" />](https://kaggle.com) repo 👋</h1>

# Table of content
- [<h1 align="center">Welcome to our <img src="https://kaggle.com/static/images/site-logo.png" height="50" style="margin-bottom:-15px" /> repo 👋</h1>](#h1-aligncenterwelcome-to-our--repo-h1)
- [Table of content](#table-of-content)
- [Description](#description)
      - [Data](#data)
      - [Evaluation](#evaluation)
- [Results](#results)
- [Computing Resources](#computing-resources)
      - [Runtime](#runtime)
- [Reproductibility](#reproductibility)
      - [Environment](#environment)
      - [Downloading data](#downloading-data)
      - [Training](#training)
          - [Single model](#single-model)
          - [Multiple model](#multiple-model)
      - [Prediction](#prediction)
- [Author](#author)

# Description
<h1 align="center"><img src="images/logos-defi-insa.svg" height="500"></h1>
This is our repository for the  5th edition of the so-called Défi IA.

This edition of the Défi IA pertains to NLP. The task is straightforward: assign the correct job category to a job description. This is thus a **multi-class classification** task with 28 classes to choose from.

#### Data
The data has been retrieved from CommonCrawl. The latter has been famously used to train OpenAI's GPT-3 model. The data is therefore representative of what can be found on the English speaking part of the Internet, and thus contains a certain amount of bias. One of the goals of this competition is to design a solution that is both accurate as well as fair.
The **train set** contains **217,197 sample** of job descriptions  as well as their labels and genders.
The **test set** contains **54,300 sample** of job descriptions as well as their genders. This is the set used for submissions.


#### Evaluation

The original aspect of this competition is that there will be 3 tracks on which solutions will be ranked. First of all, solutions are ranked according to the Macro F1 metric, which will be used to build the [Kaggle leaderboard](https://www.kaggle.com/c/defi-ia-insa-toulouse/leaderboard). From Scikit-learn documentation :

> The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.
************
#  Results
We ended at the `14th place` in the private leaderboard with a private score of `0.81026`.

Below are all of our submissions from latest to first submission.


| N° | Submissions                                   | Private Score | Public Score |
|----|----------------------------------------------:|--------------:|-------------:|
| 10 | Roberta Bert Xlnet 2 epochs voting lenght 256 | **0.81026**   | 0.80776      |
| 9  | Roberta Bert Xlnet 2 epochs voting            | 0.80636       | 0.80179      |
| 8  | Roberta base 2 epochs first phrase 20 characs | 0.69292       | 0.68064      |
| 7  | Roberta base 2 epochs clean text              | 0.79886       | 0.79813      |
| 6  | Roberta Large 3 epochs                        | 0.77888       | 0.78147      |
| 5  | Roberta Base 5 epochs                         | 0.79442       | 0.79318      |
| 4  | xlnet-base-cased 2 epochs                     | 0.79953       | 0.79171      |
| 3  | Roberta-base 2 epochs                         | 0.79903       | 0.79789      |
| 2  | Bert-base-cased 2 epochs                      | 0.79475       | 0.79647      |
| 1  | ULMFIT                                        | 0.78005       | 0.77978      |






*********
# Computing Resources


Prototyping was done on our `local desktop` or using free cloud resources such as `Google Colab` or `Kaggle Kernel`.

Local desktop specs:
- CPU :  i7 4770K 8 threads
- GPU : Nvidia GTX 1060 6 Go
- Ram : 16 Go
- OS  : Ubuntu 18.04

Complete training was done on Google Cloud Platform VM:
- CPU : Xeon 8 threads
- GPU : Nvidia Tesla V100 16 Go
- Ram : 30 Go
- OS  : Ubuntu 18.04

#### Runtime


**Training runtime per epoch using Roberta base:**


|         GPU | lenght=128 batch=16 | lenght=128 batch=32 | lenght=256 batch=16 | lenght=256 batch=32 |
|------------:|--------------------:|--------------------:|--------------------:|--------------------:|
|    GTX 1060 |            1h 23min |       Out of memory |       Out of memory |       Out of memory |
|    Tesla T4 |         26 min 40 s |         21 min 29 s |         45 min 52 s |          42 min 44s |
| Tesla  V100 |         14 min 35 s |          9 min 18 s |          20 min 2 s |         15 min 17 s |



**Preprocessing runtime by preprocessing task :**
| Nb Core | Cleaning |    Language | Complexity |        Distance |
|--------:|---------:|------------:|-----------:|----------------:|
|       1 |     30 s | 13 min 57 s | 2 min 19 s | 1 h 56 min 21 s |
|       2 |     16 s |  6 min 57 s |       55 s |         1h 34 s |
|       4 |      9 s |  3 min 45 s |       26 s |     29 min 54 s |
|       8 |      8 s |  3 min 16 s |       24 s |     20 min 44 s |


*********
# Reproductibility
#### Environment
In order to reproduce our result, you have to recreate our virtual environment. We recommend using `conda` to create an environment and to install the dependencies using the `requirements.txt` file.

```bash
conda create -n defi_ia python=3.7
conda activate defi_ia
pip install -r requirements.txt
```

#### Downloading data
You also  have to download all the data in the `/data` directory. A bash script is provided to do so. You have to execute the bash script inside the data directory.

```bash
cd data/
bash download_data.sh
```

#### Training
To train a single or multiple model at the same time, use the provided script inside the `/script` directory.

###### Single model
You can chose the model and its hyperparamters in the beginning of the script `run_single.py` :

```python
TEST = False
SAMPLE = 1000
EPOCH = 2
LENGTH = 128
BATCH = 16
FAMILY = "roberta"
FAMILYMODEL = "roberta-base"
```
**Where :**
- TEST : whether you just want to see if the whole script run (`True`) on a subset of the data (`SAMPLE`) or not
- EPOCH : number of epoch you want to train on
- LENGHT : lenght use in the transformers model, how many characters should the model take as input
- BATCH : batchsize use for training and testing
- FAMILY : type of model or architecture  you want to use (bert, roberta, xlnet, [see more](https://huggingface.co/transformers/pretrained_models.html))
- FAMILYMODEL : model id [see more](https://huggingface.co/transformers/pretrained_models.html)

Once you have set the parameters of the model you want to train, execute the script inside the script directory:

```bash
cd script/
python run_single.py
```

The trained model will be saved as a `pickle` file in the `/model` directory and can be used to do some inference later on.
The saving format is as follow :  `FAMILYMODEL_LENGTH_EPOCH.pkl`.
Ex: `roberta-base_32_2.pkl` for a roberta-base model train on a lenght of 32 for 2 epochs

###### Multiple model
Another script is provided to train multiple model at the same time. The same parameters can be found in the beginning of the script `run_multiple.py` except that you have to manually set the `FAMILY` and `FAMILYMODEL` inside the script :

```python
model = ClassificationModel(
    FAMILY, FAMILYMODEL, num_labels=len(eval_df.labels.unique()), args=model_args)
```
The models is saved the same way as for training a single model (see before).


#### Prediction
To run prediction from a trained model, import it with `pickle` and use the `predict` method on data. This methods return 2 outputs : label prediction, probabilities of each label.

```python
import pickle

model_path = "../model/"
data_path = "../data/"
test = pickle.load(open(data_path + "test.pkl", "rb"))

roberta = pickle.load(open(model_path+"roberta.pkl", "rb"))
test["roberta"], _ = roberta.predict(test.cleaned)
```
Change `roberta.pkl` into the name of your model. A few models are provided in the `/model` directory but you have to download them using the provided bash script :

```bash
cd model/
bash download_models.sh
```
*************
# Author
**Team 3TP** Mastère Spécialisé Valorisation des Données Massives (**VALDOM**)


<img src=images/logo_insa_toulouse.png height="30"/> <img src=images/logo_n7.png height="30"/>

👤 **Premchanok BAMRUNG**
👤 **Thibault DELMON**
👤 **Thibaut HERNANDEZ**
👤 **Thomas HUSTACHE**
**********