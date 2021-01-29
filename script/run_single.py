TEST = True
SAMPLE = 1000
EPOCH = 2
LENGTH = 128
BATCH = 16
FAMILY = "roberta"
FAMILYMODEL = "roberta-base"

print("Import")
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
from transformers import logging
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd

logging.set_verbosity_warning()

print("Load Data")

data_path = "../data/"


train = pickle.load(open(data_path + "train.pkl", "rb"))
test = pickle.load(open(data_path + "test.pkl", "rb"))


train.category = train.category.astype(int)

xtrain, xtest = train_test_split(
    train, test_size=0.33, random_state=42, stratify=train.category
)
train_df = pd.DataFrame()
train_df["text"] = xtrain.cleaned
train_df["labels"] = xtrain.category


eval_df = pd.DataFrame()
eval_df["text"] = xtest.cleaned
eval_df["labels"] = xtest.category


if TEST:
    eval_df = eval_df[0:SAMPLE]
    train_df = eval_df[0:SAMPLE]




print("Defining model")

# Optional model configuration
model_args = ClassificationArgs(
    num_train_epochs=EPOCH,
    no_save=True,
    overwrite_output_dir=True,
    save_eval_checkpoints=False,
    save_model_every_epoch=False,
    save_optimizer_and_scheduler=False,
    max_seq_length=LENGTH,
    fp16=True,
    train_batch_size=BATCH,
    eval_batch_size=BATCH,
)

# Create a ClassificationModel
model = ClassificationModel(
    FAMILY, FAMILYMODEL, num_labels=len(eval_df.labels.unique()), args=model_args
)
print("Model training")
# Train the model
model.train_model(train_df)

print("Metrics")
predictions, raw_outputs = model.predict(eval_df.text.values)
f1 = f1_score(eval_df.labels.values, predictions, average="weighted")
accuracy = accuracy_score(eval_df.labels.values, predictions)

print("F1 score : ", f1)
print("Accuracy score : ", accuracy)

print("Saving model")
model_path = "../model/"


pickle.dump(
    model,
    open(
        model_path + FAMILYMODEL + "_" + str(LENGTH) + "_" + str(EPOCH) + ".pkl", "wb"
    ),
)

print("Creating Kaggle Submission")

submissions = pd.read_csv(data_path + "template_submissions.csv")

if TEST:
    test = test[0:SAMPLE]
    submissions = submissions[0:SAMPLE]

test_predictions, raw_outputs = model.predict(test.cleaned.values)


submissions.Category = test_predictions
submissions.to_csv(data_path + "submissions.csv", index=False)

print("Done.")
