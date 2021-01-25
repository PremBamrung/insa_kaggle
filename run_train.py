TEST = True
SAMPLE = 1000
print("Import")
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle 
from datetime import datetime
from transformers import logging
logging.set_verbosity_warning()

print("Load Data")
# train = pd.read_csv("train.csv").dropna()
# test = pd.read_csv("test.csv").dropna()

train=pickle.load(open("train.pkl","rb"))
test=pickle.load(open("test.pkl","rb"))


train.category = train.category.astype(int)

xtrain, xtest = train_test_split(
    train, test_size=0.33, random_state=42, stratify=train.category)
train_df = pd.DataFrame()
train_df["text"] = xtrain.cleaned
train_df["labels"] = xtrain.category


eval_df = pd.DataFrame()
eval_df["text"] = xtest.cleaned
eval_df["labels"] = xtest.category


if TEST:
  eval_df = eval_df[0:SAMPLE]
  train_df = eval_df[0:SAMPLE]


from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

print("Defining model")

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=2,
                                no_save=True,
                                overwrite_output_dir=True,
                                save_eval_checkpoints=False,
                                save_model_every_epoch=False,
                                save_optimizer_and_scheduler=False,
                                max_seq_length=256,
                                fp16=True,
                                train_batch_size=16,
                                eval_batch_size=16)

# Create a ClassificationModel
model = ClassificationModel(
    'roberta',
    'roberta-base',
    num_labels=len(eval_df.labels.unique()),
    args=model_args
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
now=datetime.now()
model_name=f"model_{now.strftime('%y_%m_%d_%H%M')}.pkl"

pickle.dump( model, open(model_name, "wb" ) )
print("Creating Kaggle Submission")

submissions = pd.read_csv("template_submissions.csv")

if TEST:
  test = test[0:SAMPLE]
  submissions = submissions[0:SAMPLE]

test_predictions, raw_outputs = model.predict(test.cleaned.values)


submissions.Category = test_predictions
submissions.to_csv('submissions.csv', index=False)

print("Done.")
