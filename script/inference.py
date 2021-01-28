TEST = False
N = 1000
import pickle

model_path = "../model/"
data_path = "../data/"
train = pickle.load(open(data_path + "train.pkl", "rb"))
test = pickle.load(open(data_path + "test.pkl", "rb"))

if TEST:
    train.cleaned = train.cleaned[0:N]
    test.cleaned = test.cleaned[0:N]

print("Roberta Model")
roberta = pickle.load(open("roberta.pkl", "rb"))
train["roberta"], _ = roberta.predict(train.cleaned)
test["roberta"], _ = roberta.predict(test.cleaned)
del roberta

print("Roberta256 Model")
roberta256 = pickle.load(open("roberta256.pkl", "rb"))
train["roberta256"], _ = roberta256.predict(train.cleaned)
test["roberta256"], _ = roberta256.predict(test.cleaned)
del roberta256

print("Bert Model")
bert = pickle.load(open("bert.pkl", "rb"))
train["bert"], _ = bert.predict(train.cleaned)
test["bert"], _ = bert.predict(test.cleaned)
del bert

print("Bert256 Model")
bert256 = pickle.load(open("bert256.pkl", "rb"))
train["bert256"], _ = bert256.predict(train.cleaned)
test["bert256"], _ = bert256.predict(test.cleaned)
del bert256

print("Xlnet Model")
xlnet = pickle.load(open("xlnet.pkl", "rb"))
train["xlnet"], _ = xlnet.predict(train.cleaned)
test["xlnet"], _ = xlnet.predict(test.cleaned)
del xlnet

print("Xlnet256 Model")
xlnet256 = pickle.load(open("xlnet256.pkl", "rb"))
train["xlnet256"], _ = xlnet256.predict(train.cleaned)
test["xlnet256"], _ = xlnet256.predict(test.cleaned)
del xlnet256

print("Saving file.")
pickle.dump(train, open("train.pkl", "wb"))
pickle.dump(test, open("test.pkl", "wb"))
print("Done.")