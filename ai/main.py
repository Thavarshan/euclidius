from nltk import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import re
import nltk
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.nn import functional as F
from torchtext import data
from collections import Counter
import torch
from torch import nn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import spacy
import pandas as pd
import numpy as np
import random
import os
import dill

SEED = 32
random.seed(SEED)


# For Text processing
nltk.download('punkt')
nltk.download('stopwords')
nlp = English()
tokenizer = Tokenizer(nlp.vocab)
stops = stopwords.words('english')
snowstem = SnowballStemmer('english')
portstem = PorterStemmer()


class DataFrameDataset(data.Dataset):
    def __init__(self, df, text_field, label_field, is_test=False, **kwargs):
        fields = [('comment_text', text_field), ('toxic', label_field)]
        examples = []
        for i, row in df.iterrows():
            if row['class'] == 2:
                label = 0
            else:
                label = 1
            text = row['tweet']
            examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(examples, fields, **kwargs)


def deviceSelect():
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('running on CUDA gpu 0')
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device = 'mps'
    #     print('running on m1 gpu')
    else:
        device = 'cpu'
        print('running on cpu')
    return torch.device(device)


def loadData(path, test_size=0.2):
    labeledDf = pd.read_csv(path)
    print(labeledDf.head())
    return train_test_split(labeledDf, test_size=test_size)


def removepunc(my_str):  # function to remove punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def myTokenizer(x):
    return [snowstem.stem(word.text) for word in
            tokenizer(removepunc(re.sub(
                r"\s+\s+", " ", re.sub(r"[^A-Za-z0-9()!?\'\`\"\r+\n+]", " ", x.lower()))).strip())
            if (word.text not in stops and not hasNumbers(word.text))]


def processText():
    TEXT = data.Field(tokenize=myTokenizer, batch_first=True, fix_length=140)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)
    trainData, testData = loadData('ai/data/tweets.csv')
    torchDataset = DataFrameDataset(trainData, TEXT, LABEL)
    torchTest = DataFrameDataset(testData, TEXT, LABEL)
    train_data, valid_data = torchDataset.split(
        split_ratio=0.8,
        random_state=random.seed(SEED)
    )
    TEXT.build_vocab(train_data, min_freq=3)
    LABEL.build_vocab(train_data)
    # No. of unique tokens in text
    print('Size of TEXT vocabulary:', len(TEXT.vocab))
    # No. of unique tokens in label
    print('Size of LABEL vocabulary:', len(LABEL.vocab))
    # Commonly used words
    print(TEXT.vocab.freqs.most_common(10))
    return train_data, valid_data, torchTest, (TEXT, LABEL)


class TextTransformer(nn.Module):
    def __init__(self, vocabSize, device):
        super(TextTransformer, self).__init__()
        self.wordEmbeddings = nn.Embedding(vocabSize, 140)
        self.positionEmbeddings = nn.Embedding(140, 20)
        self.transformerLayer = nn.TransformerEncoderLayer(160, 8)
        self.linear1 = nn.Linear(160,  64)
        self.linear2 = nn.Linear(64,  1)
        self.linear3 = nn.Linear(140,  16)
        self.linear4 = nn.Linear(16,  1)
        self.device = device

    def forward(self, x):
        positions = (torch.arange(0, 140).reshape(1, 140) +
                     torch.zeros(x.shape[0], 140)).to(self.device)
        # broadcasting the tensor of positions
        sentence = torch.cat((self.wordEmbeddings(
            x.long()), self.positionEmbeddings(positions.long())), axis=2)
        attended = self.transformerLayer(sentence)
        linear1 = F.relu(self.linear1(attended))
        linear2 = F.relu(self.linear2(linear1))
        # reshaping the layer as the transformer outputs a 2d tensor (or 3d considering the batch size)
        linear2 = linear2.view(-1, 140)
        linear3 = F.relu(self.linear3(linear2))
        out = torch.sigmoid(self.linear4(linear3))
        return out


def calculateMetrics(ypred, ytrue):
    acc = accuracy_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred)
    f1_average = f1_score(ytrue, ypred, average='macro')
    return 'f1 score: ' + str(round(f1, 3)) + ' f1 average: ' + str(round(f1_average, 3)) + ' accuracy: ' + str(round(acc, 3))


def saveVocab(vocab, file):
    path = Path(file).parent.absolute().mkdir(parents=True, exist_ok=True)
    print('Saving vocab file ' + vocab.__str__() + ' --> ' + file)
    torch.save(vocab, file, pickle_module=dill)


def loadVocab(file):
    if os.path.isfile(file):
        print('Reading vocab file: ' + file)
        return torch.load(file, pickle_module=dill)
    else:
        print('Error reading file: ' + file + '. file do not exist.')


def saveModel(model, file):
    path = Path(file).parent.absolute().mkdir(parents=True, exist_ok=True)
    print('Saving vocab file ' + model.__str__() + ' --> ' + file)
    torch.save(model.state_dict(), file, pickle_module=dill)


def loadModel(file, vocabSize, device):
    if os.path.isfile(file):
        print('Reading model file: ' + file)
        model = TextTransformer(vocabSize, device)
        model.load_state_dict(torch.load(
            file, map_location=torch.device(device), pickle_module=dill))
        return model
    else:
        print('Error reading file: ' + file + '. file do not exist.')


def inference(model, vocab, inString, device):
    model.eval()  # switching to evaluation mode.
    model.to(device)
    x = vocab.process([inString])
    x = x.to(device)
    pred = model(x).squeeze().cpu().detach()
    print(pred)
    return torch.round(pred).numpy().item()


def trainModel(device, episodeCount=20, batchSize=128):
    train_data, valid_data, test_data, (TEXT, LABEL) = processText()
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batchSize,
        device=device,
        sort=False,
        shuffle=False)
    myTransformer = TextTransformer(len(TEXT.vocab), device)
    myTransformer.to(device)
    optimizer = optim.Adagrad(myTransformer.parameters(), lr=0.001)
    for i in range(episodeCount):
        trainpreds = torch.tensor([])
        traintrues = torch.tensor([])
        for batch in train_iterator:
            X = batch.comment_text
            y = batch.toxic
            myTransformer.zero_grad()
            pred = myTransformer(X).squeeze()
            trainpreds = torch.cat((trainpreds, pred.cpu().detach()))
            traintrues = torch.cat((traintrues, y.cpu().detach()))
            err = F.binary_cross_entropy(pred, y)
            err.backward()
            optimizer.step()
        err = F.binary_cross_entropy(trainpreds, traintrues)
        print('train BCE loss: ', err.item(), calculateMetrics(
            torch.round(trainpreds).numpy(), traintrues.numpy()))
        valpreds = torch.tensor([])
        valtrues = torch.tensor([])
        for batch in valid_iterator:
            X = batch.comment_text
            y = batch.toxic
            valtrues = torch.cat((valtrues, y.cpu().detach()))
            pred = myTransformer(X).squeeze().cpu().detach()
            # print(valtrues.shape)
            valpreds = torch.cat((valpreds, pred))
        err = F.binary_cross_entropy(valpreds, valtrues)
        print('validation BCE loss: ', err.item(), calculateMetrics(
            torch.round(valpreds).numpy(), valtrues.numpy()))
    return myTransformer, TEXT


# if __name__ == '__main__':
    # device = deviceSelect()
    # # needed for training and saving the models.
    # model, TEXT = trainModel(device, 5)
    # saveVocab(TEXT, 'ai/vocab/TEXT_obj.pth')
    # saveModel(model, 'ai/model/textTransformer_states.pth')
    # # inference
    # TEXT = loadVocab('ai/vocab/TEXT_obj_kaggle_trained_2.pth')
    # vocabSize = len(TEXT.vocab)
    # model = loadModel(
    #     'ai/model/textTransformer_states_kaggle_trained_2.pth', vocabSize, device)
    # print(inference(model, TEXT, 'The shit just blows me..claim you so faithful and down for somebody but still fucking with hoes!', device))
    # print(inference(model, TEXT, "!!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out...", device))
    # print(inference(model, TEXT, "!!!!!!!!!!!!! RT @ShenikaRoberts: The shit you hear about me might be true or it might be faker than the bitch who told it to ya &#57361;", device))
