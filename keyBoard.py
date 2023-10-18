from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import joblib

TRAIN_TEXT_FILE_PATH = 'TrainingSet(Games).txt'

with open(TRAIN_TEXT_FILE_PATH) as text_file:
    text_sample = text_file.readlines()
text_sample = ' '.join(text_sample)


def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
    print(sorted_chars)
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])

    return sequence, char_to_idx, idx_to_char


sequence, char_to_idx, idx_to_char = text_to_seq(text_sample)


def evaluate(model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)

    inp = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char

    return predicted_text


class TextRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = TextRNN(input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
model.to(device)

model = joblib.load('TextRNN.joblib')
from pynput import keyboard
import os

counter = 0
length_of_pred = 30
Variable = ""
with keyboard.Events() as events:
    for event in events:
        if event.key == keyboard.Key.esc:
            break   #Выход из слушателя
        else:
            if ((counter % 2) == 0):
                os.system('clear')
                unsorted_char = str(event.key)
                char = unsorted_char[1]
                Variable = Variable + char
                print('Start: ',Variable)
                model.eval()

                print(evaluate(
                    model,
                    char_to_idx,
                    idx_to_char,
                    temp=0.3,
                    prediction_len=length_of_pred,
                    start_text=Variable
                )
                )
                length_of_pred += 1
            counter +=1