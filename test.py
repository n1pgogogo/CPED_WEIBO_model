import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os, sys
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

device = torch.device("cuda:0")

class TCModel(nn.Module):
    def __init__(self, embedding_dimension, num_labels=5, num_classes=3):
        super(TCModel, self).__init__()
        self.num_classes = num_classes
        n = 30
        
        self.dense1 = nn.Linear(embedding_dimension, n)
        self.dense2 = nn.Linear(embedding_dimension, n)
        self.dense3 = nn.Linear(embedding_dimension, n)
        self.dense4 = nn.Linear(embedding_dimension, n)
        self.dense5 = nn.Linear(embedding_dimension, n)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(n)
        self.bn2 = nn.BatchNorm1d(n)
        self.bn3 = nn.BatchNorm1d(n)
        self.bn4 = nn.BatchNorm1d(n)
        self.bn5 = nn.BatchNorm1d(n)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.output_dense1 = nn.Linear(n, num_classes)
        self.output_dense2 = nn.Linear(n, num_classes)
        self.output_dense3 = nn.Linear(n, num_classes)
        self.output_dense4 = nn.Linear(n, num_classes)
        self.output_dense5 = nn.Linear(n, num_classes)

    def forward(self, x):
        x1 = self.dense1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)
        outputs1 = F.softmax(self.output_dense1(x1), dim = 1)

        x2 = self.dense2(x)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)
        outputs2 = F.softmax(self.output_dense2(x2), dim = 1)

        x3 = self.dense3(x)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        x3 = self.dropout3(x3)
        outputs3 = F.softmax(self.output_dense3(x3), dim = 1)

        x4 = self.dense4(x)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)
        x4 = self.dropout4(x4)
        outputs4 = F.softmax(self.output_dense4(x4), dim = 1)

        x5 = self.dense5(x)
        x5 = self.bn5(x5)
        x5 = self.relu5(x5)
        x5 = self.dropout5(x5)
        outputs5 = F.softmax(self.output_dense5(x5), dim = 1)
        return outputs1, outputs2, outputs3, outputs4, outputs5
    
embedding_dimension = 768

model_cnn = TCModel(embedding_dimension).to(device)
model_cnn.load_state_dict(torch.load("./model/best_model.pt", map_location=device))
optimizer = torch.optim.SGD(model_cnn.parameters(), lr = 1e-3, momentum=0.9)
criterion = nn.CrossEntropyLoss()

d_data, d_label = torch.load("data/weibo_200_gpt.pth", map_location=device)
train, val, label, val_label = train_test_split(d_data, d_label, test_size=0.3, random_state=55)

model_cnn.eval()
with torch.no_grad():
    for i in range(1):
        output1, output2, output3, output4, output5 = model_cnn(val)

        print("N 精确率: ", precision_score(val_label[:, 4].to("cpu"), torch.max(output5, dim = 1)[1].to("cpu"), average='weighted'))
        print("N 召回率: ", recall_score(val_label[:, 4].to("cpu"), torch.max(output5, dim = 1)[1].to("cpu"), average='weighted'))
        print("N F1值: ", f1_score(val_label[:, 4].to("cpu"), torch.max(output5, dim = 1)[1].to("cpu"), average='weighted'))
        print("N 准确率: ", accuracy_score(val_label[:, 4].to("cpu"), torch.max(output5, dim = 1)[1].to("cpu")))

        print("E 精确率: ", precision_score(val_label[:, 3].to("cpu"), torch.max(output4, dim = 1)[1].to("cpu"), average='weighted'))
        print("E 召回率: ", recall_score(val_label[:, 3].to("cpu"), torch.max(output4, dim = 1)[1].to("cpu"), average='weighted'))
        print("E F1值: ", f1_score(val_label[:, 3].to("cpu"), torch.max(output4, dim = 1)[1].to("cpu"), average='weighted'))
        print("E 准确率: ", accuracy_score(val_label[:, 3].to("cpu"), torch.max(output4, dim = 1)[1].to("cpu")))

        print("O 精确率: ", precision_score(val_label[:, 0].to("cpu"), torch.max(output1, dim = 1)[1].to("cpu"), average='weighted'))
        print("O 召回率: ", recall_score(val_label[:, 0].to("cpu"), torch.max(output1, dim = 1)[1].to("cpu"), average='weighted'))
        print("O F1值: ", f1_score(val_label[:, 0].to("cpu"), torch.max(output1, dim = 1)[1].to("cpu"), average='weighted'))
        print("O 准确率: ", accuracy_score(val_label[:, 0].to("cpu"), torch.max(output1, dim = 1)[1].to("cpu")))

        print("A 精确率: ", precision_score(val_label[:, 2].to("cpu"), torch.max(output3, dim = 1)[1].to("cpu"), average='weighted'))
        print("A 召回率: ", recall_score(val_label[:, 2].to("cpu"), torch.max(output3, dim = 1)[1].to("cpu"), average='weighted'))
        print("A F1值: ", f1_score(val_label[:, 2].to("cpu"), torch.max(output3, dim = 1)[1].to("cpu"), average='weighted'))
        print("A 准确率: ", accuracy_score(val_label[:, 2].to("cpu"), torch.max(output3, dim = 1)[1].to("cpu")))

        print("C 精确率: ", precision_score(val_label[:, 1].to("cpu"), torch.max(output2, dim = 1)[1].to("cpu"), average='weighted'))
        print("C 召回率: ", recall_score(val_label[:, 1].to("cpu"), torch.max(output2, dim = 1)[1].to("cpu"), average='weighted'))
        print("C F1值: ", f1_score(val_label[:, 1].to("cpu"), torch.max(output2, dim = 1)[1].to("cpu"), average='weighted'))
        print("C 准确率: ", accuracy_score(val_label[:, 1].to("cpu"), torch.max(output2, dim = 1)[1].to("cpu")))

        print(torch.max(output1, dim = 1)[1])
        # print("N", np.mean(torch.eq(torch.max(output5, dim = 1)[1], val_label[:, 4]).to("cpu").numpy().astype(int)))
        # print("E", np.mean(torch.eq(torch.max(output4, dim = 1)[1], val_label[:, 3]).to("cpu").numpy().astype(int)))
        # print("O", np.mean(torch.eq(torch.max(output1, dim = 1)[1], val_label[:, 0]).to("cpu").numpy().astype(int)))
        # print("A", np.mean(torch.eq(torch.max(output3, dim = 1)[1], val_label[:, 2]).to("cpu").numpy().astype(int)))
        # print("C", np.mean(torch.eq(torch.max(output2, dim = 1)[1], val_label[:, 1]).to("cpu").numpy().astype(int)))