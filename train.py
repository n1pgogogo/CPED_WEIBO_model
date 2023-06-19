import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os, sys
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda:0")

class TCModel(nn.Module):
    def __init__(self, embedding_dimension, num_labels=5, num_classes=3):
        super(TCModel, self).__init__()
        self.num_classes = num_classes
        
        self.dense1 = nn.Linear(embedding_dimension, 30)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(30)
        self.relu = nn.ReLU()
        
        self.output_dense1 = nn.Linear(30, num_classes)
        self.output_dense2 = nn.Linear(30, num_classes)
        self.output_dense3 = nn.Linear(30, num_classes)
        self.output_dense4 = nn.Linear(30, num_classes)
        self.output_dense5 = nn.Linear(30, num_classes)

    def forward(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x) # 30

        outputs1 = F.softmax(self.output_dense1(x), dim = 1)
        outputs2 = F.softmax(self.output_dense2(x), dim = 1)
        outputs3 = F.softmax(self.output_dense3(x), dim = 1)
        outputs4 = F.softmax(self.output_dense4(x), dim = 1)
        outputs5 = F.softmax(self.output_dense5(x), dim = 1)
        return outputs1, outputs2, outputs3, outputs4, outputs5

embedding_dimension = 768

model_cnn = TCModel(embedding_dimension).to(device)
# model_cnn.load_state_dict(torch.load("../6_cped_test_model/model/best_model.pt", map_location=device))
optimizer = torch.optim.Adam(model_cnn.parameters(), lr = 1e-2)
# optimizer = torch.optim.SGD(model_cnn.parameters(), lr = 1e-3, momentum=0.9)

def compute_class_weights(labels):
    _, counts = torch.unique(labels, return_counts=True)
    total_samples = labels.size(0)
    class_weights = torch.FloatTensor([total_samples / count for count in counts])
    return class_weights.to(device)

d_data, d_label = torch.load("data/weibo_200_gpt.pth", map_location=device)
# val, val_label = torch.load("data/trains.pth", map_location=device)
train, val, label, val_label = train_test_split(d_data, d_label, test_size=0.3, random_state=55)

criterion1 = nn.CrossEntropyLoss(weight=torch.tensor([0.98, 0.87, 0.14], dtype=torch.float32).to(device))
criterion2 = nn.CrossEntropyLoss(weight=torch.tensor([0.88, 0.60, 0.52], dtype=torch.float32).to(device))
criterion3 = nn.CrossEntropyLoss(weight=torch.tensor([0.89, 0.84, 0.27], dtype=torch.float32).to(device))
criterion4 = nn.CrossEntropyLoss(weight=torch.tensor([0.96, 0.82, 0.22], dtype=torch.float32).to(device))
criterion5 = nn.CrossEntropyLoss(weight=torch.tensor([0.82, 0.82, 0.35], dtype=torch.float32).to(device))

# Initialize variables for early stopping and model saving
best_dev_loss = np.inf
early_stop_counter = 0
early_stop_patience = 99
# Initialize lists to keep track of training and validation loss
train_loss_history = []
dev_loss_history = []

epochs = 99999
for epoch in range(epochs):
    model_cnn.train()
    
    train_loss = 0.0
    train_steps = 0
    for i in range(1):
        optimizer.zero_grad()
        output1, output2, output3, output4, output5 = model_cnn(train)

        loss1 = criterion1(output1, label[:, 0])
        loss2 = criterion2(output2, label[:, 1])
        loss3 = criterion3(output3, label[:, 2])
        loss4 = criterion4(output4, label[:, 3])
        loss5 = criterion5(output5, label[:, 4])

        total_loss = loss1 + loss2 + loss3 + loss4 + loss5
        total_loss.backward()

        # noise_scale = 0.01
        # for param in model_cnn.parameters():
            # param.grad += noise_scale * torch.randn_like(param.grad)

        optimizer.step()

        train_loss += total_loss.item()
        train_steps += 1

        print("Train N", np.mean(torch.eq(torch.max(output5, dim = 1)[1], label[:, 4]).to("cpu").numpy().astype(int)))
        print("Train E", np.mean(torch.eq(torch.max(output4, dim = 1)[1], label[:, 3]).to("cpu").numpy().astype(int)))
        print("Train O", np.mean(torch.eq(torch.max(output1, dim = 1)[1], label[:, 0]).to("cpu").numpy().astype(int)))
        print("Train A", np.mean(torch.eq(torch.max(output3, dim = 1)[1], label[:, 2]).to("cpu").numpy().astype(int)))
        print("Train C", np.mean(torch.eq(torch.max(output2, dim = 1)[1], label[:, 1]).to("cpu").numpy().astype(int)))

    train_loss /= train_steps
    train_loss_history.append(train_loss)

    model_cnn.eval()
    with torch.no_grad():
        # Initialize variables to keep track of validation loss
        dev_loss = 0.0
        dev_steps = 0
        N = 0
        E = 0
        O = 0
        A = 0
        C = 0

        for i in range(1):
            output1, output2, output3, output4, output5 = model_cnn(val)

            loss1 = criterion1(output1, val_label[:, 0])
            loss2 = criterion2(output2, val_label[:, 1])
            loss3 = criterion3(output3, val_label[:, 2])
            loss4 = criterion4(output4, val_label[:, 3])
            loss5 = criterion5(output5, val_label[:, 4])

            total_loss = loss1 + loss2 + loss3 + loss4 + loss5
            dev_loss += total_loss.item()
            dev_steps += 1

            print("N", torch.max(output5, dim = 1)[1])
            print("E", torch.max(output4, dim = 1)[1])
            print("O", torch.max(output1, dim = 1)[1])
            print("A", torch.max(output3, dim = 1)[1])
            print("C", torch.max(output2, dim = 1)[1])
            N = np.mean(torch.eq(torch.max(output5, dim = 1)[1], val_label[:, 4]).to("cpu").numpy().astype(int))
            E = np.mean(torch.eq(torch.max(output4, dim = 1)[1], val_label[:, 3]).to("cpu").numpy().astype(int))
            O = np.mean(torch.eq(torch.max(output1, dim = 1)[1], val_label[:, 0]).to("cpu").numpy().astype(int))
            A = np.mean(torch.eq(torch.max(output3, dim = 1)[1], val_label[:, 2]).to("cpu").numpy().astype(int))
            C = np.mean(torch.eq(torch.max(output2, dim = 1)[1], val_label[:, 1]).to("cpu").numpy().astype(int))

        dev_loss /= dev_steps
        dev_loss_history.append(dev_loss)

        print(f'train_loss={train_loss:.4f}, dev_loss={dev_loss:.4f}')


        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            early_stop_counter = 0
            with open("README.md", "w", encoding="utf-8") as f:
                f.write(f"N: {N} E: {E} O: {O} A: {A} C: {C}")
            torch.save(model_cnn.state_dict(), "model/best_model.pt")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                break

plt.plot([i for i in range(len(train_loss_history))], train_loss_history, color = "r")
plt.plot([i for i in range(len(dev_loss_history))], dev_loss_history, color = "b")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["Loss", "Validation Loss"])
plt.savefig("./loss.jpg")
