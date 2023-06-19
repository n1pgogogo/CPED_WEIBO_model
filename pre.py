import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel

devices = torch.device("cuda:1")

# tokenizer = AutoTokenizer.from_pretrained("voidful/albert_chinese_base", cache_dir = "./.cache")
# model = AutoModel.from_pretrained("voidful/albert_chinese_base", cache_dir = "./.cache")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="../2_model_filter_info/.cache/models--voidful--albert_chinese_base/snapshots/84609bee8677d5b06e0c98adec78698e2d66f83f/")
model = AutoModel.from_pretrained(pretrained_model_name_or_path="../2_model_filter_info/.cache/models--voidful--albert_chinese_base/snapshots/84609bee8677d5b06e0c98adec78698e2d66f83f/")

# train_dataset = pd.read_csv("./data/train_split.csv", usecols=["Speaker", "Utterance",
#                                                                "Neuroticism", "Extraversion",
#                                                                "Openness", "Agreeableness",
                                                            #    "Conscientiousness"]).applymap(lambda x: 2 if x == "high" else (0 if x == "low" else x))
# valid_dataset = pd.read_csv("./data/valid_split.csv", usecols=["Speaker", "Utterance",
#                                                                "Neuroticism", "Extraversion",
#                                                                "Openness", "Agreeableness",
#                                                                "Conscientiousness"]).applymap(lambda x: 2 if x == "high" else (0 if x == "low" else x))
train_dataset = pd.read_csv("./data/test_split.csv", usecols=["Speaker", "Utterance",
                                                               "Neuroticism", "Extraversion",
                                                               "Openness", "Agreeableness",
                                                               "Conscientiousness"]).applymap(lambda x: 2 if x == "high" else (0 if x == "low" else x))

model.to(devices)

def st(word):
    token = tokenizer(word, return_tensors="pt", max_length=512, truncation = True, padding="max_length").to(devices)
    # token = tokenizer(word, return_tensors="pt", max_length=512, truncation = True, padding="max_length").to(devices)
    outputs = model(**token)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[:, 0, :]

with torch.no_grad():
    data = None
    label = None
    c = np.unique(train_dataset[train_dataset["Openness"] != "unknown"]["Speaker"])
    for i in c:
        u = 0
        t_data = None
        for j, row in train_dataset[train_dataset["Speaker"] == i].iterrows():
            t = st(row["Utterance"])
            u += 1
            if u == 1:
                t_data = t
            else:
                t_data = torch.maximum(t_data, t)

        print("{}/{}说做人，最重要的就是开心啦~~".format(i, len(np.unique(train_dataset["Speaker"]))))
        if data == None:
            data = t_data
            # O C A E N
            label = torch.from_numpy(np.array([[
                train_dataset[train_dataset["Speaker"] == i]["Openness"].iloc[0],
                train_dataset[train_dataset["Speaker"] == i]["Conscientiousness"].iloc[0],
                train_dataset[train_dataset["Speaker"] == i]["Agreeableness"].iloc[0],
                train_dataset[train_dataset["Speaker"] == i]["Extraversion"].iloc[0],
                train_dataset[train_dataset["Speaker"] == i]["Neuroticism"].iloc[0]]], dtype=np.uint8))
        else:
            data = torch.concat([data, t_data], axis = 0)
            label = torch.concat([
                torch.from_numpy(np.array([[
                    train_dataset[train_dataset["Speaker"] == i]["Openness"].iloc[0],
                    train_dataset[train_dataset["Speaker"] == i]["Conscientiousness"].iloc[0],
                    train_dataset[train_dataset["Speaker"] == i]["Agreeableness"].iloc[0],
                    train_dataset[train_dataset["Speaker"] == i]["Extraversion"].iloc[0],
                    train_dataset[train_dataset["Speaker"] == i]["Neuroticism"].iloc[0]]], dtype=np.uint8)),
                label
            ], axis = 0)

    print(label.shape)
    print(data.shape)
    torch.save([data, label], "./data/test.pth")