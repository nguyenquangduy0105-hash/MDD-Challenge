import pandas as pd
import pickle

train = pd.read_csv("Path_to_train_and_dev")
test = pd.read_csv("Path_to_test")

train.to_pickle("train.pkl")
test.to_pickle("test.pkl")