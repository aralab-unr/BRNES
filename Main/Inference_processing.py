import pandas as pd
import numpy as np
import sys
import os


path = './Inference/'

folder = os.fsencode(path)

filenames = []
mainDF = pd.DataFrame()

for file in os.listdir(folder):
    filename = os.fsdecode(file)
    print("Processed files",filename)
    df = pd.read_csv(path+filename)
    mainDF = pd.concat([mainDF, df], axis=1)

mainDF.drop('Unnamed: 0', axis=1, inplace=True)

mainDF.to_csv('./ProcessedOutput/ProcessedInference.csv', index = False)
print("#################\nProcessing done [Generated file: ProcessedInference.csv]. Look into the ProcessedOutput folder")