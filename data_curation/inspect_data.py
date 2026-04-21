import pandas as pd

# Load datasets
df1 = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/raw/20191119_rawdata.csv")
df2 = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/raw/20200201_rawdata.csv")
mp = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/raw/materials_project.csv")

print("==== DATASET 1 ====")
print(df1.shape)
print(df1.columns)
print(df1.head())

print("\n==== DATASET 2 ====")
print(df2.shape)
print(df2.columns)
print(df2.head())

print("\n==== MATERIALS PROJECT ====")
print(mp.shape)
print(mp.columns)
print(mp.head())
