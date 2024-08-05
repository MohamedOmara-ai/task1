import pandas as pd
import numpy as np
import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('username/dataset-name')


data = pd.read_csv('/kaggle/input/dataset-name/your_data.csv')

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', etc.
data = imputer.fit_transform(data)

 import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
 import MinMaxScaler
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)