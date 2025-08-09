import numpy as np
import pandas as pd
import sklearn.decomposition as dec 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
import os

# اطمینان از وجود پوشه images
if not os.path.exists('images'):
    os.makedirs('images')

df = pd.read_csv('datasetprojmabahes.csv')

for i in range(1, len(df)):
    if pd.isna(df.loc[i, 'FlightNumber']):
        df.loc[i, 'FlightNumber'] = df.loc[i-1, 'FlightNumber'] + 10

df['FlightNumber'] = df['FlightNumber'].astype(int)

df['From_To'] = df['From_To'].str.replace('_', ' ')
df['From'] = df['From_To'].str.split(' ').str[0]
df['To'] = df['From_To'].str.split(' ').str[1]
df['From'] = df['From'].str.capitalize()
df['To'] = df['To'].str.capitalize()
df = df.drop('From_To', axis=1)

df['Airline'] = df['Airline'].str.replace('[^a-zA-Z ]', '', regex=True)

df['RecentDelays'] = df['RecentDelays'].apply(literal_eval)
delays = df['RecentDelays'].apply(pd.Series)
delays.columns = ['delay_{}'.format(n+1) for n in range(len(delays.columns))]
df = pd.concat([df.drop('RecentDelays', axis=1), delays], axis=1)

print(df.describe())

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

df[numeric_cols].hist(figsize=(12, 8), bins=15, edgecolor='black')
plt.suptitle('histogram of Numeric Columns', fontsize=16)
plt.tight_layout()
plt.savefig('images/histogram_numeric_columns.png')
plt.close()

delay_cols = [c for c in ['delay_1', 'delay_2', 'delay_3'] if c in df.columns]

df['TotalDelays'] = df[delay_cols].sum(axis=1)
df['NumDelays'] = df[delay_cols].notna().sum(axis=1)
df['AverageDelay'] = df[delay_cols].mean(axis=1)

data = df[['TotalDelays', 'NumDelays', 'AverageDelay', 'delay_1', 'delay_2', 'delay_3']]
data = data.fillna(0)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

pca = dec.PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio for each component:", explained_variance)

if delay_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[delay_cols], palette="Set2")
    plt.title('Boxplot of Delay Columns')
    plt.xticks(rotation=45)
    plt.savefig('images/boxplot_delay_columns.png')
    plt.close()

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance) + 1))
plt.savefig('images/scree_plot.png')
plt.close()

cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.savefig('images/cumulative_explained_variance.png')
plt.close()

plt.figure(figsize=(8, 6))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('images/correlation_heatmap.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['TotalDelays'], cmap='viridis', alpha=0.7)
plt.title('PCA: First Two Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Total Delays')
plt.savefig('images/pca_scatter_plot.png')
plt.close()
