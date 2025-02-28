import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load Dataset (Using a Sample Retail Dataset)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# Display first few rows
print("Dataset Preview:\n", df.head())

# Selecting Features for Clustering
features = df[['total_bill', 'tip', 'size']]  # Adjust as per dataset

# Data Preprocessing - Standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Finding the Optimal Number of Clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()

# Apply K-Means Clustering (Choosing K=3 from Elbow Method)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualizing Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['total_bill'], y=df['tip'], hue=df['Cluster'], palette='viridis', s=100)
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.title('Customer Segmentation Based on Spending Behavior')
plt.show()

# Interactive Plot (Optional)
fig = px.scatter(df, x='total_bill', y='tip', color=df['Cluster'].astype(str), 
                 title="Customer Clusters (Interactive)", size_max=10)
fig.show()

# Save clustered data
df.to_csv("customer_segments.csv", index=False)
print("Clustering complete! Results saved as 'customer_segments.csv'")
