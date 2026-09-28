import pandas as pd

df = pd.read_excel("Sales_Revenue_Analysis_Dataset.xlsx")

customer_data = df.groupby("Customer_ID").agg(
    Total_Orders=("Order_ID", "nunique"),
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum")
).reset_index()

print(customer_data.head())
print("Number of customers:", len(customer_data))

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = customer_data[[
    "Total_Orders",
    "Total_Sales",
    "Total_Profit"
]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

customer_data["Cluster"] = kmeans.fit_predict(scaled_features)

print(customer_data.head(10))
cluster_summary = customer_data.groupby("Cluster").agg(
    Customers=("Customer_ID", "count"),
    Avg_Orders=("Total_Orders", "mean"),
    Avg_Sales=("Total_Sales", "mean"),
    Avg_Profit=("Total_Profit", "mean")
).round(2)

print("\nCluster Summary:")
print(cluster_summary)
customer_data.to_csv("customer_segments_new.csv", index=False)

print("\nCustomer segmentation saved successfully!")
import matplotlib.pyplot as plt

cluster_counts = customer_data["Cluster"].value_counts().sort_index()

plt.bar(cluster_counts.index.astype(str), cluster_counts.values)

plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.title("Customer Segmentation using K-Means")

plt.show()
cluster_sales = customer_data.groupby("Cluster")["Total_Sales"].mean()

plt.figure()

plt.bar(
    cluster_sales.index.astype(str),
    cluster_sales.values
)

plt.xlabel("Customer Cluster")
plt.ylabel("Average Sales")
plt.title("Average Sales by Customer Cluster")

plt.show()
plt.figure()

for cluster in sorted(customer_data["Cluster"].unique()):
    cluster_data = customer_data[customer_data["Cluster"] == cluster]

    plt.scatter(
        cluster_data["Total_Sales"],
        cluster_data["Total_Profit"],
        label=f"Cluster {cluster}"
    )

plt.xlabel("Total Sales")
plt.ylabel("Total Profit")
plt.title("Customer Segments: Sales vs Profit")
plt.legend()

plt.show()