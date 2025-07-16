import pandas as pd

# Đường dẫn tới file train_results.csv
csv_path = "E:/VHL Project/Chemistry/VHL_Chemistrygit/VHL_Chemistrygit/data/processed/train_results.csv"

# Đọc file csv
df = pd.read_csv(csv_path)

# Sort theo cột Test_R2 giảm dần
df_sorted = df.sort_values(by="Test_R2", ascending=False)

# In ra top 10 model tốt nhất
print(df_sorted.head(10))

# Nếu muốn lưu lại file đã sort
df_sorted.to_csv("E:/VHL Project/Chemistry/VHL_Chemistrygit/VHL_Chemistrygit/data/processed/train_results_sorted.csv", index=False)