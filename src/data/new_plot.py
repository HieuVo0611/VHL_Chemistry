import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# df_meta là DataFrame chứa cột 'Pb' và 'Cd'
df_meta = pd.read_csv(r"E:\VHL Project\Chemistry\VHL_Chemistrygit\VHL_Chemistrygit\data\processed\metadata_pb_cd.csv")
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_meta, x='Pb', y='Cd')
plt.title("Mối quan hệ giữa Pb và Cd")
plt.xlabel("Nồng độ Pb")
plt.ylabel("Nồng độ Cd")
plt.grid(True)
plt.show()