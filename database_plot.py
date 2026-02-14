import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('game_data.csv')

sns.pairplot(data, hue='skill_level')
plt.show()

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
