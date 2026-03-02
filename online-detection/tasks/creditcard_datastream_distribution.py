import ssl
import os
import matplotlib.pyplot as plt
import seaborn as sns
from river import datasets


if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


print("Загрузка данных из River...")
dataset = datasets.CreditCard()
times = []

for x, y in dataset:
    times.append(x['Time'])


fig, ax = plt.subplots(1, 2, figsize=(15, 6))


sns.histplot(times, bins=50, kde=True, ax=ax[0], color='royalblue')
ax[0].set_title('Distribution of Transactions (Histogram)')
ax[0].set_xlabel('Time (seconds from first transaction)')
ax[0].set_ylabel('Frequency')

# Violin Chart
sns.violinplot(x=times, ax=ax[1], color='lightseagreen')
ax[1].set_title('Distribution of Transactions (Violin Plot)')
ax[1].set_xlabel('Time (seconds)')

plt.tight_layout()
plt.show()