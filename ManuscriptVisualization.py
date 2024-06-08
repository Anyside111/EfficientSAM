import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'iou_dice_score_madsam_tune.csv'
data = pd.read_csv(file_path)

# Filter data for 'EfficientSAM (VIT-tiny)' and 'EfficientSAM (VIT-small)'
vit_tiny_data = data[data['Model'] == 'EfficientSAM (VIT-tiny)'].copy()
vit_small_data = data[data['Model'] == 'EfficientSAM (VIT-small)'].copy()

# Extract lung number from 'Image Sticker'
vit_tiny_data.loc[:, 'Lung Number'] = vit_tiny_data['Image Sticker'].str.extract(r'lung_(\d+)')
vit_small_data.loc[:, 'Lung Number'] = vit_small_data['Image Sticker'].str.extract(r'lung_(\d+)')

# Calculate mean IoU for each lung number and sort by decreasing order
vit_tiny_data['Lung Number'] = vit_tiny_data['Lung Number'].astype(int)
vit_small_data['Lung Number'] = vit_small_data['Lung Number'].astype(int)

tiny_means = vit_tiny_data.groupby('Lung Number')['IoU'].mean().sort_values(ascending=False)
small_means = vit_small_data.groupby('Lung Number')['IoU'].mean().sort_values(ascending=False)

vit_tiny_data['Lung Number'] = pd.Categorical(vit_tiny_data['Lung Number'], categories=tiny_means.index, ordered=True)
vit_small_data['Lung Number'] = pd.Categorical(vit_small_data['Lung Number'], categories=small_means.index, ordered=True)

# Calculate mean and standard deviation
vit_tiny_mean = vit_tiny_data['IoU'].mean()
vit_tiny_std = vit_tiny_data['IoU'].std()
vit_small_mean = vit_small_data['IoU'].mean()
vit_small_std = vit_small_data['IoU'].std()

# Create subplots for VIT-tiny and VIT-small
fig, axs = plt.subplots(2, 1, figsize=(20, 16))

# VIT-tiny subplot
sns.violinplot(ax=axs[0], x='Lung Number', y='IoU', data=vit_tiny_data)
axs[0].set_title('Violin Plot of IoU for Each Lung (EfficientSAM VIT-tiny)')
axs[0].axhline(y=vit_tiny_mean, color='r', linestyle='--', label=f'Mean IoU = {vit_tiny_mean:.2f} ± {vit_tiny_std:.2f}')
axs[0].legend()

# VIT-small subplot
sns.violinplot(ax=axs[1], x='Lung Number', y='IoU', data=vit_small_data)
axs[1].set_title('Violin Plot of IoU for Each Lung (EfficientSAM VIT-small)')
axs[1].axhline(y=vit_small_mean, color='r', linestyle='--', label=f'Mean IoU = {vit_small_mean:.2f} ± {vit_small_std:.2f}')
axs[1].legend()

# Adjust layout
plt.tight_layout()

# Save the combined plot
plt.savefig('iou_violin_plot_vit_tiny_and_small_mean_std_sorted.png')

# Show the plot
plt.show()
