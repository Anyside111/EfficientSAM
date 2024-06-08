import json
import pandas as pd

# Load the CSV data
csv_file_path = "iou_dice_score_madsam_tune.csv"
data = pd.read_csv(csv_file_path)

# Filter data for IoU and Dice scores greater than or equal to 0.7
filtered_data = data[(data['IoU'] >= 0.7) & (data['Dice'] >= 0.7)]

# Define the original prompts
prompts = {
    "General Analysis": "Pretend you are an advanced pulmonologist. Analyze the provided image of biomedical lung segmentation. Describe the segmented regions, their significance, and the metrics provided (IoU and Dice scores). Explain what the colors represent and the importance of the segmented areas in a medical context.",
    "Detailed Region Analysis": "Pretend you are an advanced pulmonologist. Examine the image of biomedical lung segmentation. Identify the key regions segmented in the lungs, explain the role of these regions in lung health, and interpret the metrics shown (IoU and Dice scores). Discuss the implications of the segmentation accuracy on potential medical diagnoses.",
    "Metrics Interpretation": "Pretend you are an advanced pulmonologist. Interpret the IoU and Dice scores presented in the biomedical lung segmentation image. Explain what these scores indicate about the accuracy of the segmentation and how they might affect medical outcomes. Discuss any potential improvements or concerns based on these metrics.",
    "Color Representation": "Pretend you are an advanced pulmonologist with focus on using different color when exaiming lung CT scan with segmented potential tumor. Describe the color-coding used in the biomedical lung segmentation image. Explain what each color represents, how it differentiates various lung regions or abnormalities, and the significance of these distinctions in medical imaging.",
    "Medical Context": "Pretend you are an advanced pulmonologist. Provide a medical interpretation of the lung segmentation image. Discuss the clinical relevance of the segmented regions, the significance of the highlighted areas, and how the segmentation results could influence patient diagnosis and treatment. Explain the importance of the IoU and Dice scores in this context."
}

# Append CSV data to the prompts
integrated_prompts = {}
for index, row in filtered_data.iterrows():
    for key, prompt in prompts.items():
        integrated_prompt = f"{prompt}\n\nImage Sticker: {row['Image Sticker']}, Slice Index: {row['Slice Index']}, Model: {row['Model']}, IoU: {row['IoU']}, Dice: {row['Dice']}"
        integrated_prompts[f"{key} - {row['Image Sticker']} {row['Slice Index']}"] = integrated_prompt

# Save to a JSON file
output_file_path = "integrated_segmentation_prompts.json"
with open(output_file_path, 'w') as file:
    json.dump(integrated_prompts, file, indent=4)
