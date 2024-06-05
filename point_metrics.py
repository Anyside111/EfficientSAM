import os

import cv2
import nibabel as nib
import numpy as np

from EvaluationMetrics import compute_iou, compute_dice


def convert_to_rgb(hu_image, min_hu=-1200, max_hu=3000):
    # normalize HU values to 0-255
    # as HU values range from -1000 to 1000
    hu_image_clipped = np.clip(hu_image, min_hu, max_hu)
    np_img_normalized = (hu_image_clipped - min_hu) / (max_hu - min_hu) * 255
    np_img_normalized = np_img_normalized.astype(np.uint8)

    rgb_image = np.stack([np_img_normalized] * 3, axis=-1)
    return rgb_image


def process_folders(training_file_path, label_file_path, output_folder, predict_output_folder):
    img_index = os.path.basename(training_file_path).split('_')[1].split('.')[0]  # 'XXX' from 'lung_XXX.nii.gz'

    # load the training and label images
    training_img = nib.load(training_file_path)
    label_img = nib.load(label_file_path)
    batched_point_labels = np.array([[[1]]]).astype(np.float32)
    training_data = training_img.get_fdata()
    label_data = label_img.get_fdata()

    iou_result = dict()
    dice_result = dict()
    # traverse through the slices
    for slice_index in range(label_data.shape[2]):
        label_slice = label_data[:, :, slice_index]

        # check if the slice has any positive pixels (target pixels)
        if np.any(label_slice > 0):
            label_slice = (label_slice * 255).astype(np.uint8)  # Scale label

            predict_im_path = os.path.join(predict_output_folder, f'{img_index}_{slice_index}.png')
            predict_im = cv2.imread(predict_im_path)
            predict_mask = predict_im[:, :, 0]

            iou_result[f'{img_index}-{slice_index}'] = compute_iou(predict_mask, label_slice)
            dice_result[f'{img_index}-{slice_index}'] = compute_dice(predict_mask, label_slice)

    return iou_result, dice_result


training_data_folder = '../dataset/Task06_Lung/imagesTr'
label_data_folder = '../dataset/Task06_Lung/labelsTr'
output_folder = '../dataset/Task06_Lung/output'
predict_output_folder = '../dataset/Task06_Lung/predict_output'

iou_dict = dict()
dice_dict = dict()
for filename in os.listdir(training_data_folder):
    if filename.endswith('.nii.gz') and not filename.startswith('._'):
        training_file_path = os.path.join(training_data_folder, filename)
        label_file_path = os.path.join(label_data_folder, filename)
        iou_result, dice_result = process_folders(training_file_path, label_file_path, output_folder,
                                                  predict_output_folder)

        iou_result = {f'{filename}-{k}': v for k, v in iou_result.items()}
        dice_result = {f'{filename}-{k}': v for k, v in dice_result.items()}
        iou_dict.update(iou_result)
        dice_dict.update(dice_result)
    else:
        print(f'One of the paths does not exist: {training_data_folder}, {label_data_folder}')

print("mean iou:", np.mean(list(iou_dict.values())))
print("mean dice:", np.mean(list(dice_dict.values())))

# save to csv
import pandas as pd

iou_dice_data = []
for k, v in iou_dict.items():
    filename, img_index, slice_index = k.split('-')

    iou_dice_data.append({
        "Image Sticker": filename,
        "Slice Index": slice_index,
        "Model": "efficient_sam_vitt.onnx",
        "IoU": iou_dict[k],
        "Dice": dice_dict[k]
    })

iou_dice_df = pd.DataFrame(iou_dice_data)
csv_path = "iou_dice_scores.csv"
iou_dice_df.to_csv(csv_path, index=False)
print("CSV saved to:", csv_path)
