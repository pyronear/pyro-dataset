import glob
import os
import shutil
import random


folders = glob.glob('Data/All_Data/**/**/images')
for folder in folders:
        
        label_folder = folder.replace('images', 'labels')
        if not os.path.isdir(label_folder):
            label_folder = label_folder.replace('All_Data', 'Labels')

        labels = glob.glob(label_folder + '/*')
        
        new_folder = 'Dataset_All/smoke/' if 'smoke' in folder else 'Dataset_All/FP/'
            
        for label_file in labels:
            # Labels
            label_file2 = label_file.replace('Labels', new_folder) if 'Labels' in label_file else label_file.replace('All_Data', new_folder)
            os.makedirs(os.path.split(label_file2)[0], exist_ok=True)
            shutil.copy(label_file, label_file2)
            # Images
            img_file = folder + '/' + os.path.split(label_file)[1].replace('.txt', '.jpg')
            img_file2 = label_file2.replace('labels', 'images').replace('.txt', '.jpg')
            os.makedirs(os.path.split(img_file2)[0], exist_ok=True)
            shutil.copy(img_file, img_file2)
            
        print(os.path.split(img_file2)[0], f"{len(labels)} images")

   
print(f"False positive {len(glob.glob('Data/Dataset_All/FP/**/**/images/*.jpg'))} images")
print(f"Smoke {len(glob.glob('Data/Dataset_All/smoke/**/**/images/*.jpg'))} images")
            

