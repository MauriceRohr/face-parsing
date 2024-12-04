from inference import FaceParser
import os
from PIL import Image
import cv2
import numpy as np
import argparse
import tqdm

FP_labels = {'background':0,'skin':1,'cloth':16,'hair':17}


def create_FLAME_masks(input_folder,output_folder,file_ending=".png",visualize=False,flip=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    input_files = os.listdir(input_folder)
    input_files = [_f for _f in input_files if _f.endswith(file_ending)]
    print(f"Found {len(input_files)} {file_ending} images")
    print("Computing masks for FLAME ...")
    for _f_in in tqdm.tqdm(input_files):
        input_img =Image.open(os.path.join(input_folder,_f_in))
        size_inp = input_img.size
        input_img_full = np.array(input_img)
        input_img = input_img.resize((512, 512), Image.BILINEAR)
        if flip:
            input_img = np.array(input_img)
            input_img = cv2.flip(input_img,1)
        face_parser = FaceParser()
        mask = face_parser.create_mask(input_img)

        mask_corrected = cv2.resize(mask.astype(np.uint8), (size_inp[0], size_inp[1]))
        if flip:
            mask_corrected=cv2.flip(mask_corrected,1)

        flame_mask = ~((mask_corrected==FP_labels['hair']) | (mask_corrected==FP_labels['background']) | (mask_corrected==FP_labels['cloth']))

        kernel = np.ones((5,5),np.uint8)
        flame_mask_filt = cv2.morphologyEx(flame_mask.astype(np.uint8)*255,cv2.MORPH_OPEN,kernel)
        out_img = flame_mask_filt
        if visualize:
            input_img_full[~(flame_mask_filt>128)] = 0
            out_img=cv2.cvtColor(input_img_full,cv2.COLOR_RGB2BGR)
            
        cv2.imwrite(os.path.join(output_folder,_f_in),out_img)
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='FLAME mask creator',
                    description='Creates image masks that are optimized for use in FLAME model',
                    epilog='Fuck you')
    parser.add_argument('input_folder')           # positional argument
    parser.add_argument('output_folder')           # positional argument
    parser.add_argument('-e', '--file_ending',default=".png")      # option that takes a value
    parser.add_argument('-v', '--visualize',
                    action='store_true')  # on/off flag
    parser.add_argument('-f', '--flip',
                    action='store_true')  # on/off flag
    args = parser.parse_args()
    
    create_FLAME_masks(args.input_folder,args.output_folder,args.file_ending,args.visualize,args.flip)
