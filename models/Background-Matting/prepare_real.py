#######################################
# Prepares training data. Takes a path to a directory of videos + captured backgrounds, dumps frames, extracts human
# segmentations. Also takes a path of background videos. Creates a training CSV file with lines of the following format,
# by using all but the last 80 frames of each video and iterating repeatedly over the background frames as needed.

#$image;$captured_back;$segmentation;$image+20frames;$image+2*20frames;$image+3*20frames;$image+4*20frames;$target_back

path = "ak/"
background_path = "ak/"
output_csv = "Video_data_train.csv"

#######################################

import os
from itertools import cycle
from tqdm import tqdm

with open(output_csv, "w") as f:
    video = "ak"
    n = len(os.listdir(video))
    print(n)
    assert n % 2 == 0
    n //= 2
    for j in range(1, n + 1 - 5):
        img_name = video + "/%04d_img.png" % j
        captured_back = video + ".png"
        seg_name = video + "/%04d_masksDL.png" % j
        mc1 = video + "/%04d_img.png" % (j + 1)
        mc2 = video + "/%04d_img.png" % (j + 2)
        mc3 = video + "/%04d_img.png" % (j + 3)
        mc4 = video + "/%04d_img.png" % (j + 4)
        target_back = "ak.png"
        csv_line = f"{img_name};{captured_back};{seg_name};{mc1};{mc2};{mc3};{mc4};{target_back}\n"
        f.write(csv_line)

print(f"Done, written to {output_csv}")
