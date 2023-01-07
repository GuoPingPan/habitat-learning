import os
import cv2
from glob import glob
from tqdm import tqdm
import time

files = glob('1/*')
# print(sorted(files))
# a = files[0].split('-')[-1]
# a = a.rjust(10,'0')
# print(a)
# exit()

# files_name = [file.split('-')[-1].rjust(10,'0') for file in files]
# files_name = sorted(files_name)

# for f,name in zip(files,files_name):
#     os.rename(f,os.path.join('1',name))
#
# exit()
# print(files)
# exit()
files = sorted(files)
print(files)

temp = cv2.imread(files[0])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
video = cv2.VideoWriter('output.mp4', fourcc, fps, (temp.shape[1], temp.shape[0]))

for img_path in files:
    print(img_path)
    img = cv2.imread(img_path)

    # cv2.imshow('img',img)

    # if cv2.waitKey(int(1000/fps)) == ord('q'):
    #     break

    video.write(img)

video.release()