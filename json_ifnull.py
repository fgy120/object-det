#-*- coding:utf-8 –*-
import cv2
import os
import glob
import json
import collections
import numpy as np
from PIL import Image
from labelme import utils
# import moxing as mox
# mox.file.shift('os', 'mox')

if __name__ == "__main__":
    src_dir = 'F:\\lunwen\\code\\MyData\\pest_dete_reco\\data\\dete_dataset\\train'
    # src_dir ='F:\\lunwen\\dataset\\test'
    # dst_dir = 'F:\\lunwen\\dataset\\test'
    
    # if not os.path.exists(dst_dir):
    #     os.makedirs(dst_dir)
    # 先收集一下文件夹中图片的格式列表，例如 ['.jpg', '.JPG']
    # exts = dict()
    filesnames = os.listdir(src_dir)
    for filename in filesnames:
        name, ext = filename.split('.')
        if ext == 'json':
            # if exts.__contains__(ext):
            #     exts[ext] += 1
            # else:
            #     exts[ext] = 1

    # anno = collections.OrderedDict()  # 这个可以保证保存的字典顺序和读取出来的是一样的，直接使用dict()的话顺序会很乱（小细节哦）
    # for key in exts.keys():
        # for img_file in glob.glob(os.path.join(src_dir, '*.' + key)):
            # file_name = os.path.basename(img_file)
            # print(f"Processing {file_name}")
#             print("imgfile",img_file)
            # img=Image.open(img_file)
#             img = cv2.imread(img_file)
            # img=img.convert('L')
            # h, w = img.size   # 统计了一下，所有图片的宽度里面，1344是占比较多的宽度中最小的那个，因此
            #                         # 都等比例地将宽resize为1344(这里可以自己修改)
            # w_new = 800
            # h_new = 800#int(h / w * w_new)  # 高度等比例缩放
            # ratio_w = w_new / w  # 标注文件里的坐标乘以这个比例便可以得到新的坐标值
            # ratio_h = h_new / h
            # img2 = img.resize((w_new,h_new), Image.BICUBIC)
            # img2.save(os.path.join(dst_dir, file_name))
#             img_resize = cv2.resize(img, (w_new, h_new))  # resize中的目标尺寸参数为(width, height)
#             cv2.imwrite(os.path.join(dst_dir, file_name), img_resize)

            # 接下来处理标注文件json中的标注点的resize
            json_file = os.path.join(src_dir, name + '.json')
            # save_to = open(os.path.join(dst_dir, name + '.json'), 'w',encoding='utf-8')
            # w2=0
            # h2=0
            # print(json_file)
            # with open(json_file, encoding='utf-8') as f:
            anno = json.load(open(json_file, encoding='utf-8'))
            # for shape in anno["shapes"]:
                # print(shape['label'])
            imgname=anno["imagePath"]
            n=imgname.split('.')[0]
            if n!=name:
                print(json_file)
                print("false")
                break

#                     points = shape["points"]
#                     print("points",points)
# #                     points = (np.array(points) * ratio).astype(int8).tolist()
# #                     for i in range(len(points)):
# #                         print()
#                     points[0][0]=(points[0][0] * ratio_h)#.astype(float16)#.tolist()
#                     points[1][0]=(points[1][0] * ratio_h)#.astype(float16)#.tolist()
#                     w2=abs(points[1][0]-points[0][0])
#                     points[0][1]=(points[0][1] * ratio_w)#.astype(float16)#.tolist()
#                     points[1][1]=(points[1][1] * ratio_w)#.astype(float16)#.tolist()
#                     h2=abs(points[0][1]-points[1][1])
#                     shape["points"] = points
#                     print("points2",points)
                # 注意下面的img_resize编码加密之前要记得将通道顺序由BGR变回RGB
                # img3 = np.array(img2) 
                # anno['imageData']=str(utils.img_arr_to_b64(img3), encoding='utf-8')
                # anno['imageHeight']=800
                # anno['imageWidth']=800
                # print(h2,w2)
            # json.dump(anno, save_to, indent=4,ensure_ascii =False)
    print("Done")