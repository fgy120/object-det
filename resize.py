import cv2
import os
import glob
import json
import collections
import numpy as np
from labelme import utils


if __name__ == "__main__":
    src_dir = './srcDir'
    dst_dir = './dstDir'
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # 先收集一下文件夹中图片的格式列表，例如 ['.jpg', '.JPG']
    exts = dict()
    filesnames = os.listdir(src_dir)
    for filename in filesnames:
        name, ext = filename.split('.')
        if ext != 'json':
            if exts.__contains__(ext):
                exts[ext] += 1
            else:
                exts[ext] = 1

    anno = collections.OrderedDict()  # 这个可以保证保存的字典顺序和读取出来的是一样的，直接使用dict()的话顺序会很乱（小细节哦）
    for key in exts.keys():
        for img_file in glob.glob(os.path.join(src_dir, '*.' + key)):
            file_name = os.path.basename(img_file)
            print(f"Processing {file_name}")
            img = cv2.imread(img_file)
            (h, w, c) = img.shape   # 统计了一下，所有图片的宽度里面，1344是占比较多的宽度中最小的那个，因此
                                    # 都等比例地将宽resize为1344(这里可以自己修改)
            w_new = 1344
            h_new = int(h / w * w_new)  # 高度等比例缩放
            ratio = w_new / w  # 标注文件里的坐标乘以这个比例便可以得到新的坐标值
            img_resize = cv2.resize(img, (w_new, h_new))  # resize中的目标尺寸参数为(width, height)
            cv2.imwrite(os.path.join(dst_dir, file_name), img_resize)

            # 接下来处理标注文件json中的标注点的resize
            json_file = os.path.join(src_dir, file_name.split('.')[0] + '.json')
            save_to = open(os.path.join(dst_dir, file_name.split('.')[0] + '.json'), 'w')
            with open(json_file, 'rb') as f:
                anno = json.load(f)
                for shape in anno["shapes"]:
                    points = shape["points"]
                    points = (np.array(points) * ratio).astype(int).tolist()
                    shape["points"] = points

                # 注意下面的img_resize编码加密之前要记得将通道顺序由BGR变回RGB
                anno['imageData']=str(utils.img_arr_to_b64(img_resize[..., (2, 1, 0)]), encoding='utf-8')
                json.dump(anno, save_to, indent=4)
    print("Done")