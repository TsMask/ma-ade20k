import os, sys
import argparse
import tensorflow as tf
from moxing.framework import file
import cv2
import time

# 执行参数 python notebook/detect_image.py --image notebook/test.jpg --show_mask True --mask_alpha 0.6 --version V0002
# 外部参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='notebook/test.jpg', help='data jpg file.')
parser.add_argument('--show_mask', type=bool, default=True, help='show image mask.')
parser.add_argument('--mask_alpha', type=float, default=0.3, help='show image mask alpha.')
parser.add_argument('--version', type=str, default='V0002', help='model version')
ARGS = parser.parse_args()

show_mask = ARGS.show_mask # 图片语义掩膜
mask_alpha = ARGS.mask_alpha # 掩膜透明度 0.1 - 0.9fds
version = ARGS.version # 模型版本

# 执行所在路径， V0xxx 表示模型版本号
source_path = os.path.join(os.getcwd(), "model/" + version + "/model")
sys.path.append(source_path)

from core.semantic import load_ade20k_model, image_preporcess, image_segment

# obs桶路径
obs_path = "obs://puddings/ma-ade20k/notebook/out/image"

# 输出目录
out_path = "notebook/out/image"

# 输出目录存在需要删除里边的内容
if os.path.exists(out_path):
    file.remove(out_path, recursive=True)
os.makedirs(out_path)

if __name__ == "__main__":
    # 载入模型
    model = load_ade20k_model(os.path.join(source_path, 'deeplabv3_xception65_ade20k.h5'))
    
    # 读取图片
    image = cv2.imread(ARGS.image)
    
    prev_time = time.time()

    # 图片预处理
    resized_image, pad = image_preporcess(image)
    # 识别并绘制
    predict = model.predict(resized_image)
    segment = image_segment(predict, pad, image.shape[:2])

    # 是否用语义掩膜覆盖
    if show_mask:
        cv2.addWeighted(segment, mask_alpha, image, 1-mask_alpha,0, image)
    else:
        image = segment

    # 绘制时间
    curr_time = time.time()
    exec_time = curr_time - prev_time
    print("识别耗时: %.2f ms" %(1000*exec_time))

    # print("识别结果：", recognizer)

    # 绘制保存
    cv2.imwrite(out_path + "/output" + str(round(prev_time * 1000)) + ".jpg", image) 

    # 复制保存到桶
    print("输出目录：" + out_path)
    file.copy_parallel(out_path, obs_path)
    
    # 显示窗口
    # cv2.namedWindow('image_result', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('image_result', image)
    # 退出窗口
    # cv2.waitKey(0)
    # 任务完成后释放内容
    # cv2.destroyAllWindows()
    