import os, sys
import argparse
import tensorflow as tf
from moxing.framework import file
import cv2
import time

# 执行参数 python notebook/detect_video.py --video notebook/test.mp4 --show_mask True --mask_alpha 0.6 --version V0002
# 外部参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='notebook/test.mp4', help='data mp4 file.')
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

# 输出目录
out_path = "notebook/out/video"

# 输出目录存在需要删除里边的内容
if os.path.exists(out_path):
    file.remove(out_path, recursive=True)
os.makedirs(out_path)

# 帧数，用于通过帧数取图
frameNum = 0

# obs桶路径
obs_path = "obs://puddings/ma-ade20k/notebook/out/video"

if __name__ == "__main__":
    # 载入模型
    model = load_ade20k_model(os.path.join(source_path, 'deeplabv3_xception65_ade20k.h5'))

    # 读取视频
    video = cv2.VideoCapture(ARGS.video)

    # 输出保存视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_out = cv2.VideoWriter(out_path + "/outputVideo.mp4", fourcc, fps, size)

    # 视频是否可以打开，进行逐帧识别绘制
    while video.isOpened:
        # 视频读取图片帧
        retval, frame = video.read()
        if retval:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # 读取失败、结束后释放所有内容
            video.release()
            video_out.release()
            print("没有图像！尝试使用其他视频")
            break

        print('识别帧：%d/%d' % (frameNum, video.get(7)))
        prev_time = time.time()

        # 图片预处理
        resized_image, pad = image_preporcess(frame)
        # 识别并绘制
        predict = model.predict(resized_image)
        segment = image_segment(predict, pad, frame.shape[:2])

        # 是否用语义掩膜覆盖
        if show_mask:
            cv2.addWeighted(segment, mask_alpha, frame, 1-mask_alpha,0, frame)
        else:
            frame = segment

        # 绘制时间
        curr_time = time.time()
        exec_time = curr_time - prev_time
        print("识别耗时: %.2f ms" %(1000*exec_time))

         # 视频输出保存
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_out.write(result)

        # 每300帧取图进行分割保存
        if(frameNum % 300 == 0):
            cv2.imwrite(os.path.join(out_path, str(frameNum) + ".jpg"), result)
        frameNum += 1

        # 绘制视频显示窗
        # cv2.namedWindow("video_reult", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("video_reult", result)
        # 退出窗口
        # if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    # 保存统计总数并复制保存到桶
    print("输出目录：" + out_path)
    # 复制保存到桶
    file.copy_parallel(out_path, obs_path)
    
    # 任务完成后释放所有内容
    video.release()
    video_out.release()
    # cv2.destroyAllWindows()
