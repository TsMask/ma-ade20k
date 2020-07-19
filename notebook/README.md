# 开发环境模型调试

在开发环境 notebook 中将目录中的 `model目录/V版本`，`notebook目录` 进行 Sync Obs 同步。

- `model目录/V版本` 中可能含有多个版本，只需要选择需要调试的版本目录进行同步即可
- `notebook目录` 中含 `detect_image.py文件` 和 `detect_video.py文件` 是便于自己本机运行和在开发环境 `Terminal - TensorFlow-2.1.0` 中执行。

## 终端执行方式

1. 打开 `Terminal` 后命令，可以看到已同步的 `model` 和 `notebook`

```shell
cd work

# V0015 是你同步的模型版本
ls -lh model/V0002/model

ls -lh notebook
```

2. 先切换到 `tf-2.1.0` 环境

```shell
source /home/ma-user/anaconda3/bin/activate TensorFlow-2.1.0
# (TensorFlow-2.1.0) sh-4.3$
```

3. 选择执行你需要的识别 py

```shell
# 图片识别 detect_image.py 文件
python notebook/detect_image.py --image notebook/test.jpg --show_mask True --mask_alpha 0.6 --version V0002

# 视频识别 detect_video.py 文件
python notebook/detect_video.py --video notebook/test.mp4 --show_mask True --mask_alpha 0.6 --version V0002
```

**输入参数**

|    名称    |                说明                |
| :--------: | :--------------------------------: |
|   image    |            图片文件路径            |
|   video    |            视频文件路径            |
| show_mask  |      图片语义掩膜，默认 true       |
| mask_alpha |   掩膜透明度 0.1 - 0.9，默认 0.3   |
|  version   | 选择你同步的模型版本号，默认 V0xxx |
