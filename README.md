# 基于 ModelArts 平台模型部署预测 —— DeeplabV3 模型

## 介绍

项目使用华为云 `ModelArts` AI 开发平台进行训练部署，采用 `deeplabv3-xception65-ade20k` 算法模型进行识别预测。

模型识别预测结果输出语义分割图

> 可以自己本机运行，需要安装 `tf2.1-gpu` 和 `NVIDIA GPU CUDA`

## 目录结构

将目录文件夹上传至已创建的桶中，文件过多可能上传失败，建议分目录选择上传。

```text
ma-ade20k

┌── infer                               模型推理
│   ├── core
│   ├── config.json
│   ├── customize_service.py
│   └── deeplabv3_xception65_ade20k.h5
├── log                                 模型训练其他日志
├── model                               模型训练版本输出
├── notebook                            开发环境模型调试
│   ├── detect_image.py
│   ├── detect_video.py
│   ├── run.ipynb
│   ├── test.jpg
│   └── test.mp4
├── train                               模型训练
│   └── saved_model.py
└── README.md
```

## 使用前提

拥有一个华为云账号

- EI 企业智能 —— ModelArts
- 存储 —— 对象存储服务 OBS

`ModelArts` 平台需要在全局配置中添加访问密钥才能使用的自动学习、数据管理、Notebook、训练作业、模型和服务可能需要使用对象存储功能，若没有添加访问密钥，则无法使用对象存储功能。

`对象存储服务 OBS` 创建一个桶进行文件的存储。

选择服务地区：**华北-北京四**

## 创建开发环境

在 `ModelArts` 平台中使用 _开发环境>Notebook_ 创建一个工作环境（TF-2.1.0&Pytorch-1.4.0-python3.6 | GPU）和选择对象存储服务（OBS）桶内的已创建或已上传的文件夹进行创建，之后可以启动已创建 `notebook` 进行在线的开发调试和具体的模型训练。使用一些关联文件需要同步到开发环境 `work` 文件夹内，注意同步文件大小存在指定大小内。

## 训练模型

在 `ModelArts` 平台中使用 _训练管理>训练作业_ 创建：

1. 算法来源为常用框架（TensorFlow | TF-2.1.0-python3.6）
2. 代码目录，选择已上传的 `train` 目录
3. 启动文件，选择 `train` 目录内的 `saved_model.py` 文件
4. 数据来源为数据存储位置，选择已创建桶中已上传的 `dataset` 数据文件夹
5. 训练输出位置，选择已上传或创建名为 `model` 的文件夹
6. 作业日志路径，选择已上传或创建名为 `log` 的文件夹
7. 选择 **公共资源池>GPU** 训练更佳
8. 运行参数，参考下表：

|      名称       |  类型  |                    说明                    |
| :-------------: | :----: | :----------------------------------------: |
|    data_url     | string |  已创建桶中已上传的 `dataset` 数据文件夹   |
|    train_url    | string | 已上传或已创建名为 `model` 的文件夹/V 版本 |
|    num_gpus     | number |         拥有 GPU 数，默认固定为 1          |

## 模型推理

在完成上面的模型训练后，你可以在已上传或已创建名为 `model` 的文件夹/V 版本内看到模型文件，但是还不能直接部署，还需要编写平台的模型包规范。因为使用的是 `TF-2.1.0-python3.6` 环境训练，所以使用 `TensorFlow` 模型推理代码进行编写。

使用已经上传的 `infer` 文件夹中的推理文件：

- 方法一：使用开发环境 `notebook` 进行复制 `infer` 文件夹中的推理文件到对应版本模型的 `model` 文件夹内。
- 方法二：直接在 `对象存储服务 OBS` 桶内目录选择对应版本模型的 `model` 文件夹，选择上传对象上传 `infer` 文件夹内的推理文件。

完成上传后，就可以在 `ModelArts` 平台中使用 _模型管理>模型_ 导入：

1. 元模型来源，选择 **从训练中选择** 或 **从对象存储服务（OBS）中选择**
2. 根据元模型来源选择对应的模型版本，**选择训练作业** 或 **选择元模型**
3. 部署类型，选择在线服务即可

推理模型参数如下：

**输入参数**

|    名称    |  类型  |              说明              |
| :--------: | :----: | :----------------------------: |
|   image    |  file  |          上传图片文件          |
| show_mask  | number |  是否输出图片语义掩膜，默认 1  |
| mask_alpha | number | 掩膜透明度 0.1 - 0.9，默认 0.5 |

**输出参数**

|      名称       |  类型  |        说明         |
| :-------------: | :----: | :-----------------: |
| predicted_image | string | 识别绘制图片 base64 |
| recognizer_data | object |        空{}         |

## 模型部署

在完成上面的模型推理后，就可以在 `ModelArts` 平台中使用 _部署上线>在线服务_ 部署。选择模型及配置是你导入训练推理编写后得到的模型和对应的模型版本号，选好 **CPU 规格**下一步直接提交，等待服务启动完成。

服务启动后，可以直接使用服务提供的预测进行图片的预测查看。

## 建议阅读

`ModelArts` 平台：

- [模型包规范介绍](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0091.html)
- [ModelArts 平台常见问题](https://support.huaweicloud.com/modelarts_faq/modelarts_05_0014.html)
- [MoXing 开发指南](https://support.huaweicloud.com/moxing-devg-modelarts/modelarts_11_0001.html)

[PixelLib](https://pixellib.readthedocs.io/en/latest/index.html) 是为使用几行代码执行图像和视频分割而创建的库。
它是一个灵活的库，可轻松将图像和视频分割集成到软件解决方案中。

[PixelLib GitHub仓库](https://github.com/ayoolaolafenwa/PixelLib)


