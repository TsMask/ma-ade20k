# 模型管理 - 推理文件夹

[模型包规范介绍](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0091.html)

- 模型包里面必须包含 `model` 文件夹，`model` 文件夹下面放置模型文件，模型配置文件，模型推理代码。
- 模型配置文件必需存在，文件名固定为 `config.json` , 有且只有一个，模型配置文件编写请参见[模型配置文件编写说明](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0092.html)。
- 模型推理代码文件是可选的。推荐采用相对导入方式（Python import）导入自定义包。如果需要此文件，则文件名固定为 `customize_service.py` , 此文件有且只能有一个，模型推理代码编写请参见[模型推理代码编写说明](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0093.html)。

[自定义脚本代码示例（TensorFlow 2.1）](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0301.html)

## 推理文件上传

```text
你创建并上传的到OBS桶内的文件夹
│
├── infer                           模型推理文件
│   ├── core                        识别所需文件
│   ├── config.json
│   ├── customize_service.py
│   └── deeplabv3_xception65_ade20k.h5
...
```

将该文件夹内的文件上传至对应训练输出的模型版本文件夹里的 `model` 文件夹下。

## 推理模型参数

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
