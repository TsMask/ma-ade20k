from model_service.tfserving_model_service import TfServingBaseService
from core.semantic import load_ade20k_model, image_preporcess, image_segment, image_to_base64
from skimage import io
import cv2

# 推理服务
class ade20k_service(TfServingBaseService):
    count = 1               # 预测次数
    model_object = None     # 模型实例

    def _preprocess(self, data):
        temp_data = {}
        
        # 遍历提交参数取值，image必传，配置默认值
        for k, v in data.items():
            if k == 'image':
                # 参数的默认值
                temp_data['show_mask'] = int(data['show_mask']) if 'show_mask' in data else 1
                temp_data['mask_alpha'] = float(data['mask_alpha']) if 'mask_alpha' in data else 0.5
                
                # file_name, file_content 图片字典数据
                for _, file_content in v.items():
                    image = io.imread(file_content)
                    temp_data[k] = image

        # 加载模型实例
        if(self.model_object == None):
            print('--加载模型实例--')
            self.model_object = load_ade20k_model('model/1/deeplabv3_xception65_ade20k.h5')

        return temp_data

    def _postprocess(self, data):
        outputs = {}

        # 输入参数
        image = data['image']
        show_mask = data['show_mask']
        mask_alpha = data['mask_alpha']

        # 图片预处理
        resized_image, pad = image_preporcess(image)
        # 识别并绘制
        predict = self.model_object.predict(resized_image)
        segment = image_segment(predict, pad, image.shape[:2])

        # 是否用语义掩膜覆盖
        if show_mask:
            cv2.addWeighted(segment, mask_alpha, image, 1-mask_alpha,0, image)
        else:
            image = segment

        # 预测次数+1
        print('预测次数：', self.count)
        self.count += 1

        # 输出识别处理的base64图片
        itb64 = image_to_base64(image)
        outputs['predicted_image'] = itb64
        outputs['recognizer_data'] = {}
        return outputs

    def _inference(self, data):
        return data
