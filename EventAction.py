import cv2
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import torch
from PIL import Image
#facebook transforms

class EventAction:
    def __init__(self):
        # 加载预训练模型和处理器
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
        pass


    # 提取视频帧
    def extract_frames_from_video(self,video_path, num_frames=8, frame_size=(224, 224)):
        cap = cv2.VideoCapture(video_path)
        frames = []

        frame_count = 0
        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
            frame = cv2.resize(frame, frame_size)  # 调整为指定尺寸
            frames.append(frame)
            frame_count += 1

        cap.release()
        return frames



    def predict_img_action(self,image_path):
        tp=Image.open(image_path).convert("RGB")
        inputs = self.processor(tp, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        print(logits)
        print(logits.shape)

    def predict_video_avcction(self,video_path: str):

        # 从视频文件中提取帧
        frames = self.extract_frames_from_video(video_path)

        # 处理输入数据
        inputs = self.processor(frames, return_tensors="pt")

        # 推理并输出结果
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # 获取排序后的前五个最大值
        top5_logits = sorted_logits[:, :5]

        # 获取排序后前五个最大值的索引
        top5_indices = sorted_indices[:, :5]

        print("Top 5 sorted logits:")
        print(top5_logits)

        print("Top 5 indices of sorted logits:")
        print(top5_indices)

        li=[]
        for an,class_ in zip(top5_logits.squeeze(),top5_indices.squeeze()):
            li.append({self.model.config.id2label[class_.item()]:an.item()})
        print(li)
        # print(logits.shape)
        # print(logits.argmax(-1))
        # print(logits.argmax(-1).item())
        # print(self.model.config.id2label.values())
        # with open('actions.txt',mode='w') as f:
        #     f.write("\n".join(self.model.config.id2label.values()))

        print("Predicted class:", self.model.config.id2label[predicted_class_idx])
        return li

if __name__ == '__main__':
    ev=EventAction()
    ev.predict_img_action("D:\\aibase\\CLIP-In2\\images\\0013_UPSCALE_Black_Mixture_ComfyUI.jpg")