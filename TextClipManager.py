import os
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from faissManager import FaissManager

class TextClipManager:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.tags_descriptions = self.load_tags_image_desc()
        self.tags_faiss = self.load_tags_faiss_database()

    def load_tags_faiss_database(self):
        """ 向现有 FAISS 索引添加图像特征 """
        if os.path.exists("tags_index.faiss"):
            index = faiss.read_index('tags_index.faiss')  # 读取现有的索引
            print("Loaded existing index.")
        else:
            print("No existing index found. Creating a new one.")
            index = faiss.IndexFlatIP(768)  # 如果没有找到索引，则创建一个新的索引

        return index
    def load_tags_image_desc(self):
        """添加图片索引"""
        if os.path.exists("tags_descriptions.npy"):
            image_descriptions = np.load('tags_descriptions.npy', allow_pickle=True).item()
            return image_descriptions
        else:
            print("No existing index found.Will creating a new one.")
            return {}
    def save_tags_index_faiss(self,prompt_datas):
        # 保存图像路径或描述信息
        # 使用一个字典来存储图像的索引与其文件名的映射
        print("更新索引中---------")
        if type(prompt_datas) == list:
            for img_path in tqdm(prompt_datas):
                t_index = self.tags_descriptions.keys()
                if len(t_index) == 0:
                    t_index = -1
                else:
                    t_index = sorted(t_index)[-1]
                index = int(t_index) + 1
                self.tags_descriptions[index] =img_path
        else:
            t_index=self.tags_descriptions.keys()
            if len(t_index) == 0:
                t_index = -1
            else:
                t_index = sorted(t_index)[-1]
            index = int(t_index) + 1
            self.tags_descriptions[index] = prompt_datas
        # print("索引信息", self.tags_descriptions)
        # 保存描述信息到文件
        np.save('tags_descriptions.npy', self.tags_descriptions)
    def add_tag_faiss_databases(self,description:str):
        embeddings = self.model.encode(description, convert_to_tensor=True)
        dd = embeddings.detach().numpy().reshape(1, -1)
        self.tags_faiss.add(dd)
        # 保存 FAISS 索引
        faiss.write_index(self.tags_faiss, 'tags_index.faiss')
        # 保存图像路径或描述信息
        self.save_tags_index_faiss(description)

    def search_faiss_database(self, prompe="sea", search_num: int = 5):
        # 获取查询图像特征并归一化
        try:
            p_embeddings = self.model.encode(prompe, convert_to_tensor=True)
            # 查询 FAISS 索引
            D, I = self.tags_faiss.search(p_embeddings.detach().numpy().reshape(1, -1), search_num)  # 2 为返回的最近邻个数
            reslut = []
            for i, dist in zip(I[0], D[0]):
                print(f"Index: {i}, Distance: {dist}, rs: {self.tags_descriptions[i]}")
                reslut.append({"rs": self.tags_descriptions[i], "distance": float(dist)})
        except:
            print("----查询-----[search_faiss_database]Error")
            return None


        return reslut

    def web_request_tags(self,image_path):
        # 请求的 URL
        url = "http://127.0.0.1:8877/fast_tags"
        # 提供图片路径，确保路径正确
        # 准备请求数据
        data = {
            "image_path": image_path
        }
        # 发送 POST 请求
        response = requests.post(url, json=data)
        # 检查返回的状态码
        if response.status_code == 200:
            # 如果请求成功，解析返回的 JSON 数据
            result = response.json()['data']
            print("返回的标签数据:", result)
            return result
        else:
            print(f"请求失败，状态码: {response.status_code}")


    def test_rn(self,r:list):
        for i in tqdm(r,desc="Progress Tags Ver"):
            data=self.web_request_tags(i)
            self.add_tag_faiss_databases(data)

    def add_fs(self,datas_tags):
        for i in tqdm(datas_tags,desc="Progress Tags Ver"):
            self.add_tag_faiss_databases(i)

if __name__ == "__main__":
    tx=TextClipManager()
    tx.test_rn([f"D:/aibase/CLIP-In2/images/{t}" for t in os.listdir("images")])
    while True:
        ip=input("输入：")
        print(tx.search_faiss_database(ip))

