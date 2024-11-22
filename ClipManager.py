
import requests
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fer import FER
from transformers import CLIPProcessor, CLIPModel
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from tqdm import tqdm
from PIL import Image
from clip_interrogator import Interrogator, Config
from faissManager import FaissManager
from showImage import ImageViewer
from sentence_transformers import SentenceTransformer, util
import cv2

class ClipManager:
    def __init__(self):
        # 加载CLIP模型到GPU
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


        self.texcoder_model=SentenceTransformer('all-mpnet-base-v2')

        self.tags_faiss=FaissManager("tags",with_index_file=True,dimensions=768,useIDMap=False)

        self.faiss_img = FaissManager("index", with_index_file=True, dimensions=512)
        
        self.faiss_index = self.load_faiss_database()
        config = Config(cache_path="cache",device="cpu")
        self.ci = Interrogator(config)
        self.image_descriptions=self.load_image_desc()
        self.detector = FER()

    def text_to_vector(self,text, max_length=77):
        """将长文本切分并分别嵌入"""
        # 切分为多段
        tokens = self.tokenizer(text, truncation=False)["input_ids"]
        chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        print(chunks)
        vectors = []
        for chunk in chunks:
            inputs = self.tokenizer.pad({"input_ids": [chunk]}, return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            vectors.append(text_features)

        # 聚合多个嵌入 (取平均值)
        return torch.mean(torch.stack(vectors), dim=0)


    # 根据标签预测图片内容
    def predict_text(self, labels, image_path):
        image_embeds = self.get_image_description(image_path)
        text_embeds = torch.stack([self.get_text_description(label) for label in labels])
        logits_per_image = torch.matmul(image_embeds, text_embeds.T).squeeze()
        # 使用 softmax 得到每个标签的概率
        probs = torch.softmax(logits_per_image, dim=0)
        # 获取最可能的标签索引
        predicted_index = torch.argmax(probs).item()
        predicted_label = labels[predicted_index]


    def get_image_description(self, image_path):
        # 使用 PIL.Image 加载图像
        if type(image_path) == list:
            img=[Image.open(img_p).convert("RGB") for img_p in image_path]
        else:
            img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt", padding=True)

        outputs = self.model.get_image_features(**inputs)
        image_features = outputs.squeeze()
        return image_features


    def get_text_description(self, text_description):
        # 将文本转换为嵌入向量
        inputs = self.processor(text=text_description, return_tensors="pt", padding=True,truncation=True)
        outputs = self.model.get_text_features(**inputs)
        text_features = outputs.squeeze()
        return text_features

    def img_to_natural_sceneery(self,image_path):
        image = Image.open(image_path).convert("RGB")
        return self.ci.interrogate_natural(image,max_flavors=2)

    def img_to_actions(self,image_path):
        image = Image.open(image_path).convert("RGB")
        return self.ci.interrogate_actions(image,max_flavors=2)

    def img_to_emo(self,image_path):
        image = Image.open(image_path).convert("RGB")
        return self.ci.interrogate_emo(image,max_flavors=2)

    def img_to_art(self,image_path):
        image = Image.open(image_path).convert("RGB")
        return self.ci.interrogate_art(image,max_flavors=2)
    def img_to_area(self,image_path):
        image = Image.open(image_path).convert("RGB")
        return self.ci.interrogate_area(image,max_flavors=2)
    def img_to_good_item(self,image_path):
        image = Image.open(image_path).convert("RGB")
        return self.ci.interrogate_gooditem(image,max_flavors=2)

    def img_to_qg(self,img_path):
        image = cv2.imread(img_path)
        # 检查图像是否成功加载
        if image is None:
            print("Error: Unable to load image.")
            return None
        else:
            # 图像成功加载，继续处理
            print("Image loaded successfully.")
        # 预测情感
        emotion, score = self.detector.top_emotion(image)
        print(f'Predicted Emotion: {emotion}, Confidence: {score}')
        return emotion

    def img_to_lable_txt(self,image_path):


        data={
            "描述":self.search_image_to_prompt(image_path),
            "场景":self.img_to_area(image_path),
            "对象":self.img_to_good_item(image_path),
            "活动":self.img_to_actions(image_path),
            "情感":self.img_to_emo(image_path),
            "情绪":self.img_to_emo(image_path),
            "风格":self.img_to_art(image_path),
            "地理位置":self.img_to_scene(image_path),
        }




        return data

    def process_image_multithreaded(self, image_path):
        # 定义任务与对应的方法
        tasks = {
            "描述": self.search_image_to_prompt,
            "场景": self.img_to_area,
            "对象": self.img_to_good_item,
            "活动": self.img_to_actions,
            "情感": self.img_to_emo,
            "情绪": self.img_to_qg,
            "风格": self.img_to_art,
            "地理位置": self.img_to_scene,
        }
        # 用字典存储结果
        results = {}
        # 使用线程池执行任务
        with ThreadPoolExecutor() as executor:
            # 提交任务
            future_to_key = {
                executor.submit(func, image_path): key
                for key, func in tasks.items()
            }

            # 收集结果
            for future in future_to_key:
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = f"Error: {e}"  # 捕获异常并记录
        return results




    # 定义归一化函数
    def normalize_embeddings(self,embeddings):
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norm


    def search_index_exits(self,images_path):
        exits_path=self.image_descriptions.values()
        if type(images_path) == list:
            not_in = []
            for img_path in images_path:
                if img_path not in exits_path:
                    not_in.append(img_path)
            return not_in
        else:
            if images_path not in exits_path:
                return images_path

        return None

    def save_index_faiss(self,images_paths):
        # 保存图像路径或描述信息
        # 使用一个字典来存储图像的索引与其文件名的映射

        print("更新索引中---------")

        if type(images_paths) == list:
            for img_path in tqdm(images_paths):
                t_index = self.image_descriptions.keys()
                if len(t_index) == 0:
                    t_index = -1
                else:
                    t_index = sorted(t_index)[-1]
                index = int(t_index) + 1
                self.image_descriptions[index] =img_path
        else:
            t_index=self.image_descriptions.keys()
            if len(t_index) == 0:
                t_index = -1
            else:
                t_index = sorted(t_index)[-1]
            index = int(t_index) + 1
            self.image_descriptions[index] = images_paths


        # image_descriptions = {i: image_paths[i] for i in range(len(image_paths))}
        print("索引信息", self.image_descriptions)
        # 保存描述信息到文件
        np.save('image_descriptions.npy', self.image_descriptions)



    def add_all_databases(self,image_paths=None):
        if image_paths is None:
            return
        self.add_faiss_databases(image_paths)
        self.add_tags_faiss(image_paths)



    def add_faiss_databases(self,image_paths=None):
        if image_paths is None:
            return
        image_paths = self.search_index_exits(image_paths)
        if image_paths is None or len(image_paths) == 0:
            return
        print(image_paths)
        if type(image_paths) == list and len(image_paths)>1:
            # image_paths=[f"images/{n}" for n in os.listdir("images")]
            # 获取图像特征
            pp = self.get_image_description(image_paths)
            dd = pp.detach().numpy()
            # 归一化图像特征
            dd = self.normalize_embeddings(dd)
        else:
            pp=self.get_image_description(image_paths)
            dd = pp.detach().numpy().reshape(1,-1)
            dd = self.normalize_embeddings(dd)
        self.faiss_index.add(dd)

        # 保存 FAISS 索引
        faiss.write_index(self.faiss_index, 'image_index.faiss')



        # 保存图像路径或描述信息

        self.save_index_faiss(image_paths)

    def web_rq_tags(self,pt):
        # 请求的 URL
        url = "http://127.0.0.1:8877/tags"
        # 提供图片路径，确保路径正确
        # 准备请求数据
        data = {
            "image_path": pt
        }
        # 发送 POST 请求
        response = requests.post(url, json=data)
        # 检查返回的状态码
        if response.status_code == 200:
            # 如果请求成功，解析返回的 JSON 数据
            print(response.json())
            result = response.json()['data']
            print("返回的标签数据:", result)
            return result
        else:
            print(f"请求失败，状态码: {response.status_code}")

    def add_tags_faiss(self,image_paths,web_rq=False):
        datas_tags=[]
        #TODO 获取描述信息
        for tgs in tqdm(image_paths,desc="Progress Images Tags And MS"):
            if web_rq:
                js_dts=";".join(list(filter(None,self.web_rq_tags(tgs).values())))
            else:
                js_dts = ";".join(list(filter(None,self.process_image_multithreaded(tgs).values())))
            datas_tags.append(js_dts)
        self.texcoder.add_fs(datas_tags)



    def search_no_tags(self,prompt_:str,search_num:int =3):
        decodedxt = self.get_text_description(prompt_).detach().numpy().reshape(1, -1)
        rt = self.search_faiss_database(decodedxt, Ver=True, search_num=search_num)
        try:
            result=rt[0:search_num]
        except:
            result=rt
        return result

    def search_finall(self,prompt_:str,search_num:int =3):
        text_= self.texcoder.search_faiss_database(prompt_,search_num=search_num)
        all_data=[]
        if text_ is None or len(text_)==0:
            decodedxt = self.get_text_description(prompt_).detach().numpy().reshape(1, -1)
            rt = self.search_faiss_database(decodedxt, Ver=True, search_num=search_num)
            all_data.extend(rt)
        else:
            for t in text_:
                decodedxt=self.get_text_description(t['rs']).detach().numpy().reshape(1, -1)

                rt=self.search_faiss_database(decodedxt,Ver=True,search_num=search_num)
                all_data.extend(rt)

        #去重
        # 去重逻辑
        seen = set()
        result = []
        for item in all_data:
            if item['image'] not in seen:
                result.append(item)
                seen.add(item['image'])

        # 输出去重结果
        try:
           result = result[0:search_num]  # 切片
        except:
            result=result

        return result[0:search_num]


    def search_faiss_database(self,prompe="sea",Ver=False,search_num:int=5):
        # 获取查询图像特征并归一化
        try:
            if Ver:
                tt = prompe.reshape(1, -1)
            else:
                tt = self.get_text_description(prompe).detach().numpy().reshape(1, -1)
                tt = self.normalize_embeddings(tt)

            # 查询 FAISS 索引
            D, I = self.faiss_index.search(tt, search_num)  # 2 为返回的最近邻个数
            reslut = []
            for i, dist in zip(I[0], D[0]):
                print(f"Index: {i}, Distance: {dist}, Image: {self.image_descriptions[i]}")
                reslut.append({"image": self.image_descriptions[i], "distance": float(dist)})
        except:
            print("-----查询--------【search_faiss_database-img】Error")
            return None

        return sorted(reslut, key=lambda x: x["distance"], reverse=True)






    def search_image_to_prompt(self,image_path):
        image = Image.open(image_path).convert("RGB")
        return self.ci.interrogate_fast_ex(image,max_flavors=2)




    def img_to_scene(self,image_path):
        image = Image.open(image_path).convert("RGB")
        return self.ci.interrogate_scene(image,max_flavors=2)

    def search_faiss_database_prompt(self,image_paths):
        # 获取查询图像特征并归一化
        tt = self.get_image_description(image_paths).detach().numpy().reshape(1, -1)
        tt = self.normalize_embeddings(tt)
        # 查询 FAISS 索引
        print("开始查询----",image_paths)
        D, I = self.prompt_faiss_index.search(tt, 5)  # 2 为返回的最近邻个数
        print("Distances:", D)
        print("Indices:", I)
        # 加载存储的图像描述信息
        # image_descriptions = np.load('image_descriptions.npy', allow_pickle=True).item()
        result=[]
        # 打印搜索结果的描述信息
        print("Search Results:")
        for i, dist in zip(I[0], D[0]):
            print(f"Index: {i}, Distance: {dist}, Image: {self.prompt_data[i]}")
            result.append({"rs": self.image_descriptions[i], "distance": float(dist)})
        return result
    def find_most_similar_image(self, prodict_text):
        image_embeddings = self.load_image_embeddings()
        text_features = self.get_text_description(prodict_text).detach().numpy()
        max_similarity = -1
        most_similar_image = None
        for img_path, img_features in image_embeddings.items():
            # 计算文本特征与图像特征之间的余弦相似度
            similarity = cosine_similarity([text_features], [img_features])[0][0]
            # 如果当前相似度更高，更新最相似的图像
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_image = img_path

        return most_similar_image, max_similarity


    #预测图片
    def prodect_lable_imgs(self,lable=['sea','fish','sky','human',"A shark swims in the deep ocean"]):
        img_ = self.get_image_description("images/3/38524000.jpg").detach().numpy()
        text_ = self.get_text_description(lable).detach().numpy()
        qe = cosine_similarity(text_, img_.reshape(1, -1))  # 这里 text_ 是一个二维数组
        relable = np.reshape(lable, (-1, 1))
        result = np.hstack((relable, qe))
        result_st = result[result[:, 1].argsort()[::-1]]
        print(result_st)
        probs = qe.softmax(dim=0)
        print(probs)
        return result_st

    #预测文字
    def prodect_text_searchimgs(self,prompt,images : list =None):
        text_=self.get_text_description(prompt).detach().numpy()
        images_=self.get_image_description(images).detach().numpy()
        print(images_.shape)
        print(text_.shape)
        qe = cosine_similarity(images_,text_.reshape(1,-1))  # 这里 text_ 是一个二维数组
        relable = np.reshape(images, (-1, 1))
        result = np.hstack((relable, qe))
        result_st = result[result[:, 1].argsort()[::-1]]
        print(result_st)
        return result_st
        # qe_tensor = torch.tensor(qe)  # 转换为张量以使用 softmax
        # # 使用 softmax 将相似度转换为概率
        # probs = torch.softmax(qe_tensor, dim=0)
        # print(probs)

    def load_faiss_database(self):
        """ 向现有 FAISS 索引添加图像特征 """
        if os.path.exists("image_index.faiss") :
            index = faiss.read_index('image_index.faiss')  # 读取现有的索引
            print("Loaded existing index.")
        else:
            print("No existing index found. Creating a new one.")
            index = faiss.IndexFlatIP(512)  # 如果没有找到索引，则创建一个新的索引

        return index


    def load_image_desc(self):
        """添加图片索引"""
        if os.path.exists("image_descriptions.npy"):
            image_descriptions = np.load('image_descriptions.npy', allow_pickle=True).item()
            return image_descriptions
        else:
            print("No existing index found.Will creating a new one.")
            return {}


    def search_tensor_data(self,prompt: str):
        print(self.ci.interrogate_fast_ms(prompt))

    def search_fast_nocaption(self,image_path):
        image = Image.open(image_path).convert("RGB")
        print(self.ci.interrogate_fast_nocaption(image))



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

    def show_img(self,img_path):
        # 打开图片
        image = Image.open(img_path)
        # 显示图片
        image.show()

    def tk_show_image(self,images_path):
        viewer = ImageViewer(images_path)
        viewer()



    def check_faiss_database(self,check_name):
        """ 向现有 FAISS 索引添加图像特征 """
        if os.path.exists(f"index/{check_name}.faiss") :
            return True
        else:
           return False

    def fin_add_img_faiss(self,img_dts,add_id,Classification_name : list = None):
        # ver_data = np.random.rand(3, 512)
        ver_data = self.get_image_description(img_dts).detach().numpy()
        dt = {
            'data': [i for i in img_dts]
        }
        print("Add Main Faiss Database")
        self.faiss_img.add_datas2(dt, ver_data, add_id)
        if Classification_name is not None and len(Classification_name) > 0:
            # 去除空字符串和仅包含空格的内容
            cleaned_data = [item for item in Classification_name if item and item.strip()]
            for Class_name in cleaned_data:
                db = FaissManager(Class_name, True, 512)
                print(f"Add {Class_name} Faiss Database")
                db.add_datas2(dt, ver_data, add_id)




    def fin_add_tag_faiss_databases(self,descriptions:list):
        embeddings=self.get_textcoder_np(descriptions)
        self.tags_faiss.add_datas_tags(descriptions,embeddings,)

    def fin_query_tags_databases(self,prompt:str = None):
        if prompt is None:
            return

        embeddings = self.texcoder_model.encode(prompt, convert_to_tensor=True)
        embeddings = embeddings.detach().numpy().reshape(1, -1)
        data=self.tags_faiss.query_tags(embeddings)
        print(data)
        return data



    def fin_del_img_faiss(self,del_id,Classification_name=None):
        #TODO 修改删除逻辑
        if Classification_name is None:
            self.faiss_img.delete_faiss_index2(del_id)
        else:
            if self.check_faiss_database(Classification_name):
                db=FaissManager(Classification_name,True,512)
                db.delete_faiss_index2(del_id)


 
 
    def fin_search_img_faiss(self,ver_data,Classification_name=None,search_num:int = 5):
            # ver_data = np.random.rand(1, 512)
            # [{'rsid': 33333, 'distance': 127.42607879638672}, {'rsid': 99999, 'distance': 127.10645294189453},
            #  {'rsid': 873288, 'distance': 124.19203186035156}, {'rsid': 99999, 'distance': 123.10406494140625},
            #  {'rsid': 99999, 'distance': 122.38831329345703}, {'rsid': 99999, 'distance': 121.77877044677734},
            #  {'rsid': 873288, 'distance': 120.44474792480469}, {'rsid': 873288, 'distance': 120.1497802734375},
            #  {'rsid': 99999, 'distance': 119.60147094726562}, {'rsid': 33333, 'distance': 119.51825714111328}]
        if Classification_name is None:
            return self.faiss_img.search_faiss_index2(ver_data, search_num)
        else:
            if self.check_faiss_database(Classification_name):
                db = FaissManager(Classification_name,True,512)
                return  db.search_faiss_index2(ver_data, search_num)
            else:
                return []


    def fin_multi_level_search(self,prompt,Classification_names:list=None,search_num:int = 5):
        vector_data = self.get_text_description(prompt).detach().numpy().reshape(1, -1)

        #主数据结构目录数据（包含分类）
        main_data=self.fin_search_img_faiss(vector_data, Classification_name=None, search_num=search_num)
        if Classification_names is None:
            return main_data
        #分类结果数据
        data=[]
        #区孔
        cleaned_data = [item for item in Classification_names if item and item.strip()]

        for Classname in cleaned_data:
            rs=self.fin_search_img_faiss(ver_data=vector_data,Classification_name=Classname,search_num=search_num)
            data.extend(rs)
        #TODO 对主结构数据和分类结果数据综合取值 serch_num个
        #数据结果[{'rsid': 'jfddafaef', 'distance': 132.1553192138672}, {'rsid': '873288', 'distance': 131.04873657226562}, {'rsid': '873288', 'distance': 130.16526794433594}, {'rsid': 'jfddafaef', 'distance': 127.8250503540039}, {'rsid': 'dfahfhag', 'distance': 127.30360412597656}, {'rsid': 'jfddafaef', 'distance': 125.51935577392578}, {'rsid': 'dfahfhag', 'distance': 124.85501098632812}, {'rsid': '873288', 'distance': 124.62327575683594}, {'rsid': 'dfahfhag', 'distance': 123.88580322265625}]
        # 合并主数据和分类结果
        # 合并主数据和分类结果
        combined_data = main_data + data  # 使用 + 操作符合并列表

        # 去重并排序
        unique_data = {}
        for item in combined_data:
            # 如果 rsid 已存在，保留距离较小的条目
            if item['rsid'] not in unique_data or item['distance'] > unique_data[item['rsid']]['distance']:
                unique_data[item['rsid']] = item

        # 转为列表并按 distance 排序
        sorted_data = sorted(unique_data.values(), key=lambda x: x['distance'],reverse=True)

        # 截取前 search_num 个结果
        return sorted_data[:search_num]


    def get_textcoder_np(self,descriptions: list =None):
        if descriptions is None:
            return None
        embeddings = self.texcoder_model.encode(descriptions, convert_to_tensor=True)
        embeddings = embeddings.detach().numpy()
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def fin_add_finall(self,img_dts,add_id,Classification_name : list = None,add_tags=False,web_rq=False):

        #checkImages——path
        for img in  img_dts:
            if not os.path.exists(img):
                raise f"{img} not exists check_your images"

        if add_tags:
            #TODO 获取描述和标签
            datas_tags = []
            # TODO 获取描述信息
            for tgs in tqdm(img_dts, desc="Progress Images Tags And MS"):
                if web_rq:
                    js_dts = ";".join(list(filter(None, self.web_rq_tags(tgs).values())))
                else:
                    js_dts = ";".join(list(filter(None, self.process_image_multithreaded(tgs).values())))
                datas_tags.append(js_dts)

            embeddings = self.get_textcoder_np(datas_tags)
            self.tags_faiss.add_datas_tags(datas_tags,embeddings)
        self.fin_add_img_faiss(img_dts,add_id,Classification_name)


    def fn_search_finall(self,prompt,search_num:int = 5,Classification_names:list = None,start_tags=False):
        rsdata=prompt
        if start_tags:
            rsdata = self.fin_query_tags_databases(prompt)

        if type(rsdata) is list:
            return self.fin_multi_level_search(rsdata[0]['rs'],Classification_names,search_num=search_num)
        else:
            return self.fin_multi_level_search(prompt,Classification_names,search_num=search_num)


























































if __name__ == "__main__":
    fm = ClipManager()
    while True:
        num=int(input("Press Enter:\n 1 : add  / 2 : del / 3 : search \n"))
        if num==1:

            add_num=str(input("Press Enter: ADD NUMBERID \n"))

            ml = input("PASS目录:\n")
            Class_name = input("Press Enter:(查询的标签) \n")
            classt = []
            classt.append(Class_name)
            if len(classt) == 0:
                classt = None
            img_dts=[f"images/{ml}/{m}" for m in os.listdir(f"images/{ml}")]
            fm.fin_add_img_faiss(img_dts,add_num,classt)
        elif num==2:
            Class_name=input("Press Enter:(查询的标签) \n")
            classt=[]
            classt.append(Class_name)
            if len(classt)==0:
                classt=None

            del_num = int(input("Press Enter: DEL NUMBER\n"))
            fm.fin_del_img_faiss(del_num,classt)

            ver_data = np.random.rand(1, 512)
            fm.fin_search_img_faiss(ver_data,10)

        elif num==3:
            prompt=input("Press Enter(PROMPT):\n")
            Class_name=input("Press Enter:(查询的标签) \n").split(',')

            # classt=[]
            # classt.append(Class_name)
            # if len(classt)==0:
            #     classt=None
            # ver_data = np.random.rand(1, 512)
            print(fm.fin_multi_level_search(prompt, Class_name,10))




