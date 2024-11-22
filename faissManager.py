import os

import numpy as np
import faiss


class FaissManager:
    
    def __init__(self,index_save_name,with_index_file,dimensions,useIDMap=True):
        #加载数据集
        self.index_save_name = index_save_name
        self.descriptions_name = index_save_name
        self.with_index_file=with_index_file
        self.faiss_index ,self.descriptions_index=self.load(dimensions=dimensions,faiss_name=self.index_save_name,with_index_file=with_index_file,useIDMap=useIDMap)


        pass
    
    def load(self, dimensions : int =512 , faiss_name : str = "index.faiss",with_index_file : bool = True,useIDMap=True):
        """ 向现有 FAISS 索引添加图像特征 """
        if os.path.exists(f"index/{faiss_name}.faiss"):
            index = faiss.read_index(f'index/{faiss_name}.faiss')  # 读取现有的索引
            print("Loaded existing index.")
        else:
            print("No existing index found. Creating a new one.")
            if useIDMap:
                index = faiss.IndexIDMap(faiss.IndexFlatIP(dimensions))  # 如果没有找到索引，则创建一个新的索引
            else:
                index = faiss.IndexFlatIP(dimensions)
        if with_index_file:
            """添加图片索引"""
            if os.path.exists(f"index/{faiss_name}.npy"):
                descriptions_index = np.load(f'index/{faiss_name}.npy', allow_pickle=True).item()
            else:
                print("No existing index found.Will creating a new one.")
                descriptions_index = {}
        else:
            descriptions_index = {}

        return index,descriptions_index

    def get_next_id(self):
        print(self.descriptions_index)

        if not self.descriptions_index:  # 检查字典是否为空
            max_vl = 0
        else:
            ids = [t['desid'] for t in self.descriptions_index.values()]
            max_vl = max(ids) if ids else 0  # 处理无有效值的情况

        print(max_vl)
        return max_vl + 1

    #添加数据
    def add_datas(self, img_descriptions : dict ,img_group_ver : np.ndarray = None, img_group_id : int = 0 ):
        """
        向量
        🐖id
        """
        #TODO 加载现用的列表组查看是否重复Id 重复退出
        #添加向量数据集
        ids = [img_group_id for _ in range(len(img_group_ver))]
        np_datas=np.array(ids)
        self.faiss_index.add_with_ids(img_group_ver,np_datas)

        #TODO 保存向量数据
        self.save_faiss_index()

        #保存索引
        if self.with_index_file:
            self.save_npy_index(img_descriptions, img_group_id)

    #添加数据
    def add_datas2(self, img_descriptions : dict ,img_group_ver : np.ndarray = None, img_group_id : str = None ):
        """
        向量
        🐖id
        """
        if img_group_id is None:
            raise "img_group_id is None"

        get_id = self.get_next_id()
        # max(self.descriptions_index.values())
        #TODO 加载现用的列表组查看是否重复Id 重复退出
        #添加向量数据集
        ids = [get_id for _ in range(len(img_group_ver))]
        np_datas=np.array(ids)
        self.faiss_index.add_with_ids(img_group_ver,np_datas)

        #TODO 保存向量数据
        self.save_faiss_index()

        #保存索引
        if self.with_index_file:
            self.save_npy_index2(img_descriptions, img_group_id,get_id)
    #添加数据
    def add_datas_tags(self, tags_descriptions : list ,tags_group_ver : np.ndarray = None ):
        """
        向量
        🐖id
        """
        #TODO 加载现用的列表组查看是否重复Id 重复退出
        #添加向量数据集
        self.faiss_index.add(tags_group_ver)
        #TODO 保存向量数据
        self.save_faiss_index()

        #保存索引
        if self.with_index_file:
            self.save_tags_npy(tags_descriptions)

    def save_tags_npy(self,tsgd_):

        for ts in tsgd_:
            if self.descriptions_index ==  {}:
                st = 0
            else:
                st = max(self.descriptions_index.keys()) + 1
            self.descriptions_index[st] = ts
            np.save(f'index/{self.descriptions_name}.npy', self.descriptions_index)

    def query_tags(self,query_vector:np.ndarray=None,search_num : int =1):
        print(self.faiss_index.ntotal)
        if self.faiss_index.ntotal>search_num*2:
            distances, indices=self.faiss_index.search(query_vector, search_num*2)
        elif self.faiss_index.ntotal > search_num:
            distances, indices=self.faiss_index.search(query_vector, search_num)
        else:
            distances, indices=self.faiss_index.search(query_vector, self.faiss_index.ntotal)
        print("Search result indices:", indices)
        print("Search result distances:", distances)
        reslut = []
        for i, dist in zip(indices[0], distances[0]):
            print(f"Index: {i}, Distance: {dist}, rsid: {self.descriptions_index[i]}")
            reslut.append({"rs": self.descriptions_index[i], "distance": float(dist)})
        if len(reslut)>search_num:
            return reslut[0:search_num]
        return reslut
    #保存
    def save_faiss_index(self):
        faiss.write_index(self.faiss_index, f'index/{self.index_save_name}.faiss')


    def save_npy_index2(self, img_descriptions : dict = None,img_group_id : str =None,desid=None):
        #描述信息
        if img_group_id is None:
            raise f"{self.descriptions_name}.npy Add Error img_group_id is None"
        #TODO 其他描述信息另说
        descriptions = {
            'data': img_descriptions['data'],
            'desid':desid,

        }
        if img_group_id not in list(self.descriptions_index.keys()):
            self.descriptions_index[img_group_id]=descriptions

        else:
            self.descriptions_index[img_group_id]['data']=list(set(self.descriptions_index[img_group_id]['data'] + descriptions['data']))


        np.save(f'index/{self.descriptions_name}.npy', self.descriptions_index)

        pass


    def save_npy_index(self, img_descriptions : dict = None,img_group_id : int =None):
        #描述信息
        if img_group_id is None:
            raise f"{self.descriptions_name}.npy Add Error img_group_id is None"
        #TODO 其他描述信息另说
        descriptions = {
            'data': img_descriptions['data'],
        }
        if img_group_id not in list(self.descriptions_index.keys()):
            self.descriptions_index[img_group_id]=descriptions

        else:
            self.descriptions_index[img_group_id]['data']=list(set(self.descriptions_index[img_group_id]['data'] + descriptions['data']))


        np.save(f'{self.descriptions_name}.npy', self.descriptions_index)

        pass

    #查询
    def search_faiss_index(self,query_vector,search_num : int = 5):
        print(self.faiss_index.ntotal)
        if self.faiss_index.ntotal>search_num*2:
            distances, indices=self.faiss_index.search(query_vector, search_num*2)
        elif self.faiss_index.ntotal > search_num:
            distances, indices=self.faiss_index.search(query_vector, search_num)
        else:
            distances, indices=self.faiss_index.search(query_vector, self.faiss_index.ntotal)


        print("Search result indices:", indices)
        print("Search result distances:", distances)
        reslut = []
        for i, dist in zip(indices[0], distances[0]):
            print(f"Index: {i}, Distance: {dist}, rsid: {self.descriptions_index[i]['data']}")
            reslut.append({"rsid": i, "distance": float(dist)})
        if len(reslut)>search_num:
            return reslut[0:search_num]
        return reslut
    #查询
    def search_faiss_index2(self,query_vector,search_num : int = 5):
        print(self.faiss_index.ntotal)
        if self.faiss_index.ntotal>search_num*2:
            distances, indices=self.faiss_index.search(query_vector, search_num*2)
        elif self.faiss_index.ntotal > search_num:
            distances, indices=self.faiss_index.search(query_vector, search_num)
        else:
            distances, indices=self.faiss_index.search(query_vector, self.faiss_index.ntotal)


        print("Search result indices:", indices)
        print("Search result distances:", distances)
        reslut = []
        for i, dist in zip(indices[0], distances[0]):

            # print(f"Index: {i}, Distance: {dist}, rsid: {self.descriptions_index[i]['data']}")
            for key,value in self.descriptions_index.items():
                if i==value['desid']:
                    print(f"Index: {i}, Distance: {dist}, rsid: {key}")

                    reslut.append({"rsid": key, "distance": float(dist)})
                    break
            # reslut.append({"rsid": i, "distance": float(dist)})
        if len(reslut)>search_num:

            return reslut[0:search_num]
        return reslut


    #删除
    def delete_faiss_index(self,del_num : int = None):
        if del_num is None:
            raise "Delete Faiss index is None"
        self.faiss_index.remove_ids(np.array([del_num]))

    def delete_faiss_index2(self,del_num : str = None):
        if del_num is None:
            raise "Delete Faiss index is None"

        #查出id
        ids = self.descriptions_index[del_num]['desid']
        self.faiss_index.remove_ids(np.array([ids]))


    #修改
    def change_faiss_index(self,old_num,new_faiss_index : int):
       pass

if __name__ == "__main__":
    fm = FaissManager("index",with_index_file=True,dimensions=512)
    ver_data=np.random.rand(3,512)
    dt = {
        'data': [f'image{i}' for i in range(int(ver_data.shape[0]))]
    }
    fm.add_datas2(dt,ver_data,'873288')

    while True:
        num=int(input("Press Enter:\n 1 : add  / 2 : del / 3 : search \n"))
        if num==1:
            add_num=str(input("Press Enter: ADD NUMBER"))
            ver_data = np.random.rand(3, 512)
            dt = {
                'data': [f'image{i}' for i in range(int(ver_data.shape[0]))]
            }
            fm.add_datas2(dt,ver_data,add_num)
        elif num==2:
            del_num = int(input("Press Enter: ADD NUMBER"))
            fm.delete_faiss_index(del_num)

            ver_data = np.random.rand(1, 512)
            fm.search_faiss_index(ver_data,10)

        elif num==3:
            ver_data = np.random.rand(1, 512)
            print(fm.search_faiss_index(ver_data, 10))






