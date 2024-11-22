import os
import uvicorn
from ClipManager import ClipManager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
cp = ClipManager()

app = FastAPI()

# 默认路由，返回简单的 JSON
@app.get("/")
async def hello():
    return {"Hello": "World"}


# 定义接收图片路径的请求体格式
class ImageRequest(BaseModel):
    imagepath: str


# 定义接收图片路径的请求体格式
class VideoActionRequest(BaseModel):
    videopath: str


# 定义接收提示文本的请求体格式
class TextRequest(BaseModel):
    prompt: str
    search_num: int


class AddFaissIndex(BaseModel):
    imgpaths: list




#添加索引
class AddIndex(BaseModel):
    imgpaths: list
    add_id :str
    classification_name : list
    add_tags : bool

class SearchIndex(BaseModel):
    prompt: str
    search_num :int
    classification_name : list
    start_tags : bool


@app.post("/addindex")
async def add_index(request: AddIndex):
    try:
        cp.fin_add_finall(request.imgpaths,request.add_id,request.classification_name,request.add_tags)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search_prompt")
async def search_prompt(request: SearchIndex):
    try:
        data=cp.fn_search_finall(request.prompt,request.search_num,request.classification_name,request.start_tags)
        return {"success": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))






@app.post("/img_to_lable")
async def img_to_natural_scenery(request: ImageRequest):
    try:
        data = cp.process_image_multithreaded(request.imagepath)
        return {"state": "success", "data": data}
    except Exception as e:
        return {"state": "error", "message": str(e)}














#
#
#
# # 图片生成描述语句接口，接收 JSON 数据
# @app.post("/img_to_description")
# async def img_to_txt(request: ImageRequest):
#     try:
#         # 从请求体中获取 imagepath
#         description = cp.search_image_to_prompt(request.imagepath)
#         return {"state": "success", "data": description}
#     except Exception as e:
#         return {"state": "error", "message": str(e)}
#
#
# # 文字查询图片接口（无tags)查询
# @app.post("/search_no_tags")
# async def txt_to_img(request: TextRequest):
#     try:
#         searchnum = request.search_num
#         if searchnum == 0:
#             return {"state": "success", "data": None}
#         # 从请求体中获取 prompt
#         data = cp.search_no_tags(request.prompt, search_num=searchnum)
#         print(data)
#         return {"state": "success", "data": data}
#     except Exception as e:
#         return {"state": "error", "message": str(e)}
#
#
# # 添加图片索引
# @app.post("/add_index_img")
# async def add_index_img(request: AddFaissIndex):
#     try:
#         cp.add_faiss_databases(request.imgpaths)
#         return {"state": "success", "data": None}
#     except Exception as e:
#         return {"state": "error", "message": str(e)}
#
#
# # 添加标签索引
# @app.post("/add_index_tags")
# async def add_index_img(request: AddFaissIndex):
#     try:
#         cp.add_tags_faiss(request.imgpaths)
#         return {"state": "success", "data": None}
#     except Exception as e:
#         return {"state": "error", "message": str(e)}
#
#
# # 图片查询自然场景
# @app.post("/img_to_natural_scenery")
# async def img_to_natural_scenery(request: ImageRequest):
#     try:
#         data = cp.interrogate_naturalimg_to_natural_sceneery(request.imagepath)
#         return {"state": "success", "data": data}
#     except Exception as e:
#         return {"state": "error", "message": str(e)}
#
#
# # 图片查询标签
# @app.post("/img_to_lable")
# async def img_to_natural_scenery(request: ImageRequest):
#     try:
#         data = cp.process_image_multithreaded(request.imagepath)
#         return {"state": "success", "data": data}
#     except Exception as e:
#         return {"state": "error", "message": str(e)}
#
#
# # 视频预测行为
# @app.post("/video_action")
# async def video_action(request: VideoActionRequest):
#     try:
#         data = eva.predict_video_avcction(request.videopath)
#         return {"state": "success", "data": data}
#     except Exception as e:
#         return {"state": "error", "message": str(e)}
#
#
# ##以标签行为查询图片
# @app.post("/search_with_tags")
# async def search_with_tags(request: TextRequest):
#     try:
#         searchnum = request.search_num
#         if searchnum == 0:
#             return {"state": "success", "data": None}
#         data = cp.search_finall(request.prompt, searchnum)
#         return data
#     except Exception as e:
#         return {"state": "error", "message": str(e)}
#

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8877,workers=1)
