import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import torch_npu
from torch_npu.contrib import transfer_to_npu


import glob
import transformers

from torch.utils.data import Dataset
from myclip.train.clip_trainer import CLIPTrainer


import torch.distributed as dist

import copy
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange
# import cv2
from random import choice
from PIL import Image

import gzip
from io import BytesIO
import base64
from torch.utils.data import  IterableDataset
import random

# from myclip.model.clip_strc.myclip_box import LongCLIPModel
# from myclip.model.clip_strc.myclip_filip import LongCLIPModel
from myclip.model.clip_strc.myclip_clstext import LongCLIPModel
# fg_clip_newtopk
# from myclip.model.clip_strc.fg_clip_newtopk import LongCLIPModel

# from myclip.model.clip_strc.configuration_clip import CLIPConfig
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


import gc





local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

# global sub_path_list

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    base_model: Optional[str] = field(default=None)
    download_root: Optional[str] = field(default=None)
    # parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    log_scale: float = 4.6052

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    # max_seq_length: int = 77
    max_seq_length: int = 77*4-60
    base_seq_length: int = 77
    box_image_size: int = 224
    add_box_loss: bool = field(default=False)
    use_longcaption: bool = field(default=False)
    train_with_laion: bool = field(default=False)
    laion_longcaption_root: Optional[str] = field(default=None)
    
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    train_use_word_size: int = 8
    text_model_lr: Optional[float] = None
    text_only_long: bool = field(default=False)


from datetime import datetime
    
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.npu.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()

    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 data_args: DataArguments,
                 img_preprocess=None,tokenizer=None):
        super(LazySupervisedDataset, self).__init__()

        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            list_data_dict = json.load(open(data_path, "r", encoding="utf-8"))
        else:
            json_files = glob.glob(os.path.join(data_path, '*.json'))
            list_data_dict = []
            for json_file in json_files:
                list_data_dict += json.load(open(json_file, "r",encoding="utf-8"))

            jsonl_files = glob.glob(os.path.join(data_path, '*.jsonl'))
            for jsonl_file in jsonl_files:
                list_data_dict += json.load(open(jsonl_file, "r",encoding="utf-8"))
        
        # temp_list_data_dict = []
        # for item in list_data_dict:
        #     temp_list_data_dict.extend(item)
        # list_data_dict = temp_list_data_dict
        

        rank0_print("Formatting inputs...Skip in lazy mode")

        self.total_len = 1000
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

        # print(len(self.list_data_dict))

        self.data_args = data_args
        self.preprocess = img_preprocess
        self.image_root = data_args.image_folder
        self.max_length = data_args.max_seq_length
        self.base_length = data_args.base_seq_length
        self.box_image_size = data_args.box_image_size
        self.add_box_loss = data_args.add_box_loss




    def __len__(self):
        return len(self.list_data_dict)

    def rm_coco(self, curlist):

        new_list = [item for item in curlist if "coco" not in item['image']]
        return new_list


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        import time
        # s1 = time.time()
        caption = self.list_data_dict[i]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        
        # caption_short = caption.split(". ")[0]
        caption_short = self.list_data_dict[i]['conversations'][1]['short']
        caption_short = caption_short.replace("\n", " ")
        caption_short = caption_short.replace("\"","")
        caption_short = caption_short.replace("image","")
        caption_short = caption_short.replace("picture","")
        caption_short = caption_short.replace("photo","")
        caption_short = caption_short.replace("[", "")
        caption_short = caption_short.replace("]", "")
        caption_short = "a photo of "+caption_short
        
        lastname = self.list_data_dict[i]['image']

        if "coco" in lastname:
            image_name = "/mm-datasets/public/"+lastname
        elif "llava" in lastname:
            # llava/llava_pretrain/images/00273/002738569.jpg
            image_name = "/mm-datasets/public/LLaVA_data/pretrain/"+lastname.replace("llava/llava_pretrain/","")
        elif "sam" in lastname:
            image_name = "/mm-datasets/public/sam_pre50/"+lastname.split("/")[-1]
        elif "data-12m" in lastname:
            # data-12m/coyo_image_8/00052/000528224.jpg
            image_name = "/mm-datasets/public/grit-20m/"+lastname
        else:
            image_name = self.image_root + self.list_data_dict[i]['image']

        try:
            image = Image.open(image_name).convert("RGB")
        except:
            print("read from mm-datasets-lycc")
            image_name = image_name.replace("/mm-datasets","/mm-datasets-lycc")
            image = Image.open(image_name).convert("RGB")

 
        

        if self.box_image_size == 224:
            image = image.resize((self.box_image_size, self.box_image_size)) 
        else:
            width, height = image.size
            # 检查宽度和高度是否都小于3
            if width < 3 or height < 3:
                # 调整大小为336x336像素
                image = image.resize((self.box_image_size, self.box_image_size))

        # image = image.resize((self.box_image_size, self.box_image_size)) 
        image_tensor = self.preprocess.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # text = self.tokenizer(caption, truncate=True).to(device=image_tensor.device)
        text =  torch.tensor(self.tokenizer([caption], max_length=self.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)
        # short_text = self.tokenizer(caption_short, truncate=True).to(device=image_tensor.device)
        short_text = torch.tensor(self.tokenizer([caption_short], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)        


        if self.add_box_loss:
            box_images = []
            box_texts = []
            short_terms_max_similarity_boxes = self.list_data_dict[i]['short_terms_max_similarity_boxes']
            # crop box images

            for box_keyname in short_terms_max_similarity_boxes.keys():

                box = short_terms_max_similarity_boxes[box_keyname]["box"]

                # 获取图像的尺寸
                width, height = image.size
                # 将box信息转换为像素坐标

                similarity_value = short_terms_max_similarity_boxes[box_keyname]["similarity"]
                
                if similarity_value==0:
                    left = int(box[0] * width)
                    top = int(box[1] * height)
                    right = int(box[2] * width)
                    bottom = int(box[3] * height)
                else:
                    from torchvision.ops import box_convert
                    boxes = torch.tensor(box)
                    boxes = boxes * torch.Tensor([width, height, width, height])
                    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").tolist()
                    left,top,right,bottom = xyxy[0],xyxy[1],xyxy[2],xyxy[3]
                    

                # 确保坐标是正确的，左上角的坐标应该小于右下角的坐标
                if left >= right or top >= bottom:
                    raise ValueError("Box coordinates are invalid.")

                # 使用PIL的crop方法裁剪图像
                cropped_image = image.crop((left, top, right, bottom))
                cropped_image = cropped_image.convert("RGB")
                cropped_image = cropped_image.resize((self.box_image_size, self.box_image_size)) 
                box_image = self.preprocess.preprocess(cropped_image, return_tensors='pt', do_resize=False)['pixel_values'][0]
                
                # box_image = self.preprocess.preprocess(cropped_image, return_tensors='pt')['pixel_values'][0]
                # (cropped_image, return_tensors='pt', do_resize=False, do_center_crop=False)
                box_kn = box_keyname.replace("\n", " ")
                box_caption = "a photo of "+box_kn
                box_text = torch.tensor(self.tokenizer([box_caption], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)        
                box_images.append(box_image)
                box_texts.append(box_text)

            box_images = torch.stack(box_images)
            # 5*3*384*384
            box_texts = torch.cat(box_texts,dim=0)
            # 5*274


        data_dict = {}
        data_dict['image'] = image_tensor
        data_dict['text'] = text
        data_dict['short_text'] = short_text
        data_dict['add_box_loss'] = self.add_box_loss

        if self.add_box_loss:
            data_dict['box_images'] = box_images
            data_dict['box_texts'] = box_texts
            
        return data_dict

import torch.distributed as dist

def list_all_files(rootpath):
    _files = []

    for item_path in os.listdir(rootpath):
        for sub in os.listdir(os.path.join(rootpath,item_path)):
            jsonpath = os.path.join(rootpath,item_path,sub)
            _files.append(jsonpath)
    
    return _files




class Liaon2B_With_Longcaption(IterableDataset):
    def __init__(self, path, data_args,
                 img_preprocess=None,tokenizer=None) -> None:
        super().__init__()
        # self.args=args
        self.path=path
        # with open('../AIGC_list.txt','r') as f:
        #     self.sub_path_list=f.read().strip().split('\n')
        with open("/wangbin-home-shcdt/image_text_match/npu_longclip/code_test/train_use_list_add12M.json","r",encoding="utf-8") as f:
            self.sub_path_lists = json.load(f)
        # self.sub_path_list=list_all_files(self.path)
        # /lmm-shcdt-datasets/laion-2b-decompress-as-gzip/00037/01335.gz
        # random.shuffle(self.sub_path_list)

        # self.sub_path_list = self.sub_path_list[:1024]

        # self.tokenizer_clip,self.tokenizer_ali=tokenizer_list       
        self.processor = img_preprocess
        self.tokenizer = tokenizer
        self.max_length = data_args.max_seq_length
        self.base_length = data_args.base_seq_length
        self.box_image_size = data_args.box_image_size
        self.use_longcaption = data_args.use_longcaption
        self.laion_longcaption_root = data_args.laion_longcaption_root

        # bad_files = ["/lmm-shcdt-datasets/laion-2b-decompress-as-gzip/00112/00037.gz","/lmm-shcdt-datasets/laion-2b-decompress-as-gzip/00116/00408.gz","/lmm-shcdt-datasets/laion-2b-decompress-as-gzip/00050/00679.gz"]
        # # self.sub_path_list = self.sub_path_list-bad_files
        # set1 = set(self.sub_path_list)
        # set2 = set(bad_files)
        # self.sub_path_list = list(set1-set2)
        # self.random_seeds = [333,44,555,666,7777,8888,9999,567]
        
        # history_list = self.read_files_from_directory("/wangbin-home-shcdt/image_text_match/npu_longclip/laion_history")
        # history_list = list(set(history_list))

        # # last_list = self.sub_path_list-history_list
        # set1 = set(self.sub_path_list)
        # set2 = set(history_list)
        # self.sub_path_list = list(set1-set2)
        

        # self.sub_path_list = self.sub_path_list+history_list


        self.mult_GPU=True
        self.rank_res = 0
        self.world_size = 1
        try:
            self.rank_res = int(os.environ.get('RANK'))
            self.world_size = int(os.environ.get('WORLD_SIZE'))
            print('word_size, ', self.world_size)
            print('rank_res, ', self.rank_res)
            print('file_num, ', len(self.sub_path_lists[0]))
        except:
            self.mult_GPU=False

        # self.cur_rank_sublist = []

    def read_files_from_directory(self, directory_path):
        """
        遍历指定目录下的所有.txt文件读取文件内容
        并将内容按行分割后添加到列表中。
        
        :param directory_path: 要遍历的文件夹路径
        :return: 包含所有文件名的列表
        """
        # 初始化一个空列表用于存储文件名
        all_filenames = []

        # 检查给定路径是否存在且为文件夹
        if not os.path.isdir(directory_path):
            print(f"错误：'{directory_path}' 不是一个有效的文件夹路径")
            return all_filenames

        # 遍历文件夹
        for filename in os.listdir(directory_path):
            # 检查是否为.txt文件
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                # 打开文件并读取内容
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 按行读取文件内容，并去除每行末尾的换行符
                    lines = file.read().splitlines()
                    # 将读取到的行添加到列表中
                    all_filenames.extend(lines)

        return all_filenames

    def _sample_generator(self, intval_num):
        
        # WorkerInfo(id=0, num_workers=8, seed=5105788733417564587, dataset=<__main__.AIGC_dataset object at 0x7f07d5243820>)

        epoch_num = 0
        import copy
        
        while True:
            print("============================================= EOPCH START IN NUM: "+str(epoch_num)+" =============================================")

            cur_list = self.sub_path_lists[0]
            print(cur_list[:10])
                        
            worker_info = torch.utils.data.get_worker_info()
            # print(worker_info)
            for file_index in range(len(cur_list)):
                sub_path = cur_list[file_index]
                if file_index % (self.world_size*worker_info.num_workers) == worker_info.num_workers*intval_num+worker_info.id:                            
                    if not os.path.isfile(sub_path):
                        with open('bad_file.txt','a') as bf:
                            bf.write(sub_path+'\n')
                        continue

                    if self.use_longcaption and "laion-2b-decompress-as-gzip" in sub_path:
                        longcaption_lastname = sub_path.split("laion-2b-decompress-as-gzip/")[-1].replace(".gz",".json.gz")
                        # longcaption_gzip_save_path = "/mm-datasets/public/laion2b_longcaption_gzips/"+longcaption_lastname
                        longcaption_gzip_save_path = self.laion_longcaption_root+longcaption_lastname

                        if not os.path.isfile(longcaption_gzip_save_path):
                            with open('no_longcaption.txt','a') as ssf:
                                ssf.write(sub_path+'\n')
                            continue

                    try:
                        # subcaption
                        with gzip.open(sub_path,'r') as f:
                            # sub_path = "/lmm-shcdt-datasets/laion-2b-decompress-as-gzip/00000/00999.gz"
                            use_longcaption = self.use_longcaption
                            if use_longcaption and "laion-2b-decompress-as-gzip" in sub_path:
                                longcaption_lastname = sub_path.split("laion-2b-decompress-as-gzip/")[-1].replace(".gz",".json.gz")
                                # longcaption_gzip_save_path = "/mm-datasets/public/laion2b_longcaption_gzips/"+longcaption_lastname
                                longcaption_gzip_save_path = self.laion_longcaption_root+longcaption_lastname

                                if not os.path.isfile(longcaption_gzip_save_path):
                                    use_longcaption = False
                                else:
                                    with gzip.open(longcaption_gzip_save_path,'r') as jf:
                                        longcaption_json = jf.read()

                                    longcaption_dict = json.loads(longcaption_json)

                            

                            for line in f:
                                try:
                                    line = line.decode("utf-8").strip()
                                    datas = line.split("\t")

                                    imgkey, width, height, title = datas[:4]

                                    imgb64 = datas[-1]

                                    if len(datas)>5:
                                        title = "\t".join(datas[3:-1])
                                        # NOTE maybe string too long, set not more than 50
                                        if len(title)>50:
                                            title = title[:50]


                                    caption = title
                                    # urlsafe_
                                    # get long caption
                                    if use_longcaption:
                                        try:
                                            if "laion-2b-decompress-as-gzip" in sub_path:
                                                # "url": "http://i01.i.aliimg.com/img/pb/490/914/590/590914490_310.jpg"
                                                imgkey = imgkey.split('"')[3]
                                                # http://i01.i.aliimg.com/img/pb/490/914/590/590914490_310.jpg

                                                longcaption = longcaption_dict[imgkey]["long_caption"]
                                            else:
                                                longcaption = imgkey
                                        except:
                                            longcaption = caption+caption+caption+caption
                                            # continue
                                    else:
                                        longcaption = caption


                                    try:
                                        image_ = Image.open(BytesIO(base64.b64decode(imgb64)))
                                    except:
                                        image_ = Image.open(BytesIO(base64.urlsafe_b64decode(imgb64)))
            
                                    image_ = image_.convert('RGB')                         
                                    

                                    # if longcaption != caption:
                                    #     print(longcaption)

                                    

                                    if self.box_image_size > 336:
                                        image_ = image_.resize((self.box_image_size,self.box_image_size))
                                        image_tensor = self.processor.preprocess(image_, return_tensors='pt', do_resize=False, do_center_crop=False)['pixel_values'][0]
                                        # image_tensor = self.processor.preprocess(image_, return_tensors='pt')['pixel_values'][0]
                                    else:
                                        if self.box_image_size == 224:
                                            image_ = image_.resize((self.box_image_size,self.box_image_size))

                                        try:
                                            image_tensor = self.processor.preprocess(image_, return_tensors='pt')['pixel_values'][0]
                                        except:
                                            image_ = image_.resize((self.box_image_size,self.box_image_size))
                                            image_tensor = self.processor.preprocess(image_, return_tensors='pt')['pixel_values'][0]


                                    text =  torch.tensor(self.tokenizer([longcaption], max_length=self.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)
                                    short_text = torch.tensor(self.tokenizer([caption], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)        
                                    
                                    del image_
                                    del longcaption
                                    del caption
                                    del line


                                    data_dict = {}
                                    data_dict['image'] = image_tensor
                                    data_dict['text'] = text
                                    data_dict['short_text'] = short_text


                                    yield data_dict
                                except Exception as e:
                                    print(e)
                                    continue
                                    
                            
                            if use_longcaption:
                                del longcaption_dict

                            # collected = gc.collect()
                            # print(f"Number of objects collected: {collected}")


                            id_v = str(worker_info.id)
                            history_root = "/wangbin-home-shcdt/image_text_match/npu_longclip/laion_history_1/"
                            write_in_filename = history_root+"rank_"+str(intval_num)+"_id_"+str(id_v)+"_usedfilename.txt"
                            with open(write_in_filename,'a') as wif:
                                wif.write(sub_path+'\n')

                    except Exception as e:
                        with open('bad_file.txt','a') as ebf:
                            ebf.write(sub_path+'\n')
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! failed whole one gzip file!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print(e)

            epoch_num+=1
    def __iter__(self):
        sample_iterator = self._sample_generator(self.rank_res)
        return sample_iterator


from prefetch_generator import BackgroundGenerator



import io
import webdataset as wds


def make_my_webdata(data_path: str, data_args: DataArguments,
                 img_preprocess=None,
                 tokenizer=None):

    image_folder="/mm-datasets/public/refine_longclip_webdata/{1..10}.tar"
    transform=None
    image_root = data_args.image_folder

    def to_item(sample):


        image = sample[0]
        jsonfile = sample[1]
        
        caption = jsonfile['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        # caption_short = caption.split(". ")[0]
        caption_short = jsonfile['conversations'][1]['short']
        caption_short = caption_short.replace("\n", " ")
        caption_short = caption_short.replace("\"","")
        caption_short = caption_short.replace("image","")
        caption_short = caption_short.replace("picture","")
        caption_short = caption_short.replace("photo","")
        caption_short = caption_short.replace("[", "")
        caption_short = caption_short.replace("]", "")
        caption_short = "a photo of "+caption_short
        
        image = image.convert("RGB")

        box_image_size = data_args.box_image_size

        if box_image_size == 224:
            image = image.resize((box_image_size, box_image_size)) 
        else:
            width, height = image.size
            # 检查宽度和高度是否都小于3
            if width < 3 or height < 3:
                # 调整大小为336x336像素
                image = image.resize((box_image_size, box_image_size))

        # image = image.resize((self.box_image_size, self.box_image_size)) 
        image_tensor = img_preprocess.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # text = self.tokenizer(caption, truncate=True).to(device=image_tensor.device)
        text =  torch.tensor(tokenizer([caption], max_length=data_args.max_seq_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)
        # short_text = self.tokenizer(caption_short, truncate=True).to(device=image_tensor.device)
        short_text = torch.tensor(tokenizer([caption_short], max_length=data_args.base_seq_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)        

        data_dict = {}
        data_dict['image'] = image_tensor
        data_dict['text'] = text
        data_dict['short_text'] = short_text

        return data_dict

    dataset = wds.DataPipeline(
            wds.ResampledShards(image_folder),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(2048, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(transform, handler=wds.warn_and_continue),
            wds.map(to_item, handler=wds.warn_and_continue),
        )

    return dataset


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        batch = {}
        images = [instance['image'] for instance in instances]
        batch['image'] = torch.stack(images)
        texts = [instance['text'] for instance in instances]
        batch['text_long'] = torch.cat(texts,dim=0)
        short_texts = [instance['short_text'] for instance in instances]
        batch['text_short'] = torch.cat(short_texts,dim=0)

        return batch


def make_supervised_data_module(data_args,img_preprocess,tokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    

    if data_args.train_with_laion:

        # sub_path_list = list_all_files(data_args.data_path)
        # bad_files = ["/lmm-shcdt-datasets/laion-2b-decompress-as-gzip/00112/00037.gz","/lmm-shcdt-datasets/laion-2b-decompress-as-gzip/00116/00408.gz","/lmm-shcdt-datasets/laion-2b-decompress-as-gzip/00050/00679.gz"]
        # # self.sub_path_list = self.sub_path_list-bad_files
        # set1 = set(sub_path_list)
        # set2 = set(bad_files)
        # sub_path_list = list(set1-set2)

        train_dataset = Liaon2B_With_Longcaption(
                                path=data_args.data_path,
                                data_args=data_args,
                                img_preprocess=img_preprocess,tokenizer=tokenizer,)
        # train_dataset = make_my_webdata(
        #                 data_path=data_args.data_path,
        #                 data_args=data_args,
        #                 img_preprocess=img_preprocess,tokenizer=tokenizer,)
    else:
        train_dataset = LazySupervisedDataset(
                                    data_path=data_args.data_path,
                                    data_args=data_args,
                                    img_preprocess=img_preprocess,tokenizer=tokenizer,)
        # train_dataset = make_my_webdata(
        #                 data_path=data_args.data_path,
        #                 data_args=data_args,
        #                 img_preprocess=img_preprocess,tokenizer=tokenizer,)

                                
    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    compute_dtype = torch.float32


    # Load pretrained model, tokenizer, and image processor
    from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPConfig



    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    # tokenizer.pad_token_id = 1


    image_processor = CLIPImageProcessor.from_pretrained(model_args.base_model)
    # cur_config = CLIPConfig.from_pretrained(model_args.base_model)
    # cur_config.text_config.eos_token_id = tokenizer.eos_token_id
    # cur_config.text_config.pad_token_id = tokenizer.pad_token_id
    # cur_config.text_config.bos_token_id = tokenizer.bos_token_id

    # print(tokenizer.pad_token_id,tokenizer.eos_token_id)
    
    # cur_config.text_config.max_position_embeddings = 248
    
    # model = LongCLIPModel.from_pretrained(model_args.base_model)
    model = LongCLIPModel.from_pretrained(model_args.model_name_or_path)
    # model = LongCLIPModel(cur_config)
    # model.apply(model._init_weights)
    
    config = model.config
    import numpy as np

    model.logit_scale = torch.nn.Parameter(torch.ones([]) * model_args.log_scale)  
    model.logit_scale_siglip = torch.nn.Parameter(torch.ones([])* np.log(10))
    model.logit_bias_siglip = torch.nn.Parameter(torch.ones([])*-10.0)

    model.resize_postion_embeding()
    model.copy_weight()
    model.text_only_long = training_args.text_only_long

    model.world_size = training_args.train_use_word_size
    model.pad_token_id = tokenizer.pad_token_id
    
    if data_args.box_image_size>336:
        model.interpolate_pos_encoding = True


    # if not model.text_only_long:
    #     # freeze old postion-embedding
    #     # pass
    #     for p in model.vision_model.parameters():
    #         p.requires_grad = False

    #     for p in model.visual_projection.parameters():
    #         p.requires_grad = False


    data_module = make_supervised_data_module(data_args=data_args,img_preprocess=image_processor,tokenizer=tokenizer,)
    
    model.to(dtype=compute_dtype, device=training_args.device)

    # old: --gradient_checkpointing_kwargs {"use_reentrant":True} \
    training_args.gradient_checkpointing_kwargs = {"use_reentrant":False}

    trainer = CLIPTrainer(model=model,
                        args=training_args,
                        **data_module)

    # while True:
    #     try:
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,output_dir=training_args.output_dir)

    # except:
    #     continue


if __name__ == "__main__":
    train()
