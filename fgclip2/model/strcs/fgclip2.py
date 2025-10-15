import torch
import torch.nn as nn
import math

from typing import Any, Optional, Tuple, Union
import torch.distributed.nn as nn_dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union

import torch.distributed as dist
from torch.nn import AvgPool2d


from .modeling_fgclip2 import Fgclip2TextModel,Fgclip2VisionModel,Fgclip2Model,Fgclip2MultiheadAttentionPoolingHead,Fgclip2Output
from .configuration_fgclip2 import Fgclip2Config, Fgclip2TextConfig, Fgclip2VisionConfig
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
import math
from torchvision.ops import roi_align
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class PNTextLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PNTextLoss, self).__init__()

    def forward(self, inputs):            
        sim_sum = torch.sum(inputs.exp(), dim=1)
        loss = -1.0*torch.log(1.0/ sim_sum)
        loss = torch.mean(loss)
        return loss



class FG_CLIP2_Model(Fgclip2Model):
    config_class = Fgclip2Config
    main_input_name = "text_long"

    def __init__(self, config: Fgclip2Config):
        super().__init__(config)

        text_config = config.text_config
        vision_config = config.vision_config

        r'''
        # First, initialize the text and vision models with proper attention implementation
        # defalut is sdpa
        # NOTE If you need to train FG_CLIP2_Model and your device supports flash_attn, you can open them! 
        text_config._attn_implementation = "flash_attention_2"
        vision_config._attn_implementation = "flash_attention_2"

        '''
        
        text_model = Fgclip2TextModel._from_config(text_config)
        vision_model = Fgclip2VisionModel._from_config(vision_config)

        # Second, get the text and vision submodules (for backward compatibility)
        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model
        self.vision_config = vision_config

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))


        self.embed_dim = text_config.hidden_size

        self.longtext_head = nn.Linear(self.embed_dim, self.embed_dim)
        self.boxtext_head = nn.Linear(self.embed_dim, self.embed_dim)
        self.dense_feature_head = Fgclip2MultiheadAttentionPoolingHead(vision_config)

        # Initialize weights and apply final processing
        self.thresholds = 0.0
        self.pad_token_id = 0
        self.world_size = 0
        self.loss_type = None

        
        # Initialize weights and apply final processing
        self.post_init()


    def copy_weight(self,):
        with torch.no_grad():
            self.longtext_head.weight.data.copy_(self.text_model.head.weight.data)
            self.longtext_head.bias.data.copy_(self.text_model.head.bias.data)
            self.boxtext_head.weight.data.copy_(self.text_model.head.weight.data)
            self.boxtext_head.bias.data.copy_(self.text_model.head.bias.data)

    def copy_dense_feature_head(self,):
        with torch.no_grad():
            self.dense_feature_head.load_state_dict(self.vision_model.head.state_dict())

    def resize_postion_embeding(self, newsize=196):

        old_position_embedding = self.text_model.embeddings.position_embedding
        old_position_embedding_res = self.text_model.embeddings.position_embedding_res
        old_position_embedding_ori = self.text_model.embeddings.position_embedding_ori
        
        
        positional_embedding_pre = self.text_model.embeddings.position_embedding.weight.data

        length, dim = positional_embedding_pre.shape
        keep_len = 20
        posisitonal_embedding_new = torch.zeros([4*length-3*keep_len, dim], dtype=positional_embedding_pre.dtype)
        for i in range(keep_len):
            posisitonal_embedding_new[i] = positional_embedding_pre[i]
        for i in range(length-1-keep_len):
            posisitonal_embedding_new[4*i + keep_len] = positional_embedding_pre[i + keep_len]
            posisitonal_embedding_new[4*i + 1 + keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4

        posisitonal_embedding_new[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
                
        positional_embedding_res = posisitonal_embedding_new.clone()

        self.text_model.embeddings.position_embedding_ori.weight.data = posisitonal_embedding_new
        self.text_model.embeddings.position_embedding_ori.num_embeddings = posisitonal_embedding_new.shape[0]
        
        self.text_model.embeddings.position_embedding_res.weight.data = positional_embedding_res
        self.text_model.embeddings.position_embedding_res.num_embeddings = positional_embedding_res.shape[0]

        old_position_embedding_ori_requires_grad = old_position_embedding_ori.weight.requires_grad
        self.text_model.embeddings.position_embedding_ori.requires_grad_(old_position_embedding_ori_requires_grad)

        old_position_embedding_res_requires_grad = old_position_embedding_res.weight.requires_grad
        self.text_model.embeddings.position_embedding_res.requires_grad_(old_position_embedding_res_requires_grad)



    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        need_densefeature: Optional[bool] = None,
    ) -> torch.FloatTensor:

        # Use Fgclip2Model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        
        pooled_output = vision_outputs[1]
        if need_densefeature:
            return pooled_output,self.get_dense_feature(vision_outputs.last_hidden_state,pixel_attention_mask)
        else:
            return pooled_output


    def get_image_box_roi_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        box_info=None,
        box_mask = None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        feature_map = vision_outputs.last_hidden_state#[:, 1:, :]
        # ,attention_mask=pixel_attention_mask
        feature_map = self.get_dense_feature(feature_map,attention_mask=pixel_attention_mask)

        bs = pixel_values.shape[0]
        real_feature_maps = []
        for bs_index in range(bs):
            spatial_values = spatial_shapes[bs_index]
            real_h = spatial_values[0].item()
            real_w = spatial_values[1].item()
            real_pixel_tokens_num = real_w*real_h
            real_feature_map = feature_map[bs_index][:real_pixel_tokens_num]
            real_feature_map = real_feature_map.view(1,real_h,real_w,-1)
            real_feature_map = real_feature_map.permute(0, 3, 1, 2)
            real_feature_maps.append(real_feature_map)


        x_roi_list = []

        for bs_index in range(bs):
            real_feature_map = real_feature_maps[bs_index]
            cur_x_roi = roi_align(real_feature_map.type(torch.float32), box_info[bs_index], (1, 1), 1.0, -1, True)[..., 0, 0]
            x_roi_list.append(cur_x_roi)

        x_rois = torch.cat(x_roi_list,dim=0)
        # """
        x_rois = x_rois / x_rois.norm(p=2, dim=-1, keepdim=True)


        return x_rois



    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        
        denormed_boxes = []
        # print("normed_boxes, ", normed_boxes.shape)
        for boxes in normed_boxes:
            # print("boxes, ", boxes)
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes.type(torch.float32))
        return denormed_boxes



    def get_dense_feature(self,feature_map,attention_mask=None):

        probe = feature_map
        hidden_state = feature_map


        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            # print(attention_mask.shape) [1, 1, 256, 256]
            attention_mask = attention_mask.repeat(1, self.dense_feature_head.num_heads, 1, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.dense_feature_head.attention(probe,hidden_state,hidden_state, attn_mask=attention_mask)[0]
        residual = hidden_state
        hidden_state = self.dense_feature_head.layernorm(hidden_state)
        hidden_state = residual+self.dense_feature_head.mlp(hidden_state)

        feature_map = hidden_state

        return feature_map

    def forward(
        self,
        text_short: Optional[torch.LongTensor] = None,
        text_long: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        box_infos: Optional[torch.FloatTensor] = None,
        box_images: Optional[torch.FloatTensor] = None,
        box_texts: Optional[torch.LongTensor] = None,
        box_nums: Optional[torch.LongTensor] = None,
        text_long_flag: Optional[torch.LongTensor] = None,
        hard_images: Optional[torch.FloatTensor] = None,
        hard_infos: Optional[torch.FloatTensor] = None,
        hard_texts: Optional[torch.LongTensor] = None,
        hard_nums: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        add_box_loss: bool = False,
        use_hard_neg: bool = False,
    ) -> Union[Tuple, Fgclip2Output]:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        rank = dist.get_rank()
         
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        
        short_text_outputs = self.text_model(
                input_ids=text_short,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        short_text_embeds = short_text_outputs[1]
        short_text_embeds = short_text_embeds / short_text_embeds.norm(p=2, dim=-1, keepdim=True)


        if text_long is not None:
            long_text_outputs = self.text_model(
                    input_ids=text_long,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    walk_type="long",
                )
            long_text_embeds = long_text_outputs[1]
            long_text_embeds = self.longtext_head(long_text_embeds)
            long_text_embeds = long_text_embeds / long_text_embeds.norm(p=2, dim=-1, keepdim=True)
 

        image_embeds = vision_outputs[1]
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)


        if add_box_loss:
            box_valid_count_all = sum(int(box_nums[i]) for i in range(box_nums.shape[0]))
            if box_valid_count_all == 0:
                add_box_loss = False

        if use_hard_neg:
            hard_valid_count_all = sum(int(hard_nums[i]) for i in range(hard_nums.shape[0]))
            if hard_valid_count_all == 0:
                use_hard_neg = False

        if add_box_loss or use_hard_neg:

            feature_map = vision_outputs.last_hidden_state #[:, 1:, :]
            feature_map = self.get_dense_feature(feature_map,attention_mask=pixel_attention_mask)

            bs = pixel_values.shape[0]
            real_feature_maps = []
            for bs_index in range(bs):
                spatial_values = spatial_shapes[bs_index]
                real_h = spatial_values[0].item()
                real_w = spatial_values[1].item()
                real_pixel_tokens_num = real_w*real_h
                real_feature_map = feature_map[bs_index][:real_pixel_tokens_num]
                real_feature_map = real_feature_map.view(1,real_h,real_w,-1)
                real_feature_map = real_feature_map.permute(0, 3, 1, 2)
                real_feature_maps.append(real_feature_map)



        if add_box_loss:

            box_size = box_infos.shape[-1]
            box_infos = box_infos.reshape(bs, -1, box_size)

            x_roi_list = []

            for bs_index in range(bs):
                real_feature_map = real_feature_maps[bs_index]
                cur_box_infos = box_infos[bs_index].unsqueeze(dim=0)
                cur_original_bboxes = self._denormalize_boxes(cur_box_infos, real_feature_map)
                cur_x_roi = roi_align(real_feature_map.type(torch.float32), cur_original_bboxes, (1, 1), 1.0, -1, True)[..., 0, 0]
                x_roi_list.append(cur_x_roi)

            x_rois = torch.cat(x_roi_list,dim=0)

            bbox_vision_outputs = x_rois.type(torch.bfloat16)
            
            
            bbox_text_outputs = self.text_model(
                    input_ids=box_texts,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    walk_type="box",
                )


            bbox_text_embeds = bbox_text_outputs[1]
            bbox_text_embeds = self.boxtext_head(bbox_text_embeds)
            bbox_text_embeds = bbox_text_embeds / bbox_text_embeds.norm(p=2, dim=-1, keepdim=True)

            bbox_image_embeds = bbox_vision_outputs
            bbox_image_embeds = bbox_image_embeds / bbox_image_embeds.norm(p=2, dim=-1, keepdim=True)        
            

        if use_hard_neg:
            box_size = hard_infos.shape[-1]
            hard_infos = hard_infos.reshape(bs, -1, box_size)


            x_roi_list = []

            for bs_index in range(bs):
                real_feature_map = real_feature_maps[bs_index]
                cur_box_infos = hard_infos[bs_index].unsqueeze(dim=0)
                cur_original_bboxes = self._denormalize_boxes(cur_box_infos, real_feature_map)
                cur_x_roi = roi_align(real_feature_map.type(torch.float32), cur_original_bboxes, (1, 1), 1.0, -1, True)[..., 0, 0]
                x_roi_list.append(cur_x_roi)

            x_rois = torch.cat(x_roi_list,dim=0)


            # print("x_rois, ", x_rois.shape)
            hard_bbox_image_embeds = x_rois.type(torch.bfloat16)
            hard_bbox_image_embeds = hard_bbox_image_embeds / hard_bbox_image_embeds.norm(p=2, dim=-1, keepdim=True)

            hard_bbox_text_outputs = self.text_model(
                    input_ids=hard_texts,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    walk_type="box",
                )
            hard_bbox_text_embeds = hard_bbox_text_outputs[1]
            hard_bbox_text_embeds = self.boxtext_head(hard_bbox_text_embeds)
            hard_bbox_text_embeds = hard_bbox_text_embeds / hard_bbox_text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        

        logit_scale = self.logit_scale.exp()
        logit_bias = self.logit_bias
        if self.loss_type == "gather":
            if text_long is not None:
                loss_long = self.all_gather_siglip_loss_(image_embeds,long_text_embeds,logit_scale,logit_bias,rank)
            loss_short = self.all_gather_siglip_loss_(image_embeds,short_text_embeds,logit_scale,logit_bias,rank)
        elif self.loss_type == "reduce":
            if text_long is not None:
                loss_long = self.all_reduce_siglip_loss(image_embeds,long_text_embeds,logit_scale,logit_bias,rank)
            loss_short = self.all_reduce_siglip_loss(image_embeds,short_text_embeds,logit_scale,logit_bias,rank)
        else:
            assert self.loss_type is not None


        if text_long is not None:
            loss = loss_long+loss_short
        else:
            loss = loss_short
            return Fgclip2Output(
                loss=loss,
            )


        try:
            if add_box_loss:

                box_loss_weight = 0.2
                region_cc_loss_weight = 0.1
                distill_loss_weight = 0.4

                bs = box_nums.shape[0]
                bbox_size = int(bbox_text_embeds.shape[0]/bs)
                box_weight = torch.zeros([bs, bbox_size], device=bbox_text_embeds.device)
                for i in range(bs):
                    valid_count = int(box_nums[i])
                    box_weight[i][:valid_count] = 1
                box_weight = box_weight.reshape(1, bbox_text_embeds.shape[0]).squeeze()
                select_index = box_weight.nonzero()
                bbox_text_embeds = bbox_text_embeds[select_index,:].squeeze()
                bbox_image_embeds = bbox_image_embeds[select_index,:].squeeze()

                loss_bbox_itcl = self.pairwise_contrastive_loss(bbox_image_embeds, bbox_text_embeds, bbox_image_embeds.device, self.logit_scale_finegraind)
                loss_bbox_rcc = self.hard_category_contrastive_loss(bbox_text_embeds)
                loss = loss + box_loss_weight*loss_bbox_itcl + region_cc_loss_weight*loss_bbox_rcc



            if use_hard_neg:
                hard_box_loss_weight = 0.5

                bs = hard_nums.shape[0]
                bbox_size = int(hard_bbox_image_embeds.shape[0]/bs)
                box_weight = torch.zeros([bs, bbox_size], device=hard_bbox_image_embeds.device)
                for i in range(bs):
                    valid_count = int(hard_nums[i])
                    box_weight[i][:valid_count] = 1
                box_weight = box_weight.reshape(1, hard_bbox_image_embeds.shape[0]).squeeze()
                select_index = box_weight.nonzero()
                hard_bbox_image_embeds = hard_bbox_image_embeds[select_index,:].squeeze()
                loss_bbox_hitc, threshold= self.hard_contrastive_total_loss(hard_bbox_image_embeds, hard_bbox_text_embeds, hard_bbox_text_embeds.device, self.thresholds, self.logit_scale_hardneg)
                loss = loss + hard_box_loss_weight*loss_bbox_hitc
            else:
                threshold = self.thresholds

            sum_threshold = self.all_reduce_threshold(threshold)
            mean_threshold = sum_threshold/self.world_size

            upper_bound = 10
            self.thresholds=torch.clamp(mean_threshold,0,upper_bound).item()

        except:
            pass

        return Fgclip2Output(
            loss=loss,
        )


    def all_reduce_threshold(self,threshold):

        if not dist.is_initialized():
            raise RuntimeError("Distributed training is not initialized.")

        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, dtype=torch.float32, device='cuda')

        if threshold.dim() == 0:
            threshold = threshold.unsqueeze(0)

        dist.all_reduce(threshold, op=dist.ReduceOp.SUM)

        return threshold

    def pairwise_contrastive_loss(self, image_features_long, text_features_long, device, logit_scale=1.0):
        batch_size, c = image_features_long.shape
        labels = torch.eye(batch_size, device=device, dtype=torch.float)#.repeat(batch_size, 1)
        logits_per_image = logit_scale.exp() * image_features_long @ text_features_long.T
        logits_per_text = logit_scale.exp() * text_features_long @ image_features_long.T
        temp1 = F.cross_entropy(logits_per_text, labels)
        temp2 = F.cross_entropy(logits_per_image, labels)

        loss = (temp1+temp2)/2
        return loss

    def hard_category_contrastive_loss(self, text_features_long):
        batch_size, t_dim = text_features_long.shape
        similarity = text_features_long @ text_features_long.T
        eyeweights = torch.ones((batch_size, batch_size), device=text_features_long.device, dtype=torch.float)-torch.eye(batch_size, device=text_features_long.device, dtype=torch.float)
        similarity = torch.einsum('bp,bp->bp', eyeweights, similarity)
        weights = torch.zeros((batch_size, batch_size), device=text_features_long.device, dtype=torch.float)
        similarity = torch.where(similarity > 0.95, weights, similarity)
        values , indices = similarity.topk(10, dim=1, largest=True, sorted=True)
        criterion = PNTextLoss()
        loss = criterion(values)
        return loss
    
    def hard_contrastive_loss(self, image_features_long, text_features_long, device, logit_scale=1.0):
        batch_size, c = image_features_long.shape
        text_features_long = text_features_long.reshape(batch_size, 11, -1)
        labels = torch.zeros(batch_size, device=device, dtype=torch.long)#.repeat(batch_size, 1)
        predict = logit_scale.exp() * torch.einsum('bp,bdp->bd', image_features_long, text_features_long)
        loss = F.cross_entropy(predict, labels)
        return loss


    def hard_contrastive_total_loss(self, image_features, text_features, device, thresholds, logit_scale=1.0):
        batch_size, c = image_features.shape
        text_features = text_features.reshape(batch_size, 11, -1)
                
        gt_text_features=text_features[:,0,:]
        da_text_features=text_features[:,1:,:]
        reshape_da_text_features = da_text_features.reshape(-1, gt_text_features.shape[-1])
        
        all_text_features=torch.cat([gt_text_features, reshape_da_text_features])
        logits_per_image = logit_scale.exp() * image_features @ all_text_features.T
        logits_per_text = logit_scale.exp() * gt_text_features @ image_features.T
        
        da_logits_per_image= logit_scale.exp() * (da_text_features @ image_features.unsqueeze(-1)).squeeze()
        logits_per_image_gt = logit_scale.exp() * image_features @ gt_text_features.T
        logits_per_image_sda = torch.cat([logits_per_image_gt, da_logits_per_image], dim=-1)
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        pair_loss = self.hard_contrastive_loss(image_features, text_features, device, logit_scale)

        cmr_loss,thresholds=self.get_cmr_loss(logits_per_image,da_logits_per_image,thresholds)
        

        imc_loss_weight = 0.2
        cmr_loss_weight = 0.4
        contrastive_loss_weight = 0.6
        pair_loss_weight = 3 #
        total_loss = contrastive_loss_weight*loss + pair_loss_weight*pair_loss + cmr_loss*cmr_loss_weight# + imc_loss*imc_loss_weight
        return total_loss, thresholds

    def get_cmr_loss(self, gt_logits_per_image , da_logits_per_image, thresholds):
        # calculating cmr loss
        gt_similarity=gt_logits_per_image.diag().reshape(-1,1).expand(da_logits_per_image.shape)
        cmr_loss=nn.functional.relu((thresholds+da_logits_per_image-gt_similarity))

        mask = da_logits_per_image!=0
        average_similarity_for_types = (da_logits_per_image*mask).sum()/mask.sum()
        thresholds=gt_similarity.mean()-average_similarity_for_types
        return cmr_loss.mean(),thresholds

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            eyevalue = torch.eye(num_logits, device=device, dtype=torch.float)
            labels = 2 * eyevalue.bfloat16() + labels
        
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid((labels * logits).float()).sum() / image_features.shape[0]
        return loss


    def all_gather_siglip_loss_(self, image_features, text_features, logit_scale, logit_bias, cur_rank, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        text_features_all = torch.stack(nn_dist.all_gather(text_features), dim=0)

        for i in range(self.world_size):
            loss += float(i != cur_rank) * self._loss(
                image_features,
                text_features_all[i],
                logit_scale,
                logit_bias,
                negative_only=True,
            )

        return loss


    def all_reduce_siglip_loss(self, image_features, text_features, logit_scale, logit_bias, cur_rank, no_longtext_indices=None, output_dict=False):
        
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)


        for i in range(self.world_size):

            text_from_other = torch.distributed.nn.all_reduce(
                text_features * (cur_rank == i),
                torch.distributed.ReduceOp.SUM,
            )

            loss += float(i != cur_rank) * self._loss(
                image_features,
                text_from_other,
                logit_scale,
                logit_bias,
                negative_only=True,
            )

        return loss




