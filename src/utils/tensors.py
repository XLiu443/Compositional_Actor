import torch
import random
import ipdb
import copy
import numpy as np
import src.utils.rotation_conversions as geometry


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    lenbatch = [len(b[0][0][0]) for b in batch]
   # lenbatch = [len(b[0][0][0][0]) for b in batch]
    databatchTensor = collate_tensors(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
    return batch


humanact12_action_mask = {
    0: torch.tensor([0.6, 0.7, 0.7, 0.5, 0.8, 0.8, 0.5, 0.9, 0.9, 0.6, 1, 1, 0.5, 0.7, 0.7, 0.4, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    1: torch.tensor([0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    2: torch.tensor([0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.6, 0.6, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    3: torch.tensor([0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    4: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    5: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.9, 0.9, 1, 1, 1, 1]),
    6: torch.tensor([0.1, 0.8, 0.8, 0.2, 0.9, 0.9, 0.3, 1, 1, 0.4, 1, 1, 0.4, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    7: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    8: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    9: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.8, 0.8, 0.9, 0.9, 1, 1, 1, 1]),
    10: torch.tensor([0.1, 0.4, 0.4, 0.1, 0.4, 0.4, 0.1, 0.5, 0.5, 0.1, 0.6, 0.6, 0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 1, 1, 1, 1, 1, 1]),
    11: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 1, 1, 1, 1, 1, 1, 1, 1]),
}



def video_combine(video1, video2, action1, action2, alpha):               #video1 in shape 60*24*6

    old_mask1 = humanact12_action_mask[action1]                           #字典中找到mask1, mask2
    old_mask2 = humanact12_action_mask[action2]

    video_mask1 = alpha*old_mask1 / (alpha*old_mask1 + (1-alpha)*old_mask2)       #mask1, mask2做归一化
    video_mask2 = (1-alpha)*old_mask2 / (alpha*old_mask1 + (1-alpha)*old_mask2)   

    video_mask1 = video_mask1.unsqueeze(0).unsqueeze(2)                   #调整维度让maks1匹配视频每一帧，调整后video_mask1 in shape60*24*6 
    video_mask1 = video_mask1.repeat(60, 1, 6)
    video_mask2 = video_mask2.unsqueeze(0).unsqueeze(2)
    video_mask2 = video_mask2.repeat(60, 1, 6)

    video1_firstframe_mask = video_mask1[0,:,:]                           #取video_mask1第一帧
    video2_firstframe_mask = video_mask2[0,:,:]

    video1_frame = video1[0,:,:]                                          #取video1第一帧
    video1_frame = video1_frame.unsqueeze(0).repeat(60, 1, 1)             #调整video1_frame的维度到60*24*6
    video1_delta = video1 - video1_frame                                  #计算video1的delta, video1每一帧-video1第一帧, video1_delta in shape60*24*6
            
    
    video2_frame = video2[0,:,:]
    video2_frame = video2_frame.unsqueeze(0).repeat(60, 1, 1)
    video2_delta = video2 - video2_frame

    standardframe =  video1[0,:,:]*video1_firstframe_mask + video2[0,:,:]*video2_firstframe_mask     #计算标准帧，video1第一帧*mask1+video2第一帧*mask2
    standardframe = standardframe.unsqueeze(0).repeat(60, 1, 1)                                      #调整标准帧维度到60*24*6
 
    compose_delta = video1_delta * video_mask1 + video2_delta * video_mask2                         #合成数据, video1的delta*mask1 + video2的delta*mask2, 合成后的数据in shape60*24*6
    compose_delta = compose_delta + standardframe                                                   #加上标准帧成为最终的合成数据, shape60*24*6
    
    return compose_delta



def collate_function(batch): 
  #  ipdb.set_trace()
    databatch = [b[0] for b in batch]                   #databatch是12个原始video
    labelbatch = [b[1] for b in batch]                  #labelbatch是12个原始video的action
 #   ret_trbatch = [b[2] for b in batch]

    action_num = len(set(labelbatch))                   #action_num=一个batch有多少不同的action
    instance_num = len(databatch) // action_num         #instance_num=每个action有多少video
    generate_sequences = []
    actions1 = []
    actions2 = []
    alphas = []
    for i in range(action_num):                         #每个video都和随机选择的另一个action的video合成, 两个for循环遍历
        for j in range(instance_num): 
            alpha = 0.5                                 
            ind1 = i*instance_num + j                   #第一个video在databatch中的索引
    #        print("ind1",ind1) 
            video1 = databatch[ind1]                    #通过索引找到video1
            action1 = labelbatch[ind1]                  #找到video1的action
    #        print("action1",action1)
            avai_action = copy.deepcopy( list(  np.arange(action_num) ) )       
            avai_action.remove(i)                        
            ind2 = random.sample(avai_action, 1)[0]*instance_num + random.sample( list(np.arange(instance_num)), 1)[0]     #随机选择video2的索引, video2的action不等于video1的action
            video2 = databatch[ind2]                    #找到video2
            action2 = labelbatch[ind2]                  #video2的action
    #        print("action2",action2)
            new_sequence = video_combine(video1, video2, action1, action2, alpha)       #生成合成视频
            ret = new_sequence.permute(1, 2, 0).contiguous()                        
 
            generate_sequences.append(ret)
            #generate_sequences.append(new_sequence)
            actions1.append(action1)
            actions2.append(action2)
            alphas.append(alpha)
  #  ipdb.set_trace()
  #  generate_sequences = torch.stack(generate_sequences)
    actions1 = torch.tensor(actions1) 
    actions2 = torch.tensor(actions2)
    alphas =  torch.tensor(alphas)
 
  #  lenbatch = [len(b[0][0][0]) for b in batch]
    lenbatch = [len(batch[0][0]) for b in batch]
    databatchTensor = collate_tensors(generate_sequences)   #合成的video转化为tensor

    databatch_origin = []
    for i in range(len(databatch)):
        databatch_origin.append( databatch[i].permute(1, 2, 0).contiguous() )
    databatch_origin = collate_tensors(databatch_origin)    

    databatchTensor_origin = collate_tensors(databatch_origin)   #原始video转化为tensor
    labelbatchTensor_origin = torch.as_tensor(labelbatch)  

    databatchTensor_new = torch.cat( (databatchTensor, databatchTensor_origin), axis=0 )       #cat合成video和原始video
    actions1_new = torch.cat( (actions1, labelbatchTensor_origin), axis=0 )                    #cat合成video用到的action1和原始video的action
    actions2_new = torch.cat( (actions2, labelbatchTensor_origin), axis=0 )                    #cat合成video用到的action2和原始video的action

    lenbatchTensor = torch.as_tensor(lenbatch)
    lenbatchTensor = lenbatchTensor.repeat(2)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)

    batch = {"x": databatchTensor_new, "y1": actions1_new, "y2": actions2_new,
             "mask": maskbatchTensor, "lengths": lenbatchTensor, "alphas": alphas}
  
    return batch
