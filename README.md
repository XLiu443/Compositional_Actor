# ACTOR

```
How to run

python -m src.train.train_cvae --modelname cvae_transformer_rc_rcxyz_kl --pose_rep rot6d --lambda_kl 1e-5 --jointstype vertices --batch_size 12 --num_frames 60 --num_layers 8 --lr 0.0001 --glob --no-vertstrans --dataset humanact12 --num_epochs 5000 --snapshot 100 --folder exps/humanact12
```
用原始的数据+合成的数据training \
实现的思路是每个batch通过BatchSampler抽12个video作为原始数据，batch内通过collate_function合成12个video作为合成数据


入口文件 src/train/train_cvae.py \
line20 修改DataLoader\
BatchSampler抽12个video(3个不同的action, 每一个action抽4个video) \
collate_function重新封装batch, 一个batch包括12个原始的video+12个合成的video, 一共24个video


###
BatchSampler定义 src/samplers/batch_sampler.py \
在Reid RandomIdentitySampler(Sampler)基础上修改，RandomIdentitySampler目标是抽N个人，每个人K张图片，一个batch=N*K。修改后BatchSampler抽12个video(3个不同的action, 每一个action抽4个video)

###
collate_function定义 src/utils/tensors.py 

line43\
字典humanact12_action_mask定义12类action的mask


line60 \
增加video_combine(video1, video2, action1, action2, alpha)合成video1和video2


line95 \
collate_function(batch)重新封装batch 


###
其他的修改都在src/models/architectures/transformer.py 

Encoder_TRANSFORMER里面

line95\
x, y1, y2, mask, alphas = batch["x"], batch["y1"], batch["y2"], batch["mask"], batch["alphas"]
拿到batch里面的video数据和action label


line 120\
mix_muquery = torch.cat( (self.muQuery[y1], self.muQuery[y2]), axis = 1) \
cat两个action(也就是y1 y2)的mu_token, 比如y1=[0, 0, 0, 0, 6, 6, 6, 6, 5, 5, 5, 5, 0, 0, 0, 0, 6, 6, 6, 6, 5, 5, 5, 5]
y2 = [6, 6, 5, 5, 5, 5, 0, 5, 0, 0, 0, 6, 0, 0, 0, 0, 6, 6, 6, 6, 5, 5, 5, 5]
前12个元素表示合成video使用哪两个action(比如y1=0,y2=6), 后面12个元素是原始video的action，所以y1和y2一样


line 123\
mix_sigmaquery =  torch.cat( (self.sigmaQuery[y1], self.sigmaQuery[y2]), axis = 1) \
cat两个action(也就是y1,, y2)的sigma_token


line 124 \
xseq = torch.cat((mix_muquery[None], mix_sigmaquery[None], x), axis=0) \
cat mu_token, sigma_token和x(输入的video, 维度是60*24 *512)给transformer


Decoder_TRANSFORMER里面

line196 \
z, y1, y2, mask, lengths, alphas = batch["z"], batch["y1"], batch["y2"], batch["mask"], batch["lengths"], batch["alphas"]
拿到latent z和action labely1 y2


line217 \
z_action = torch.cat( (self.actionBiases[y1], self.actionBiases[y2]), axis = 1) \
cat两个action的action_token, cat之后z_action的维度是24*512


line218 \
z = z + z_action
z是latent, 维度24*512























