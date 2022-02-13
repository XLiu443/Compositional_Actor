# ACTOR

```
How to run


python -m src.train.train_cvae --modelname cvae_transformer_rc_rcxyz_kl --pose_rep rot6d --lambda_kl 1e-5 --jointstype vertices --batch_size 20 --num_frames 60 --num_layers 8 --lr 0.0001 --glob --no-vertstrans --dataset humanact12 --num_epochs 5000 --snapshot 100 --folder exps/humanact12

```
training只用Humanact12原始数据，没有合成的数据，目的是测试cat两个condition feature是否work

改动的地方都在 src/models/architectures/transformer.py

Encoder_TRANSFORMER里面

line 116\
mix_muquery = torch.cat( (self.muQuery[y], self.muQuery[y]), axis = 1) \
cat两个action(也就是y)的mu_token, mix_muquery的维度是20*512

line 117\
mix_sigmaquery =  torch.cat( (self.sigmaQuery[y], self.sigmaQuery[y]), axis = 1) \
cat两个action(也就是y)的sigma_token, mix_sigmaquery的维度是20*512

line 118 \
xseq = torch.cat((mix_muquery[None], mix_sigmaquery[None], x), axis=0) \
cat mu_token, sigma_token和x(输入的video, 维度是60*20*512)给transformer


Decoder_TRANSFORMER里面

line208 \
z_action = torch.cat( (self.actionBiases[y], self.actionBiases[y]), axis = 1) \
cat两个action的action_token, cat之后z_action的维度是20*512

line209 \
z = z + z_action
z是latent, 维度20*512


















