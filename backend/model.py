import torch
import clip 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from collections import OrderedDict 
from torch import distributed as dist
from transformers import CLIPTokenizer


class GALIP_CONTAINER:
    def __init__(self, netG, netD, netC, text_encoder, image_encoder, optimizerG, optimizerD, scaler_G, scaler_D, device):
        self.netG = netG
        self.netD = netD
        self.netC = netC
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.scaler_G = scaler_G
        self.scaler_D = scaler_D
        self.device = device


class GalipModelSingleton:
    _instance = None

    def __new__(cls, model_path, clip_model, *args, **kwargs):
        if cls._instance is None:
            # Create the instance if not already created
            cls._instance = super(GalipModelSingleton, cls).__new__(cls)
            cls._instance.galip_container = initialize_gan_model(clip_model)
            cls._instance.epoch = load_model_galip(cls._instance.galip_container, model_path)
            cls._instance.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        return cls._instance

    def get_model(self):
        return self.galip_container, self.tokenizer




class CLIP_Model:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            # Instantiate the CLIP model only once
            cls._instance = super(CLIP_Model, cls).__new__(cls)
            cls.device = "cpu"
            cls.model, cls.preprocessor = clip.load("ViT-B/32", device=cls.device)
        return cls._instance

    @classmethod
    def get_model(cls):
        """Returns the CLIP model and preprocessor."""
        if cls._instance is None:
            cls()  # Ensure instance is created
        return cls._instance.model, cls._instance.preprocessor
    
class CLIP_IMG_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_IMG_ENCODER, self).__init__()
        model = CLIP.visual
        # print(model)
        self.define_module(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, model):
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer
        self.ln_post = model.ln_post
        self.proj = model.proj

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def transf_to_CLIP_input(self,inputs):
        device = inputs.device
        if len(inputs.size()) != 4:
            raise ValueError('Expect the (B, C, X, Y) tensor.')
        else:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
            inputs = ((inputs+1)*0.5-mean)/var
            return inputs

    def forward(self, img: torch.Tensor):
        x = self.transf_to_CLIP_input(img)
        x = x.type(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid =  x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
#         selected = [1,4,8]
        selected = [2,3,4,6,8,10]
        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                local_features.append(x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return torch.stack(local_features, dim=1), x.type(img.dtype)

class CLIP_TXT_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_TXT_ENCODER, self).__init__()
        self.define_module(CLIP)
        # print(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, CLIP):
        self.transformer = CLIP.transformer
        self.vocab_size = CLIP.vocab_size
        self.token_embedding = CLIP.token_embedding
        self.positional_embedding = CLIP.positional_embedding
        self.ln_final = CLIP.ln_final
        self.text_projection = CLIP.text_projection
        self.context_length = self.positional_embedding.shape[0]

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        sent_emb = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return sent_emb, x
    
class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, imsize):
        super(G_Block, self).__init__()
        self.imsize = imsize
        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, h, y):
        h = F.interpolate(h, size=(self.imsize, self.imsize))
        return self.shortcut(h) + self.residual(h, y)


class D_Block(nn.Module):
    def __init__(self, fin, fout, k, s, p, res, CLIP_feat):
        super(D_Block, self).__init__()
        self.res, self.CLIP_feat = res, CLIP_feat
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        if self.res==True:
            self.gamma = nn.Parameter(torch.zeros(1))
        if self.CLIP_feat==True:
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, CLIP_feat=None):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        
        if (self.res==True)and(self.CLIP_feat==True):
            return x + self.gamma*res + self.beta*CLIP_feat
        elif (self.res==True)and(self.CLIP_feat!=True):
            return x + self.gamma*res
        elif (self.res!=True)and(self.CLIP_feat==True):
            return x + self.beta*CLIP_feat
        else:
            return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return h


class M_Block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, cond_dim, k, s, p):
        super(M_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, k, s, p)
        self.fuse1 = DFBLK(cond_dim, mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, k, s, p)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        self.learnable_sc = in_ch != out_ch
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, text):
        h = self.conv1(h)
        h = self.fuse1(h, text)
        h = self.conv2(h)
        h = self.fuse2(h, text)
        return h

    def forward(self, h, c):
        return self.shortcut(h) + self.residual(h, c)

class CLIP_Mapper(nn.Module):
  def __init__(self,CLIP):
    super(CLIP_Mapper, self).__init__()
    model = CLIP.visual
    # print(model)
    self.define_module(model)
    for param in self.parameters():
      param.requires_grad = False

  def define_module(self,model):
    self.conv1 = model.conv1
    self.class_embedding = model.class_embedding
    self.positional_embedding = model.positional_embedding
    self.ln_pre = model.ln_pre
    self.transformer = model.transformer

  @property
  def dtype(self):
    return self.conv1.weight.dtype


  def forward(self,img:torch.Tensor,prompts:torch.Tensor):
    x = img.type(self.dtype)
    prompts = prompts.type(self.dtype)
    grid = x.size(-1)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
    # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)
    # NLD -> LND
    x = x.permute(1, 0, 2)
    # Local features
    selected = [1,4,7,12]
#     selected = [1,2,3,4,5,6,7,8]
    begin,end = 0 , 12
    prompt_idx = 0
    for i in range(begin, end):
            if i in selected:
                prompt = prompts[:,prompt_idx,:].unsqueeze(0)
                prompt_idx = prompt_idx+1
                x = torch.cat((x,prompt), dim=0)
                x = self.transformer.resblocks[i](x)
                x = x[:-1,:,:]
            else:
                x = self.transformer.resblocks[i](x)
    return x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype)

class CLIP_Adapter(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, G_ch, CLIP_ch, cond_dim, k, s, p, map_num, CLIP):
        super(CLIP_Adapter, self).__init__()
        self.CLIP_ch = CLIP_ch
        self.FBlocks = nn.ModuleList([])
        self.FBlocks.append(M_Block(in_ch, mid_ch, out_ch, cond_dim, k, s, p))
        for i in range(map_num-1):
            self.FBlocks.append(M_Block(out_ch, mid_ch, out_ch, cond_dim, k, s, p))
        self.conv_fuse = nn.Conv2d(out_ch, CLIP_ch, 5, 1, 2)
        self.CLIP_ViT = CLIP_Mapper(CLIP)
        self.conv = nn.Conv2d(768, G_ch, 5, 1, 2)
        #
        self.fc_prompt = nn.Linear(cond_dim, CLIP_ch*8)

    def forward(self,out,c):
        prompts = self.fc_prompt(c).view(c.size(0),-1,self.CLIP_ch)
        for FBlock in self.FBlocks:
            out = FBlock(out,c)
        fuse_feat = self.conv_fuse(out)
        map_feat = self.CLIP_ViT(fuse_feat,prompts)
        return self.conv(fuse_feat+0.1*map_feat)


class NetG(nn.Module):
  def __init__(self,ngf,nz,cond_dim,imsize,ch_size,mixed_precision,CLIP):
    super(NetG, self).__init__()
    self.ngf = ngf
    self.mixed_precision = mixed_precision
    self.code_sz,self.code_ch,self.mid_ch = 7,64,32
    self.CLIP_ch = 768
    self.fc_code = nn.Linear(nz,self.code_sz*self.code_sz*self.code_ch)
    self.mapping = CLIP_Adapter(self.code_ch, self.mid_ch, self.code_ch, ngf*8, self.CLIP_ch, cond_dim+nz, 3, 1, 1, 4, CLIP)
    self.GBlocks = nn.ModuleList([])
    in_out_pairs = list(get_G_in_out_chs(ngf, imsize))
    imsize = 4
    for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
        if idx<(len(in_out_pairs)-1):
            imsize = imsize*2
        else:
            imsize = 224
        self.GBlocks.append(G_Block(cond_dim+nz, in_ch, out_ch, imsize))
    self.to_rgb = nn.Sequential(
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(out_ch, ch_size, 3, 1, 1),
    )

  def forward(self, noise, c, eval=False):
    with torch.cuda.amp.autocast() if self.mixed_precision and not eval else dummy_context_mgr() as mp:
      cond = torch.cat((noise, c), dim=1)
      out = self.mapping(self.fc_code(noise).view(noise.size(0), self.code_ch, self.code_sz, self.code_sz), cond)
      # fuse text and visual features
      for GBlock in self.GBlocks:
          out = GBlock(out, cond)
      # convert to RGB image
      out = self.to_rgb(out)
    return out


class NetD(nn.Module):
    def __init__(self, ndf, imsize, ch_size, mixed_precision):
        super(NetD, self).__init__()
        self.mixed_precision = mixed_precision
        self.DBlocks = nn.ModuleList([
            D_Block(768, 768, 3, 1, 1, res=True, CLIP_feat=True),
            D_Block(768, 768, 3, 1, 1, res=True, CLIP_feat=True),
        ])
        self.main = D_Block(768, 512, 3, 1, 1, res=True, CLIP_feat=False)

    def forward(self, h):
        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
            out = h[:,0]
            for idx in range(len(self.DBlocks)):
                out = self.DBlocks[idx](out, h[:,idx+1])
            out = self.main(out)
        return out
    

class NetC(nn.Module):
    def __init__(self, ndf, cond_dim, mixed_precision):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.mixed_precision = mixed_precision
        self.joint_conv = nn.Sequential(
            nn.Conv2d(512+512, 128, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            )

    def forward(self, out, cond):
        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
            cond = cond.view(-1, self.cond_dim, 1, 1)
            cond = cond.repeat(1, 1, 7, 7)
            h_c_code = torch.cat((out, cond), 1)
            out = self.joint_conv(h_c_code)
        return out
    

def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs

def prepare_data(data, text_encoder, device):
    imgs, captions, CLIP_tokens = data
    imgs, CLIP_tokens = imgs.to(device), CLIP_tokens.to(device)
    sent_emb, words_embs = encode_tokens(text_encoder, CLIP_tokens)
    return imgs, captions, CLIP_tokens, sent_emb, words_embs


def encode_tokens(text_encoder, caption):
    # encode text
    with torch.no_grad():
        sent_emb,words_embs = text_encoder(caption)
        sent_emb,words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs

def predict_loss(predictor, img_feature, text_feature, negtive):
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err


def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.mean(F.relu(1. - output))
    else:
        err = torch.mean(F.relu(1. + output))
    return err

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def MA_GP_MP(img, sent, out, scaler):
    grads = torch.autograd.grad(outputs=scaler.scale(out),
                            inputs=(img, sent),
                            grad_outputs=torch.ones_like(out),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    inv_scale = 1./(scaler.get_scale()+float("1e-8"))
    #inv_scale = 1./scaler.get_scale()
    grads = [grad * inv_scale for grad in grads]
    with torch.cuda.amp.autocast():
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0,grad1),dim=1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def MA_GP_FP32(img, sent, out):
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp

class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
    

def initialize_gan_model(CLIP, 
                              ngf=64, 
                              nz=100, 
                              cond_dim=512, 
                              imsize=224, 
                              ch_size=3, 
                              mixed_precision=True, 
                              ndf=64, 
                              lr_g=0.0001, 
                              lr_d=0.0004, 
                              betas=(0.0, 0.9)) -> GALIP_CONTAINER:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    if torch.cuda.is_available():
        netG = nn.DataParallel(NetG(ngf, nz, cond_dim, imsize, ch_size, mixed_precision, CLIP))
        netD = nn.DataParallel(NetD(ndf, imsize, ch_size, mixed_precision))
        netC = nn.DataParallel(NetC(ndf, cond_dim, mixed_precision))
    else:
        netG = NetG(ngf, nz, cond_dim, imsize, ch_size, mixed_precision, CLIP)
        netD = NetD(ndf, imsize, ch_size, mixed_precision)
        netC = NetC(ndf, cond_dim, mixed_precision)
    
    # Initialize text and image encoders
    text_encoder = CLIP_TXT_ENCODER(CLIP)
    image_encoder = CLIP_IMG_ENCODER(CLIP)
    
    # Move models to device and set precision
    netG.float().to(device)
    netD.float().to(device)
    netC.float().to(device)
    
    # Initialize optimizers
    D_params = list(netD.parameters()) + list(netC.parameters())
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=betas)
    optimizerD = torch.optim.Adam(D_params, lr=lr_d, betas=betas)
    
    # Initialize scalers for mixed precision training
    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()

    # Return the models, encoders, optimizers, and scalers
    return GALIP_CONTAINER (
        netG,
        netD,
        netC,
        text_encoder,
        image_encoder,
        optimizerG,
        optimizerD,
        scaler_G,
        scaler_D,
        device
    )


def load_model_galip(components, load_path):
    checkpoint = torch.load(load_path, map_location=torch.device(components.device))
    
    # Loading model components
    components.netG.load_state_dict(checkpoint['model']['netG'], strict=False)
    components.netD.load_state_dict(checkpoint['model']['netD'], strict=False)
    components.netC.load_state_dict(checkpoint['model']['netC'], strict=False)
    components.text_encoder.load_state_dict(checkpoint['model']['text_encoder'], strict=False)
    components.image_encoder.load_state_dict(checkpoint['model']['image_encoder'], strict=False)

    # Loading optimizers
    components.optimizerG.load_state_dict(checkpoint['optimizers']['optimizer_G'])
    components.optimizerD.load_state_dict(checkpoint['optimizers']['optimizer_D'])
    
    return checkpoint['epoch']

