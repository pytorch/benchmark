import os
import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks.tensorboard import *
from fastai.vision.gan import *
from .deoldify.generators import *
from .deoldify.critics import *
from .deoldify.dataset import *
from .deoldify.save import *
from .deoldify.filters import MasterFilter, ColorizerFilter
from .deoldify.visualize import ModelImageVisualizer
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile

from .deoldify import mydevice
from .deoldify.device_id import DeviceId

path = Path(__file__).parent/'train_data'
path_hr = path
path_lr = path/'bandw'

proj_id = 'StableModel'

gen_name = proj_id + '_gen'
pre_gen_name = gen_name + '_0'
crit_name = proj_id + '_crit'

name_gen = proj_id + '_image_gen'
path_gen = path/name_gen

nf_factor = 2
pct_start = 1e-8

def get_data(bs:int, sz:int, keep_pct:float):
    return get_colorize_data(sz=sz, bs=bs, crappy_path=path_lr, good_path=path_hr, 
                             random_seed=None, keep_pct=keep_pct)

def get_crit_data(classes, bs, sz):
    src = ImageList.from_folder(path, include=classes, recurse=True).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=sz)
           .databunch(bs=bs).normalize(imagenet_stats))
    return data

def create_training_images(fn,i):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn).convert('LA').convert('RGB')
    img.save(dest)  
    
def save_preds(dl):
    i=0
    names = dl.dataset.items
    
    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1
    
def save_gen_images():
    if path_gen.exists(): shutil.rmtree(path_gen)
    path_gen.mkdir(exist_ok=True)
    data_gen = get_data(bs=bs, sz=sz, keep_pct=0.085)
    save_preds(data_gen.fix_dl)
    PIL.Image.open(path_gen.ls()[0])

class Model:
    def __init__(self, device=None, jit=False):
        self.device = device
        self.jit = jit
        if self.device=='cuda':
            mydevice.set(device=DeviceId.GPU0)
        elif self.device=='cpu': 
            mydevice.set(device=DeviceId.CPU)
        from .deoldify.loss import FeatureLoss, WassFeatureLoss 

        # Create black and white training images
        if not path_lr.exists():
            il = ImageList.from_folder(path_hr)
            parallel(create_training_images, il.items)
        # Create model
        bs=10
        sz=64
        keep_pct=1.0
        data_gen = get_colorize_data(sz=sz, bs=bs, crappy_path=path_lr, good_path=path_hr, 
                             random_seed=None, keep_pct=keep_pct)
        learn_gen = gen_learner_wide(data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor)
        self.module = learn_gen 

        
    def get_module(self):
        raise NotImplementedError()

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        self.gen_loss = F.l1_loss
        self.arch = models.resnet101
        self.module.model.eval()

        render_factor = 35
        source_path = str(pathlib.Path(__file__).parent)+'/test_images/image_1.jpg'
        results_dir = str(pathlib.Path(__file__).parent)+'/result_images'
        filtr = MasterFilter([ColorizerFilter(learn=self.module)], render_factor=render_factor)
        vis = ModelImageVisualizer(filtr, results_dir=results_dir)
        for _ in range(niter):
            vis.plot_transformed_image(path=source_path, render_factor=render_factor, compare=True)

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        self.module.fit_one_cycle(niter, pct_start=0.8, max_lr=slice(1e-3))

if __name__ == "__main__":
    m = Model(device='cuda', jit=False)
    m.train(niter=1)
    m.eval(niter=1)
