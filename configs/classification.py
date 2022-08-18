from argparse import Namespace as _Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN as _IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD as _IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as _F

# =========> shared <=================================
seed = 42
size = 224
type = 'CLS'
# =========> dataset <=================================
data = _Namespace()
data.name = 'DefaultCLS'  # ['DefaultCLS', 'ImageFolderLMDB']
data.root = '/path/to/imagenet'
data.loader_type = 'pil'
data.sampler = 'naive'
data.nb_classes = 1000

data.train_transforms = [
	dict(type='timm_create_transform', input_size=size, is_training=True, color_jitter=0.4,
		 auto_augment='rand-m9-mstd0.5-inc1', interpolation='random',
		 re_prob=0.25, re_mode='pixel', re_count=1),
]
data.test_transforms = [
	dict(type='Resize', size=int(size / 0.875), interpolation=_F.InterpolationMode.BICUBIC),
	dict(type='CenterCrop', size=size),
	dict(type='ToTensor'),
	dict(type='Normalize', mean=_IMAGENET_DEFAULT_MEAN, std=_IMAGENET_DEFAULT_STD, inplace=True),
]

# =========> model <=================================
model = _Namespace()
model.name = 'eatformer_mobile'
# model.name = 'eatformer_lite'
# model.name = 'eatformer_tiny'
# model.name = 'eatformer_mini'
# model.name = 'eatformer_small'
# model.name = 'eatformer_medium'
# model.name = 'eatformer_base'

model.model_kwargs = dict(checkpoint_path='pretrained/eatformer_mobile.pth', ema=False, strict=True, num_classes=data.nb_classes)
# model.model_kwargs = dict(checkpoint_path='pretrained/eatformer_lite.pth', ema=False, strict=True, num_classes=data.nb_classes)
# model.model_kwargs = dict(checkpoint_path='pretrained/eatformer_tiny.pth', ema=False, strict=True, num_classes=data.nb_classes)
# model.model_kwargs = dict(checkpoint_path='pretrained/eatformer_mini.pth', ema=False, strict=True, num_classes=data.nb_classes)
# model.model_kwargs = dict(checkpoint_path='pretrained/eatformer_small.pth', ema=False, strict=True, num_classes=data.nb_classes)
# model.model_kwargs = dict(checkpoint_path='pretrained/eatformer_medium.pth', ema=False, strict=True, num_classes=data.nb_classes)
# model.model_kwargs = dict(checkpoint_path='pretrained/eatformer_base.pth', ema=False, strict=True, num_classes=data.nb_classes)

# =========> optimizer <=================================
optim = _Namespace()
optim.lr = 5e-4
optim.optim_kwargs = dict(name='adamw', betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, amsgrad=False)
# =========> trainer <=================================
trainer = _Namespace()
trainer.name = 'CLS'
trainer.checkpoint = 'runs'
trainer.resume_dir = ''  # a higher priority than model.model_kwargs['checkpoint_path'], e.g., CLS_eatformer_tiny_DefaultCLS_20220515-181212
trainer.cuda_deterministic = False
trainer.epoch_full = 300

trainer.scheduler_kwargs = dict(
	name='cosine', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=optim.lr / 100,
	warmup_lr=optim.lr / 1000, warmup_iters=-1, cooldown_iters=0, warmup_epochs=10, cooldown_epochs=0, use_iters=True,
	patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=0, decay_rate=0.1,)

trainer.data = _Namespace()
trainer.data.batch_size = 2048
trainer.data.batch_size_per_gpu = None
trainer.data.batch_size_test = None
trainer.data.batch_size_per_gpu_test = 125
trainer.data.num_workers_per_gpu = 8
trainer.data.drop_last = True
trainer.data.pin_memory = True
trainer.data.persistent_workers = False

trainer.mixup_kwargs = dict(
	mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=1.0, switch_prob=0.5,
	mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=data.nb_classes)

trainer.start_test_epoch = 200
trainer.every_test_epoch = 5
trainer.find_unused_parameters = False
trainer.syn_BN = True
trainer.scaler = 'native'  # [none, native, apex]
trainer.ema = None  # [ None, 0.99996 ]
trainer.ema_start_epoch = 100

# =========> loss <=================================
loss = _Namespace()
loss.loss_terms = [
	dict(type='SoftTargetCE', name='CE', lam=1.0, fp32=True),
	# dict(type='CLSKDLoss', name='CLSKDLoss', lam=1.0, kd_type='hard', tau=1.0),
]

loss.clip_grad = 5.0
loss.create_graph = False
loss.retain_graph = False

# =========> logging <=================================
logging = _Namespace()
logging.log_terms_train = [
	dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
	dict(name='data_t', fmt=':>5.3f'),
	dict(name='optim_t', fmt=':>5.3f'),
	dict(name='lr', fmt=':>7.6f'),
	dict(name='CE', fmt=':>5.3f', add_name='avg'),
	# dict(name='CLSKDLoss', fmt=':>5.3f', add_name='avg'),
]
logging.log_terms_test = [
	dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
	dict(name='top1_cnt', fmt=':>6.0f', show_name='sum'),
	dict(name='top5_cnt', fmt=':>6.0f', show_name='sum'),
	dict(name='top_all', fmt=':>6.0f', show_name='sum'),
]
logging.train_reset_log_per = 50
logging.train_log_per = 50
logging.train_save_model_epoch = 1
logging.test_log_per = 50
