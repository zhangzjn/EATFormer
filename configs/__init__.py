import importlib
from argparse import Namespace
from util.net import get_timepc


def get_cfg(opt_terminal):
	opt_terminal.cfg_path = opt_terminal.cfg_path.split('.')[0].replace('/', '.')
	dataset_lib = importlib.import_module(opt_terminal.cfg_path)
	cfg_terms = dataset_lib.__dict__
	ks = list(cfg_terms.keys())
	for k in ks:
		if k.startswith('_'):
			del cfg_terms[k]
	cfg = Namespace(**dataset_lib.__dict__)
	for key, val in opt_terminal.__dict__.items():
		cfg.__setattr__(key, val)
	cfg.task_start_time = get_timepc()
	return cfg
