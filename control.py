import argparse
from omegaconf import OmegaConf
from control import Controller

if __name__ == "__main__":
	
	# argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='controller/control.yml')
	parser.add_argument('--device', default=0)
	parser.add_argument('--object_class', type=str, default='tableware')
	parser.add_argument('--ik_mode', action='store_true')

	# process cfg
	args, unknown = parser.parse_known_args()
	cfg = OmegaConf.load(args.config)

	# set device
	if args.device == 'cpu':
		cfg.device = 'cpu'
	elif args.device == 'any':
		cfg.device = 'cuda'
	else:
		cfg.device = f'cuda:{args.device}'

	# set parameters
	cfg.ik_mode = args.ik_mode
	cfg.object_class = args.object_class

	# initialize controller
	controller = Controller(cfg)
	
	# control
	controller.recognition_and_control()
