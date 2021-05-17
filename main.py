from parameter import *
from trainer import Trainer
# from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils1 import make_folder
import torch

def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    data_loader = Data_Loader(config.train, config.dataset, config.image_path, config.imsize,
                             config.batch_size, shuf=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        trainer = Trainer(data_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

if __name__ == '__main__':
    # torch.cuda.set_device(0)
    # config = get_parameters()
    # config.version = config.dataset + '_' + config.model + '_' + config.version
    # print(config)
    # main(config)
    #
    # config = get_parameters2()
    # config.version = config.dataset + '_' + config.model + '_' + config.version
    # print(config)
    # main(config)
    #
    # #
    # config = get_parameters3()
    # config.version = config.dataset + '_' + config.model + '_' + config.version
    # print(config)
    # main(config)
    #
    # config = get_parameters4()
    # config.version = config.dataset + '_' + config.model + '_' + config.version
    # print(config)
    # main(config)

    config = get_parameters6()
    config.version = config.dataset + '_' + config.model + '_' + config.version
    print(config)
    main(config)

