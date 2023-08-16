import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 对应0卡
import torch
import torch.distributed as dist

from models import model_dict, TrainTask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':
    # reference https://stackoverflow.com/questions/38050873/can-two-python-argparse-objects-be-combined/38053253
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args()
    MODEL = model_dict[default_opt.model_name]
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    # dist.init_process_group(backend='nccl',
    #                         init_method='env://')
    # torch.cuda.set_device(dist.get_rank())
    model = MODEL(opt)
    # print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in  model.parameters())))

    # model.fit()  # train
    model.fit1(100000)   # test

