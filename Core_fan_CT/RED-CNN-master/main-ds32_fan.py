import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = args.save_path + 'ds' + str(args.ds_factor) + '_fig/'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             ds_factor = args.ds_factor,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader, device)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--saved_path', type=str, default='./data/CT/fan/')
    parser.add_argument('--save_path', type=str, default='./save/fan_ds32-1000/')
    parser.add_argument('--test_patient', type=str, default='test')
    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--ds_factor', type=int, default=32, help='from {64，32,16, 8, 4}')
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160, help='liver=-160, -110.0, lung=-1350')
    parser.add_argument('--trunc_max', type=float, default=240, help='liver=240, 190.0, lung=150')
    parser.add_argument('--trunc_mode', type=str, default='HU', help='HU、others(liver,lung trunc)')
    parser.add_argument('--transform', type=bool, default=False)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--print_iters', type=int, default=100)
    parser.add_argument('--decay_epoch', type=int, default=300)
    parser.add_argument('--save_epoch', type=int, default=200)
    parser.add_argument('--test_epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    main(args)
