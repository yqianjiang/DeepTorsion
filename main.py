import argparse
import os

from src.train.train_STN import fit_model_dir

# to fit model: python main.py --fit data_dir result_dir --gpu GPU_num --stn
# to test model: python main.py --test data_dir result_dir --gpu GPU_num


parser = argparse.ArgumentParser()
required = parser.add_argument_group('required arguments')

required.add_argument("--fit", help="fit data, use with -fit data_directory", nargs=1, type=str, metavar=("PATH"))
required.add_argument("--test", help="test data, use with -test data_directory", nargs=2, type=str, metavar=("PATH","PATH"))
parser.add_argument("--gpu", help="GPU number", default="0", type=str, metavar=("INT"))
parser.add_argument("--stn", help="If use STN network or not", action="store_true")
parser.add_argument("--size", help="", default="(60, 720)", type=str, metavar=("INT, INT"))
parser.add_argument("--batchsize", help="", default=16, type=int, metavar=("INT"))
parser.add_argument("--epoch", help="", default=100, type=int, metavar=("INT"))
parser.add_argument("--range", help="degree range to train the network, just enter the max number, default is 5.", default=5, type=float, metavar=("FLOAT"))
parser.add_argument("--rotate", help="number of rotation to generate for each image, default=16", default=16, type=int, metavar=("INT"))
parser.add_argument("--modelpath", help="model save path for --fit mode; model path for --test mode", type=str, metavar=("PATH"))


args = parser.parse_args()

# set arguments
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=arg.gpu

stn = args.stn

# fit or test mode
if args.fit is not None:
    data_dir = args.fit
    h, w = args.size
    n_rotate = args.rotate
    max_degree = args.range
    model_save_dir = args.modelpath
    batch_size = args.batchsize
    epochs = args.epoch
    condition = "fit"
    model = fit_model_dir(data_dir=data_dir,
                      n_rotate = n_rotate,
                      h = h, w = w,
                      batch_size=batch_size,
                      epochs=epochs,
                      add_noise=True,
                      stn = stn,
                      condition = condition,
                      degree_sampling="uniform",
                      max_degree = max_degree,
                      model_save_dir=model_save_dir)
elif args.test is not None:
    data_dir = args.test
    test()



