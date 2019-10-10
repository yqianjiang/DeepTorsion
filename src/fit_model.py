from .train.train_STN import fit_model_dir
import sys
import os
import time

# to run this script: python fit_model.py GPU_num net_type
# e.g., python fit_model.py 0 stn

if __name__ == "__main__":

    train_dir = "data/training/polar/"

    os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
    net_type = str(sys.argv[2])
    if net_type == 'stn':
        stn = True
    elif net_type == 'simple':
        stn = False
    else:
        raise ValueError('unkown option :{}'.format(sys.argv[2]))

    h = 60
    w = 720
    n_rotate = 16
    max_degree = 5
    degree_sampling="uniform"
    model_save_dir = None
    noise_mode = "glints" # noGlints, noNoise
    batch_size = 16
    epochs = 100

    if noise_mode == "noNoise":
        add_noise = False
    else:
        add_noise = True

    if stn:
        condition = degree_sampling+'_STN8_'+noise_mode+'_ro'+str(n_rotate)+'_bs16_max'+str(max_degree)
    else:
        condition = degree_sampling+'_'+noise_mode+'_ro'+str(n_rotate)+'_bs16_max'+str(max_degree)
                 
    print("condition:", condition)
    print("fitting model...")

    model = fit_model_dir(data_dir=train_dir,
                      n_rotate = n_rotate,
                      h = h, w = w,
                      batch_size=batch_size,
                      epochs=epochs,
                      add_noise=add_noise,
                      stn = stn,
                      condition = condition,
                      degree_sampling=degree_sampling,
                      max_degree = max_degree,
                      model_save_dir=model_save_dir)


