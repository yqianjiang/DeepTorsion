import sys
import os
import logging
from torsion_inference.torsion_inference import torsion_inferer

from keras.models import load_model
import keras.backend as K
from utils.load_stn import load_stn
import time


# metrics for loading torsion_model
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# loss function for loading seg_model
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0000001) / (K.sum(y_true_f) + K.sum(y_pred_f) +  0.0000001)

def dice_coef_multilabel(y_true, y_pred, numLabels=4):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])

    #video_dir = "/mnt/DeepVOG3D/example/videos/torsion_vid.avi"
    video_dir = "/mnt/Torsion/test_data/clips/"
    
    output_dir = 'results/torsion_infer/'
    visual_save_path = 'results/torsion_infer/visual/'

    batch_size = 32
    h = 60
    w = 720   # the training width
    #noise_mode = "noise_noGlints"
    #noise_mode = "STN8_glints0.3"
    noise_mode = "glints"
    condition = "uniform_"+noise_mode+"_all_"+str(w)+"_ro16_bs16_max5"

    

    # config logger
    logging.basicConfig(level=logging.INFO, 
                        filename=output_dir+'logs/'+condition+'.log',
                        filemode='w',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )
    
    my_logger = logging.getLogger('MyLogger')
    my_logger.setLevel(logging.DEBUG)

    my_logger.info("loading models..")

    # load seg_model
    seg_model = load_model("utils/gen_data/models/multiple_layers.h5", custom_objects = {"dice_coef_multilabel":dice_coef_multilabel})
    

    # load torsion_model
    model_path = "model/"+condition+".h5"
    torsion_model = load_model(model_path, custom_objects={"rmse": rmse})
    #torsion_model = None
    #torsion_model = load_stn(model_path)

    inferer = torsion_inferer(seg_model, torsion_model, my_logger)

    print_prefix = ""
    #method = ["network", "cc", "hoi", "stn"]
    method = ["network"]
    for video_name in os.listdir(video_dir):
        video_path=video_dir+video_name
        for infer_method in method:
            my_logger.info("infering torsion angle by method: "+infer_method)
            csv_path = output_dir+'csv/'+video_name+"_"+infer_method+"_2.csv"
            visual_save_path2 = visual_save_path+video_name+"_"+infer_method+"_2.mp4"
            if os.path.isfile(csv_path):
                print("file", csv_path, " already exist!")
                #continue
            
            start = time.time()
            inferer.predict(video_path, csv_path, output_vis_path = visual_save_path2, infer_method = infer_method, batch_size=batch_size, print_prefix=print_prefix)
            my_logger.info(infer_method+"  spend time for video" + video_name + ":")
            my_logger.info(time.time()-start)

    my_logger.info("finish")
