# DeepTorsion

DeepTorsion is a solution for torsional eye tracking based on deep convolutional neural networks.

## Getting Started

### Prerequisites
A python distribution is required, and the necessary packages are list in requirements.txt, you can run:
```
$ pip install -r requirements.txt 
```

### Quick Start
To fit the model with your own training data, use "--fit data_dir" as follow:
```
$ python main.py --fit data/training

# use --modelpath model_save_path to save the trained model
$ python main.py --fit data/training --modelpath model/model.h5
```


To test the trained model, "--modelpath" flag is required to provide the model to test. run:
```
$ python main.py --test data/testing --modelpath model/model.h5
```

You can specify some arguments for training, such as batch size, epochs, image size, as follow:

To specify image size, use --size (h, w)
```
$ python main.py --fit data/training --modelpath model/model.h5 --size 60,720 --batchsize 16 --gpu 0
```

To use the STN Network, add --stn
```
$ python main.py --fit data/training --modelpath model/model.h5 --stn
```


## Authors

* **Yingqian Jiang** - *Implementation and validation*
* **Seyed-Ahmad Ahmadi** - *Research study concept*
* **Yiu Yuk Hoi** - *Initial work*

## Links to other related papers
Yiu YH, Aboulatta M, Raiser T, Ophey L, Flanagin VL, zu Eulenburg P, Ahmadi SA. DeepVOG: Open-source Pupil Segmentation and Gaze Estimation in Neuroscience using Deep Learning. Journal of neuroscience methods. vol. 324, 2019, DOI: https://doi.org/10.1016/j.jneumeth.2019.05.016


## License

This project is licensed under the GNU General Public License v3.0 (GNU GPLv3) License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

We thank our fellow researchers at the German Center for Vertigo and Balance Disorders for help in acquiring data for training and validation of pupil segmentation and gaze estimation. In particular, we would like to thank Theresa Raiser, Dr. Virginia Flanagin and Prof. Dr. Peter zu Eulenburg.
