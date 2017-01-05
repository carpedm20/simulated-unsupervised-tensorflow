# Simulated+Unsupervised (S+U) learning in TensorFlow

TensorFlow implementation of [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/abs/1612.07828).

![model](./assets/SimGAN.png)


## Requirements

- Python 2.7
- [TensorFlow](https://www.tensorflow.org/) 0.12.0
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- [tqdm](https://github.com/tqdm/tqdm)

## Usage

To generate synthetic dataset:

1. Run [UnityEyes](http://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) with changing `resolution` to `640x480` and `Camera parameters` to `[0, 0, 20, 40]`.
2. Move generated images into `data/gaze/UnityEyes`.

The `data` directory should looks like:

    data
    ├── gaze
    │   ├── MPIIGaze
    │   │   └── Data
    │   │       └── Normalized
    │   │           ├── p00
    │   │           ├── p01
    │   │           └── ...
    │   └── UnityEyes # contains images of UnityEyes
    │       ├── 1.jpg
    │       ├── 2.jpg
    │       └── ...
    ├── __init__.py
    ├── gaze_data.py
    ├── hand_data.py
    └── utils.py

To train a model (samples will be generated in `samples` directory):

    $ python main.py
    $ tensorboard --logdir=logs --host=0.0.0.0

To test with an existing model:

    $ python main.py --is_train=False --synthetic_image_dir="./data/gaze/UnityEyes/"


## Results

(in progress)

The paper is lack of details for some hyperparameters such as `B` and `lambda`.

![result_0104](./assets/results_0104.png)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
