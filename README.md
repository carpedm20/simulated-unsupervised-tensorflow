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

To refine all synthetic images with a pretrained model:

    $ python main.py --is_train=False --synthetic_image_dir="./data/gaze/UnityEyes/"


## Differences with the paper

- Used an Adam optimizer not a Stochatstic Gradient Descent.
- Only used 83K (14% of 1.2M used by the paper) synthetic images from `UnityEyes`.
- Manually choose hyperparameters for `B` and `lambda` because those are not specified in the paper.


## Training results

For these synthetic images,

![UnityEyes_sample](./assets/UnityEyes_samples.png)

Result of `lambda=1.0` after 4,000 steps.

![Refined_sample_with_lambd=1.0](./assets/lambda=1.0.png)

Result of `lambda=0.5` after 4,000 steps.

![Refined_sample_with_lambd=0.5](./assets/lambda=0.5.png)

Result of `lambda=0.1` after 4,000 steps.

![Refined_sample_with_lambd=0.1](./assets/lambda=0.1.png)

Training loss of discriminator and refiner when `lambda` is `1.0` (blue), `0.5` (purple) and `0.1` (green).

![Refined_sample_with_lambd=0.1](./assets/loss_lambda=1.0,0.5,0.1.png)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
