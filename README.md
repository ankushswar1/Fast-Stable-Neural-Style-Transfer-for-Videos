# Fast-Stable-Neural-Style-Transfer-for-Videos
## What It Does
We present an architecture for fast, stable style transfer of videos that preserves style invariance, temporal consistency, and the style and content features of inputs. 

We use the AdaIN operator to make our image-transformation network both content- and style-invariant (encoder and decoder parameters are preserved across styles).

We also use a noise-resilient loss function to train a temporally-consistent style transfer algorithm that can operate in real-time, by stylizing each frame independently rather than using 3D convolutions or optical flow, which are computationally expensive and time consuming.

See `test.py` and `train.py` for information on how to train and test our algorithm, respectively.

## Results & More Information

Our approach effectively stylizes videos with any combination of style and content inputs and preserves the temporal consistency and quality of the original inputs. See `Final Report.pdf` and `Final Poster.pdf` for quantitative and qualitative validations of our approach.

## Contributions
Both authors (Michelle Bao, Ankush Swarnakar) contributed to the project equally and worked on developing the architecture and evaluation metrics. Our work is highly inspired by the adaptive instance normalization work of Huang et al. in "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" and Mur.AI's video style transfer work in "Stabilizing neural style-transfer for video."

## Acknowledgements
This was our final project for CS231N, Stanford's awesome deep learning class on computer vision and convolutional neural networks. We'd like to thank the incredible course staff (Fei-Fei Li, Justin Johnson & Serena Yeung) and the TAs for introducing us to CNNs and deep learning and guiding us throughout this project!

