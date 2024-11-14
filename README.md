# ViT-VAE
My personal modification of "Learning Traces by Yourself: Blind Image Forgery
Localization via Anomaly Detection with ViT-VAE". <br>
Originally from https://github.com/media-sec-lab/ViT-VAE <br>

<p align='center'>  
  <img src='https://github.com/media-sec-lab/ViT-VAE/blob/master/fig.png' width='700'/>
</p>

# Installation

The code requires Python 3.8.20 and PyTorch 1.11.0

```bash
conda create -n vitvae python=3.8
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```


# Usage
For training and validation use csv file.

```bash
bash script.sh
```

You can run test.py to generate the predicted results of the test image.<br>

For testing:

```bash
python test.py 
```

img_path: Path of the image. 

noise_path: Path of Noiseprint feature. <br>

Note: ViT-VAE needs to use the Noiseprint feature. <br>
You need to replace `main_extraction.py ` with `.\tools\main_extraction_ViT-VAE.py ` in [Noiseprint](https://github.com/grip-unina/noiseprint). <br>
This code is used to generate the noise map of the image.


save_path: Save path of prediction results (not thresholded).


mask_path: Ground-truth of the image.

# Acknowledgments
[Noiseprint: https://github.com/grip-unina/noiseprint](https://github.com/grip-unina/noiseprint)
