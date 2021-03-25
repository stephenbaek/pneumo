# Pneumo
Pneumothorax segmentation using TensorFlow.

## Get Started

### Clone this repository
```bash
git clone https://github.com/stephenbaek/pneumo.git
cd pneumo
```

### Install dependencies
The code has been developed and tested with `TensorFlow==2.3.2`. But it should work fine with any version of TensorFlow 2. After configuring your own version of TensorFlow, run the following command to install the other dependencies.

```bash
pip install -r requirements.txt
```

### Data download
Please download data manually from https://www.kaggle.com/jesperdramsch/siim-acr-pneumothorax-segmentation-data. The original SIIM-ACR Kaggle Challenge no longer maintains the original dataset. After downloading the zip file, extract it under `data\siim-acr` folder, so that the files have the following structure:
```bash
- data
|---- siim-acr
|    |---- dicom-images-train
|    |---- dicom-images-test
|    | train-rle.csv
```

### Run example
A simple U-Net example file is available under `examples/unet.ipynb`.

Rough workflow should look as follows.
```python
import pneumo
ds = pneumo.data.load_siim('data/siim-acr')

model = pneumo.models.UNet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=pneumo.losses.combined_loss(1,4,3),
              metrics=[pneumo.metrics.dice])

model_history = model.fit(
    ds,
    epochs=1,
    steps_per_epoch=10,
)
```