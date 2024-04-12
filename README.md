# Ignision
A Computer Vision based project relying on object detection and localization to detect the presence of fire in an image and display its location.
## Dataset
The images and XML annotations are sourced from the  [Fire Data Annotations Image Dataset version 5](https://universe.roboflow.com/fire-detection/fire-data-annotations/dataset/5) which was released on Jul 6, 2022, and is available on **roboflow**.

## Model
### Environment Setup

<pre>
<code class="language-bash">
python -m venv env_name <span style="color: green">#Create a virtual env named env_name</span>
env_name\Scripts\activate <span style="color: green">#For Windows: Activate the virtual env</span>
pip install -r Ignision/requirements.txt <span style="color: green">#Download the required libraries</span>
</code>
</pre>
##### Note: Download the ['weights.h5'](https://drive.google.com/drive/folders/1e9YafLoB9IHpa0aY33fx_l-_ztxoUM7u) file from the provided Google Drive link.

### Train
<pre>
<code class="language-bash">
python -m Ignision --train --learning-rate=1e-4 --epochs=4 --save-best-to="newweights.h5" --checkpoint-dir="checkpoint.h5" --load-from="Ignision\weights.h5" --dropout=0.2<span style="color: green"></span>
<span style="color: green">#These parameters are merely exemplary; adjust them according to your requirements.</span>
</code>
</pre>

### Predictions
There are two ways to run predictions on images:

1. `--predict`: Predicting and displaying results
2. `--predict-to-file`:Predicting and writing results to an image file named `pred.png`
<pre>
<code class="language-bash">
python -m Ignision --predict="file_path" --load-from="Ignision\weights.h5"<span style="color: green"></span>
python -m Ignision --predict-to-file="file_path" --load-from="Ignision\weights.h5"<span style="color: green"></span>
</code>
</pre>
## Results
<img src="results/0.jpg" width="350"/>&nbsp;&nbsp;&nbsp;&nbsp;
<img src="results/0 pred.jpg" width="350"/> 
<br>
<br>
<img src="results/1.jpg" width="350"/>&nbsp;&nbsp;&nbsp;&nbsp;
<img src="results/1pred.jpg" width="350"/> 
<br>
<br>
<img src="results/3.jpg" width="350"/>&nbsp;&nbsp;&nbsp;&nbsp;
<img src="results/3 pred.jpg" width="350"/> 
<br>
<br>
<img src="results/2.jpg" width="350"/>&nbsp;&nbsp;&nbsp;&nbsp;
<img src="results/2 pred.jpg" width="350"/> 
<br>
<br>
<img src="results/4.jpg" width="350"/>&nbsp;&nbsp;&nbsp;&nbsp;
<img src="results/4 pred.jpg" width="350"/> 
