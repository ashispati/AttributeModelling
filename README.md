[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-ff69b4.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

## AttributeModelling

### About
Code for the paper: **Latent Space Regularization for Explicit Control of Musical Attributes**. Implements the models and regularization technique to encode selected musical attributes along specific dimensions of the latent of a VAE trained to reconstruct individual measures of music. 

### Installation and Use
1. Requires `python3.x` and `pytorch1.0.0`
2. Download or clone this repository. Navigate to the root folder and run `python setup.py install`.
3. Download the folder linked [here](https://drive.google.com/open?id=1sh5zXo-D5AyaamJ_k1ZmHop3EDEE5CJU). Unzip it and place the `datasets` and `folk_raw_data` folders in the `AttributeModelling/data` folder.
4. Run the `train_measure_vae.py` script with appropriate arguments to train or test the models.

### Additonal Information

#### Training Parameters
All the model variants were trained using the same parameters to ensure consistency.
* Optimizer: Adam (`b1=0.9, b2=0.999, e=1e-8`)
* Learning Rate: `1e-4`
* Batch-Size: `256`
* Number of Epochs: `30`
* Beta (for VAE training): `1e-3`

#### Interpretability Metric
The scores were computed using the method proposed by [Adel et al.](http://proceedings.mlr.press/v80/adel18a.html). The computation steps are:
1. All data-points in the held-out test set are passed through the encoder of the trained model to obtain the corresponding latent vectors.
2. For each attribute `a`, the latent space dimension `r` which has the maximum mutual information with `a` is computed.
3. A simple linear regression model is then fit to predict `a` given `z_r`, i.e. the value of the latent code for dimension `r`.
4. The interpretability metric is finally the regression score (coefficienct of determination R2) for this regression model

The regression scores (higher is better) are shown in the table below: 

| Model Type 	| Rhythmic Complexity 	| Pitch Range 	| Average  	|
|------------	|---------------------	|-------------	|----------	|
| RHY        	| 0.8364              	| 1.1E-06    	  | 0.42     	|
| PR         	| 0.014               	| 0.9625      	| 0.49     	|
| RHY-PR     	| 0.8339              	| 0.9681      	| 0.90     	|
| Base       	| 4.2E-07            	  | 1.5E-05    	  | 7.9E-06 	|

### Attribution

This research work is published as as conference paper at the ICML ML4MD Workshop, 2019.

> Ashis Pati, Alexander Lerch. "Latent Space Regularization for Explicit Control of Musical Attributes", ICML Workshop on Machine Learning for Music Discovery Workshop (ML4MD), Extended Abstract, Long Beach, CA, USA, 2019.

```
@inproceedings{pati19latent-reg,
  author = {Pati, Ashis and Lerch, Alexander},
  booktitle = {ICML Workshop on Machine Learning for Music Discovery Workshop (ML4MD), Extended Abstract},
  location = {Long Beach, CA, USA},
  title = {Latent Space Regularization for Explicit Control of Musical Attributes},
  year = {2019}
}
```
Please cite the above publication if you are using the code/data in this repository in any manner. 

<a name="License"></a>License
--------------------
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
