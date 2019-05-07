## Additonal Information

### Training Parameters
All the model variants were trained using the same parameters to ensure consistency.
* Optimizer: Adam (`b1=0.9, b2=0.999, e=1e-8`)
* Learning Rate: `1e-4`
* Batch-Size: `256`
* Number of Epochs: `30`

### Interpretability Metric
The scores were computed using the method proposed by [Adel et al.](http://proceedings.mlr.press/v80/adel18a.html). The computation steps are:
1. All data-points in the held-out test set are passed through the encoder of the trained model to obtain the corresponding latent vectors.
2. For each attribute `a`, the latent space dimension `r` which has the maximum mutual information with `a` is computed.
3. A simple linear regression model is then fit to predict `a` given `z_r`, i.e. the value of the latent code for dimension `r`.
4. The interpretability metric is finally the regression score (coefficienct of determination R2) for this regression model

The regression scores (higher is better) are shown in the table below: 

| Model Type 	| Rhythmic Complexity 	| Pitch Range 	| Average  	|
|------------	|---------------------	|-------------	|----------	|
| RHY        	| 0.8364              	| 1.13E-06    	| 0.42     	|
| PR         	| 0.014               	| 0.9625      	| 0.49     	|
| RHY-PR     	| 0.8339              	| 0.9681      	| 0.90     	|
| Base       	| 4.23E-07            	| 1.54E-05    	| 7.90E-06 	|
