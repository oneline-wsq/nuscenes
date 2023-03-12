# Sentence label generation based on nuScenes dataset

This code allows you to generate sentence label based on nuScenes dataset. The concept of the sentence label comes from Tesla's AI day in 2021. In the AI day sharing, they proposed an autoregressive lane language model.

Inspired by this, we decided to build and train a similar model on a publicly available dataset. This repository is focused on the preliminary data preparation and the generation of the labels required for network training.

## Data processing flow

<div align=center>
<img src="https://github.com/oneline-wsq/nuscenes/blob/master/the%20pipeline%20of%20scentence%20label%20generation.png" width="500" > 
</div>

1. Find the map patch corresponding to the current frame according to ego pose;
2. Intercept the lane + lane connector in the map patch;
3. Find the key point according to the incoming degree & outcoming degree provided by the connectivity;
4. Divide the lanes according to length and curvature to determine the control points;
5. Iterate/search the key points & control points in the map patch, and generate the language of lanes;

The final generated result is shown in the following figure.

<div align=center>
<img src="https://github.com/oneline-wsq/nuscenes/blob/master/visible%20data/0.jpg?raw=true" width="800"><img src="https://github.com/oneline-wsq/nuscenes/blob/master/visible%20data/2.jpg?raw=true" width="800"> 
<img src="https://github.com/oneline-wsq/nuscenes/blob/master/visible%20data/33.jpg?raw=true" width="800"><img src="https://github.com/oneline-wsq/nuscenes/blob/master/visible%20data/35.jpg?raw=true" width="800"> 
</div>



