# Sentence label generation based on nuScenes dataset

This code allows you to generate sentence label based on nuScenes dataset. The concept of the sentence label comes from Tesla's AI day in 2021. In the AI day sharing, they proposed an autoregressive lane language model.

Inspired by this, we decided to build and train a similar model on a publicly available dataset. This repository is focused on the preliminary data preparation and the generation of the labels required for network training.

## Data processing flow
![image](https://github.com/oneline-wsq/nuscenes/blob/master/the%20pipeline%20of%20scentence%20label%20generation.png)

1. Find the map patch corresponding to the current frame according to ego pose;
2. Intercept the lane + lane connector in the map patch;
3. Find the key point according to the incoming degree & outcoming degree provided by the connectivity;
4. Divide the lanes according to length and curvature to determine the control points;
5. Iterate/search the key points & control points in the map patch, and generate the language of lanes;

The final generated result is shown in the following figure.

<div align=center>
<img src="[https://github.com/BIT-MJY/Active-SLAM-Based-on-Information-Theory/blob/master/img/1-2.png](https://img-blog.csdn.net/2018061215200776?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70](https://github.com/oneline-wsq/nuscenes/blob/master/visible%20data/0.jpg))" width="180" height="105"> width="180" height="105"/>
</div>


<center class="half">
    <img src="[https://img-blog.csdn.net/2018061215200776?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70](https://github.com/oneline-wsq/nuscenes/blob/master/visible%20data/0.jpg)" width="200"/><img src="[https://img-blog.csdn.net/20180612152032532?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70](https://github.com/oneline-wsq/nuscenes/blob/master/visible%20data/2.jpg)" width="200"/>
      <img src="[https://img-blog.csdn.net/2018061215200776?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70](https://github.com/oneline-wsq/nuscenes/blob/master/visible%20data/33.jpg)" width="200"/><img src="[https://img-blog.csdn.net/20180612152032532?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70](https://github.com/oneline-wsq/nuscenes/blob/master/visible%20data/35.jpg)" width="200"/>


## Prediction Process

We have also made a tiny video of the prediction process. You can get a more intuitive understanding.

