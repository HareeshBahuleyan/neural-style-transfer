# Neural Style Transfer
Given two images A and B, the task is to transfer the *style* of image A to that of image B, while being able to preserve the *content* of image A. 

## Instructions
- Specify the source and style image paths, and other configurations in `config.py`
- Run `run_style_transfer.ipynb` in a cell-by-cell manner, the output at each iteration is saved in `images/results/`
Refer to `style-transfer-vgg.ipynb` notebook for step by step procedures and detailed notes.

## Results
Source Image:
<br>
<img src="images/originals/mili.jpg" width="600"/>
<br>

Style Image:
<br>
<img src="images/styles/style_5.jpg" width="600"/>
<br>

Stylized Version of Source Image:
<br>
<img src="images/results/res_at_iteration_9.png" width="600"/>
<br>

Transformed Image:

## Acknowledgements
This project was done following the lecture materials of the [fast.ai course](http://course.fast.ai/lessons/lesson8.html) offered by Jeremy Howard 
