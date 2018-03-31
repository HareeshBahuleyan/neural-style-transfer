config = dict(
    original_image_path = 'images/originals/mili.jpg',
    style_image_path = 'images/styles/style_5.jpg',
    content_layer_name = 'block4_conv1', # which layer to take for content loss
    style_loss_conv_blocks = [1,2,3,4,5], # which conv blocks to use for style loss # use all 5 as default
    style_wgts = [0.05, 0.2, 0.2, 0.25, 0.3], # weights for the style layers # should sum to 1.0
    image_size = 600, # Assume square images, discarding the aspect ratio
    lambda_coeff = 1.0, # The coefficient of the content loss term in the objective function
    iterations = 10,
    start_img = 'n', # Whether to start from a random normal or random uniform distribution noise image
)