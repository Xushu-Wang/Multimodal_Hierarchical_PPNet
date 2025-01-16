import torch.nn as nn 


conv_layer = nn.Conv2d(4, 5, 
                      kernel_size=3,
                      stride=1,
                      padding=1) 

print(conv_layer.kernel_sizes)
print(conv_layer.strides)
print(conv_layer.paddings)
