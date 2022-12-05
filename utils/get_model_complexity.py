import torchvision.models as models
from ptflops import get_model_complexity_info

model = models.resnet50()
dummy_size = (3, 256, 256)

macs, params = get_model_complexity_info(model, dummy_size, as_strings=True, print_per_layer_stat=True, verbose=True)
                                
print('computational complexity:', macs)
print('number of parameters:', params)
# 1 mac = 2 flops
# pip install ptflops

