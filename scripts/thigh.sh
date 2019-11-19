#!/bin/bash

# fcn8s
# python ../train.py --config ../configs/fcn8s.yml
# python ../test.py --config ../configs/fcn8s.yml

# fcn16s
# python ../train.py --config ../configs/fcn16s.yml
# python ../test.py --config ../configs/fcn16s.yml

# fcn32s
# python ../train.py --config ../configs/fcn32s.yml
# python ../test.py --config ../configs/fcn32s.yml

# plain u
# python ../train.py --config ../configs/unet.yml
python ../test.py --config ../configs/unet.yml

# atten u
# python ../train.py --config ../configs/atten_unet.yml
# python ../test.py --config ../configs/atten_unet.yml

# resu
# python ../train.py --config ../configs/resunet.yml
# python ../test.py --config ../configs/resunet.yml

# resatt unet
# python ../train.py --config ../configs/res_atten_unet.yml
python ../test.py --config ../configs/res_atten_unet.yml


# psp
# python ../train.py --config ../configs/psp.yml

