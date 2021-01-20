#!/bin/bash

# launch the evaluation for volumetric triangulation
python $(cd $(dirname $0) && cd ../../scripts && pwd -P)/hpn_trainer.py --config $(cd $(dirname $0) && cd ../../configs && pwd -P)/eval/human36m_vol_softmax.yaml --local_rank 26 --seed 26 --logdir $(cd $(dirname $0) && cd ../../../logs/hpn && pwd -P)
