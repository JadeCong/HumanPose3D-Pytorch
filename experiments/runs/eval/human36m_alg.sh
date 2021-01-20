#!/bin/bash

# launch the evaluation for algebraic triangulation
python $(cd $(dirname $0) && cd ../../scripts && pwd -P)/hpn_trainer.py --config $(cd $(dirname $0) && cd ../../configs && pwd -P)/eval/human36m_alg.yaml --eval --local_rank 0 --seed 26 --logdir $(cd $(dirname $0) && cd ../../../logs/hpn && pwd -P)
