from imports import *
from load_and_run import load_and_run

load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/60-20-20-no-transform",
             tr_va_split = [60,20,20],
             number_of_epochs = 2,)

# load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/40-40-20-no-transform",
#              tr_va_split = [40,40,20],
#              number_of_epochs = 1000,)

# load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/20-60-20-no-transform",
#              tr_va_split = [20,60,20],
#              number_of_epochs = 1000,)