# --------------------------------------------------------------------------------------------------------
# IMPORTANT
# --------------------------------------------------------------------------------------------------------
#
# This is the main development file.
# Please make sure you only add justified code here, otherwise your code may be rejected by the reviewers.
# For exploratory and experimental code, please use the `online-detection/tasks` folder.
#
# --------------------------------------------------------------------------------------------------------

VERBOSE = True

# Load dataset

from river.datasets import CreditCard

dataset = CreditCard()
if VERBOSE: print("Is the dataset downloaded? ", dataset.is_downloaded, "\n")
