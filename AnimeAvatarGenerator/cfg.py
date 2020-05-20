'''config file'''
import os


# dimensionality of the latent space
NUM_LATENT_DIMS = 100
# size of the batches
BATCH_SIZE = 128
# image size
IMAGE_SIZE = (64, 64)
# image normalization info
IMAGE_NORM_INFO = {'means': [0.5, 0.5, 0.5], 'stds': [0.5, 0.5, 0.5]}
# number of training epochs
NUM_EPOCHS = 500
# interval between saving checkpoints
SAVE_INTERVAL = 5
# images root dir
ROOTDIR = os.path.join(os.getcwd(), 'images/*')
# backup dir
BACKUP_DIR = os.path.join(os.getcwd(), 'checkpoints')
# log file path
LOGFILEPATH = {'train': os.path.join(BACKUP_DIR, 'train.log'), 'test': os.path.join(BACKUP_DIR, 'test.log')}
# optimizer config
OPTIMIZER_CFG = {'generator': {'type': 'adam', 'adam': {'lr': 1e-4, 'betas': [0.5, 0.999]}},
                 'discriminator': {'type': 'adam', 'adam': {'lr': 1e-4, 'betas': [0.5, 0.999]}}}