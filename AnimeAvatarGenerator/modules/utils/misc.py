'''
Function:
    define some util functions
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import torch
import logging


'''check dir'''
def checkDir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        return False
    return True


'''log function.'''
class Logger():
    def __init__(self, logfilepath, **kwargs):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(logfilepath),
                                      logging.StreamHandler()])
    @staticmethod
    def log(level, message):
        logging.log(level, message)
    @staticmethod
    def debug(message):
        Logger.log(logging.DEBUG, message)
    @staticmethod
    def info(message):
        Logger.log(logging.INFO, message)
    @staticmethod
    def warning(message):
        Logger.log(logging.WARNING, message)
    @staticmethod
    def error(message):
        Logger.log(logging.ERROR, message)


'''load checkpoints'''
def loadCheckpoints(checkpointspath, logger_handle):
    logger_handle.info('Loading checkpoints from %s...' % checkpointspath)
    checkpoints = torch.load(checkpointspath)
    return checkpoints


'''save checkpoints'''
def saveCheckpoints(state_dict, savepath, logger_handle):
    logger_handle.info('Saving state_dict in %s...' % savepath)
    torch.save(state_dict, savepath)
    return True


'''build optimizer'''
def buildOptimizer(params, cfg):
    if cfg['type'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=cfg['adam']['lr'], betas=(cfg['adam']['betas'][0], cfg['adam']['betas'][1]))
    else:
        raise ValueError('Unsupport type %s in buildOptimizer...' % cfg['type'])
    return optimizer