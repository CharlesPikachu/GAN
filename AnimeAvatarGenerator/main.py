'''
Function:
    main function
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import cfg
import torch
import argparse
import torch.nn as nn
from modules.utils import *
from modules.networks import *
from torchvision.utils import save_image


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='use wcgan to generate anime avatar')
    parser.add_argument('--mode', dest='mode', help='train or test', default='train', type=str)
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='the path of checkpoints', type=str)
    args = parser.parse_args()
    return args


'''main function'''
def main():
    # parse arguments
    args = parseArgs()
    assert args.mode in ['train', 'test']
    if args.mode == 'test': assert os.path.isfile(args.checkpointspath)
    # some necessary preparations
    checkDir(cfg.BACKUP_DIR)
    logger_handle = Logger(cfg.LOGFILEPATH.get(args.mode))
    start_epoch = 1
    end_epoch = cfg.NUM_EPOCHS + 1
    use_cuda = torch.cuda.is_available()
    # define the dataset
    dataset = ImageDataset(rootdir=cfg.ROOTDIR, imagesize=cfg.IMAGE_SIZE, img_norm_info=cfg.IMAGE_NORM_INFO)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    # define the loss function
    loss_func = nn.BCELoss()
    if use_cuda: loss_func = loss_func.cuda()
    # define the model
    net_g = Generator(cfg)
    net_d = Discriminator(cfg)
    if use_cuda:
        net_g = net_g.cuda()
        net_d = net_d.cuda()
    # define the optimizer
    optimizer_g = buildOptimizer(net_g.parameters(), cfg.OPTIMIZER_CFG['generator'])
    optimizer_d = buildOptimizer(net_d.parameters(), cfg.OPTIMIZER_CFG['discriminator'])
    # load the checkpoints
    if args.checkpointspath:
        checkpoints = loadCheckpoints(args.checkpointspath, logger_handle)
        net_d.load_state_dict(checkpoints['net_d'])
        net_g.load_state_dict(checkpoints['net_g'])
        optimizer_g.load_state_dict(checkpoints['optimizer_g'])
        optimizer_d.load_state_dict(checkpoints['optimizer_d'])
        start_epoch = checkpoints['epoch'] + 1
    else:
    	net_d.apply(weightsNormalInit)
    	net_g.apply(weightsNormalInit)
    # define float tensor
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    # train the model
    if args.mode == 'train':
        for epoch in range(start_epoch, end_epoch):
            logger_handle.info('Start epoch %s...' % epoch)
            for batch_idx, imgs in enumerate(dataloader):
                imgs = imgs.type(FloatTensor)
                z = torch.randn(imgs.size(0), cfg.NUM_LATENT_DIMS, 1, 1).type(FloatTensor)
                imgs_g = net_g(z)
                # --train generator
                optimizer_g.zero_grad()
                labels = FloatTensor(imgs_g.size(0), 1).fill_(1.0)
                loss_g = loss_func(net_d(imgs_g), labels)
                loss_g.backward()
                optimizer_g.step()
                # --train discriminator
                optimizer_d.zero_grad()
                labels = FloatTensor(imgs_g.size(0), 1).fill_(1.0)
                loss_real = loss_func(net_d(imgs), labels)
                labels = FloatTensor(imgs_g.size(0), 1).fill_(0.0)
                loss_fake = loss_func(net_d(imgs_g.detach()), labels)
                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_d.step()
                # --print infos
                logger_handle.info('Epoch %s/%s, Batch %s/%s, Loss_G %f, Loss_D %f' % (epoch, cfg.NUM_EPOCHS, batch_idx+1, len(dataloader), loss_g.item(), loss_d.item()))
            # --save checkpoints
            if epoch % cfg.SAVE_INTERVAL == 0 or epoch == cfg.NUM_EPOCHS:
                state_dict = {
                                'epoch': epoch,
                                'net_d': net_d.state_dict(),
                                'net_g': net_g.state_dict(),
                                'optimizer_g': optimizer_g.state_dict(),
                                'optimizer_d': optimizer_d.state_dict()
                            }
                savepath = os.path.join(cfg.BACKUP_DIR, 'epoch_%s.pth' % epoch)
                saveCheckpoints(state_dict, savepath, logger_handle)
                save_image(imgs_g.data[:25], os.path.join(cfg.BACKUP_DIR, 'images_epoch_%s.png' % epoch), nrow=5, normalize=True)
    # test the model
    else:
        z = torch.randn(imgs.size(0), cfg.NUM_LATENT_DIMS, 1, 1).type(FloatTensor)
        net_g.eval()
        imgs_g = net_g(z)
        save_image(imgs_g.data[:25], 'images.png', nrow=5, normalize=True)


'''run'''
if __name__ == '__main__':
    main()