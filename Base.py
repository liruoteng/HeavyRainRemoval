from __future__ import print_function
import time
import datetime
import shutil

from model import *
from collections import OrderedDict
from networks import GANLoss
from skimage.transform import resize

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.models

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboard_logger import configure


# Base class that serve common purpose
class Base(object):
    def __init__(self, config):
        # start from basic parameters
        self.config = config
        self.gpuid = config.gpuid
        self.mode = config.mode
        self.pretrained_weights = config.pretrained_weights
        self.criterionMSE = nn.MSELoss()
        self.criterionL1 = nn.L1Loss()
        self.criterionBCE = nn.BCELoss()
        self.criterionGAN = GANLoss(use_lsgan=False).to(self.gpuid)
        self.parallel = config.parallel

        self.ckpt_dir = config.check_point_dir
        self.model_name = 'HeavyRain-stage%s-%s' % (self.config.training_stage, str(datetime.date.today()))
        self.best_valid_acc = 0
        self.logs_dir = '.logs/'
        self.use_tensorboard = True

        # == hyper-parameters ==
        self.epoch = 0
        self.total_loss = 0
        self.input_list = []
        self.output_list = []
        self.ch_in = 6
        self.ch_out = 3
        self.ndf = 64
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.epoch_limit = config.epoch_limit
        self.LR = config.learning_rate
        self.training_stage = config.training_stage

        # ==  I/O paths ==
        self.train_dir = config.train_dir
        self.val_dir = config.val_dir
        self.test_input_dir = config.test_input_dir

        # == initialize tensors ==
        # input
        self.image_in = torch.FloatTensor(self.batch_size, 3, self.image_size, self.image_size)
        self.image_in_var = None
        self.streak_gt_var = None
        self.trans_gt_var = None
        self.atm_gt_var = None
        self.clean_gt_var = None

        # outputs
        self.st_out = None
        self.trans_out = None
        self.atm_out = None
        self.clean_out = None
        self.realrain_st = None
        self.realrain_trans = None
        self.realrain_atm = None
        self.realrain_out = torch.FloatTensor(self.batch_size, 3, self.image_size, self.image_size)
        self.loss_adv_realrain = self.criterionGAN(torch.FloatTensor(0).cuda(), True)
        self.accs = AverageMeter()
        self.probability = -1
        self.fl = -1
        self.tl = -1

        # == declare models ==
        self.G = None
        self.D = None
        self.G_optim = None # optimizer
        self.D_optim = None # optimizer
        self.vgg_model = None
        self.create_model()

        # ==  reset other infrastructure  ==
        self.reset(config)

    def create_model(self):
        # == define perceptual model==
        self.vgg_model = torchvision.models.vgg16(pretrained=True).cuda(self.gpuid)
        # == simple rain feature extractors ==
        self.G = DecompModel().cuda(self.gpuid)
        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=self.LR, betas=(0.9, 0.999))
        if self.mode == 'train' and self.training_stage == 2:
            self.D = DepthGuidedD(self.ch_in).cuda(self.gpuid)
            self.D_optim = torch.optim.Adam(self.D.parameters(), lr=self.LR * 0.1, betas=(0.9, 0.999))
        # == Multiple GPUs ==
        if self.parallel:
            self.G = torch.nn.DataParallel(self.G)
            if self.D:
                self.D = torch.nn.DataParallel(self.D)

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def reset(self, config):
        # tensorboard set up
        if self.use_tensorboard and self.mode == 'train':
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)
        # disable vgg update
        for para in self.vgg_model.parameters():
            para.requires_grad = False

    def set_gradients(self, trainable):
        if self.parallel:
            for param in self.G.module.fognet.parameters():
                param.requires_grad = trainable
            for param in self.G.module.rainnet.parameters():
                param.requires_grad = trainable
        else:
            for param in self.G.fognet.parameters():
                param.requires_grad = trainable
            for param in self.G.rainnet.parameters():
                param.requires_grad = trainable

    @staticmethod
    def trainable(model, trainable):
        for parameter in model.parameters():
            parameter.requires_grad = trainable

    def write_image_stage1(self, path):
        # tensor_zero = torch.zeros(self.batch_size, 3, self.image_size, self.image_size)
        im_in = self.input_list[0]
        recons = (im_in - (1 - self.trans_out.cpu()) * self.atm_out.cpu()) / (self.trans_out.cpu() + 0.0001) - self.st_out.cpu()
        input_row = torch.cat(self.input_list[0:-1], dim=3)
        output_row = torch.cat(
            (recons, self.st_out.cpu(), self.trans_out.cpu(), self.atm_out.cpu(), self.clean_out.cpu()), dim=3)
        painter = torch.cat((input_row, output_row), dim=2)
        img = tensor_to_image(painter)
        img = np.clip(img * 255, 0, 255)
        painter_image = Image.fromarray(img.astype(np.uint8))
        painter_image.save(path)

    def write_image_stage2(self, path):
        tensor_zero = torch.zeros(self.batch_size, 3, self.image_size, self.image_size)
        im_in = self.input_list[0]
        im_real_in = self.input_list[-1]
        recons = (im_in - (1 - self.trans_out.cpu()) * self.atm_out.cpu()) / (
                    self.trans_out.cpu() + 0.0001) - self.st_out.cpu()  # - self.st_out.cpu()
        input_row = torch.cat(self.input_list[:-1], dim=3)
        output_row = torch.cat(
            (recons, self.st_out.cpu(), self.trans_out.cpu(), self.atm_out.cpu(), self.clean_out.cpu()), dim=3)
        realrecons = (im_real_in  - (1 - self.realrain_trans.cpu()) * self.realrain_atm.cpu()) / \
                     (self.realrain_trans.cpu() + 0.0001) - self.realrain_st.cpu()
        real_row = torch.cat((im_real_in, realrecons, self.realrain_trans.cpu(), self.realrain_atm.cpu(),
                              self.realrain_out.cpu()), dim=3)
        painter = torch.cat((input_row, output_row, real_row), dim=2)
        img = tensor_to_image(painter)
        img = np.clip(img * 255, 0, 255)
        painter_image = Image.fromarray(img.astype(np.uint8))
        painter_image.save(path)

    def load_data(self, mode, aug=False):
        if mode == 'train':
            dataset = RainHazeImageDataset(self.train_dir, 'train',
                                           aug=aug,
                                           transform=transforms.Compose([ToTensor()]))
            shuff = True
        elif mode == 'val':
            dataset = RainHazeImageDataset(self.val_dir, 'val',
                                           transform=transforms.Compose([ToTensor()]))
            shuff = False
        else:
            dataset = None
            shuff = False
            print('Undefined mode', mode)
            exit()
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=shuff,
                                 drop_last=True,
                                 num_workers=self.batch_size)
        return data_loader

    def save_checkpoint(self, state, msg, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + msg + '_ckpt.pth.tar'
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, msg,  parallel, best=False, load_lr=True):
        print("[*] Loading model from {}{}.pth.tar".format(self.ckpt_dir, msg))
        filename = msg + '.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        new_state_dict = OrderedDict()

        if 'G' in ckpt.keys():
            pretrained_weights = ckpt['G']
        else:
            pretrained_weights = ckpt
        if parallel: # expect module in the pretrained weights
            for k, v, in pretrained_weights.items():
                if 'module' not in k:
                    name = 'module.' + k
                    new_state_dict[name] = v
                else:
                    new_state_dict = pretrained_weights
        else:
            for k, v in pretrained_weights.items():
                if 'module' in k:
                    name = k[7:]
                    new_state_dict[name] = v
                else:
                    new_state_dict = pretrained_weights
        self.G.load_state_dict(new_state_dict)
        # self.D.load_state_dict(ckpt['D'])
        if load_lr:
            self.LR = ckpt['lr']
        if 'epoch' in ckpt.keys():
            self.epoch = ckpt['epoch']
        if 'best_valid_acc' in ckpt.keys():
            self.best_valid_acc = ckpt['best_valid_acc']

    def load_my_state_dict(self, state_dict):
        own_state = self.G.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)


    def load_gan(self, msg, best=False, load_lr=True):
        print("Load GAN: ... ")
        print(msg)
        filename = msg + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        state_dict = ckpt['G']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        # self.D.load_state_dict(ckpt['D'])
        self.G.load_state_dict(new_state_dict)
        # Gweights = ckpt['G']
        # mydict = self.G.state_dict()
        # for name, param in Gweights.items():
        #     if 'batchnorm' in name:
        #         continue
        #     if isinstance(param, torch.nn.Parameter):
        #         param = param.data
        #     mydict[name].copy_(param)

        if load_lr:
            self.LR = ckpt['lr']
        self.epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']

    def validate(self):
        start = time.time()
        print("Validation: ", datetime.datetime.now())
        total_loss = 0
        sum_acc = 0
        count = 0
        dataloader = self.load_data('val', aug=False)
        val_dir = 'val/' + str(self.epoch) + '/'
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        with torch.no_grad():
            for i, self.input_list in enumerate(dataloader):
                print('\rCount Number:%d,' % i, end=' ')
                image_in_var = Variable(self.input_list[0]).cuda(self.gpuid)
                streak_gt_var = Variable(self.input_list[1]).cuda(self.gpuid)
                self.st_out, self.trans_out, self.atm_out, self.clean_out = self.G(image_in_var)
                self.clean_out = (image_in_var - self.st_out - (1 - self.trans_out) * self.atm_out) / (
                            self.trans_out + 0.0001)
                sum_acc += compute_psnr(self.st_out, streak_gt_var)
                if i % 100 == 0:
                    self.write_image_stage1(val_dir + str(i) + '.png')
                count = i
            avg_acc = sum_acc / count
        print("Epoch: {:02d} - Average loss: {:.3f} - Accuracy: {:.3f} Time: {}\n".format(
            self.epoch, total_loss, avg_acc, time.time() - start))
        state = {'epoch': self.epoch, 'G': self.G.state_dict(),
                 'best_valid_acc': self.best_valid_acc, 'lr': self.LR}
        if avg_acc > self.best_valid_acc:
            self.best_valid_acc = avg_acc
            self.save_checkpoint(state, 'last', True)

    def vgg(self, img, l):
        x = img
        for idx, layer in enumerate(self.vgg_model.modules()):
            if 2 <= idx <= l:
                x = layer(x)
        return x

    def test(self):
        start = time.time()
        print("Testing: ", datetime.datetime.now())
        self.load_checkpoint('pretrained', best=False)
        self.G.cuda()
        self.G = torch.optim.Adam(self.G.parameters(), lr=self.LR)
        self.batch_size = 1
        sum_acc = 0
        avg_acc = 0
        dataloader = self.load_data('val', aug=False)
        num_batch = len(dataloader) / self.batch_size
        val_dir = 'val/' + 'test' + '/'
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        with torch.no_grad():
            for i, self.input_list in enumerate(dataloader):
                print('\rCount Number:%d,' % i, end=' ')
                image_in_var = Variable(self.input_list[0]).cuda()
                clean_gt_var = Variable(self.input_list[4]).cuda()
                self.st_out, self.trans_out, self.atm_out, self.clean_out = self.G(image_in_var)
                self.clean_out = (image_in_var - self.st_out - (1 - self.trans_out) * self.atm_out) / (
                            self.trans_out + 0.001)
                self.write_image_stage1(val_dir + str(i) + '.jpg')
                sum_acc += compute_psnr(self.clean_out, clean_gt_var)
                if i == 1000:
                    break
            avg_acc = sum_acc / 1000
        print("Average accuracy: ", avg_acc)

    def predict_resize(self, iter='test'):
        print("Testing real rain images from: ", self.test_input_dir)
        if iter == 'test':
            self.load_checkpoint('pretrained2', False)
        self.file_list = os.listdir(self.test_input_dir)
        self.file_list.sort()
        num_of_seq = len(self.file_list)
        outdir = 'out/' + iter + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with torch.no_grad():
            for i in range(0, num_of_seq, 1):
                filename = os.path.join(self.test_input_dir, self.file_list[i])
                self.image_in = torch.FloatTensor(1, 3, self.image_size, self.image_size)
                print('\rTesting  %d image name:,' % i, filename, end=' ')
                rain_image = read_image(filename, noise=False)
                rain_image = resize(rain_image, [self.image_size, self.image_size])
                self.image_in[0, :, :, :] = torch.from_numpy(rain_image.transpose(2, 0, 1))
                input_var = Variable(self.image_in).cuda(self.gpuid)
                self.st_out, self.trans_out, self.atm_out, self.clean_out = self.G(input_var)
                recons = (input_var - (1 - self.trans_out) * self.atm_out) / (self.trans_out + 0.0001) - self.st_out
                painter1 = torch.cat([input_var, self.st_out, self.trans_out], dim=3)
                painter2 = torch.cat([recons, self.clean_out, self.atm_out], dim=3)
                painter = torch.cat([painter1, painter2], dim=2)
                write_tensor(painter, outdir + self.file_list[i])
        print('\n')