from __future__ import print_function
import time
import datetime
from tqdm import tqdm
from model import *
from skimage.transform import resize
from tensorboard_logger import log_value

# torch
import torch
import torch.nn.parallel
from torch.autograd import Variable

# Project specific
import Base


# set up

class ArgSimulator():
    def __init__(self):
        self.gpuid = 0
        self.batch_size = 1
        self.image_size = 256
        self.parallel = False
        self.training_stage = 2
        self.val_dir = ''
        self.train_dir = ''
        self.test_input_dir = ''
        self.learning_rate = 0.00002
        self.epoch_limit = 200
        self.mode = 'predict_real'  # predict_synthetic | predict_real
        self.check_point_dir = '.ckpt/'
        self.real_dir = 'data/web/'
        self.synthetic_dir = 'data/synthetic/Test1'
        self.pretrained_weights = 'HeavyRain-stage2-2019-05-11-76'  #
        #self.pretrained_weights = 'GAN_fullyconvcrop_epoch90_multidir_epoch93_epoch119'


class Tester(Base.Base):
    def __init__(self, config):
        super(Tester, self).__init__(config)
        self.synthetic_dir = config.synthetic_dir
        self.real_dir = config.real_dir
        self.file_list = []

    def predict_real(self, iteration='test'):
        outdir = 'out/' + iteration + '/'
        single_dir = outdir + 'single/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not os.path.exists(single_dir):
            os.makedirs(single_dir)
        if iteration == 'test':
            self.load_gan(self.pretrained_weights, False)
            #self.load_checkpoint(self.pretrained_weights, False)

        self.file_list = os.listdir(self.real_dir)
        self.file_list.sort()
        num_of_seq = len(self.file_list)
        print("Total %d rain images from: %s" % (num_of_seq, self.real_dir))

        # Test Mode: feed forward
        with torch.no_grad():
            for i in range(0, num_of_seq, 1):
                # read in image
                filename = os.path.join(self.real_dir, self.file_list[i])
                print('\rTesting  %d image name:,' % i, filename, end=' ')
                rain_image = read_image(filename, noise=False)

                # adjust image size (due to memory limitation)
                (h, w, c) = rain_image.shape
                if h > 800 or w > 800:
                    h = h / 2
                    w = w / 2
                    rain_image = resize(rain_image, [h, w])

                # adjust image size to multiple of 64
                floor_h = np.floor(h/64)
                floor_w = np.floor(w/64)
                new_h = floor_h * 64
                new_w = floor_w * 64
                #new_h , new_w = 1024-64, 1024-64
                rain_image = resize(rain_image, [new_h, new_w])

                # prepare tensors
                self.image_in = torch.FloatTensor(1, 3, new_h, new_w)
                self.image_in[0,:,:,:] = torch.from_numpy(rain_image.transpose(2, 0, 1))
                input_var = Variable(self.image_in).cuda(self.gpuid)

                # compute average atmospheric light A
                mean_atm = self.G.forward_test(input_var, None, mode='A')
                # inference
                self.st_out, self.trans_out, self.atm_out, self.clean_out = self.G.forward_test(input_var, mean_atm, mode='run')

                # save output
                out_img = np.clip(tensor_to_image(self.clean_out), 0, 1)
                recons = (input_var - (1 - self.trans_out) * self.atm_out) / self.trans_out - self.st_out
                painter1 = torch.cat([input_var, self.clean_out, recons], dim=3)
                painter2 = torch.cat([self.trans_out, self.st_out, self.atm_out], dim=3)
                painter = torch.cat([painter1, painter2], dim=2)
                write_image(out_img, single_dir + self.file_list[i])
                write_tensor(painter, outdir + self.file_list[i])

        print('\n')

    def predict_synthetic(self, epoch='test'):
        sum_psnr = 0
        outdir = 'out/' + epoch + '/'
        single_dir = outdir + 'single/'
        input_dir = os.path.join(self.synthetic_dir, 'in')
        gt_dir = os.path.join(self.synthetic_dir, 'gt')
        assert(os.path.exists(input_dir))
        assert(os.path.exists(gt_dir))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not os.path.exists(single_dir):
            os.makedirs(single_dir)
        if epoch == 'test':
            self.load_checkpoint(self.pretrained_weights, False)
        self.file_list = os.listdir(input_dir)
        self.file_list.sort()
        num_of_seq = len(self.file_list)
        print("Total % synthetic rain images from: %s" % (num_of_seq, self.synthetic_dir))
        # Test mode: feed forward
        with torch.no_grad():
            for i in range(0, num_of_seq, 1):
                # configure input and gt
                if 'Test1' in input_dir:
                    input_image_name = self.file_list[i]
                    prefix = input_image_name[0:7]
                    gt_image_name = prefix+'.png'
                else:
                    input_image_name = self.file_list[i]
                    nb = input_image_name[0:3]
                    gt_image_name =  nb+ '_GT.png'
                input_image_path = os.path.join(input_dir, input_image_name)
                gt_image_path =os.path.join(gt_dir, gt_image_name)

                # read image
                rain_image = read_image(input_image_path, noise=False)
                gt_image = read_image(gt_image_path)
                (h, w, c) = rain_image.shape
                floor_h = np.ceil(h/64)
                floor_w = np.ceil(w/64)
                new_h = floor_h * 64
                new_w = floor_w * 64
                new_h, new_w = 448, 672
                rain_image = resize(rain_image, [new_h, new_w])

                self.image_in = torch.FloatTensor(1, 3, new_h, new_w)
                self.image_in[0,:,:,:] = torch.from_numpy(rain_image.transpose(2, 0, 1))
                clean_tensor = torch.FloatTensor(self.batch_size, c, h, w)
                clean_tensor[0, :, :, :] = torch.from_numpy(gt_image.transpose((2, 0, 1)))
                clean_gt_var = F.upsample(Variable(clean_tensor).cuda(), size=(new_h, new_w), mode='bilinear')

                input_var = Variable(self.image_in).cuda(self.gpuid)
                A = self.G.forward_test(input_var, None, mode='A')
                self.st_out, self.trans_out, self.atm_out, self.clean_out = self.G.forward_test(input_var, A, mode='run')
                recons = (input_var - (1 - self.trans_out) * self.atm_out) / self.trans_out - self.st_out
                out_img = np.clip(tensor_to_image(self.clean_out), 0, 1)
                painter1 = torch.cat([input_var, self.clean_out, recons], dim=3)
                painter2 = torch.cat([self.trans_out, self.st_out, self.atm_out], dim=3)
                painter = torch.cat([painter1, painter2], dim=2)
                #write_tensor(painter, outdir + self.file_list[i])
                write_image(out_img, outdir + self.file_list[i])
                psnrsss = compute_psnr(self.clean_out, clean_gt_var)
                sum_psnr += psnrsss
                print('\rTesting  %d image name: %s,  psnr: %f,' % (i, input_image_name, psnrsss), end=' ')
        print('Average PSNR: %f', sum_psnr/num_of_seq)
        print('\n')

    def predict(self, epoch='test'):
        print("Test real rain images from: \n", self.test_input_dir)
        if epoch == 'test':
            self.load_gan(self.pretrained_weights, False)
        # get all file names
        self.file_list = os.listdir(self.test_input_dir)
        self.file_list.sort()
        num_of_seq = len(self.file_list)
        outdir = 'out/' + epoch + '/'
        ph = 640    # input height
        pw = 1280   # input width
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with torch.no_grad():
            for n in range(0, num_of_seq, 1):
                # iterate each rain image (full image)
                filename = os.path.join(self.test_input_dir, self.file_list[n])
                rain_image = read_image(filename, noise=False)
                # divide image into processable patches 768x1280 ~ 9GB memory
                (h, w, c) = rain_image.shape   # full image size
                output_image = torch.FloatTensor(1,c,h,w)
                rows = int(np.ceil(float(h)/float(ph)))
                cols = int(np.ceil(float(w)/float(pw)))
                (bh, bw) = h / rows, w / cols
                print('bh, bw, rows, cols ', bh, bw, rows, cols)
                for i in range(rows):
                    for j in range(cols):
                        patch = rain_image[i*bh:(i+1)*bh, j*bw:(j+1)*bw,:]
                        padded_patch = np.pad(patch, [((ph-bh)/2,(ph-bh)/2), ((pw-bw)/2, (pw-bw)/2), (0,0)], 'symmetric')
                        print('padded_patch size: ', padded_patch.shape)
                        input_tensor = torch.FloatTensor(1,3,ph, pw)
                        input_tensor[0,:,:,:] = torch.from_numpy(padded_patch.transpose(2,0,1))
                        input_var = Variable(input_tensor).cuda(self.gpuid)
                        self.st_out, self.trans_out, self.atm_out, self.clean_out = self.G(input_var)
                        output_patch = self.clean_out[0,:,(ph-bh)/2:ph-(ph-bh)/2, (pw-bw)/2:pw-(pw-bw)/2]
                        print('output_patch size: ', output_patch.size())
                        output_image[0,:,i*bh:(i+1)*bh, j*bw:(j+1)*bw] = output_patch

                write_tensor(output_image, outdir+self.file_list[n])
        print('\n')


if __name__ == '__main__':
    # args = get_args()
    args = ArgSimulator()
    if args.mode == 'predict_synthetic':
        tester = Tester(args)
        tester.predict_synthetic()
    elif args.mode == 'predict_real':
        tester = Tester(args)
        tester.predict_real()
