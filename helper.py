import os
import torch
import numpy as np
from PIL import Image
from scipy.misc import imresize
from torch.utils.data import Dataset


# set up
class RainHazeImageDataset(Dataset):
    def __init__(self, root_dir, mode, aug=False, transform=None):
        """
        At __init__ state, we read in all the image paths of the entire dataset instead of image data
        :param root_dir: directory of files containing the paths to all rain images
        :param mode: 'train', 'val', or 'test'
        :param aug: Whether augment the input image
        :param transform:
        """
        self.root_dir = root_dir
        self.mode = mode
        self.aug = aug
        self.transform = transform
        self.path = os.path.join(self.root_dir, (mode + '_s_rain.txt'))
        self.in_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_in.txt')))
        self.real_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_real.txt')))
        self.streak_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_streak.txt')))
        self.trans_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_trans.txt')))
        self.clean_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_clean.txt')))
        self.atm_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_atm.txt')))
        self.no_realrain = len(self.real_list)

    def __len__(self):
        return len(self.in_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            noise_trigger = True
        else:
            noise_trigger = False

        real_rain_index = np.random.randint(self.no_realrain)
        rain = read_image(self.in_list[idx], noise_trigger)
        im_gt = read_image(self.clean_list[idx]) # clean image = input - sparse - middle - dense
        st_gt = read_image(self.streak_list[idx]) # sparse streak
        trans_gt = read_image(self.trans_list[idx])  # middle streak
        atm_gt = read_image(self.atm_list[idx])   # dense streak
        realrain = read_image(self.real_list[real_rain_index])
        
        # render haze
        if np.min(trans_gt) == 0:
            print(self.trans_list[idx])

        input_list = [rain, st_gt, trans_gt, atm_gt, im_gt, realrain]
        if self.aug:
            input_list = augment(input_list)
        else:
            #input_list = ImageResize(input_list, size=256)
            input_list = RandomCrop(input_list, size=256)
        
        if self.transform:
            input_list = self.transform(input_list)

        return input_list


class FogImageDataset(Dataset):
    def __init__(self, root_dir, mode, aug=False, transform=None):
        """
        At __init__ state, we read in all the image paths of the entire dataset instead of image data
        :param root_dir: directory of files containing the paths to all rain images
        :param mode: 'train', 'val', or 'test'
        :param aug: Whether augment the input image
        :param transform:
        """
        self.root_dir = root_dir
        self.root_dir = root_dir
        self.mode = mode
        self.aug = aug
        self.transform = transform
        self.path = os.path.join(self.root_dir, (mode + '_s_rain.txt'))
        self.haze_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_haze.txt')))
        self.atm_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_atm.txt')))
        self.gt_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_gt.txt')))
        self.trans_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_trans.txt')))

    def __len__(self):
        return len(self.haze_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            noise_trigger = True

        else:
            noise_trigger = False

        haze = read_image(self.haze_list[idx])
        atm = read_image(self.atm_list[idx])
        gt = read_image(self.gt_list[idx])
        trans = read_image(self.trans_list[idx])

        (height, width, channel) = atm.shape

        input_list = [haze, trans, atm, gt]
        if self.aug:
            input_list = augment(input_list)
        #else:
        #    input_list = RandomCrop(input_list, size=224)

        if self.transform:
            input_list = self.transform(input_list)

        return input_list


def gradient(x):
    gradient_h = torch.abs(x[:,:,:,:-1] - x[:,:,:,1:])
    gradient_v = torch.abs(x[:,:,:-1,:] - x[:,:,1:,:])
    return gradient_h, gradient_v


class ToTensor(object):
    """ Conver ndarray to Tensors"""
    def __call__(self, image_list):
        # input image_list is: H x W x C
        # torch image_list is: C x H x W
        tensor_list = []
        for image in image_list:
            image = image.transpose((2, 0, 1))
            tensor_list.append(image)
        return tensor_list


def tensor_to_image(tensor):
    if type(tensor) in [torch.autograd.Variable]:
        img = tensor.data[0].cpu().detach().numpy()
    else:
        img = tensor[0].cpu().detach().numpy()
    img = img.transpose((1,2,0))
    try:
        img = np.clip(img, 0, 255)
        if img.shape[-1] == 1:
            img = np.dstack((img, img, img))
    except:
        print("invalid value catch")
        Image.fromarray(img).save('catch.jpg')
    return img


def to_tensor(x, gpuid=None):
    if type(x) in [list, tuple]:
        image_num = len(x)
        if image_num >0:
            (h,w,c) = x[0].shape
        else:
            print("No image!")
        t = torch.FloatTensor(image_num, c, h, w)
        for i in range(image_num):
            image = x[i].transpose((2, 0, 1))
            t[i,:,:,:] = torch.from_numpy(image)
        if gpuid:
            t = t.cuda(gpuid)
        return t
    elif isinstance(x, np.ndarray):
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 2:
            x = np.dstack((x,x,x))
            x = np.expand_dims(x, axis=0)
        bs, h, w, c = x.shape
        t = torch.FloatTensor(bs,c,h,w)
        x = x.transpose((0,3,1,2))
        t = torch.from_numpy(x)
        if gpuid:
            t = t.cuda(gpuid)
        return t
    else:
        print("data type not accepted!")
        return None


def to_variable(x, gpuid=3):
    v = None
    if type(x) in [list, tuple,  np.ndarray]:
        x = to_tensor(x)
    if type(x) in [torch.DoubleTensor, torch.FloatTensor]:
        if gpuid:
            x = x.cuda(gpuid)
        v = torch.autograd.Variable(x)
    else:
        print("Unrecognized data type!")
    return v


def generate_new_seq(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip())
    f.close()
    return file_list#[1:10000]


def make_col(tensor_gt, tensor_out):
    im_gt = tensor_to_image(tensor_gt)
    im_out = tensor_to_image(tensor_out)
    im_col = np.concatenate((im_out, im_gt), axis=0)
    return im_col


def make_row(*args):
    for i, v in enumerate(args):
        im = tensor_to_image(v)
        if i == 0:
            painter = im
        else:
            painter = np.concatenate((painter,im), axis=1)
    return painter


def write_sample(self):
    tensor_list = [self.image_out, self.image_in, self.streak_out, self.streak_gt, self.mask_gt, self.mask_out]
    path_list = ['./out.jpg', './in.jpg', 'streak_out.jpg', 'streak_gt.jpg', './mask_gt.jpg', './mask_out.jpg']
    for i, img_tensor in enumerate(tensor_list):
        img = img_tensor.cpu().data[0].detach().numpy()
        img = img.transpose((1,2,0))
        img = np.clip(img*255.0, 0, 255)
        if img.shape[-1]==1:
            img_file = Image.fromarray(img[:,:,0].astype(np.uint8))
            im2 = img_file.convert('L')
            im2.save(path_list[i])
        else:
            img_file = Image.fromarray(img.astype(np.uint8))
            img_file.save(path_list[i])


def write_tensor(img, path):
    im = tensor_to_image(img)
    write_image(im, path)


def write_image(img, path):
    img = np.clip(img*255, 0, 255)
    if img.shape[-1] == 1:
        img_file = Image.fromarray(img[:,:,0].astype(np.uint8))
        im2 = img_file.convert('L')
        im2.save(path)
    else:
        img_file = Image.fromarray(img[:,:,:].astype(np.uint8))
        img_file.save(path)


def read_image(image_path, noise=False):
    """
    function: read image function
    :param image_path: input image path
    :param noise: whether apply noise on image
    :return: image in numpy array, range [0,1]
    """
    img_file = Image.open(image_path)
    img_data = np.array(img_file, dtype=np.float32)
    (h,w,c) = img_data.shape
    if h < 224 or w < 224:
        img_data = imresize(img_data, [h,w])
    if len(img_data.shape) < 3:
        img_data = np.dstack((img_data, img_data, img_data))
    if noise:
        (h,w,c) = img_data.shape
        noise = np.random.normal(0,1,[h,w])
        noise = np.dstack((noise, noise, noise))
        img_data = img_data + noise
    img_data = img_data.astype(np.float32)/255.0
    img_data[img_data > 1.0] = 1.0
    img_data[img_data < 0] = 0.0
    return img_data.astype(np.float32)


def augment(input_list, scale_limit=300, crop_size=224):
    input_list = RandomHorizontalFlip(input_list)
    input_list = RandomColorWarp(input_list)
    # input_list = RandomScale(rain, streak, clean, size_limit=scale_limit)
    input_list = RandomCrop(input_list, size=crop_size)
    return input_list


def compute_psnr(est, gt):
    batch_size = est.size()[0]
    sum_acc = 0
    for i in range(batch_size):
        est_image = est.cpu().data[i].detach().numpy()
        gt_image = gt.cpu().data[i].detach().numpy()
        est_image = est_image.transpose((1,2,0))
        gt_image = gt_image.transpose((1,2,0))
        sum_acc += psnr(est_image*255, gt_image*255)
    avg_acc = sum_acc / batch_size
    return avg_acc


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return np.float64(rgb.dot(xform.T))


def psnr(es, gt):
    if len(es.shape) ==3 and es.shape[2] == 3:
        es_img = rgb2ycbcr(es)
        gt_img = rgb2ycbcr(gt)
        es_channel = es_img[:,:,0]
        gt_channel = gt_img[:,:,0]
    else:
        es_channel = es
        gt_channel = gt

    imdiff = np.float64(es_channel) - np.float64(gt_channel)
    rmse = np.sqrt(np.mean(np.square(imdiff.flatten())))
    psnr_value = 20*np.log10(255/rmse)
    return psnr_value


def load_checkpoint(self, best=False):
    """
    Load the best copy of a model. This is useful for 2 cases:

    - Resuming training with the most recent model checkpoint.
    - Loading the best validation model to evaluate on the test data.

    Params
    ------
    - best: if set to True, loads the best model. Use this if you want
      to evaluate your model on the test data. Else, set to False in
      which case the most recent version of the checkpoint is used.
    """
    print("[*] Loading model from {}".format(self.ckpt_dir))

    filename = self.model_name + '_ckpt.pth.tar'
    if best:
        filename = self.model_name + '_model_best.pth.tar'
    ckpt_path = os.path.join(self.ckpt_dir, filename)
    ckpt = torch.load(ckpt_path)

    # load variables from checkpoint
    self.start_epoch = ckpt['epoch']
    self.best_valid_acc = ckpt['best_valid_acc']
    self.lr = ckpt['lr']
    self.model.load_state_dict(ckpt['state_dict'])

    if best:
        print(
            "[*] Loaded {} checkpoint @ epoch {} "
            "with best valid acc of {:.3f}".format(
                filename, ckpt['epoch']+1, ckpt['best_valid_acc'])
        )
    else:
        print(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                filename, ckpt['epoch']+1)
        )


def RandomHorizontalFlip(input_list):
    output_list = []
    if np.random.rand() < 0.5:
        for item in input_list:
            output_list.append(np.copy(np.fliplr(item)))
    else:
        output_list = input_list
    return output_list


def RandomColorWarp(input_list):
    std_range = 0.05
    mean_range = 0
    output_list = []
    random_std = np.random.uniform(-std_range, std_range, 3)
    random_mean = np.random.uniform(-mean_range, mean_range, 3)
    random_order = np.random.permutation(3)

    for item in input_list:
        item *= (1 + random_std)
        item += random_mean
        item = item[:, :, random_order]
        output_list.append(item)

    return output_list


def RandomScale(rain, streak, clean, size_limit=300):
    h,w,c = rain.shape
    shorter_edge = min(h,w)
    ratio_limit = float(size_limit) / float(shorter_edge)
    base = max(ratio_limit, 0.5)
    ratio = np.random.rand(3) + base
    ratio[2] =1
    newh = int(round(h*ratio[0]))
    neww = int(round(w*ratio[1]))
    #rain = ndimage.interpolation.zoom(rain, ratio)
    #streak = ndimage.interpolation.zoom(streak, ratio)
    #clean = ndimage.interpolation.zoom(clean, ratio)
    #rain = cv2.resize(rain, dsize=(newh, neww), interpolation=cv2.INTER_CUBIC)
    #streak = cv2.resize(streak, dsize=(newh, neww), interpolation=cv2.INTER_CUBIC)
    #clean = cv2.resize(clean, dsize=(newh, neww), interpolation=cv2.INTER_CUBIC)
    return rain, streak, clean


def RandomCrop(input_list, size=224):
    output_list = []
    num_of_length = len(input_list)
    h, w, c = input_list[0].shape
    try:
        row = np.random.randint(h-size)
        col = np.random.randint(w-size)
    except:
        print "random low value leq high value"
        print(h, w, c)
    for i in range(num_of_length-1):
        item = input_list[i]
        item = item[row:row+size, col:col+size, :]
        output_list.append(item)
    h,w,c = input_list[-1].shape
    row = np.random.randint(h-size)
    col = np.random.randint(w-size)
    item = input_list[-1][row:row+size, col:col+size, :]
    output_list.append(item)
    assert(len(input_list)==len(output_list))
    return output_list


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_residue(tensor):
    max_channel = torch.max(tensor, dim=1, keepdim=True)
    min_channel = torch.min(tensor, dim=1, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel


def ImageResize(input_list, size=256):
    from skimage.transform import resize
    output_list = []
    for item in input_list:
        item = resize(item, [size, size])
        output_list.append(item.astype(np.float32))
    return output_list