from src.deeplabv2 import Deeplab
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as ttransforms
import cv2
import torchvision.transforms as T

label_colours = [
    # [  0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]]  # the color of ignored label(-1)
label_colours = list(map(tuple, label_colours))
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

class Segmap:
    def __init__(self):
        self.nets=dict()
        self.crop_size=(512,256)
        self.resize=True
        self.numpy_transform=False
    def set_networks(self):
        self.nets['T']=Deeplab(num_classes=19,restore_from='./weights/deeplab_gta5')
        # for net in self.nets.keys():
            # self.nets[net].cuda()
    def set_eval(self):
        self.nets['T'].eval()

    def decode_labels(self,mask, num_images=1, num_classes=19):
        """Decode batch of segmentation masks.
        Args:
        mask: result of inference after taking argmax.
        num_images: number of images to decode from the batch.
        num_classes: number of classes to predict.
        Returns:
        A batch with num_images RGB images of the same size as the input.
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.data.cpu().numpy()
        n, h, w = mask.shape
        if n < num_images:
            num_images = n
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[int(k_), int(j_)] = label_colours[int(k)]
            outputs[i] = np.array(img)
        # return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)
        return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32'))

    def _img_transform(self, image):
        if self.numpy_transform:
            image = np.asarray(image, np.float32)
            image = image[:, :, ::-1]  # change to BGR
            image -= IMG_MEAN
            image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
            new_image = torch.from_numpy(image)
        else:
            image_transforms = ttransforms.Compose([
                ttransforms.ToTensor(),
                # ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
                ttransforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
            new_image = image_transforms(image)
        return new_image
    
    def _val_sync_transform(self, img, mask):

        if self.resize:
            img = img.resize(self.crop_size, Image.BICUBIC)
            img=self._img_transform(img)
            return img
            # mask = mask.resize(self.crop_size, Image.NEAREST)

        # final transform
        # if mask:
            # img, mask = self._img_transform(img), self._mask_transform(mask)
            # return img, mask
        else: 
            img = self._img_transform(img)
            return img

    def seg(self,image):
        """
        Input : opencv format
        Output : PIL format resized (512,256)
        """
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(image)
        image=self._val_sync_transform(image,None)
        self.set_networks()
        self.set_eval()
        image=image.unsqueeze(0)
        # image=image.cuda()
        preds=self.nets['T'](image)[0]
        # for key in preds.keys():

            # pred=preds[key].data.cpu().numpy()
        preds=preds.detach().numpy()

        pred=np.argmax(preds,axis=1)
        x=self.decode_labels(pred)
        x=x.squeeze(0)
        x=np.array(x).transpose(1,2,0).astype('uint8')
        transform = T.ToPILImage()
        img=transform(x)

        return img
