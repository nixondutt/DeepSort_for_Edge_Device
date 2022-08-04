import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import tflite_runtime.interpreter as tflite
# from .model import Net
# from model import Net



class Tflite_extractor(object):
    def __init__(self,model_path):
        self.interpreter = tflite.Interpreter(model_path = model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = (10,3,128,64)
        self.input_data = np.zeros(self.input_size,dtype=np.float32)
        self.interpreter.resize_tensor_input(self.input_details[0]['index'],self.input_size)
        self.interpreter.allocate_tensors()
        logger = logging.getLogger("root.tracker")
        logger.info("LOADED deepsort tflite model")
        self.size = (64,128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    @staticmethod
    def _resize(im,size):
        return cv2.resize(im.astype(np.float32)/255., size)


    def _preprocess(self, im_crops):
        num_imgs = len(im_crops)
        im_batch = torch.cat([self.norm(Tflite_extractor._resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        inputs = im_batch.cpu().detach().numpy()  #numpy array (?, 3, 128, 64)
        frame_input = np.copy(self.input_data)
        frame_input[:num_imgs] = inputs
        return frame_input
    
    def __call__(self,im_crops):
        im_batch = self._preprocess(im_crops)
        self.interpreter.set_tensor(self.input_details[0]['index'],im_batch)
        self.interpreter.invoke()
        features = self.interpreter.get_tensor(self.output_details[0]['index'])
        return features


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Tflite_extractor("checkpoint/deepsort.tflite")
    feature = extr([img,img])
    print(feature.shape)
