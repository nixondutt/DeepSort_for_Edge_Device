import io
import sys
import os
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import onnx
import onnxruntime

# from .model import Net
from model import Net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)[
            'net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)


    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    
    def onnx_conversion(self,extr, onnx_model_path):
        # Input to the model 
        x = torch.randn(batch_size, 3,128,64, requires_grad=True)
        x = x.to(self.device)
        torch_out = extr.net(x)

        # Export the model
        torch.onnx.export(
            extr.net,
            x,
            onnx_model_path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input':{0:'batch_size'},    # variable length axis
                'output':{0:'batch_size'}})
        
        if os.path.isfile(onnx_model_path):
            ort_session = onnxruntime.InferenceSession(onnx_model_path)
        else:
            sys.stderr.write("Unable to find the onnx model file")
        #compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: Extractor.to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        
        #Compare Onnx Runtime and Pytorch Results 

        np.testing.assert_allclose(Extractor.to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        
        print("Tested with Onnxruntime, result OK!!")



        

if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    batch_size = 1
    extr = Extractor("checkpoint/ckpt.t7")
    extr.net.eval()
    onnx_model_path = "checkpoint/deepsort.onnx"
    extr.onnx_conversion(extr,onnx_model_path)
