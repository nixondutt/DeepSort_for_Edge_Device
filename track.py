import cv2
import sys
import numpy as np
from timeit import default_timer as timer
import tflite_runtime.interpreter as tflite
import os
import torch
import subprocess
import time
import yaml
import hashlib
import tempfile
from tqdm import tqdm
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from urllib.request import urlopen, Request
from pathlib import path
# from utils.utils import *
# from utils.reid import *
from PIL import Image
import argparse
import shutil




class Track(object):
    def __init__(self):
        self.opt = Track.parse_opt()
        self.names = None
        self.n_classes = 91
        self.colors = np.random.randint(0, 255, size=(self.n_classes, 3), 
                            dtype="uint8")

    @staticmethod    
    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--ext_delegate', 
            action='store_true',
            help = "Use accelarator")
        parser.add_argument(
            '--label',
            type =str, 
            default = 'deep_sort_pytorch/configs/coco.yaml')
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.56,
            help='confidence threshold')
        parser.add_argument(
            '--num_threads',
            type=int,
            default=None,
            help='Number of Threads')
        parser.add_argument(
            '--image',
            type = str,
            default = 'data/cat_dog.png',
            help = 'input image'
        )
        parser.add_argument(
            '--video',
            type = str,
            default = 'data/test_movie_3_shorts.mp4',
            help = 'input image'
        )
        parser.add_argument(
            '--input_mean',
            default = 127.5, type = float,
            help = 'input_mean'
        )
        parser.add_argument(
            '--input_std',
            default=127.5, type = float,
            help = "input standard deviation"
        )
        parser.add_argument(
            "--config_deepsort", type=str,
            default="deep_sort_pytorch/configs/deep_sort.yaml")
        opt = parser.parse_args()
        return opt


    @staticmethod
    def download_url_to_file(url,dst, hash_prefix=None, progress=True):
        file_size = None
        req = Request(url, headers={"api_ver":"deepsort"})
        u = urlopen(req)
        meta = u.info()
        if hasattr(meta,'getheaders'):
            content_length = meta.getheaders("Content-Length")
        else:
            content_length = meta.get_all("Content-Length")
        if content_length is not None and len(content_length) > 0:
            file_size = int(content_length[0])
        
        dst = os.path.expanduser(dst)
        dst_dir = os.path.dirname(dst)
        f = tempfile.NamedTemporaryFile(delete=False,dir=dst_dir)
        try:
            if hash_prefix is not None:
                sha256 = hashlib.sha256()
            with tqdm(total = file_size, disable=not progress,
                   unit= 'B',unit_scale=True, unit_divisor=1024) as pbar:
                while True:
                    buffer = u.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    if hash_prefix is not None:
                        sha256.update(buffer)
                    pbar.update(len(buffer))
            f.close()
            if hash_prefix is not None:
                digest = sha256.hexdigest()
                if digest[:len(hash_prefix)] != hash_prefix:
                    raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
                shutil.move(f.name,dst)
        finally:
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)
        

    @staticmethod
    def safe_download(file,url,url2=None,min_bytes=1E0,error_msg=''):
        """ Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
        """
        file = Path(file)
        assert_msg = f"Download file '{file}' does not exist or size is <min_bytes={min_bytes}"
        try: #url1
            Track.download_url_to_file(url,str(file))
            assert file.exists() and file.stat().st_size>min_bytes, assert_msg
        except Exception as e:#url2
            file.unlink(missing_ok=True) # remove partial download
            os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")
        finally:
            if not file.exists() or file.stat().st_size < min_bytes:
                file.unlink(missing_ok=True)
                sys.stderr.write(f" ERROR:\n{error_msg}\n")
                
    @staticmethod
    def attempt_download(file, repo=None, release = None):
        def github_assets(repository, version='latest'):
            # Return GitHub repo tag and assets
            if version != 'latest':
                version = f'tags/{version}'  # i.e. tags/v6.1
            response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
            return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets 

        file = Path(str(file).strip().replace("'",''))
        if not file.exists():
            name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
            if str(file).startswith(('http:/','https:/')):
                url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
                file = name.split('?')[0] # parse authentication 
                if Path(file).is_file():
                    print(f'Found {url} locally at {file}')
                else:
                    safe_download(file=file,url=url,min_bytes=1E5)
                return file

            try:
                tag, assets = github_assets(repo, release) 
            except Exception:
                try:
                    tag, assets = github_assets(repo) #latest release
                except Exception:
                    try:
                        tag =  subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                    except Exception:
                        tag = release
            file.parent.mkdir(parents=True, exist_ok=True)
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                min_bytes=1E5,
                error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag} or contact support team')
            

        return str(file)
            


                    
            

        










    def read_labels(self):
        with open(self.opt.label, errors = 'ignore') as f:
            self.names = yaml.safe_load(f)['names']
            print("**************************",len(self.names))

    def _set_input_tensor(self,interpreter, image):
        """ sets the input tensor """
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self,interpreter,index):
        """ Returns the output tensor at the given index """
        output_details = interpreter.get_output_details()[index]
        tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
        return tensor


    def detect_objects(self,interpreter, image):
        """
        summary - Returns a list of detection results, each a dictionary of object info. 
        """
        self._set_input_tensor(interpreter, image)
        start = timer()
        interpreter.invoke()


        # Get all output details

        boxes = self._get_output_tensor(interpreter,0)
        classes = self._get_output_tensor(interpreter,1)
        scores = self._get_output_tensor(interpreter,2)
        count = int(self._get_output_tensor(interpreter,3))
        results = []

        for i in range(count):
            if scores[i] >=self.opt.threshold and int(classes[i])==0:
                result = {
                    'bounding_box' : boxes[i],
                    'class_id' : classes[i],
                    'score' : scores[i]
                }
                results.append(result)
        return results

    @staticmethod
    def bbox_rel(*xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0], xyxy[2]])
        bbox_top = min([xyxy[1], xyxy[3]])
        bbox_w = abs(xyxy[0] - xyxy[2])
        bbox_h = abs(xyxy[1] - xyxy[3])
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    @staticmethod
    def _processframe(results, original_image):
        bbox_xywh = []
        confs = []
        for obj in results:
            ymin, xmin, ymax, xmax =  obj['bounding_box']
            xmin = int(xmin * original_image.shape[1])
            xmax = int(xmax * original_image.shape[1])
            ymin = int(ymin * original_image.shape[0])
            ymax = int(ymax * original_image.shape[0])
            xyxy = [xmin,ymin,xmax,ymax]
            x_c, y_c, bbox_w, bbox_h = Track.bbox_rel(*xyxy)
            obj_xyc = [x_c, y_c, bbox_w, bbox_h]
            bbox_xywh.append(obj_xyc)
            confs.append([obj['score']])

        return torch.Tensor(bbox_xywh), torch.Tensor(confs)
    



    def _draw_boxes(self,img, bbox, identities,infer_fps):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]        
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = [int(c) for c in self.colors[id]]
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                    t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            cv2.rectangle(img, (50,6), (370, 60), (0,0,0), -1)
            cv2.putText(img, " FPS - {:.2f}".format(infer_fps),(50,50),cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), thickness=4)  #36,18,201
        #return img



    def infer_video(self,interpreter,size):
        self.read_labels()
        cap = cv2.VideoCapture(self.opt.video)
        width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter('data/test_movie_det1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
        count =0

        # initializing_deepsort
        cfg = get_config()
        cfg.merge_from_file(self.opt.config_deepsort)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET)

        while(cap.isOpened()):
            startTime = time.time()
            count += 1
            ret, original_image = cap.read()
            if not ret:
                print("breaking..")
                break
            pre_image = cv2.resize(original_image.copy(), size, interpolation=cv2.INTER_AREA).astype(np.uint8)
            input_data = np.expand_dims(pre_image, axis = 0)
            results = self.detect_objects(interpreter,pre_image)

            if results is not None and len(results):
                xywhs,confss = Track._processframe(results,original_image)
                outputs = deepsort.update(xywhs, confss, original_image)
                infer_fps = 1/(time.time()-startTime)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    self._draw_boxes(original_image, bbox_xyxy,identities,infer_fps)                      
            else:
                deepsort.increment_ages()
                infer_fps = 1/(time.time()-startTime)
            

            out.write(original_image)
            # cv2.imwrite("sample.png",img)
            cv2.imwrite(f"data/images/sample_{count}.png",original_image)
            print(f"frame number {count}")

        cap.release()





def main():
    track = Track()
    opt = track.opt
    ext_delegate = None
    if opt.ext_delegate:
        if os.path.isfile('/usr/lib/libvx_delegate.so'):
            ext_delegate = [tflite.load_delegate('/usr/lib/libvx_delegate.so')]
            print(f'loaded {ext_delegate}')
    try:
        interpreter = tflite.Interpreter(model_path = 'weight/ssd_mobilenet_v1_1_default_1.tflite',experimental_delegates=ext_delegate,num_threads=opt.num_threads)
        # reidInterpreter = tflite.Interpreter(model_path = 'weight/model_light_reid_dynamic_int8_version2.tflite',num_threads=opt.num_threads,experimental_delegates=ext_delegate)
    except (ValueError, NameError) as e:
        sys.stderr.write(f" Unable to find  \n{e}")
    interpreter.allocate_tensors()
    floating_model = interpreter.get_input_details()[0]['dtype'] == np.float32
    _, HEIGHT, WIDTH, _ = interpreter.get_input_details()[0]['shape']
    print(f"Height and Weight accepted by the model:{HEIGHT,WIDTH}")

    track.infer_video(interpreter,(HEIGHT, WIDTH))

if __name__ == "__main__":
    main()
