from track import Track
from utils.utils import *
import sys
import os
import numpy as np 

def main():
    track = Track()
    opt = track.opt
    ext_delegate = None
    if opt.ext_delegate:
        if os.path.isfile('/usr/lib/libvx_delegate.so'):
            ext_delegate = [tflite.load_delegate('/usr/lib/libvx_delegate.so')]
            print(f'loaded {ext_delegate}')
    
    try:
        interpreter = load_model('weight/ssd_mobilenet_v1_1_default_1.tflite',experimental_delegates=opt.ext_delegate,num_threads=opt.num_threads)
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