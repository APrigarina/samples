import cv2
import numpy as np
import sys
import time
from itertools import islice
from shapely.geometry import Polygon
import optparse
import os
import sys
from os.path import join

p = optparse.OptionParser()
p.add_option('--input', '-i', default="qrcodes/detection",help="Location of directory with input images")
p.add_option('--output', '-o', default="detection_results",help="Location of output directory results are saved to")

options, arguments = p.parse_args()

dir_images = options.input
dir_results = options.output

os.mkdir(dir_results)

def detect_markers():

    total_detected = 0
    total_decoded = 0

    total = 0

    count_detected = {}
    count_decoded = {}

    qrDecoder = cv2.QRCodeDetector()

    for cat_name in sorted(os.listdir(dir_images)):
        not_detected = 0
        detected = 0

        not_decoded = 0
        decoded = 0

        input_path = os.path.join(dir_images,cat_name)
        output_path = os.path.join(dir_results,cat_name)

        input_images = [f for f in sorted(os.listdir(input_path)) if f.endswith("jpg") or f.endswith("png")]
        input_files = [f for f in sorted(os.listdir(input_path)) if f.endswith("txt")]

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for idx, input_image in enumerate(input_images):
            path_to_image = os.path.join(input_path,input_image)

            image_to_detect = cv2.imread(path_to_image)
            rectifiedImage, bbox = qrDecoder.detect(image_to_detect)

            if bbox is None:
                #print("QR Code not detected")
                not_detected += 1

            else:

                detected += 1
                path_to_file = os.path.join(output_path,input_files[idx])

                if not os.path.exists(path_to_file):
                    file = open(path_to_file, "w")

                    for point in bbox.reshape(4,2):
                        result = ' '.join([str(x) for x in point]) + ' '
                        file.write(result)
                    file.close()

                data, rectifiedImage = qrDecoder.decode(image_to_detect, bbox)

                if len(data)>0:
                    #print("Decoded Data : {}".format(data))
                    decoded += 1
                else:
                    #print("QR Code not decoded")
                    not_decoded += 1

        total_detected += detected
        total_decoded += decoded

        total += (detected + not_detected)

        count_detected[cat_name] = np.round((detected / len(input_images) * 100), 2)

        if detected != 0:
            count_decoded[cat_name] = np.round(((decoded / detected) * 100), 2)
        else:
            count_decoded[cat_name] = 0


    print("total", total)
    print("detected: {}%".format(np.round((total_detected/total) * 100, 2)))
    print("decoded: {}%".format(np.round((total_decoded/total_detected) * 100, 2)))

    print("\nCount of detected QR codes by category")
    for k, v in count_detected.items():
        print(k,v)

    print("\nCount of decoded QR codes by category")
    for k, v in count_decoded.items():
        print(k,v)

detect_markers()
