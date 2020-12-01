import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image




def processvid(filename1):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0

    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YoloV3(classes=3)

    yolo.load_weights('./weights/yolov3-custom.tf')
    logging.info('weights loaded')

    class_names = [c.strip() for c in open("./data/labels/classes.txt").readlines()]
    logging.info('classes loaded')


    vid = cv2.VideoCapture('./data/video/'+filename1)

    out = None


    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter('./static/'+filename1 , codec, fps, (width, height))
    list_file = open('detection.txt', 'w')
    frame_index = -1

    fps = 0.0
    count = 0

    countmask = []
    countunmask = []
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else:
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)


        masked=0
        unmasked=0
        newr=5.7
        ro=5.7
        countc=0
        #------------------
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            # color = colors[int(track.track_id) % len(colors)]
            # color = [i * 255 for i in color]
            # cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            # cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0) , 2)
            #cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2) original code do not uncomment
            center_y = int(((bbox[1])+(bbox[3]))/2) # ourcode
            if class_name=="with_mask":
                masked=masked+1
                countmask.append(track.track_id)
                color = (0, 255, 0)

                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                #cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)), int(bbox[1])), color, -1)
                cv2.putText(img, "With Mask" + " : " + str(int(track.track_id)%100),(int(bbox[0]), int(bbox[1]-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color , 2)
                tp=masked+unmasked
                p=masked/tp
                ef=.50
                newr=ro*(1-(ef*p))**2
                if(unmasked>0):
                    countc=countc-1
                #plotting(newr)
                s_img = cv2.imread("foo.jpg")
                s_img = cv2.resize(s_img, (60, 120))

                #img[590:590+120,300:300+60]=s_img




                #cv2.imwrite('image.png',crop_img)


            else:
                color=(0, 0,255)
                crop_img = img[int(bbox[1])-20:int(bbox[3])+20, int(bbox[0])-20:int(bbox[2]+20)]
                #cv2.imwrite("violator/{fname}.jpg".format(fname = track.track_id ),crop_img)

                #cv2.imwrite("violator/{fname}.jpg".format(fname = track.track_id ),crop_img)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                #cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)), int(bbox[1])), color, -1)
                cv2.putText(img, "Without Mask"+ " : " + str(int(track.track_id)%100),(int(bbox[0]), int(bbox[1]-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color , 2)
                countunmask.append(track.track_id)
                unmasked=unmasked+1
                countc=countc+2

                # test
                #crop_img = img[int(bbox[0]):int(bbox[1]), int(bbox[2]):int(bbox[3])]


#####



            #cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            #cv2.putText(img, "{:.1f}in".format(i+1), (int(x-5), int(y - 10)),
            #    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
#r naught
        if(newr>=2):
            cv2.rectangle(img, (int(5), int(height-120)) , ( int(600), int(height-5)),(255,255,255), 2)
            cv2.putText(img, "R0: {:.1f} ".format(newr), (int(10), int(height-100)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        elif(newr<2):
            cv2.rectangle(img, (int(5), int(height-120)) , ( int(300), int(height-5)),(255,255,255), 2)
            cv2.putText(img, "R0: {:.1f} ".format(newr), (int(10), int(height-100)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)



        cv2.putText(img, "Masked: {:.1f} ".format(masked), (int(10), int(height-80)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
#masked
        if(unmasked>masked):
            cv2.putText(img, "Un-Masked : {:.1f} ".format(unmasked), (int(10), int(height-60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200+countc), 2)
        elif(unmasked<=masked):
            cv2.putText(img, "Un-Masked : {:.1f} ".format(unmasked), (int(10), int(height-60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            # v2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2

        cv2.putText(img, "Un-Masked total : {:.1f} ".format(len(set(countunmask))), (int(10), int(height-40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(img, "Masked total : {:.1f} ".format(len(set(countmask))), (int(10), int(height-20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        #for det in detections:
        #    bbox = det.to_tlbr()
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)



        # print fps on screen

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)

        out.write(img)
        frame_index = frame_index + 1
        list_file.write(str(frame_index)+' ')
        if len(converted_boxes) != 0:
            for i in range(0,len(converted_boxes)):
                list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
        list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()

    out.release()
    list_file.close()
    cv2.destroyAllWindows()
    return newr
