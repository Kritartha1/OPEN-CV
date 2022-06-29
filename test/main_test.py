# coding=utf-8
import os
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np
from MultiMsgSync import TwoStageHostSeqSync

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")

args = parser.parse_args()

def frame_norm(frame, bbox):
    #bbox=[0.2,0.3]--> returned by neural network output. 
    #NN model returns frames into normal form within [0,1] range .
    #So,if NN returned 0.2 and frame size wwas 200 px. So , actual mormaloized frame size is 40px.
    #eg:frame.shape=[300px,400px ]
    normVals = np.full(len(bbox), frame.shape[0])#array of length bbox filled with value=frame.shape[0]
    # [300,300]
    normVals[::2] = frame.shape[1] #values at 0,2,4,6....indexes=frame.shape[1]
    # [300,300]-->[400,300]

    #normalization
    
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int) #returns normVals array but negative values are clipped to 0 and values >=1 are kept intact.
    #np.clip(np.array(bbox), 0, 1)=[0.2,0.3]
    #(np.clip(np.array(bbox), 0, 1) * normVals).astype(int)=[30,80]

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

VIDEO_SIZE = (1072, 1072)
databases = "databases"
if not os.path.exists(databases):
    os.mkdir(databases)

#Putting text in the frame
class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)#black
        self.color = (255, 255, 255)#white
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX #font style
        self.line_type = cv2.LINE_AA #type of line
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type) #font scale=1 and thickness =4
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type)

class FaceRecognition:
    def __init__(self, db_path, name) -> None:
        self.read_db(db_path)
        self.name = name
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.printed = True

    def cosine_distance(self, a, b): #cosine returns similarity between two vectors
        if a.shape != b.shape:
            
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        a_norm = np.linalg.norm(a)   #modulus of vecotr a
        b_norm = np.linalg.norm(b)   #modulus of vecotr b
        
        return np.dot(a, b.T) / (a_norm * b_norm)   #cos(theta)= A.B/mod(A).mod(B)
        #cosine value higher means angle smaller i.e similarity high!!!

    def new_recognition(self, results): #will return name array of the most matched person with value: i.e name=[disimilarity,name]
        conf = []
        max_ = 0
        label_ = None
        #labels: 1.npz, 2.npz,......
        #dic key: 1  value: data1,data2.....
        for label in list(self.labels): #in read_db method ,there is  self.labels = []
            for j in self.db_dic.get(label): #db_dic : dictionary of person data with name as label and data array as value
                conf_ = self.cosine_distance(j, results)
                if conf_ > max_:
                    max_ = conf_
                    label_ = label

        conf.append((max_, label_)) #max_=max similarity between result and a data with label as person name
        name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0], "UNKNOWN")
        #if similarity is >=0.5 ,then the face is of person name="{label_}"
        #1 - conf[0][0] is dis-similarity between result and a data with label as person name.


        # self.putText(frame, f"name:{name[1]}", (coords[0], coords[1] - 35))
        # self.putText(frame, f"conf:{name[0] * 100:.2f}%", (coords[0], coords[1] - 10))

        if name[1] == "UNKNOWN":
            self.create_db(results)
        return name

    def read_db(self, databases_path):
        self.labels = []
        for file in os.listdir(databases_path):
            filename = os.path.splitext(file)  #abc.npz   abc  npz
            if filename[1] == ".npz":
                self.labels.append(filename[0]) #getting the names from the database

#["p1","p2","p3","p4","p5","p6","p7]
        self.db_dic = {}
        for label in self.labels:
            #p1.npz --db 
            #arr1 arr2....
            #dic : key: {labels} val:[arr1,arr2,arr3,arr4,arr5,arr6]
            with np.load(f"{databases_path}/{label}.npz") as db:
                self.db_dic[label] = [db[j] for j in db.files] #saving database as key:person name and val:array of all the datas.

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1, self.color, 1, self.line_type)

    def create_db(self, results):
       
        #if name is not given as parameter then don't create db.
        if self.name is None:
            if not self.printed:
                print("Wanted to create new DB for this face, but --name wasn't specified")
                self.printed = True
            return
        print('Saving face...')
        try:
            with np.load(f"{databases}/{self.name}.npz") as db:
                #db.file
                db_ = [db[j] for j in db.files][:] #db[i][:] all contents of ith row
        except Exception as e:
            db_ = []
        db_.append(np.array(results))
        
        #database: [1],[2],[3]............
        #[1]:[1.png, 1.npz] 
        #[unknown]:[unknown.png,unknown.npz]
        #[person]:[person,png,person.npz]

        #python main.py --name "K"
        #[K]:[K.png,K.npz]

        #python main.py 
        #[unknown]:[unknown.png,unknown.npz]
        #----->[p2]:[p2.png,p2.npz]

        np.savez_compressed(f"{databases}/{self.name}", *db_)#save all data of db_ at database/name.npz file
        self.adding_new = False

print("Creating pipeline...")
pipeline = dai.Pipeline()


#cmaera node
print("Creating Color Camera...")
cam = pipeline.create(dai.node.ColorCamera)
# For ImageManip rotate you need input frame of multiple of 16
cam.setPreviewSize(1072, 1072)
cam.setVideoSize(VIDEO_SIZE)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)


#X_linkout--connect to host
host_face_out = pipeline.create(dai.node.XLinkOut)
host_face_out.setStreamName('color')
cam.video.link(host_face_out.input)

# ImageManip as a workaround to have more frames in the pool.
# cam.preview can only have 4 frames in the pool before it will
# wait (freeze). Copying frames and setting ImageManip pool size to
# higher number will fix this issue.
copy_manip = pipeline.create(dai.node.ImageManip)
cam.preview.link(copy_manip.inputImage)  #putting color camera imageoutput to image manip i/p
copy_manip.setNumFramesPool(20)
copy_manip.setMaxOutputFrameSize(1072*1072*3)

# ImageManip that will crop the frame before sending it to the Face detection NN node
face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300) #300*300
copy_manip.out.link(face_det_manip.inputImage) #setting copy_manip output to face_det_manip's input.

# NeuralNetwork
print("Creating Face Detection Neural Network...")
face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
face_det_nn.setConfidenceThreshold(0.5)
face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
# Link Face ImageManip -> Face detection NN node
face_det_manip.out.link(face_det_nn.input)

face_det_xout = pipeline.create(dai.node.XLinkOut)
face_det_xout.setStreamName("detection")
face_det_nn.out.link(face_det_xout.input)

# Script node will take the output from the face detection NN as an input and set ImageManipConfig
# to the 'age_gender_manip' to crop the initial frame
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

face_det_nn.out.link(script.inputs['face_det_in'])
# We also interested in sequence number for syncing
face_det_nn.passthrough.link(script.inputs['face_pass'])

copy_manip.out.link(script.inputs['preview'])

with open("script.py", "r") as f:
    script.setScript(f.read())

print("Creating Head pose estimation NN")

headpose_manip = pipeline.create(dai.node.ImageManip)
headpose_manip.initialConfig.setResize(60, 60)
headpose_manip.setWaitForConfigInput(True)
script.outputs['manip_cfg'].link(headpose_manip.inputConfig)
script.outputs['manip_img'].link(headpose_manip.inputImage)

headpose_nn = pipeline.create(dai.node.NeuralNetwork)
headpose_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=6))
headpose_manip.out.link(headpose_nn.input)

headpose_nn.out.link(script.inputs['headpose_in'])
headpose_nn.passthrough.link(script.inputs['headpose_pass'])

#################################################################################################################
#NN mask 1
print("Creating Mask recognition NN ImageManip")
mask_rec_manip = pipeline.create(dai.node.ImageManip)
mask_rec_manip.initialConfig.setResize(224, 224)
#mask_rec_manip should be same as scipt node ouptput image size
mask_rec_manip.setWaitForConfigInput(True)
script.outputs['manip3_cfg'].link(mask_rec_manip.inputConfig)
script.outputs['manip3_img'].link(mask_rec_manip.inputImage)

# Second stange recognition NN---Mask
print("Creating recognition Neural Network for mask...")
mask_rec_nn = pipeline.create(dai.node.NeuralNetwork)
mask_rec_nn.setBlobPath(blobconverter.from_zoo(name="sbd_mask_classification_224x224", zoo_type="depthai", shaves=6))
mask_rec_manip.out.link(mask_rec_nn.input)


mask_rec_xout = pipeline.create(dai.node.XLinkOut)
mask_rec_xout.setStreamName("mask-recognition")
mask_rec_nn.out.link(mask_rec_xout.input)
########################################################################################################################################


print("Creating face recognition ImageManip/NN")

face_rec_manip = pipeline.create(dai.node.ImageManip)
face_rec_manip.initialConfig.setResize(112, 112)
face_rec_manip.setWaitForConfigInput(True)

script.outputs['manip2_cfg'].link(face_rec_manip.inputConfig)
script.outputs['manip2_img'].link(face_rec_manip.inputImage)

face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
face_rec_nn.setBlobPath(blobconverter.from_zoo(name="face-recognition-arcface-112x112", zoo_type="depthai", shaves=6))
face_rec_manip.out.link(face_rec_nn.input)

arc_xout = pipeline.create(dai.node.XLinkOut)
arc_xout.setStreamName('recognition')
face_rec_nn.out.link(arc_xout.input)


with dai.Device(pipeline) as device:
    facerec = FaceRecognition(databases, args.name)
    sync = TwoStageHostSeqSync()
    text = TextHelper()

    queues = {}
    # Create output queues
    for name in ["color", "detection", "recognition","mask-recognition"]:
        queues[name] = device.getOutputQueue(name)

    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, object detections and face recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name) 

        msgs = sync.get_msgs()
        if msgs is not None:
            frame = msgs["color"].getCvFrame() 
            dets = msgs["detection"].detections
            #mask-recognitions = msgs["mask-recognition"]

            for i, detection in enumerate(dets):
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

                features = np.array(msgs["recognition"][i].getFirstLayerFp16())
                
                conf, name = facerec.new_recognition(features)  #conf: [cosine,name]
                

                rec = np.array(msgs["mask-recognition"][i].getFirstLayerFp16())
                #rec = mask-recognitions[i].getFirstLayerFp16()
                index = np.argmax(log_softmax(rec))
                texts = "No Mask"
                color = (0,0,255) # Red
                if index == 1:
                    texts = "Mask"
                    color = (0,255,0)
                    
                #text.putText(frame, f"{name} {(100*conf):.0f}% {texts}", (bbox[0] + 10,bbox[1] + 35))
                if not os.path.exists(f"{databases}/{name}/{name}.png"):
                    cv2.imwrite(f"{databases}/{name}/{name}.png", frame)
                text.putText(frame, f"{name} {(100*conf):.0f}% ", (bbox[0] + 10,bbox[1] + 35))
                y = (bbox[1] + bbox[3]) // 2
                cv2.putText(frame, texts, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 4)
                cv2.putText(frame, texts, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)


            cv2.imshow("color", cv2.resize(frame, (800,800)))

        if cv2.waitKey(1) == ord('q'):
            break
