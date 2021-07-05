#!/usr/bin/python3

import numpy as np
import yarp
import sys
import cv2
import pickle as pk

from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_JOINTS
from utilities import read_openpose_data, get_features


def create_bottle(output):
    centroid = output[0]
    centroid_bottle = yarp.Bottle()
    centroid_bottle.addDouble(centroid[0])
    centroid_bottle.addDouble(centroid[1])

    output_bottle = yarp.Bottle()
    output_bottle.addList().read(centroid_bottle)
    output_bottle.addInt(int(output[1]))
    output_bottle.addDouble(float(output[2]))

    return output_bottle


def draw_on_img(img, centroid, y_pred, prob):

    # write index close to the centroid
    #img = cv2.putText(img, str(id), tuple([int(centroid[0]), int(centroid[1])-100]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    if y_pred == 0:
        txt = 'EC NO - c: %0.2f' % prob
    else:
        txt = 'EC YES - c: %0.2f' % prob

    img = cv2.putText(img, txt, tuple([int(centroid[0]), int(centroid[1])-100]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return img


yarp.Network.init()

class Classifier(yarp.RFModule):
    def configure(self, rf):
        self.module_name = rf.find("module_name").asString()

        # input port for rgb image
        self.in_port_human_image = yarp.BufferedPortImageRgb()
        self.in_port_human_image.open('/classifier/image:i')
        self.in_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.in_buf_human_image = yarp.ImageRgb()
        self.in_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1], self.in_buf_human_array.shape[0])
        print('{:s} opened'.format('/classifier/image:i'))

        # input port for openpose data
        self.in_port_human_data = yarp.BufferedPortBottle()
        self.in_port_human_data.open('/classifier/data:i')
        print('{:s} opened'.format('/classifier/data:i'))

        # output port for the prediction
        self.out_port_prediction = yarp.Port()
        self.out_port_prediction.open('/classifier/pred:o')
        print('{:s} opened'.format('/classifier/pred:o'))
        #self.attach(self.out_port_prediction)

        # output port for rgb image
        self.out_port_human_image = yarp.Port()
        self.out_port_human_image.open('/classifier/image:o')
        self.out_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image = yarp.ImageRgb()
        self.out_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image.setExternal(self.out_buf_human_array.data, self.out_buf_human_array.shape[1], self.out_buf_human_array.shape[0])
        print('{:s} opened'.format('/classifier/image:o'))

        self.clf = pk.load(open('model_svm.pkl', 'rb'))
        self.threshold = 3           # to reset the buffer
        self.buffer = ((0, 0), 0, 0) # centroid, prediction and level of confidence
        self.counter = 0             # counter for the threshold
        self.svm_buffer = []

        return True

    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            out_command = 'quit'
            out_pred_bottle = self.out_port_prediction.prepare()
            out_pred_bottle.clear()
            out_pred_bottle.addString(out_command)
            self.out_port_prediction.write()
            reply.addString('quit command sent')
        else:
            print('Command {:s} not recognized'.format(command.get(0).asString()))
            reply.addString('Command {:s} not recognized'.format(command.get(0).asString()))

        return True

    def cleanup(self):
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_prediction.close()
        print('Cleanup function')

    def interruptModule(self):
        print('Interrupt function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_prediction.close()
        return True

    def getPeriod(self):
        return 0.001

    def updateModule(self):
        received_image = self.in_port_human_image.read()
        received_data = self.in_port_human_data.read(False) #non blocking

        if received_image:
            self.in_buf_human_image.copy(received_image)
            human_image = np.copy(self.in_buf_human_array)

            if received_data:
                poses, conf_poses, faces, conf_faces = read_openpose_data(received_data)
                # get features of all people in the image
                data = get_features(poses, conf_poses, faces, conf_faces)

                if data:
                    # predict model
                    # start from 2 because there is the centroid valued in the position [0,1]
                    ld = np.array(data)
                    x = ld[:, 2:(NUM_JOINTS * 2) + 2]
                    c = ld[:, (NUM_JOINTS * 2) + 2:ld.shape[1]]
                    # weight the coordinates for its confidence value
                    wx = np.concatenate((np.multiply(x[:, ::2], c), np.multiply(x[:, 1::2], c)), axis=1)

                    # return a prob value between 0,1 for each class
                    y_classes = self.clf.predict_proba(wx)
                    # take only the person with id 0, we suppose that there is only one person in the scene
                    prob = max(y_classes[0])
                    y_pred = (np.where(y_classes[0] == prob))[0]

                    if len(self.svm_buffer) == 3:
                        self.svm_buffer.pop(0)

                    self.svm_buffer.append([y_pred[0], prob])

                    count_class_0 = [self.svm_buffer[i][0] for i in range(0, len(self.svm_buffer))].count(0)
                    count_class_1 = [self.svm_buffer[i][0] for i in range(0, len(self.svm_buffer))].count(1)
                    y_winner = np.argmax([count_class_0, count_class_1])
                    prob_values = np.array(
                        [self.svm_buffer[i][1] for i in range(0, len(self.svm_buffer)) if self.svm_buffer[i][0] == y_winner])
                    prob_mean = np.mean(prob_values)
                    pred = create_bottle(((ld[0,0], ld[0,1]), y_winner, prob_mean))
                    human_image = draw_on_img(human_image, (ld[0,0], ld[0,1]), y_winner, prob_mean)

                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_prediction.write(pred)

                    self.buffer = ((ld[0,0], ld[0,1]), y_winner, prob_mean)
                    self.counter = 0
                else:
                    human_image = draw_on_img(human_image, self.buffer[0], self.buffer[1], self.buffer[2])
                    pred = create_bottle(self.buffer)

                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_prediction.write(pred)

                    self.counter = self.counter + 1

            else:
                if self.counter < self.threshold:
                    # send in output the buffer
                    human_image = draw_on_img(human_image, self.buffer[0], self.buffer[1], self.buffer[2])
                    pred = create_bottle(self.buffer)

                    # write rgb image
                    self.out_buf_human_array[:, :] = human_image
                    # write prediction bottle
                    self.out_port_prediction.write(pred)

                    self.counter = self.counter + 1
                else:
                    # send in output only the image without prediction
                    self.out_buf_human_array[:, :] = human_image

            self.out_port_human_image.write(self.out_buf_human_image)

        return True


if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("Classifier")

    # conffile = rf.find("from").asString()
    # if not conffile:
    #     print('Using default conf file')
    #     rf.setDefaultConfigFile('../config/manager_conf.ini')
    # else:
    #     rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = Classifier()
    manager.runModule(rf)