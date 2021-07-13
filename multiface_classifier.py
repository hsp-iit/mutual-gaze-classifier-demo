#!/usr/bin/python3

import numpy as np
import yarp
import sys
import pickle as pk
import distutils.util

from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_JOINTS
from utilities import read_openpose_data, get_features
from utilities import draw_on_img, get_human_idx, create_bottle


yarp.Network.init()

class MultiFaceClassifier(yarp.RFModule):

    def configure(self, rf):
        self.clf = pk.load(open('model_svm.pkl', 'rb'))
        self.MAX_FRAMERATE = bool(distutils.util.strtobool((rf.find("max_framerate").asString())))
        self.threshold = rf.find("max_propagation").asInt32()  # to reset the buffer
        self.buffer = yarp.Bottle()  # each element is ((0, 0), 0, 0) centroid, prediction and level of confidence
        self.counter = 0  # counter for the threshold
        self.svm_buffer_size = 3
        self.svm_buffer = (np.array([], dtype=object)).tolist()
        self.id_image = '%08d' % 0

        self.cmd_port = yarp.Port()
        self.cmd_port.open('/classifier/command:i')
        print('{:s} opened'.format('/classifier/command:i'))
        self.attach(self.cmd_port)

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

        # output port for rgb image
        self.out_port_human_image = yarp.Port()
        self.out_port_human_image.open('/classifier/image:o')
        self.out_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image = yarp.ImageRgb()
        self.out_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image.setExternal(self.out_buf_human_array.data, self.out_buf_human_array.shape[1], self.out_buf_human_array.shape[0])
        print('{:s} opened'.format('/classifier/image:o'))

        self.human_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

        return True

    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            print('received command QUIT')
            self.cleanup()
            reply.addString('quit command sent')
        elif command.get(0).asString() == 'get':
            print('received command GET')
            reply.copy(self.buffer)
        else:
            print('Command {:s} not recognized'.format(command.get(0).asString()))
            reply.addString('Command {:s} not recognized'.format(command.get(0).asString()))

        return True

    def cleanup(self):
        print('Cleanup function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_prediction.close()
        return True

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

        if received_image:
            self.in_buf_human_image.copy(received_image)
            human_image = np.copy(self.in_buf_human_array)

            if self.MAX_FRAMERATE:
                received_data = self.in_port_human_data.read(False)  # non blocking
            else:
                received_data = self.in_port_human_data.read()

            if received_data:
                poses, conf_poses, faces, conf_faces = read_openpose_data(received_data)
                # get features of all people in the image
                data = get_features(poses, conf_poses, faces, conf_faces)

                idx_humans_frame = []
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

                    output_pred_bottle = yarp.Bottle()
                    for itP in range(0, y_classes.shape[0]):
                        prob = max(y_classes[itP])
                        y_pred = (np.where(y_classes[itP] == prob))[0]

                        min_dist, idx = get_human_idx(self.svm_buffer, ld[itP, 0:2])
                        if (idx != None and min_dist != None) and ((len(self.svm_buffer) == y_classes.shape[0]) or (min_dist < 100)): # suppose min dist < 50 frames
                            idx_humans_frame.append(idx)
                            if len(self.svm_buffer[idx]) == 3:
                                self.svm_buffer[idx].pop(0)

                            self.svm_buffer[idx].append([ld[itP, 0], ld[itP, 1], y_pred[0], prob])

                            count_class_0 = [(self.svm_buffer[idx])[i][2] for i in range(0, len(self.svm_buffer[idx]))].count(0)
                            count_class_1 = [(self.svm_buffer[idx])[i][2] for i in range(0, len(self.svm_buffer[idx]))].count(1)
                            if (count_class_1 == count_class_0):
                                y_winner = y_pred[0]
                                prob_mean = prob
                            else:
                                y_winner = np.argmax([count_class_0, count_class_1])
                                prob_values = np.array([(self.svm_buffer[idx])[i][3] for i in range(0, len(self.svm_buffer[idx])) if (self.svm_buffer[idx])[i][2] == y_winner])
                                prob_mean = np.mean(prob_values)

                        else:
                            self.svm_buffer.append([[ld[itP, 0], ld[itP, 1], y_pred[0], prob]])
                            print("add humans to the scene: %d" % len(self.svm_buffer))
                            idx_humans_frame.append(len(self.svm_buffer)-1)
                            y_winner = y_pred[0]
                            prob_mean = prob

                        pred = create_bottle((self.id_image, (ld[itP,0], ld[itP,1]), y_winner, prob_mean))
                        output_pred_bottle.addList().read(pred)
                        human_image = draw_on_img(human_image, self.id_image, (ld[itP,0], ld[itP,1]), y_winner, prob_mean)

                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)
                    self.out_port_prediction.write(output_pred_bottle)

                    self.buffer.copy(output_pred_bottle)
                    self.id_image = '%08d' % ((int(self.id_image) + 1) % 100000)
                    self.human_image = np.copy(human_image)
                    self.counter = 0
                else:
                    if self.MAX_FRAMERATE and (self.counter < self.threshold):
                        for i in range(0, self.buffer.size()):
                            buffer = self.buffer.get(i).asList()
                            centroid = buffer.get(1).asList()
                            human_image = draw_on_img(human_image, buffer.get(0).asString(), (centroid.get(0).asDouble(), centroid.get(1).asDouble()), buffer.get(2).asInt(), buffer.get(3).asDouble())

                        self.out_buf_human_array[:, :] = human_image
                        self.out_port_human_image.write(self.out_buf_human_image)
                        self.out_port_prediction.write(self.buffer)

                        self.counter = self.counter + 1
                    # else:
                    #     self.out_buf_human_array[:, :] = human_image
                    #     self.out_port_human_image.write(self.out_buf_human_image)

                for i in range(0, len(self.svm_buffer)):
                    if not (i in idx_humans_frame):
                        self.svm_buffer.pop(i)
                        print("remove humans in the scene: %d" % len(self.svm_buffer))

            else:
                if self.MAX_FRAMERATE and (self.counter < self.threshold):
                    # send in output the buffer
                    for i in range(0, self.buffer.size()):
                        buffer = self.buffer.get(i).asList()
                        centroid = buffer.get(1).asList()
                        human_image = draw_on_img(human_image, buffer.get(0).asString(), (centroid.get(0).asDouble(), centroid.get(1).asDouble()), buffer.get(2).asInt(), buffer.get(3).asDouble())

                    # write rgb image
                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)
                    # write prediction bottle
                    self.out_port_prediction.write(self.buffer)
                    self.counter = self.counter + 1
                # else:
                #     # send in output only the image without prediction
                #     self.out_buf_human_array[:, :] = human_image
                #     self.out_port_human_image.write(self.out_buf_human_image)

        return True


if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("MultiFaceClassifier")

    conffile = rf.find("from").asString()
    if not conffile:
        print('Using default conf file')
        rf.setDefaultConfigFile('./config/classifier_conf.ini')
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = MultiFaceClassifier()
    manager.runModule(rf)