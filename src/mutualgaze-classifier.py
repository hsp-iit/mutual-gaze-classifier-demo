#!/usr/bin/python3

import numpy as np
import yarp
import sys
import pickle as pk
import distutils.util
import cv2
import time, math

from functions.config import IMAGE_HEIGHT, IMAGE_WIDTH, get_keypoints_inds
from functions.utilities import read_openpose_data, get_features
from functions.utilities import draw_on_img, create_bottle, get_mean_depth_over_area


yarp.Network.init()

class MutualGazeClassifier(yarp.RFModule):

    def configure(self, rf):
        self.model_name = rf.find("model_name").asString()
        print('SVM model file: %s' % self.model_name)
        self.clf = pk.load(open('./src/functions/' + self.model_name, 'rb'))
        self.MAX_FRAMERATE = bool(distutils.util.strtobool((rf.find("max_framerate").asString())))
        print('Max framerate: %s' % str(self.MAX_FRAMERATE))
        self.threshold = rf.find("max_propagation").asInt32()  # to reset the buffer
        print('SVM Buffer threshold: %d' % self.threshold)
        self.MAX_DURATION_MG = rf.find("duration_mutual_gaze").asInt32()  # duration of mutual gaze in milliseconds
        print('Duration of mutual gaze to be significant: %d' % self.MAX_DURATION_MG)

        self.keypoint_detector = rf.find("keypoint_detector").asString()
        self.JOINTS_POSE, self.JOINTS_FACE = get_keypoints_inds(self.keypoint_detector)
        self.NUM_JOINTS = len(self.JOINTS_POSE) + len(self.JOINTS_FACE)
        self.pose_conf_threshold = rf.find("pose_conf_threshold").asFloat32()
        self.face_conf_threshold = rf.find("face_conf_threshold").asFloat32()

        self.buffer = ('', (0, 0), 0, 0, 0)  # centroid, prediction and level of confidence
        self.counter = 0  # counter for the threshold
        self.svm_buffer_size = 3
        self.svm_buffer = []
        self.id_image = '%08d' % 0
        self.history = []

        self.cmd_port = yarp.Port()
        self.cmd_port.open('/mutualgaze/command:i')
        print('{:s} opened'.format('/mutualgaze/command:i'))
        self.attach(self.cmd_port)

        # input port for rgb image
        self.in_port_human_image = yarp.BufferedPortImageRgb()
        self.in_port_human_image.open('/mutualgaze/image:i')
        self.in_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.in_buf_human_image = yarp.ImageRgb()
        self.in_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1], self.in_buf_human_array.shape[0])
        print('{:s} opened'.format('/mutualgaze/image:i'))

        # input port for depth
        self.in_port_human_depth = yarp.BufferedPortImageFloat()
        self.in_port_human_depth_name = '/mutualgaze/depth:i'
        self.in_port_human_depth.open(self.in_port_human_depth_name)
        self.in_buf_human_depth_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        self.in_buf_human_depth = yarp.ImageFloat()
        self.in_buf_human_depth.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_depth.setExternal(self.in_buf_human_depth_array.data, self.in_buf_human_depth_array.shape[1], self.in_buf_human_depth_array.shape[0])
        print('{:s} opened'.format('/mutualgaze/depth:i'))

        # input port for openpose data
        self.in_port_human_data = yarp.BufferedPortBottle()
        self.in_port_human_data.open('/mutualgaze/data:i')
        print('{:s} opened'.format('/mutualgaze/data:i'))

        # output port for the prediction
        self.out_port_framed_prediction = yarp.Port()
        self.out_port_framed_prediction.open('/mutualgaze/framed/pred:o')
        print('{:s} opened'.format('/mutualgaze/framed/pred:o'))

        # output port for the significant mutual gaze
        self.out_port_timed_prediction = yarp.Port()
        self.out_port_timed_prediction.open('/mutualgaze/timed/pred:o')
        print('{:s} opened'.format('/mutualgaze/timed/pred:o'))

        # output port for rgb image
        self.out_port_human_image = yarp.Port()
        self.out_port_human_image.open('/mutualgaze/image:o')
        self.out_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image = yarp.ImageRgb()
        self.out_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image.setExternal(self.out_buf_human_array.data, self.out_buf_human_array.shape[1], self.out_buf_human_array.shape[0])
        print('{:s} opened'.format('/mutualgaze/image:o'))

        # propag input image
        self.out_port_propag_image = yarp.Port()
        self.out_port_propag_image.open('/mutualgaze/propag:o')
        self.out_buf_propag_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_propag_image = yarp.ImageRgb()
        self.out_buf_propag_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_image.setExternal(self.out_buf_propag_array.data, self.out_buf_propag_array.shape[1],
                                              self.out_buf_propag_array.shape[0])
        print('{:s} opened'.format('/mutualgaze/propag:o'))

        # output port for dumper
        self.out_port_human_image_dump = yarp.Port()
        self.out_port_human_image_dump.open('/mutualgaze/dump:o')
        self.out_buf_human_array_dump = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image_dump = yarp.ImageRgb()
        self.out_buf_human_image_dump.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image_dump.setExternal(self.out_buf_human_array_dump.data, self.out_buf_human_array_dump.shape[1], self.out_buf_human_array_dump.shape[0])
        print('{:s} opened'.format('/mutualgaze/dump:o'))

        self.human_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.human_image_depth = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)

        return True

    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            print('received command QUIT')
            self.cleanup()
            reply.addString('QUIT command sent')
        elif command.get(0).asString() == 'get':
            print('received command GET')
            self.out_buf_human_array_dump[:, :] = self.human_image
            self.out_port_human_image_dump.write(self.out_buf_human_image_dump)
            reply.copy(create_bottle(self.buffer))
        else:
            print('Command {:s} not recognized'.format(command.get(0).asString()))
            reply.addString('Command {:s} not recognized'.format(command.get(0).asString()))

        return True

    def cleanup(self):
        print('Cleanup function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.in_port_human_depth.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_framed_prediction.close()
        self.out_port_timed_prediction.close()
        self.out_port_human_image_dump.close()
        self.cmd_port.close()
        return True

    def interruptModule(self):
        print('Interrupt function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.in_port_human_depth.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_framed_prediction.close()
        self.out_port_timed_prediction.close()
        self.out_port_human_image_dump.close()
        self.cmd_port.close()
        return True

    def getPeriod(self):
        return 0.001

    def updateModule(self):

        received_image = self.in_port_human_image.read()
        received_depth = self.in_port_human_depth.read(False) # non blocking

        if received_image:
            self.in_buf_human_image.copy(received_image)
            human_image = np.copy(self.in_buf_human_array)

            self.human_image = np.copy(human_image)
            self.id_image = '%08d' % ((int(self.id_image) + 1) % 100000)

            if received_depth:
                self.in_buf_human_depth.copy(received_depth)
                self.human_image_depth = np.copy(self.in_buf_human_depth_array)

            if self.MAX_FRAMERATE:
                received_data = self.in_port_human_data.read(False)  # non blocking
            else:
                received_data = self.in_port_human_data.read(False)

            if received_data:
                poses, conf_poses, faces, conf_faces = read_openpose_data(received_data)

                if self.pose_conf_threshold > 0 and self.face_conf_threshold > 0:
                    poses = [pose for pose, conf in zip(poses, conf_poses) if conf.mean() > self.pose_conf_threshold]
                    conf_poses= [conf for conf in conf_poses if conf.mean() > self.pose_conf_threshold]
                    faces = [face for face, conf in zip(faces, conf_faces) if conf.mean() > self.face_conf_threshold]
                    conf_faces= [conf for conf in conf_faces if conf.mean() > self.face_conf_threshold]

                # get features of all people in the image
                data = get_features(poses, conf_poses, faces, conf_faces, self.JOINTS_POSE, self.JOINTS_FACE)

                if data:
                    # predict model
                    # start from 2 because there is the centroid valued in the position [0,1]
                    ld = np.array(data)
                    x = ld[:, 2:(self.NUM_JOINTS * 2) + 2]
                    c = ld[:, (self.NUM_JOINTS * 2) + 2:ld.shape[1]]
                    # weight the coordinates for its confidence value
                    wx = np.concatenate((np.multiply(x[:, ::2], c), np.multiply(x[:, 1::2], c)), axis=1)

                    # return a prob value between 0,1 for each class
                    y_classes = self.clf.predict_proba(wx)
                    # take only the person with id 0, we suppose that there is only one person in the scene
                    itP = 0
                    prob = max(y_classes[itP])
                    y_pred = (np.where(y_classes[itP] == prob))[0]

                    if len(self.svm_buffer) == self.svm_buffer_size:
                        self.svm_buffer.pop(0)

                    self.svm_buffer.append([y_pred[0], prob])

                    count_class_0 = [self.svm_buffer[i][0] for i in range(0, len(self.svm_buffer))].count(0)
                    count_class_1 = [self.svm_buffer[i][0] for i in range(0, len(self.svm_buffer))].count(1)
                    if count_class_1 == count_class_0:
                        y_winner = y_pred[0]
                        prob_mean = prob
                    else:
                        y_winner = np.argmax([count_class_0, count_class_1])
                        prob_values = np.array(
                            [self.svm_buffer[i][1] for i in range(0, len(self.svm_buffer)) if self.svm_buffer[i][0] == y_winner])
                        prob_mean = np.mean(prob_values)

                    if self.human_image_depth is not None:
                        depth = get_mean_depth_over_area(self.human_image_depth, [int(ld[itP,0]), int(ld[itP,1])], 30)
                    else:
                        depth = -1

                    pred = create_bottle((self.id_image, (int(ld[itP,0]), int(ld[itP,1])), depth, y_winner, prob_mean))
                    human_image = draw_on_img(human_image, self.id_image, (ld[itP,0], ld[itP,1]), y_winner, prob_mean)

                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)
                    self.out_port_framed_prediction.write(pred)
                    # propag received image
                    self.out_buf_propag_array[:, :] = self.human_image
                    self.out_port_propag_image.write(self.out_buf_propag_image)

                    self.buffer = (self.id_image, (int(ld[itP,0]), int(ld[itP,1])), depth, y_winner, prob_mean)
                    self.counter = 0

                    self.history.append((time.time(), y_winner))
                    if len(self.history) > 0:
                        start = (self.history[0])[0]
                        end = (self.history[-1])[0]
                        elapsed_ms = (end-start)*1000
                        if elapsed_ms >= self.MAX_DURATION_MG:
                            # check prediction == 1 > 95% of history
                            count_eye_contact = [self.history[i][1] for i in range(0, len(self.history))].count(1)
                            # example: 5 fps with MAX_DURATION_MG = 1500 ms (around 8 frame)
                            # send the result when 6 out 8 frame
                            if count_eye_contact > math.floor((len(self.history)/100)*80):
                                # write to the output
                                timed_pred = yarp.Bottle()
                                timed_pred.addString("mutual-gaze")
                                timed_pred.addInt32(int(elapsed_ms))
                                timed_pred.addInt32(len(self.history))
                                self.out_port_timed_prediction.write(timed_pred)
                                print("Bottle sent to port /mutualgaze/timed/pred:o: %s", timed_pred.toString())
                            else:
                                print("Time is elapsed but count_eye_contact < 80 percent. Percentage eye contact: %d" % math.floor((count_eye_contact/len(self.history))*100))

                            # reset the history
                            self.history = []
                else:
                    pred = create_bottle((self.id_image, (), -1, -1, -1))
                    human_image = cv2.putText(human_image, 'id: ' + str(self.id_image), tuple([25, 30]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                    # send in output only the image with prediction set to -1 (invalid value)
                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)
                    self.out_port_framed_prediction.write(pred)

                    self.buffer = (self.id_image, (), -1, -1, -1)
                    self.counter = 0
            else:
                if self.MAX_FRAMERATE and (self.counter < self.threshold):
                    # send in output the buffer
                    self.buffer = (self.id_image, self.buffer[1], self.buffer[2], self.buffer[3], self.buffer[4])
                    if self.buffer[1]:
                        human_image = draw_on_img(human_image, self.buffer[0], self.buffer[1], self.buffer[3], self.buffer[4])
                    else:
                        human_image = cv2.putText(human_image, 'id: ' + str(self.id_image), tuple([25, 30]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                    pred = create_bottle(self.buffer)

                    # write rgb image
                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)
                    self.out_port_framed_prediction.write(pred)

                    self.counter = self.counter + 1
                else:
                    human_image = cv2.putText(human_image, 'id: ' + str(self.id_image), tuple([25, 30]), cv2.FONT_HERSHEY_SIMPLEX,
                                              0.6, (0, 0, 255), 2,
                                              cv2.LINE_AA)
                    # send in output only the image with the id
                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)

            # propag received image
            self.out_buf_propag_array[:, :] = self.human_image
            self.out_port_propag_image.write(self.out_buf_propag_image)

        return True


if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("MutualGazeClassifier")
    rf.setDefaultConfigFile('./app/config/classifier_conf.ini')

    rf.configure(sys.argv)

    # Run module
    manager = MutualGazeClassifier()
    manager.runModule(rf)
