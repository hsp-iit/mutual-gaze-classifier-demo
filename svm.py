#!/usr/bin/python3

import numpy as np
import yarp
import pickle as pk

from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_JOINTS
from utilities import read_openpose_data, get_features


yarp.Network.init()

# initialise the model
model_file = 'model_svm.pkl'
clf = pk.load(open(model_file, 'rb'))

buffer_output = []

#if __name__ == '__main__':

in_port_human_data = yarp.BufferedPortBottle()
in_port_human_data_name = '/classifier/data:i'
in_port_human_data.open(in_port_human_data_name)

out_port_prediction = yarp.Port()
out_port_prediction_name = '/classifier/pred:o'
out_port_prediction.open(out_port_prediction_name)

try:
    while True:
        received_data = in_port_human_data.read()

        if not received_data:
            print("Openpose data not received correctly")
            continue

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
            y_classes = clf.predict_proba(wx)

            # for loop is in case of more people
            for itP in range(0, y_classes.shape[0]):
                try:
                    prob = max(y_classes[itP])
                    y_pred = (np.where(y_classes[itP] == prob))[0]

                    if len(buffer_output) == 3:
                        buffer_output.pop(0)

                    buffer_output.append([y_pred[0], prob])

                    count_class_0 = [buffer_output[i][0] for i in range(0, len(buffer_output))].count(0)
                    count_class_1 = [buffer_output[i][0] for i in range(0, len(buffer_output))].count(1)
                    y_winner = np.argmax([count_class_0, count_class_1])
                    prob_values = np.array([buffer_output[i][1] for i in range(0, len(buffer_output)) if buffer_output[i][0] == y_winner])
                    prob_mean = np.mean(prob_values)

                    # send in output normalised face centroid, true/false, confidence level
                    normalised_centroid = yarp.Bottle()
                    normalised_centroid.addDouble(ld[0])
                    normalised_centroid.addDouble(ld[1])

                    output_bottle = yarp.Bottle()
                    output_bottle.addList().read(normalised_centroid)
                    output_bottle.addInt(int(y_winner))
                    output_bottle.addDouble(float(prob_mean))

                except Exception:
                    print("skipped %d" % itP)

            out_port_prediction.write(output_bottle)

except KeyboardInterrupt:  # if ctrl-C is pressed
    print('Closing ports...')
    in_port_human_data.close()
    out_port_prediction.close()