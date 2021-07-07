#!/usr/bin/python3

import numpy as np
import yarp
import cv2
from config import JOINTS_POSE, JOINTS_FACE


def compute_centroid(points):
    mean_x = np.mean([p[0] for p in points])
    mean_y = np.mean([p[1] for p in points])

    return [mean_x, mean_y]


def joint_set(p, c):
    return (p[0] != 0.0 or p[1] != 0.0) and c > 0.1


def dist_2d(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    squared_dist = np.sum((p1 - p2)**2, axis=0)
    return np.sqrt(squared_dist)


def compute_head_face_features(pose, conf_pose, face, conf_face):

    n_joints_set = [pose[joint] for joint in JOINTS_POSE if joint_set(pose[joint], conf_pose[joint])]
    n_joints_set.extend([face[joint] for joint in JOINTS_FACE if joint_set(face[joint], conf_face[joint])])

    if len(n_joints_set) < 2:
        return None, None

    centroid = compute_centroid(n_joints_set)
    max_dist = max([dist_2d(j, centroid) for j in n_joints_set])

    new_repr_pose = [(np.array(pose[joint]) - np.array(centroid)) for joint in JOINTS_POSE]
    new_repr_face = ([(np.array(face[joint]) - np.array(centroid)) for joint in JOINTS_FACE])

    result = []

    for i in range(0, len(JOINTS_POSE)):

        if joint_set(pose[JOINTS_POSE[i]], conf_pose[JOINTS_POSE[i]]):
            result.append([new_repr_pose[i][0] / max_dist, new_repr_pose[i][1] / max_dist])
        else:
            result.append([0, 0])

    for i in range(0, len(JOINTS_FACE)):
        if joint_set(face[JOINTS_FACE[i]], conf_face[JOINTS_FACE[i]]):
            result.append([new_repr_face[i][0] / max_dist, new_repr_face[i][1] / max_dist])
        else:
            result.append([0, 0])

    flat_list = [item for sublist in result for item in sublist]

    for j in JOINTS_POSE:
        if conf_pose[j] > 0.1:
            flat_list.append(conf_pose[j])
        else:
            flat_list.append(0)

    for j in JOINTS_FACE:
        if conf_face[j] > 0.1:
            flat_list.append(conf_face[j])
        else:
            flat_list.append(0)

    return flat_list, centroid


# compute features for all people in the image
def get_features(poses, conf_poses, faces, conf_faces):
    data = []

    for itP in range(0, len(poses)):
        try:
            # compute facial keypoints coordinates w.r.t. to head centroid
            features, centr = compute_head_face_features(poses[itP], conf_poses[itP], faces[itP], conf_faces[itP])
            # if minimal amount of facial keypoints was detected
            if features is not None:
                featMap = np.asarray(features)
                #featMap = np.reshape(featMap, (1, 3 * NUM_JOINTS))
                centr = np.asarray(centr)
                #centr = np.reshape(centr, (1, 2))
                poseFeats = np.concatenate((centr, featMap))

                data.append(poseFeats)
        except Exception as e:
            print("Got Exception: " + str(e))

    return data


def load_many_poses(data):
    poses = []
    confidences = []

    for person in data:
        poses.append(np.array(person)[:, 0:2])
        confidences.append(np.array(person)[:, 2])

    return poses, confidences


def load_many_faces(data):
    faces = []
    confidences = []

    for person in data:
        faces.append(np.array(person)[:, 0:2])
        confidences.append(np.array(person)[:, 2])

    return faces, confidences


def read_openpose_data(received_data):
    body = []
    face = []
    if received_data:
        received_data = received_data.get(0).asList()
        for i in range(0, received_data.size()):
            keypoints = received_data.get(i).asList()

            # if person is not None:
            #     keypoints = person.get(0).asList()

            if keypoints:
                body_person = []
                face_person = []
                for y in range(0, keypoints.size()):
                    part = keypoints.get(y).asList()
                    if part:
                        if part.get(0).asString() == "Face":
                            for z in range(1, part.size()):
                                item = part.get(z).asList()
                                face_part = [item.get(0).asDouble(), item.get(1).asDouble(), item.get(2).asDouble()]

                                face_person.append(face_part)
                        else:
                            body_part = [part.get(1).asDouble(), part.get(2).asDouble(), part.get(3).asDouble()]

                        body_person.append(body_part)

                if body_person and face_person:
                    body.append(body_person)
                    face.append(face_person)

    poses, conf_poses = load_many_poses(body)
    faces, conf_faces = load_many_faces(face)

    return poses, conf_poses, faces, conf_faces


def get_human_idx(buffer_output, centroid):
    #suppose that in the two first element of buffer output there is the centroid
    dist = []
    for i in range(0, len(buffer_output)):
        buffer_person = buffer_output[i]
        if len(buffer_person) != 0:
            dist.append(dist_2d((buffer_person[-1])[0:2], centroid))

    if len(dist) != 0:
        min_dist = min(dist)
        idx = dist.index(min_dist)
        return min_dist, idx
    else:
        return None, None


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
        txt = 'EC NO'
    else:
        txt = 'EC YES'

    img = cv2.putText(img, txt, tuple([int(centroid[0]), int(centroid[1])-120]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    img = cv2.putText(img, 'c: %0.2f' % prob, tuple([int(centroid[0]), int(centroid[1]) - 90]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return img
