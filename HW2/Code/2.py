def meanError(data, label):
    u_labels = np.unique(label)
    for i in u_labels:
        cmean = np.mean(data[label == i], axis = 0)
        x = []
        [x.append(cmean[i]) for i in range(len(cmean))]
        x = [x]
        print(np.shape(data[label == i]), np.shape(x))
        distances = cdist(data[label == i], x ,'euclidean')
        print("mean distance for cluster" + str(i) + " " + str(np.mean(distances)))