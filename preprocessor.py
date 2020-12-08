from mnist import MNIST
import numpy as np


def get_training_files(path):
    mndata = MNIST(path)
    mndata.gz = True
    return mndata.load_training()


def get_testing_files(path):
    mndata = MNIST(path)
    mndata.gz = True
    return mndata.load_testing()


def save_to_txt(values, labels, n, filename):
    labels_delim = []
    for i in range(len(labels)):
        labels_delim.append(str(labels[i]) + " | ")
    file_testing = open(filename, "w")
    totxt = np.array(list(zip(labels_delim, values)))
    np.savetxt(fname=file_testing, X=totxt[:n], fmt='%s')


def main():
    mn_images, mn_labels = get_training_files('./')
    mn_images_testing, mn_labels_testing = get_testing_files('./')

    save_to_txt(mn_images, mn_labels, 10000, "bayes_input_10k.txt")
    save_to_txt(mn_images_testing, mn_labels_testing, 50, "bayes_input_testing.txt")


if __name__ == '__main__':
    main()
