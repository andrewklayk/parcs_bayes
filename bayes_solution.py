# import gmpy2
from Pyro4 import expose
import random

from requests.packages.urllib3.connectionpool import xrange


class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        print("Inited")

    def solve(self):
        print("Job Started")
        print("Workers %d" % len(self.workers))

        training_images, training_labels, test_images, test_labels = self.read_input()

        try:
            import numpy as np
        except:
            import subprocess
            import sys
            subprocess.check_call([sys.executable,
                                   '-m', 'pip', 'install', 'numpy'])
            import numpy as np

        images_chunks = np.array_split(training_images, len(self.workers))
        labels_chunks = np.array_split(training_labels, len(self.workers))

        # map
        mapped = []
        for i in xrange(0, len(self.workers)):
            print("map %d" % i)
            mapped.append(self.workers[i].mymap((images_chunks[i]).tolist(), (labels_chunks[i]).tolist()))

        # reduce
        priors, means, var = self.myreduce(mapped)
        try:
            import scipy.stats as stats
        except:
            import subprocess
            import sys
            subprocess.check_call([sys.executable,
                                   '-m', 'pip', 'install', 'scipy'])
            import scipy.stats as stats

        log_probabilities = np.ndarray((np.size(test_images, axis=0), 10))
        for i in range(np.size(test_images, axis=0)):
            for k in range(10):
                log_probabilities[i][k] = np.sum(stats.norm(means[k], (var[k])).logpdf(test_images[i]))
        full_img_logprobs = (log_probabilities)
        classes = np.argmax(full_img_logprobs, axis=1)

        # output
        self.write_output(list(zip((classes).tolist(), test_labels)))

        print("Job Finished")

    @staticmethod
    @expose
    def mymap(images, labels):
        means = []
        covs = []
        priors = []
        var = []

        images_by_labels = [[], [], [], [], [], [], [], [], [], []]
        for i in range(len(images)):
            images_by_labels[labels[i]].append(images[i])

        try:
            import numpy as np
        except:
            import subprocess
            import sys
            subprocess.check_call([sys.executable,
                                   '-m', 'pip', 'install', 'numpy'])
            import numpy as np

        for i in xrange(10):
            priors.append(float(labels.count(i)) / float(len(labels)))
            means.append(np.mean(images_by_labels[i], axis=0).tolist())
            # covs.append(np.cov(images_by_labels[i], rowvar=False, bias=False))
            var.append((np.std(images_by_labels[i], axis=0) + 0.01).flatten().tolist())
        return priors, means, var

    @staticmethod
    @expose
    def myreduce(mapped):
        print("reduce")
        priors_f = [0] * 10
        means_f = [0] * 10
        covs_f = [0] * 10
        vars_f = [0] * 10
        try:
            import numpy as np
        except:
            import subprocess
            import sys
            subprocess.check_call([sys.executable,
                                   '-m', 'pip', 'install', 'numpy'])
            import numpy as np

        for map_unit in mapped:
            # covs_mapped = np.fromstring(map_unit.value[2], sep=',').reshape((10, 784, 784))
            print("reduce loop")
            for i in xrange(10):
                priors_f[i] = priors_f[i] + map_unit.value[0][i]
                means_f[i] = ((np.array(means_f[i]) + np.array(map_unit.value[1][i])) / float(len(mapped))).tolist()
                # covs_f[i] = (np.array(covs_f[i]) + np.array(pickle.loads(map_unit.value[2][i])) / float(
                #     len(mapped))).tolist()
                vars_f[i] = ((np.array(vars_f[i]) + np.array(map_unit.value[2][i])) / float(len(mapped))).tolist()

        print("reduce done")
        return priors_f, means_f, vars_f

    def read_input(self):
        try:
            import numpy as np
        except:
            import subprocess
            import sys
            subprocess.check_call([sys.executable,
                                   '-m', 'pip', 'install', 'numpy'])
            import numpy as np
        file = open(self.input_file_name, "r")
        Lines = file.readlines()
        testing = False
        images = []
        labels = []
        images_testing = []
        labels_testing = []
        for line in Lines:
            if line.find("TESTING") != -1:
                testing = True
                continue
            if testing:
                split = line.split(" | ")
                labels_testing.append(int(split[0]))
                images_testing.append((np.fromstring(split[1].strip().replace('[', ''), sep=', ', dtype=int)).tolist())
            else:
                split = line.split(" | ")
                labels.append(int(split[0]))
                images.append((np.fromstring(split[1].strip().replace('[', ''), sep=', ', dtype=int)).tolist())

        return images, labels, images_testing, labels_testing

    def write_output(self, output):
        f = open(self.output_file_name, 'w')
        acc = 0
        for o in output:
            if o[0] == o[1]:
                acc += 1
            f.write("predicted: {0} | actual: {1};   ".format(o[0], o[1]))
        f.write('\n')
        f.write('accuracy = {0}%'.format(float(acc) / float(len(output)) * 100))
        f.close()
        print("output done")
