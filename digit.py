import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
import os
import sys
import csv


"""
Digit class
    Represent every digit in two dimension using PCA

"""
class Digit(object):

    def __init__(self, _digits_dir):
        # Initializing the Digit model
        self.digits_dir = _digits_dir
        # 5 classes: 0 ~ 4
        self.digit_classes = 5

        # Fix the random genertor for writing report only
        np.random.seed(42)

        # Training data
        # self.X_train dim: 5000 * 784
        # self.y_train dim: 5000 * 1
        self.X_train = np.loadtxt(open(self.digits_dir + '/X_train.csv'), delimiter=',')
        self.y_train = np.loadtxt(open(self.digits_dir + '/T_train.csv'), dtype='int', delimiter= ',')
        #print(self.y_train.dtype)

        # Testing data
        # self.X_test dim: 2500 * 784
        # self.y_test dim : 2500 * 1
        self.X_test = np.loadtxt(open(self.digits_dir + '/X_test.csv'), delimiter=',')
        self.y_test = np.loadtxt(open(self.digits_dir + '/T_test.csv'), dtype='int', delimiter= ',')
        self.reduce_dim()
    
    def reduce_dim(self):
        # Use PCA to reduce dim: 784 -> 2
        n_components = 2
        pca = PCA(n_components=n_components, svd_solver='randomized',
                  whiten=True).fit(self.X_train)
        self.X_train = pca.transform(self.X_train) # 5000 * 2
        self.X_test = pca.transform(self.X_test)   # 2500 * 2


        #print(self.X_train.shape)
        #print(self.X_test.shape)

    def nu_svm_classify(self, kernel, degree=0, fig_num=0):
        '''Running nu-svm to do digit classification and return testing accuracy.
        Args:
            kernel: string, optional
                    Specifies the kernel type to be used in the algorithm
                    It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
            degree: int, optional (default=3)
                    Degree of the polynomial kernel function. Ignored by all other kernels.
            fig_num: the new figure number
        Returns: 
            test_acc: a float represent mean accuracy on test set.
        '''
        # Training
        clf = svm.NuSVC(kernel=kernel, degree=degree)
        clf.fit(self.X_train, self.y_train)

        # Drawing
        plt.figure(fig_num)
        plt.clf()

        # Randomly draw 150 training data points
        train_indices = np.arange(self.X_train.shape[0])
        np.random.shuffle(train_indices)
        sample_num = 30 * 5 
        draw_indices = train_indices[: sample_num]
        #print(draw_indices)
        plt.scatter(self.X_train[draw_indices, 0], self.X_train[draw_indices, 1], s=3, 
                    c=self.y_train[draw_indices], zorder=10, cmap=plt.cm.Paired)

        # Circle out the testing data
        #plt.scatter(self.X_test[:, 0], self.X_test[:, 1], s=80, facecolors='none', zorder=10)

        # Randommly draw 30 support vector
        np.random.shuffle(clf.support_)
        sample_support_num = 30
        draw_indices = clf.support_[: sample_support_num]
        #print(support_draw_indices)
        plt.scatter(self.X_train[draw_indices, 0], self.X_train[draw_indices, 1], s=30, 
                    c=self.y_train[draw_indices], zorder=10, cmap=plt.cm.Paired)
        '''plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                    s=30, zorder=10)'''

        # Randomlu draw 30 outliers
        outlier_indices = np.where(clf.predict(self.X_train) != self.y_train)[0]
        np.random.shuffle(outlier_indices)
        sample_outlier_num = 30
        draw_indices = outlier_indices[: sample_outlier_num]
        plt.scatter(self.X_train[draw_indices, 0], self.X_train[draw_indices, 1], s=30,
                    c=self.y_train[draw_indices], zorder=10, cmap=plt.cm.Paired, marker='+')

        
        plt.axis('tight')
        x_min = np.r_[self.X_train[:, 0], self.X_test[:, 0]].min() - 1
        x_max = np.r_[self.X_train[:, 0], self.X_test[:, 0]].max() + 1
        y_min = np.r_[self.X_train[:, 1], self.X_test[:, 1]].min() - 1
        y_max = np.r_[self.X_train[:, 1], self.X_test[:, 1]].max() + 1

        XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.contourf(XX, YY, Z, cmap=plt.cm.Accent)

        title = ''
        if kernel == 'poly':
            title = kernel + ' with degreee ' + str(degree) 
        else:
            title = kernel

        plt.title(title)

        plt.savefig(title + '.png')
        #plt.show()

        # Return testing accuracy
        return clf.score(self.X_test, self.y_test)

    def c_svm_classify(self, kernel, degree=0):
        '''Running c-svm to do digit classification and return testing accuracy.
        Args:
            kernel: string, optional
                    Specifies the kernel type to be used in the algorithm
                    It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
            degree: int, optional (default=3)
                    Degree of the polynomial kernel function. Ignored by all other kernels.
        Returns: 
            test_acc: a float represent mean accuracy on test set.
        '''
        clf = svm.SVC(kernel=kernel, degree=degree)
        clf.fit(self.X_train, self.y_train)
        return clf.score(self.X_test, self.y_test)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    digits = Digit(str(sys.argv[1]))
    for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
        if kernel == 'poly':
            for degree in xrange(2, 5):
                nu_test_acc = digits.nu_svm_classify(kernel=kernel, degree=degree, fig_num=fig_num+degree)
                c_test_acc = digits.c_svm_classify(kernel=kernel, degree=degree)
                print('Kernel: ' + kernel + ',  ' + 'degree: ' + str(degree) + ',   ' +
                      'Nu Accuracy: ' + str(nu_test_acc) + ',  C Accuracy: ' + str(c_test_acc))
        else:
            nu_test_acc = digits.nu_svm_classify(kernel=kernel, fig_num=fig_num)
            c_test_acc = digits.c_svm_classify(kernel=kernel)
            print('Kernel: ' + kernel + ',  ' +  'Nu Accuracy: ' + str(nu_test_acc) +
                  ', C Accuracy: ' + str(c_test_acc))



