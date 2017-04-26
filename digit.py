import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
import os
import sys
import csv


class Digit(object):
    """Digit class
    Represent every digit in two dimension using PCA
    Also contain method for doing classification with svm
    """

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

    def svm_classify(self, svm_type='nu', kernel='rbf', degree=0, to_draw = False):
        '''Running nu-svm to do digit classification and return testing accuracy.
        Args:
            svm_type: string, optional
                    Specifies the svm type to be used in the algorithm
            kernel: string, optional
                    Specifies the kernel type to be used in the algorithm
                    It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
            degree: int, optional (default=3)
                    Degree of the polynomial kernel function. Ignored by all other kernels.
            to_draw: int, optional (default=False)
                    Specifies whether to draw the figure.
        Returns: 
            test_acc: a float represent mean accuracy on test set.
        '''
        # Training
        if svm_type == 'nu':
            clf = svm.NuSVC(kernel=kernel, degree=degree)
        elif svm_type == 'c':
            clf = svm.SVC(kernel=kernel, degree=degree)
        else:
            print('No this type of svm')
            return 0

        clf.fit(self.X_train, self.y_train)
        
        if to_draw:
            self.draw_classifier(clf)

        # Return testing accuracy
        return clf.score(self.X_test, self.y_test)

    def draw_classifier(self, clf, fig_num=0):
        '''Plot the decision boundary, outlier, and support vectors.
        Args:
            clf: svm object, optional
                Specifies the svm for doing classification when drawing.
            fig_num: int, optional (default=0)
                Specifies the figure number.
        '''
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

        #print(type(clf).__name__)
        title = type(clf).__name__ + '_'
        if kernel == 'poly':
            title += kernel + ' with degreee ' + str(degree) 
        else:
            title += kernel

        plt.title(title)

        plt.savefig(title + '.png')
        #plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    digits = Digit(str(sys.argv[1]))
    for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
        if kernel == 'poly':
            for degree in xrange(2, 5):
                nu_test_acc = digits.svm_classify(svm_type='nu', kernel=kernel, degree=degree, to_draw=True)
                c_test_acc = digits.svm_classify(svm_type='c', kernel=kernel, degree=degree)
                print('Kernel: ' + kernel + ',  ' + 'degree: ' + str(degree) + ',   ' +
                      'Nu Accuracy: ' + str(nu_test_acc) + ',  C Accuracy: ' + str(c_test_acc))
        else:
            nu_test_acc = digits.svm_classify(svm_type='nu', kernel=kernel, to_draw=True)
            c_test_acc = digits.svm_classify(svm_type='c', kernel=kernel)
            print('Kernel: ' + kernel + ',  ' +  'Nu Accuracy: ' + str(nu_test_acc) +
                  ', C Accuracy: ' + str(c_test_acc))



