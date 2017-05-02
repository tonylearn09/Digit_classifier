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
        self.digit_classes = 5
        # Fix the random genertor for writing report only
        np.random.seed(42)

        # Load training data
        self.X_train = np.loadtxt(open(self.digits_dir + '/X_train.csv'), delimiter=',')
        self.y_train = np.loadtxt(open(self.digits_dir + '/T_train.csv'), dtype='int', delimiter= ',')

        # Seperate out 20% of training data for validation
        indices = np.arange(self.X_train.shape[0])
        np.random.shuffle(indices)
        val_num = int(self.X_train.shape[0] * 0.2)
        # self.X_val dim: 1000 * 784
        # self.y_val dim: 1000 * 1
        self.X_val = self.X_train[indices[:val_num], :]
        self.y_val = self.y_train[indices[:val_num]]
        print(self.X_val.shape)
        print(self.y_val.shape)

        # self.X_train dim: 4000 * 784
        # self.y_train dim: 4000 * 1
        self.X_train  = self.X_train[indices[val_num:], :]
        self.y_train = self.y_train[indices[val_num:]]
        print(self.X_train.shape)
        print(self.y_train.shape)

        # Testing data
        # self.X_test dim: 2500 * 784
        # self.y_test dim : 2500 * 1
        self.X_test = np.loadtxt(open(self.digits_dir + '/X_test.csv'), delimiter=',')
        self.y_test = np.loadtxt(open(self.digits_dir + '/T_test.csv'), dtype='int', delimiter= ',')

        # Set up data for drawing in 2D (PCA)
        self.reduce_dim()
    
    def reduce_dim(self):
        # Use PCA to reduce dim: 784 -> 2
        n_components = 2
        pca = PCA(n_components=n_components, svd_solver='randomized',
                  whiten=True).fit(self.X_train)
        self.X_train_pca = pca.transform(self.X_train) # 4000 * 2
        self.X_val_pca = pca.transform(self.X_val) # 1000 * 2
        self.X_test_pca = pca.transform(self.X_test)   # 2500 * 2

    def svm_classify(self, svm_type='nu', C=1.0, nu=0.5, kernel='rbf', degree=0, to_draw = False):
        '''Running nu-svm to do digit classification and return testing accuracy.
        Args:
            C: float, optional (default=1.0)
                Penalty parameter of error term for c-svm. Ignored by all other svms.
            nu: float, optional (default=0.5)
                Upper bound on the fraction of training errors and a lower bound of the fraction
                of support vectors for nu-svm. Ignored by all other svms.
            svm_type: string, optional
                    Specifies the svm type to be used in the algorithm
                    It must be one of 'nu' or 'C' or 'c'.
            kernel: string, optional
                    Specifies the kernel type to be used in the algorithm
                    It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
            degree: int, optional (default=3)
                    Degree of the polynomial kernel function. Ignored by all other kernels.
            to_draw: int, optional (default=False)
                    Specifies whether to draw the figure.
        Returns: 
            (val_acc, test_acc): a float represent mean accuracy on test set.
        '''
        # Training
        if svm_type == 'nu':
            clf = svm.NuSVC(nu=nu, kernel=kernel, degree=degree)
        elif svm_type == 'c' or svm_type == 'C':
            clf = svm.SVC(C=C, kernel=kernel, degree=degree)
        else:
            print('No this type of svm')
            return None 

        clf.fit(self.X_train, self.y_train)
        
        if to_draw:
            self.draw_classifier(clf)

        # Return validation accuracy and testing accuracy
        return clf.score(self.X_val, self.y_val), clf.score(self.X_test, self.y_test)

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

        x_min = np.r_[self.X_train_pca[:, 0], self.X_val_pca[:, 0]].min() - 1
        x_max = np.r_[self.X_train_pca[:, 0], self.X_val_pca[:, 0]].max() + 1
        y_min = np.r_[self.X_train_pca[:, 1], self.X_val_pca[:, 1]].min() - 1
        y_max = np.r_[self.X_train_pca[:, 1], self.X_val_pca[:, 1]].max() + 1
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        # Randomly draw 300 training data points
        #train_indices = np.arange(self.X_train_pca.shape[0])
        #np.random.shuffle(train_indices)
        #sample_num = 60 * 5 
        #draw_indices = train_indices[: sample_num]
        #print(draw_indices)
        markers = 'osDH*'
        colors = 'bgrcy'
        for i, marker, color in zip(np.arange(1, self.digit_classes+1), markers, colors):
            idx = np.where(self.y_train == i)
            plt.scatter(self.X_train_pca[idx, 0], self.X_train_pca[idx, 1],
                        marker=marker, c=color, s=3, zorder=10)
            #plt.scatter(self.X_train_pca[draw_indices, 0], self.X_train_pca[draw_indices, 1], s=3, 
                        #c=self.y_train[draw_indices], zorder=10, cmap=plt.cm.Paired)
                        #marker=self.y_train[draw_indices], zorder=10)


        # Circle out the testing data
        #plt.scatter(self.X_test[:, 0], self.X_test[:, 1], s=80, facecolors='none', zorder=10)

        # Randommly draw 30 support vector
        np.random.shuffle(clf.support_)
        sample_support_num = 50
        draw_indices = clf.support_[: sample_support_num]
        #print(support_draw_indices)
        for i, color in zip(np.arange(1, self.digit_classes+1), colors):
            idx = np.where(self.y_train[draw_indices] == i)
            plt.scatter(self.X_train_pca[idx, 0], self.X_train_pca[idx, 1], s=10,
                        c=color, zorder=10, marker='x')
            #print self.X_train_pca[idx, 0], self.X_train_pca[idx, 1]
        #plt.scatter(self.X_train_pca[draw_indices, 0], self.X_train_pca[draw_indices, 1], s=30, 
                    #c=self.y_train[draw_indices], zorder=10, cmap=plt.cm.Paired)
                    #c=self.y_train[draw_indices], zorder=10)

        # Randomlu draw 30 outliers
        outlier_indices = np.where(clf.predict(self.X_train) != self.y_train)[0]
        np.random.shuffle(outlier_indices)
        sample_outlier_num = 50
        draw_indices = outlier_indices[: sample_outlier_num]
        for i, color in zip(np.arange(1, self.digit_classes+1), colors):
            idx = np.where(self.y_train[draw_indices] == i)
            plt.scatter(self.X_train_pca[idx, 0], self.X_train_pca[idx, 1], s=10,
                        c=color, zorder=10, marker='+')
            #print self.X_train_pca[idx, 0], self.X_train_pca[idx, 1]
        #plt.scatter(self.X_train_pca[draw_indices, 0], self.X_train_pca[draw_indices, 1], s=30,
                    #c=self.y_train[draw_indices], zorder=10, cmap=plt.cm.Paired, marker='+')
                    #c=self.y_train[draw_indices], zorder=10, marker='+')

        
        plt.axis('tight')
        '''x_min = np.r_[self.X_train[:, 0], self.X_val[:, 0]].min() - 1
        x_max = np.r_[self.X_train[:, 0], self.X_val[:, 0]].max() + 1
        y_min = np.r_[self.X_train[:, 1], self.X_val[:, 1]].min() - 1
        y_max = np.r_[self.X_train[:, 1], self.X_val[:, 1]].max() + 1

        XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.contourf(XX, YY, Z, cmap=plt.cm.Accent)'''

        #print(type(clf).__name__)
        title = type(clf).__name__ + '_'
        if kernel == 'poly':
            title += kernel + ' with degreee ' + str(degree) 
        else:
            title += kernel

        plt.title(title)

        plt.savefig(title + '.pdf')
        #plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    digits = Digit(str(sys.argv[1]))
    for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
        if kernel == 'poly':
            for degree in xrange(2, 5):
                nu_val_acc, nu_test_acc = digits.svm_classify(svm_type='nu', nu=0.5,
                                                              kernel=kernel, degree=degree, to_draw=True)
                c_val_acc, c_test_acc = digits.svm_classify(svm_type='c', C=4.0,
                                                            kernel=kernel, degree=degree, to_draw=False)
                print('Kernel: ' + kernel + ',  ' + 'degree: ' + str(degree) + ',   ' +
                      'Nu val Accuracy: ' + str(nu_val_acc) + ',  C val Accuracy: ' + str(c_val_acc))
                print('Nu test acc: ' + str(nu_test_acc) + ',  C test acc: ' + str(c_test_acc))
        else:
            nu_val_acc, nu_test_acc = digits.svm_classify(svm_type='nu', nu=0.5, kernel=kernel, to_draw=True)
            c_val_acc, c_test_acc = digits.svm_classify(svm_type='c', C=1.0, kernel=kernel, to_draw=False)
            print('Kernel: ' + kernel + ',  ' +  'Nu val Accuracy: ' + str(nu_val_acc) +
                  ', C val Accuracy: ' + str(c_val_acc))
            print('Nu test acc: ' + str(nu_test_acc) + ',  C test acc: ' + str(c_test_acc))



