from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,\
GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from xgboost.sklearn import XGBClassifier
# from imblearn.ensemble import BalancedRandomForestClassifier
# from imblearn.ensemble import EasyEnsembleClassifier
# from imblearn.ensemble import RUSBoostClassifier
# from lightgbm.sklearn import LGBMClassifier
# from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
# from sklearn.linear_model import ARDRegression

from copy import deepcopy
import pandas as pd
import numpy as np

from .primitive import Primitive

class RandomForestClassifierPrim(Primitive):
    def __init__(self):
        super(RandomForestClassifierPrim, self).__init__(name='RandomForestClassifier')
        self.id = 1
        # self.name = 'RandomForestClassifier'
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "A random forest classifier. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default)."
        self.accept_type = 'c'
        self.model = RandomForestClassifier(random_state=0, n_jobs=5)
        code = 'rfc = RandomForestClassifier(random_state=0, n_jobs=5)\n'
        code += 'rfc.fit(train_x, train_y)\n'
        code += 'pred_y = rfc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = RandomForestClassifier(random_state=0, n_jobs=5)
        model.fit(train_x, train_y)
        pred_y =  model.predict(test_x)
        return pred_y

class AdaBoostClassifierPrim(Primitive):
    def __init__(self):
        super(AdaBoostClassifierPrim, self).__init__(name='AdaBoostClassifier')
        self.id = 2
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "An AdaBoost classifier. An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. This class implements the algorithm known as AdaBoost-SAMME."
        self.accept_type = 'c'
        code = 'abc = AdaBoostClassifier(random_state=0)\n'
        code += 'abc.fit(train_x, train_y)\n'
        code += 'pred_y = abc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True
    def transform(self, train_x, train_y, test_x):
        model = AdaBoostClassifier(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class BaggingClassifierPrim(Primitive):
    def __init__(self):
        super(BaggingClassifierPrim, self).__init__(name='BaggingClassifier')
        self.hyperparams = []
        self.id = 3
        self.type = 'Classifier'
        self.description = "A Bagging classifier. A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it. This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting. If samples are drawn with replacement, then the method is known as Bagging. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches."
        self.accept_type = 'c'
        code = 'bc = BaggingClassifier(random_state=0, n_jobs=5)\n'
        code += 'bc.fit(train_x, train_y)\n'
        code += 'pred_y = bc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = BaggingClassifier(random_state=0, n_jobs=5)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y



class BernoulliNBClassifierPrim(Primitive):
    def __init__(self):
        super(BernoulliNBClassifierPrim, self).__init__(name='BernoulliNBClassifier')
        self.hyperparams = []
        self.id = 4
        self.type = 'Classifier'
        self.description = "Naive Bayes classifier for multivariate Bernoulli models. Like MultinomialNB, this classifier is suitable for discrete data. The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features."
        self.accept_type = 'c'
        code = 'bnb = BernoulliNB()\n'
        code += 'bnb.fit(train_x, train_y)\n'
        code += 'pred_y = bnb.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = BernoulliNB()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

# class ComplementNBClassifierPrim(Primitive):
#     def __init__(self):
#         super(ComplementNBClassifierPrim, self).__init__(name='ComplementNBClassifier')
#         self.hyperparams = []
#         self.id = 5
#         self.type = 'Classifier'
#         self.description = "The Complement Naive Bayes classifier described in Rennie et al. (2003). The Complement Naive Bayes classifier was designed to correct the “severe assumptions” made by the standard Multinomial Naive Bayes classifier. It is particularly suited for imbalanced data sets."
#         self.accept_type = 'd'
#         code = 'cnb = ComplementNB()\n'
#         code += 'cnb.fit(train_x, train_y)\n'
#         code += 'pred_y = cnb.predict(test_x)\n'
#         self.code = code

#     def can_accept(self, data):
#         return self.can_accept_d(data, 'Classification')

#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True

#     def transform(self, train_x, train_y, test_x):
#         model = ComplementNB()
#         model.fit(train_x, train_y)
#         pred_y = model.predict(test_x)
#         return pred_y

class DecisionTreeClassifierPrim(Primitive):
    def __init__(self):
        super(DecisionTreeClassifierPrim, self).__init__(name='DecisionTreeClassifier')
        self.hyperparams = []
        self.id = 2
        self.type = 'Classifier'
        self.description = "A decision tree classifier."
        self.accept_type = 'c'
        code = 'dtc = DecisionTreeClassifier(random_state=0)\n'
        code += 'dtc.fit(train_x, train_y)\n'
        code += 'pred_y = dtc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True
    
    def transform(self, train_x, train_y, test_x):
        model = DecisionTreeClassifier(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class ExtraTreesClassifierPrim(Primitive):
    def __init__(self):
        super(ExtraTreesClassifierPrim, self).__init__(name='ExtraTreesClassifier')
        self.hyperparams = []
        self.id = 6
        self.type = 'Classifier'
        self.description = "An extra-trees classifier. This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting."
        self.accept_type = 'c'
        code = 'etc = ExtraTreesClassifier(random_state=0, n_jobs=5)\n'
        code += 'etc.fit(train_x, train_y)\n'
        code += 'pred_y = etc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = ExtraTreesClassifier(random_state=0, n_jobs=5)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class GaussianNBClassifierPrim(Primitive):
    def __init__(self):
        super(GaussianNBClassifierPrim, self).__init__(name='GaussianNBClassifier')
        self.hyperparams = []
        self.id = 7
        self.type = 'Classifier'
        self.description = "Gaussian Naive Bayes (GaussianNB). Can perform online updates to model parameters via partial_fit method."
        self.accept_type = 'c'
        code = 'gnb = GaussianNB()\n'
        code += 'gnb.fit(train_x, train_y)\n'
        code += 'pred_y = gnb.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = GaussianNB()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class GaussianProcessClassifierPrim(Primitive):
    def __init__(self):
        super(GaussianProcessClassifierPrim, self).__init__(name='GaussianProcessClassifierPrim')
        self.hyperparams = []
        self.id = 8
        self.type = 'Classifier'
        self.description = "Gaussian process classification (GPC) based on Laplace approximation. The implementation is based on Algorithm 3.1, 3.2, and 5.1 of Gaussian Processes for Machine Learning (GPML) by Rasmussen and Williams. Internally, the Laplace approximation is used for approximating the non-Gaussian posterior by a Gaussian. Currently, the implementation is restricted to using the logistic link function. For multi-class classification, several binary one-versus rest classifiers are fitted. Note that this class thus does not implement a true multi-class Laplace approximation."
        self.accept_type = 'c'
        code = 'gpc = GaussianProcessClassifier()\n'
        code += 'gpc.fit(train_x, train_y)\n'
        code += 'pred_y = gpc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = GaussianProcessClassifier()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class GradientBoostingClassifierPrim(Primitive):
    def __init__(self):
        super(GradientBoostingClassifierPrim, self).__init__(name='GradientBoostingClassifier')
        self.hyperparams = []
        self.id = 9
        self.type = 'Classifier'
        self.description = "Gradient Boosting for classification. GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced."
        self.accept_type = 'c'
        code = 'gbc = GradientBoostingClassifier(random_state=0)\n'
        code += 'gbc.fit(train_x, train_y)\n'
        code += 'pred_y = gbc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = GradientBoostingClassifier(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class KNeighborsClassifierPrim(Primitive):
    def __init__(self):
        super(KNeighborsClassifierPrim, self).__init__(name='KNeighborsClassifier')
        self.hyperparams = []
        self.id = 3
        self.type = 'Classifier'
        self.description = "Classifier implementing the k-nearest neighbors vote."
        self.accept_type = 'c'
        code = 'knc = KNeighborsClassifier(n_jobs=5)\n'
        code += 'knc.fit(train_x, train_y)\n'
        code += 'pred_y = knc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = KNeighborsClassifier()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class LinearDiscriminantAnalysisPrim(Primitive):
    def __init__(self):
        super(LinearDiscriminantAnalysisPrim, self).__init__(name='LinearDiscriminantAnalysisPrim')
        self.hyperparams = []
        self.id = 11
        self.type = 'Classifier'
        self.description = "Linear Discriminant Analysis. A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule. The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix. The fitted model can also be used to reduce the dimensionality of the input by projecting it to the most discriminative directions."
        self.accept_type = 'c'
        code = 'lda = LinearDiscriminantAnalysis()\n'
        code += 'lda.fit(train_x, train_y)\n'
        code += 'pred_y = lda.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = LinearDiscriminantAnalysis()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class LinearSVCPrim(Primitive):
    def __init__(self):
        super(LinearSVCPrim, self).__init__(name='LinearSVC')
        self.hyperparams = []
        self.id = 12
        self.type = 'Classifier'
        self.description = "Linear Support Vector Classification. Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples. This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme."
        self.accept_type = 'c'
        code = 'lsvc = LinearSVC(random_state=0)\n'
        code += 'lsvc.fit(train_x, train_y)\n'
        code += 'pred_y = lsvc.predict(test_x)\n'
        self.code = code
    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = LinearSVC(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class LogisticRegressionPrim(Primitive):
    def __init__(self):
        super(LogisticRegressionPrim, self).__init__(name='LogisticRegression')
        self.hyperparams = []
        self.id = 4
        self.type = 'Classifier'
        self.description = "Logistic Regression (aka logit, MaxEnt) classifier. In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, and uses the cross- entropy loss if the ‘multi_class’ option is set to ‘multinomial’. (Currently the ‘multinomial’ option is supported only by the ‘lbfgs’, ‘sag’ and ‘newton-cg’ solvers.) This class implements regularized logistic regression using the ‘liblinear’ library, ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers. It can handle both dense and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit floats for optimal performance; any other input format will be converted (and copied). The ‘newton-cg’, ‘sag’, and ‘lbfgs’ solvers support only L2 regularization with primal formulation. The ‘liblinear’ solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty."
        self.accept_type = 'c'
        code = 'lr = LogisticRegression(random_state=0, n_jobs=5, multi_class=\'auto\')\n'
        code += 'lr.fit(train_x, train_y)\n'
        code += 'pred_y = lr.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = LogisticRegression(solver='liblinear',random_state=0, n_jobs=5)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


# class LogisticRegressionCVPrim(Primitive):
#     def __init__(self):
#         super(LogisticRegressionCVPrim, self).__init__(name='LogisticRegressionCV')
#         self.hyperparams = []
#         self.id = 14
#         self.type = 'Classifier'
#         self.description = "Logistic Regression CV (aka logit, MaxEnt) classifier. See glossary entry for cross-validation estimator. This class implements logistic regression using liblinear, newton-cg, sag of lbfgs optimizer. The newton-cg, sag and lbfgs solvers support only L2 regularization with primal formulation. The liblinear solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty. For the grid of Cs values (that are set by default to be ten values in a logarithmic scale between 1e-4 and 1e4), the best hyperparameter is selected by the cross-validator StratifiedKFold, but it can be changed using the cv parameter. In the case of newton-cg and lbfgs solvers, we warm start along the path i.e guess the initial coefficients of the present fit to be the coefficients got after convergence in the previous fit, so it is supposed to be faster for high-dimensional dense data. For a multiclass problem, the hyperparameters for each class are computed using the best scores got by doing a one-vs-rest in parallel across all folds and classes. Hence this is not the true multinomial loss."
#         self.accept_type = 'c'
#         code = 'lrcv = LogisticRegressionCV(random_state=0, n_jobs=5, multi_class=\'auto\')\n'
#         code += 'lrcv.fit(train_x, train_y)\n'
#         code += 'pred_y = lrcv.predict(test_x)\n'
#         self.code = code
#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Classification')

#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True

#     def transform(self, train_x, train_y, test_x):
#         model = LogisticRegressionCV(random_state=0, n_jobs=5, multi_class='auto')
#         model.fit(train_x, train_y)
#         pred_y = model.predict(test_x)
#         return pred_y

# class MultinomialNBPrim(Primitive):
#     def __init__(self):
#         super(MultinomialNBPrim, self).__init__(name='MultinomialNB')
#         self.hyperparams = []
#         self.id = 16
#         self.type = 'Classifier'
#         self.description = "Naive Bayes classifier for multinomial models. The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work."
#         self.accept_type = 'd'
#         code = 'mnb = MultinomialNB()\n'
#         code += 'mnb.fit(train_x, train_y)\n'
#         code += 'pred_y = mnb.predict(test_x)\n'
#         self.code = code

#     def can_accept(self, data):
#         return self.can_accept_d(data, 'Classification')

#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True

#     def fit(self, data):
#         data = handle_data(data)
#         self.model.fit(data, data['Y'])

#     def transform(self, train_x, train_y, test_x):
#         model = MultinomialNB()
#         model.fit(train_x, train_y)
#         pred_y = model.predict(test_x)
#         return pred_y

class NearestCentroidPrim(Primitive):
    def __init__(self):
        super(NearestCentroidPrim, self).__init__(name='NearestCentroid')
        self.hyperparams = []
        self.id = 14
        self.type = 'Classifier'
        self.description = "Nearest centroid classifier. Each class is represented by its centroid, with test samples classified to the class with the nearest centroid."
        self.accept_type = 'c'
        code = 'nc = NearestCentroid()\n'
        code += 'nc.fit(train_x, train_y)\n'
        code += 'pred_y = nc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = NearestCentroid()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class PassiveAggressiveClassifierPrim(Primitive):
    def __init__(self):
        super(PassiveAggressiveClassifierPrim, self).__init__(name='PassiveAggressiveClassifier')
        self.hyperparams = []
        self.id = 15
        self.type = 'Classifier'
        self.description = "Passive Aggressive Classifier"
        self.hyperparams_run = {'default': True}
        self.accept_type = 'c'
        code = 'pac = PassiveAggressiveClassifier(random_state=0, n_jobs=5)\n'
        code += 'pac.fit(train_x, train_y)\n'
        code += 'pred_y = pac.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = PassiveAggressiveClassifier(random_state=0, n_jobs=5)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

# class QuadraticDiscriminantAnalysisPrim(Primitive):
#     def __init__(self):
#         super(QuadraticDiscriminantAnalysisPrim, self).__init__(name='QuadraticDiscriminantAnalysis')
#         self.hyperparams = []
#         self.id = 17
#         self.type = 'Classifier'
#         self.description = "Quadratic Discriminant Analysis. A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule. The model fits a Gaussian density to each class."
#         self.accept_type = 'c'
#         code = 'qda = QuadraticDiscriminantAnalysis()\n'
#         code += 'qda.fit(train_x, train_y)\n'
#         code += 'pred_y = qda.predict(test_x)\n'
#         self.code = code

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Classification')

#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True

#     def transform(self, train_x, train_y, test_x):
#         model = QuadraticDiscriminantAnalysis()
#         model.fit(train_x, train_y)
#         pred_y = model.predict(test_x)
#         return pred_y

class RidgeClassifierPrim(Primitive):
    def __init__(self):
        super(RidgeClassifierPrim, self).__init__(name='RidgeClassifier')
        self.hyperparams = []
        self.id = 16
        self.type = 'Classifier'
        self.description = "Classifier using Ridge regression."
        self.accept_type = 'c'
        code = 'rc = RidgeClassifier(random_state=0)\n'
        code += 'rc.fit(train_x, train_y)\n'
        code += 'pred_y = rc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = RidgeClassifier(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class RidgeClassifierCVPrim(Primitive):
    def __init__(self):
        super(RidgeClassifierCVPrim, self).__init__(name='RidgeClassifierCV')
        self.hyperparams = []
        self.id = 17
        self.type = 'Classifier'
        self.description = "Ridge classifier with built-in cross-validation. By default, it performs Generalized Cross-Validation, which is a form of efficient Leave-One-Out cross-validation."
        self.accept_type = 'c'
        code = 'rccv = RidgeClassifierCV()\n'
        code += 'rccv.fit(train_x, train_y)\n'
        code += 'pred_y = rccv.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = RidgeClassifierCV()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class SGDClassifierPrim(Primitive):
    def __init__(self):
        super(SGDClassifierPrim, self).__init__(name='SGDClassifier')
        self.id = 18
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Linear classifiers (SVM, logistic regression, a.o.) with SGD training. This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning, see the partial_fit method. For best results using the default learning rate schedule, the data should have zero mean and unit variance. This implementation works with data represented as dense or sparse arrays of floating point values for the features. The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM). The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection."
        self.accept_type = 'c'
        code = 'sgdc = SGDClassifier(random_state=0, n_jobs=5, loss=\'log\')\n'
        code += 'sgdc.fit(train_x, train_y)\n'
        code += 'pred_y = sgdc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = SGDClassifier(random_state=0, n_jobs=5, loss='log')
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class SVCPrim(Primitive):
    def __init__(self):
        super(SVCPrim, self).__init__(name='SVC')
        self.hyperparams = []
        self.id = 5
        self.type = 'Classifier'
        self.description = "C-Support Vector Classification. The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples. The multiclass support is handled according to a one-vs-one scheme. For details on the precise mathematical formulation of the provided kernel functions and how gamma, coef0 and degree affect each other, see the corresponding section in the narrative documentation: Kernel functions."
        self.accept_type = 'c'
        code = 'svc = SVC(random_state=0, probability=True)\n'
        code += 'svc.fit(train_x, train_y)\n'
        code += 'pred_y = svc.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = SVC(random_state=0, probability=True)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

# class XGBClassifierPrim(Primitive):
#     def __init__(self):
#         super(XGBClassifierPrim, self).__init__(name='XGBClassifier')
#         self.hyperparams = []
#         self.id = 24
#         self.type = 'Classifier'
#         self.description = "XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples."
#         self.model = XGBClassifier(random_state=0, n_jobs=5)
#         self.accept_type = 'xgb'
#         code = 'xgbc = XGBClassifier(random_state=0, n_jobs=5)\n'
#         code += 'xgbc.fit(train_x, train_y)\n'
#         code += 'pred_y = xgbc.predict(test_x)\n'
#         self.code = code

#     def can_accept(self, data):
#         # data = handle_data(data)
#         if data.empty:
#             return False
#         cols = data
#         num_cols = data._get_numeric_data().columns
#         cat_cols = list(set(cols) - set(num_cols))
#         if not len(cat_cols) == 0:
#             return False
#         return True

#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True

#     def transform(self, train_x, train_y, test_x):
#         model = XGBClassifier(random_state=0, n_jobs=5)
#         model.fit(train_x, train_y)
#         pred_y = model.predict(test_x)
#         return pred_y

class BalancedRandomForestClassifierPrim(Primitive):
    def __init__(self):
        super(BalancedRandomForestClassifierPrim, self).__init__(name='BalancedRandomForestClassifier')
        self.hyperparams = []
        self.id = 20
        self.type = 'Classifier'
        self.description = "A balanced random forest classifier. A balanced random forest randomly under-samples each boostrap sample to balance it."
        self.accept_type = 'c'
        code = 'brf = BalancedRandomForestClassifier(random_state=0, n_jobs=5)\n'
        code += 'brf.fit(train_x, train_y)\n'
        code += 'pred_y = brf.predict(test_x)\n'
        self.code = code
    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = BalancedRandomForestClassifier(random_state=0, n_jobs=5)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class EasyEnsembleClassifierPrim(Primitive):
    def __init__(self):
        super(EasyEnsembleClassifierPrim, self).__init__(name='EasyEnsembleClassifier')
        self.hyperparams = []
        self.id = 21
        self.type = 'Classifier'
        self.description = "Bag of balanced boosted learners also known as EasyEnsemble. This algorithm is known as EasyEnsemble [1]. The classifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling."
        self.accept_type = 'c'
        code = 'eec = EasyEnsembleClassifier(random_state=0, n_jobs=5)\n'
        code += 'eec.fit(train_x, train_y)\n'
        code += 'pred_y = eec.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = EasyEnsembleClassifier(random_state=0, n_jobs=5)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class RUSBoostClassifierPrim(Primitive):
    def __init__(self):
        super(RUSBoostClassifierPrim, self).__init__(name='RUSBoostClassifier')
        self.hyperparams = []
        self.id = 22
        self.type = 'Classifier'
        self.description = "Random under-sampling integrating in the learning of an AdaBoost classifier. During learning, the problem of class balancing is alleviated by random under-sampling the sample at each iteration of the boosting algorithm."
        self.accept_type = 'c'
        code = 'rc = RUSBoostClassifier(random_state=0)\n'
        code += 'rc.fit(train_x, train_y)\n'
        code += 'pred_y = rc.predict(test_x)\n'
        self.code = code
    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = RUSBoostClassifier(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

# class LGBMClassifierPrim(Primitive):
#     def __init__(self):
#         super(LGBMClassifierPrim, self).__init__(name='LGBMClassifier')
#         self.hyperparams = []
#         self.id = 28
#         self.type = 'Classifier'
#         self.accept_type = 'c'
#         self.description = "LightGBM is a gradient boosting framework that uses tree based learning algorithms."
#         code = 'lc = LGBMClassifier(random_state=0, n_jobs=5)\n'
#         code += 'lc.fit(train_x, train_y)\n'
#         code += 'pred_y = lc.predict(test_x)\n'
#         self.code = code
        

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Classification')

#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True

#     def transform(self, train_x, train_y, test_x):
#         model = LGBMClassifier(random_state=0, n_jobs=5)
#         model.fit(train_x, train_y)
#         pred_y = model.predict(test_x)
#         return pred_y

class ARDRegressionPrim(Primitive):
    def __init__(self):
        super(ARDRegressionPrim, self).__init__(name='ARDRegression')
        self.hyperparams = []
        self.id = 23
        self.type = 'Regressor'
        self.description = "Bayesian ARD regression. Fit the weights of a regression model, using an ARD prior. The weights of the regression model are assumed to be in Gaussian distributions. Also estimate the parameters lambda (precisions of the distributions of the weights) and alpha (precision of the distribution of the noise). The estimation is done by an iterative procedures (Evidence Maximization)"
        self.accept_type = 'c_r'
        code = 'ard = ARDRegression()\n'
        code += 'ard.fit(train_x, train_y)\n'
        code += 'pred_y = ard.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = ARDRegression()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class AdaBoostRegressorPrim(Primitive):
    def __init__(self):
        super(AdaBoostRegressorPrim, self).__init__(name='AdaBoostRegressor')
        self.hyperparams = []
        self.id = 24
        self.type = 'Regressor'
        self.description = "An AdaBoost regressor. An AdaBoost [1] regressor is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction. As such, subsequent regressors focus more on difficult cases."
        self.accept_type = 'c_r'
        code = 'adb = AdaBoostRegressor(random_state=0)\n'
        code += 'adb.fit(train_x, train_y)\n'
        code += 'pred_y = adb.predict(test_x)\n'
        self.code = code
    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def transform(self, train_x, train_y, test_x):
        model = AdaBoostRegressor(random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y

class BaggingRegressorPrim(Primitive):
    def __init__(self):
        super(BaggingRegressorPrim, self).__init__(name='BaggingRegressor')
        self.hyperparams = []
        self.id = 25
        self.type = 'Regressor'
        self.description = "A Bagging regressor. A Bagging regressor is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it. This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [1]. If samples are drawn with replacement, then the method is known as Bagging [2]. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [3]. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [4]."
        self.accept_type = 'c_r'
        code = 'br = BaggingRegressor(random_state=0, n_jobs=5)\n'
        code += 'br.fit(train_x, train_y)\n'
        code += 'pred_y = br.predict(test_x)\n'
        self.code = code

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True
    
    def transform(self, train_x, train_y, test_x):
        model = BaggingRegressor(random_state=0, n_jobs=5)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y