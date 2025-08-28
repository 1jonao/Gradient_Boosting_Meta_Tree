import numpy as np

# Loss_{hoge}_{hoge} class means it's not defined in bayesian context, 
# i.e. it doesn't consider prior distribution knowledge or assume non-informative prior.

# BR_{hoge}_{hoge}, on the other hand take in to account (conjugate) prior distribution  
# and derived from minimization of bayes risk function.

# {hoge}_Discrete_{hoge} means it's specified for discrete (category) objective 
# {hoge}_Continuous_{hoge} means it's specified for continuous (numerical) objective

def Loss_Continuous_Squared_Error(y_pred, y_true):
    grad = -2*(y_pred - y_true)
    hess = 0*y_true + 2
    return grad, hess

class ClassificationCriterion():
    def __init__(
            self,
            y_vecs: np.ndarray,
            y_hat_vecs: np.ndarray,
            weights_vecs: np.ndarray,
            n_instances: int,
            n_classes: int,
            depth=0,
            h_g=1.0,
            ):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        y_vecs : numpy.array
            target objective variables consists (n_instances*n_classes) 1-hot vectors. 
        y_vecs : numpy.array
            predicted objective variables consists (n_instances*n_classes) vectors. It can either be 1-hot or probability vector.  
        n_instances : int
            The number of instances on Y, the dimensionality of the prediction
        n_classes : int
            The number of unique classes in Y.
        """
        self.y_vecs = y_vecs
        self.y_hat_vecs = y_hat_vecs
        self.weights_vecs = weights_vecs
        self.n_instances = n_instances
        self.n_classes = n_classes
        self.depth = depth
        self.h_g = h_g

class Loss_Discrete_ErrorRate(ClassificationCriterion):
    r"""L1 norm loss criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

    count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The error rate loss is then defined as

    error_rate = -\sum_{n=1}^{N}\sum_{k=0}^{K-1} y_{n,k}|y_{n,k} - q_{n,k}|
    """
    def calc(self):
        return np.sum((self.y_vecs - self.y_hat_vecs)[self.y_vecs == 1] * self.weights_vecs)
    

def _squared_error(y_vec:np.ndarray,y_hat_vec:np.ndarray,num:int):
    return np.sum((y_vec - y_hat_vec)**2) / num

def _misclassification_rate(y_vec:np.ndarray,y_hat_vec:np.ndarray,sample_size:int):
    return np.count_nonzero(y_vec != y_hat_vec)/sample_size

def _one_hot_encoding(y_vec:np.ndarray,num_classes:int):
    return np.eye(num_classes)[y_vec]

def _squared_error_clf(y_vecs:np.ndarray,y_hat_vecs:np.ndarray,num:int):
    return np.sum((y_vecs - y_hat_vecs)**2) / num / 2

def _abs_error_clf(y_vecs:np.ndarray,y_hat_vecs:np.ndarray,num:int):
    return np.sum(np.abs(y_vecs - y_hat_vecs)) / num

def _cross_entropy(y_vecs:np.ndarray,y_hat_vecs:np.ndarray,num:int):
    return - np.sum(y_vecs * np.log(y_hat_vecs)) / num 

def _gini(y_vec:np.ndarray,num_classes:int):
    count_list = np.bincount(y_vec,minlength=num_classes)
    total_count = np.sum(count_list)
    return 1 - np.sum((count_list/total_count)**2) 

if __name__ == '__main__':
    print('debug')
    from bayesml import categorical

    # gm = categorical.GenModel(3)
    # gm.set_params()
    # y_vecs = gm.gen_sample(3)
    # print(y_vecs)
    # y_hat_vecs = np.tile(np.array([0.6, 0.3, 0.1]), (3,1))
    # weights_vecs = np.full(3, 0.1)
    # er = Loss_Discrete_ErrorRate(y_vecs,y_hat_vecs,weights_vecs,10,3)
    # print(er.calc())

    NUM_CLASSES = 2
    NUM_DATA = 5
    a = np.random.randint(0,NUM_CLASSES,NUM_DATA)
    b = np.random.randint(0,NUM_CLASSES,NUM_DATA)
    print(a)
    print(b)
    print(_squared_error(a,b,NUM_DATA))
    print(_misclassification_rate(a,b,NUM_DATA))
    print(_gini(a,NUM_CLASSES))

    print("-----------------------")
    c = _one_hot_encoding(a,NUM_CLASSES)
    d = _one_hot_encoding(b,NUM_CLASSES)
    print(c)
    print(d)
    print(_squared_error_clf(c,d,NUM_DATA))
    print(_abs_error_clf(c,d,NUM_DATA))

    print("========================")
    y_hat_vecs = np.array([
        [0.1, 0.3, 0.6],
        [0.8, 0.1, 0.1],
        [0.2, 0.5, 0.3]
    ])
    y_vecs = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    print(_squared_error_clf(y_vecs,y_hat_vecs,3))
    print(_abs_error_clf(y_vecs,y_hat_vecs,3))
    print(_cross_entropy(y_vecs,y_hat_vecs,3))
    