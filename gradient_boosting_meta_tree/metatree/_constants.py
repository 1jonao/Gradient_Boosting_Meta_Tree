# Code Author
# Yuta Nakahara <y.nakahara@waseda.jp>
import matplotlib.pyplot as plt
from .. import bernoulli, categorical, normal, linearregression

_CMAP = plt.get_cmap("Blues")
MODELS = {
    bernoulli,
    categorical,
    normal,
    linearregression,
    }
DISCRETE_MODELS = {
    bernoulli,
    categorical,
    }
CONTINUOUS_MODELS = {
    normal,
    linearregression,
    }
CLF_MODELS = {
    bernoulli,
    categorical,
    }
REG_MODELS = {
    normal,
    linearregression,
    }

