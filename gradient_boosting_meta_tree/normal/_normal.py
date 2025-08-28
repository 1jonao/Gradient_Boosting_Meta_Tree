# Code Author
# Noboru Namegaya <n.noboru20180403@toki.waseda.jp>
# Koshi Shimada <shimada.koshi.re@gmail.com>
# Naoki Ichijo <1jonao@fuji.waseda.jp> # the known_precision version
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Document Author
# Noboru Namegaya <n.noboru20180403@toki.waseda.jp>
# Koshi Shimada <shimada.koshi.re@gmail.com>
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
import warnings
import numpy as np
from scipy.stats import norm as ss_norm
from scipy.stats import gamma as ss_gamma 
from scipy.stats import t as ss_t
from scipy.special import gammaln
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution

    Parameters
    ----------
    mu : float, optional
        a real number, by default 0.0
    tau : float, optional
        a positive real number, by default 1.0
    h_m : float, optional
        a real number, by default 0.0
    h_kappa : float, optional
        a positive real number, by default 1.0
    h_alpha : float, optional
        a positive real number, by default 1.0
    h_beta : float, optional
        a positive real number, by default 1.0
    h_tau : float, optional
        a positive real number, by default 1.0
    known_precision : bool, optional
        If ``True``, the precision parameter of the prior distribution is set to be known, by default False
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(self,mu=0.0,tau=1.0,h_m=0.0,h_kappa=1.0,h_alpha=1.0,h_beta=1.0,h_tau=1.0,known_precision=False, seed=None):
        self.rng = np.random.default_rng(seed)

        # params
        self.mu = None
        self.tau = None
        self.set_params(mu,tau)

        self.known_precision = known_precision

        # h_params
        self.h_m = None
        self.h_tau = None
        self.h_kappa = None
        self.h_alpha = None
        self.h_beta = None
        if self.known_precision:
            self.set_h_params(h_m=h_m, h_tau=h_tau)
        else:
            self.set_h_params(h_m=h_m,h_kappa=h_kappa,h_alpha=h_alpha,h_beta=h_beta)


    def get_constants(self):
        """Get constants of GenModel.

        This model does not have any constants. 
        Therefore, this function returns an emtpy dict ``{}``.
        
        Returns
        -------
        constants : an empty dict
        """
        return {}

    def set_h_params(self,h_m=None,h_kappa=None,h_alpha=None,h_beta=None, h_tau=None, known_precision=None, tau=None):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_m : float, optional
            a real number, by default None
        h_kappa : float, optional
            a positive real number, by default None
        h_alpha : float, optional
            a positive real number, by default None
        h_beta : float, optional
            a positive real number, by default None
        h_tau : float, optional
            a positive real number, by default None
        """
        if h_m is not None:
            self.h_m = _check.float_(h_m,'h_m',ParameterFormatError)
        if h_kappa is not None:
            self.h_kappa = _check.pos_float(h_kappa,'h_kappa',ParameterFormatError)
        if h_alpha is not None:
            self.h_alpha = _check.pos_float(h_alpha,'h_alpha',ParameterFormatError)
        if h_beta is not None:
            self.h_beta = _check.pos_float(h_beta,'h_beta',ParameterFormatError)
        if h_tau is not None:
            self.h_tau = _check.pos_float(h_tau,'h_tau',ParameterFormatError)
        if known_precision is not None:
            self.known_precision = _check.boolean(known_precision,'known_precision',ParameterFormatError)
        if (self.known_precision) and (tau is not None):
            self.tau = _check.pos_float(tau,'tau',ParameterFormatError)
        return self

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float}
            * ``"h_m"`` : The value of ``self.h_m``
            * ``"h_kappa"`` : The value of ``self.h_kappa``
            * ``"h_alpha"`` : The value of ``self.h_alpha``
            * ``"h_beta"`` : The value of ``self.h_beta``
            * ``"h_tau"`` : The value of ``self.h_tau``
        """
        return {"h_m":self.h_m, "h_kappa":self.h_kappa, "h_alpha":self.h_alpha, "h_beta":self.h_beta, "h_tau":self.h_tau, "known_precision": self.known_precision}
    
    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.mu`` and ``self.tau``.
        """
        if self.known_precision:
            self.mu = self.rng.normal(loc=self.h_m,scale=1.0/np.sqrt(self.h_tau))
        else:
            self.tau = self.rng.gamma(shape=self.h_alpha,scale=1.0/self.h_beta)
            self.mu = self.rng.normal(loc=self.h_m,scale=1.0/np.sqrt(self.tau * self.h_kappa))
        return self

    def set_params(self,mu=None,tau=None):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        mu : float, optional
            a real number :math:`mu \in \mathbb{R}`, by default None.
        tau : float, optional
            a positive real number, by default None.
        """
        if mu is not None:
            self.mu = _check.float_(mu,'mu',ParameterFormatError)
        if tau is not None:
            self.tau = _check.pos_float(tau,'tau',ParameterFormatError)
        return self

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:float}
            * ``"mu"`` : The value of ``self.mu``
            * ``"tau"`` : The value of ``self.tau``
        """
        return {"mu":self.mu, "tau":self.tau}

    def gen_sample(self,sample_size):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        x : numpy ndarray
            1 dimensional array whose size is ``sammple_size``.
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        return self.rng.normal(loc=self.mu,scale=1.0/np.sqrt(self.tau),size=sample_size)
        
    def save_sample(self,filename,sample_size):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"x\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int
            A positive integer
        
        See Also
        --------
        numpy.savez_compressed
        """
        np.savez_compressed(filename,x=self.gen_sample(sample_size))

    def visualize_model(self,sample_size=1000,hist_bins=10):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 1000
        hist_bins : float, optional
            A positive float, by default 10

        Examples
        --------
        >>> from bayesml import normal
        >>> model = normal.GenModel()
        >>> model.visualize_model()

        .. image:: ./images/normal_example.png
        """
        
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        _check.pos_int(hist_bins, 'hist_bins', DataFormatError)
        fig, ax = plt.subplots()
        sample = self.gen_sample(sample_size)

        plt.title(f"PDF and normalized histogram")
        ax.hist(sample,density=True,label=f"normalized hist n={sample_size}",bins=hist_bins)

        x = np.linspace(sample.min()-(sample.max()-sample.min())*0.25,
                        sample.max()+(sample.max()-sample.min())*0.25,
                        100)
        y = ss_norm.pdf(x,self.mu,1.0/np.sqrt(self.tau))
        plt.plot(x, y, label=f"Normal PDF mu={self.mu}, tau={self.tau}")
        ax.set_xlabel("Realization")
        ax.set_ylabel("Probability or frequency")
        
        plt.legend()
        plt.show()

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    h0_m : float, optional
        a real number, by default 0.0
    h0_kappa : float, optional
        a positive real number, by default 1.0
    h0_alpha : float, optional
        a positive real number, by default 1.0
    h0_beta : float, optional
        a positive real number, by default 1.0
    h0_tau : float, optional
        a positive real number, the value only used when known_precision is set to True, by default 1.0
    h0_tau_x : float, optional
        a positive real number, the value only used when known_precision is set to True, by default 1.0
    known_precision : bool, optional
        If ``True``, the precision parameter of the prior distribution is set to be known, by default False
    
    Attributes
    ----------
    hn_m : float
        a real number
    hn_kappa : float
        a positive real number
    hn_alpha : float
        a positive real number
    hn_beta : float
        a positive real number
    p_mu : float
        a real number
    p_lambda : float
        a positive real number
    p_nu : float
        a positive real number
    """

    def __init__(self,h0_m=0.0,h0_kappa=1.0,h0_alpha=1.0,h0_beta=1.0,h0_tau=1.0,h0_tau_x=1.0,known_precision=False):
        self.known_precision = known_precision
        # h0_params
        self.h0_m = None
        self.h0_tau = None
        self.h0_tau_x = None
        self.h0_kappa = None
        self.h0_alpha = None
        self.h0_beta = None
        if self.known_precision:
            self.set_h0_params(h0_m=h0_m, h0_tau=h0_tau, h0_tau_x=h0_tau_x)
        else:
            self.set_h0_params(h0_m=h0_m,h0_kappa=h0_kappa,h0_alpha=h0_alpha,h0_beta=h0_beta)

        # hn_params
        # self.hn_m = self.h0_m
        # if self.known_precision:
        #     self.hn_tau = self.h0_tau
        # else:
        #     self.hn_kappa = None
        #     self.hn_alpha = None
        #     self.hn_beta = None

        # p_params
        self.p_mu = None
        self.p_lambda = None
        if self.known_precision:
            pass
        else:
            self.p_nu = None

        # sample size
        self._n = 0

        # sample statistics
        self.x_sum = 0.0
        self.x_squared_sum = 0.0

    def get_constants(self):
        """Get constants of LearnModel.

        This model does not have any constants. 
        Therefore, this function returns an emtpy dict ``{}``.
        
        Returns
        -------
        constants : an empty dict
        """
        return {}

    def set_h0_params(self,h0_m=None,h0_kappa=None,h0_alpha=None,h0_beta=None, h0_tau=None, h0_tau_x=None, known_precision=None):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        h0_m : float, optional
            a real number, by default None
        h0_kappa : float, optional
            a positive real number, by default None
        h0_alpha : float, optional
            a positive real number, by default None
        h0_beta : float, optional
            a positive real number, by default None
        h0_tau : float, optional
            a positive real number, by default None
        h0_tau_x : float, optional
            a positive real number, by default None
        """
        if h0_m is not None:
            self.h0_m = _check.float_(h0_m,'h0_m',ParameterFormatError)
        if h0_kappa is not None:
            self.h0_kappa = _check.pos_float(h0_kappa,'h0_kappa',ParameterFormatError)
        if h0_alpha is not None:
            self.h0_alpha = _check.pos_float(h0_alpha,'h0_alpha',ParameterFormatError)
        if h0_beta is not None:
            self.h0_beta = _check.pos_float(h0_beta,'h0_beta',ParameterFormatError)
        if h0_tau is not None:
            self.h0_tau = _check.pos_float(h0_tau,'h0_tau',ParameterFormatError)
        if h0_tau_x is not None:
            self.h0_tau_x = _check.pos_float(h0_tau_x,'h0_tau_x',ParameterFormatError)
        if known_precision is not None:
            self.known_precision = _check.boolean(known_precision,'known_precision',ParameterFormatError)
        self.reset_hn_params()
        return self

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the posterior distribution.

        Returns
        -------
        h0_params : dict of {str: float}
            * ``"h0_m"`` : The value of ``self.h0_m``
            * ``"h0_kappa"`` : The value of ``self.h0_kappa``
            * ``"h0_alpha"`` : The value of ``self.h0_alpha``
            * ``"h0_beta"`` : The value of ``self.h0_beta``
            * ``"h0_tau"`` : The value of ``self.h0_tau``
            * ``"h0_tau_x"`` : The value of ``self.h0_tau_x``
        """
        return {"h0_m":self.h0_m, "h0_kappa":self.h0_kappa, "h0_alpha":self.h0_alpha, "h0_beta":self.h0_beta, "h0_tau":self.h0_tau, "h0_tau_x":self.h0_tau_x, "known_precision": self.known_precision}
    
    def set_hn_params(
            self,
            # hn_params
            hn_m=None,
            hn_kappa=None,
            hn_alpha=None,
            hn_beta=None,
            hn_tau=None,
            # to avoid error
            hn_tau_x=None,
            known_precision=None,
        ):
        """Set updated values of the hyperparameter of the posterior distribution.
        
        Parameters
        ----------
        hn_m : float, optional
            a real number, by default None
        hn_kappa : float, optional
            a positive real number, by default None
        hn_alpha : float, optional
            a positive real number, by default None
        hn_beta : float, optional
            a positive real number, by default None
        hn_tau : float, optional
            a positive real number, by default None
        """
        # reset statistics
        self._n = 0
        self.x_sum = 0.0
        self.x_squared_sum = 0.0
        # hn_params
        if hn_m is not None:
            self.hn_m = _check.float_(hn_m,'hn_m',ParameterFormatError)
        if hn_kappa is not None:
            self.hn_kappa = _check.pos_float(hn_kappa,'hn_kappa',ParameterFormatError)
        if hn_alpha is not None:
            self.hn_alpha = _check.pos_float(hn_alpha,'hn_alpha',ParameterFormatError)
        if hn_beta is not None:
            self.hn_beta = _check.pos_float(hn_beta,'hn_beta',ParameterFormatError)
        if hn_tau is not None:
            self.hn_tau = _check.pos_float(hn_tau,'hn_tau',ParameterFormatError)
        # hn_tau_x is not used even if the precision is known. 
        # known_precision is not updated even if the value is passed. 
        # These imputs are to avoid error while calling reset_hn_params().

        self.calc_pred_dist()
        return self

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float}
            * ``"hn_m"`` : The value of ``self.hn_m``
            * ``"hn_kappa"`` : The value of ``self.hn_kappa``
            * ``"hn_alpha"`` : The value of ``self.hn_alpha``
            * ``"hn_beta"`` : The value of ``self.hn_beta``
            * ``"hn_tau"`` : The value of ``self.hn_tau``
        """
        returned_dict = {}
        key_hn_params = ["hn_m","hn_kappa","hn_alpha","hn_beta","hn_tau"]
        for key in key_hn_params:
            if hasattr(self,key):
                returned_dict[key] = getattr(self,key)
            else:
                returned_dict[key] = None
        return returned_dict
        # return {"hn_m":self.hn_m, "hn_kappa":self.hn_kappa, "hn_alpha":self.hn_alpha, "hn_beta":self.hn_beta, "hn_tau":self.hn_tau}
    
    def _check_sample(self,x):
        return _check.floats(x,'x',DataFormatError)

    def update_posterior(self,x):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy.ndarray
            Real numbers.
        """
        x = self._check_sample(x)
        try:
            n=x.size
        except:
            n=1
        self._update_posterior(x)
        return self

    def _update_posterior(self,x):
        """Update posterior without input check."""
        n=x.size
        self._n += n
        self.x_sum += np.sum(x)
        self.x_bar = self.x_sum / self._n
        self.x_squared_sum += np.sum(x**2)

        if self.known_precision:
            self.hn_m = (self.hn_tau * self.hn_m + self.h0_tau_x * self.x_sum) / (self.hn_tau + self.h0_tau_x * n)
            self.hn_tau += self.h0_tau_x * n
        else:
            self.hn_beta += (np.sum((x-self.x_bar )**2) + n*self.hn_kappa / (self.hn_kappa + n) * (self.x_bar  - self.hn_m)**2 ) / 2.0
            self.hn_m =  (self.hn_kappa * self.hn_m + self.x_sum) / (self.hn_kappa + n)
            self.hn_kappa += n
            self.hn_alpha += n*0.5
        return self

    def estimate_params(self,loss="squared",dict_out=False):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Note that the criterion is applied to estimating ``mu`` and ``tau`` independently.
        Therefore, a tuple of the student's t-distribution and the gamma distribution will be returned when loss=\"KL\"

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".
        dict_out : bool, optional
            If ``True``, output will be a dict, by default ``False``.
        
        Returns
        -------
        estimates : tuple of {float, None, or rv_frozen}
            * ``mu_hat`` : the estimate for mu
            * ``tau_hat`` : the estimate for tau
            The estimated values under the given loss function. If it is not exist, `None` will be returned.
            If the loss function is \"KL\", the posterior distribution itself will be returned
            as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """
        if loss == "squared":
            if self.known_precision:
                tau_hat = self.hn_tau
            else:
                tau_hat = self.hn_alpha / self.hn_beta
            if dict_out:
                return {'mu':self.hn_m,'tau':tau_hat}
            else:
                return self.hn_m, tau_hat
        elif loss == "0-1":
            if self.known_precision:
                tau_hat = self.hn_tau
            else:
                if self.hn_alpha > 1.0:
                    tau_hat = (self.hn_alpha - 1.0) / self.hn_beta
                else:
                    # warnings.warn("hn_alpha is less than 1.0. The estimate of tau is set to 0.0.",ParameterFormatWarning)
                    tau_hat = 0.
            if dict_out:
                return {'mu':self.hn_m,'tau':tau_hat}
            else:
                return self.hn_m, tau_hat
        elif loss == "abs":
            if self.known_precision:
                tau_hat = self.hn_tau
            else:
                tau_hat = ss_gamma.median(a=self.hn_alpha,scale=1/self.hn_beta)
            if dict_out:
                return {'mu':self.hn_m,'tau':tau_hat}
            else:
                return self.hn_m, tau_hat
        elif loss == "KL":
            if self.known_precision:
                return ss_norm(loc=self.hn_m,scale=1.0/np.sqrt(self.hn_tau)) # NOTE: There's no posterior distribution for tau in known precision case.
            else:
                return ss_t(loc=self.hn_m,scale=np.sqrt(self.hn_beta / self.hn_alpha / self.hn_kappa),df=2*self.hn_alpha),ss_gamma(a=self.hn_alpha,scale=1.0/self.hn_beta)
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))

    def estimate_interval(self,credibility=0.95):
        """Credible interval of the parameter.

        Parameters
        ----------
        credibility : float, optional
            A posterior probability that the interval conitans the paramter, by default 0.95

        Returns
        -------
        (mu_lower, mu_upper),(tau_lower, tau_upper): float
            The lower and the upper bounds of the intervals, tau_lower and tau_upper are both h0_tau when the precision is known.
        """
        _check.float_in_closed01(credibility,'credibility',CriteriaError)
        if self.known_precision:
            return (ss_norm.interval(credibility,
                                     loc=self.hn_m,
                                     scale=1.0/np.sqrt(self.hn_tau)),
                    (self.h0_tau, self.h0_tau) # NOTE: Return the pair of h0_tau when the precision is known.
                    )
        else:
            return (ss_t.interval(credibility,
                                loc=self.hn_m,
                                scale=np.sqrt(self.hn_beta / self.hn_alpha / self.hn_kappa),
                                df=2*self.hn_alpha),
                    ss_gamma.interval(credibility,a=self.hn_alpha,scale=1.0/self.hn_beta))
    
    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import normal
        >>> gen_model = normal.GenModel(mu=1.0,tau=1.0)
        >>> learn_model = normal.LearnModel()
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()
        
        .. image:: ./images/normal_posterior.png
        """
        if self.known_precision:
            mu_pdf = self.estimate_params(loss="KL")
            x1 = np.linspace(self.hn_m-4.0*np.sqrt(1.0/self.hn_tau),
                            self.hn_m+4.0*np.sqrt(1.0/self.hn_tau),
                            100)
            fig = plt.figure()
            plt.plot(x1,mu_pdf.pdf(x1))
            plt.xlabel("mu")
            plt.ylabel("Density")
        else:
            mu_pdf, tau_pdf = self.estimate_params(loss="KL")
            num_subplots = 2
            x1 = np.linspace(self.hn_m-4.0*np.sqrt(self.hn_beta / self.hn_alpha / self.hn_kappa),
                            self.hn_m+4.0*np.sqrt(self.hn_beta / self.hn_alpha / self.hn_kappa),
                            100)
            fig, axes = plt.subplots(1,num_subplots)
            # for mu
            axes[0].plot(x1,mu_pdf.pdf(x1))
            axes[0].set_xlabel("mu")
            axes[0].set_ylabel("Density")
            #for tau
            x2 = np.linspace(max(1.0e-8,self.hn_alpha/self.hn_beta-4.0*np.sqrt(self.hn_alpha)/self.hn_beta),
                            self.hn_alpha/self.hn_beta+4.0*np.sqrt(self.hn_alpha)/self.hn_beta,
                            100)
            axes[1].plot(x2,tau_pdf.pdf(x2))
            axes[1].set_xlabel("tau")
            axes[1].set_ylabel("Density")

        fig.tight_layout()
        plt.show()
    
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: float}
            * ``"p_mu"`` : The value of ``self.p_mu``
            * ``"p_lambda"`` : The value of ``self.p_lambda``
            * ``"p_nu"`` : The value of ``self.p_nu``
        """
        if self.known_precision:
            return {"p_mu":self.p_mu, "p_lambda":self.p_lambda}
        else:
            return {"p_mu":self.p_mu, "p_lambda":self.p_lambda, "p_nu":self.p_nu}
    
    def calc_pred_dist(self):
        """Calculate the parameters of the predictive distribution."""
        self.p_mu = self.hn_m
        if self.known_precision:
            self.p_lambda = (self.hn_tau * self.h0_tau_x) / (self.hn_tau + self.h0_tau_x)
        else:
            self.p_nu = 2*self.hn_alpha
            self.p_lambda = self.hn_kappa / (self.hn_kappa+1) * self.hn_alpha / self.hn_beta
        return self

    def _calc_pred_density(self,x):
        if self.known_precision:
            return ss_norm.pdf(x,loc=self.p_mu,scale=1.0/np.sqrt(self.p_lambda))
        else:
            return ss_t.pdf(x,loc=self.p_mu,scale=1.0/np.sqrt(self.p_lambda),df=self.p_nu)
    
    def _calc_pred_log_density(self,x): # ss_tにはlogdensityあるが，線形回帰とかは別かも
        return ss_t.logpdf(x,loc=self.p_mu,scale=1.0/np.sqrt(self.p_lambda),df=self.p_nu)

    def make_prediction(self,loss="squared"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".
        
        Returns
        -------
        Predicted_value : {int, numpy.ndarray}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as numpy.ndarray.
        """
        if loss == "squared" or loss == "0-1" or loss == "abs":
            return self.p_mu
        elif loss == "KL":
            if self.known_precision:
                return ss_norm(loc=self.p_mu,scale=1.0/np.sqrt(self.p_lambda))
            else:
                return ss_t(loc=self.p_mu,scale=1.0/np.sqrt(self.p_lambda),df=self.p_nu)
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))

    def pred_and_update(self,x,loss="squared"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : float
            A real number
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".
        
        Returns
        -------
        Predicted_value : {int, numpy.ndarray}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as numpy.ndarray.
        """
        _check.float_(x,'x',DataFormatError)
        self.calc_pred_dist()
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x)
        return prediction

    def calc_log_marginal_likelihood(self):
        """Calculate log marginal likelihood

        Returns
        -------
        log_marginal_likelihood : float
            The log marginal likelihood.
        """
        if self.known_precision:
            return 0.5 * (
                self._n * (
                    - np.log(2*np.pi) + np.log(self.p_lambda))
                + np.log(self.h0_tau)
                - np.log(self.hn_tau)
                - self.p_lambda * self.x_squared_sum
                - self.h0_tau * self.h0_m**2
                + self.hn_tau * self.hn_m**2
            ) # NOTE: こちらは全部自分で手計算したのであまり自信ない
            # return 0.5 * (
            #     self._n * ( - np.log(2*np.pi) + np.log(self.h0_tau_x))
            #     + np.log(self.h0_tau) - np.log(self.hn_tau)
            #     - self.h0_tau_x * self.x_squared_sum
            #     - self.h0_tau * self.h0_m**2
            #     - (self.h0_tau_x*self.x_sum + self.h0_tau*self.h0_m)**2 / (self.hn_tau)
            # ) # NOTE: ChatGPTと話し合いながらもう一度計算したが，実際に計算してみると，真のパラメータを既知として与えても最大にならない
        else:
            return (
                self.h0_alpha * np.log(self.h0_beta)
                - self.hn_alpha * np.log(self.hn_beta)
                + gammaln(self.hn_alpha)
                - gammaln(self.h0_alpha)
                + 0.5 * (
                    np.log(self.h0_kappa)
                    - np.log(self.hn_kappa)
                    - self._n * np.log(2*np.pi)
                )
            )