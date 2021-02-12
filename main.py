# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pymc3 as pm
import arviz as az
import theano.tensor as tt
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 0

class GaussianModel:
    def __init__(self, mu_1, mu_2, sigma_1, sigma_2, sigma_3, X, Y_obs):
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.sigma_3 = sigma_3
        self.X = X
        self.Y_obs = Y_obs


    def run(self):
        basic_model = pm.Model()
        with basic_model:
            slope = pm.Normal("slope", mu=self.mu_1, sigma=self.sigma_1)
            intercept = pm.Normal("intercept", mu=self.mu_2, sigma=self.sigma_2)
            mu = slope * self.X + intercept
            pm.Normal("Y_obs", mu=mu, sigma=self.sigma_3, observed=self.Y_obs)
            self.trace = pm.sample(1000)
            az.plot_trace(self.trace, show=True)
            print(az.summary(self.trace, round_to=2))

class UniformModel:
    def __init__(self, lower_1, lower_2, upper_1, upper_2, sigma_3, X, Y_obs):
        self.lower_1 = lower_1
        self.lower_2 = lower_2
        self.upper_1 = upper_1
        self.upper_2 = upper_2
        self.sigma_3 = sigma_3
        self.X = X
        self.Y_obs = Y_obs


    def run(self):
        basic_model = pm.Model()
        with basic_model:
            slope = pm.Uniform("slope", lower=self.lower_1, upper=self.upper_1)
            intercept = pm.Uniform("intercept", lower=self.lower_2, upper=self.upper_2)
            mu = slope * self.X + intercept
            pm.Normal("Y_obs", mu=mu, sigma=self.sigma_3, observed=self.Y_obs)
            self.trace = pm.sample(1000)
            az.plot_trace(self.trace, show=True,)
            print(az.summary(self.trace, round_to=2))

def run():
    X = [x for x in range(-5, 5)]
    Y_obs = [5.75003, 5.1568, 3.26414, 0.84894, 2.09686, 0.94026, 0.36411, -1.83959, -1.976, -2.93363]


    gauss1 = GaussianModel(mu_1=0, mu_2=1, sigma_1=1, sigma_2=1, sigma_3=0.1, X=X, Y_obs=Y_obs)
    gauss1.run()
    #gauss2 = GaussianModel(mu_1=5, mu_2=10, sigma_1=0.1, sigma_2=0.1, sigma_3=0.1, X=X, Y_obs=Y_obs)
    #gauss2.run()
    #gauss3 = GaussianModel(mu_1=0, mu_2=1, sigma_1=1, sigma_2=1, sigma_3=20, X=X, Y_obs=Y_obs)
    #gauss3.run()
    #uniform1 = UniformModel(lower_1=-1, lower_2=0, upper_1=1, upper_2=2, sigma_3=0.1, X=X, Y_obs=Y_obs)
    #uniform1.run()

    slope_1 = gauss1.trace["slope"].mean()
    #slope_2 = gauss2.trace["slope"].mean()
    #slope_3 = gauss3.trace["slope"].mean()
    #slope_4 = uniform1.trace["slope"].mean()
    intercept_1 = gauss1.trace["intercept"].mean()
    #intercept_2 = gauss2.trace["intercept"].mean()
    #intercept_3 = gauss3.trace["intercept"].mean()
    #intercept_4 = uniform1.trace["intercept"].mean()
    Y_1 = np.multiply(slope_1, X) + intercept_1
    #Y_2 = np.multiply(slope_2, X) + intercept_2
    #Y_3 = np.multiply(slope_3, X) + intercept_3
    #Y_4 = np.multiply(slope_4, X) + intercept_4

    plt.figure()
    plt.title('Y_observed')
    plt.plot(X, Y_obs, 'bo')
    plt.plot(X, Y_1)
    #plt.plot(X, Y_2)
    #plt.plot(X, Y_3)
    #plt.plot(X, Y_4)
    plt.show()



if __name__ == '__main__':
    run()