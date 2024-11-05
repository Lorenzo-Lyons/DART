import torch
import gpytorch
import numpy as np


# produce input data of a sine wave between 0 and 2pi
n = 100
x = torch.linspace(0, 2*np.pi, n)
y = torch.sin(2 * x) + torch.randn(n)*0.1

# produce SVGP model
# SVGP 
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

class SVGPModel(ApproximateGP):
    def __init__(self,inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

# initialize inducing points by selecting them randomly from the input data
n_inducing = 5
#inducing_points = x[torch.randperm(n)[:n_inducing]] # random inputs
inducing_points = x[:n_inducing] # first n datapoints
# intialize SVGP model
model = SVGPModel(inducing_points)

# initialize likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# training loop
model.train()
likelihood.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# Use the adam optimizer
mll = gpytorch.mlls.VariationalELBO(likelihood, model, y.numel())
training_iterations = 100

# plot training data and inducing points
import matplotlib.pyplot as plt
# plotting body frame accelerations and data fitting results 
fig, (ax) = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
fig.subplots_adjust(top=0.96,
bottom=0.055,
left=0.05,
right=0.995,
hspace=0.345,
wspace=0.2)

ax.plot(x.numpy(), y.numpy(), 'k.')
#plot inducing points as vertical lines
for i in range(n_inducing):
    ax.axvline(inducing_points[i].item(), color='r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Training Data')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-3, 3)
# plot predictive mean and confidence interval
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 2 * np.pi, 51)
    observed_pred = likelihood(model(test_x))
    lower, upper = observed_pred.confidence_region()
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

plt.ion()

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(x)
    loss = -mll(output, y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()

    # update inducing points location on plot
    ax.clear()
    ax.plot(x.numpy(), y.numpy(), 'k.')
    #plot updated inducing points as vertical lines
    for i in range(n_inducing):
        ax.axvline(model.variational_strategy.inducing_points[i].item(), color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Training Data')
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-3, 3)
    # plot predictive mean and confidence interval
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 2 * np.pi, 51)
        observed_pred_likelyhood = likelihood(model(test_x))
        lower_likelyhood, upper_likelyhood = observed_pred_likelyhood.confidence_region()
        ax.plot(test_x.numpy(), observed_pred_likelyhood.mean.numpy(), 'b')
        ax.fill_between(test_x.numpy(), lower_likelyhood.numpy(), upper_likelyhood.numpy(), alpha=0.3)

        # obseved pred epistemic uncertainty
        observed_pred_model = model(test_x)
        lower_model, upper_model = observed_pred_model.confidence_region()
        ax.fill_between(test_x.numpy(), lower_model.numpy(), upper_model.numpy(), alpha=0.3, color='k')

        
    plt.pause(0.1)
plt.ioff()
plt.show()
    


