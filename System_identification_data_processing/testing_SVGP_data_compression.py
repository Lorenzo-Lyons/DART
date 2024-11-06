import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt


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
    


def plot_GP(ax,x,y,x_subset,y_subset,model,likelihood,resolution):
    resolution = resolution * 0.99
    ax.plot(x.numpy(), y.numpy(), 'k.')
    #plot inducing points as vertical lines
    for i in range(n_inducing):
        ax.axvline(model.variational_strategy.inducing_points.data[i].item(), color='r',alpha=0.3, label="Inducing Points" if i == 0 else "")
    # Plot data subset as gray vertical bars with width specified by resolution
    # for i in range(len(x_subset)):
    #     subset_point = x_subset[i].item()
    #     ax.axvspan(subset_point - resolution / 2, subset_point + resolution / 2, 
    #                color='gray',edgecolor='none', alpha=0.5, label="Data Subset" if i == 0 else "")


    # plot subset with an orange circle
    ax.plot(x_subset.numpy(), y_subset.numpy(), 'o', color='orange',alpha=0.5)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_likelyhood = likelihood(model(x))
        lower_likelyhood, upper_likelyhood = observed_pred_likelyhood.confidence_region()
        ax.plot(x.numpy(), observed_pred_likelyhood.mean.numpy(), 'b')
        ax.fill_between(x.numpy(), lower_likelyhood.numpy(), upper_likelyhood.numpy(), alpha=0.3)

        # obseved pred epistemic uncertainty
        observed_pred_model = model(x)
        lower_model, upper_model = observed_pred_model.confidence_region()
        ax.fill_between(x.numpy(), lower_model.numpy(), upper_model.numpy(), alpha=0.3, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Training Data')
    ax.set_xlim(-0.1,0.1+ 2 * np.pi)
    ax.set_ylim(-3, 3)






# initialize inducing points by selecting them randomly from the input data
n_inducing = 10
inducing_points = torch.linspace(0, 2*np.pi, n_inducing)
resolution = x[1].item()-x[0].item()
#inducing_points = x[torch.randperm(n)[:n_inducing]] # random inputs
#inducing_points = x[:n_inducing] # first n datapoints
# define data subset as the inducing points
subset_indexes = list(range(n_inducing))
x_subset = x.clone()[subset_indexes]
y_subset = y.clone()[subset_indexes]


training_iterations = 200

# plot training data and inducing points





# intialize SVGP model
model = SVGPModel(inducing_points)
model.train()
# initialize likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.train()

optimizer = torch.optim.Adam([
{'params': model.parameters()},
{'params': likelihood.parameters()},
], lr=0.1)

# Use the adam optimizer
mll = gpytorch.mlls.VariationalELBO(likelihood, model, y.numel())

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(x)
    loss = -mll(output, y)
    loss.backward()
    optimizer.step()

# now remove the lengthscale and the outputscale from the training process
model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
model.covar_module.raw_outputscale.requires_grad = False



# plotting body frame accelerations and data fitting results 
fig, (ax) = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
plt.ion()
plot_GP(ax,x,y,x_subset,y_subset,model,likelihood,resolution)



# evalaute max uncertainty rateo
prior_stdd = np.sqrt(model.covar_module.outputscale.item())
preds = model(x) # evaluate predictive posterior on full dataset
max_stdd = torch.max(preds.variance).item()**0.5 # evaluate max uncertainty

# evalaute max rateo
stdd_rateo_threshold = max_stdd/prior_stdd 
acceptable_rateo_increase = 1.0 #we allow 10% increase in the rateo



#initially assign the x subset to the inducing points
# x_subset = x_subset.detach()  # Detach to avoid any gradient tracking
# model.variational_strategy.inducing_points.data = x_subset



# start training loop
max_stdd_rateo = 1 + stdd_rateo_threshold # set to high value to start the loop

# set inducing points to the subset
inducing_points = torch.unsqueeze(x.clone()[:n_inducing],1)


# now data compression
while max_stdd_rateo > stdd_rateo_threshold * acceptable_rateo_increase:
    # update inducing points first guess
    model.variational_strategy.inducing_points.data = inducing_points

    # now just optimize the inducing points locations
    # Only optimize the inducing points
    optimizer = torch.optim.Adam([
        {'params': [param for name, param in model.named_parameters()
                    if 'raw_lengthscale' not in name and 'outputscale' not in name]}
    ], lr=0.1)

    # Use the adam optimizer
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_subset.numel())

    for i in range(200):
        optimizer.zero_grad()
        output = model(x_subset)
        loss = -mll(output, y_subset)
        loss.backward()
        optimizer.step()

    # plotting results
    ax.clear()
    plot_GP(ax,x,y,x_subset,y_subset,model,likelihood,resolution)

    # --- finished plotting ---




    # update points in dataset and inducing points

    # evaluate maximum uncertainty
    prior_stdd = np.sqrt(model.covar_module.outputscale.item())


    preds = model(x)
    stdd = preds.variance.sqrt()  # Get standard deviations

    # Sort uncertainties in descending order and get the indices
    sorted_indices = torch.argsort(stdd, descending=True)

    # Find the first index with the highest uncertainty that is not in subset_indexes
    for idx in sorted_indices:
        if idx.item() not in subset_indexes:
            max_uncertainty_idx = idx.item()
            max_stdd = stdd[max_uncertainty_idx].item()
            max_stdd_rateo = max_stdd/prior_stdd
            subset_indexes.append(max_uncertainty_idx)
            break
    else:
        print("All high-uncertainty points are already in the subset.")





    print(f"max_stdd_rateo = {max_stdd_rateo:.4f}, stdd_rateo_threshold = {stdd_rateo_threshold:.4f}")


    #update the subset of data
    x_subset = x.clone()[subset_indexes] # select chosen points
    y_subset = y.clone()[subset_indexes]


    # !! also update the inducing points !!
    # identify the least informative inducing point
    # evaluate kernel matrix
    KZZ = model.covar_module(model.variational_strategy.inducing_points)

    # select row with the highest sum
    sum_KZZ = torch.sum(KZZ,1)

    # find the index of the least informative inducing point
    least_informative_idx = torch.argmax(sum_KZZ).item()


    # add line to plot the next data point to be added
    ax.axvline(x[max_uncertainty_idx].item(), color='g',alpha=0.3)
    # add dashed line on the inducing point that will be eliminated
    ax.axvline(model.variational_strategy.inducing_points[least_informative_idx].item(), color='k',linestyle='--',alpha=0.3)
    plt.pause(0.1)


    a = 1
    # replace the least informative inducing point with the new data point
    with torch.no_grad():
        inducing_points[least_informative_idx] = x.clone()[max_uncertainty_idx]
    #   model.variational_strategy.inducing_points[least_informative_idx] = x[max_uncertainty_idx]


plt.ioff()
plt.show()
    


