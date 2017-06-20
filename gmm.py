# How to use this code
#
# 1. Install numpy, sklearn and ipython
# 2. Run ipython in the directory where you have gmm.py
# 3. In ipython:
#
# - Load the script
# >>> %run gmm.py
#
# - Generate some data
# this generates 20 points from a distribution with mean -1 and sd 2 and
# 40 points from a distribution with mean 2 and sd 1
# >>> data = make_data([(-1, 2, 20), (2, 1, 40)])
#
# - Play with EM and the various parameters
# >>> em_gmm(data, 2, use_kmeans=True)

import numpy as np

# use sklearn for K-Means
import sklearn
import sklearn.cluster

EPS = 0.0000001

# pdf of the normal distribution
def norm_pdf(x, mean=0., sd=1.):
  return 1/(sd * np.sqrt(2 * np.pi)) * np.exp(- (x - mean)** 2/ (2 * sd ** 2))


# generate date according to specifications
# specs is a list of (mean, sd, number of samples) for each distribution
def make_data(specs):
  return np.concatenate([
    np.random.normal(loc=mean, scale=sd, size=n) for\
    (mean, sd, n) in specs])

# use sklearn to do k-means
def do_kmeans(data, k):
  km = sklearn.cluster.KMeans(n_clusters=k)
  km.fit(data)

  means = km.cluster_centers_.reshape((-1,))

  #initialize standard deviations with distances between random cluster centers
  sds = []
  for i in range(means.shape[0]):
    # choose any 2 means and take half the distance between them
    x, y = np.random.choice(means, 2, replace=False)
    sds.append((x-y)/2)
  sds = np.abs(np.array(sds))

  return (means, sds)

# expectation maximization for gmm
# use_kmeans: whether to initialize using kmeans or randomly
# use_priors: whether to model the prior distribution;
# this attaches a weight to each distribution that tells us
# the percentage of points generated from that distribution
def em_gmm(data, k, use_kmeans=False, use_priors=False):
  N = len(data)
  # data as a Nx1 matrix
  vdata = data.reshape((-1, 1))
  if use_kmeans:
    means, sds = do_kmeans(vdata, k)
  else:
    means = np.random.uniform(low=-2, high=2, size=k)
    sds = np.random.uniform(low=-2, high=2, size=k)

  priors = np.ones(k)
  priors /= sum(priors)

  print "Initial values:\nMeans: %s\nSDs: %s\nPriors: %s\n" % (means, sds, priors)

  # this encodes the weights for each sample/distribution pair
  # probs[i, j] tells us the probability that sample i came from distribution j
  probs = np.zeros((N, k))

  improvement = 1000
  iteration = 0

  if use_priors:
    print "Theta = (%s, %s, %s)" % (means, sds, priors)
  else:
    print "Theta = (%s, %s)" % (means, sds)

  # start of EM
  while improvement > EPS:
    # E-step: compute probs using current parameters
    for j in range(k):
      if use_priors:
        probs[:, j] = norm_pdf(data, mean=means[j], sd=sds[j]) * priors[j]
      else:
        probs[:, j] = norm_pdf(data, mean=means[j], sd=sds[j])

    # normalize probabilities row-wise
    probs[probs < EPS] = EPS # make sure we don't have 0 probabilities
    probs = probs / probs.sum(axis=1)[:, np.newaxis]

    # M-step: compute parameters using probs
    if use_priors:
      new_priors = probs.sum(axis=0)
      new_priors /= new_priors.sum()
    new_means = (probs * vdata).sum(axis=0) / probs.sum(axis=0)
    aux = np.tile(new_means, (N, 1))
    vdiffs = np.abs(vdata - aux)
    new_sds = np.sqrt((probs * vdiffs**2).sum(axis=0) / probs.sum(axis=0))

    # maximum mean or sd change
    improvement = max(np.max(np.abs(new_means - means)),
                     np.max(np.abs(new_sds - sds)))

    # update the parameters
    means = new_means
    sds = new_sds
    if use_priors:
      priors = new_priors

    sds[np.isnan(sds)] = EPS # in case we get a NaN, make it epsilon

    print "Step %s\nMeans: %s\nSDs: %s\nPriors: %s\nImprovement: %s\n" % (iteration, means, sds, priors, improvement)
    iteration += 1

  print "Final weights\n", probs
  print "Final\nMeans: %s\nSDs: %s\nPriors: %s\nImprovement: %s\n" % (means, sds, priors, improvement)

