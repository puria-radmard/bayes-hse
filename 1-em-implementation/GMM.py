from GMM_utils import *

def E_step(X, pi, mu, sigma):
    """
    Performs E-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    pi: (C), mixture component weights 
    mu: (C x d), mixture component means
    sigma: (C x d x d), mixture component covariance matrices
    
    Returns:
    gamma: (N x C), probabilities of clusters for objects
    """
    N = X.shape[0] # number of objects
    C = pi.shape[0] # number of clusters
    d = mu.shape[1] # dimension of each object
    gamma = np.zeros((N, C)) # distribution q(T)

    ### YOUR CODE HERE

    for i in range(N):

      # Place initial values
      for c in range(C):
        gamma[i][c] = (
            f_gaussian(
                X=X[i],
                mu=mu[c],
                sigma=sigma[c]
            )
        )
      # Softmax over values
      original_gamma = gamma
      gamma[i] = pi * gamma[i]

      gamma[i] = normalise_probs(
          gamma[i]
      )

      try:
        assert abs(sum(gamma[i]) - 1) < 0.0001
      except:
        import pdb; pdb.set_trace()
    
    return gamma


def M_step(X, gamma):
    """
    Performs M-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    gamma: (N x C), distribution q(T)  
    
    Returns:
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    """
    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    ### YOUR CODE HERE
    
    # Start with means:
    mu = M_step_mus(X=X, gamma=gamma)
    assert mu.shape == (C, d)

    sigma = M_step_sigma(X=X, gamma=gamma, mu=mu)
    assert sigma.shape == (C, d, d)

    pi = M_step_pi(X=X, gamma=gamma)
    assert pi.shape == tuple([C])

    return pi, mu, sigma


def vlb_first_term(X, pi, mu, sigma, gamma):
  
    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    first_term = 0

    for i in range(N):
      for c in range(C):
        q = gamma[i][c]
        weighted_g_prob = pi[c] * f_gaussian(X[i], mu[c], sigma[c])
        first_term += q * math.log(weighted_g_prob + np.finfo(float).eps)

    return first_term


def vlb_second_term(X, pi, mu, sigma, gamma):

    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    second_term = 0

    for i in range(N):
      for c in range(C):
        q = gamma[i][c]
        second_term -= q * math.log(q + np.finfo(float).eps)
    
    return second_term


def compute_vlb(X, pi, mu, sigma, gamma):
    """
    Each input is numpy array:
    X: (N x d), data points
    gamma: (N x C), distribution q(T)  
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    
    Returns value of variational lower bound
    """
    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    ### YOUR CODE HERE
    first_term = vlb_first_term(X, pi, mu, sigma, gamma)
    second_term = vlb_second_term(X, pi, mu, sigma, gamma)

    loss = first_term + second_term

    return loss


def train_EM(X, C, rtol=1e-3, max_iter=100, restarts=10):
    '''
    Starts with random initialization *restarts* times
    Runs optimization until saturation with *rtol* reached
    or *max_iter* iterations were made.
    
    X: (N, d), data points
    C: int, number of clusters
    '''
    N = X.shape[0] # number of objects
    d = X.shape[1] # dimension of each object
    best_loss = None
    best_pi = None
    best_mu = None
    best_sigma = None

    for r in range(restarts):
        try:
            ### YOUR CODE HERE

            # Initialise parameters:

            pi = np.random.rand(C)    
            mu = np.random.rand(C, d)

            sigma = generate_spd_sigma(C, d)

            prev_iter_loss = - np.inf

            for j in range(max_iter):
              gamma = E_step(X, pi, mu, sigma)
              pi, mu, sigma = M_step(X, gamma)
              loss = compute_vlb(X, pi, mu, sigma, gamma)

              if abs(loss - prev_iter_loss) < rtol:
                break              

              try:
                assert loss >= prev_iter_loss
              except AssertionError:
                raise ValueError(f"Lower bound just went down from {prev_iter_loss} to {loss}")                

              prev_iter_loss = loss
            
            # Check this - maximisation right?
            if best_loss == None or loss > best_loss:
              best_loss = loss
              best_pi = pi
              best_mu = mu
              best_sigma = sigma

            print(f"Restart {r} gives loss {loss} after {j+1} iterations")

        except np.linalg.LinAlgError:
            print("Singular matrix: components collapsed")
            pass

    return best_loss, best_pi, best_mu, best_sigma