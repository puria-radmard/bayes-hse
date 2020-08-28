from GMM import *

def load_sample(fname):
    samples = np.load(fname)
    X = samples['data']
    pi0 = samples['pi0']
    mu0 = samples['mu0']
    sigma0 = samples['sigma0']
    plt.scatter(X[:, 0], X[:, 1], c='grey', s=30)
    plt.axis('equal')
    plt.savefig("raw-sample")
    plt.close()

    return X, pi0, mu0, sigma0

if __name__ == "__main__":
    X, _, _, _ = load_sample('samples.npz')
    
    best_loss, best_pi, best_mu, best_sigma = train_EM(X, 3)

    gamma = E_step(X, best_pi, best_mu, best_sigma)
    labels = gamma.argmax(axis=1)
    colors = np.array([(31, 119, 180), (255, 127, 14), (44, 160, 44)]) / 255.
    plt.scatter(X[:, 0], X[:, 1], c=colors[labels], s=30)
    plt.axis('equal')
    plt.savefig("clustered")