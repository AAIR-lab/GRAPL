from scipy.stats import chisquare
import random
import numpy as np

def generate_samples(num_samples, distribution):

    samples = {i: 0 for i in range(len(distribution))}
    for _ in range(num_samples):
        
        current_weight = 0
        for j in range(len(distribution)):

            current_weight += distribution[j]
            if random.random() <= current_weight:
            
                samples[j] = samples.setdefault(j, 0) + 1
                break
                
    return samples
            

def perform_chisquare(distribution, num_samples, obs):

    distribution = distribution * num_samples
    chi_test = chisquare(f_obs=obs, f_exp=distribution)
    print(chi_test)

if __name__ == "__main__":

    NUM_SAMPLES = 5
    dist1 = np.asarray([0.2, 0.8])
    dist2 = np.asarray([0.5, 0.5])
    dist = dist1
    
    samples = generate_samples(NUM_SAMPLES, dist1)
    
    samples = np.asarray([samples[k] for k in sorted(samples.keys())])
    print(samples)

    
    perform_chisquare(dist, NUM_SAMPLES, samples)
