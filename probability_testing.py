# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import binom

# # Number of tosses
# num_tosses = 2000

# # Number of heads observed
# num_heads_observed = 1850

# # Range of biases (from 0% to 100%)
# biases = np.arange(0, 101, 0.01) / 100.0


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import binom

# # Number of tosses
# num_tosses = 2000

# # Number of heads observed
# num_heads_observed = 1300

# probabilities = []
# for bias in biases:
#     # Probability of success (getting heads) for each trial
#     p = bias
    
#     # Probability of getting exactly num_heads_observed heads with bias p
#     probability = binom.pmf(num_heads_observed, num_tosses, p)
#     probabilities.append(probability)

# # Replace nan values with zero
# probabilities = np.nan_to_num(probabilities)

# # Find the bias value with the highest probability
# max_probability_index = np.argmax(probabilities)
# bias_with_max_probability = biases[max_probability_index]

# print(f"The most likely bias was {bias_with_max_probability}")


# from scipy.stats import binom

# # Number of tosses
# num_tosses = 2000

# # Number of heads observed
# num_heads_observed = 1850

# # Define the range of bias values
# lower_bound_bias = bias_with_max_probability -1
# upper_bound_bias = bias_with_max_probability +1

# # Initialize the probability
# probability_within_range = 0

# # Calculate the probability within the specified range
# for bias in range(int(lower_bound_bias * 100), int(upper_bound_bias * 100) + 1):
#     p = bias / 100.0
#     probability_within_range += binom.pmf(num_heads_observed, num_tosses, p)

# print("Probability that the true value is between 89% and 91%:", probability_within_range)



# # Plotting
# plt.plot(biases, probabilities)
# plt.xlabel('Bias')
# plt.ylabel('Probability')
# plt.title('Probability of observing 1850 heads out of 2000 tosses for different biases')
# plt.grid(True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import binom

# # Number of tosses
# num_tosses = 2000

# # Number of heads observed
# num_heads_observed = 1300

# # Range of biases (from 0% to 100%)
# biases = np.arange(0, 101, 0.01) / 100.0

# # Calculate probabilities for each bias
# probabilities = []
# for bias in biases:
#     # Probability of success (getting heads) for each trial
#     p = bias
    
#     # Probability of getting exactly num_heads_observed heads with bias p
#     probability = binom.pmf(num_heads_observed, num_tosses, p)
#     probabilities.append(probability)

# # Replace nan values with zero
# probabilities = np.nan_to_num(probabilities)

# # Find the bias value with the highest probability
# max_probability_index = np.argmax(probabilities)
# bias_with_max_probability = biases[max_probability_index]

# print(f"The most likely bias was {bias_with_max_probability:.2%}")

# # Define the range of bias values within 1% of the most likely bias
# lower_bound_bias = bias_with_max_probability - 0.05
# upper_bound_bias = bias_with_max_probability + 0.05

# print(f"lower bound bias = {lower_bound_bias}")
# print(f"Upper bound bias = {upper_bound_bias}")

# # Initialize the probability within the specified range
# probability_within_range = 0

# # Calculate the probability within the specified range
# for bias in np.arange(lower_bound_bias, upper_bound_bias, 0.01):
#     probability_within_range += binom.pmf(num_heads_observed, num_tosses, bias)

# print("Probability that the true value is within 1% of the most likely bias:",
#       probability_within_range)

# # Plotting
# plt.plot(biases, probabilities)
# plt.xlabel('Bias')
# plt.ylabel('Probability')
# plt.title('Probability of observing 1300 heads out of 2000 tosses for different biases')
# plt.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Number of tosses
num_tosses = 10586

# Number of heads observed
num_heads_observed = 9781

# Mean and standard deviation of the binomial distribution
mean = num_tosses * 0.5  # Assuming unbiased coin
std_dev = np.sqrt(num_tosses * num_heads_observed/num_tosses * (1-num_heads_observed/num_tosses))  # Variance for binomial distribution is np(1-p)

# Range of biases (from 0% to 100%)
biases = np.linspace(0, 1.0, 1000)

# Calculate the probabilities using the normal approximation to the binomial distribution
probabilities = norm.pdf((num_heads_observed - biases * num_tosses) / std_dev) / std_dev


# Find the maximum probability bias
max_probability_bias = biases[np.argmax(probabilities)]

intervals = np.linspace(0, 0.2, 1000)
likelyhoods = []
for interval in intervals:
    # Define the range of bias values within 1% of the most likely bias
    lower_bound_bias = max(0, max_probability_bias - interval)
    upper_bound_bias = min(1.0, max_probability_bias + interval)

    # Calculate the probability within the specified range
    probability_within_range = norm.cdf((upper_bound_bias * num_tosses - num_heads_observed) / std_dev) \
                            - norm.cdf((lower_bound_bias * num_tosses - num_heads_observed) / std_dev)
    
    likelyhoods.append(probability_within_range)


# Find the first interval value that had a likelihood of less than 0.95
for index, likelyhood in enumerate(likelyhoods):
    if likelyhood > 0.95:
        break

print(f"The error is {intervals[[index - 1]]}")

# Plotting
# plt.plot(intervals, likelyhoods)
# plt.xlabel('Possible Error Values')
# plt.ylabel('Probability that Error is Lower than the x Value ')
# plt.title('Error and Confidence')
# plt.grid(True)
# plt.show()
    

print(f"The most likely bias was {max_probability_bias:.2%}")
print("Probability that the true value is within 1% of the most likely bias:",
      probability_within_range)

#Plotting
plt.plot(biases, probabilities)
plt.xlabel('Possible Values for True Model Accuracy')
plt.ylabel('Probability Density')
plt.title("Alkene Probability Density Distribution for Possible Accuracies")
plt.grid(True)
plt.show()