import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

features2 = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features2)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Your code here to process the auto data
# print(hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data, auto_labels, 10))

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data
#print(hw3.xval_learning_alg(hw3.perceptron, review_bow_data, review_labels, 10))
#print(hw3.xval_learning_alg(hw3.averaged_perceptron, review_bow_data, review_labels, 10))

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[9]["images"]
d1 = mnist_data_all[0]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n_samples, m, n = x.shape
    a = np.empty((m*n, 0))
    
    for i in range(n_samples):
        temp = np.empty((1, 0))
        for j in range(m):
            temp = np.concatenate((temp, x[i,j:j+1,]), axis=1)
        a = np.concatenate((a, temp.T), axis=1)
    return a

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    d, n = x.shape
    a = np.zeros((d, 1))
    
    for i in range(d):
        a[i,:] = np.average(x[i,:])

    return a



def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    d, n = x.shape
    a = np.zeros((n, 1))
    
    for i in range(n):
        a[i,:] = np.average(x[:,i:i+1])

    return a


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    m, n = x.shape
    a = np.zeros((2, 1))
    sep = int(np.floor(m/2))
    odd_rows = 0 if m % 2 == 0 else 1

    
    a[0,0:1] = np.sum(row_average_features(x[0:sep,:]))/sep
    a[1,0:1] = np.sum(row_average_features(x[sep:,:]))/(sep + odd_rows)
    
    return a
    


#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data
n_samples, d, n = data.shape
data_list = list()

# ROW

for i in data:
    data_list.append([row_average_features(i)])

dataN = np.vstack(data_list)

# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(dataN), labels)
print(acc)


# COL
data_list = list()
for i in data:
    data_list.append([col_average_features(i)])

dataN = np.vstack(data_list)

# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(dataN), labels)
print(acc)


# TOP BOTTOM
data_list = list()
for i in data:
    data_list.append([top_bottom_features(i)])

dataN = np.vstack(data_list)

# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(dataN), labels)
print(acc)





