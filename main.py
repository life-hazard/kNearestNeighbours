# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import data
from numpy import *
import operator

import matplotlib
import matplotlib.pyplot as plt


# create data set and labels
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# create classifier:
def classify(input_data_to_classify, data_set, labels, k):
    # .shape(number_of_rows, elements_in_row); shape[0] shows number of rows
    data_set_size = data_set.shape[0]

    # calculate distance
    '''
    to calculate distance we make a matrix of the input points,
    then we subtract the points from data set from the input points | (a0 - b0) (a1 - b1)
    then each of those is squared | (a0 - b0)^2 (a1 - b1)^2
    then we sum them | (a0 - b0)^2 + (a1 - b1)^2
    then take square root which is the same as ^1/2
    now each value of the array is a calculated distance between input point and points from data set
    '''
    # tile(A, reps) <- constructs array by repeating A the number of time given by reps
    differential_matrix = tile(input_data_to_classify, (data_set_size, 1)) - data_set   # (a0 - b0) (a1 - b1)
    squared_differential_matrix = differential_matrix ** 2  # (a0 - b0)^2 (a1 - b1)^2
    squared_distances = squared_differential_matrix.sum(axis=1)     # (a0 - b0)^2 + (a1 - b1)^2
    distances = squared_distances ** 0.5    # ( (a0 - b0)^2 + (a1 - b1)^2 )^1/2

    # sort distances in increasing order | indices - plural of index
    # it doesn't rearrange the distances but shows at which index is the smallest to biggest distance
    sorted_distances_indices = distances.argsort()

    class_count = {}

    # for every point in number first (lowest) k distances
    for i in range(k):
        # vote on a class of input data
        # chooses a label for the lowest distance
        vote_I_label = labels[sorted_distances_indices[i]]
        # to class count dictionary put label as key and count as count + 1
        class_count[vote_I_label] = class_count.get(vote_I_label, 0) + 1

    # sort class count - put it into list of tuples, then sort the tuples using itemgetter from operator module
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    # return first (most counted) label from dictionary - majority class prediction
    return sorted_class_count[0][0]


def file_to_matrix(filename):
    file = open(filename)
    # create an array where each line is a string value
    array_of_lines = file.readlines()
    # get number of all values
    number_of_lines = len(array_of_lines)
    # create a matrix as big as the array
    returned_matrix = zeros((number_of_lines, 3))
    # create vector for each label
    class_label_vector = []
    index = 0

    for lines in array_of_lines:
        # remove trailing new lines
        lines = lines.rstrip('\n')
        # create a list of values from each line
        list_from_line = lines.split('\t')
        # change line in matrix with given index to the end into three first elements of the list
        returned_matrix[index, :] = list_from_line[0:3]
        index += 1
        # add label to the label array
        if list_from_line[3] == 'largeDoses':
            class_label_vector.append(3)
        elif list_from_line[3] == 'smallDoses':
            class_label_vector.append(2)
        else:
            class_label_vector.append(1)

    # print(class_label_vector)
    # print(returned_matrix)
    return returned_matrix, class_label_vector


# normalizing numeric values
def normalize_values(data_set):
    # to normalize it is common to put values into range <0,1> or <-1,1>
    # to scale to range 0 to 1 apply formula: new_value = (old_value-min) / (max-min) <- min and max are smallest and
    # largest values in the dataset

    # find biggest and smallest values in data set, and make a new range
    # 0 from .min(0) and .max(0) allows to take min and max form columns, not the rows
    min_values = data_set.min(0)
    max_values = data_set.max(0)
    range_set = max_values - min_values
    # create a new matrix to fill with the shape of data set
    normalized_data_set = zeros(shape(data_set))
    # number of lines in data set
    number_of_lines = data_set.shape[0]
    # tile(A, reps) <- constructs array by repeating A the number of time given by reps
    # new value = old value - min
    normalized_data_set = data_set - tile(min_values, (number_of_lines, 1))
    # new value = (old - min) / (max - min)
    normalized_data_set = normalized_data_set / tile(range_set, (number_of_lines, 1))

    return normalized_data_set, range_set, min_values


# testing classifier
def dating_class_test():
    # the data isn't sorted so we can use 10% of data from beginning or end of the data set
    # error rate = total number of errors / total number of data points tested
    # ratio of sth idk
    testing_percentage = 0.10

    # take data from file
    dating_data_matrix, dating_labels = file_to_matrix('datingTestSet.txt')
    # normalize data set
    normalized_data_set, ranges, min_values = normalize_values(dating_data_matrix)

    # get number of text vectors
    number_of_lines = normalized_data_set.shape[0]
    # decide how many vectors are to be tested
    number_of_test_vectors = int(number_of_lines * testing_percentage)

    error_count = 0.0

    for i in range(number_of_test_vectors):

        # classify(input_data_to_classify, data_set, labels, k)
        # input data to classify - first three elements of each vector
        # data set - whole data set which excludes chosen 10%
        # labels - all labels excluding chosen 10%
        classifier_result = classify(normalized_data_set[i, :],
                                     normalized_data_set[number_of_test_vectors:number_of_lines, :],
                                     dating_labels[number_of_test_vectors:number_of_lines],
                                     4)
        print(f'The classifier came out with the result {classifier_result}, the real answer is {dating_labels[i]}')
        # add to error count if the result is different from expected result
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print(f'total error rate is {error_count/float(number_of_test_vectors)}')


# predictor function
def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    # get input from user
    video_games_percent = float(input('Percentage of time spent on video games: '))
    flier_miles = float(input('Flier miles earned per year: '))
    liters_of_ice_cream = float(input('Liters of ice cream consumed per year: '))
    # take data set from file
    dating_data_matrix, dating_labels = file_to_matrix('datingTestSet.txt')
    # normalize data
    normalized_data_matrix, range_set, min_values = normalize_values(dating_data_matrix)
    # put input into the classifier
    input_array = array([flier_miles, video_games_percent, liters_of_ice_cream])
    classifier_result = classify((input_array - min_values) / range_set, normalized_data_matrix, dating_labels, 3)
    print('You will probably like this person', result_list[classifier_result - 1])


if __name__ == '__main__':
    group, labels = createDataSet()
    # print(group, labels, sep='\n')
    # print(classify([0.6, 0.8], group, labels, 3))

    data_matrix, data_labels = file_to_matrix('datingTestSet.txt')

    print(data_labels)
    # making a plot from matrix
    plot_figure = plt.figure()
    ax = plot_figure.add_subplot(111)
    # make dots: xaxis, yaxis, marker size & color
    ax.scatter(data_matrix[:, 1], data_matrix[:, 2], 15.0*array(data_labels), array(data_labels))
    plt.figlegend(data_labels)
    plt.show()
    normalize_values(data_matrix)
    classify_person()
