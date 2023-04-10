import numpy as np


class CategoricalCrossEntropy:
    @staticmethod
    def compute_loss(labels, predictions, epsilon=1e-8):
        '''
       The function to compute the categorical cross entropy loss, given training labels and prediction
       :param labels:[numpy array]: Training labels
       :param predictions:[numpy array]: Predicted labels
       :param epsilon:[float default=1e-8]: A small value for applying clipping for stability
       :return:[float]: The computed value of loss.
       '''
        predictions /= np.sum(predictions, axis=0, keepdims=True)
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        return -np.sum(labels * np.log(predictions))

    @staticmethod
    def compute_derivative(labels, predictions):
        '''
        The function to compute the derivative values of categorical cross entropy values, given labels and prediction
        :param labels:[numpy array]: Training labels
        :param predictions:[numpy array]: Predicted labels
        :return:[numpy array]: The computed derivatives of categorical cross entropy function.
        '''
        return labels - predictions

class Root_Mean_Squared_Error:
    @staticmethod
    def compute_loss(labels, predictions, epsilon=1e-8):
        '''
       The function to compute the categorical cross entropy loss, given training labels and prediction
       :param labels:[numpy array]: Training labels
       :param predictions:[numpy array]: Predicted labels
       :param epsilon:[float default=1e-8]: A small value for applying clipping for stability
       :return:[float]: The computed value of loss.
       '''

        diff = predictions - labels
        differences_squared = diff ** 2
        mean_diff = differences_squared.mean()
        return np.sqrt(mean_diff)

    @staticmethod
    def compute_derivative(labels, predictions):
        '''
        The function to compute the derivative values of categorical cross entropy values, given labels and prediction
        :param labels:[numpy array]: Training labels
        :param predictions:[numpy array]: Predicted labels
        :return:[numpy array]: The computed derivatives of categorical cross entropy function.
        '''

        diff = predictions - labels
        # return np.sum(diff)/(len(labels) * np.sqrt((1/len(labels)) * (diff ** 2)))
        return diff/np.sqrt(len(labels))

class Mean_Squared_Error:
    @staticmethod
    def compute_loss(labels, predictions, epsilon=1e-8):
        '''
       The function to compute the categorical cross entropy loss, given training labels and prediction
       :param labels:[numpy array]: Training labels
       :param predictions:[numpy array]: Predicted labels
       :param epsilon:[float default=1e-8]: A small value for applying clipping for stability
       :return:[float]: The computed value of loss.
       '''

        diff = predictions - labels
        differences_squared = diff ** 2
        mean_diff = differences_squared.mean()
        return mean_diff

    @staticmethod
    def compute_derivative(labels, predictions):
        '''
        The function to compute the derivative values of categorical cross entropy values, given labels and prediction
        :param labels:[numpy array]: Training labels
        :param predictions:[numpy array]: Predicted labels
        :return:[numpy array]: The computed derivatives of categorical cross entropy function.
        '''

        diff = predictions - labels

        return 2 * diff / len(labels)

class Mean_Absolute_Error:
    @staticmethod
    def compute_loss(labels, predictions, epsilon=1e-8):
        '''
       The function to compute the categorical cross entropy loss, given training labels and prediction
       :param labels:[numpy array]: Training labels
       :param predictions:[numpy array]: Predicted labels
       :param epsilon:[float default=1e-8]: A small value for applying clipping for stability
       :return:[float]: The computed value of loss.
       '''

        diff = labels - predictions
        abs_diff = np.absolute(diff)
        mean_diff = abs_diff.mean()
        return mean_diff

    @staticmethod
    def compute_derivative(labels, predictions):
        '''
        The function to compute the derivative values of categorical cross entropy values, given labels and prediction
        :param labels:[numpy array]: Training labels
        :param predictions:[numpy array]: Predicted labels
        :return:[numpy array]: The computed derivatives of categorical cross entropy function.
        '''
        return np.array([1 if predictions[i]>labels[i] else -1 for i in range(len(labels))])
