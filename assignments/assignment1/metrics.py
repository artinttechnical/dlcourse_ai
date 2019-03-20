import numpy as np

def binary_classification_metrics(prediction, ground_truth):

    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    positive_pred = prediction == 1
    negative_pred = prediction == 0
    #print(prediction, "\n", ground_truth)
   
    tp_rate = np.count_nonzero(ground_truth[positive_pred])
    fn_rate = np.count_nonzero(ground_truth[negative_pred])
    tn_rate = ground_truth[negative_pred].shape[0] - fn_rate
    fp_rate = ground_truth[positive_pred].shape[0] - tp_rate

    print(tp_rate, fn_rate, tn_rate, fp_rate)
    
    precision = tp_rate / (tp_rate + fp_rate) if tp_rate + fp_rate > 0 else 0
    recall = tp_rate / (tp_rate + fn_rate) if tp_rate + fn_rate > 0 else 0 
    accuracy = (tp_rate + tn_rate) / (tp_rate + tn_rate + fp_rate + fn_rate)
    f1 = 2 * precision * recall / (precision + recall) if tp_rate != 0 else 0
    #print(f1)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    #print("Pred ", prediction)
    #print("Truth ", ground_truth)
    return np.count_nonzero(np.equal(prediction, ground_truth)) / prediction.shape[0]
    return 0
