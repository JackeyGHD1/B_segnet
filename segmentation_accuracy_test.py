import numpy as np

# settings
import settings
FLAGS = settings.FLAGS

NUM_CLASSES = 3

labels = ("0", "1", "2")


def global_accuracy(annotations, predicts, answer):
    accuracy_list = []
    print("*************global accuracy*************")
    for annotation, predict in zip(annotations, predicts):
        collect = len(np.where((annotation==predict)==True)[0])
        accuracy = float(collect) / annotation.size
        accuracy_list.append(accuracy)
    return answer == round(sum(accuracy_list)/len(accuracy_list), 2)

def class_average_accuracy(annotations, predicts, answer):
    class_accuracy_list = [[] for x in range(NUM_CLASSES)]
    print("*************class average accuracy*************")
    for annotation, predict in zip(annotations, predicts):
        for class_index in range(NUM_CLASSES):
            TP = true_positive(annotation, predict, class_index)
            num_true_labels = len(np.where(annotation == class_index)[0])
            if num_true_labels == 0:
                class_accuracy_list[class_index].append("NULL")
            else:
                class_accuracy_list[class_index].append(float(TP) / num_true_labels)
    total_accuracy = 0
    null_count = 0
    for class_accuracy, label in zip(class_accuracy_list, labels):
        if class_accuracy.count("NULL") == len(class_accuracy):
            null_count += 1
        else:
            while "NULL" in class_accuracy:
                class_accuracy.remove("NULL")
            total_accuracy += sum(class_accuracy)/(len(class_accuracy))
    return answer == round(total_accuracy/(NUM_CLASSES - null_count), 2)


def mean_interaction_over_union(annotations, predicts, answer):
    class_accuracy_list = [[] for x in range(NUM_CLASSES)]
    print("*************mean interaction over union accuracy**************")
    for annotation, predict in zip(annotations, predicts):
        for class_index in range(NUM_CLASSES):
            num_true_labels = len(np.where(annotation==class_index)[0])
            TP = true_positive(annotation, predict, class_index)
            FP = false_positive(annotation, predict, class_index)
            FN = false_negative(annotation, predict, class_index)
            if num_true_labels == 0:
                class_accuracy_list[class_index].append("NULL")
            else:
                class_accuracy_list[class_index].append(float(TP)/(TP+FP+FN))
    total_accuracy = 0
    null_count = 0
    for class_accuracy, label in zip(class_accuracy_list, labels):
        if class_accuracy.count("NULL") == len(class_accuracy):
            null_count += 1
        else:
            while "NULL" in class_accuracy:
                class_accuracy.remove("NULL")
            total_accuracy += sum(class_accuracy)/len(class_accuracy)
    return answer == round(total_accuracy/(NUM_CLASSES - null_count), 2)


def true_positive(annotation, predict, class_index):
    annotation = annotation.flatten()
    predict = predict.flatten()
    TP = 0
    for i in range(len(annotation)):
        if annotation[i] == predict[i] == class_index:
            TP += 1
    return TP

def false_positive(annotation, predict, class_index):
    annotation = annotation.flatten()
    predict = predict.flatten()
    FP = 0
    for i in range(len(annotation)):
        if predict[i] == class_index and predict[i] != annotation[i]:
            FP += 1
    return FP

def false_negative(annotation, predict, class_index):
    annotation = annotation.flatten()
    predict = predict.flatten()
    FN = 0
    for i in range(len(annotation)):
        if annotation[i] == class_index and annotation[i] != predict[i]:
            FN += 1
    return FN

if __name__ == '__main__':
    annotations = np.array([[0,1,2,2,2]])
    predicts = np.array([[0,0,0,2,2]])
    g_answer = 0.60
    c_answer = 0.56
    m_answer = 0.33
    print global_accuracy(annotations, predicts, g_answer)
    print class_average_accuracy(annotations, predicts, c_answer)
    print mean_interaction_over_union(annotations, predicts, m_answer)


