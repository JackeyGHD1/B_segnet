import numpy as np

# settings
import settings
FLAGS = settings.FLAGS

NUM_CLASSES = FLAGS.num_classes

labels = ("Sky", "Building", "Column_Pole", "Road", "Sidewalk", "Tree", "SignSymbol", "Fence", "Car"
    , "Pedestrian", "Bicyclist", "LaneMkgsDriv", "Void")


def global_accuracy(annotations, predicts, f):
    accuracy_list = []
    print("*************global accuracy*************")
    print >> f, "*************global accuracy*************"
    for annotation, predict in zip(annotations, predicts):
        collect = len(np.where((annotation==predict)==True)[0])
        accuracy = float(collect) / annotation.size
        accuracy_list.append(accuracy)
    print("[global accuracy]: %f" % (sum(accuracy_list)/len(accuracy_list)))
    print >> f, "[global accuracy]: %f" % (sum(accuracy_list)/len(accuracy_list))

def class_average_accuracy(annotations, predicts, f):
    class_accuracy_list = [[] for x in range(NUM_CLASSES)]
    print("*************class average accuracy*************")
    print >> f, "*************class average accuracy*************"
    for annotation, predict in zip(annotations, predicts):
        for class_index in range(NUM_CLASSES):
            TP = true_positive(annotation, predict, class_index)
            num_true_labels = len(np.where(annotation==class_index)[0])
            if num_true_labels == 0:
                class_accuracy_list[class_index].append("NULL")
            else:
                class_accuracy_list[class_index].append(float(TP)/num_true_labels)
    total_accuracy = 0
    null_count = 0
    for class_accuracy, label in zip(class_accuracy_list, labels):
        if class_accuracy.count("NULL") == len(class_accuracy):
            print("%s: None" % label)
            print >> f, "%s: None" % label
            null_count += 1
        else:
            while "NULL" in class_accuracy:
                class_accuracy.remove("NULL")
            total_accuracy += sum(class_accuracy)/(len(class_accuracy))
            print("%s: %f" % (label, (sum(class_accuracy)/len(class_accuracy))))
            print >> f, "%s: %f" % (label, (sum(class_accuracy)/len(class_accuracy)))
    print("[class average accuracy]: %f" % (total_accuracy/(NUM_CLASSES - null_count)))
    print >> f, "[class average accuracy]: %f" % (total_accuracy/(NUM_CLASSES - null_count))


def mean_interaction_over_union(annotations, predicts, f):
    class_accuracy_list = [[] for x in range(NUM_CLASSES)]
    print("*************mean interaction over union accuracy*************")
    print >> f, "*************mean interaction over union accuracy*************"
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
            print("%s: None" % label)
            print >> f, "%s: None" % label
            null_count += 1
        else:
            while "NULL" in class_accuracy:
                class_accuracy.remove("NULL")
            total_accuracy += sum(class_accuracy)/len(class_accuracy)
            print("%s: %f" % (label, (sum(class_accuracy)/len(class_accuracy))))
            print >> f, "%s: %f" % (label, (sum(class_accuracy)/len(class_accuracy)))
    print("[mean interaction over union accuracy]: %f" % (total_accuracy/(NUM_CLASSES - null_count)))
    print >> f, "[mean interaction over union accuracy]: %f" % (total_accuracy/(NUM_CLASSES - null_count))

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





