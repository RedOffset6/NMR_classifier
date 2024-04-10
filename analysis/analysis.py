import numpy as np
import pandas as pd
import os
import openpyxl
import re
import matplotlib.pyplot as plt
import pickle as pkl

print(os.listdir("data"))

raw_data = pd.read_excel("data/accuracy.xlsx")
data = {}

experiment_dict = {}

for label in raw_data:

    #
    #  First doing some sting splitting that seperates out the componants of the label
    #

    #this gets the experiment type
    experiment_type = label.split(",")[1]

    #getting the fraction of the data that was used
    fraction = label.split("]")[0][1:]
    
    #getting the experiment type
    fragment1 = label.split("cnn")[0].split("]")[1]

    #getting the name of the functional group and the number of samples
    fg_and_samples = re.findall(r'[a-zA-Z]+|\d+', fragment1)
    functional_group = fg_and_samples[0]
    num_samples = fg_and_samples[1]
    #print (f"{functional_group} , {num_samples}")

    #extracting value from dataframe and series
    accuracy = raw_data[label][0]

    # Creating the inner dictionary if it doesn't exist
    if experiment_type not in experiment_dict:
        experiment_dict[experiment_type] = {}
    
    if functional_group not in experiment_dict[experiment_type]:
        experiment_dict[experiment_type][functional_group] = {}
    
    if fraction not in experiment_dict[experiment_type][functional_group]:
        experiment_dict[experiment_type][functional_group][fraction] = [accuracy, int(num_samples), label]

    # # Creating the dictionary for the current experiment type if it doesn't exist
    # if fraction not in experiment_dict[experiment_type]:
    #     experiment_dict[experiment_type][fraction] = {'accuracy': [], 'num_samples': []}
    
    # # Storing accuracy and number of samples
    # experiment_dict[experiment_type][fraction]['accuracy'].append(accuracy)
    # experiment_dict[experiment_type][fraction]['num_samples'].append(num_samples)

with open('data/confusion_matrices.pkl', 'rb') as f:
    confusion_matrices = pkl.load(f)


################################
#                              #
#      PLOTTING FUNCTIONS      #
#                              #
################################
        
def training_curve_plotter(experiment_type = "COMBINED", functional_group = "alkene"):
    num_datapoints = []
    accuracy = []
    highest_score = [0,"key", "EXPT TYPE"]
    for fraction in experiment_dict[experiment_type][functional_group]:
        sub_list = experiment_dict[experiment_type][functional_group][fraction]
        num_datapoints.append(sub_list[1])
        accuracy.append(sub_list[0])
        if accuracy>highest_score[0]:
            highest_score[0] = accuracy
            highest_score[1] = sub_list[2]
            highest_score[2] = experiment_type
    print (accuracy)
    print(num_datapoints)

    # Sort the data based on num_datapoints
    sorted_data = sorted(zip(num_datapoints, accuracy))
    num_datapoints, accuracy = zip(*sorted_data)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(num_datapoints, accuracy, marker='o', linestyle='-')
    plt.xlabel('Number of Training Data Points')
    plt.ylabel('Accuracy %')
    plt.title(f'{experiment_type} {functional_group} Training Curve')
    plt.grid(True)
    plt.savefig(f"plots/training_curves/{experiment_type}{functional_group}.jpeg")
    return highest_score
#plots which overlay the training curves of all experiments

def calculate_f1_score(confusion_matrix):
    # Extract true positives (TP), false positives (FP), and false negatives (FN) from confusion matrix
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]

    # Calculate precision
    precision = TP / (TP + FP)

    # Calculate recall
    recall = TP / (TP + FN)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

def calculate_matthews_score(confusion_matrix):
    # Extract true positives (TP), false positives (FP), and false negatives (FN) from confusion matrix
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[0][0]

    # Calculate Matthews correlation coefficient (MCC)
    numerator = (TP * TN) - (FP * FN)
    denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    matthews_score = numerator / denominator if denominator != 0 else 0

    return matthews_score

def f1_curve_plot(functional_group = "alkene"):
    plt.figure(figsize=(8, 6))
    experiments = ["HSQC", "HMBC", "COMBINED", "COSY", "ORIGINAL_COSY"]
    for experiment_type in experiments:
        num_datapoints = []
        f1_scores = []
        for fraction in experiment_dict[experiment_type][functional_group]:
            sub_list = experiment_dict[experiment_type][functional_group][fraction]
            cm = np.array(confusion_matrices[sub_list[2]])
            
            print(cm)

            # Calculate the F1 score
            f1 = calculate_f1_score(cm)
            #print(f"f1 = {f1}")

            if sub_list[0] > 53:
                f1_scores.append(f1)
                num_datapoints.append(sub_list[1])
                
            else:   
                print(sub_list[2])
                print(cm)
                print(f1)
        
        # print (accuracy)
        # print(num_datapoints)

        # Sort the data based on num_datapoints
        #print(f"{f1_scores}, {num_datapoints}")
        sorted_data = sorted(zip(num_datapoints, f1_scores))
        num_datapoints, f1_scores = zip(*sorted_data)

        plt.plot(num_datapoints, f1_scores, marker='o', linestyle='-', label = experiment_type)
        
    # Plotting
    plt.xlabel('Number of Training Data Points')
    plt.ylabel('F1 Score')
    plt.title(f'{functional_group.capitalize()} F1 Training Curves')
    plt.grid(True)
    plt.legend()
    # plt.show()

    plt.savefig(f"plots/f1_curves/{functional_group}.jpeg")

def matthews_curve_plot(functional_group = "alkene"):
    plt.figure(figsize=(8, 6))
    experiments = ["HSQC", "HMBC", "COMBINED", "COSY", "ORIGINAL_COSY"]
    highest_score = [0,"key", "EXPT TYPE"]
    for experiment_type in experiments:
        num_datapoints = []
        matthews_scores = []
        for fraction in experiment_dict[experiment_type][functional_group]:
            sub_list = experiment_dict[experiment_type][functional_group][fraction]
            cm = np.array(confusion_matrices[sub_list[2]])
            
            # print(cm)

            # Calculate the F1 score
            matthews_score = calculate_matthews_score(cm)
            #print(f"f1 = {f1}")
            #adds the highest matthews score to the list 
            if matthews_score > highest_score[0]:
                highest_score[0] = matthews_score
                highest_score[1] = sub_list[2]
                highest_score[2] = experiment_type

            if sub_list[0] > 53:
                matthews_scores.append(matthews_score)
                num_datapoints.append(sub_list[1])
                
            # else:   
            #     print(sub_list[2])
            #     print(cm)
            #     print(matthews_score)
        
        # print (accuracy)
        # print(num_datapoints)

        # Sort the data based on num_datapoints
        #print(f"{f1_scores}, {num_datapoints}")
        sorted_data = sorted(zip(num_datapoints, matthews_scores))
        num_datapoints, matthews_scores = zip(*sorted_data)

        plt.plot(num_datapoints, matthews_scores, marker='o', linestyle='-', label = experiment_type)
    return highest_score
        
    # Plotting
    plt.xlabel('Number of Training Data Points')
    plt.ylabel('Matthews Score')
    plt.title(f'{functional_group.capitalize()} Mattehws Score Training Curves')
    plt.grid(True)
    plt.legend()
    # plt.show()

    plt.savefig(f"plots/matthews_curves/{functional_group}.jpeg")
    


def multi_training_curve_plot(functional_group = "alkene"):
    plt.figure(figsize=(8, 6))
    experiments = ["HSQC", "HMBC", "COMBINED", "COSY", "ORIGINAL_COSY"]
    highest_score = [0,"key", "EXPT TYPE"]
    for experiment_type in experiments:
        num_datapoints = []
        accuracy = []
        for fraction in experiment_dict[experiment_type][functional_group]:
            sub_list = experiment_dict[experiment_type][functional_group][fraction]
            if sub_list[0] > 53:
                num_datapoints.append(sub_list[1])
                accuracy.append(sub_list[0])
            if sub_list[0]>highest_score[0]:
                highest_score[0] = sub_list[0]
                highest_score[1] = sub_list[2]
                highest_score[2] = experiment_type
            
        # print (accuracy)
        # print(num_datapoints)

        # Sort the data based on num_datapoints
        sorted_data = sorted(zip(num_datapoints, accuracy))
        num_datapoints, accuracy = zip(*sorted_data)

        plt.plot(num_datapoints, accuracy, marker='o', linestyle='-', label = experiment_type)


    # Plotting
    plt.axhline(y=98.7, color='r', linestyle='--', label = "Lower Error Bound")
    plt.xlabel('Number of Training Data Points')
    plt.ylabel('Accuracy %')
    plt.title(f'{functional_group.capitalize()} Training Curves')
    plt.grid(True)
    plt.legend()
    # plt.show()

    plt.savefig(f"plots/training_curves/{functional_group}.jpeg")
    return highest_score



#PLOTS TRAINING CURVES FOR ALL FGS AND EXPERIMENTS
functional_groups = ["ketone", "alcohol", "aldehyde", "imine", "alkene", "amide", "epoxide", "ether"]
functional_groups = ["aldehyde"]
highest_accs = []
for functional_group in functional_groups:
    acc = multi_training_curve_plot(functional_group=functional_group)
    highest_accs.append(acc)

print("PRINTING THE HIGHEST ACC SCORES")
print(highest_accs)
#PLOTS F1 TRAINING CURVES FOR ALL FGS AND EXPERIMENTS
# functional_groups = ["ketone", "alcohol", "aldehyde", "imine", "alkene", "amide", "epoxide", "ether"]
# for functional_group in functional_groups:
#     f1_curve_plot(functional_group=functional_group)
    
#PLOTS MATTHEWS CURVES
functional_groups = ["ketone", "alcohol", "aldehyde", "imine", "alkene", "amide", "epoxide", "ether"]
highest_scores = []
for functional_group in functional_groups:
    highest_score = matthews_curve_plot(functional_group=functional_group)
    highest_scores.append(highest_score)
print ("Printing best matthews scores")
print(highest_scores)