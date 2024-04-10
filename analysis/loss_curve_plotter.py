import numpy as np
import pandas as pd
import os
import openpyxl
import re
import matplotlib.pyplot as plt
import pickle as pkl


loss_curves = pd.read_excel("data/loss_curves.xlsx")

print(loss_curves)

best_models_list = [[0.8523767124835117, '[1.0]ketone14170cnn.pth,COMBINED', 'COMBINED'], [0.7454920281508797, '[1.0]alcohol36570cnn.pth,COMBINED', 'COMBINED'], [0.9988033069703293, '[0.25]aldehyde3350cnn.pth,HSQC', 'HSQC'], [0.8751119261809347, '[1.0]imine5720cnn.pth,COMBINED', 'COMBINED'], [0.9797384283079459, '[1.0]alkene14610cnn.pth,COMBINED', 'COMBINED'], [0.7987083846452094, '[0.5]amide6590cnn.pth,COMBINED', 'COMBINED'], [0.7910685403768397, '[1.0]epoxide9850cnn.pth,COMBINED', 'COMBINED'], [0.849491578102138, '[1.0]ether52930cnn.pth,COMBINED', 'COMBINED']]

model_indices = []
for model in best_models_list:
    model_indices.append(model[1])


# Iterate through each model index
for model_index in model_indices:
    # Get the loss curve from the DataFrame using the model index as the column name
    loss_curve = loss_curves[model_index]
        #this gets the experiment type
    experiment_type = model_index.split(",")[1]

    #getting the fraction of the data that was used
    fraction = model_index.split("]")[0][1:]
    
    #getting the experiment type
    fragment1 = model_index.split("cnn")[0].split("]")[1]

    #getting the name of the functional group and the number of samples
    fg_and_samples = re.findall(r'[a-zA-Z]+|\d+', fragment1)
    functional_group = fg_and_samples[0]
    # Plot the loss curve
#     plt.plot(loss_curve, label=functional_group.capitalize())

# # Set plot labels and legend
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Curves')

# # Add 1 to each epoch label
# current_ticks = plt.xticks()[0]
# new_ticks = [tick + 1 for tick in current_ticks]
# new_ticks = [int(tick) for tick in new_ticks]  # Convert to integers
# plt.xticks(current_ticks, new_ticks)
# plt.xlim([0,9])
# plt.legend()
# plt.show()

loss = [0.41291199411246016, 0.23059510044473502, 0.1771318229624409, 0.13276118998099493, 0.07807230710490953, 0.05496370418225419, 0.06360129591765332, 0.028131887489967058, 0.01988082348398205, 0.020248107821451616, 0.013263705961617226, 0.007691453242943594, 0.026544611266848598, 0.0015099066331529455, 0.0004598454659462116, 0.00016614209759835486, 9.648110136944028e-05, 5.6937803565052986e-05, 2.694560789308596e-05, 1.2416824485650247e-05, 6.4316579314783565e-06, 3.6111736527251673e-06, 2.380542447440075e-06, 1.7177650407674804e-06, 1.2190933853672825e-06, 8.237580293819119e-07, 6.020199362207915e-07, 4.2376848825979735e-07, 3.050009901640282e-07, 2.1809694789858575e-07, 1.598477366073723e-07, 1.1613475810238978e-07, 8.323782218267683e-08, 5.8416598051454655e-08, 4.147308495695378e-08, 2.794746279310512e-08, 1.8423241086542074e-08, 1.2483623918584825e-08, 8.753126354268967e-09, 5.793736626433368e-09, 4.001429748540347e-09, 3.0010724503772335e-09, 1.9798742070828413e-09, 1.4796954803633109e-09, 1.0420390952601014e-09, 7.711089296229296e-10, 5.835418954574097e-10, 5.001787857243907e-10, 3.542933064253801e-10, 3.1261174534783274e-10, 3.334525227810875e-10, 1.8756704658759586e-10, 1.4588548240452951e-10, 1.2504470186575583e-10, 4.168156728858528e-11, 4.168156728858528e-11, 2.084078364429264e-11, 1.2504470186575583e-10, 0.0, 0.0, 0.2903434820905306, 0.08341559245821169, 0.004477203108817059, 0.0006449507937234304, 0.0003803401456631684, 0.00027636372847776933, 0.00023472675905704738, 0.00020881638766969955, 0.035502548430777096, 0.07500399071299442, 0.012891684011313772, 0.003063832709323901, 0.0018834741634325084, 0.0016815258082810958, 0.014147050255009185, 0.0028497169895694073, 0.00015044322638554366, 0.00012512382057723562, 0.00011544045062001379, 0.00010770038624137961, 0.00010071012476516321, 9.469357100224627e-05, 8.935035432004463e-05, 8.412390159936049e-05, 7.860137426790865e-05, 7.373298090577892e-05, 6.897116319585624e-05, 6.414957058581456e-05, 5.934864229521882e-05, 9.271519396316605e-05, 4.797199895679238e-05, 4.347934978090482e-05, 3.9381225941982634e-05, 3.528474534373063e-05, 3.106126119547712e-05, 2.7146694789171455e-05, 2.3553676288267977e-05, 2.02765417467086e-05, 1.6655605593993984e-05, 1.3658251853283292e-05]
epochs = []
for i, _ in enumerate(loss):
    epochs.append(i)
print(epochs)
print(loss)
plt.plot(epochs,loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.show()