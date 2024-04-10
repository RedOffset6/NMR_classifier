from index_generation import read_pickle, molecule_plotter

index_smile_dict = read_pickle("index_smile_dict.pkl")

#print(index_smile_dict)

molecule_plotter(index_smile_dict["99994"], filename = "smileinvestigation.png")

print(f'Smile string = {index_smile_dict["99994"]}')

#print(index_smile_dict["99994"])



