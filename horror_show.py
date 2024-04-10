#importing some of my functions
from index_generation import read_pickle
from index_generation import molecule_plotter
from index_generation import img_read_and_plot

index_smile_dict = read_pickle("index_smile_dict.pkl")
alkene_list = read_pickle("alkene_list.pkl")

print (len(alkene_list))

molecule_plotter(index_smile_dict["11231"], filename = "alkene_hopefully.png")

print(alkene_list)