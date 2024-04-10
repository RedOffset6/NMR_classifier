#importing some of my functions
from index_generation import read_pickle
from index_generation import molecule_plotter
from index_generation import img_read_and_plot

index_smile_dict = read_pickle("index_smile_dict.pkl")


bad_ketones = [
    "100073",
    "100269",
    "100168",
    "100269",
    "100168",
    "100073",
    "100083",
    "100090",
    "100266",
    "100263"
]



bad_amides = [
    "100082",
    "100082",
    "10010",
    "100148",
    "100148",
    "100169",
    "100094",
    "10010",
    "100082",
    "100074"
]

bad_alcohols  =  [
    "100029",
    "100008",
    "100022",
    "100014",
    "100022",
    "100008",
    "100011",
    "100014",
    "100029",
    "100005"]

for index in bad_alcohols:

    molecule_plotter(index_smile_dict[index], filename = f"molecule_plot/bad_alcohols/molecule_{index}.png")