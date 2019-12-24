import numpy as NP
import pandas as PD
from matplotlib import pyplot as Plot
from pandas import DataFrame as DF
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.cluster import AgglomerativeClustering as Cluster
from scipy.cluster import hierarchy


"""Global variable definition"""
SMILES_Path = "C:\\Workbench\\Computer Science\\Python Programming\\Carbon-Nitrogen-Coupling\\SMILES_material\\"
Distance_Path = "C:\\Workbench\\Computer Science\\Python Programming\\Carbon-Nitrogen-Coupling\\distance\\"


"""Reading SMILES material and calculate the distance of molecules"""
def ReadSMILES():
    global SMILES_Path, Distance_Path
    global Molecule_List, Molecule_Dice_Distance_Table, Molecule_Tanimoto_Distance_Table
    Fingerprint_List_Temp = []
    Molecule_List = PD.read_csv(SMILES_Path + "list.csv", header = None)
    print("Reading SMILES ... Done!")

    print("Now calculating the distance of molecules ...")
    Molecule_List.rename(columns = {0: "Name"}, inplace = True) 
    Molecule_Dice_Distance_Table = Molecule_List.copy()
    Molecule_Tanimoto_Distance_Table = Molecule_List.copy()
    for Molecule_Name in Molecule_List["Name"]:    # The first column is the name of molecules
        SMILES_file = open(SMILES_Path + Molecule_Name + ".smi", mode = "r")
        Molecule_Object = AllChem.MolFromSmiles(SMILES_file.readline())
        SMILES_file.close()
        Fingerprint_List_Temp.append(AllChem.GetMorganFingerprintAsBitVect(Molecule_Object, radius = 2, nBits = 2048))
    Molecule_List[1] = Fingerprint_List_Temp
    Molecule_List.rename(columns = {1: "Fingerprint"}, inplace = True) 
    for Ref_Molecule in Molecule_List.index:
        Ref_Fingerprint = Molecule_List.at[Ref_Molecule, "Fingerprint"]
        Col_Name = Molecule_List.at[Ref_Molecule, "Name"]
        Dice_Dist_List = []
        Tanimoto_Dist_List = []
        for Cmp_Molecule in Molecule_List.index:
            Cmp_Fingerprint = Molecule_List.at[Cmp_Molecule, "Fingerprint"]
            Dice_Dist = DataStructs.DiceSimilarity(Ref_Fingerprint, Cmp_Fingerprint, returnDistance = True)
            Tanimoto_Dist = DataStructs.TanimotoSimilarity(Ref_Fingerprint, Cmp_Fingerprint, returnDistance = True)
            Dice_Dist_List.append(Dice_Dist)
            Tanimoto_Dist_List.append(Tanimoto_Dist)
        Molecule_Dice_Distance_Table[Col_Name] = Dice_Dist_List
        Molecule_Tanimoto_Distance_Table[Col_Name] = Tanimoto_Dist_List

    print("Now printing the distance of molecules ...")
    Molecule_Dice_Distance_Table.to_csv(Distance_Path + "dice_distance.csv", index = False)
    Molecule_Tanimoto_Distance_Table.to_csv(Distance_Path + "tanimoto_distance.csv", index = False)
    print("Calculation done!")


"""Cluster analysis"""
def WardClusterAnalysis():
    global Distance_Path
    global Molecule_Dice_Distance_Table, Molecule_Tanimoto_Distance_Table

    print("Now making cluster analysis ...")
    Ward_Dice = Cluster(n_clusters = 7)
    Ward_Dice.fit(Molecule_Dice_Distance_Table.drop("Name", axis = 1))
    print("According to dice distance, the labels of different molecules are: ")
    print(Ward_Dice.labels_)
    Molecule_Label_Dice = PD.concat([Molecule_Dice_Distance_Table["Name"], PD.Series(Ward_Dice.labels_)], axis = 1)
    Molecule_Label_Dice.columns = ["Name", "Label"]
    Molecule_Label_Dice.to_csv(Distance_Path + "Ward_method_labels_dice.csv", index = False)
#    OutputName = Distance_Path + "Ward_method_labels_dice.dat"
#    Output = open(OutputName, mode = "w")
#    Output.write(str(Ward_Dice.labels_))
#    Output.close()
    Ward_Tanimoto = Cluster(n_clusters = 7)
    Ward_Tanimoto.fit(Molecule_Tanimoto_Distance_Table.drop("Name", axis = 1))
    print("According to Tanimoto distance, the labels of different molecules are: ")
    print(Ward_Tanimoto.labels_)
    Molecule_Label_Tanimoto = PD.concat([Molecule_Tanimoto_Distance_Table["Name"], PD.Series(Ward_Tanimoto.labels_)], axis = 1)
    Molecule_Label_Tanimoto.columns = ["Name", "Label"]
    Molecule_Label_Tanimoto.to_csv(Distance_Path + "Ward_method_labels_Tanimoto.csv", index = False)
#    OutputName = Distance_Path + "Ward_method_labels_Tanimoto.dat"
#    Output = open(OutputName, mode = "w")
#    Output.write(str(Ward_Dice.labels_))
#    Output.close()


    print("Now drawing dendrogram for Ward method ...")
    linked_array_dice = hierarchy.ward(Molecule_Dice_Distance_Table.drop("Name", axis = 1))
    hierarchy.dendrogram(linked_array_dice)
    Plot.xlabel("Compounds", fontsize=12)
    Plot.xticks([])
    Plot.ylabel("Cluster Distance", fontsize=12)
    Plot.title('Dendrogram for Ward Method', fontsize=14)
    Plot.savefig(Distance_Path + "dendrogram_using_dice_distance.png", dpi = 300, format = "png")
    Plot.cla()
    linked_array_tanimoto = hierarchy.ward(Molecule_Tanimoto_Distance_Table.drop("Name", axis = 1))
    hierarchy.dendrogram(linked_array_tanimoto)
    Plot.xlabel("Compounds", fontsize=12)
    Plot.xticks([])
    Plot.ylabel("Cluster Distance", fontsize=12)
    Plot.title('Dendrogram for Ward Method', fontsize=14)
    Plot.savefig(Distance_Path + "dendrogram_using_tanimoto_distance.png", dpi = 300, format = "png")
    print("Drawing done!")


"""Main program"""
def main():
    ReadSMILES()
    WardClusterAnalysis()


if __name__ == '__main__':
    main()

