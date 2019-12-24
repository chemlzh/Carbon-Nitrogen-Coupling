import pandas as PD
import numpy as NP
from matplotlib import pyplot as Plot
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import RidgeCV as RCV
from sklearn.linear_model import BayesianRidge as BR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.svm import LinearSVR as LSVR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.pipeline import Pipeline as PL
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2


"""Global variable definition"""
Data_Path = "C:\\Workbench\\Computer Science\\Python Programming\\Carbon-Nitrogen-Coupling\\descriptor_and_yield_data\\"
Result_Path = "C:\\Workbench\\Computer Science\\Python Programming\\Carbon-Nitrogen-Coupling\\training_result\\"

Aryl_Halide_DM_1 = NP.array([0.770806, 0.754193, 1.059432], dtype = "float64") 
Aryl_Halide_DM_2 = NP.array([2.894904, 2.97534,  2.775967, 2.32974, 2.410215, 2.177715], dtype = "float64") 
Aryl_Halide_DM_3 = NP.array([3.519084, 3.579429, 3.355629, 2.089781, 2.086875, 1.988625], dtype = "float64") 

Additive_DM_1 = NP.array([3.059464, 2.980515, 3.387206, 3.203762, 3.44567, 4.695918, 
                          4.848261, 3.973089, 2.516823, 2.703396, 3.85783, 3.719695, 4.564313], dtype = "float64")
Additive_DM_2 = NP.array([2.94125, 2.752454, 3.268545, 3.94018, 3.210447, 3.198581, 
                          3.184221, 3.036368, 3.974105], dtype = "float64")


"""Machine learning models"""
pipeKNN = PL([("kNearestNeighbors", KNR(n_neighbors = 8))])
pipeSVM = PL([("SVM", SVR(cache_size = 1024, gamma = "scale"))])
pipeLSVM = PL([("LinearSVM", LSVR())])
pipeRCV = PL([("RidgeCV", RCV(alphas = NP.logspace(-6, 2, 5000)))])
pipeBR = PL([("BayesianRidge", BR())])
pipeMLP = PL([("MultiLayerPerceptron", MLPR(hidden_layer_sizes = (100), activation= "logistic", solver = "adam", 
                                            max_iter = 10000, random_state = 1551))])
pipeRFR = PL([("RandomForest", RFR(n_estimators = 500))])
pipeDTR = PL([("DecisionTree", DTR(max_depth = 15, min_samples_leaf = 3, random_state = 1551))])


"""Reading data"""
def ReadData():
    global Data_Path, Data_Source
    Data_Source = PD.read_csv(Data_Path + "unscaled_summary_with_yield.csv", float_precision = 'round_trip')
    Data_Source.fillna(0, inplace = True)
    print("Reading descriptor and yield data ... Done!")


"""Standardizing data"""
def StandardizeData():
    global X, Y, Data_Source
    X = (Data_Source.drop(["yield"], axis=1)).values
    Y = Data_Source["yield"].values
    ScalerX = Scaler();   ScalerX.fit(X, NP.all(Y));
    ScalerX = ScalerX.transform(X, NP.all(Y))
    print("Standardizing ... Done!")


"""Spliting data randomly"""
def SplitDataRandomly():
    global Data_Path, Data_Source
    global X, Y, TrainData_X, TrainData_Y, TestData_X, TestData_Y
    StandardizeData()
    TrainData_X, TestData_X, TrainData_Y, TestData_Y = train_test_split(X, Y, test_size = 1/3, random_state = 1551)
    print("Spliting data ... Done!")
    

"""Spliting data into groups, which have similar aryl halides"""
def SplitData_SimilarArylHalide():
    global Data_Source
    global Aryl_Halide_DM_1, Aryl_Halide_DM_2, Aryl_Halide_DM_3
    global X, Y, TrainData_X, TrainData_Y, TestData_X, TestData_Y
    Aryl_Halide_DM_Train = NP.hstack((Aryl_Halide_DM_1, Aryl_Halide_DM_2))
    Aryl_Halide_DM_Test = Aryl_Halide_DM_3.copy()
    StandardizeData()
    TrainLine = Data_Source[Data_Source["aryl_halide_dipole_moment"].isin(Aryl_Halide_DM_Train)].index.values
    TestLine = Data_Source[Data_Source["aryl_halide_dipole_moment"].isin(Aryl_Halide_DM_Test)].index.values
    TrainData_X = X[TrainLine].copy()
    TrainData_Y = Y[TrainLine].copy()
    TestData_X = X[TestLine].copy()
    TestData_Y = Y[TestLine].copy()
    print("Spliting data ... Done!")


"""Spliting data into groups, which have randomly selected aryl halides"""
def SplitData_RandomlySelectedArylHalide():
    global Data_Source
    global Aryl_Halide_DM_1, Aryl_Halide_DM_2, Aryl_Halide_DM_3
    global X, Y, TrainData_X, TrainData_Y, TestData_X, TestData_Y
    StandardizeData()
    TrainLine = NP.array([], dtype = "int64") 
    TestLine = NP.array([], dtype = "int64") 
    for DM in Aryl_Halide_DM_1:
        TempLine = Data_Source[Data_Source["aryl_halide_dipole_moment"].isin([DM])].index.values
        LineCount = TempLine.size
        RandomArray = NP.arange(TempLine.shape[0])
        NP.random.shuffle(RandomArray)
        TrainLine = NP.hstack((TrainLine, TempLine[RandomArray[0: LineCount * 2 // 3]]))
        TestLine = NP.hstack((TestLine, TempLine[RandomArray[LineCount * 2 // 3: LineCount]]))
    for DM in Aryl_Halide_DM_2:
        TempLine = Data_Source[Data_Source["aryl_halide_dipole_moment"].isin([DM])].index.values
        LineCount = TempLine.size
        RandomArray = NP.arange(TempLine.shape[0])
        NP.random.shuffle(RandomArray)
        TrainLine = NP.hstack((TrainLine, TempLine[RandomArray[0: LineCount * 2 // 3]]))
        TestLine = NP.hstack((TestLine, TempLine[RandomArray[LineCount * 2 // 3: LineCount]]))
    for DM in Aryl_Halide_DM_3:
        TempLine = Data_Source[Data_Source["aryl_halide_dipole_moment"].isin([DM])].index.values
        LineCount = TempLine.size
        RandomArray = NP.arange(TempLine.shape[0])
        NP.random.shuffle(RandomArray)
        TrainLine = NP.hstack((TrainLine, TempLine[RandomArray[0: LineCount * 2 // 3]]))
        TestLine = NP.hstack((TestLine, TempLine[RandomArray[LineCount * 2 // 3: LineCount]]))
    TrainData_X = X[TrainLine].copy()
    TrainData_Y = Y[TrainLine].copy()
    TestData_X = X[TestLine].copy()
    TestData_Y = Y[TestLine].copy()
    print("Spliting data ... Done!")


"""Spliting Data into groups, which have similar additives"""
def SplitData_SimilarAdditive():
    global Data_Source
    global Additive_DM_1, Additive_DM_2
    global X, Y, TrainData_X, TrainData_Y, TestData_X, TestData_Y
    Additive_DM_Train = Additive_DM_1
    Additive_DM_Test = Additive_DM_2
    StandardizeData()
    TrainLine = Data_Source[Data_Source["additive_dipole_moment"].isin(Additive_DM_Train)].index.values
    TestLine = Data_Source[Data_Source["additive_dipole_moment"].isin(Additive_DM_Test)].index.values
    TrainData_X = X[TrainLine].copy()
    TrainData_Y = Y[TrainLine].copy()
    TestData_X = X[TestLine].copy()
    TestData_Y = Y[TestLine].copy()
    print("Spliting data ... Done!")


"""Spliting data into groups, which have randomly selected additives"""
def SplitData_RandomlySelectedAdditive():
    global Data_Source
    global Additive_DM_1, Additive_DM_2
    global X, Y, TrainData_X, TrainData_Y, TestData_X, TestData_Y
    StandardizeData()
    TrainLine = NP.array([], dtype = "int64") 
    TestLine = NP.array([], dtype = "int64") 
    for DM in Additive_DM_1:
        TempLine = Data_Source[Data_Source["additive_dipole_moment"].isin([DM])].index.values
        LineCount = TempLine.size
        RandomArray = NP.arange(TempLine.shape[0])
        NP.random.shuffle(RandomArray)
        TrainLine = NP.hstack((TrainLine, TempLine[RandomArray[0: LineCount * 2 // 3]]))
        TestLine = NP.hstack((TestLine, TempLine[RandomArray[LineCount * 2 // 3: LineCount]]))
    for DM in Additive_DM_2:
        TempLine = Data_Source[Data_Source["additive_dipole_moment"].isin([DM])].index.values
        LineCount = TempLine.size
        RandomArray = NP.arange(TempLine.shape[0])
        NP.random.shuffle(RandomArray)
        TrainLine = NP.hstack((TrainLine, TempLine[RandomArray[0: LineCount * 2 // 3]]))
        TestLine = NP.hstack((TestLine, TempLine[RandomArray[LineCount * 2 // 3: LineCount]]))
    TrainData_X = X[TrainLine].copy()
    TrainData_Y = Y[TrainLine].copy()
    TestData_X = X[TestLine].copy()
    TestData_Y = Y[TestLine].copy()
    print("Spliting data ... Done!")


"""Model training and analysis"""
def TrainingAndAnalysis(DataSourceName):
    global pipeKNN, pipeSVM, pipeLSVM, pipeRCV, pipeBR, pipeMLP, pipeRFR, pipeDTR
    global TrainData_X, TrainData_Y, TestData_X, TestData_Y
    pipeKNN.fit(TrainData_X, TrainData_Y)
    ModelAssessment(pipeKNN, "kNearestNeighbors", DataSourceName)
    pipeSVM.fit(TrainData_X, TrainData_Y)
    ModelAssessment(pipeSVM, "SVM", DataSourceName)
    pipeLSVM.fit(TrainData_X, TrainData_Y)
    ModelAssessment(pipeLSVM, "LinearSVM", DataSourceName)
    pipeRCV.fit(TrainData_X, TrainData_Y)
    ModelAssessment(pipeRCV, "RidgeCV", DataSourceName)
    pipeBR.fit(TrainData_X, TrainData_Y)
    ModelAssessment(pipeBR, "BayesianRidge", DataSourceName)
    pipeMLP.fit(TrainData_X, TrainData_Y)
    ModelAssessment(pipeMLP, "MultiLayerPerceptron", DataSourceName)
    pipeRFR.fit(TrainData_X, TrainData_Y)
    ModelAssessment(pipeRFR, "RandomForest", DataSourceName)
    pipeDTR.fit(TrainData_X, TrainData_Y)
    ModelAssessment(pipeDTR, "DecisionTree", DataSourceName)
    print("Training and Analysing ... Done!")


"""Model assessment"""
def ModelAssessment(pipe, pipeName, DataSourceName):
    global TrainData_X, TrainData_Y, TestData_X, TestData_Y

    TrainPrediction_Y = pipe.predict(TrainData_X)
    Train_R2 = R2(TrainData_Y, TrainPrediction_Y)
    Train_RMS_score = NP.sqrt(MSE(TrainData_Y, TrainPrediction_Y))
#    print(Train_R2, Train_RMS_score)
    Plot.plot(TrainPrediction_Y, TrainData_Y, "ob") 
    Plot.title("Train R^2 = " + str(Train_R2) + ", Train RMSD = " + str(Train_RMS_score))
    Plot.savefig(Result_Path + DataSourceName + "_" + pipeName + "_Training.png", dpi = 300, format = "png")
    Plot.cla()

    TestPrediction_Y = pipe.predict(TestData_X)
    Test_R2 = R2(TestData_Y, TestPrediction_Y)
    Test_RMS_score = NP.sqrt(MSE(TestData_Y, TestPrediction_Y))
#    print(Test_R2, Test_RMS_score)
    Plot.plot(TestPrediction_Y, TestData_Y, "ob") 
    Plot.title("Test R^2 = " + str(Test_R2) + ", Test RMSD = " + str(Test_RMS_score))
    Plot.savefig(Result_Path + DataSourceName + "_" + pipeName + "_Testing.png", dpi = 300, format = "png")
    Plot.cla()


"""First workflow"""
def Workflow_1():
    SplitDataRandomly()
    TrainingAndAnalysis("Random_Data")


"""Second workflow"""
def Workflow_2():
    SplitData_SimilarArylHalide()
    TrainingAndAnalysis("Data_With_Similar_Aryl_Halides")


"""Third workflow"""
def Workflow_3():
    SplitData_SimilarAdditive()
    TrainingAndAnalysis("Data_With_Similar_Additives")


"""Fourth workflow"""
def Workflow_4():
    SplitData_RandomlySelectedArylHalide()
    TrainingAndAnalysis("Stratified_Sampling_For_Aryl_Halides")


"""Fifth workflow"""
def Workflow_5():
    SplitData_RandomlySelectedAdditive()
    TrainingAndAnalysis("Stratified_Sampling_For_Additives")


"""Main program"""
def main():
    ReadData()
#    Workflow_1()
#    Workflow_2()
#    Workflow_3()
#    Workflow_4()
    Workflow_5()


if __name__ == '__main__':
    main()

