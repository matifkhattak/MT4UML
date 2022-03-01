#https://www.kaggle.com/hellbuoy/online-retail-k-means-hierarchical-clustering

# import required libraries for dataframe and visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# import required libraries for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import SilhouetteVisualizer

def loadData():
    #####Step 1 : Reading and Understanding Data########
    # Reading the data on which analysis needs to be done
    retail = pd.read_csv('OnlineRetail.csv', sep=",", encoding="ISO-8859-1", header=0)
    retail.head()
    retail.shape
    retail.info()
    retail.describe()
    return retail

def dataCleaning(retail):
    ############Step 2 : Data Cleansing####################
    # Calculating the Missing Values % contribution in DF
    df_null = round(100*(retail.isnull().sum())/len(retail), 2)
    df_null
    # Droping rows having missing values
    retail = retail.dropna()
    retail.shape
    # Changing the datatype of Customer Id as per Business understanding
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    return retail

def dataPreparation(retail,MRType):
    ######Step 3 : Data Preparation########
    # New Attribute : Monetary
    retail['Amount'] = retail['Quantity']*retail['UnitPrice']
    rfm_m = retail.groupby('CustomerID')['Amount'].sum()
    rfm_m = rfm_m.reset_index()
    rfm_m.head()
    # New Attribute : Frequency
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
    rfm_f = rfm_f.reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    rfm_f.head()
    # Merging the two dfs
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm.head()
    # New Attribute : Recency
    # Convert to datetime to proper datatype
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%m/%d/%Y %H:%M')
    # Compute the maximum date to know the last transaction date
    max_date = max(retail['InvoiceDate'])
    max_date
    # Compute the difference between max date and transaction date
    retail['Diff'] = max_date - retail['InvoiceDate']
    retail.head()
    # Compute last transaction date to get the recency of customers
    rfm_p = retail.groupby('CustomerID')['Diff'].min()
    rfm_p = rfm_p.reset_index()
    rfm_p.head()
    # Extract number of days only
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm_p.head()
    # Merge tha dataframes to get the final RFM dataframe
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
    rfm.head()
    # Outlier Analysis of Amount Frequency and Recency
    attributes = ['Amount','Frequency','Recency']
    plt.rcParams['figure.figsize'] = [10,8]
    sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
    plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
    plt.ylabel("Range", fontweight = 'bold')
    plt.xlabel("Attributes", fontweight = 'bold')
    # Removing (statistical) outliers for Amount
    Q1 = rfm.Amount.quantile(0.05)
    Q3 = rfm.Amount.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

    # Removing (statistical) outliers for Recency
    Q1 = rfm.Recency.quantile(0.05)
    Q3 = rfm.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

    # Removing (statistical) outliers for Frequency
    Q1 = rfm.Frequency.quantile(0.05)
    Q3 = rfm.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]

    # Rescaling the attributes
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    # Instantiate
    scaler = StandardScaler()
    # fit_transform
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled.shape

    if(MRType=="MR2"):
        print("MR2-Followup")
        rfm_df_scaled = scaler.fit_transform(rfm_df_scaled)

    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

    if (MRType == "MR3"):
        print("MR3-Followup")
        rfm_df_scaled["Amount1"] = rfm_df_scaled["Amount"]
        rfm_df_scaled["Frequency1"] = rfm_df_scaled["Frequency"]
        rfm_df_scaled["Recency1"] = rfm_df_scaled["Recency"]

    return rfm, rfm_df_scaled

def buildModel(rfm, rfm_df_scaled,MRType,valueToIncrement=None):
    ######Step 4 : Building the Model######

    ##Finding the Optimal Number of Clusters
    # Elbow-curve/SSD Method

    #ssd = []
    #range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    #fig, ax = plt.subplots(4, 2, figsize=(15, 8))
    #for num_clusters in range_n_clusters:
    #    agglomerativeModel = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='average')
        # silhouette visualizer
        #q, mod = divmod(num_clusters, 2)
        #visualizer = SilhouetteVisualizer(agglomerativeModel, colors='yellowbrick', ax=ax[q-1][mod])
        # end
   #     agglomerativeModel.fit(rfm_df_scaled)
        #ssd.append(agglomerativeModel.inertia_)

        #visualizer.fit(rfm_df_scaled)  # Fit the data to the visualizer
    #visualizer.show()

        # plot the SSDs for each n_clusters
    #plt.plot(ssd)

    # Silhouette analysis Method
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        # intialise kmeans
        agglomerativeModel = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='average')
        agglomerativeModel.fit(rfm_df_scaled)
        cluster_labels = agglomerativeModel.labels_
        # silhouette score
        silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
        print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

    ## Single linkage:
    #mergings = linkage(rfm_df_scaled, method="single", metric='euclidean')
    #dendrogram(mergings)
    #plt.show()
    ## Complete linkage
    #mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
    #dendrogram(mergings)
    #plt.show()

    ## Average linkage
    mergings = linkage(rfm_df_scaled, method="average", metric='euclidean')
    dendrogram(mergings)
    #plt.show()

    #Cutting the Dendrogram based on K
    # 3 clusters
    #cluster_label = cut_tree(mergings, n_clusters=3).reshape(-1, )
    clusterModel = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
    clusterModel.fit_predict(rfm_df_scaled)
    # Assign cluster labels
    rfm['Cluster_Id'] = clusterModel.labels_
    ###rfm['Cluster_Label'] = cluster_label
    # Plot Cluster Id vs Amount
    #sns.boxplot(x='Cluster_Id', y='Amount', data=rfm)
    # Plot Cluster Id vs Frequency
    #sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)
    # Plot Cluster Id vs Recency
    #sns.boxplot(x='Cluster_Id', y='Recency', data=rfm)

    return rfm,rfm_df_scaled,clusterModel

def printModelInfo(clusterModel,rfm, rfm_df_scaled):
    print("")

def hierarchicalAlgorithm(MRType = "", rfm_WithClassLabels = None, rowIndex = None, valueToIncrement = None):
    #####Step 1 : Reading and Understanding Data########
    retail = loadData()

    ############Step 2 : Data Cleansing#################
    retail = dataCleaning(retail)

    ######Step 3 : Data Preparation########
    rfm, rfm_df_scaled = dataPreparation(retail,MRType)

    ######Step 4 : Building the Model######
    ######Step 4 : Building the Model######
    if (MRType == "MR1"):
        print("MR1-Followup")

        # MR1.1: adding single point
        ##duplicateRow = {'Amount': rfm_df_scaled['Amount'].iloc[rowIndex],'Frequency': rfm_df_scaled['Frequency'].iloc[rowIndex],'Recency':  rfm_df_scaled['Recency'].iloc[rowIndex]} # dynamically preparing the datarow, so that we can identify the row/instance no that if added, the MR will be violated

        #duplicateRow = {'Amount': rfm_df_scaled['Amount'].iloc[4],
        #                'Frequency': rfm_df_scaled['Frequency'].iloc[4],
        #                'Recency': rfm_df_scaled['Recency'].iloc[4]}  # index4 datapoint belong to cluster2
        #rfm = rfm.append(duplicateRow, ignore_index=True)
        #rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)

        # MR1.2 adding multiple points (each belonging to different cluster).
        #duplicateRow = {'Amount': rfm_df_scaled['Amount'].iloc[85],
        #                'Frequency': rfm_df_scaled['Frequency'].iloc[85],
        #                'Recency': rfm_df_scaled['Recency'].iloc[85]}  # index84 datapoint belong to cluster0
        #rfm = rfm.append(duplicateRow, ignore_index=True)
        #rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)
        #duplicateRow = {'Amount': rfm_df_scaled['Amount'].iloc[104],
        #                'Frequency': rfm_df_scaled['Frequency'].iloc[104],
        #                'Recency': rfm_df_scaled['Recency'].iloc[104]}  # index104 datapoint belong to cluster1
        #rfm = rfm.append(duplicateRow, ignore_index=True)
        #rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)
        #duplicateRow = {'Amount': rfm_df_scaled['Amount'].iloc[4],
        #                'Frequency': rfm_df_scaled['Frequency'].iloc[4],
        #                'Recency': rfm_df_scaled['Recency'].iloc[4]}  # index4 datapoint belong to cluster2
        #rfm = rfm.append(duplicateRow, ignore_index=True)
        #rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)

        #MR1.3 not valid for Agglomerative clustering.


    elif (MRType == "MR4"):
        print("MR4-Followup")

        # For instance at this index, we got inconsistent outputs = 2,12,13,16,20,21,25,26,29,31,38,39,42,47,51
        ## (i) removing single point
        #rfm.drop(rowIndex, inplace=True) #rowIndex is used to first identify the indexes for which the model gives inconsistent results for both source and follow-up execution
        #rfm_df_scaled.drop(rowIndex,inplace=True)

        #rfm.drop(2, inplace=True) #use the actual index
        #rfm_df_scaled.drop(2, inplace=True)

        ## (ii) removing multiple points (each belonging to different cluster i.e., from cluster0 and cluster2)
        rfm.drop(2, inplace=True) #For source execution, data instance at index 2 belongs to cluster2
        rfm_df_scaled.drop(2,inplace=True)
        rfm.drop(12, inplace=True) #For source execution, data instance at index 12 belongs to cluster0
        rfm_df_scaled.drop(12, inplace=True)

        # (iii) Removing 1000 rows belonging to cluster_Id = 0
        #rfm = pd.DataFrame(rfm).reset_index() #reset the index because the indexes in rfm, rfm_WithClassLabels, and rfm_df_scaled not the same
        #rfm_WithClassLabels = pd.DataFrame(rfm_WithClassLabels).reset_index()
        #newDF = rfm_WithClassLabels.loc[rfm_WithClassLabels['Cluster_Id'] == 0]
        # print(newDF.head(1000).index)
        #rfm = rfm.drop(newDF.head(1000).index)
        #rfm_df_scaled = rfm_df_scaled.drop(newDF.head(1000).index)
        #print(rfm_WithClassLabels)
        #exit(0)

    elif (MRType == "MR5"):
        print("MR5-Followup")
        rfm['NewInformativeAttribute'] = 0  # Add a new uninformative attribute (any constant) to all the instances
        # OR rfm['NewInformativeAttribute'] = 69 # Add a new uninformative attribute (any constant) to all the instances

    elif (MRType == "MR6" or MRType=="MR6Followup"):
        print("MR6-Source")

        newBoundryInstance = {'Amount': -0.34051651, 'Frequency': -0.346977995,
                              'Recency': 0.532813235}  # Average of centeroids belonging to Cluster 0 and 2
        # Treated as source: time1
        rfm = rfm.append(newBoundryInstance, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append(newBoundryInstance, ignore_index=True)
        if (MRType=="MR6Followup"):
            print("MR6-Followup")
            # time2: In next step, now shuffle the data-points, treated as follow-up: the result for this new instance + other instances should remain consistent
            # Note: When executing time2 code, don't comment out the time1 execution code. For follow-up execution, both the time1 and time2 code should be uncommented and executed.
            rfm = sklearn.utils.shuffle(rfm, random_state=1)
            rfm_df_scaled = sklearn.utils.shuffle(rfm_df_scaled, random_state=1)

    elif (MRType == "MR7"):
        print("MR7-Followup")
        print("ValueToIncrement = ", valueToIncrement)

        rfm['Amount'] = rfm['Amount'] + valueToIncrement
        rfm['Frequency'] = rfm['Frequency'] + valueToIncrement
        rfm['Recency'] = rfm['Recency'] + valueToIncrement
        rfm_df_scaled['Amount'] = rfm_df_scaled['Amount'] + valueToIncrement
        rfm_df_scaled['Frequency'] = rfm_df_scaled['Frequency'] + valueToIncrement
        rfm_df_scaled['Recency'] = rfm_df_scaled['Recency'] + valueToIncrement

    elif (MRType == "MR8"):
        print("MR8-Followup")
        rfm['Amount'] = rfm['Amount'] * valueToIncrement
        rfm['Frequency'] = rfm['Frequency'] * valueToIncrement
        rfm['Recency'] = rfm['Recency'] * valueToIncrement
        rfm_df_scaled['Amount'] = rfm_df_scaled['Amount'] * valueToIncrement
        rfm_df_scaled['Frequency'] = rfm_df_scaled['Frequency'] * valueToIncrement
        rfm_df_scaled['Recency'] = rfm_df_scaled['Recency'] * valueToIncrement

    elif (MRType == "MR9"):
        print("MR9-Followup")
        #### reset the index values, because these dataframes have different indexes than the rfm_df_scaled
        rfm_WithClassLabels = pd.DataFrame(rfm_WithClassLabels).reset_index() # so that all dataframes i.e. rfm_WithClassLabels, rfm and rfm_df_scaled have same indexes
        rfm = pd.DataFrame(rfm).reset_index() # so that all dataframes i.e. rfm_WithClassLabels, rfm and rfm_df_scaled have same indexes

        #print(rfm_WithClassLabels)
        #print(rfm)
        #print(rfm_df_scaled)
        #print(rfm_WithClassLabels[rfm_WithClassLabels["Cluster_Id"] == 0])
        #print(rfm_df_scaled.loc[10])

        # MR9.1: replace one instance of cluster 0#####
        #rfm['Amount'].loc[12] = 2.812697 # At index 10,12,15,50,65,...,4207,4246,4248,4249,4252 instances belong to cluster 0, so we selected index 10 element
        #rfm['Frequency'].loc[12] = 0.533783
        #rfm['Recency'].loc[12] = -0.599520
        #rfm_df_scaled['Amount'].loc[12] = 2.812697
        #rfm_df_scaled['Frequency'].loc[12] = 0.533783
        #rfm_df_scaled['Recency'].loc[12] = -0.599520

        #### MR9.2: replace all instances of cluster 0#####

        #rfm['Amount'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = 2.812697 # the indexes assigned to rfm and rfm_WithClassLabels are not properly odered but the same indexes are assigned to instances to both dataframes, so this is the reason that the filter is applied on rfm_WithClassLabels not the dfWithProperIndexes
        #rfm['Frequency'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = 0.533783
        #rfm['Recency'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = -0.599520
        #rfm_df_scaled['Amount'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = 2.812697
        #rfm_df_scaled['Frequency'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = 0.533783
        #rfm_df_scaled['Recency'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = -0.599520

        #####MR9.3###################################
        # print("MR9.3-Followup")
        ### Inside the main function, we can see in the rfm_source that at these indexes 10,12,15,50,65,...,4207,4246,4248,4249,4252 the data points are assigned to cluster0. We can update them with any of the index data (let update them with 10 index data in rfm_df_scaled dataframe)
        # print(rfm_WithClassLabels.head(20))
        # print(rfm_df_scaled.head(20))
        rfm_df_scaled.at[12, 'Amount'] = 2.812697
        rfm_df_scaled.at[12, 'Frequency'] = 0.533783
        rfm_df_scaled.at[12, 'Recency'] = -0.599520
        rfm_df_scaled.at[15, 'Amount'] = 2.812697
        rfm_df_scaled.at[15, 'Frequency'] = 0.533783
        rfm_df_scaled.at[15, 'Recency'] = -0.599520
        rfm_df_scaled.at[50, 'Amount'] = 2.812697
        rfm_df_scaled.at[50, 'Frequency'] = 0.533783
        rfm_df_scaled.at[50, 'Recency'] = -0.599520
        rfm_df_scaled.at[65, 'Amount'] = 2.812697
        rfm_df_scaled.at[65, 'Frequency'] = 0.533783
        rfm_df_scaled.at[65, 'Recency'] = -0.599520
        rfm_df_scaled.at[420, 'Amount'] = 2.812697
        rfm_df_scaled.at[420, 'Frequency'] = 0.533783
        rfm_df_scaled.at[420, 'Recency'] = -0.599520

    elif (MRType == "MR10"):
        print("MR10-Followup")  # Perform swapping of features
        rfm = rfm[['Frequency', 'Recency', 'Amount', 'CustomerID']]
        rfm_df_scaled = rfm_df_scaled[['Frequency', 'Recency', 'Amount']]

    elif (MRType == "MR11"):
        print("MR11-Followup")
        rfm_WithClassLabels = pd.DataFrame(rfm_WithClassLabels).reset_index()  # so that all dataframes i.e. rfm_WithClassLabels, rfm and rfm_df_scaled have same indexes
        rfm = pd.DataFrame(rfm).reset_index()

        rfm['NewInformativeAttribute'] = 0
        rfm['NewInformativeAttribute'].loc[rfm_WithClassLabels[
                                               'Cluster_Id'] == 0] = 0.3  # the indexes assigned to rfm and rfm_WithClassLabels are not properly odered but the same indexes are assigned to instances to both dataframes, so this is the reason that the filter is applied on rfm_WithClassLabels not the dfWithProperIndexes
        rfm['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 1] = 0.6
        rfm['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 2] = 0.9
        rfm_df_scaled['NewInformativeAttribute'] = 0
        rfm_df_scaled['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = 0.3
        rfm_df_scaled['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 1] = 0.6
        rfm_df_scaled['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 2] = 0.9

    elif (MRType == "MR12"):
        print("MR12-Followup")
        ##MR12.1: Reversing the data points##
        rfm  = rfm.loc[::-1]
        rfm_df_scaled = rfm_df_scaled.loc[::-1]

        ##MR12.2: Changing the order of data points/shuffling randomly## Fix the random seed because when we are building the model, we use rfm_df_scaled to build the model and then assigning the predicted label to the 'Cluster_Id' column in rfm dataframe. So, the order of data-points in both dataframes should remain the same, otherwise, in one dataframe the datapoint#1 may be at index/location 5, whereas in other dataframe it may be at index/location 25
        #rfm = sklearn.utils.shuffle(rfm, random_state=1)
        #rfm_df_scaled = sklearn.utils.shuffle(rfm_df_scaled, random_state=1)

    elif (MRType == "MR13"):
        print("MR13-Followup")
        rfm['Amount'] = rfm['Amount'] * -1
        rfm['Frequency'] = rfm['Frequency'] * -1
        rfm['Recency'] = rfm['Recency'] * -1
        rfm_df_scaled['Amount'] = rfm_df_scaled['Amount'] * -1
        rfm_df_scaled['Frequency'] = rfm_df_scaled['Frequency'] * -1
        rfm_df_scaled['Recency'] = rfm_df_scaled['Recency'] * -1

    elif (MRType == "MR14"):
        print("MR14-Followup")
        rfm_WithClassLabels = pd.DataFrame(rfm_WithClassLabels).reset_index()  # so that all dataframes i.e. rfm_WithClassLabels, rfm and rfm_df_scaled have same indexes

        #print(rfm_WithClassLabels)
        #print(rfm_df_scaled)

        # MR14.1 adding new data point(s) with no informative attributes should not change the output (all features have 0 value).
        # duplicateRow = {'Amount': 0, 'Frequency': 0, 'Recency': 0}
        # for i in range(100):
        #    rfm = rfm.append(duplicateRow, ignore_index=True)
        #    rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)

        # Element at index0 (-0.723738  -0.752888  2.301611) and index1(1.731617   1.042467 -0.906466) both belong to cluster#2. Let add multiple data points inbetween these two data points
        # MR14.2 Add multiple points in between these two points to make this cluster more compact/dense
        rfm = pd.DataFrame(rfm).reset_index()
        rfm = rfm.append({'Amount': -0.5, 'Frequency': -0.6, 'Recency': 2.1}, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append({'Amount': -0.5, 'Frequency': -0.6, 'Recency': 2.1}, ignore_index=True)
        rfm = rfm.append({'Amount': -0.3, 'Frequency': -0.4, 'Recency': 1.7}, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append({'Amount': -0.3, 'Frequency':-0.4, 'Recency': 1.7}, ignore_index=True)
        rfm = rfm.append({'Amount': -0.1, 'Frequency': -0.2, 'Recency': 0.6}, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append({'Amount': -0.1, 'Frequency': -0.2, 'Recency': 0.6}, ignore_index=True)
        rfm = rfm.append({'Amount': 1.71, 'Frequency': 1.02, 'Recency': 0.88}, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append({'Amount': 1.71, 'Frequency': 1.02, 'Recency': 0.88}, ignore_index=True)
        rfm = rfm.append({'Amount': 1.72, 'Frequency': 1.03, 'Recency': -0.89}, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append({'Amount': 1.72, 'Frequency': 1.03, 'Recency': -0.89}, ignore_index=True)

    rfm, rfm_df_scaled, clusterModel = buildModel(rfm, rfm_df_scaled, MRType, valueToIncrement)


    #####Step 4.2: Print Information#####
    #printModelInfo(clusterModel,rfm, rfm_df_scaled)

    ######Step 5 : Final Analysis########
    # Inference:
    # Hierarchical Clustering with 3 Cluster Labels

    # Customers with Cluster_Label 2 are the customers with high amount of transactions as compared to other customers.
    # Customers with Cluster_Label 2 are frequent buyers.
    # Customers with Cluster_Label 0 are not recent buyers and hence least of importance from business point of view.


    return rfm, rfm_df_scaled,clusterModel


if __name__ == '__main__':
    ##########################################################
    # ============Metamorphic Relations (MRs)=================#
    ##########################################################

    ###############################
    #####Source Executions======
    ###############################
    #print("#####Source Execution#####")

    rfm_Source, rfm_df_scaled_Source, hierarchicalModel = hierarchicalAlgorithm()
    #print(rfm_Source.shape)
    #exit(0)

    # =========MR#1: Duplicating single, and multiple instances (each belonging to different class) =============#

    ##To find the exact/actual data rows (thier indexes) for which the model produces inconsistent results for the source and follow-up executions
    #rfm_Source = pd.DataFrame(rfm_Source).reset_index()
    #conflictDataInstances = []
    #for rowIndex in range(4292):
    #    rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR1", rfm_Source,rowIndex)
    #    mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #    if (len(inconsistentOutDataRows)>0):
    #        conflictDataInstances.append(rowIndex)
    #        print("For instance at this index, we got inconsistent outputs = ", rowIndex)
    #print("Instances indexes for which we got different outputs = ", conflictDataInstances)
    # Found data instances indexes for which violations have been found. These are just few found within first 120 iterations
    # For instance at this index, we got inconsistent outputs = 4,13, 20, 21, 23, 37, 43, 51, 56, 74, 78, 79, 85, 90,91, 98, 99,101,104,105,108,109,114
    #rfm_Source = pd.DataFrame(rfm_Source).reset_index()
    #print(rfm_Source.loc[[4,13, 20, 21, 23, 37, 43, 51, 56, 74, 78, 79, 85, 90,91, 98, 99,101,104,105,108,109,114]])
    #exit(0)

    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR1")
    #rfm_Source = pd.DataFrame(rfm_Source).reset_index() # reset the index, as they are not in sequential order but for rfm_Followup it is sequential
    ## Merge (using inner join based on the indexes) the source(having all datapoints) with follow-up(with the left data), the common data points should be allocated to same clusters for both the source and follow-up executions
    #mergedDataFrame = pd.merge(rfm_Source,rfm_Followup,left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("Inconsistent Data Rows at specific index =", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#2: Apply normalization on the normalizeed data =============#
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR2")
    ## Merge (using inner join based on the indexes) the source(having all datapoints) with follow-up(with the left data), the common data points should be allocated to same clusters for both the source and follow-up executions
    #mergedDataFrame = pd.merge(rfm_Source,rfm_Followup,left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("Inconsistent Data Rows at specific index =", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#3: Addition of features by copying/duplicating the original feature set=============#
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR3")
    ## Merge (using inner join based on the indexes) the source(having all datapoints) with follow-up(with the left data), the common data points should be allocated to same clusters for both the source and follow-up executions
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("Inconsistent Data Rows at specific index =", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#4: Removing one, more instances (each belonging to different cluster) should not change the output =============#

    # ================================================#

    ##To find the exact/actual data rows (thier indexes) for which the model produces inconsistent results for the source and follow-up executions
    #conflictDataInstances = []
    #for rowIndex in range(4292):
    #    rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR4", rfm_Source,rowIndex)
    #    mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #    if (len(inconsistentOutDataRows)>0):
    #        conflictDataInstances.append(rowIndex)
    #        print("For instance at this index, we got inconsistent outputs = ", rowIndex)
    #print("Instances indexes for which we got different outputs = ", conflictDataInstances)
    # Found data instances indexes for which violations have been found. These are just few found within first 55 iterations
    # For instance at this index, we got inconsistent outputs = 2,12,13,16,20,21,25,26,29,31,38,39,42,47,51
    #print(rfm_Source.loc[[2,12,13,16,20,21,25,26,29,31,38,39,42,47,51]])

    # ==========Execute Actual MR=============#
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR4", rfm_Source)
    ## Merge (using inner join based on the indexes) the source(having all datapoints) with follow-up(with the left data), the common data points should be allocated to same clusters for both the source and follow-up executions
    #rfm_Source = pd.DataFrame(rfm_Source).reset_index() #Very Important Note: Uncomment it only when executing MR4.3
    #mergedDataFrame = pd.merge(rfm_Source,rfm_Followup,left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("Inconsistent Data Rows at specific index =", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#5: Adding datapoint(s) with 0 features' values should not change the output=============#
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR5",rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source,rfm_Followup,left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#6: Consistent with reprediction=============#
    #rfm_SourceMR6, rfm_df_scaled_SourceMR6, hierarchicalModel_SourceMR6 = hierarchicalAlgorithm("MR6", rfm_Source)
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel_Followup = hierarchicalAlgorithm("MR6Followup", rfm_Source)
    #mergedDataFrame = pd.merge(rfm_SourceMR6,rfm_Followup,left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print(rfm_SourceMR6)
    #print(rfm_Followup)

    # =========MR#7: Shifting the data set features by a constant i.e., x + 2=============#
    ##To find the exact/actual data rows (thier indexes) for which the model produces inconsistent results for the source and follow-up executions
    ## I run a lot of iterations with different values but found no inconsistency
    #valueToIncrement = 0.0099
    #conflictDataInstances = []
    #for rowIndex in range(500):
    #    print("Iteration#1: ", rowIndex)
    #    rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR7", rfm_Source,rowIndex,valueToIncrement)
    #    mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #    print("Inconsistent = ", (inconsistentOutDataRows.head()).to_string())
    #    if len(inconsistentOutDataRows)>0:
    #        conflictDataInstances.append(valueToIncrement)
    #        print("Value for which we got inconsistent outputs = ", valueToIncrement)
    #    valueToIncrement = valueToIncrement + (0.0097*3)
    #print("Values for which we got different outputs = ", conflictDataInstances)
    # ========================================================#

    # ==========Actual MR Execution=============#
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR7",rfm_Source,-1,2)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#8: Scaling the data set features by a constant i.e., x * 2=============#
    ##To find the exact/actual data rows (thier indexes) for which the model produces inconsistent results for the source and follow-up executions
    ## I run a lot of iterations with different values but found no inconsistency
    #valueToIncrement = 0.0099
    #conflictDataInstances = []
    #for rowIndex in range(500):
    #    print("Iteration#1: ", rowIndex)
    #    rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR8", rfm_Source,rowIndex,valueToIncrement)
    #    mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #    print("Inconsistent = ", (inconsistentOutDataRows.head()).to_string())
    #    if len(inconsistentOutDataRows)>0:
    #        conflictDataInstances.append(valueToIncrement)
    #        print("Value for which we got inconsistent outputs = ", valueToIncrement)
    #    valueToIncrement = valueToIncrement + (0.0097*3)
    #print("Values for which we got different outputs = ", conflictDataInstances)
    # ========================================================#

    # ==========Actual MR Execution=============#
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR8",rfm_Source,-1,2)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#9:MR_replace=============#

    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR9",rfm_Source)
    #rfm_Source = pd.DataFrame(rfm_Source).reset_index()
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head(20)).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)


    # =========MR#10:Swapping the features=============#
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR10")
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#11: Adding uninformative attribute(s)=============#
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR11",rfm_Source)
    #rfm_Source = pd.DataFrame(rfm_Source).reset_index()
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#12:Reversing the data-points=============#
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR12",rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#13: Multiple all the features with -1=============#
    #rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR13",rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#14: Enhancing the compactness of specific cluster=============#
    rfm_Followup, rfm_df_scaled_Followup, hierarchicalModel = hierarchicalAlgorithm("MR14",rfm_Source)
    rfm_Source = pd.DataFrame(rfm_Source).reset_index() # reset the index, as they are not in sequential order but for rfm_Followup it is sequential
    mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head(20)).to_string())
    print(rfm_Source)
    print(rfm_Followup)
