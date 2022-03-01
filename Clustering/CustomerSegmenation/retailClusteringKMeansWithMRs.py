#https://www.kaggle.com/hellbuoy/online-retail-k-means-hierarchical-clustering
##plotting clusters centeroids in scatterplot
#https://stackoverflow.com/questions/54240144/distance-between-nodes-and-the-centroid-in-a-kmeans-cluster

# To install yellowbrick package (for visualization) with conda, run in terminal: conda install -c districtdatalabs yellowbrick
# import required libraries for dataframe and visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.metrics import pairwise_distances_argmin_min
from yellowbrick.cluster import SilhouetteVisualizer
# import required libraries for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

def loadData():
    # Reading the data on which analysis needs to be done
    retail = pd.read_csv('OnlineRetail.csv', sep=",", encoding="ISO-8859-1", header=0)
    retail.head()
    retail.shape
    retail.info()
    retail.describe()
    return retail

def dataCleaning(retail):
    # Calculating the Missing Values % contribution in DF
    df_null = round(100 * (retail.isnull().sum()) / len(retail), 2)
    df_null
    # Droping rows having missing values
    retail = retail.dropna()
    retail.shape
    # Changing the datatype of Customer Id as per Business understanding
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    return retail

def dataPreparation(retail,MRType):
    # New Attribute : Monetary
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']
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
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%m/%d/%Y %H:%M')
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
    attributes = ['Amount', 'Frequency', 'Recency']
    plt.rcParams['figure.figsize'] = [10, 8]
    sns.boxplot(data=rfm[attributes], orient="v", palette="Set2", whis=1.5, saturation=1, width=0.7)
    plt.title("Outliers Variable Distribution", fontsize=14, fontweight='bold')
    plt.ylabel("Range", fontweight='bold')
    plt.xlabel("Attributes", fontweight='bold')
    # Removing (statistical) outliers for Amount
    Q1 = rfm.Amount.quantile(0.05)
    Q3 = rfm.Amount.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Amount >= Q1 - 1.5 * IQR) & (rfm.Amount <= Q3 + 1.5 * IQR)]

    # Removing (statistical) outliers for Recency
    Q1 = rfm.Recency.quantile(0.05)
    Q3 = rfm.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Recency >= Q1 - 1.5 * IQR) & (rfm.Recency <= Q3 + 1.5 * IQR)]

    # Removing (statistical) outliers for Frequency
    Q1 = rfm.Frequency.quantile(0.05)
    Q3 = rfm.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Frequency >= Q1 - 1.5 * IQR) & (rfm.Frequency <= Q3 + 1.5 * IQR)]

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
        rfm_df_scaled["Frequen1"] = rfm_df_scaled["Frequency"]
        rfm_df_scaled["Recency1"] = rfm_df_scaled["Recency"]

    return rfm, rfm_df_scaled

def buildModel(rfm, rfm_df_scaled,MRType,valueToIncrement=None):
    # k-means with some arbitrary/random k
    #kmeans = KMeans(n_clusters=4, max_iter=50,random_state=42)
    #kmeans.fit(rfm_df_scaled)
    #kmeans.labels_

    ##Finding the Optimal Number of Clusters
    # Elbow-curve/SSD Method
    ssd = []
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    fig, ax = plt.subplots(4, 2, figsize=(15, 8))

    #for num_clusters in range_n_clusters:
    #    kmeans = KMeans(n_clusters=num_clusters, max_iter=50,random_state=42)
    #    # silhouette visualizer
    #    #q, mod = divmod(num_clusters, 2)
    #    #visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
    #    # end
    #    kmeans.fit(rfm_df_scaled)
    #    ssd.append(kmeans.inertia_)

    #visualizer.fit(rfm_df_scaled)  # Fit the data to the visualizer
    #visualizer.show()

    # plot the SSDs for each n_clusters
    #plt.plot(ssd)

    # Silhouette analysis Method
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        # initialize kmeans
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50,random_state=42)
        kmeans.fit(rfm_df_scaled)
        cluster_labels = kmeans.labels_
        # silhouette score
        silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
        print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

    # Final model with k=3, fix the random seed and selects initial cluster centers for k-mean clustering in a smart way to speed up convergence using init='k-means++'
    if (MRType == "MR3"):
        startpts = np.array([[-0.30, -0.26, 0.9,-0.30, -0.26, 0.9], [1.2, 2.8, -0.41,1.2, 2.8, -0.41], [-0.02, -0.24, -0.17,-0.02, -0.24, -0.17]], np.float64)
    elif(MRType == "MR7"):
        startpts = np.array([[-0.30+valueToIncrement ,-0.26+valueToIncrement,  0.9+valueToIncrement], [1.2+valueToIncrement,2.8+valueToIncrement, -0.41+valueToIncrement], [-0.02+valueToIncrement,-0.24+valueToIncrement, -0.17+valueToIncrement]], np.float64) # shift the centroid with some constant i.e., x + 2
    elif (MRType == "MR8"):
        startpts = np.array([[-0.30*valueToIncrement ,-0.26*valueToIncrement,  0.9*valueToIncrement], [1.2*valueToIncrement,2.8*valueToIncrement, -0.41*valueToIncrement], [-0.02*valueToIncrement,-0.24*valueToIncrement, -0.17*valueToIncrement]],np.float64)  # scale the centroid with some constant i.e., x*2
    elif (MRType == "MR10"):
        startpts = np.array([[-0.26,  0.9,-0.30], [2.8, -0.41,1.2], [-0.24, -0.17,-0.02]], np.float64) # Change order from ['Amount','Frequency','Recency'] ['Frequency','Recency','Amount']
    elif (MRType == "MR11"):
        startpts = np.array([[-0.30 ,-0.26,  0.9,0.3], [1.2,2.8, -0.41,0.6], [-0.02,-0.24, -0.17,0.9]], np.float64)
    elif (MRType == "MR13"):
        startpts = np.array([[0.30 ,0.26,  -0.9], [-1.2,-2.8, 0.41], [0.02,0.24, 0.17]], np.float64)
    else:
        startpts = np.array([[-0.30 ,-0.26,  0.9], [1.2,2.8, -0.41], [-0.02,-0.24, -0.17]], np.float64)

    print("Initial/Starting Centeroids = " , startpts)
    kmeans = KMeans(n_clusters=3, max_iter=50,init=startpts, n_init=1,random_state=42) #ini='k-means++'
    kmeans.fit(rfm_df_scaled)
    #kmeans.labels_
    # For verification of MR, we can use the results of predictions made.
    #print(kmeans.predict([[-0.486919 ,-0.533455,  1.519395], [1.89444,2.069808, -0.688633]]))
    # assign the label
    rfm['Cluster_Id'] = kmeans.labels_
    # Box plot to visualize Cluster Id vs Frequency
    sns.boxplot(x='Cluster_Id', y='Amount', data=rfm)
    # Box plot to visualize Cluster Id vs Frequency
    # sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)
    # Box plot to visualize Cluster Id vs Recency
    # sns.boxplot(x='Cluster_Id', y='Recency', data=rfm)

    # Clusters Visualization
    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(x='CustomerID', y='Amount', hue='Cluster_Id', data=rfm, palette='bright')
    #plt.show()

    return rfm,rfm_df_scaled, kmeans

def printModelInfo(kmeans,rfm, rfm_df_scaled):
    # print centeroids
    print("Identified Cluster Centeroids: ",kmeans.cluster_centers_)
    # pritn centeroids class labels
    print("Centeroids class labels: ",kmeans.predict(kmeans.cluster_centers_))
    # print (Sum of squared distances of samples to their closest cluster center)
    print("SSD: ",kmeans.inertia_)
    # print (Number of iterations run)
    print("Number of iterations used",kmeans.n_iter_)

    ######Print information used to verify the outputs######
    print("=========Print information used to verify the outputs========")
    # print (the nearest point to clusters)

    closest,distance = pairwise_distances_argmin_min(kmeans.cluster_centers_[0].reshape(1,-1), rfm_df_scaled)
    #print("Shape: ", rfm_df_scaled)
    #print("Nearest point index to cluster[0]: ", closest0)
    print("Cluster[0] centeroid: ", kmeans.cluster_centers_[0])
    print("Nearest point to cluster[0] centeroid: ",rfm_df_scaled.iloc[closest])
    print("Distance of Nearest point to cluster[0] centeroid: ", distance)

    closest, distance = pairwise_distances_argmin_min(kmeans.cluster_centers_[1].reshape(1, -1), rfm_df_scaled)
    print("Cluster[1] centeroid: ", kmeans.cluster_centers_[1])
    print("Nearest point to cluster[1] centeroid: ", rfm_df_scaled.iloc[closest])
    print("Distance of Nearest point to cluster[1] centeroid: ", distance)

    closest, distance = pairwise_distances_argmin_min(kmeans.cluster_centers_[2].reshape(1, -1), rfm_df_scaled)
    print("Cluster[2] centeroid: ", kmeans.cluster_centers_[2])
    print("Nearest point to cluster[2] centeroid: ", rfm_df_scaled.iloc[closest])
    print("Distance of Nearest point to cluster[2] centeroid: ", distance)

    #### Verify using number of points in each cluster ####
    #print("Count of data points in Cluster0: ", )
    #print("Count of data points in Cluster1: ", )
    #print("Count of data points in Cluster2: ", )

    #https://stackoverflow.com/questions/45234336/value-at-kmeans-cluster-centers-in-sklearn-kmeans

def kmeansAlgorithm(MRType = "", rfm_WithClassLabels = None, rowIndex = None, valueToIncrement = None):
    #####Step 1 : Reading and Understanding Data########
    retail = loadData()

    ############Step 2 : Data Cleansing#################
    retail = dataCleaning(retail)

    ######Step 3 : Data Preparation########
    rfm, rfm_df_scaled = dataPreparation(retail,MRType)

    ######Step 4 : Building the Model######
    if(MRType =="MR1"):
        print("MR1-Followup")

        # MR1.1: adding single point
        #duplicateRow = {'Amount': 1.731617, 'Frequency': 1.042467, 'Recency': -0.906466} #Cluster_1, index1
        #rfm = rfm.append(duplicateRow, ignore_index=True)
        #rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)

        # MR1.2 adding multiple points (each belonging to different cluster).
        #duplicateRow = {'Amount' : -0.723738, 'Frequency' : -0.752888, 'Recency' : 2.301611} #Cluster_0, index0
        #rfm = rfm.append(duplicateRow, ignore_index=True)
        #rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)
        #duplicateRow = {'Amount': 1.731617, 'Frequency': 1.042467, 'Recency': -0.906466}  # Cluster_1, index1
        #rfm = rfm.append(duplicateRow, ignore_index=True)
        #rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)
        #duplicateRow = {'Amount': 0.300128, 'Frequency': -0.463636, 'Recency': -0.183658} #Cluster_2, index2
        #rfm = rfm.append(duplicateRow, ignore_index=True)
        #rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)

        # MR1.3 adding duplicate centroids (found during source execution belonging to each cluster)
        duplicateRow = {'Amount' : -0.50237552, 'Frequency' : -0.51846606, 'Recency' : 1.54212771} #belonging to Cluster_0
        rfm = rfm.append(duplicateRow, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)
        duplicateRow = {'Amount': 2.08754921, 'Frequency': 2.10491086, 'Recency': -0.70109576}  #belonging to Cluster_1
        rfm = rfm.append(duplicateRow, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)
        duplicateRow = {'Amount': -0.1786575, 'Frequency': -0.17548993, 'Recency': -0.47650124} #belonging to Cluster_2
        rfm = rfm.append(duplicateRow, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)



    elif (MRType == "MR4"):
        print("MR4-Followup")

        # For instance at this index, we got inconsistent outputs = 50,65,70,85,99,101,104,155,167
        ## (i) removing single point
        #rfm.drop(rowIndex, inplace=True)
        #rfm_df_scaled.drop(rowIndex,inplace=True)

        rfm.drop(50, inplace=True)
        rfm_df_scaled.drop(50, inplace=True)

        ## (ii) removing multiple points (each belonging to different cluster)
        #rfm.drop(50, inplace=True) #For source execution, data instance at index 50 belongs to cluster1
        #rfm_df_scaled.drop(50,inplace=True)
        #rfm.drop(65, inplace=True) #For source execution, data instance at index 50 belongs to cluster2
        #rfm_df_scaled.drop(65, inplace=True)
        #rfm.drop(85, inplace=True) #For source execution, data instance at index 85 belongs to cluster0
        #rfm_df_scaled.drop(85, inplace=True)

        # (iii) Removing 1000 rows belonging to cluster_Id = 0
        #rfm = pd.DataFrame(rfm).reset_index()  # reset the index because the indexes in rfm, rfm_WithClassLabels, and rfm_df_scaled not the same
        #rfm_WithClassLabels = pd.DataFrame(rfm_WithClassLabels).reset_index()
        #newDF = rfm_WithClassLabels.loc[rfm_WithClassLabels['Cluster_Id'] == 0]
        #print(newDF.head(1000).index)
        #rfm = rfm.drop(newDF.head(1000).index)
        #rfm_df_scaled = rfm_df_scaled.drop(newDF.head(1000).index)


    elif (MRType == "MR5"):
        print("MR5-Followup")
        rfm['NewInformativeAttribute'] = 0 # Add a new uninformative attribute (any constant) to all the instances
        # OR rfm['NewInformativeAttribute'] = 69 # Add a new uninformative attribute (any constant) to all the instances

    elif (MRType == "MR6" or MRType=="MR6Followup"):
        print("MR6-Source")
        #rfm = rfm.drop(labels=['Frequency'],axis=1)
        #rfm_df_scaled = rfm_df_scaled.drop(labels=['Frequency'],axis=1)

        newBoundryInstance = {'Amount': -0.34051651, 'Frequency': -0.346977995, 'Recency': 0.532813235} # Average of centeroids belonging to Cluster 0 and 2
        # Treated as source: time1
        rfm = rfm.append(newBoundryInstance, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append(newBoundryInstance, ignore_index=True)

        if (MRType=="MR6Followup"):
            print("MR6-Followup")
            #time2: In next step, now shuffle the data-points, treated as follow-up: the result for this new instance + other instances should remain consistent
            # Note: When executing time2 code, don't comment out the time1 execution code. For follow-up execution, both the time1 and time2 code should be uncommented and executed.
            rfm = sklearn.utils.shuffle(rfm,random_state=1)
            rfm_df_scaled = sklearn.utils.shuffle(rfm_df_scaled,random_state=1)

    elif (MRType == "MR7"):
        print("MR7-Followup")
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
        rfm_WithClassLabels = pd.DataFrame(rfm_WithClassLabels).reset_index()
        rfm = pd.DataFrame(rfm).reset_index()

        #print(rfm_WithClassLabels)
        #print(rfm)
        #print(rfm_df_scaled)
        #print(rfm_WithClassLabels[rfm_WithClassLabels["Cluster_Id"] == 0])
        #print(rfm_df_scaled.loc[0])

        # MR9.1: replace one instance of cluster 0#####
        #rfm['Amount'].loc[4] = -0.723738 # At index 0,4,6,7,8,...4289,4290, instances belong to cluster 0, so we selected index 0 element
        #rfm['Frequency'].loc[4] = -0.752888
        #rfm['Recency'].loc[4] = 2.301611
        #rfm_df_scaled['Amount'].loc[4] = -0.723738
        #rfm_df_scaled['Frequency'].loc[4] = -0.752888
        #rfm_df_scaled['Recency'].loc[4] = 2.301611

        #### MR9.2: replace all instances of cluster 0#####

        #rfm['Amount'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = -0.723738 # the indexes assigned to rfm and rfm_WithClassLabels are not properly odered but the same indexes are assigned to instances to both dataframes, so this is the reason that the filter is applied on rfm_WithClassLabels not the dfWithProperIndexes
        #rfm['Frequency'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = -0.752888
        #rfm['Recency'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = 2.301611
        #rfm_df_scaled['Amount'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = -0.723738
        #rfm_df_scaled['Frequency'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = -0.752888
        #rfm_df_scaled['Recency'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = 2.301611

        #####MR9.3###################################
        #print("MR9.3-Followup")
        ### Inside the main function, we can see in the rfm_source that at these indexes 0,4,6,7,8,...4289,4290 the data points are assigned to cluster0. We can update them with any of the index data (let update them with 6 index data in rfm_df_scaled dataframe)
        #print(rfm_WithClassLabels.head(20))
        #print(rfm_df_scaled.head(20))
        rfm_df_scaled.at[7, 'Amount'] = -0.673036
        rfm_df_scaled.at[7, 'Frequency'] = -0.732939
        rfm_df_scaled.at[7, 'Recency'] = 1.093632
        rfm_df_scaled.at[0, 'Amount'] = -0.673036
        rfm_df_scaled.at[0, 'Frequency'] = -0.732939
        rfm_df_scaled.at[0, 'Recency'] = 1.093632
        rfm_df_scaled.at[8, 'Amount'] = -0.673036
        rfm_df_scaled.at[8, 'Frequency'] = -0.732939
        rfm_df_scaled.at[8, 'Recency'] = 1.093632
        rfm_df_scaled.at[4289, 'Amount'] = -0.673036
        rfm_df_scaled.at[4289, 'Frequency'] = -0.732939
        rfm_df_scaled.at[4289, 'Recency'] = 1.093632
        rfm_df_scaled.at[4290, 'Amount'] = -0.673036
        rfm_df_scaled.at[4290, 'Frequency'] = -0.732939
        rfm_df_scaled.at[4290, 'Recency'] = 1.093632

    elif (MRType == "MR10"):
        print("MR10-Followup") # Perform swapping of features
        rfm = rfm[['Frequency','Recency','Amount','CustomerID']]
        rfm_df_scaled = rfm_df_scaled[['Frequency','Recency','Amount']]

    elif(MRType == "MR11"):
        print("MR11-Followup")
        dfWithProperIndexes = pd.DataFrame(rfm_WithClassLabels).reset_index() # prepare a new dataset with reset index option so that all data instances are assigned with proper index incrementally. This dataframe and rfm_df_scaled dataframe will now have same indexes for the instances

        rfm['NewInformativeAttribute'] = 0
        rfm['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = 0.3 # the indexes assigned to rfm and rfm_WithClassLabels are not properly odered but the same indexes are assigned to instances to both dataframes, so this is the reason that the filter is applied on rfm_WithClassLabels not the dfWithProperIndexes
        rfm['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 1] = 0.6
        rfm['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 2] = 0.9
        rfm_df_scaled['NewInformativeAttribute'] = 0
        rfm_df_scaled['NewInformativeAttribute'].loc[dfWithProperIndexes['Cluster_Id'] == 0] = 0.3
        rfm_df_scaled['NewInformativeAttribute'].loc[dfWithProperIndexes['Cluster_Id'] == 1] = 0.6
        rfm_df_scaled['NewInformativeAttribute'].loc[dfWithProperIndexes['Cluster_Id'] == 2] = 0.9

    elif (MRType == "MR12"):
        print("MR12-Followup")
        ##MR12.1: Reversing the data points##
        #rfm  = rfm.loc[::-1]
        #rfm_df_scaled = rfm_df_scaled.loc[::-1]

        ##MR12.2: Changing the order of data points/shuffling randomly## Fix the random seed because when we are building the model, we use rfm_df_scaled to build the model and then assigning the predicted label to the 'Cluster_Id' column in rfm dataframe. So, the order of data-points in both dataframes should remain the same, otherwise, in one dataframe the datapoint#1 may be at index/location 5, whereas in other dataframe it may be at index/location 25
        rfm = sklearn.utils.shuffle(rfm,random_state=1)
        rfm_df_scaled = sklearn.utils.shuffle(rfm_df_scaled,random_state=1)

    elif (MRType == "MR13"):
        print("MR13-Followup")
        rfm['Amount'] = rfm['Amount'] * -1
        rfm['Frequency'] = rfm['Frequency'] * -1
        rfm['Recency'] = rfm['Recency'] * -1
        rfm_df_scaled['Amount'] = rfm_df_scaled['Amount'] * -1
        rfm_df_scaled['Frequency'] = rfm_df_scaled['Frequency'] * -1
        rfm_df_scaled['Recency'] = rfm_df_scaled['Recency'] * -1

    elif(MRType == "MR14"):
        print("MR14-Followup")


        # MR14.1 adding new data point(s) with no informative attributes should not change the output (all features have 0 value).
        # duplicateRow = {'Amount': 0, 'Frequency': 0, 'Recency': 0}
        # for i in range(100):
        #    rfm = rfm.append(duplicateRow, ignore_index=True)
        #    rfm_df_scaled = rfm_df_scaled.append(duplicateRow, ignore_index=True)

        # 14.2 Add multiple points in between these two points to make this cluster more compact/dense
        # Cluster[1]centeroid:   2.08754921  2.10491086 -0.70109576
        # nearest point#1411:  	1.89444   2.069808 -0.688633

        rfm = rfm.append({'Amount': 2.07, 'Frequency': 2.09, 'Recency': -0.692}, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append({'Amount': 2.07, 'Frequency': 2.09, 'Recency': -0.692}, ignore_index=True)
        rfm = rfm.append({'Amount': 1.93, 'Frequency': 2.08, 'Recency': -0.694}, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append({'Amount': 1.93, 'Frequency': 2.08, 'Recency': -0.694}, ignore_index=True)
        rfm = rfm.append({'Amount': 1.95, 'Frequency': 2.085, 'Recency': -0.696}, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append({'Amount': 1.95, 'Frequency': 2.085, 'Recency': -0.696}, ignore_index=True)
        rfm = rfm.append({'Amount': 1.87, 'Frequency': 2.08, 'Recency': -0.698}, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append({'Amount': 1.87, 'Frequency': 2.08, 'Recency': -0.698}, ignore_index=True)
        rfm = rfm.append({'Amount': 1.88, 'Frequency': 2.07, 'Recency': -0.699}, ignore_index=True)
        rfm_df_scaled = rfm_df_scaled.append({'Amount': 1.88, 'Frequency': 2.07, 'Recency': -0.699}, ignore_index=True)

    rfm, rfm_df_scaled, kmeans = buildModel(rfm, rfm_df_scaled,MRType,valueToIncrement)
    #####Step 4.1: Print Information#####
    printModelInfo(kmeans,rfm, rfm_df_scaled)

    ######Step 5 : Final Analysis########
    #Inference:
    #K-Means Clustering with 3 Cluster Ids

    #Customers with Cluster Id 0 are the customers with high amount of transactions as compared to other customers.
    #Customers with Cluster Id 1 are frequent buyers.
    #Customers with Cluster Id 2 are not recent buyers and hence least of importance from business point of view.
    return rfm, rfm_df_scaled,kmeans

if __name__ == '__main__':

    ##########################################################
    #============Metamorphic Relations (MRs)=================#
    ##########################################################

    ###############################
    #####Source Executions======
    ###############################
    print("#####Source Execution#####")
    rfm_Source, rfm_df_scaled_Source, kmeansModel_Source = kmeansAlgorithm()
    #print(rfm_Source.loc[[50,65,70,85,99,101,104,155,167], :])


    ###testData = rfm_df_scaled_Source.copy()
    ###testDataPredictions_Source = kmeansModel_Source.fit_predict(testData)

    ###############################
    #####Follow-up Executions======
    ###############################
    print("#####Follow-up Execution#####")

    #=========MR#1: Duplicating single, and multiple instances (each belonging to different class) =============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR1")
    #rfm_Source = pd.DataFrame(rfm_Source).reset_index() # reset the index, as they are not in sequential order but for rfm_Followup it is sequential
    ## Merge (using inner join based on the indexes) the source(having all datapoints) with follow-up(with the left data), the common data points should be allocated to same clusters for both the source and follow-up executions
    #mergedDataFrame = pd.merge(rfm_Source,rfm_Followup,left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("Inconsistent Data Rows at specific index =", (inconsistentOutDataRows.head()).to_string())
    # print(rfm_Source)
    # print(rfm_Followup)


    # =========MR#2: Apply normalization on the normalizeed data =============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR2")
    ## Merge (using inner join based on the indexes) the source(having all datapoints) with follow-up(with the left data), the common data points should be allocated to same clusters for both the source and follow-up executions
    #mergedDataFrame = pd.merge(rfm_Source,rfm_Followup,left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("Inconsistent Data Rows at specific index =", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)


    # =========MR#3: Addition of features by copying/duplicating the original feature set=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR3")
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
    #    rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR4", rfm_Source,rowIndex)
    #    testDataPredictions_Followup = kmeansModel_Followup.predict(testData)
    #    if not(pd.DataFrame(testDataPredictions_Source).equals(pd.DataFrame(testDataPredictions_Followup))):
    #        conflictDataInstances.append(rowIndex)
    #        print("For instance at this index, we got inconsistent outputs = ", rowIndex)
    #print("Instances indexes for which we got different outputs = ", conflictDataInstances)
    #Found data instances indexes for which violations have been found. These are just few found within first 170 iterations
    #For instance at this index, we got inconsistent outputs = 50,65,70,85,99,101,104,155,167
    # ========================================================#

    # ==========Actual MR=============#

    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR4", rfm_Source)
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
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR5")
    #mergedDataFrame = pd.merge(rfm_Source,rfm_Followup,left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#6: Consistent with reprediction=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR6")

    #rfm_SourceMR6, rfm_df_scaled_SourceMR6, kmeansModel_Followup = kmeansAlgorithm("MR6")
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR6Followup")
    #mergedDataFrame = pd.merge(rfm_SourceMR6,rfm_Followup,left_index=True, right_index=True)
    #print(mergedDataFrame.head())
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print(rfm_SourceMR6)
    #print(rfm_Followup)

    # =========MR#7: Shifting the data set features by a constant i.e., x + 2=============#

    #================================================#
    ##To find the exact/actual data rows (thier indexes) for which the model produces inconsistent results for the source and follow-up executions
    ## I run a lot of iterations with different values but found no inconsistency
    #valueToIncrement = 0.0099
    #conflictDataInstances = []
    #for rowIndex in range(100):
    #    rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR7", rfm_Source,rowIndex,valueToIncrement)
    #    mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #    print("Inconsistent = ", (inconsistentOutDataRows.head()).to_string())
    #    if len(inconsistentOutDataRows)>0:
    #        conflictDataInstances.append(valueToIncrement)
    #        print("Value for which we got inconsistent outputs = ", valueToIncrement)
    #    valueToIncrement = valueToIncrement + (0.0097*3)
    #print("Values for which we got different outputs = ", conflictDataInstances)
    #========================================================#

    #==========Actual MR Execution=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR7")
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head()).to_string())


    # =========MR#8: Scaling the data set features by a constant i.e., x * 2=============#
    # ================================================#
    ##To find the exact/actual data rows (thier indexes) for which the model produces inconsistent results for the source and follow-up executions
    ## I run a lot of iterations with different values but found no inconsistency
    #valueToIncrement = 0.0099
    #conflictDataInstances = []
    #for rowIndex in range(100):
    #    rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR8", rfm_Source, rowIndex,
    #                                                                                 valueToIncrement)
    #     mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #    print("Inconsistent = ", (inconsistentOutDataRows.head()).to_string())
    #    if len(inconsistentOutDataRows) > 0:
    #        conflictDataInstances.append(valueToIncrement)
    #        print("Value for which we got inconsistent outputs = ", valueToIncrement)
    #    valueToIncrement = valueToIncrement + (0.0097 * 3)
    #print("Values for which we got different outputs = ", conflictDataInstances)
    #======================================#

    # ==========Actual MR Execution=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR8")
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)


    # =========MR#9:MR_replace=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR9",rfm_Source)
    #rfm_Source = pd.DataFrame(rfm_Source).reset_index()
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#10:Swapping the features=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR10")
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)

    # =========MR#11: Adding informative attribute(s)=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR11",rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#12:Reversing the data-points=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR12")
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows[0:5]).to_string())
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#13: Multiple all the features with -1=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR13")
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#14: Enhancing the compactness of specific cluster=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR14")
    #rfm_Source = pd.DataFrame(rfm_Source).reset_index() # reset the index, as they are not in sequential order but for rfm_Followup it is sequential
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print(rfm_Source)
    #print(rfm_Followup)

    # =========MR#16: Rotate by 45 degrees=============#
    #rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR16")
    #testDataPredictions_Followup = kmeansModel_Followup.predict(testData)
    #print(pd.DataFrame(testDataPredictions_Source).equals(pd.DataFrame(testDataPredictions_Followup)))
    #print(np.where(testDataPredictions_Source != testDataPredictions_Followup))
    #rfm_Source = pd.DataFrame(rfm_Source).reset_index()
    #print(rfm_Source.iloc[703])
    #print(rfm_Followup.iloc[703])

    # =========Removal of features (one or more) should not change the output (this may not be a valid MR, so remove it=============#
    # rfm_Followup, rfm_df_scaled_Followup, kmeansModel_Followup = kmeansAlgorithm("MR6")
    # testDataPredictions_Followup = kmeansModel_Followup.predict(testData.drop(labels=['Frequency'],axis=1))
    # print(pd.DataFrame(testDataPredictions_Source).equals(pd.DataFrame(testDataPredictions_Followup)))
    # print(np.where(testDataPredictions_Source != testDataPredictions_Followup))