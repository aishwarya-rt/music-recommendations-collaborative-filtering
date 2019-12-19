# music-recommendations-collaborative-filtering
Leveraged python libraries to perform Data Mining and Modelling techniques on user data from the “last.fm" app to design, analyze and implement an approach for a music recommendation system that takes user clusters’ variables for customized song recommendations. Analyzed various Similarity measures and Clustering algorithms and derived insights using Tableau to optimize the recommendation vector and achieved a recall of 73%.

Problem Statement:
With the increase in the use of e-commerce sites, it has become very easy for the users to find the items of their interest without wasting a lot of time. Websites like Amazon and Ebay examples for Recommender systems which provide recommendations to the users based on their search history and purchase history. Recommender systems provide recommendations of almost all the items ranging from books to movies to music.
When a user tries to find an item using search engines, for example, Google Search engine and Yahoo Search engine, the user needs to type the exact name of the item. The data in the internet is huge which makes it very difficult for the user to find the items of his interest. Hence, there is a need for a system which learns the likes and dislikes of the user and generates recommendations based on his interest. Many algorithms need to be used while designing a recommender system.
Research to implement an algorithm that would address all the facets of music has been tried for quite some time; but it is unreasonable to expect a system to outperform human intuition, which is difficult to quantify. Choices are influenced by human behaviour, Sometimes, and user data in its unprocessed form does not reveal important relationships between two users/items. Also, considering the field of music, where the size of music library is ever increasing, we need a viable linear algebraic approach that can address the computationally intensive approach of recommendation. Gathering data does not require much effort; however processing Big Data is problematic.
The music recommendation system is used to recommend the songs and generate the playlist that the user may likely listen using the user based collaborative filtering. The goal of this system is to compute a scoring function that aggregates the result of computing similarities between users and between items. We focus on reviewing the strategy of user-based collaborative filtering. For experimental purpose we explore different metrics to measure the similarity of users and items such as Euclidean distance, cosine metric Pearson correlation and others. Finally, we compare different evaluations metrics that represent the effectiveness of the recommender system.

Approach:
The approach here is based on certain hypotheses about the factors that can affect the choices of users when they listen to songs. Considering these factors, the algorithm tries to create a more intelligent recommendation than simply suggesting a song or a band to listen to. Generally, music recommenders suggest a song or an artist by plainly noting the users behaviour and the kind of songs they scan through; however, it is important to note that a user may not like all the albums of a particular artist or all the songs on a particular album. The recommendation system should work so that it will give an enriching experience to its listener. Music portal Last.fm is leveraged to collect information and Metadata for approximately thousand users.
This project is an investigation of using collaborative filtering techniques for a music recommender system. Collaborative filtering is the technology that focuses on the relationships between users and between items to make a prediction. The goal of the recommender system is to compute a scoring function that aggregates the result of computing similarities between users and between items. We focus on the reviewing the strategy of collaborative filtering: user-based recommendations. For experimental purpose we explore different metrics to measure the similarity of users and items such as Euclidean distance, cosine metric, Pearson correlation and others. Finally, we compare different evaluations metrics that represent the effectiveness of the recommender system.

Music Recommendation:
We are offered options among different things that we come across throughout a day. We hear song on a radio, see a movie, read about some books, or see different clothes/accessories. We form an opinion: we like them, don’t like them or sometimes we don’t even care. This happens unconsciously. Although these all seem random, we inherently follow a pattern and we call it personal taste. We tend to like similar things. For example, if someone likes bacon-lettuce- tomato sandwiches, then there are good chances that that person will also like a club sandwich because they are very similar only with turkey replacing the bacon. We follow these kinds of patterns inherently. In the crux, recommendations are all about pattern recognition and finding similarities.

Music is omnipresent. It is no surprise that there are millions of songs at everyone’s fingertips. In  fact,  given   the  number  of  songs,  bands,  and  artists  coming  up,  music  listeners  are overwhelmed by choices. They are always looking for ways to discover new music so that it will match their taste. This has given birth to the field of music recommendations. In the past few years, there have been many services like Pandora, Spotify, and Last.fm that have come up in order to find a perfect solution, but haven’t been completely successful. Choices are influenced by interests, trust, and liking towards any particular object and these emotions are very difficult to quantify especially for a machine or software. Hence, it has been a very difficult experience   for   these   service   providers   to   give   a   fulfilling   experience.   Every   music recommendations system works on a given set of hypotheses, which they believe will result in the effective recommendations.
There are two fundamental styles of music recommendations:
Collaborative filtering and Content-based Filtering. The next section describes the two methodologies used by existing music recommender. 

Collaborative Filtering:
It is an approach in which information is gathered about the users’ preferences for any particular item (books, videos, news articles, songs, etc.). The knowledge captured is then structured and used against all the unknown items and make intelligent predictions that a user might enjoy. In collaborative filtering, the interaction between users and items is important. The system relies on the past history to derive a suitable model for an entity. The historical data acts as an input to the system. The preference or user history can be derived in two ways:
A typical Collaborative filtering system would have a data bank of the user’s preferences for all the songs they have browsed or purchased. Essentially, we have a list of m users {u1, u2, u3 …um} along with a list of n songs {s1, s2, s3… sn}. As described previously, with the help of implicit or explicit ratings, user preferences are noted for all items i. The preference vectors of all the users are then converted to user-item matrix. 

Preprocessing:
This dataset contains <user, timestamp, artist, song> tuples collected from Last.fm API. Since the dataset is too large to be processed we are splitting the million song dataset into 20 files with 5,00,000 records each.
Total Lines:19,150,868
Unique Users:992
Artists with MBID:107,528
Artists without MBDID:69,420
Since the dataset is too large to be processed we are splitting the million song dataset into 40 files with 5,00,000 records each. Here we take the 250 Mb .tsv MSD into 40 separate .tsv files with 5 Lakhs records in each.
The Split Current Document to Several Files command on EmEditor allows you to split the current document into several files either every user-specified number of lines, or before every bookmarked line. It also allows you to specify a header and/or footer to each separated file. The new Combine Documents into a Single File command allows you to combine multiple documents into one file. Both features use an intuitive wizard that allows you to control the parameters of each operation.

Now in this project we are considering 100 users out of 992 users because it will be easier for us to form the User-Item matrix. For this we have split the 40 files containing 5 lakhs records each into separate files for each user i.e., 100 .csv files after conversion . We used python to make this conversion and splitting. 

The recommender references training data to generate recommendation and the performance is measured on test sessions. We set the experimental settings with an intention to make similar situations in the industries when they use recommendation techniques. Each log record in the log table contains user identification, song identification, username, timestamp, album name, album identification. 
We removed the records in which the users who has listened less than 250 songs and the songs which were listened by less than 10 users. We consider when only the log records which have the users and songs which satisfy the above criteria. Thus, by removing these unwanted logs, we ignore those data when calculating the rating of items.
After data pre processing, the training data set contains a total of  5,503  items and 100 users. The detailed statistical information of training dataset is described below. 

Statistical Information of Datasets
Description Number of Records before pre processing
Number of Records :       19,150,868
Number of items :           2,31,681 
Number of users :            992
Description Number of Records after pre processing
Number of distinct items :     5,503
Number of distinct users :     100
The dataset for this project contains activities of users whose listening history has been captured. For every song that a user listens to, its activity is recorded in the following format:

User Id-Each user is assigned a user Id.
Song Id – A unique identifier is attributed to each Song.
Time Stamp– The time for which the user listened to the song is entered.
Album Id -  A unique identifier is attributed to each album to which song belongs to.
Album name - A name to a particular album is assigned.
Song name - A name to a particular song is assigned.   

Formation of Sessions :
	User logs obtained from pre-processing of data are divided into sessions. A session is defined as fixed time slot of a day. Once pre-processing of data is done, for each user i, where i is such that 0 < i < 1000, we use the timestamps to perform an analysis to get a suitable threshold value of a session length. We can define a session such that the difference between the timestamps of any two consecutive songs is not greater than the threshold session value decided above. Here, we are working on a hypothesis that the users' choices of songs is influenced by external factors and that there exists a degree of correlation between any two songs that are listened to in the same user-sessions. Multiple such sessions are formed for each user i in the database as each of them has a listening history that spans over two years. We have taken four sessions for each day of equal intervals i.e from 0 a.m to 6 a.m as S1, 6a.m to 12 p.m as S2 , 12 p.m to 18 p.m as S3 and 18p.m to 24p.m as S4. We store all the user sessions in a flat file that falls into one of the three broad session’s blocks and process them to extract the pair of all the songs that are played together in that particular session.
Number of sessions formed: 4
Session 1: 
Number of items :    2059
Number of users :     33
Session 2: 
Number of items :    2221  
Number of users :     34
Session 3: 
Number of items :     5231
Number of users :     38
Session 4: 
Number of items :    4862 
Number of users :     38


Song List Generation :
In order to generate the user item matrix we to generate Song List of unique songs and frequency of different users corresponding to the song. We used the pandas package and added all the data to a big dataframe and removed the duplicates. Now we performed a group by and found the  count of each song.

Generating User-Item Matrix Stage 
Once we get the pairs of songs for each user i, we compose a user vector which consists of all the songs that are played in the user's history. Then we cross match it with the top  songs. We construct a user-item matrix for users x  songs; so that the value in each cell aij in the matrix is directly proportional to the number of times a user i has listened to song j. We call this matrix as Matrix M, which is a sparse matrix. We have considered 100 users and 5,503 songs for the construction of the matrix. 
We have imported Numpy, Pandas and Collections packages. Numpy is Numeric python used for higher mathematical operations on data structures. Pandas is used to handle dat frames. Collections has the operations to handle data. For making the matrix we started off by making a dataframe with count greater than the minimum number of songs. Using the filtered song-count list and collections.counter we populated a user-song matrix with respective user frequencies for corresponding songs.

Uploads: UserItemMatrix.png

Normalization Stage 
In statistics and applications of statistics, normalization can have a range of meanings. In the simplest cases, normalization of ratings means adjusting values measured on different scales to a notionally common scale, often prior to averaging. In more complicated cases, normalization may refer to more sophisticated adjustments where the intention is to bring the entire probability distributions of adjusted values into alignment. In the case of normalization of scores in educational assessment, there may be an intention to align distributions to a normal distribution. A different approach to normalization of probability distributions is quantile normalization, where the quantiles of the different measures are brought into alignment.

In general three types of normalizations can be used.

1. Min-max Normalization :  Min-max algorithm transforms the data set from one range to another.
People generally go for 0–1 range when v’=(v-min)/(max-min) is the new value for a particular value in the data set.
            v’ = (v-min)/(max-min) * (newmax-newmin) + newmin

Uploads: NormalizationMinMax.png

 2.  Root Mean Square Normalization :  Root mean square normalization is used    for normalizing the data. Normalization is generally done to get the complete data into one form so that the data can be used accurately. RMSN=(√(∑Ri*Ri))/n. Normalization is scaling technique or a mapping technique or a pre processing stage. 
def normalize(d) :
    	df_norm=(((d-d.mean())**2)/d.shape[1])**(½) , return df_norm.
      
Uploads: RMSNormalization.png

3. Normalization using Standard Scaler : The result of standardization (or Z-score normalization) is that the features will be rescaled so that they’ll have the properties of a standard normal distribution with
μ=0μ=0 and σ=1σ=1
where μμ is the mean (average) and σσ is the standard deviation from the mean; standard scores (also called z scores) of the samples are calculated as follows:
z=x−μσz=x−μσ
Standardizing the features so that they are centered around 0 with a standard deviation of 1 is not only important if we are comparing measurements that have different units, but it is also a general requirement for many machine learning algorithms. Intuitively, we can think of gradient descent as a prominent example (an optimization algorithm often used in logistic regression, SVMs, perceptrons, neural networks etc.); with features being on different scales, certain weights may update faster than others since the feature values xjxj play a role in the weight updates
Δwj=−η∂J∂wj=η∑i(t(i)−o(i))x(i)j,Δwj=−η∂J∂wj=η∑i(t(i)−o(i))xj(i),
def normalize(d) :
 	data_norm = df  # Has training + test data frames combined to form single data frame
    	normalizer = StandardScaler()
    	data_array = normalizer.fit_transform(data_norm.as_matrix())
   	return pd.DataFrame(data_array)

Uploads: SSNormalization

K-Fold Cross Validation:
Cross-validation, sometimes called rotation estimation, is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice. In a prediction problem, a model is usually given a dataset of known data on which training is run (training dataset), and a dataset of unknown data (or first seen data) against which the model is tested (called the validation dataset or testing set). The goal of cross validation is to define a dataset to "test" the model in the training phase (i.e., the validation set), in order to limit problems like overfitting, give an insight on how the model will generalize to an independent dataset (i.e., an unknown dataset, for instance from a real problem), etc.
One round of cross-validation involves partitioning a sample of data into complementary subsets, performing the analysis on one subset (called the training set), and validating the analysis on the other subset (called the validation set or testing set). To reduce variability, in most methods multiple rounds of cross-validation are performed using different partitions, and the validation results are combined (e.g. averaged) over the rounds to estimate a final predictive model.
K fold cross validation is performed for selecting testing and training data (here k =10). Users are divided into 10 folds, where 7 folds will  be used for training and 3 folds will be used for test data. As the iterations proceed, the 7:3 ratio will be altered to various values. We have considered the following cases,
Case 1:  We have used random training data set of 70% and test data set of 30%.
Case 2:  We have used random training data set of 80% and test data set of 20%.
Case 3:  We have used random training data set of 80% and test data set of 20%.

Clustering Stage:
Now we apply the clustering algorithm on all the data points of Matrix m x n. We just use Matrix V, as our algorithmic approach is to find the similarities between all of the required songs and use that data as the corner stone of our recommendation system. We form user clusters and song clusters with some prescribed conditions and  threshold values. After clustering, we get the group of all the similar songs. We store all the clusters into a flat file for the future decision making process (recommendations). 
The process of grouping a set of physical or abstract objects into classes of similar objects is called clustering. A cluster is a collection of data objects that are similar to one another within the same cluster and are dissimilar to the objects in other clusters. A cluster of data objects can be treated collectively as one group and so may be considered as a form of data compression. Although classification is an effective means for distinguishing groups or classes of objects, it requires the often costly collection and labelling of a large set of training tuples or patterns, which the classifier uses to model each group. It is often more desirable to proceed in the reverse direction: First partition the set of data into groups based on data similarity (e.g., using clustering), and then assign labels to the relatively small number of groups. Additional advantages of such a clustering-based process are that it is adaptable to changes and helps single out useful features that distinguish different groups.

Centroid-Based Technique: The k-Means Method
The k-means algorithm takes the input parameter, k, and partitions a set of n objects into k clusters so that the resulting intracluster similarity is high but the inter cluster similarity is low. Cluster similarity is measured in regard to the mean value of the objects in a cluster, which can be viewed as the cluster’s centroid or center of gravity. “How does the k-means algorithm work?” The k-means algorithm proceeds as follows.
	First, it randomly selects k of the objects, each of which initially represents a cluster mean or center. For each of the remaining objects, an object is assigned to the cluster to which it is the most similar, based on the distance between the object and the cluster mean. It then computes the new mean for each cluster. This process iterates until the criterion function converges. Typically, the square-error criterion is used, defined as 
E = i=1n    p∈ci  |p-mi|2
Example: Clustering by k-means partitioning. Let k = 3; that is, the user would like the objects to be partitioned into three clusters.
	According to the algorithm, we arbitrarily choose three objects as the three initial cluster centers, where cluster centers are marked by a “+”. Each object is distributed to a cluster based on the cluster center to which it is the nearest. Such a distribution forms silhouettes encircled by dotted curves. Next, the cluster centers are updated. That is, the mean value of each cluster is recalculated based on the current objects in the cluster. Using the new cluster centers, the objects are redistributed to the clusters based on which cluster center is the nearest. Such redistribution forms new silhouettes encircled by dashed curves.
This process iterates, the process of iteratively re-assigning objects to clusters to improve the partitioning is referred to as iterative relocation. Eventually, no redistribution of the objects in any cluster occurs, and so the process terminates. The resulting clusters are returned by the clustering process. The algorithm attempts to determine k partitions that minimize the square-error function. It works well when the clusters are compact clouds that are rather well separated from one another. The method is relatively scalable and efficient in processing large data sets because the computational complexity of the algorithm is O(nkt), where n is the total number of objects, k is the number of clusters, and t is the number of iterations.  Normally, k<n and t <n. The method often terminates at a local optimum.

K-Means Algorithm :
1: Initialize K points as initial centroids
2: repeat
3:	Form K clusters by assigning each point to its closest centroid
4:	Recompute the centroid of each cluster.
5: until centroids do not change
Method:
(1) Arbitrarily choose k objects from D as the initial cluster centers;
(2) Repeat
(3) (Re) assign each object to the cluster, to which the object is the most similar , based on the mean value of the objects in the cluster;
(4) Update the cluster means, i.e., calculate the mean value of the objects for each  cluster; until no change;                         

The k-means method, however, can be applied only when the mean of a cluster is defined. This may not be the case in some applications, such as when data with categorical attributes are involved. The necessity for users to specify k, the number of clusters, in advance can be seen as a disadvantage. The k-means method is not suitable for discovering clusters with non convex shapes or clusters of very different size. Moreover, it is sensitive to noise and outlier data points because a small number of such data can substantially influence the mean value.
There are quite a few variants of the k-means method. These can differ in the selection of the initial k means, the calculation of dissimilarity, and the strategies for calculating cluster means. An interesting strategy that often yields good results is to first apply a hierarchical agglomeration algorithm, which determines the number of clusters and finds an initial clustering, and then use iterative relocation to improve the clustering.
Another variant to k-means is the k-modes method, which extends the k-means paradigm to cluster categorical data by replacing the means of clusters with modes, using new dissimilarity measures to deal with categorical objects and a frequency-based method to update modes of clusters. The k-means and the k-modes methods can be integrated to cluster data with mixed numeric and categorical values.The EM (Expectation-Maximization) algorithm extends the k-means paradigm in a different way. Whereas the k-means algorithm assigns each object to a cluster, in EM each object is assigned to each cluster according to a weight representing its probability of membership. In other words, there are no strict boundaries between clusters. Therefore, new means are computed based on weighted measures.	
  
User-based Clusters 
Each user from the user-item matrix of one of the session (S1,S2,S3 S4) is considered as a user vector. User clusters for a session are formed by using the following hierarchical agglomerative clustering algorithm 
Algorithm User_clusters_with Sessions() 
Input: User-Item Matrix of a particular session 
Output: User Clusters 

Method: 
Begin 
1. Consider each user vector I1, I2,..Ik where k is the number of distinct items rated by all users 
2. Initialize threshold_cutoff value 
3. Consider the first user and put in C1 
4. For all remaining users repeat the steps from 4 to 8 
5. Find the similarity of the useri with all the clusters formed so far 
6. Put the useri in the cluster with more similarity 
7. If the useri is not in the threshold value of any cluster 
8. Create a new cluster 
    end 

Output
Clusters at k=5 :[1, 2, 6, 8, 11, 12, 14, 17, 21, 23, 24, 25, 26, 29, 31, 32, 33, 35, 39, 40, 41, 42, 49, 52, 53, 55, 56, 58, 59, 60, 63, 64, 65, 66, 67], [13], [0, 5, 7, 9, 10, 15, 16, 19, 20, 30, 34, 36, 38, 43, 44, 45, 46, 54, 57, 61, 69], [28], [3, 4, 18, 22, 27, 37, 47, 48, 50, 51, 62, 68]]

Clusters at k=6 : [[2, 3, 5, 6, 8, 9, 11, 14, 15, 19, 20, 22, 27, 28, 32, 35, 42, 48, 49, 50, 55, 56, 57, 59, 62, 63, 64, 65, 67, 68, 69], [4, 26, 29, 37], [0, 1, 10, 12, 13, 16, 17, 18, 21, 23, 24, 25, 30, 31, 33, 34, 36, 39, 43, 44, 45, 46, 47, 51, 52, 54, 58, 60, 61, 66], [7], [38], [40, 41, 53]]

Clusters at k=7 :
[[0, 1, 2, 3, 6, 11, 13, 14, 15, 18, 20, 22, 24, 26, 28, 30, 35, 38, 42, 45, 46, 51, 60, 63, 66], [5, 8, 9, 12, 16, 17, 19, 21, 23, 31, 32, 33, 36, 37, 39, 41, 44, 47, 52, 53, 54, 55, 56, 57, 58, 61, 62, 65, 67, 68, 69], [25], [27], [43], [4, 7, 10, 29, 34, 40, 48, 49, 50, 64], [59]]

Recommendation using Most popular songs
Another Approach is to use the data frame containing the Most popular songs i.e the songs that have been listened to by many users and recommending only those songs to the new users. In this the data transformation starts by, selecting a subset of the data (ex: the first 10,000 songs). We then merge the song and artist_name into one column, aggregated by number of time a particular song is listened too in general by all users.

Doing data transformation allows us to further simplify our dataset and make it easy and simple to understand. Next step, we’re going to follow a naive approach when building a recommendation system. We’re going to count the number of unique users and songs in our subset of data. We arbitrarily pick 20% as our testing size. We then used a popularity based recommender class as a blackbox to train our model. We create an instance of popularity based recommender class and feed it with our training data. 
This system is a naive approach and not personalized. It first get a unique count of user_id (ie the number of time that song was listened to in general by all user) for each song and tag it as a recommendation score. The recommend function then accept a user_id and output the top ten recommended song for any given user. Keeping in my that since this is the naive approach, the recommendation is not personalized and will be the same for all users.
Recommendation using Time-Sessions songs
In user-based collaborative filtering algorithm, current users' nearest neighbors are used to recommend items because they have similar preference, but users' preference varies with time, which often affects the accuracy of the recommendation. As a result of the varying users' preference, many researches about recommendation systems are focusing on the time factor, to find a way to make up for the change in preferences of users.
During the recommendation process, the proposed model orders the items by time for each user as a sequence. The sequence is called time-behaviour sequence.
First it finds the last item from current user's time-behaviour sequence which represents the newest preference of the current user.
Secondly, it locates the item in nearest neighbours’ timebehavior sequence and saves the timestamp of the item.
Lastly, it recommends the items whose timestamps are greater than the saved timestamp from the nearest neighbours’ time-behaviour sequence.
It can catch the newest preferences of the users and increase the accuracy of recommendation, without changing the training phase.
 
 Here, we are considering only the top four songs with the highest lift values i.e
S156813, S451721, S582699, S652635
• Now the unique songs heard by the above users are
user_13, user_7, user_55, user_89, user_S56, user_36, user_98, user_180,user_49,…....
• Cluster 2 contains the above unique songs in a greater number, therefore we need to
recommend those top k songs of the cluster 2.
 
KNN Algorithm 
KNN is a machine learning algorithm to find clusters of similar users based on common songs ratings, and make predictions using the average rating of top-k nearest neighbours. For example, we first present ratings in a matrix with the matrix having one row for each item and one column for each user.
We then find the k item that has the most similar user engagement vectors. In this case, Nearest Neighbors of item id 5= [7, 4, 8, …]. Starting from the original data set, we will be only looking at the popular songs. In order to find out which songs are popular, we combine songs data with ratings data(count). We combine the rating data with the total rating count data, this gives us exactly what we need to find out which songs are popular and filter out lesser-known songs
 
Implementing KNN
We convert our table to a 2D matrix, and fill the missing values with zeros (since we will calculate distances between rating vectors). We then transform the values(ratings) of the matrix data frame into a scipy sparse matrix for more efficient calculations.
Finding the Nearest Neighbors
We use unsupervised algorithms with sklearn.neighbors. The algorithm we use to compute the nearest neighbors is “brute”, and we specify “metric=cosine” so that the algorithm will calculate the cosine similarity between rating vectors. Finally, we fit the model. 
Test our model and make some recommendations
In this step, the KNN algorithm measures distance to determine the “closeness” of instances. It then classifies an instance by finding its nearest neighbors, and picks the most popular class among the neighbors
.Algorithm
1: Classify (X,Y,x) X:Training data, Y: class labels of X
2: for i=1 to m do
3:	Compute distance d(Xi,x)
4: end for
5: Compute set I containing indices for the k-smallest distances d(Xi,x)
6: Return majority label for {Yi  where i ε I }


Mean Vector Generation
Mean vector is a central tendency obtained by mean of all vector attributes of the vectors of the respective clusters. For this firstly we sum all the vectors in a cluster and divide each vector attribute with the number of vectors in the cluster to get the mean vector of the respective cluster.


Recommending Songs Stage:
The songs are sorted based on the frequency count and top-n songs are recommended. Further calculating and comparing the test users for each test user and  for K-folds.  Once we have constructed the model from user and song Clusters, we can give recommendations based on the patterns observed in the user's history and session information that can be extracted from the above process. At recommendation stage test set users and songs objects are retrieved and test user song matrix is generated. From this each user vector is considered one by one and their songs are compared with generated clusters. Depending on comparison result songs of corresponding cluster are recommended. In this system we are going to recommend  songs to the users
Recommendations from user clusters
We select songs from user clusters. After forming user clusters the user is mapped to one of the cluster and other user’s song vectors present in this clusters are considered and frequent items among the recommended to the user.
Recommendation stage 
After getting the user clusters and item clusters for each session, we use these clusters to recommend items to new users by using the following Algorithm for recommendations. 

Algorithm Recommendation_ with Sessions () 
Input: User Clusters and Item Clusters 
Output: Set of Recommendations for new users 
Method: 
begin 
1. map the new user to the user clusters to which he/she is most similar 
2. map the new user to the item cluster based on the items listened 
3. consider the recommendations from step1 i.e user clusters and step2 i.e item clusters for the new user 
4. Let I1, I2, ….Ik are the items which are common in both recommendations ( user clusters and item clusters )
5. recommend the common items to the new user.
end

Similarity Measures
	Before clustering, a similarity/distance measure must be determined. The measure reflects the degree of closeness or separation of the target objects and should correspond to the characteristics that are believed to distinguish the clusters embedded in the data measure is also crucial for cluster analysis, especially for a particular type of clustering algorithms.
Some of the similarity measures are:
i) Simple matching coefficient, distance
	Simple matching coefficient and simple matching distance. In many cases, these characteristics are dependent on the data or the problem context at hand, and there is no measure that is universally best for all kinds of clustering problems.
Moreover, choosing an appropriate similarity are useful when both positive and negative values carried equal information .for example , gender has symmetry attribute because number of male and female give equal information.
Given two vectors of Boolean (there or not) features, and summarizing variables:
Example: p and q, have only binary attributes
Compute similarities using the following quantities
		M01 = the number of attributes where p was 0 and q was 1
		M10 = the number of attributes where p was 1 and q was 0
		M00 = the number of attributes where p was 0 and q was 0
		M11 = the number of attributes where p was 1 and q was 1
Simple Matching and Jaccard Coefficients
(SMC)  = number of matches / number of attributes
             = (M11 + M00) ((M01 + M10 + M11 + M00)) 
ii) Jaccard Similarity Coefficient
	The Jaccard coefficient, which is sometimes referred to as the Tanimoto coefficient, measures similarity as the intersection divided by the union of the objects. For text document, the Jaccard coefficient compares the sum weight of shared terms to the sum weight of terms that are present in either of the two documents but are not the shared terms.
J = number of 11 matches / number of not-both-zero attributes values
   = (M11) (M01 + M10 + M11)
iii) Pearson Correlation Coefficient
	Pearson’s correlation coefficient is another measure of the extent to which two vectors are related. There are different forms of the Pearson correlation coefficient formula .This is also a similarity measure. However, unlike the other measures, it ranges from +1 to −1.
To compute correlation, we standardize data objects, p and q, and then take their dot product
                              pk'=pk-mean(p)std(p)
                                       qk'=(qk-meanq)std(q) 
                                           Correlation(p,q)=p'.q' 		
iv) Manhattan Distance
		The distance between two points measured along axes at right angles. In a  plane with p1 at (x1, y1) and p2 at (x2, y2), it is 
                             	|x1 - x2| + |y1 - y2|
Although we have many similarity measuring algorithms we are dealing with only Cosine similarity and Euclidean distance as we are using them for clustering in our project. 
v) Cosine Similarity
	The cosine similarity between two vectors (or two documents on the Vector Space) is a measure that calculates the cosine of the angle between them. This metric is a measurement of orientation and not magnitude.
                                          a.b= |a||b| cos θ)|a||b|            Here a,b are vectors
	       
vi) Supremum Distance 
The Minkowski distance is a generalization of the Euclidean distance.
With the measurement,  xik ,  i = 1, … , N,  k = 1, … , p, the Minkowski distance is
where λ ≥ 1.  It is also called the Lλ metric.
λ = 1 : L1 metric, Manhattan or City-block distance.
λ = 2 : L2 metric, Euclidean distance.
λ → ∞ : L∞ metric, Supremum distance.
limλ→∞= k=1pxik-xjk1= max(∣∣xi1-xj1∣∣,...,∣∣xip-xjp∣∣) 



vii) JPSP Simiarity
  Similarity(x,y) = Proximity(x,y) * Significance(x,y) * Popularity(x,y)
  Proximityx,y= 1(1+e-Rxp-Ryp )
  Significancex,y= 11+e(-|(R(xp)-R(med))-(R(yp)-R(med))|)
  Popularityx,y= 11+e(-|R((xp)-R(yp))/2-Up)

Sim(x,y)jpsp = Sim(x,y) * Jaccard Coefficient

Evaluation - Testing the Model
Root Mean Square Error :
Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are :
                                               RMSE =(I=1N (Zfi - zoi)2N)
Precision : Precision takes all retrieved documents into account, but it can also be evaluated at a given cut-off rank, considering only the topmost results returned by the system. This measure is called precision at n or P@n. 
Precision= TPTP+FP
Recall :  Recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.
Recall= TPTP+FN
F-Measure : A measure that combines precision and recall is the harmonic mean of precision and recall, the traditional F-measure or balanced F-score:
F-Measure= 2RPR+P


