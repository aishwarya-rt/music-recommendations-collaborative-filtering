import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
import nltk

from sklearn.cluster import KMeans

'''three different methods for normalization'''


def normalize(d):
    '''df_norm = (((d - d.mean()) ** 2) / d.shape[1]) ** (1 / 2)
    return df_norm'''

    x = d.values  # returns a numpy array
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

    '''data_norm = df  # Has training + test data frames combined to form single data frame
    normalizer = StandardScaler()
    data_array = normalizer.fit_transform(data_norm.as_matrix())

    return pd.DataFrame(data_array)'''

def euclidean_distance(train,test):
    train = np.asarray(train)
    test = np.asarray(test)
    temp = train - test
    temp = [x ** 2 for x in temp]
    temp = np.divide(temp, len(temp))
    dist = np.sum(temp)
    dist = dist ** (1 / 2)
    return dist.tolist()


def cosine_similarity(train, test):
    train = np.asarray(train)
    test = np.asarray(test)
    temp = train * test
    d = np.sum(temp)
    train = [x ** 2 for x in train]
    d1 = np.sum(train)
    d1 = d1 ** (1 / 2)
    test = [x ** 2 for x in test]
    d2 = np.sum(test)
    d2 = d2 ** (1 / 2)
    return d / (d1 * d2)


def pearson_correlation(train, test):
    train = np.asarray(train)
    test = np.asarray(test)
    n = train.size
    temp = train * test
    d = np.sum(temp)
    d1 = np.sum(train)
    d2 = np.sum(test)
    numerator = n * d - d1 * d2
    train = [x ** 2 for x in train]
    d11 = np.sum(train)
    test = [x ** 2 for x in test]
    d21 = np.sum(test)
    denominator = (d11 - (d1 ** 2)) * (d21 - (d2 ** 2))
    denominator = denominator ** 1 / 2
    return numerator / denominator


def manhattan_distance(train, test):
    train = np.asarray(train)
    test = np.asarray(test)
    temp = train - test
    temp = np.absolute(temp)
    d = np.sum(temp)
    return d


def jaccard_coefficient(train, test):
    return sklearn.metrics.jaccard_similarity_score(train, test)


def evaluate_nearest(tdf,test,nc):
    min = 0
    flag = nc
    sum = [0 for k in range(tdf.shape[1])]
    for j in range(nc):
        for i in range(len(tdf)):
            templ=tdf.iloc[i]
            temp = euclidean_distance(templ,test)
            if (temp < min):
                min = temp
                flag = i
        tempk=tdf.iloc[flag].tolist()
        sum=np.add(sum,tempk)
        tdf = tdf.drop(tdf.index[[flag]])
        print('Shape : ',tdf.shape[0],tdf.shape[1])
    sum=sum/nc
    return sum

'''root mean square error'''


def rmse(test, rv):
    temp = 0
    sum = 0
    n = len(rv)
    for i in range(0, n):
        temp = (rv[i] - test[i]) ** 2
        temp = temp / n
        sum = sum + temp
    sum = sum ** 1 / 2
    return sum


'''mean absolute error'''


def mae(test, rv):
    temp = 0
    sum = 0
    n = len(rv)
    for i in range(0, n):
        temp = (rv[i] - test[i])
        temp = temp / n
        sum = sum + temp
    return abs(sum)


'''precision recall f-measure'''


def prfm(test, rv):
    t = listtobinary(test)
    r = listtobinary(rv)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n = len(t)
    for i in range(0, n):
        if (t[i] == 1 and r[i] == 1):
            tp = tp + 1
        elif (t[i] == 1 and r[i] == 0):
            fn = fn + 1
        elif (t[i] == 0 and r[i] == 1):
            fp = fp + 1
        elif (t[i] == 0 and r[i] == 0):
            tn = tn + 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fm = (2 * precision * recall) / (precision + recall)
    prfm = [precision, recall, fm, tp, tn, fp, fn]
    print(prfm)
    return prfm


def listtobinary(listpassed):
    temp = listpassed
    for i in range(0, len(temp)):
        if (temp[i] != 0):
            temp[i] = 1
    return temp

rma=0
ma=0
pa=0
ra=0
fa=0
itercount=10
for q in range(itercount):
    df = pd.read_csv('UIMatrix_selectedcolumns.csv')
    tsidcol = list(df.columns.values)
    #del tsidcol[1]
    '''df=df.sample(70)
    print(df)'''
    d = df._get_numeric_data()
    d = normalize(d)
    print(d)
    traindf = d.sample(frac = 0.9)
    testdf = d.drop(traindf.index)
    neighbour_count  = 5
    row_count = traindf.shape[0]


    rvdfl = []

    for i in range(testdf.shape[0]):
        test = testdf.iloc[i]
        recommend_vector = []
        recommend_vector = evaluate_nearest(traindf,test,neighbour_count)
        # print(recommend_vector)
        print(i)
        rvdfl.append(recommend_vector)
    rvdf = pd.DataFrame(rvdfl)

    print('Recommended vector :')
    print(rvdf)

    '''rvdf = pd.DataFrame(rvdfl, columns=tsidcol)
    print('Recommended vector :')
    print(rvdf)
    print('testdf:', testdf)
    testdf = np.asarray(testdf)
    testdf = pd.DataFrame(testdf, columns=tsidcol)
    print(testdf)
    sdf = pd.read_csv('postmatriclistsc10.csv')
    sdf = sdf.sort_values(by=['count'], ascending=False)
    columns = 60  # eval(input("enter columns"))
    sdf = sdf.head(columns)
    sidlist = sdf['songid'].tolist()
    print(sidlist)
    testdf = testdf[testdf.columns.intersection(sidlist)]
    print('testdf :', testdf)
    rvdf = rvdf[rvdf.columns.intersection(sidlist)]
    print('rvdf :', rvdf)'''
    '''rmse'''
    sum = 0
    for i in range(testdf.shape[0]):
        sum = sum + rmse(testdf.iloc[i], rvdf.iloc[i])
    rmsev = sum / testdf.shape[0]
    print(testdf.shape[1])
    print('rmse : ', rmsev)
    '''mae'''
    sum = 0
    for i in range(testdf.shape[0]):
        sum = sum + mae(testdf.iloc[i], rvdf.iloc[i])
    maev = sum / testdf.shape[0]
    print('mae:', maev)
    '''prfm'''
    precisiont = 0
    recallt = 0
    fmt = 0
    for i in range(testdf.shape[0]):
        templist = []
        templist = prfm(testdf.iloc[i], rvdf.iloc[i])
        precisiont = precisiont + templist[0]
        recallt = recallt + templist[1]
        fmt = fmt + templist[2]
    precisionv = precisiont / testdf.shape[0]
    recallv = recallt / testdf.shape[0]
    fmv = fmt / testdf.shape[0]
    print('Precision :', precisionv, 'Recall :', recallv, 'F-Measure :', fmv)
    rma = rma + rmsev
    ma = ma + maev
    pa = pa + precisionv
    ra = ra + recallv
    fa = fa + fmv
print(rma / itercount, ma / itercount, pa / itercount, ra / itercount, fa / itercount, )

