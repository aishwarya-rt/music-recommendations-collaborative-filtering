import pandas as pd
import numpy as np
import collections
#fq=pd.read_csv('FirstQuarter.csv')
sq=pd.read_csv('SecondQuarter.csv')
tq=pd.read_csv('ThirdQuarter.csv')
ftq=pd.read_csv('FourthQuarter.csv')
#fq=fq.head(50)
sq=sq.head(50)
tq=tq.head(50)
fq=ftq.head(50)
print(fq)
songlist=fq['songname'].tolist()
Count_Col=fq.shape[0]
print(Count_Col)
songidcol=[]
count=0
log=0
sid=0
for son in songlist:
    songidcol.append('s%d'%sid)
    sid=sid+1
fq['songid']=songidcol
Count_Row=eval(input("enter min no of users"))
useridrow=[]
for u in range(1,101):
    useridrow.append('user%d'%u)
Matrix = pd.DataFrame(np.zeros((Count_Row, Count_Col)),index=useridrow,columns=songidcol)
'''i is the range of users,change accordingly'''
for i in range(1,101):
    if(i!=61):
        userdf=pd.read_csv('user%d.csv'%i)
    usersonglist=userdf.iloc[:,5].tolist()
    #usersonglist.insert(0, userdf.columns[5])
    counter = collections.Counter(usersonglist)
    j=0
    for j in usersonglist:
            print(log)
            log=log+1
            if (j in songlist):
                c = songlist.index(j)
                Matrix.iloc[i-1][c]=counter[j]
    counter.clear()
print(Matrix)
Matrix.to_csv('ftq.csv', encoding='utf-8')