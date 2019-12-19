import numpy as np
import pandas as pd
import collections
count=0
log=0
sid=0
min_songs=10#eval(input("Enter minimum no of songs"))
df = pd.read_csv('sl2.csv')
df=df[df['count'] >= min_songs]
print(df)
songlist=df['songname'].tolist()
Count_Col=df.shape[0]
print(Count_Col)
songidcol=[]
for son in songlist:
    songidcol.append('s%d'%sid)
    sid=sid+1
df['songid']=songidcol
Count_Row=eval(input("enter min no of users"))
useridrow=[]
for u in range(1,101):
    useridrow.append('user%d'%u)
Matrix = pd.DataFrame(np.zeros((Count_Row,Count_Col)),index=useridrow,columns=songidcol)
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
Matrix.to_csv('UIMatrix_sctest.csv', encoding='utf-8')
df.to_csv('postmatriclisttest.csv',encoding='utf-8')