import csv
import pandas as pd
useridlist = []
songnamelist = []
songindex=[]
timestamplist = []
sid=0
'''change the below range from 6 to the number of last file+1'''


'''it is necessary all the split files are in encoding-utf-8,normally fies exist
in this format but once google how to convert csv files to encoding utf-8 ani'''
for i in range (1,6):
        if i==4 or i==5:
            with open('p%d.csv'%i, mode='r') as infile:
                    reader = csv.reader(infile)
                    for rows in reader:
                        useridlist.append(rows[0])
                        songnamelist.append(rows[5])
                        timestamplist.append(rows[1])
        else :
            with open('p%d.csv' % i, mode='r',encoding='utf-8') as infile:
                reader = csv.reader(infile)
                for rows in reader:
                    useridlist.append(rows[0])
                    songnamelist.append(rows[5])
                    timestamplist.append(rows[1])
print(i)
d=pd.DataFrame()
d['userid'] = useridlist
d['songname'] = songnamelist
d['timestamp'] = timestamplist
print(d)
#d=d.drop_duplicates(keep='first')
d[['timestamp','time']] = d['timestamp'].str.split('T',expand=True)
d[['hours','minutes','seconds']] = d['time'].str.split(':',expand=True)
#d.columns = ['hours','minutes','seconds']
print(d)
#df = d.groupby(['songname']).size().reset_index(name='count')
d.to_csv('songtimelist.csv')
