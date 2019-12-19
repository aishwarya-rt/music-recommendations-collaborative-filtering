import csv
'''j represents the p files'''
for j in range(5,6):
    '''i is the user index,change to the one in your case'''
    for i in range(100,101):
        count=0
        with open('p%d.csv'%j,'r') as fin:
            for row in csv.reader(fin, delimiter=','):
                if row[0] == 'user_000%d' %i:
                    count = count+1
            print ( count )

            if count>250:
                with open('p%d.csv'%j, 'r') as fi ,open ('user%d.csv' %i,'a', encoding='utf-8') as fout :
                    writer = csv.writer(fout, delimiter=',')
                    for row in csv.reader(fi, delimiter=','):
                        if row[0] == 'user_000%d' %i:
                            writer.writerow(row)
