import json
import csv
f= open('test_data.csv', 'w', newline='',encoding='utf-8')
writer = csv.writer(f)
writer.writerow(["sentence","gene","disease","relation","gene_index","disease_index"])
with open(r"C:\Users\Venkatesh Dharmaraj\Downloads\NLP Project\TBGA\TBGA_test.txt",'r') as f:
    for line in f:
        d=json.loads(line)
        # print(d)
        writer.writerow([d['text'],d['h']['name'],d['t']['name'],d['relation'],d['h']['pos'],d['t']['pos']])
