dict_file='vocab.txt'
with open (dict_file,'w') as fdict:
    for i in range(500):
        if i<200:
            fdict.write(str(i) + "\n")
        elif i ==200:
            fdict.write("<@@@>\n")
        else:
            fdict.write(str(i-1)+"\n")
    fdict.write(str(499))
