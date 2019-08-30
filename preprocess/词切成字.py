def is_eng(word):
    flag=True
    for uchar in word:
        if uchar < u'\u0061' or uchar > u'\u007a':
            flag=False
    return flag
fin=open('data/train-content-org.txt')
fout=open('data/train-label-org.txt','w')
stop_words=list('abcdefghijklmnopqrstuvwxyz我的和与于之')
for line in fin.readlines():
    chars=[]
    line=line.strip()
    for word in line.split():
        if word not in stop_words:
            if is_eng(word)==True:
                chars.append(word)
            else:
                chars.extend(list(word))
    fout.write(' '.join(chars)+'\n')

fout.close()