import os
correct = 0
recall = 0
FP = 0
recall_num = 0
total_num = 0
with open('result.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        path, pred = line.split(' ')
        img_name = os.path.basename(path)
        img_num = int(os.path.splitext(img_name)[0])
        total_num += 1
        if img_num <=300:
            if int(pred)==0:
                correct += 1
            else:
                FP += 1
        else:
            recall_num += 1
            if int(pred)==1:
                recall += 1
                correct +=1
print('recall is {} and precision is {} and accuracy is {}'.format(float(recall)/recall_num,float(recall)/(recall+FP),float(correct)/total_num))
