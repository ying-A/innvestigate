import os
import matplotlib.pyplot as plt
loglist = "C:\\Users\\trio\\Desktop\\level2_logs"
dirfiles = os.listdir(loglist)
logs  = []
methods_invaccs = []
methods = []
y_pre_accs = []
for x in dirfiles:
   log = os.path.join(loglist,x)
   logs.append(log)
   sub = x.find("_")
   methods.append(x[0:sub])
   with open(log,"r")as flog:
       line = flog.readline()
       method_invaccs = []
       y_pre_accs = []
       while(line):
           _,y_pre_acc,invacc = line.split()
           y_pre_accs.append(float(y_pre_acc))
           method_invaccs.append(float(invacc))
           line = flog.readline()
       methods_invaccs.append(method_invaccs)
epoch = [0,4,8,12,16,20,24,28,32,36]
plt.title("Methods_Acc", fontsize=24)
plt.xlabel("epoch", fontsize=14)
plt.ylabel("acc", fontsize=14)
for i,method_invaccs in enumerate(methods_invaccs):
    plt.plot(epoch,method_invaccs, label=methods[i])
plt.plot(epoch,y_pre_accs, label="y_pre_acc")
plt.legend()
plt.show()

