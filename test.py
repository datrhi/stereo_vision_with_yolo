# Python program to illustrate
# Append vs write mode
# file1 = open("config.txt","w")
# L = ["This is Delhi\n","This is Paris\n","This is London\n"] 
# file1.write("Test\n1\n2")
# file1.close()
  
# Append-adds at last
file1 = open("config.txt","a")#append mode
file1.write("Today \n")
file1.close()
  
file1 = open("config.txt","r")
print("Output of Readlines after appending") 
cfgs = file1.readlines()
print([cfg[:len(cfg)-1] for cfg in cfgs])
# print()
file1.close()
  
# # Write-Overwrites
# file1 = open("config.txt","w")#write mode
# file1.write("Tomorrow \n")
# file1.close()
  
# file1 = open("config.txt","r")
# print("Output of Readlines after writing") 
# print(file1.readlines())
# print()
# file1.close()