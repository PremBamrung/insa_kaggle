import os 

message=input("Submission message : ")

command=f'kaggle competitions submit -c defi-ia-insa-toulouse -f submissions.csv -m "{message}"'
os.system(command)

print("Submited.")


