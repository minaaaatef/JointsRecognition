import os


path = r"/home/mina_atef0/Desktop/AlphaPose/chossen"

count = 0
# command = "source venv/bin/activate"
# os.system(command)
# os.system("which python3")
couldRename = 0

for subdir, dirs, files in os.walk(path,topdown=True):
  for file in files:
      if str(file).endswith('.json'):
        continue
      count += 1    
      print("proccessing file#" + str(count)) 
      print(file) 
      pathtofile =  os.path.join(subdir,file)
      command = "python3 video_demo.py --video {} --outdir {}".format(pathtofile,subdir)
      os.system(command)
      try:
        os.rename(os.path.join(subdir,'alphapose-results.json'),os.path.join(subdir,file+'.json'))
      except:
        couldRename += 1



print("couldn't rename  " +str(couldRename) )

print("made  " +str(count) )