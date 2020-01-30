import os


path = r"/home/mina/Downloads/chossen"

for subdir, dirs, files in os.walk(path)
  for file in files:    
      pathtofile =  os.path.join(subdir,file)
      command = "python video_demo.py --video {} --output {}".format(pathtofile,subdir)
      os.system(command)