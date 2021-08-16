import pickle
fr=open('pissd_kitti_results.pkl','rb')
inf = pickle.load(fr)
doc = open('pissd_kitti_results.txt', 'a')
print(inf, file=doc)
