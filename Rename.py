import os
import sys

PATH_0 = './Images/'
PATH_1 = './Images/Balle_verte/'
PATH_2 = './Images/Balle_rouge/'
PATH_3 = './Images/No_ball/'
PATH_4 = './Images/Balle_bleue/'

new = 0
new_name = PATH+'img'+str(new)+'.png'

for i in range(0,1000):
	try : 
		name = PATH+'img'+str(i)+'.png'
		os.rename(name, new_name)
		new = new + 1 
		new_name = PATH+'img'+str(new)+'.png'
	except :
   		print ("Image non existante")