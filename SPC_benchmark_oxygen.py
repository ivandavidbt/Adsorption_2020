#SPC MODEL

import numpy
import re
import json
from matplotlib import pyplot
#from collections import defaultdict
import itertools
import random
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import copy
#from matplotlib.animation import FuncAnimation
#from scipy import interpolate
#from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
#import sys
import matplotlib.backends.backend_pdf
"numpy.set_printoptions(threshold=numpy.nan)"


######### VALORES INICIALES #########

#TODOS LOS VALORES EN NANÓMETROS
#TODAS LAS ENERGÍAS DADAS EN kJ/mol


#potentials=[-8838.120303, -6942.7176, -5495.6, -3933.6696, -2840.74, -2115.67, -1461.6, -1009.589, -721.9425, -469.85, -76.77]
potentials=numpy.linspace(-28,10,2)

repetitions=range(1)
particles=2
iterations=50000		
#Temperatura en Kelvin
T=298.15

def lennard_jones(rij):
	u=2.601*((0.3166/rij)**12 - (0.3166/rij)**6)
	return u
def coulomb(rij,q1,q2):
	v=((138.62*q1*q2)/rij)
	return v
def steele(zij):
	z=15.056*( ((5.47*10**(-6)) /(zij**10)) - ((0.01134)/(zij**4)) - ((0.01127)/((0.205+zij)**3))  )
	return z

#Caja de Simulación
values={'x_value':1, 'y_value':1, 'z_value':1}


#Valores constantes
r_1=0.1 #Distancia de O a H
d=0.057 #Esta es la distancia entre O y el intersecto de los hidrógenos
l=0.082 #Esta es la ditancia entre el intersecto y los hidrógenos
q_H=0.4238
q_O=-0.8476
delta=0.01
cutoff=1
mean=[]

for repetition in repetitions:

	lista_final=[]

	for pot in tqdm(potentials):

		#Listas y diccionarios que se van a usar en la simulaión 
		#Por código toca definirlos
		e_list=[]
		i_list=[]
		final=[]
		interaction=[]
		energies={}
		distances={}
		energy={}
		total_distances={}
		info={}
		positions=[]
		accepted=0
		insertion=0
		part=[]






		#Positions por itertools

		positions= itertools.product(numpy.linspace(0 , values['x_value'] - 0.01 , particles), [values['y_value']/2], [values['z_value']/2])
		positions=list(positions)

		#Cantidad de moléculas, dispuestas de manera aleatoria en la caja de simulación
		#for i in range(particles):
		#	positions.append([random.uniform(0,values['x_value']),random.uniform(0,values['y_value']),random.uniform(0,values['z_value'])])
		#positions=list(positions)
		#positions contiene las posiciones de los oxígenos de cada molécula


		for i, position in enumerate(positions):
			#Estos ángulos se trazan con el origen en el oxígeno
			#polar es un ángulo con respecto al eje z
			#azimut es el ángulo phi en coordenadas esféricas
			#gamma es el ángulo de rotación del plano en el que se encuentran los hidrógenos
			polar=numpy.pi/2 - (random.uniform(-numpy.pi/2,numpy.pi/2))
			azimut= random.uniform(0,2*numpy.pi)
			gamma = random.uniform(0,numpy.pi)

			#Posición del punto medio sin rotar
			#El punto medio es una extensión de m que intersecta una linea que une los hidrógenos
			T1 = [d*numpy.cos(polar)*numpy.cos(azimut),
				 d*numpy.cos(polar)*numpy.sin(azimut),
				 d*numpy.sin(polar)]

			#Posición de H1 sin rotar
			T2 = [d*numpy.cos(polar)*numpy.cos(azimut) + l*numpy.sin(azimut),
				 d*numpy.cos(polar)*numpy.sin(azimut) - l*numpy.cos(azimut),
				 d*numpy.sin(polar)]

			#Posición de H2 sin rotar
			T3 = [d*numpy.cos(polar)*numpy.cos(azimut) - l*numpy.sin(azimut),
				 d*numpy.cos(polar)*numpy.sin(azimut) + l*numpy.cos(azimut),
				 d*numpy.sin(polar)]
			
			#W y V son vectores unitarios perpendiculares que definen el plano de rotación gamma
			w = numpy.array(T3) - numpy.array(T2)
			W = w / numpy.linalg.norm(w)

			v = numpy.cross(T3,T2)
			V = v /numpy.linalg.norm(v)

			#H111 y H222 son las posiciones finales de los hidrógenos al sumar el vector punto medio T1
			H11 = l*numpy.sin(gamma)*V + l*numpy.cos(gamma)*W
			H111 = T1 + H11
			H222 = T1 - H11

			#info es el corazón del programa
			#info tiene como key el número de la molécula
			#Y como value tiene diccionarios conteniendo las posiciones de cada estructura (oxígeno, punto m, H1 y H2)
			info[i]={
			'pos': position,
			'pos1': [H111[0] + position[0], H111[1] + position[1], H111[2] + position[2]], 
			'pos2': [H222[0] + position[0], H222[1] + position[1], H222[2] + position[2]]}

		#Generar la lista de interacciones vacía
		for i in info.keys():
				for value1 in ['pos','pos1','pos2']:
					for h in info.keys():
						for value2 in ['pos','pos1','pos2']:
							if i != h:
								interaction.append(((i,value1),(h,value2)))

		'''print(interaction)'''

		#Esta parte se hace para tomar solo las interacciones que importan, en forma de tuplas (molécula, estructura),(molécula estructura)
		#descartando las repetidas, las interacciones propias y las interacciones de oxígenos con algo más que no sea oxígeno									
		interaction = [tuple(sorted(i)) for i in interaction]

		#Este set quita las interacciones repetidas
		final=set(interaction)
		interaction=sorted(list(final))
		#print(interaction)
		for f in final:
			distances[f] = list()

		'''print(interaction)
		print('close')'''
		#Esto calcula las distancias iniciales, el ciclo en z se hace a parte porque no existen condiciones periódicas de frontera en ese eje
		for f in distances.keys():
			
			for h, coordinate in enumerate (('x','y')):

				#info contiene la información de cada molécula por lo que por ejemplo
				#info{1:{pos1:1,2,2}} es la posición x,y,z del hidrógeno 1 en la mólécula 1
				if numpy.abs(info[f[0][0]][f[0][1]][h] - info[f[1][0]][f[1][1]][h]) > float(values[('%s' % coordinate + '_value')])/2:
					distances[f].append(numpy.abs(numpy.abs(info[f[0][0]][f[0][1]][h] - info[f[1][0]][f[1][1]][h]) - float(values[('%s' % coordinate + '_value')])))
					
				elif numpy.abs(info[f[0][0]][f[0][1]][h] - info[f[1][0]][f[1][1]][h]) <= float(values[('%s' % coordinate + '_value')])/2:
					distances[f].append (numpy.abs(info[f[0][0]][f[0][1]][h]-info[f[1][0]][f[1][1]][h]))
					
				else: 
					distances[f].append(info[f[0][0]][f[0][1]][h] - info[f[1][0]][f[1][1]][h])

			distances[f].append(numpy.abs(info[f[0][0]][f[0][1]][2] - info[f[1][0]][f[1][1]][2]))

			

		#Norma
		for distance in distances:
			total_distances[distance]=numpy.linalg.norm(distances[distance])

		#print(total_distances)

		#Diccionarios vacíos de energías en términos de las distancias (lennard jones y coulomb) y de partículas individuales (steele)
		for distance in total_distances.keys():
			energy[distance]=list()

		#Esto es potencial de steele
		for molecule in info.keys():
			energy[molecule]=list()

		for distance in total_distances.keys():

			if total_distances[distance] < cutoff:
				#print('a')
		#Los keys de distances y total_distances son tuplas (Númerodemolécula, posición (O,H1,H2))

			#pos es la posición de los oxígenos, por lo que esta primera parte solo tomas las interacciones oxígeno-oxígeno

			#En esta parte se agrega inmediatamente la interacción LJ como la de Coulomb

				if distance[0][1] == 'pos' and distance[1][1]== 'pos':
					u = lennard_jones(total_distances[distance])
					v = coulomb(total_distances[distance],q_O,q_O)
					energy[distance].append(u+v)
					
				#Coulomb entre H1, H2 y O (O-O se hace arriba)
				else:
					if 'pos' not in distance[0] and 'pos' not in distance[1]:
						q1=q_H
						q2=q_H
					else:
						q1=q_O
						q2=q_H
							
					v = coulomb(total_distances[distance],q1,q2)
					energy[distance].append(v)
					
			else:
				energy[distance]=[]

		#Para el número de la molécula, potencial de Steele
		for molecule in info.keys():
			z=steele(info[molecule]['pos'][2])
			#t=steele(values['z_value'] - info[molecule]['pos'][2])
			energy[molecule]=[]
			energy[molecule].append(z)

		total_list=[]
		for e in energy.values():
			if len(e)>0:
				total_list.append(e[0])

		first_energy=sum(total_list)

		'''plt=pyplot.figure()
		ax=plt.add_subplot(111,projection='3d')
		for position in info.keys():
			x,y,z=zip(info[position]['pos'])
			xh1,yh1,zh1=zip(info[position]['pos1'])
			xh2,yh2,zh2=zip(info[position]['pos2'])
			#xm,ym,zm=zip(info[position]['posm'])

			ax.scatter(x,y,z,color='grey', s=350)
			#ax.scatter(xm,ym,zm,color='black', s=100)
			ax.scatter(xh1,yh1,zh1,color='red', s=350)
			ax.scatter(xh2,yh2,zh2,color='red', s=350)
			ax.set_xlim3d(0,values['x_value'])
			ax.set_ylim3d(0,values['y_value'])
			ax.set_zlim3d(0,values['z_value'])
			ax.set_aspect('equal')

		pyplot.show()
		pyplot.close()'''

		total_energy = copy.deepcopy(first_energy)

		#ESTO DETERMINA LOS VALORES PARA LOS QUE SE VA A CALCULAR LA FUNCIÓN DE DISTRIBUCIÓN RADIAL


		a = random.choice(list(info.keys()))
		hist={}
		histogram=[]
		for distance in total_distances.keys():
			if (distance[0][0] == a and distance[0][1] == 'pos') or (distance[1][0] == a and distance[1][1] == 'pos'):
				if distance[0][1] == 'pos' and distance[1][1] == 'pos':
					hist[distance]=[]




		############### COMIENZAN LAS ITERACIONES #################

		for i in tqdm(range(iterations)):

		########INTENTO DE MOVIMIENTO#############

			information=copy.deepcopy(info)
			distances_new=copy.deepcopy(distances)
			energy_new=copy.deepcopy(energy)
			total_distances_new=copy.deepcopy(total_distances)
			total_energy = copy.deepcopy(total_energy)
			interact=copy.deepcopy(interaction)

			#print(information)
			#print(total_energy)

			i_list.append(i)


			#DESPLAZAMIENTO Y ROTACIÓN ALEATORIOS

			random_molecule=random.choice(list(information.keys()))
			#print(random_molecule)

			#Almacena la información inicial de la molécula que se va a mover/rotar
			stored=copy.deepcopy(information[random_molecule])


			polar = polar + random.uniform(-numpy.pi/4,numpy.pi/4)
			azimut= azimut + random.uniform(0,numpy.pi/2)
			gamma = gamma + random.uniform(0,numpy.pi/2)

			randomx=delta*(random.uniform(0,1) - 0.5)
			randomy=delta*(random.uniform(0,1) - 0.5)
			randomz=delta*(random.uniform(0,1) - 0.5)


			#Para x
			if information[random_molecule]['pos'][0] + randomx >= values['x_value']:
				x_new = information[random_molecule]['pos'][0] + randomx - values['x_value']
			elif information[random_molecule]['pos'][0] + randomx < 0:
				x_new = (information[random_molecule]['pos'][0] + randomx) + values['x_value'] - int(information[random_molecule]['pos'][0] + randomx)
			else: x_new = information[random_molecule]['pos'][0] + randomx	

			#Para y
			if information[random_molecule]['pos'][1] + randomy >= values['y_value']:
				y_new = information[random_molecule]['pos'][1] + randomy - values['y_value']
			elif information[random_molecule]['pos'][1] + randomy < 0:
				y_new = (information[random_molecule]['pos'][1]) + randomy + values['y_value'] - int(information[random_molecule]['pos'][1] + randomy)
			else: y_new = information[random_molecule]['pos'][1] + randomy	

			#Para y
			if information[random_molecule]['pos'][2] + randomz >= values['z_value']:
				z_new = information[random_molecule]['pos'][2] + randomz - values['z_value']
			elif information[random_molecule]['pos'][2] + randomz < 0:
				z_new = (information[random_molecule]['pos'][2]) + randomz + values['z_value'] - int(information[random_molecule]['pos'][2] + randomz)
			else: z_new = information[random_molecule]['pos'][2] + randomz	

			position=[]

			position = [
			x_new,
			y_new,
			z_new]

			T1 = [d*numpy.cos(polar)*numpy.cos(azimut),
				 d*numpy.cos(polar)*numpy.sin(azimut),
				 d*numpy.sin(polar)]

			#Posición de H1 sin rotar
			T2 = [d*numpy.cos(polar)*numpy.cos(azimut) + l*numpy.sin(azimut),
				 d*numpy.cos(polar)*numpy.sin(azimut) - l*numpy.cos(azimut),
				 d*numpy.sin(polar)]

			#Posición de H2 sin rotar
			T3 = [d*numpy.cos(polar)*numpy.cos(azimut) - l*numpy.sin(azimut),
				 d*numpy.cos(polar)*numpy.sin(azimut) + l*numpy.cos(azimut),
				 d*numpy.sin(polar)]
			
			w = numpy.array(T3) - numpy.array(T2)
			W = w / numpy.linalg.norm(w)

			v = numpy.cross(T3,T2)
			V = v /numpy.linalg.norm(v)

			H11 = l*numpy.sin(gamma)*V + l*numpy.cos(gamma)*W
			H22 = -l*numpy.sin(gamma)*V - l*numpy.cos(gamma)*W
			H111 = T1 + H11
			H222 = T1 + H22

			information[random_molecule]={
			'pos': position,
			'pos1': [H111[0] + position[0], H111[1] + position[1], H111[2] + position[2]], 
			'pos2': [H222[0] + position[0], H222[1] + position[1], H222[2] + position[2]]}

			#print(information[random_molecule])

			#CALCULO DE DISTANCIAS

			#for i in energy

			'''print('0initial')
			print(distances_new.keys())
			print('random')
			print(random_molecule)

			print('information')
			print(information)

			print('distances')
			print(distances.keys())
			print('total')
			print(total_distances.keys())'''

			'''print('MOVE')
			print(random_molecule)
			print(list(information.keys()))
			print('antes')
			print(list(distances_new.keys()))'''

			for distance in distances_new.keys():
				if distance[0][0] == random_molecule or distance[1][0] == random_molecule :
					distances_new[distance] = []

					#print(distance)

					for h, coordinate in enumerate (('x','y')):
						
						#info contiene la información de cada molécula por lo que por ejemplo
						#info{1:{pos1:1,2,2}} es la posición x,y,z del hidrógeno 1 en la mólécula 1
						if numpy.abs(information[distance[0][0]][distance[0][1]][h] - information[distance[1][0]][distance[1][1]][h]) > float(values[('%s' % coordinate + '_value')])/2:
							distances_new[distance].append(numpy.abs(numpy.abs(information[distance[0][0]][distance[0][1]][h] - information[distance[1][0]][distance[1][1]][h]) - float(values[('%s' % coordinate + '_value')])))
							
						elif numpy.abs(information[distance[0][0]][distance[0][1]][h] - information[distance[1][0]][distance[1][1]][h]) <= float(values[('%s' % coordinate + '_value')])/2:
							distances_new[distance].append (numpy.abs(information[distance[0][0]][distance[0][1]][h]-information[distance[1][0]][distance[1][1]][h]))
							
						else: 
							distances_new[distance].append(numpy.abs(information[distance[0][0]][distance[0][1]][h] - information[distance[1][0]][distance[1][1]][h]))

					distances_new[distance].append(numpy.abs(information[distance[0][0]][distance[0][1]][2] - information[distance[1][0]][distance[1][1]][2]))
					
					
			for distance in distances_new:
				total_distances_new[distance]=numpy.linalg.norm(distances_new[distance])

			#CÁLCULO DE LAS ENERGÍAS

			for distance in distances_new.keys():

				if total_distances[distance] < cutoff:
					if random_molecule in distance[0] or random_molecule in distance[1]:

						if distance[0][1] == 'pos' and distance[1][1]== 'pos':

							u = lennard_jones(total_distances_new[distance])
							v = coulomb(total_distances_new[distance],q_O,q_O)
							energy_new[distance]=[]
							energy_new[distance].append(u+v)
							#print('O-O')
							#print(u)
							#print(v)

						#Coulomb entre H1, H2 y O (O-O se hace arriba)
						else:

							if 'pos' not in distance[0] and 'pos' not in distance[1]:
								q1=q_H
								q2=q_H
							else:
								q1=q_O
								q2=q_H
									
							v = coulomb(total_distances_new[distance],q1,q2)
							energy_new[distance]=[]
							energy_new[distance].append(v)
							#print('H')
							#print(v)
					else: pass
				else: energy_new[distance] = []


			#AGREGA EL POTENCIAL DE STEELE CON AMBAS PAREDES PARA LA PARTÍCULA RANDOM_MOLECULE
			
			z=steele(information[random_molecule]['pos'][2])
			#t=steele(values['z_value'] - information[random_molecule]['pos'][2])
			energy_new[random_molecule]=[]
			energy_new[random_molecule].append(z)

			total_list=[]
			for e in energy_new.values():
				if len(e)>0:
					total_list.append(sum(e))

			total_energy_new=sum(total_list)

			'''print('\n')
			print(total_energy, total_energy_new)
			print(numpy.exp(-120.3317*(float(numpy.abs(total_energy_new - total_energy))/T)))'''

			if total_energy_new < total_energy:
				info=copy.deepcopy(information)
				distances=copy.deepcopy(distances_new)
				energy=copy.deepcopy(energy_new)
				total_distances=copy.deepcopy(total_distances_new)
				total_energy = total_energy_new
				accepted+=1


			#Si la energía es mayor
			elif total_energy_new > total_energy:

				#Y se cumple la condición, se acepta el cambio

				if numpy.exp(-120.3317*(float(numpy.abs(total_energy_new - total_energy))/T)) >= random.random():
							info=copy.deepcopy(information)
							distances=copy.deepcopy(distances_new)
							energy=copy.deepcopy(energy_new)
							total_distances=copy.deepcopy(total_distances_new)
							total_energy = total_energy_new
							accepted+=1
							

				#Sino, se vuelve a la energía y a la posición anterior
				else: 
					info[random_molecule]=stored
					energy=copy.deepcopy(energy)
					total_distances=copy.deepcopy(total_distances)
					distances=copy.deepcopy(distances)
					total_energy = total_energy
			
			'''print('despues')
			print(distances.keys())'''






















			################ INSERTION ##################





			information=copy.deepcopy(info)
			distances_new=copy.deepcopy(distances)
			energy_new=copy.deepcopy(energy)
			total_distances_new=copy.deepcopy(total_distances)
			total_energy = copy.deepcopy(total_energy)

			for number in range(500):
				if number not in info.keys():
					n = number
					break
			#print('\n''\n')
			#print('insertion')
			#print(n)

			#information[n]={}
			#print(information.keys())

			position=[random.uniform(0,values['x_value']),random.uniform(0,values['y_value']),random.uniform(0,values['z_value'])]

		 	#Estos ángulos se trazan con el origen en el oxígeno
			#polar es un ángulo con respecto al eje z
			#azimut es el ángulo phi en coordenadas esféricas
			#gamma es el ángulo de rotación del plano en el que se encuentran los hidrógenos
			
			polar=numpy.pi/2 - (random.uniform(-numpy.pi/2,numpy.pi/2))
			azimut= random.uniform(0,2*numpy.pi)
			gamma = random.uniform(0,numpy.pi)

			#Posición del punto medio sin rotar
			#El punto medio es una extensión de m que intersecta una linea que une los hidrógenos
			T1 = [d*numpy.cos(polar)*numpy.cos(azimut),
				 d*numpy.cos(polar)*numpy.sin(azimut),
				 d*numpy.sin(polar)]

			#Posición de H1 sin rotar
			T2 = [d*numpy.cos(polar)*numpy.cos(azimut) + l*numpy.sin(azimut),
				 d*numpy.cos(polar)*numpy.sin(azimut) - l*numpy.cos(azimut),
				 d*numpy.sin(polar)]

			#Posición de H2 sin rotar
			T3 = [d*numpy.cos(polar)*numpy.cos(azimut) - l*numpy.sin(azimut),
				 d*numpy.cos(polar)*numpy.sin(azimut) + l*numpy.cos(azimut),
				 d*numpy.sin(polar)]
			
			#W y V son vectores unitarios perpendiculares que definen el plano de rotación gamma
			w = numpy.array(T3) - numpy.array(T2)
			W = w / numpy.linalg.norm(w)

			v = numpy.cross(T3,T2)
			V = v /numpy.linalg.norm(v)

			#H111 y H222 son las posiciones finales de los hidrógenos al sumar el vector punto medio T1
			H11 = l*numpy.sin(gamma)*V + l*numpy.cos(gamma)*W
			H111 = T1 + H11
			H222 = T1 - H11

			#info es el corazón del programa
			#info tiene como key el número de la molécula
			#Y como value tiene diccionarios conteniendo las posiciones de cada estructura (oxígeno, punto m, H1 y H2)
			information[n]={
			'pos': position,
			'pos1': [H111[0] + position[0], H111[1] + position[1], H111[2] + position[2]], 
			'pos2': [H222[0] + position[0], H222[1] + position[1], H222[2] + position[2]]}	

			interact=[]

			for m in information.keys():
				for value1 in ['pos','pos1','pos2']:
					for p in information.keys():
						for value2 in ['pos','pos1','pos2']:
							if m != p:
								interact.append(((m,value1),(p,value2)))


			interact = [tuple(sorted(i)) for i in interact]

			#Este set quita las interacciones repetidas
			final=set(interact)
			interact=sorted(list(final))
			for f in interact:
				if f not in list(distances_new.keys()):
					distances_new[f] = list()

			
			#print(list(information.keys()))
			#print(list(distances_new.keys()))
			for distance in distances_new.keys():
				if distance[0][0] == n or distance[1][0] == n :
					distances_new[distance] = []

					#print('hola mundo', distance)
					
					for h, coordinate in enumerate (('x','y')):
						
						#info contiene la información de cada molécula por lo que por ejemplo
						#info{1:{pos1:1,2,2}} es la posición x,y,z del hidrógeno 1 en la mólécula 1
						if numpy.abs(information[distance[0][0]][distance[0][1]][h] - information[distance[1][0]][distance[1][1]][h]) > float(values[('%s' % coordinate + '_value')])/2:
							distances_new[distance].append(numpy.abs(numpy.abs(information[distance[0][0]][distance[0][1]][h] - information[distance[1][0]][distance[1][1]][h]) - float(values[('%s' % coordinate + '_value')])))
							
						elif numpy.abs(information[distance[0][0]][distance[0][1]][h] - information[distance[1][0]][distance[1][1]][h]) <= float(values[('%s' % coordinate + '_value')])/2:
							distances_new[distance].append (numpy.abs(information[distance[0][0]][distance[0][1]][h]-information[distance[1][0]][distance[1][1]][h]))
							
						else: 
							distances_new[distance].append(numpy.abs(information[distance[0][0]][distance[0][1]][h] - information[distance[1][0]][distance[1][1]][h]))
					
					distances_new[distance].append(numpy.abs(information[distance[0][0]][distance[0][1]][2] - information[distance[1][0]][distance[1][1]][2]))

			for distance in distances_new:
				total_distances_new[distance]=numpy.linalg.norm(distances_new[distance])



			#ENERGIES


			#print('distance')
			#print(total_distances_new)

			for distance in total_distances_new.keys():

				if total_distances_new[distance] < cutoff:
					if distance[0][0] == n or distance[1][0] == n:

						if distance[0][1] == 'pos' and distance[1][1]== 'pos':

							u = lennard_jones(total_distances_new[distance])
							v = coulomb(total_distances_new[distance],q_O,q_O)
							energy_new[distance]=[]
							energy_new[distance].append(u+v)
							#print('O-O')
							#print(u)
							#print(v)

						#Coulomb entre H1, H2 y O (O-O se hace arriba)
						else:

							if 'pos' not in distance[0] and 'pos' not in distance[1]:
								q1=q_H
								q2=q_H
							else:
								q1=q_O
								q2=q_H
									
							v = coulomb(total_distances_new[distance],q1,q2)
							energy_new[distance]=[]
							energy_new[distance].append(v)
							#print('H')
							#print(v)
					else: pass
				else: energy_new[distance] = []


			#AGREGA EL POTENCIAL DE STEELE CON AMBAS PAREDES PARA LA PARTÍCULA RANDOM_MOLECULE
			
			z=steele(information[n]['pos'][2])
			#t=steele(values['z_value'] - information[n]['pos'][2])
			energy_new[n]=[]
			energy_new[n].append(z)

			total_list=[]
			for e in energy_new.values():
				if len(e)>0:
					total_list.append(sum(e))

			total_energy_new=sum(total_list)
			#print('\n')
			#print(total_energy, total_energy_new)
			#print(numpy.exp(-float(numpy.abs(total_energy_new - total_energy))/2288.36*T))

			#print(numpy.exp(-float(numpy.abs(total_energy_new - total_energy))/(300)))

			if total_energy_new < total_energy:
				info=copy.deepcopy(information)
				distances=copy.deepcopy(distances_new)
				energy=copy.deepcopy(energy_new)
				total_distances=copy.deepcopy(total_distances_new)
				total_energy = total_energy_new
				accepted+=1
				insertion+=1



			#Si la energía es mayor
			elif total_energy_new > total_energy:

				#Y se cumple la condición, se acepta el cambio

				if ((16.863*(values['x_value']*values['y_value']*values['z_value'])*(T)**(3/2))/len(list(information.keys()))) * numpy.exp( (120.3317/T)*(pot - total_energy_new + total_energy)) >= random.random():
					info=copy.deepcopy(information)
					distances=copy.deepcopy(distances_new)
					energy=copy.deepcopy(energy_new)
					total_distances=copy.deepcopy(total_distances_new)
					total_energy = total_energy_new
					accepted+=1
					insertion+=1
					
					

				#Sino, se vuelve a la energía y a la posición anterior
				else: 
					info=copy.deepcopy(info)
					energy=copy.deepcopy(energy)
					total_distances=copy.deepcopy(total_distances)
					distances=copy.deepcopy(distances)
					total_energy = total_energy


			#print('\n')
			#print('Info keys ins')
			#print(list(info.keys()))
			#print('distances')
			#print(distances.keys())













			########### DELETION ###############

			if len(list(info.keys()))>2:



				information=copy.deepcopy(info)
				distances_new=copy.deepcopy(distances)
				energy_new=copy.deepcopy(energy)
				total_distances_new=copy.deepcopy(total_distances)
				total_energy = copy.deepcopy(total_energy)

				random_delete = random.choice(list(info.keys()))

				#print('delete')
				#print(random_delete)

				information.pop(random_delete)
				energy_new.pop(random_delete)

				dis = list(distances.keys())
				
				#print('antes')
				#print(distances.keys())
				for inte in dis:
				
					if inte[0][0] == random_delete or inte[1][0] == random_delete:
						distances_new.pop(inte, None)
						total_distances_new.pop(inte, None)
						energy_new.pop(inte, None)


					else: pass
				
				total_list=[]
				for e in energy_new.values():
					if len(e)>0:
						total_list.append(sum(e))

				total_energy_new=sum(total_list)	

				'''print('INS')
				print(total_energy, total_energy_new)
				print((((0.0593*len(list(info.keys())))/(values['x_value']*values['y_value']*values['z_value'])*(T**(3/2))) * numpy.exp( (120.3317/T)* (-pot - total_energy_new + total_energy))))'''


				if total_energy_new < total_energy:
					info=copy.deepcopy(information)
					distances=copy.deepcopy(distances_new)
					energy=copy.deepcopy(energy_new)
					total_distances=copy.deepcopy(total_distances_new)
					total_energy = total_energy_new
					accepted+=1
					insertion+=1



				#Si la energía es mayor
				elif total_energy_new > total_energy:

					#Y se cumple la condición, se acepta el cambio

					if (((0.0698*len(list(info.keys())))/(values['x_value']*values['y_value']*values['z_value'])*T) * numpy.exp( (120.3317/T)* (-pot - total_energy_new + total_energy))) >= random.random():
						info=copy.deepcopy(information)
						distances=copy.deepcopy(distances_new)
						energy=copy.deepcopy(energy_new)
						total_distances=copy.deepcopy(total_distances_new)
						total_energy = total_energy_new
						accepted+=1
						insertion+=1
						

					#Sino, se vuelve a la energía y a la posición anterior
					else: 
						info=copy.deepcopy(info)
						energy=copy.deepcopy(energy)
						total_distances=copy.deepcopy(total_distances)
						distances=copy.deepcopy(distances)
						total_energy = total_energy
						
			else: pass


			#print(total_energy)
			
			#print(info.keys())
			'''print('Info keys del')
			print(list(info.keys()))'''
			#print('distances')
			#print(distances.keys())'''

			part.append(len(list(info.keys())))
			#print(len(list(info.keys())))
			e_list.append(total_energy)

			#file= open('data', "w")
			#for 



			'''for distance in total_distances:
				if distance in hist:
					hist[distance].append(total_distances[distance])'''

		'''accepted=accepted/iterations

		pyplot.figure()
		pyplot.scatter(i_list,e_list, label='acceptance rate' + '%f' % accepted )
		#pyplot.ylim((int(e_list[-1]),e_list[int(iterations/2)]))
		pyplot.legend(loc='best')
		pyplot.ylabel('kJ/mol')
		pyplot.grid(True)
		pyplot.xlabel('mcs')
		pyplot.show()
		pyplot.close()'''


		#Configuración final


		pdf = matplotlib.backends.backend_pdf.PdfPages("output_" + '%s' %repetition + '_' +  '%s' %pot + '.pdf')
		plt=pyplot.figure()
		ax=plt.add_subplot(111,projection='3d')
		for position in info.keys():
			x,y,z=zip(info[position]['pos'])
			xh1,yh1,zh1=zip(info[position]['pos1'])
			xh2,yh2,zh2=zip(info[position]['pos2'])
			#xm,ym,zm=zip(info[position]['posm'])

			ax.scatter(x,y,z,color='grey', s=250)
			#ax.scatter(xm,ym,zm,color='black', s=100)
			ax.scatter(xh1,yh1,zh1,color='red', s=250)
			ax.scatter(xh2,yh2,zh2,color='red', s=250)
			ax.set_xlim3d(0,values['x_value'])
			ax.set_ylim3d(0,values['y_value'])
			ax.set_zlim3d(0,values['z_value'])
			#ax.set_aspect('equal')

		pdf.savefig()
		pyplot.close()

		pyplot.figure()
		pyplot.scatter(i_list,part)
		pdf.savefig()

		pyplot.figure()
		pyplot.scatter(i_list[10:],e_list[10:])
		pdf.savefig()
		pyplot.close()

		pdf.close()


		promedio=numpy.mean(numpy.array(part[int(2*iterations/3):-1]))
		lista_final.append(promedio)

		final_positions = []

		for position in info.keys():
			final_positions.append(info[position]['pos'])

		save= [i_list, part, e_list, final_positions]
		print(type(save))
		with open('file_' + '%s' % repetition + '_' + '%s' % pot + '.json','w') as js:
    		# human-readable for more complicated data
  			json.dump(save, js) 


		#guardar columna y fila con iteraciones (part y i_list)



	'''for value in hist.values():
		v=len(value)/2
		histogram.append(sum(value[int(v):-1])/(v))'''


	#FUNCIÓN DE DISTRIBUCIÓN RADIAL
	'''pyplot.figure()
	pyplot.hist(histogram,10)
	pyplot.xlim(0,1)
	pyplot.show()
	pyplot.close()'''

	'''print(potentials)
	print(lista_final)'''


	save= str((numpy.array([potentials, lista_final]).T))

	file = open('configurations_ ' + '%s' % repetition + '.txt','w')
	file.write(save)
	file.close()

	mean.append(lista_final)

"""mean = numpy.array(mean)
average=numpy.mean(mean, axis=0, dtype=float)
error = numpy.std(mean,axis=0, dtype=float)

pdf = matplotlib.backends.backend_pdf.PdfPages("final_" + '%s' %repetition + '.pdf')
#print(average)
#print(error)
pyplot.figure()
pyplot.scatter(potentials,average)
pyplot.errorbar(potentials,average,yerr=error)
pdf.savefig()
pyplot.close()
pdf.close()"""