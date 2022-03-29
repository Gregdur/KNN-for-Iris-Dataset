# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:39:10 2021

@author: grego
"""

import math
import numpy as np


#on charge les données et on les range dans une liste globale
def LoadData():
    file = open("data.csv","r")
    data = []
    for ligne in file:
        if ligne != "\n":    
            currentline = ligne.split(",")
            lastcolumn = currentline[6].replace("\n","")
            currentline = [float(currentline[i]) for i in range(6)]
            data.append([currentline,lastcolumn])
    file2 = open("preTest.csv","r")
    data2 = []
    for ligne in file2:
        if ligne != "\n":    
            currentline = ligne.split(",")
            lastcolumn = currentline[6].replace("\n","")
            currentline = [float(currentline[i]) for i in range(6)]
            data2.append([currentline,lastcolumn])
    return data,data2

def LoadData2():
    file = open("data.csv","r")
    data = []
    for ligne in file:
        if ligne != "\n":    
            currentline = ligne.split(",")
            lastcolumn = currentline[6].replace("\n","")
            currentline = [float(currentline[i]) for i in range(6)]
            data.append([currentline,lastcolumn])
    file2 = open("preTest.csv","r")
    for ligne in file2:
        if ligne != "\n":    
            currentline = ligne.split(",")
            lastcolumn = currentline[6].replace("\n","")
            currentline = [float(currentline[i]) for i in range(6)]
            data.append([currentline,lastcolumn])
    print(len(data))
    return data

def LoadTest():
    file=open("finalTest.csv","r")
    test=[]
    for ligne in file:
        if ligne != "\n":    
            currentline = ligne.split(",")
            currentline = [float(currentline[i]) for i in range(6)]
            test.append([currentline])
    print (len(test))
    return test

def GetClasse():
    file = open("caurier.txt","w")
    train=LoadData2()
    test=LoadTest()
    for individu in test:
        classe=AlgoKnn(train,individu,4,False)
        file.write(classe+"\n")
    file.close()

#séparer les données pour l'apprentissage
def Separate(data,data2):
    a=0
    train = []
    test = []
    for i in range(len(data)):
        a+=1
        if a % 10 :
            train.append(data[i])
            train.append(data2[i])
        else:
            test.append(data[i])     
            test.append(data2[i])
    return train,test

def Distance(PointA, PointB):
    return math.sqrt(sum([math.pow(PointA[i]-PointB[i],2) for i in range(len(PointA))]))

#Retourne la classe la plus présente dans une liste
def ClasseDominante(liste):
    return max(liste,key=liste.count)


def AlgoKnn(train, test, k, afficherPredict):
    tab = []    
    for individu in train:
#        tab.append([Distance(individu[0],test),individu])
        tab.append([Distance(individu[0],test[0]),individu])
    tab.sort()
    kplusproche = []
    #les k premirs
    for i in range (k):
        kplusproche.append(tab[i][1][1])    
    #Affiche que si afficherPredict vaut true
    if afficherPredict:
        quality=kplusproche.count((ClasseDominante(kplusproche)))*100/k
        print("la classe prédit selon les",k,"plus proches voisins est :",ClasseDominante(kplusproche),"avec une qualité de :",quality, "\nla vraie classe est :",test[1])
    return ClasseDominante(kplusproche)


def ConfusionMatrix(k,afficherMat):    
    dico = {"classA":0,"classB":1,"classC":2,"classD":3,"classE":4}
    matriceConf=np.zeros(shape=(5,5))
    data1,data2=LoadData()
    train,test = Separate(data1,data2)
    for individu in test:
        i = dico[individu[1]]
        j = dico[AlgoKnn(train,individu,k,False)]
        matriceConf[i,j]+=1    
    if afficherMat:
        print("\n\n\n\nMatrice de confusion :\n")
        print("Classe estimées :")    
        print (matriceConf)    
    val=0
    for i in np.diag(matriceConf):
        val+=i   #somme de la diagonale de la matrice de confusion 
#    print(val)
    return val

def BestK():
    maxk=20
    stat=[]
    for k in range(1,maxk):
        y=ConfusionMatrix(k,False)
        stat.append(y)
    print("Maximum de bonne réponses",max(stat))
    print("Atteint pour k =",stat.index(max(stat))+1)
    return stat.index(max(stat))+1
    
if __name__ == "__main__":
    k=BestK()
    ConfusionMatrix(k,True)
#    GetClasse()
    