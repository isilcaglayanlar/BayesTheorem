# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:54:40 2020

@author: ISIL

"""

#240201057

import pandas as pd
import numpy as np
import random

#Required objects that has already explained below

class condProbability:
    def __init__(self,x_value,group,probability):
        self.x = x_value
        self.g = group
        self.p = probability

class posteriorProbability:
    def __init__(self,group1,group2,probability):
        self.g1 = group1
        self.g2 = group2
        self.p = probability
        
class test:
    def __init__(self,probability,check_value):
        self.p = probability
        self.c = check_value
        
#Functions
        
def separateArrayIntoTwoParts(array,length1,length2): #according to given percentages, seperation occurs RANDOMLY
    
    values = []
    
    values80 = []
    values20 = []
    
    min_len = int((len(array)*length1)/100)
    
    random_indexes = random.sample(range(0, len(array)),min_len)
    for i in range (len(array)):
        if(i in random_indexes):
            values20.append(array[i])
        else:
            values80.append(array[i])
   
    values.append(values20)
    values.append(values80)
    
    return values #returns array of seperated arrays

# p(Ck|x)
def calculateConditionalProbability(min_value,max_value,data,feature,Ck):
    #Ck for calculating conditional probability according to desired class type (Ck=0 or Ck=1)
    
    class_counter = 0
    total_counter = 0
    
    for j in range (len(data)):
        control_data = data[j][32]
        feature_val =  data[j][colnames.index(feature)]
            
        if(feature_val >= min_value and feature_val < max_value): #if we know x
            if(Ck==1):
                if(control_data >= 10): #what's the number of values exists in given class
                    class_counter += 1
                    
            elif(Ck==0):
                if(control_data < 10): #what's the number of values exists in given class
                    class_counter += 1
                
            total_counter += 1
                    
    return (class_counter/total_counter)

#p(Ck)
def calculateEvidence(data,check_value,check_index,Ck):
    
    class_counter = 0
    total_counter = 0
    for i in range (len(data)):
        if(Ck == 1):
            if(data[i][check_index] >= 10):
                class_counter += 1
        else:
            if(data[i][check_index] < 10):
                class_counter += 1
            
        total_counter += 1
    
    return (class_counter/total_counter)
            
#P(Ck|xI,xB)=(P(Ck|xI)*p(Ck|xB))/p(Ck) without normalization    
def calculatePosteriorProbability(data,cond_probs,Ck):
   
    dividend = 1.0
    for i in range (len(cond_probs)):
       dividend = (dividend*((cond_probs[i])))
    
    evidence = calculateEvidence(data,10,32,Ck)
    posterior = dividend/evidence
    
    return posterior
      
#P(Ck|xI,xB)=(P(Ck|xI)*p(Ck|xB))/p(Ck) with normalization
def findPosteriors(data,features,interval):
    
    #GROUPS
    #[interval[0]-interval[1]] : group1
    #[interval[1]-interval[2]] : group2
    #[interval[2]-interval[3]] : group3
    
    #REQUIRED CALCULATIONS FOR NORMALIZATION OF BAYES
    #P(Ck) for Ck=1(PASS) and P(Ck) for Ck=0(FAIL)
    
    #--------------------------REQUIRED PART FOR THE DIVIDEND-----------------------------------------
    #PASS CONDITION Ck = 1 
    
    #first, p(Ck|xI) and p(Ck|xB) must be calculated
    cond_probs = []
    for i in range (len(features)): #for each feature (xI and xB)
        for j in range (len(interval)-1):
            cond_prob = calculateConditionalProbability(interval[j],interval[j+1],data,features[i],1) #for each future and for each part of the interval mentioned in line 283 
            p = condProbability(features[i],(j+1),cond_prob) #new object for storing feature,group and probability information
            cond_probs.append(p) #there are 2(feature number)*3(group number) = 6 conditional probability has calculated
    
    feature_number = int(len(features))
    partition = int(len(cond_probs)/feature_number) #calculate how many groups exist
    
    
    pass_posteriors = []
    
    #there are 3 conditional probability for feature1 and 3 for feature2
    #the first 3 entry of cond_probs list is feture1's cond. prob and last 3 is feature2's cond. prob.
    #2 for loop for combination of 3x3 events
    for i in range (partition): #FIRST 3
        cond_prob1 = ((cond_probs[i]).p)
        group1 = ((cond_probs[i]).g)
        for j in range (partition,partition*feature_number): #LAST 3
            cond_prob2 = ((cond_probs[j]).p)
            group2 = ((cond_probs[j]).g)
            cond_prob=[cond_prob1,cond_prob2] #required 2 cond. prob. to calculate posterior probability p(Ck|xI) and p(Ck|xB)
            posterior = calculatePosteriorProbability(data,cond_prob,1)
            temp_p = posteriorProbability(group1,group2,posterior) #new object for storing group and posterior probability information
            pass_posteriors.append(temp_p) #there are 3x3=9 post. prob. has calculated
          
    #--------------------------REQUIRED PART FOR THE DIVIDER-----------------------------------------
    #FAIL CONDITION Ck = 0 
    cond_probs = []
    for i in range (len(features)):
        for j in range (len(interval)-1):
            cond_prob = calculateConditionalProbability(interval[j],interval[j+1],data,features[i],0)
            p = condProbability(features[i],(j+1),cond_prob)
            cond_probs.append(p)
      
    feature_number = int(len(features))
    partition = int(len(cond_probs)/feature_number)
   
    fail_posteriors = []
    for i in range (partition):
        cond_prob1 = ((cond_probs[i]).p)
        group1 = ((cond_probs[i]).g)
        for j in range (partition,partition*feature_number):
            cond_prob2 = ((cond_probs[j]).p)
            group2 = ((cond_probs[j]).g)
            cond_prob=[cond_prob1,cond_prob2]
            posterior = calculatePosteriorProbability(data,cond_prob,0)
            temp_p = posteriorProbability(group1,group2,posterior)
            fail_posteriors.append(temp_p)
            
    #NORMALIZATION
    #PASS/(PASS+FAIL)
    final_pass_posteriors= []
    for i in range (len(pass_posteriors)):
        pass_posteriors[i].p = (pass_posteriors[i].p)/(((pass_posteriors[i]).p)+((fail_posteriors[i]).p))
        final_pass_posteriors.append(pass_posteriors[i])
    
    """
    #this part can be used to check whether the sum of fail and pass conditions' probability is equal to 1.0 
    #FAIL/(PASS+FAIL)
    final_fail_posteriors= []
    for i in range (len(fail_posteriors)):
        fail_posteriors[i].p = ((fail_posteriors[i]).p)/(((pass_posteriors[i]).p)+((fail_posteriors[i]).p))
        final_fail_posteriors.append(fail_posteriors[i])
     
    #PRINT PASTERIOR PROBABILITIES
    print("Group1   Group2   Probability(Ck='PASS') Probability(Ck='FAIL')")
    for i in range (len(final_pass_posteriors)):
        print("  ",(final_pass_posteriors[i]).g1,"     ",(final_pass_posteriors[i]).g2,"           %1.4f                 %1.4f" % ((final_pass_posteriors[i]).p,(final_fail_posteriors[i]).p))
    """
    return final_pass_posteriors #returns 9 posterior probabilities with the info of groups

def testPrediction(data,final_pass_posteriors,interval,features,check_index):
   
    prediction = []
    
    #grouping each data according to its features and assign to them the related posterior probability calculated in prediction part
    for i in range (len(data)):
      
        groups = []
        
        single_data1 = data[i][colnames.index(features[0])]
        single_data2 = data[i][colnames.index(features[1])]
        single_data = [single_data1,single_data2]
        
        #assigning groups to test data according to prediction to decide its class
        for j in range (len(single_data)):
            if(single_data[j] >= interval[0] and single_data[j] < interval[1]):
                groups.append(1)
            elif(single_data[j] >= interval[1] and single_data[j] < interval[2]):
                groups.append(2)
            elif(single_data[j] >= interval[2] and single_data[j] < interval[3]):
                groups.append(3)
        
        for k in range (len(final_pass_posteriors)):
            posterior = final_pass_posteriors[k]
            if (posterior.g1 == groups[0] and posterior.g2 == groups[1]):
                temp_test_object = test(posterior.p,data[i][check_index]) #new object to store posterior probability and data will be checked (G3 in that case)
        
        prediction.append(temp_test_object)
        
    return prediction

def classification(prediction,class1,class2,threshold):
    
    #according to given threshold assign a class to datas (if >0.5 "Pass", else "Fail" in that case)
    classified_data = []
    
    for i in range (len(prediction)):
        if(prediction[i].p > threshold):
            classified_data.append([class1,prediction[i].c])
        else:
            classified_data.append([class2,prediction[i].c])
            
    return classified_data


def printContingencyTable(classified_data):
    pp=0
    pf=0
    fp=0
    ff=0

    for i in range(len(classified_data)):
    	if classified_data[i][0] == class1 and (classified_data[i][1] >= 10):  #pass predicted, pass in reality
    		pp += 1
    	elif classified_data[i][0] == class1 and (classified_data[i][1] < 10): #pass predicted, fail in reality
    		pf += 1
    	elif classified_data[i][0] == class2 and (classified_data[i][1] >= 10):#fail predicted, pass in reality
    		fp += 1
    	elif classified_data[i][0] == class2 and (classified_data[i][1] < 10): #fail predicted, fail in reality
    		ff += 1
    
    print("\n      PREDICTION")
    print("      PASS  FAIL  ALL")
    print("PASS  ",pp,"  ",fp,"  ",pp+fp)
    print("FAIL  ",pf,"   ",ff," ",pf+ff)
    print("ALL   ",pf+pp,"  ",ff+fp," ",pp+fp+pf+ff)
    

        
#initialization of parameters                  
        
colnames=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob',
          'reason','guardian','traveltime','studytime','failures','schoolsup','famsup',
          'paid','activities','nursery','higher','internet','romantic','famrel','freetime',
          'goout','Dalc','Walc','health','absences','G1','G2','G3']
          
d = pd.read_csv("student-mat.csv",sep=";")
values = d.values

#separating "values" into parts for training and computing posterior probabilities and test the prediction performance
seperated_arrays = separateArrayIntoTwoParts(values,20,80)
values20 = seperated_arrays[0]
values80 = seperated_arrays[1]
    
features = ['G1','G2'] #features that i choose to decide on the class

interval = [0,6,10,21] #Because G1 and G2 is from 0 to 20, the interval is divided into 3 parts to make calculation easier

#TRAINING BY USING 80% OF DATA
pass_posteriors = findPosteriors(values80,features,interval) #list of every case's posterior probability (for Ck=1 [PASS]) 

#TESTING THE PREDICTION
class1 = "PASS"
class2 = "FAIL"

threshold = 0.5

#TESTING FOR PREDICTION BY USING 20% OF DATA
prediction = testPrediction(values20,pass_posteriors,interval,features,32)

#Classification according to prediction results
classified_data = classification(prediction,class1,class2,threshold)

# Calculating value of contingency table cells
printContingencyTable(classified_data)