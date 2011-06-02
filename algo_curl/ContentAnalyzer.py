from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import sys, math, random
from Numeric import *
from sys import argv
import cPickle as pickle
import nltk.util
import urllib2
import socket
import numpy
import pycurl
import time
import csv

class Bayesian:
                     
    def __init__(self,count,c):
        global C
        global word_list_all
        self.count = count
        self.c = c
        if (self.count == 0):
            word_list,A = Bucket_Processing().return_()
            self.word_list = word_list
            self.word_list_all = []
            self.A = A                                                  
        if not(self.count == 0):
            
            self.A = Bucket_Processing().normalise(C)
            self.word_list = word_list_all                            
    def learning(self):                                       
        global C
        global word_list_all
        ini_length_word_list = len(self.word_list)
        length_link_text,length_list_link_text =0,0
        region_text = zeros([7],Int)
        region_link = zeros([7],Int) 
        i,site_value_text,site_value_link=0,0.0,0.0
        word_list_text = self.word_list                                                            
        word_list_link = self.word_list                                                            
        text = pickle.load(open('pfile_text.p'))
        link_text = str(text).lower().split(None)
        length_link_text = len(link_text)
        text = pickle.load(open('pfile_link.p'))
        list_link_text = str(text).lower().split(None)
        if (self.c==0):
            list_link_text=[]
            length_list_link_text=0
        length_list_link_text = len(list_link_text)

        for word in link_text: 
            if word in word_list_text:
                ind = word_list_text.index(word)
                for i in range(2,9):
                    if(self.A[ind][i]==1):             
                        region_text[i-2]+=1    
                site_value_text = site_value_text + self.A[ind][1]
                self.A[ind][0]+=1
        site_class_text = self.find_max(region_text)+2 
        mini_text,maxim_text,mini_link,maxim_link=0,0,0,0
        text_limit_good,text_limit_bad,link_limit_good,link_limit_bad =0.25*length_link_text,-0.25*length_link_text,0.1*length_list_link_text,-0.1*length_list_link_text
    
    
        if (site_value_text > text_limit_good  ):
            for word in link_text:
                if word in self.word_list:
                    ind = self.word_list.index(word)
                    if(mini_text > self.A[ind][1]):
                        mini_text = self.A[ind][1]
    
    
        if (site_value_text <=text_limit_bad ):
            for word in link_text:
                if word in self.word_list:
                    ind = self.word_list.index(word)
                    if(maxim_text < self.A[ind][1]):
                        maxim_text = self.A[ind][1]        
    
    
        for word in list_link_text: 
            if word in word_list_link:
                ind = word_list_link.index(word) 
                for i in range(2,9):
                    if(self.A[ind][i]==1):             
                        region_link[i-2]+=1 
                site_value_link = site_value_link + self.A[ind][1]
                self.A[ind][0]+=1 
        site_class_link = self.find_max(region_link)+2 
    
    
        if (site_value_link > link_limit_good ):
            for word in link_text:
                if word in self.word_list:
                    ind = self.word_list.index(word)
                    if(mini_link > self.A[ind][1]):
                        mini_link = self.A[ind][1]
    
    
        if (site_value_link <=link_limit_bad ):
            for word in link_text:
                if word in self.word_list:
                    ind = self.word_list.index(word)
                    if(maxim_link < self.A[ind][1]):
                        maxim_link = self.A[ind][1]   
        text_limit_good,text_limit_bad,link_limit_good,link_limit_bad =0.25*length_link_text,-0.25*length_link_text,0.1*length_list_link_text,-0.1*length_list_link_text
    
    
        if (site_value_text > text_limit_good): 
            for word in link_text:
                if word not in word_list_link:               
                    word_list_link.append(word)
                    ind = self.word_list.index(word)
                    if((site_value_text > text_limit_good)and(site_value_link > link_limit_good)):
                        self.A[ind][1] = abs(min(mini_link,mini_text))                       
                    else:
                        self.A[ind][1] = -max(maxim_link,maxim_text,site_value_text,site_value_link)                        
                    f=0
                    for i in range(1,7):
                        if (region_text[i]>=1):
                            f=1
                            self.A[ind][i+2]=1
                        if (f==0):
                            self.A[ind][2]=1                                                                         
    
    
        if (site_value_text <= text_limit_bad): 
            for word in link_text:
                if word not in word_list_link:                
                    word_list_link.append(word)
                    ind = self.word_list.index(word)
                    if((site_value_text > text_limit_good)and(site_value_link > link_limit_good)):
                        self.A[ind][1] = abs(min(mini_link,mini_text))                       
                    else:
                        self.A[ind][1] = -max(maxim_link,maxim_text,site_value_text,site_value_link)                        
                    f=0
                    for i in range(1,7):
                        if (region_text[i]>=1):
                            f=1
                            self.A[ind][i+2]=1
                        if (f==0):
                            self.A[ind][2]=1                           
   
   
        if (site_value_link > link_limit_good): 
            for word in list_link_text:
                if word not in word_list_text:                
                    word_list_text.append(word)
                    ind = self.word_list.index(word)
                    if((site_value_text > text_limit_good)and(site_value_link > link_limit_good)):
                        self.A[ind][1] = abs(min(mini_link,mini_text))                  
                    else:
                        self.A[ind][1] = -max(maxim_link,maxim_text,site_value_text,site_value_link)                        
                    f=0
                    for i in range(1,7):
                        if (region_link[i]>=1):
                            f=1
                            self.A[ind][i+2]=1
                        if (f==0):
                            self.A[ind][2]=1                        
    
    
        if (site_value_link <= link_limit_bad): 
            for word in list_link_text:
                if word not in word_list_text:                     
                    word_list_text.append(word)
                    ind = self.word_list.index(word)
                    if((site_value_text > text_limit_good)and(site_value_link > link_limit_good)):
                        self.A[ind][1] = abs(min(mini_link,mini_text))           
                    else:

                        self.A[ind][1] = -max(maxim_link,maxim_text,site_value_text,site_value_link)
                    f=0
                    for i in range(1,7):
                        if (region_link[i]>=1):
                            f=1
                            self.A[ind][i+2]=1
                        if (f==0):
                            self.A[ind][2]=1 
                        
        word_list_all = self.word_list
        C = self.A
        return region_text,region_link,length_list_link_text,length_link_text
    
    def find_max(self,a):
        maxi=0
        max_index=0
        for i in range(6):
            if(maxi < a[i]):
                maxi=a[i]
                max_index = i
        return max_index

class Bucket_Processing:
    
    
    def __init__(self):
        self.word_list = []
        self.A = zeros([1000000,9],Float)
        self.i = 0

    def return_(self):
        
        self.perform()
        return self.word_list,self.A    
    
    def frequency(self,text,freq,num):
        word_list_temp ,word_list_tempo,links_list,word_list_temporary=[],[],[],[]   
        word_list_temp = str(text).lower().split(None)  
        for word in word_list_temp:
            if word not in self.word_list:
                self.word_list.append(word)
        for word in word_list_temp:
            freq[word] = freq.get(word, 0) + 1       
            keys = freq.keys()
        
        for ind,word in enumerate(self.word_list):
            try:
                self.A[ind][0]= self.A[ind][0] + int(freq[word]) 
            except Exception as e:
                continue  
        norm_f =0.0
        p=0.0
    
        for word in freq:
            norm_f = norm_f + freq[word]
    
        for word in freq:
            p= freq[word]/norm_f
            freq[word]=p 
    
        for word in word_list_temp:
            ind = self.word_list.index(word)
            if(self.A[ind][2] == 1):
                self.A[ind][2] = 0
                self.A[ind][num] = 1
            elif(self.A[ind][2]!=1):
                self.A[ind][num] = 1
       
        for ind,word in enumerate(self.word_list):                                                    
            try:
                if(self.A[ind][2]==1):
                    self.A[ind][1] =  self.A[ind][1] + float(freq[word])
                elif(self.A[ind][2]!=1):
                    self.A[ind][1] =  self.A[ind][1] - float(freq[word])
            except Exception as e:
                continue
        return self.A 
    def normalise(self,B):
        norm_fac=0.0
        for ind,word in enumerate(self.word_list):
            norm_fac = norm_fac + B[ind][0]
        for ind,word in enumerate(self.word_list):
            B[ind][1] = (B[ind][1])/norm_fac
        return B   
            

    def perform(self):
        good_word_freq,bad_porn_freq,bad_violence_freq,bad_racism_freq,bad_drugs_freq,bad_alcohol_freq,bad_tobacco_freq = {},{},{},{},{},{},{}
        good_text,porn_text,violence_text,racism_text,drugs_text, alcohol_text,tobacco_text = [],[],[],[],[],[],[]

        good_text = self.openandstem('good_temp.txt')
        self.frequency(good_text,good_word_freq,2)
    
        porn_text = self.openandstem('porn_temp.txt')
        self.frequency(porn_text,bad_porn_freq,3)
    
        violence_text = self.openandstem('violence_temp.txt')
        self.frequency(violence_text,bad_violence_freq,4)
    
        racism_text = self.openandstem('racism_temp.txt')
        self.frequency(racism_text,bad_racism_freq,5)
    
        drugs_text = self.openandstem('drugs_temp.txt')  
        self.frequency(drugs_text,bad_drugs_freq,6)
    
        alcohol_text = self.openandstem('alcohol_temp.txt')
        self.frequency(alcohol_text,bad_alcohol_freq,7)
    
        tobacco_text = self.openandstem('tobacco_temp.txt')
        self.frequency(tobacco_text,bad_tobacco_freq,8) 
    
        return True
 
                  
    def openandstem(self,file1):
        doc = open(file1, 'r').read()
        return Stemming().stem(doc)    

class Content_Classifier_1:

    
    def __init__(self,c,count):
        self.count = count 
        self.c = c
        self.region_text,self.region_link,self.length_list_link_text,self.length_link_text = Bayesian(self.count,self.c).learning()
            
    def porn_text(self):
        if (self.region_text[1]>self.length_link_text/6): 
            return 1
        elif (self.region_text[1]<=self.length_link_text/6):
            return 0
        
    def porn_link(self):
        if (self.region_link[1]>self.length_list_link_text/6):
            return 1
        elif(self.region_link[1]<=self.length_list_link_text/6):
            return 0
        
    def violence_text(self):
        if (self.region_text[2]>self.length_link_text/6):
            return 1
        elif (self.region_text[2]<=self.length_link_text/6):
            return 0
    def violence_link(self):
        if (self.region_link[2]>self.length_list_link_text/6):
            return 1
        elif(self.region_link[2]<=self.length_list_link_text/6):
            return 0
        
        
    def racism_text(self):
        if (self.region_text[3]>self.length_link_text/6):       
            return 1
        elif (self.region_text[3]<=self.length_link_text/6):
            return 0

    def racism_link(self):
        if (self.region_link[3]>self.length_list_link_text/6):       
            return 1
        elif(self.region_link[3]<=self.length_list_link_text/6):
            return 0        

    def drugs_text(self):
        if (self.region_text[4]>self.length_link_text/6):
            return 1
        elif (self.region_text[4]<=self.length_link_text/6):
            return 0
    def drugs_link(self):
        if (self.region_link[4]>self.length_list_link_text/6):
            return 1
        elif(self.region_link[4]<=self.length_list_link_text/6):
            return 0
        
        
    def alcohol_text(self):
        if (self.region_text[5]>self.length_link_text/6):
            return 1
        elif (self.region_text[5]<=self.length_link_text/6):
            return 0

    def alcohol_link(self):
        if (self.region_link[5]>self.length_list_link_text/6):
            return 1
        elif(self.region_link[5]<=self.length_list_link_text/6):
            return 0        
        
    def tobacco_text(self):
        if (self.region_text[6]>self.length_link_text/6):
            return 1

        elif (self.region_text[6]<=self.length_link_text/6):
            return 0
        
        
    def tobacco_link(self):
        if (self.region_link[6]>self.length_list_link_text/6):
            return 1
        elif(self.region_link[6]<=self.length_list_link_text/6):
            return 0        
        
        
class Content_Classifier_2:

    
    def __init__(self,c,count):
        self.count = count
        self.c = c
        self.region_text,self.region_link,self.length_list_link_text,self.length_link_text = Bayesian(self.count,self.c).learning()
         
    def porn_text(self):
        if (self.region_text[1]>self.length_link_text/2): 
            return 5       
        elif (self.region_text[1]>self.length_link_text/4 and self.region_text[1]<=self.length_link_text/2):
            return 4        
        elif (self.region_text[1]<=self.length_link_text/4 and self.region_text[1]>self.length_link_text/6):
            return 3       
        elif(self.region_text[1]<=self.length_link_text/6 and self.region_text[1]>self.length_link_text/8):
            return 2
        elif(self.region_text[1]<=self.length_link_text/8):
            return 1
        
        
    def porn_link(self):
      if (self.region_link[1]>self.length_list_link_text/2): 
          return 5
      elif (self.region_link[1]>self.length_list_link_text/4 and self.region_link[1]<=self.length_list_link_text/2):  
          return 4
      elif (self.region_link[1]<=self.length_list_link_text/4 and self.region_link[1]>self.length_list_link_text/6):   
          return 3       
      elif(self.region_link[1]<=self.length_list_link_text/6 and self.region_link[1]>self.length_list_link_text/8):
          return 2       
      elif(self.region_link[1]<=self.length_list_link_text/8):
          return 1    
    
    def violence_text(self):
        global count
        if (self.region_text[2]>self.length_link_text/2):
            return 5         
        elif (self.region_text[2]>self.length_link_text/4 and self.region_text[2]<=self.length_link_text/2):
            return 4           
        elif (self.region_text[2]<=self.length_link_text/4 and self.region_text[2]>self.length_link_text/6):
            return 3           
        elif (self.region_text[2]<=self.length_link_text/6 and self.region_text[2]>self.length_link_text/8):
            return 2          
        elif(self.region_text[2]<=self.length_link_text/8):
            return 1         
                       
    def violence_link(self):
      if (self.region_link[2]>self.length_list_link_text/2):
          return 5
      elif (self.region_link[2]>self.length_list_link_text/4 and self.region_link[2]<=self.length_list_link_text/2):
          return 4    
      elif (self.region_link[2]<=self.length_list_link_text/4 and self.region_link[2]>self.length_list_link_text/6):
          return 3         
      elif (self.region_link[2]<=self.length_list_link_text/6 and self.region_link[2]>self.length_list_link_text/8):
          return 2         
      elif(self.region_link[2]<=self.length_list_link_text/8):
          return 1         
      
        
    def racism_text(self):
      global count
      if (self.region_text[3]>self.length_link_text/2):
          return 5          
      elif (self.region_text[3]>self.length_link_text/4 and self.region_text[3]<=self.length_link_text/2):
          return 4         
      elif (self.region_text[3]<=self.length_link_text/4 and self.region_text[3]>self.length_link_text/6):
          return 3            
      elif(self.region_text[3]<=self.length_link_text/6 and self.region_text[3]>self.length_link_text/8):
          return 2         
      elif(self.region_text[3]<=self.length_link_text/8):
          return 1               
        
    def racism_link(self):
      if (self.region_link[3]>self.length_list_link_text/2):
          return 5
      elif (self.region_link[3]>self.length_list_link_text/4 and self.region_link[3]<=self.length_list_link_text/2):
          return 4 
      elif (self.region_link[3]<=self.length_list_link_text/4 and self.region_link[3]>self.length_list_link_text/6):
          return 3
      elif(self.region_link[3]<=self.length_list_link_text/6 and self.region_link[3]>self.length_list_link_text/8):
          return 2
      elif(self.region_link[3]<=self.length_list_link_text/8):
          return 1 
      
      
    def drugs_text(self):
      global count
      if (self.region_text[4]>self.length_link_text/2):  
          return 5  
      elif (self.region_text[4]>self.length_link_text/4 and self.region_text[4]<=self.length_link_text/2):  
          return 4 
      elif (self.region_text[4]<=self.length_link_text/4 and self.region_text[4]>self.length_link_text/6): 
          return 3   
      elif(self.region_text[4]<=self.length_link_text/6 and self.region_text[4]>self.length_link_text/8):   
          return 2 
      elif(self.region_text[4]<=self.length_link_text/8):
          return 1 
                
    def drugs_link(self):
      if (self.region_link[4]>self.length_list_link_text/2):
          return 5
      elif (self.region_link[4]>self.length_list_link_text/4 and self.region_link[4]<=self.length_list_link_text/2):
          return 4
      elif (self.region_link[4]<=self.length_list_link_text/4 and self.region_link[4]>self.length_list_link_text/6):
          return 3
      elif(self.region_link[4]<=self.length_list_link_text/6 and self.region_link[4]>self.length_list_link_text/8):
          return 2
      elif(self.region_link[4]<=self.length_list_link_text/8):
          return 1
      
      
    def alcohol_text(self):
      global count
      if (self.region_text[5]>self.length_link_text/2):
          return 5
      elif (self.region_text[5]>self.length_link_text/4 and self.region_text[5]<=self.length_link_text/2):  
          return 4
      elif (self.region_text[5]<=self.length_link_text/4 and self.region_text[5]>self.length_link_text/6): 
          return 3  
      elif(self.region_text[5]<=self.length_link_text/6 and self.region_text[5]>self.length_link_text/8):   
          return 2
      elif(self.region_text[5]<=self.length_link_text/8):
          return 1
      
      
    def alcohol_link(self):
      if (self.region_link[5]>self.length_list_link_text/2):
          return 5
      elif (self.region_link[5]>self.length_list_link_text/4 and self.region_link[5]<=self.length_list_link_text/2):
          return 4
      elif (self.region_link[5]<=self.length_list_link_text/4 and self.region_link[5]>self.length_list_link_text/6):
          return 3
      elif(self.region_link[5]<=self.length_list_link_text/6 and self.region_link[5]>self.length_list_link_text/8):
          return 2
      elif(self.region_link[5]<=self.length_list_link_text/8):
          return 1
      
      
    def tobacco_text(self):
      global count
      if (self.region_text[6]>self.length_link_text/2):   
          return 5
      elif (self.region_text[6]>self.length_link_text/4 and self.region_text[6]<=self.length_link_text/2):
          return 4
      elif (self.region_text[6]<=self.length_link_text/4 and self.region_text[6]>self.length_link_text/6):  
          return 3 
      elif(self.region_text[6]<=self.length_link_text/6 and self.region_text[6]>self.length_link_text/8):   
          return 2
      elif(self.region_text[6]<=self.length_link_text/8):   
          return 1
            
    def tobacco_link(self):
      if (self.region_link[6]>self.length_list_link_text/2):
          return 5
      elif (self.region_link[6]>self.length_list_link_text/4 and self.region_link[6]<=self.length_list_link_text/2):
          return 4
      elif (self.region_link[6]<=self.length_list_link_text/4 and self.region_link[6]>self.length_list_link_text/6):
          return 3
      elif(self.region_link[6]<=self.length_list_link_text/6 and self.region_link[6]>self.length_list_link_text/8):
          return 2
      elif(self.region_link[6]<=self.length_list_link_text/8):          
          return 1 

class ContentAnalyzer_Utils:    
          
    def __init__(self,html,links):
        self.html = html
        self.links = links
        self.k = 0
        self.timeout =10
        socket.setdefaulttimeout(self.timeout)
      
    def read_text(self):                              
        seed_list = []
        pure_text = nltk.util.clean_html(self.html) 
        stemmed_text = Stemming().stem(pure_text) 
        
        seed_list = self.links
        link_stemmed_text = Stemming().stem(self.fetch_page(seed_list))
    
        return stemmed_text, link_stemmed_text,self.k

    def fetch_page(self,seed_list):               
	    link_pure_text = []
	    for seed in seed_list: 
	    	try:
                    t = Test()
                    c = pycurl.Curl()
                    c.setopt(pycurl.HEADER, 1)  
                    c.setopt(pycurl.FOLLOWLOCATION, 1)
                    c.setopt(pycurl.URL,seed)
                    c.setopt(pycurl.WRITEFUNCTION, t.body_callback)
                    c.perform()
                    c.close()
                    link_pure_text.append(nltk.util.clean_html(t.contents)) 
                    self.k = self.k+1
                    if(self.k==4):
                        break
	    	except Exception as e:
                    continue	
            return link_pure_text  
    
class Stemming:
    
    def __init__(self):
        pass
    
    def stem(self,input_text):
       tokenizer = RegexpTokenizer('\s+', gaps=True)
       stemmed_text=[]
       lemmatizer = WordNetLemmatizer()
       stemmer = LancasterStemmer() 
       text = tokenizer.tokenize(str(input_text))
       filtered_text = self.stopword(text)            
       for word in filtered_text:
           if word.isalpha():
               stemmed_text.append(stemmer.stem(word).lower())
      
       ' '.join(stemmed_text)
      
       return stemmed_text    
   
   
    def stopword(self,text):
          filtered_text = text[:]  
          stopset = set(stopwords.words('english'))      
          for word in text:                # iterate over word_list
             if len(word) < 3 or word.lower() in stopset: 
                 filtered_text.remove(word) 
          return filtered_text 
             
                    
class Test:
        def __init__(self):
                self.contents = ''

        def body_callback(self, buf):
                self.contents = self.contents + buf  

class ContentAnalyzer:

    def __init__(self):
        global data1,data2,i
        data1  = zeros([13],Int)
        data2 = zeros([13],Int)
        i = 0         
    
    def Classify_1(self,string,html,link):
        start_time = time.time()
        global i
        global j
        global data1
        if(i==0):
            
            stemmed_text,link_stemmed_text,c = ContentAnalyzer_Utils(html,link).read_text()
            open('Result_1.csv','w')
            open('Result_2.csv','w') 
            pickle.dump(stemmed_text, open('pfile_text.p','wb'))
            pickle.dump(link_stemmed_text,open('pfile_link.p','wb'))
            
            p = Content_Classifier_1(c,i)
            data1[1] = p.porn_text()
            data1[2] = p.porn_link()
            data1[3] = p.violence_text()
            data1[4] = p.violence_link()
            data1[5] = p.racism_text()
            data1[6] = p.racism_link()
            data1[7] = p.drugs_text()
            data1[8] = p.drugs_link()
            data1[9] = p.alcohol_text()
            data1[10] = p.alcohol_link()
            data1[11] = p.tobacco_text()
            data1[12] = p.tobacco_link()
            i = 1 
        elif(i==1):
            stemmed_text,link_stemmed_text,c = ContentAnalyzer_Utils(html,link).read_text()
    	
            pickle.dump(stemmed_text, open('pfile_text.p','wb'))
            pickle.dump(link_stemmed_text,open('pfile_link.p','wb'))
   
            p = Content_Classifier_1(c,i)
            
            data1[1] = p.porn_text()
            data1[2] = p.porn_link()
            data1[3] = p.violence_text()
            data1[4] = p.violence_link()
            data1[5] = p.racism_text()
            data1[6] = p.racism_link()
            data1[7] = p.drugs_text()
            data1[8] = p.drugs_link()
            data1[9] = p.alcohol_text()
            data1[10] = p.alcohol_link()
            data1[11] = p.tobacco_text()
            data1[12] = p.tobacco_link()  	
    
        fd_1 = open('Result_1.csv','a')
        s = []
        reswriter = csv.writer(fd_1,delimiter  = ' ')
        s[:] = data1
        reswriter.writerow([string,s])
        print (time.time()-start_time),"Elapsed Time using time.time()"

       
    def Classify_2(self,string,html,link):
        start_time = time.time()
        global i
        global j
        global data2
        if(i==0):
            stemmed_text,link_stemmed_text,c = ContentAnalyzer_Utils(html,link).read_text()
            open('Result_1.csv','w')
            open('Result_2.csv','w')
            pickle.dump(stemmed_text, open('pfile_text.p','wb'))
            pickle.dump(link_stemmed_text,open('pfile_link.p','wb'))
            p = Content_Classifier_2(c,i)
            
            data2[1] = p.porn_text()
            data2[2] = p.porn_link()
            data2[3] = p.violence_text()
            data2[4] = p.violence_link()
            data2[5] = p.racism_text()
            data2[6] = p.racism_link()
            data2[7] = p.drugs_text()
            data2[8] = p.drugs_link()
            data2[9] = p.alcohol_text()
            data2[10] = p.alcohol_link()
            data2[11] = p.tobacco_text()
            data2[12] = p.tobacco_link()
            i = 1
	
        elif(i==1):
            stemmed_text,link_stemmed_text,c = ContentAnalyzer_Utils(html,link).read_text()
            pickle.dump(stemmed_text, open('pfile_text.p','wb'))
            pickle.dump(link_stemmed_text,open('pfile_link.p','wb'))
            p = Content_Classifier_2(c,i)
       
            data2[1] = p.porn_text()
            data2[2] = p.porn_link()
            data2[3] = p.violence_text()
            data2[4] = p.violence_link()
            data2[5] = p.racism_text()
            data2[6] = p.racism_link()
            data2[7] = p.drugs_text()
            data2[8] = p.drugs_link()
            data2[9] = p.alcohol_text()
            data2[10] = p.alcohol_link()
            data2[11] = p.tobacco_text()
            data2[12] = p.tobacco_link()     
        fd_2 = open('Result_2.csv','a')
        s = []
        reswriter = csv.writer(fd_2,delimiter  = ' ')
        s[:] = data2
        reswriter.writerow([string,s])      
        print (time.time()-start_time),"Elapsed Time using time.time()"


