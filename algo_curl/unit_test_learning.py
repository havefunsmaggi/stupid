import unittest
from ContentAnalyzer import Bucket_Processing
from Numeric import *
import numpy
import os


class TestBucket(unittest.TestCase):
       
   def setUp(self):
       
       
       self.text = 'hello world'
       self.freq = {}
       self.num =  0
       self.bucket = Bucket_Processing()
       # def tearDown(self):
       self.A = zeros([9],Float)
       self.wordlist = []
       
   def tearDown(self):
      self.text = None
      self.freq = None
      self.num = None
      self.bucket = None
      self.A = None
      self.wordlist = None
      
   def test_frequency(self):
          
      self.assertEqual(self.bucket.frequency(self.text,self.freq,self.num), self.A)
      
   def test_normalise(self):
      self.assertEqual(self.bucket.normalise(self.A), self.A)
      
   def test_perform(self):
      self.assertTrue(self.bucket.perform()) 

   def test_openandstem(self):
      open('file1.txt','w').write('hello my program')
      self.assertEqual(self.bucket.openandstem('file1.txt'),['hello','program'])
      os.remove('file1.txt')
   def test_return(self):
      self.assertTrue(self.bucket.return_())

def suite():

    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(TestBucket))

    return suite    
                      
if __name__ == "__main__":

   #unittest.main()

   suiteFew = unittest.TestSuite()       
    
   suiteFew.addTest(TestBucket("test_frequency"))
   suiteFew.addTest(TestBucket("test_normalise"))
   suiteFew.addTest(TestBucket("test_perform"))
   suiteFew.addTest(TestBucket("test_openandstem"))
   suiteFew.addTest(TestBucket("test_return"))
   unittest.TextTestRunner(verbosity=5).run(suite())   


