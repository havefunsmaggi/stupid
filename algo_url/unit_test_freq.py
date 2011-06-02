import unittest
from Numeric import *
import numpy
from ContentAnalyzer import Bucket_Processing

class testContentAnalyzer_bucket(unittest.TestCase):


    def setUp(self):

        self.freq = Bucket_Processing()
        self.word_freq = {}

    def testFrequency(self):
        
        A = zeros([9],Float)  
        self.assertEqual(self.freq.frequency("Hi Hi hi hi HI",self.word_freq,3),A)
    
def suite():

    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(testContentAnalyzer_bucket)) 

    return suite
if __name__ == '__main__':

    unittest.main()
    #suiteFew = unittest.TestSuite()

    #suiteFew.addTest(testBlogger("testFrequency"))


    #unittest.TextTestRunner(verbosity=2).run(suiteFew)

    #unittest.TextTestRunner(verbosity=1).run(suite())

