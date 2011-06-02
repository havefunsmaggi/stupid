import unittest
import cPickle as pickle
from ContentAnalyzer import Bayesian
from ContentAnalyzer import ContentAnalyzer
from ContentAnalyzer import Stemming
import nltk.util
import os
class TestBayes(unittest.TestCase):
    
    def setUp(self):
        c = ContentAnalyzer()
        p = open('unittest.html').read()
        pickle.dump(Stemming().stem(nltk.util.clean_html(p)), open('pfile_text.p','wb'))
        pickle.dump(Stemming().stem(nltk.util.clean_html(p)), open('pfile_link.p','wb'))    
        self.Bayes = Bayesian(0,1)

    def tearDown(self):
        self.Bayes = None         
        os.remove('pfile_text.p')
        os.remove('pfile_link.p')     
    
    def test_learning(self):
        region_text = [ 43, 25, 24, 1, 18, 10, 3]
        region_link =  [ 43, 25, 24, 1, 18, 10, 3]
        length_list_link_text = 155
        length_link_text = 155
        self.assertEqual(self.Bayes.learning(),(region_text,region_link,length_list_link_text,length_link_text))

def suite():

    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(TestBayes))

    return suite     
       
       
if __name__ == "__main__":
    
    #unittest.main()
    suiteFew = unittest.TestSuite()       
    
    suiteFew.addTest(TestBayes("test_learning"))
    
    unittest.TextTestRunner(verbosity=1).run(suite())       

 
