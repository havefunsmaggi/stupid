import unittest
import urllib2 
from ContentAnalyzer import ContentAnalyzer_Utils, Bayesian,Bucket_Processing


class TestUtils(unittest.TestCase):



   def setUp(self):
      
      self.html  = 'hello world'
      self.link = ''
      self.Utils = ContentAnalyzer_Utils(self.html,self.link) 

   def tearDown(self):
      self.html = None
      self.link = None
      self.Utils = None
   def test_read_text(self):
          
       self.assertEqual(self.Utils.read_text(),  (['hello', 'world'], [], 0))
           
   def test_fetch_page(self):
         
      self.assertEqual(self.Utils.fetch_page(self.link), []) 

def suite():

    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(TestUtils))

    return suite     


if __name__ == "__main__":
      
   #unittest.main()  
      
   suiteFew = unittest.TestSuite()       
    
   suiteFew.addTest(TestUtils("test_read_text"))
   suiteFew.addTest(TestUtils("test_fetch_page"))
   
   unittest.TextTestRunner(verbosity=2).run(suite())       
    
                      
                      


