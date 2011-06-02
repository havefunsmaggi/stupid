import unittest

from ContentAnalyzer import Stemming

class testContentAnalyzer_stemming(unittest.TestCase):

    def setUp(self):

        self.stemmer = Stemming()

    def tearDown(self):
        self.stemmer = None
    
    def testStemmedWords(self):

        stemmed_text = ['jump']

        self.assertEqual(self.stemmer.stem("Jumped"), stemmed_text)

    def testStopWords(self):

        stop_knock = []

        self.assertEqual(self.stemmer.stopword(['The','a','an','and']),stop_knock)
    
def suite():

    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(testContentAnalyzer_stemming))

    return suite

if __name__ == '__main__':

    #unittest.main()
    
    suiteFew = unittest.TestSuite()

    suiteFew.addTest(testBlogger("testStemmedWords"))

    suiteFew.addTest(testBlogger("testStopWords"))

    #unittest.TextTestRunner(verbosity=2).run(suiteFew)

    unittest.TextTestRunner(verbosity=2).run(suite())


