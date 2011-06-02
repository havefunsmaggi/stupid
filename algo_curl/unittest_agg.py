import unittest
import unit_test_utils
import unit_test_Bayesian
import unit_test_stemming
import unit_test_learning

suite1 = unit_test_stemming.suite()
suite2 = unit_test_utils.suite()
suite3 = unit_test_learning.suite()
suite4 = unit_test_Bayesian.suite()

suite = unittest.TestSuite()
suite.addTest(suite1)
suite.addTest(suite2)
suite.addTest(suite3)
suite.addTest(suite4)

unittest.TextTestRunner(verbosity=4).run(suite)
