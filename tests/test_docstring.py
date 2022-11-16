import unittest
import doctest

modules = (
    "galnet.utils",
    "galnet.layers.padding",
)

suite = unittest.TestSuite()
for mod in modules:
    suite.addTest(doctest.DocTestSuite(mod))
runner = unittest.TextTestRunner()
runner.run(suite)