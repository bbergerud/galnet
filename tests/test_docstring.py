import unittest
import doctest

modules = (
    "galnet.layers.blocks",
    "galnet.layers.padding",
    "galnet.layers.recurrent",
    "galnet.layers.upsample",
    "galnet.utils",
)

suite = unittest.TestSuite()
for mod in modules:
    suite.addTest(doctest.DocTestSuite(mod))
runner = unittest.TextTestRunner()
runner.run(suite)