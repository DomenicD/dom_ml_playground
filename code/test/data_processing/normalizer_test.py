import unittest
from code.src.data_processing.normalizer import Normalizer

class NormalizerTest(unittest.TestCase):
    def test_default_normalizer(self):
        normalizer = Normalizer()

        self.assertAlmostEqual(.27, normalizer.norm_input(.27))
        self.assertAlmostEqual(.27, normalizer.denorm_input(
            normalizer.norm_input(.27)))

        self.assertAlmostEqual(.27, normalizer.norm_output(.27))
        self.assertAlmostEqual(.27, normalizer.denorm_output(.27))


    def test_norm_input(self):
        normalizer = Normalizer(in_min = -50, in_max = 50,
                                norm_min = -3, norm_max = 3)
        
        self.assertAlmostEqual(-2.94, normalizer.norm_input(-49))
        self.assertAlmostEqual(0, normalizer.norm_input(0))
        self.assertAlmostEqual(1.98, normalizer.norm_input(33))


    def test_denorm_input(self):
        normalizer = Normalizer(in_min = -50, in_max = 50,
                                norm_min = -3, norm_max = 3)
        
        self.assertAlmostEqual(-49, normalizer.denorm_input(-2.94))
        self.assertAlmostEqual(0, normalizer.denorm_input(0))
        self.assertAlmostEqual(33, normalizer.denorm_input(1.98))


    def test_norm_output(self):
        normalizer = Normalizer(out_min = -20, out_max = 70,
                                norm_min = -3, norm_max = 3)
        
        self.assertAlmostEqual(-20, normalizer.norm_output(0))
        self.assertAlmostEqual(25, normalizer.norm_output(.5))
        self.assertAlmostEqual(2.5, normalizer.norm_output(.25))
        self.assertAlmostEqual(61, normalizer.norm_output(.9))


    def test_denorm_output(self):
        normalizer = Normalizer(out_min = -20, out_max = 70,
                                norm_min = -3, norm_max = 3)
        
        self.assertAlmostEqual(0, normalizer.denorm_output(-20))
        self.assertAlmostEqual(.5, normalizer.denorm_output(25))
        self.assertAlmostEqual(.25, normalizer.denorm_output(2.5))
        self.assertAlmostEqual(.9, normalizer.denorm_output(61))
