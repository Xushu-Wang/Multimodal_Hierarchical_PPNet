import unittest
import torch
from dataio import tree

class TestGeneticOneHot(unittest.TestCase): 

    def test_with_zero_encode_unknown(self): 
        goh = tree.GeneticOneHot(length=8, zero_encode_unknown=True)
        seq = "ACGT"
        pred = goh(seq)
        truth = torch.cat([torch.eye(4, dtype=torch.float), torch.zeros(4, 4)], dim=1)
        self.assertTrue(torch.all(pred==truth))

    def test_without_zero_encode_unknown(self): 
        goh = tree.GeneticOneHot(length=8, zero_encode_unknown=False)
        seq = "ACGT"
        pred = goh(seq)
        truth = torch.cat([torch.eye(4, dtype=torch.float), torch.zeros(4, 4)], dim=1)
        truth = torch.cat([torch.Tensor([0] * 4 + [1] * 4).reshape(1, -1), truth], dim=0)
        self.assertTrue(torch.all(pred==truth))


if __name__ == "__main__": 
    unittest.main()

