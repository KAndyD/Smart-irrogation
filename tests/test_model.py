import json
import unittest
import numpy as np
from common import ROOT
from model import fronts,crowding,generate,simulate,run,coverage

class TestNSGA(unittest.TestCase):
    def test_fronts_against_hand_computed_example(self):
        f=np.array([[0,2],[1,1],[2,0],[2,2],[3,3],[1,1]])
        result=fronts(f)
        self.assertEqual(set(result[0]),{0,1,2,5}); self.assertEqual(set(result[1]),{3}); self.assertEqual(set(result[2]),{4})
    def test_crowding_boundaries(self):
        f=np.array([[0,2],[1,1],[2,0]])
        d=crowding(f,np.arange(3)); self.assertTrue(np.isinf(d[0]) and np.isinf(d[2])); self.assertEqual(d[1],2)
    def test_water_balance(self):
        data=dict(initial_moisture=22,rain=[0,0],evaporation=[3,3],lower_comfort=20,upper_comfort=30,capacity=45)
        f,h=simulate(np.array([0.,0.]),data)
        np.testing.assert_array_equal(h,[[19,16]]); np.testing.assert_array_equal(f,[[0,5,0]])
    def test_equal_budget_bounds_and_determinism(self):
        cfg=json.loads((ROOT/'config.json').read_text()); cfg['generations']=2; data=generate(cfg['data_seed'],cfg['days'])
        a=run(cfg,data,1); b=run(cfg,data,1); c=run(cfg,data,1,np.ones(3)/3)
        np.testing.assert_array_equal(a[0],b[0]); self.assertEqual(a[3],3*cfg['population']); self.assertEqual(a[3],c[3])
        self.assertTrue(np.all((a[0]>=0)&(a[0]<=cfg['max_irrigation'])))
        self.assertEqual(len(fronts(a[1])[0]),len(a[1])); self.assertEqual(coverage(a[1],a[1]),1.)
if __name__=='__main__': unittest.main()
