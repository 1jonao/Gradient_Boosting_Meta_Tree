import numpy as np
from typing import Optional, Union, Generator
class GenThresholdCandidates():
    def __init__(
            self,
            seed: Union[int, Generator, None],
            ):
        self.rng = np.random.default_rng(seed)

    def all_thresholds(
        self,
        x_continuous_vecs: np.ndarray,
        feature: int,
        num_thresholds: None,
        ): # FIXME: feature = -1でも動いてしまう
        thresholds = np.sort(x_continuous_vecs[:,feature])[:-1]
        thresholds = np.unique(thresholds) # 同一の閾値を除外
        return thresholds
    
    def by_histogram(
        self,
        x_continuous_vecs: np.ndarray,
        feature: int,
        num_thresholds: Union[int,str],
        ):
        if type(num_thresholds) is int:
            if num_thresholds > x_continuous_vecs.shape[1]:
                _num_thresholds = x_continuous_vecs.shape[1]
            else:
                _num_thresholds = num_thresholds
        elif num_thresholds == 'log2':
            _num_thresholds = int(np.log2(x_continuous_vecs.shape[1])+1)
        # if x_continuous_vecs[:,feature]
        x_min = np.min(x_continuous_vecs[:,feature])
        x_max = np.max(x_continuous_vecs[:,feature])
        if x_min == x_max:
          return np.array([x_min])
        else:
          thresholds = np.arange(x_min + (x_max-x_min) / (_num_thresholds + 1), x_max, (x_max - x_min) / (_num_thresholds + 1))
          thresholds = np.unique(thresholds) # 同一の閾値を除外
          return thresholds[:_num_thresholds]
        
    def by_quantile(
        self,
        x_continuous_vecs: np.ndarray,
        feature: int,
        num_thresholds: int,
        ):
        if num_thresholds > x_continuous_vecs.shape[1]:
            num_thresholds = x_continuous_vecs.shape[1]
        else:
            pass
        step = 100 / (num_thresholds + 1)
        percintiles = np.arange(0,100 + step, step)[1:-1]
        thresholds = np.percentile(x_continuous_vecs[:,feature],percintiles)
        thresholds = np.unique(thresholds) # 同一の閾値を除外
        return thresholds
    
    def by_random(
        self,
        x_continuous_vecs: np.ndarray,
        feature: int,
        num_thresholds: int,
        ):
        x_min = np.min(x_continuous_vecs[:,feature])
        x_max = np.max(x_continuous_vecs[:,feature])
        thresholds = self.rng.uniform(x_min,x_max,num_thresholds)
        thresholds = np.unique(thresholds) # 同一の閾値を除外
        return thresholds

if __name__ == '__main__':
    a = np.array([
      [1,1,1,1,1,1],
      [2,2,2,2,2,2],
      [3,3,3,3,3,3],
      [4,4,4,4,4,4],
    ])

    b = np.random.random((2,5))
    print(a)

    gen = GenThresholdCandidates(0)
    print('a-----')
    print('all:', gen.all_thresholds(a, 0, None))
    print('hist:', gen.by_histogram(a, 1, 'log2'))
    print('quantile:', gen.by_quantile(a, 2, 6))
    print('random:', gen.by_random(a, 3, 5))
    print('b-----')
    print('all:', gen.all_thresholds(b, 0, None))
    # print('hist:', gen.by_histogram(b, 1, 1))
    # print('quantile:', gen.by_quantile(b, 2, 6))
    # print('random:', gen.by_random(b, 3, 5))
