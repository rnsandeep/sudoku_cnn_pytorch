# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/sudoku
'''
import numpy as np
from hyperparams import Hyperparams as hp

def load_data(type="train"):
    '''Loads training / test data.
    
    Args
      type: A string. Either `train` or `test`.
    
    Returns:
      X: A 3-D array of float. Entire quizzes. 
         Has the shape of (# total games, 9, 9) 
      Y: A 3-D array of int. Entire solutions.
        Has the shape of (# total games, 9, 9)
    '''
    fpath = hp.train_fpath if type=="train" else hp.test_fpath
    lines = open(fpath, 'r').read().splitlines()[1:]
    nsamples = len(lines)
    
    X = np.zeros((nsamples, 9*9), np.float32)  
    Y = np.zeros((nsamples, 9*9), np.int32) 
    
    for i, line in enumerate(lines): #[:1000]):
        quiz, solution = line.split(",")
        for j, (q, s) in enumerate(zip(quiz, solution)):
            X[i, j], Y[i, j] = q, s
    
    X = np.reshape(X, (-1, 9, 9))
    Y = np.reshape(Y, (-1, 9, 9))
    return X, Y
        
