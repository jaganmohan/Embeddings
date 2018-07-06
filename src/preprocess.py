
from abc import ABCMeta, abstractmethod

class Preprocess:
    __metalclass__ = ABCMeta
    """
      Wrapper for preprocessing of data, to be implemented
      for custom data
    """
    
    def __init__(self):
        super().__init__()
        
        
    @abstractmethod
    def process_data(self):
        pass

