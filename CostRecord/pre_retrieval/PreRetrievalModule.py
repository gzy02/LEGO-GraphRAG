from abc import ABC, abstractmethod
from utils.Query import Query
from utils.Timer import TimedClass

import inspect
class PreRetrievalModule(ABC, TimedClass):
    @abstractmethod
    def process(self, query: Query) -> Query:
        """Process the query and return the updated query object"""
        pass

    def __str__(self):
        init_signature = inspect.signature(self.__init__)
        params = init_signature.parameters
        params_str = ', '.join(
            [f"{name}={getattr(self, name)}" for name in params if name != 'self'])
        return f"{self.__class__.__name__}({params_str})"
