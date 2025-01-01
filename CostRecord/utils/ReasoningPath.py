from typing import List, Tuple, Dict


class ReasoningPath:
    def __init__(self, entity: str):
        self.entity = str(entity)
        self.relational_path: List[str] = []
        self.entities_path: List[str] = []

    def __str__(self):
        join_str = " -> "
        ret = self.entity
        for i in zip(self.relational_path, self.entities_path):
            ret += join_str+i[0]+join_str+i[1]
        return ret

    def add_triple(self, triple: Tuple[str, str, str]):
        self.relational_path.append(triple[1])
        self.entities_path.append(triple[2])

    def __repr__(self) -> str:
        return self.__str__()
