from typing import List, Tuple, Dict


class ReasoningPath:
    def __init__(self, entity: str, path: str = None):
        self.entity = str(entity)
        self.relational_path: List[str] = []
        self.entities_path: List[str] = []
        if path:
            self.str2path(path)

    def __str__(self):
        join_str = " -> "
        ret = self.entity
        for rel, ent in zip(self.relational_path, self.entities_path):
            ret += join_str + rel + join_str + ent
        return ret

    def add_triple(self, triple: Tuple[str, str, str]):
        self.relational_path.append(triple[1])
        self.entities_path.append(triple[2])

    def __repr__(self) -> str:
        return self.__str__()

    def str2path(self, path: str):
        path_parts = path.split(" -> ")
        if len(path_parts) < 1:
            return
        self.entity = path_parts[0]
        self.relational_path = path_parts[1::2]
        self.entities_path = path_parts[2::2]
