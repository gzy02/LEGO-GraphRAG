from dataclasses import dataclass
from typing import List, Tuple, Dict
from utils.Tools import Tools


@dataclass
class Entity:
    eid: str
    name: str

    @classmethod
    def from_eid(cls, eid: str):
        tool = Tools()
        name = tool.id2name(eid)
        return cls(eid=eid, name=name)

    @classmethod
    def from_name(cls, name: str):
        tool = Tools()
        eid = tool.name2id(name)
        return cls(eid=eid, name=name)

    def __str__(self):
        return self.eid
