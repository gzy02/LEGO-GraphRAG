from retrieval.RetrievalModule import RetrievalModule
from utils.Query import Query
import igraph as ig
from utils.ReasoningPath import ReasoningPath
from typing import List, Tuple, Dict
from utils.PromptTemplate import filter_prompt, FILTER_PERSONA, RETRIEVAL_PERSONA, extract_prompt
from utils.LLM import LocalLLM
from utils.SemanticModel import SemanticModel
from copy import deepcopy


class RetrievalModuleLLM(RetrievalModule):
    def __init__(self, llm_model: LocalLLM = None, rank_model: SemanticModel = None, hop: int = 4, top_k: int = -1, beam_width: int = 8, thre: int = 64):
        super().__init__()
        self.hop = hop
        self.top_k = top_k
        self.beam_width = beam_width
        self.thre = thre
        self.llm_model = llm_model
        self.rank_model = rank_model

        self.tokenizer = self.llm_model.tokenizer
        self.target = self.llm_model.tokenizer.model_max_length//2 - \
            self.token_count(FILTER_PERSONA)  # 16k

    def token_count(self, query):
        tokenized_prediction = self.tokenizer.encode(query)
        return len(tokenized_prediction)

    def get_user_input(self, ranked_corpus: List[str], question: str, entities, target: int) -> Tuple[str, int]:
        l, r = 0, len(ranked_corpus)
        best_prompt = ""
        best_length = 0
        while l < r:
            m = (l + r) // 2
            prompt = self.prepare_prompt(question, entities, ranked_corpus[:m])
            length = self.token_count(prompt)
            if length > target:
                r = m
            else:
                l = m + 1
                if length > best_length:
                    best_prompt = prompt
                    best_length = length
        return best_prompt, l

    async def aprocess(self, query: Query) -> Query:
        return self.process(query)

    def process(self, query: Query) -> Query:
        query.reasoning_paths = self._get_reasoning_paths(query)
        return query

    def _get_reasoning_paths(self, query: Query) -> List[ReasoningPath]:
        reasoning_paths = []
        G = query.subgraph
        for entity in query.entities:
            try:
                start_vertex_id = G.vs.find(name=entity).index
            except:
                continue

            # Initialize beam with the starting entity
            beams = [([start_vertex_id], ReasoningPath(
                entity=G.vs[start_vertex_id]["name"]))]

            for step in range(self.hop):
                next_beams = []

                for path, reasoning_path in beams:
                    current_vertex_id = path[-1]
                    neighbors = G.vs[current_vertex_id].neighbors(mode="out")

                    for index, neighbor in enumerate(neighbors):
                        if index >= self.thre:
                            break
                        neighbor_id = neighbor.index
                        edge_id = G.get_eid(current_vertex_id, neighbor_id)
                        new_path = path + [neighbor_id]

                        cur_reasoning_path = deepcopy(reasoning_path)
                        edge = G.es[edge_id]
                        source_vertex = G.vs[edge.source]["name"]
                        target_vertex = G.vs[edge.target]["name"]
                        relation = edge["name"]
                        cur_reasoning_path.add_triple(
                            (source_vertex, relation, target_vertex))

                        next_beams.append((new_path, cur_reasoning_path))

                # Retain the top-k beams based on their scores
                path_dict = {str(reasoning_path): (path, reasoning_path)
                             for path, reasoning_path in next_beams}
                corpus = list(path_dict.keys())
                sorted_paths = self.rank_model.top_k(
                    query.question, corpus, -1)
                user_input, length = self.get_user_input(
                    sorted_paths, query.question, query.entities, self.target)
                llm_paths, input_token, output_token = self.extract_paths(
                    path_dict, user_input)
                query.input_tokens += input_token
                query.output_tokens += output_token
                if len(llm_paths) > self.beam_width:
                    llm_paths = llm_paths[:self.beam_width]
                elif len(sorted_paths) < self.beam_width:
                    for path in sorted_paths:
                        if path not in llm_paths:
                            llm_paths.append(path_dict[path])
                        if len(llm_paths) == self.beam_width:
                            break
                next_beams = llm_paths
                beams = next_beams

                # Add the top beams to the reasoning paths
                for _, reasoning_path in beams:
                    reasoning_paths.append(reasoning_path)

        path_dict = {str(reasoning_path)                     : reasoning_path for reasoning_path in reasoning_paths}
        corpus = list(path_dict.keys())
        sorted_paths = self.rank_model.top_k(
            query.question, corpus, self.top_k)
        query.st_tokens += self.rank_model.token_count(query.question)+sum(
            [self.rank_model.token_count(path) for path in corpus])
        reasoning_paths = [path_dict[path] for path in sorted_paths]
        return reasoning_paths

    def extract_paths(self, path_dict, user_input) -> List[ReasoningPath]:
        resp = self.llm_model.invoke(FILTER_PERSONA, user_input)
        answer = resp['response']
        filtered_paths = self.parse_answer(answer, path_dict)
        return filtered_paths, resp['input_tokens'], resp['output_tokens']

    def prepare_prompt(self, question: str, entities: List[str], corpus: List[str]) -> str:
        return filter_prompt.format(question=question, entities='; '.join(entities), corpus='\n'.join(corpus))

    def parse_answer(self, answer: str, path_dict: Dict[str, ReasoningPath]) -> List[ReasoningPath]:
        return [path_dict[path] for path in path_dict if path in answer]
