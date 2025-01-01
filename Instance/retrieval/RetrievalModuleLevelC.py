from retrieval.RetrievalModule import RetrievalModule
from utils.Query import Query
import igraph as ig
from utils.ReasoningPath import ReasoningPath
from typing import List, Tuple, Dict
from utils.PromptTemplate import filter_prompt, FILTER_PERSONA, RETRIEVAL_PERSONA, extract_prompt
from utils.LLM import LLM
from utils.SemanticModel import SemanticModel
from copy import deepcopy
import config


class RetrievalModuleLLM(RetrievalModule):
    def __init__(self, llm_model: LLM = None, rank_model: SemanticModel = None, hop: int = 4, top_k: int = -1, beam_width: int = 8, thre: int = 64):
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

    def process(self, query: Query) -> Query:
        return self.aprocess(query)

    async def aprocess(self, query: Query) -> Query:
        query.reasoning_paths = await self._get_reasoning_paths_async(query)
        return query

    async def _get_reasoning_paths_async(self, query: Query) -> List[ReasoningPath]:
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
                llm_paths = await self.extract_paths_async(
                    path_dict, user_input)
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

        path_dict = {str(reasoning_path): reasoning_path for reasoning_path in reasoning_paths}
        corpus = list(path_dict.keys())
        sorted_paths = self.rank_model.top_k(
            query.question, corpus, self.top_k)
        reasoning_paths = [path_dict[path] for path in sorted_paths]
        return reasoning_paths

    async def extract_paths_async(self, path_dict, user_input) -> List[ReasoningPath]:
        resp = await self.llm_model.ainvoke(FILTER_PERSONA, user_input)
        answer = resp[0]
        filtered_paths = self.parse_answer(answer, path_dict)
        return filtered_paths

    def prepare_prompt(self, question: str, entities: List[str], corpus: List[str]) -> str:
        return filter_prompt.format(question=question, entities='; '.join(entities), corpus='\n'.join(corpus))

    def parse_answer(self, answer: str, path_dict: Dict[str, ReasoningPath]) -> List[ReasoningPath]:
        return [path_dict[path] for path in path_dict if path in answer]


class RetrievalModuleLLMTriples(RetrievalModule):
    def __init__(self, llm_model: LLM = None, rank_model: SemanticModel = None, hop: int = 4, beam_width: int = 32, thre: int = 32):
        super().__init__()
        self.hop = hop
        self.beam_width = beam_width
        self.thre = thre
        self.llm_model = llm_model
        self.rank_model = rank_model

    async def aprocess(self, query: Query) -> Query:
        query.reasoning_paths = await self._get_reasoning_paths_async(
            query.subgraph, query.entities, query.question)
        return query

    def process(self, query: Query) -> Query:
        query.reasoning_paths = self.get_reasoning_paths(
            query.subgraph, query.entities,  query.question)
        return query

    def get_reasoning_paths(self, G: ig.Graph, entities: List[str], question: str) -> List[ReasoningPath]:
        reasoning_paths = []
        for entity in entities:
            try:
                start_vertex_id = G.vs.find(name=entity).index
            except:
                continue
            visited = {start_vertex_id: None}
            queue = [(start_vertex_id, None, 0)]

            while queue:
                next_queue = []

                for current_vertex_id, parent_edge, hop in queue:
                    if hop == self.hop or G.degree(current_vertex_id, mode="out") == 0:
                        path = []
                        while current_vertex_id is not None:
                            parent_info = visited[current_vertex_id]
                            if parent_info is not None:
                                parent_vertex_id, edge_id = parent_info
                                path.append(
                                    (parent_vertex_id, current_vertex_id, edge_id))
                                current_vertex_id = parent_vertex_id
                            else:
                                break
                        path.reverse()

                        reasoning_path = ReasoningPath(entity)
                        for source_vertex_id, target_vertex_id, edge_id in path:
                            edge = G.es[edge_id]
                            source_vertex = G.vs[edge.source]["name"]
                            target_vertex = G.vs[edge.target]["name"]
                            relation = edge["name"]
                            reasoning_path.add_triple(
                                (source_vertex, relation, target_vertex))
                        reasoning_paths.append(reasoning_path)
                    else:
                        visited_set = set(visited.keys())
                        for neighbor in G.vs[current_vertex_id].neighbors(mode="out"):
                            neighbor_id = neighbor.index
                            if neighbor_id not in visited_set:
                                edge_id = G.get_eid(
                                    current_vertex_id, neighbor_id)
                                next_queue.append((neighbor_id, edge_id))
                                visited[neighbor_id] = (
                                    current_vertex_id, edge_id)

                if len(next_queue) == 0:
                    break

                new_paths = self.extract_paths(G, next_queue, question, entity)
                queue = [(neighbor_id, edge_id, hop + 1)
                         for neighbor_id, edge_id in new_paths]

        return list(set(reasoning_paths))

    async def _get_reasoning_paths_async(self, G: ig.Graph, entities: List[str], question: str) -> List[ReasoningPath]:
        reasoning_paths = []
        for entity in entities:
            try:
                start_vertex_id = G.vs.find(name=entity).index
            except:
                continue
            visited = {start_vertex_id: None}
            queue = [(start_vertex_id, None, 0)]

            while queue:
                next_queue = []

                for current_vertex_id, parent_edge, hop in queue:
                    if hop == self.hop or G.degree(current_vertex_id, mode="out") == 0:
                        path = []
                        while current_vertex_id is not None:
                            parent_info = visited[current_vertex_id]
                            if parent_info is not None:
                                parent_vertex_id, edge_id = parent_info
                                path.append(
                                    (parent_vertex_id, current_vertex_id, edge_id))
                                current_vertex_id = parent_vertex_id
                            else:
                                break
                        path.reverse()

                        reasoning_path = ReasoningPath(entity)
                        for source_vertex_id, target_vertex_id, edge_id in path:
                            edge = G.es[edge_id]
                            source_vertex = G.vs[edge.source]["name"]
                            target_vertex = G.vs[edge.target]["name"]
                            relation = edge["name"]
                            reasoning_path.add_triple(
                                (source_vertex, relation, target_vertex))
                        reasoning_paths.append(reasoning_path)
                    else:
                        visited_set = set(visited.keys())
                        for neighbor in G.vs[current_vertex_id].neighbors(mode="out"):
                            neighbor_id = neighbor.index
                            if neighbor_id not in visited_set:
                                edge_id = G.get_eid(
                                    current_vertex_id, neighbor_id)
                                next_queue.append((neighbor_id, edge_id))
                                visited[neighbor_id] = (
                                    current_vertex_id, edge_id)

                if len(next_queue) == 0:
                    break

                new_paths = await self.extract_paths_async(G, next_queue, question, entity)
                queue = [(neighbor_id, edge_id, hop + 1)
                         for neighbor_id, edge_id in new_paths]

        return list(set(reasoning_paths))

    def extract_paths(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str, entity_name: str) -> List[Tuple[int, int]]:
        if len(paths) <= self.beam_width:
            return paths
        extract_prompt = self._construct_extract_prompt(
            G, paths, question, entity_name)
        results = self.llm_model.invoke(RETRIEVAL_PERSONA, extract_prompt)[0]
        filtered_paths = self._filter_paths_by_results(G, paths, results)
        return self._adjust_paths(G, question, filtered_paths, paths)

    async def extract_paths_async(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str, entity_name: str) -> List[Tuple[int, int]]:
        if len(paths) <= self.beam_width:
            return paths
        extract_prompt = self._construct_extract_prompt(
            G, paths, question, entity_name)
        resp = await self.llm_model.ainvoke(RETRIEVAL_PERSONA, extract_prompt)
        answer = resp[0]
        filtered_paths = self._filter_paths_by_results(G, paths, answer)
        return self._adjust_paths(G, question, filtered_paths, paths)

    def _adjust_paths(self, G: ig.Graph, question: str, new_paths: List[Tuple[int, int]], original_paths: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(new_paths) > self.beam_width:
            new_paths = self._rank_paths(
                G, new_paths, question, self.beam_width)
        if len(new_paths) < self.beam_width:
            for neighbor_id, edge_id in original_paths:
                if (neighbor_id, edge_id) not in new_paths:
                    new_paths.append((neighbor_id, edge_id))
                if len(new_paths) == self.beam_width:
                    break
        return new_paths

    def _filter_paths_by_results(self, G: ig.Graph, paths: List[Tuple[int, int]], results: List[str]) -> List[Tuple[int, int]]:
        return [(neighbor_id, edge_id) for neighbor_id, edge_id in paths if self._format_path(G, edge_id) in results]

    def _construct_extract_prompt(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str, entity_name: str) -> str:
        if len(paths) > self.thre:
            paths = self._rank_paths(G, paths, question, self.thre)

        total_paths = [self._format_path(G, edge_id) for _, edge_id in paths]
        return extract_prompt.format(beam_width=self.beam_width, question=question, entity_name=entity_name, total_paths="\n".join(total_paths))

    def _format_path(self, G: ig.Graph, edge_id: int) -> str:
        edge = G.es[edge_id]
        return f"{G.vs[edge.source]['name']}, {edge['name']}, {G.vs[edge.target]['name']}"

    def _rank_paths(self, G: ig.Graph, paths: List[Tuple[int, int]], question: str, top_k: int) -> List[Tuple[int, int]]:
        corpus = [self._format_path(G, edge_id) for _, edge_id in paths]
        top_k_paths = self.rank_model.top_k(question, corpus, top_k)
        ranked_paths = [
            (neighbor_id, edge_id)
            for path in top_k_paths
            for neighbor_id, edge_id in paths
            if self._format_path(G, edge_id) == path
        ]
        return ranked_paths
