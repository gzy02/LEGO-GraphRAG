############################################
# region Pre-Retrieval Module

PRERETIEVAL_EDGE_PERSONA = """You are an expert in identifying and selecting the most relevant relational paths from a given corpus to answer specific questions. Your role involves analyzing the provided question and related entities, and then determining the most useful relational paths that can help address the question effectively. Your selection should be guided by the relevance, specificity, and accuracy of each path in relation to the entities mentioned in the question. Ensure that the paths you select are concise and directly related to the question.
"""

PRERETIEVAL_NODE_PERSONA = """You are an expert in identifying and selecting the most relevant entities from a given corpus to answer specific questions. Your role involves analyzing the provided question and related entities, and then determining the most useful entities that can help address the question effectively. Your selection should be guided by the relevance, specificity, and accuracy of each entity in relation to the entities mentioned in the question. Ensure that the entities you select are concise and directly related to the question.
"""

PRERETIEVAL_TRIPLE_PERSONA = """You are an expert in identifying and selecting the most relevant triples from a given corpus to answer specific questions. Your role involves analyzing the provided question and related entities, and then determining the most useful triples that can help address the question effectively. Your selection should be guided by the relevance, specificity, and accuracy of each triple in relation to the entities mentioned in the question. Ensure that the triples you select are concise and directly related to the question.
"""
PRERETIEVAL_EDGE_ONE_SHOT = """
# Example

## Input:
Given the relations: 
people.person.nationality
people.person.sibling_s
people.profession.specialization_of
people.ethnicity.included_in_group
people.person.place_of_birth
people.professional_field.professions_in_this_field
people.person.profession
people.person.parents
people.person.children
people.deceased_person.place_of_death
location.location.people_born_here
people.ethnicity.includes_groups
people.person.spouse_s
people.profession.part_of_professional_field
people.profession.corresponding_type
people.place_lived.person
people.person.education
people.person.employment_history
time.event.people_involved
people.sibling_relationship.sibling
people.person.ethnicity
people.place_lived.location
people.person.gender
people.marriage.spouse
people.person.languages
people.profession.people_with_this_profession
people.person.places_lived
people.profession.specializations

Generate some valid relation paths that can be helpful for answering the following question: what is the name of justin bieber brother

## Output:
people.person.children; people.person.parents
people.person.nationality; people.person.place_of_birth
people.person.gender; people.person.gender
people.sibling_relationship.sibling; people.sibling_relationship.sibling

# Query
Now answer the following question, you only need to output useful paths, nothing else!

## Input:
"""

preretrieval_edge_prompt = PRERETIEVAL_EDGE_ONE_SHOT + """
Given the relations: 
{relations}

Generate some valid relation paths that can be helpful for answering the following question: {question}

# Output:

"""
PRERETIEVAL_TRIPLE_ONE_SHOT = """
# Example

## Input:
Given the triples: 
Justin Bieber -> nationality -> Canadian
Justin Bieber -> sibling -> Jazmyn Bieber
Justin Bieber -> profession -> Singer
Justin Bieber -> place_of_birth -> London
Justin Bieber -> spouse -> Hailey Baldwin

Generate some valid triples that can be helpful for answering the following question: what is the name of justin bieber brother

## Output:
Justin Bieber -> sibling -> Jazmyn Bieber
Justin Bieber -> sibling -> Jaxon Bieber

# Query
Now answer the following question, you only need to output useful triples, nothing else!

## Input:
"""

PRERETIEVAL_NODE_ONE_SHOT = """
# Example

## Input:
Given the entities: 
Justin Bieber
Jazmyn Bieber
Jaxon Bieber
Hailey Baldwin
London
Canada

Generate some valid entities that can be helpful for answering the following question: what is the name of justin bieber brother

## Output:
Jazmyn Bieber
Jaxon Bieber

# Query
Now answer the following question, you only need to output useful entities, nothing else!

## Input:
"""

preretrieval_triples_prompt = PRERETIEVAL_TRIPLE_ONE_SHOT + """
Given the triples: 
{triples}

Generate some valid triples that can be helpful for answering the following question: {question}

# Output:

"""

preretrieval_node_prompt = PRERETIEVAL_NODE_ONE_SHOT + """
Given the entities: 
{entities}

Generate some valid entities that can be helpful for answering the following question: {question}

# Output:

"""
# endregion

############################################
# region Retrieval Module
"""
score_prompt = Please score the relations (separated by semicolon) that contribute to the question on a scale from 0 to 1 (the sum of the scores of all relations is 1).
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: language.human_language.main_country
language.human_language.countries_spoken_in
base.rosetta.languoid.parent
kg.object_profile.prominent_type
Score: 0.4, 0.3, 0.2, 0.0
language.human_language.main_country is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
language.human_language.countries_spoken_in is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
base.rosetta.languoid.parent is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.
kg.object_profile.prominent_type is not relevant and contributes nothing to the question.

Q: {}
Topic Entity: {}
Relations:
"""

RETRIEVAL_PERSONA = """You are an expert at retrieving the most relevant paths from a given input to answer specific questions. Your task is to retrieve {beam_width} paths(separated by semicolon) that contribute to the question.
"""

RETRIEVAL_ONE_SHOT = """
# Example

## Input:
Question: 
Rift Valley Province is located in a nation that uses which form of currency?

Topic Entity: 
Rift Valley Currency

Paths: 
Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling

## Output:
Kenya, location.country.currency_used, Kenyan shilling
Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity

# Query
Now answer the following question, you only need to output useful paths, nothing else!

## Input:
"""

extract_prompt = RETRIEVAL_ONE_SHOT + """
Question: 
{question}

Topic Entity: 
{entity_name}

Paths: 
{total_paths}

## Output:

"""
# endregion

############################################
# region Post-Retrieval Module
FILTER_PERSONA = """You are an expert filterer with a deep understanding of relevant information. Your task is to analyze the given question and entities, and identify the potentially useful reasoning paths from the provided corpus. For each question, you will select the most relevant paths that can help answer the question effectively. Your selection should be based on the entities mentioned in the question and their relationships with other entities in the corpus. Make sure to consider the relevance, specificity, and accuracy of each path in relation to the given question and entities. 

"""

FILTER_ONE_SHOT = """
# Example

## Input:
Given the question 'What state is home to the university that is represented in sports by George Washington Colonials men's basketball?' and entities [George Washington Colonials men's basketball], identify the potentially useful reasoning paths from the following list:

George Washington Colonials men's basketball -> sports.school_sports_team.school -> George Washington University
George Washington Colonials men's basketball -> sports.school_sports_team.school -> George Washington University -> education.educational_institution.faculty -> m.0kdpyxr
George Washington Colonials men's basketball -> sports.sports_team.sport -> Basketball
George Washington Colonials men's basketball -> sports.sports_team.arena_stadium -> Charles E. Smith Center -> location.location.containedby -> Washington, D.C.
George Washington Colonials men's basketball -> sports.school_sports_team.school -> George Washington University -> education.educational_institution.faculty -> m.0k9wvjz

## Output:
George Washington Colonials men's basketball -> sports.sports_team.arena_stadium -> Charles E. Smith Center -> location.location.containedby -> Washington, D.C.
George Washington Colonials men's basketball -> sports.school_sports_team.school -> George Washington University

# Query
Now answer the following question, you only need to output useful paths, nothing else!

## Input:
"""

filter_prompt = FILTER_ONE_SHOT + """
Given the question '{question}' and entities [{entities}], identify the potentially useful reasoning paths from the following list:

{corpus}

# Output:

"""
# endregion

############################################
# region Reasonging Module
PERSONA = """You are an expert reasoner with a deep understanding of logical connections and relationships. Your task is to analyze the given reasoning paths and provide clear and accurate answers to the questions based on these paths."""

REASONING_TEMPLATE = """Based on the reasoning paths, please answer the given question.

Reasoning Paths:
{paths}

Question:
{question}

## Output:

"""
# endregion
