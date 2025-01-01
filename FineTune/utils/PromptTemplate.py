
from config import few_shot, one_shot

############################################
# region Pre-Retrieval Module
PRERETIEVAL_PERSONA = """You are an expert at identifying and selecting the most relevant reasoning paths from a given corpus to answer specific questions. Your role involves analyzing the provided question and related entities, and then determining the most useful relation paths that can help address the question effectively. Your selection should be guided by the relevance, specificity, and accuracy of each path in relation to the entities mentioned in the question.
"""

PRERETIEVAL_ONE_SHOT = """
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

preretrieval_prompt = PRERETIEVAL_PERSONA + PRERETIEVAL_ONE_SHOT + """
Given the relations: 
{relations}

Generate some valid relation paths that can be helpful for answering the following question: {question}

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

extract_prompt = RETRIEVAL_PERSONA + RETRIEVAL_ONE_SHOT + """
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

filter_prompt = FILTER_PERSONA + FILTER_ONE_SHOT + """
Given the question '{question}' and entities [{entities}], identify the potentially useful reasoning paths from the following list:

{corpus}

# Output:

"""
# endregion

############################################
# region Reasonging Module
PERSONA = """You are an expert reasoner with a deep understanding of logical connections and relationships. Your task is to analyze the given reasoning paths and provide clear and accurate answers to the questions based on these paths."""

ONE_SHOT_EXAMPLES = """
# Example
## Input:
Based on the reasoning paths, please answer the given question and explain why.

Reasoning Paths:
Lou Seal -> sports.mascot.team -> San Francisco Giants -> sports.sports_championship_event.champion -> 2014 World Series

Question:
Lou Seal is the mascot for the team that last won the World Series when?

## Output:
Lou Seal is the mascot for the team that last won the World Series in 2014.

Explanation:
1. The reasoning path starts with "Lou Seal" and links it to "sports.mascot.team."
2. From there, it leads to "San Francisco Giants," indicating that Lou Seal is the mascot for the San Francisco Giants.
3. The path then continues to "sports.sports_championship_event.champion -> 2014 World Series," which tells us that the San Francisco Giants were the champions of the 2014 World Series.

Therefore, based on the provided reasoning paths, it can be concluded that the San Francisco Giants, represented by Lou Seal, last won the World Series in 2014.

# Query
Now answer the following question, you only need to output answer, nothing else! make sure your answer contains the entities of the above REASONING PATHS.

## Input:
"""

FEW_SHOT_EXAMPLES = """
# Examples
## Input:
Based on the reasoning paths, please answer the given question and explain why

Reasoning Paths:
Northern District -> location.administrative_division.first_level_division_of -> Israel -> government.form_of_government.countries -> Parliamentary system

Question:
What type of government is used in the country with Northern District?

## Output:
Parliamentary system

Explanation:
1. "Northern District" is a location within some country.
2. The reasoning path mentions "Northern District -> location.administrative_division.first_level_division_of -> Israel," indicating that the Northern District is part of Israel.
3. It further states "Israel -> government.form_of_government.countries," suggesting that Israel's form of government is being discussed.
4. The last part of the reasoning path indicates that Israel has a "Parliamentary system."

Therefore, based on the provided reasoning paths, it can be concluded that the type of government used in the country with the Northern District(Israel) is a Parliamentary system.

## Input:
Based on the reasoning paths, please answer the given question and explain why.

Reasoning Paths:
1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals -> sports.sports_team.arena_stadium -> Busch Stadium
1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals -> sports.sports_team.arena_stadium -> Roger Dean Stadium

Question:
Where is the home stadium of the team who won the 1946 World Series championship?

## Output:
Busch Stadium

Explanation:
1. 1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals -> sports.sports_team.arena_stadium -> Busch Stadium

The reasoning path leads us to the St. Louis Cardinals as the team that won the 1946 World Series, and Busch Stadium is the stadium associated with the St. Louis Cardinals. Therefore, Busch Stadium is the home stadium of the team that won the 1946 World Series championship.

## Input:
Based on the reasoning paths, please answer the given question and explain why.

Reasoning Paths:
Lou Seal -> sports.mascot.team -> San Francisco Giants -> sports.sports_championship_event.champion -> 2014 World Series

Question:
Lou Seal is the mascot for the team that last won the World Series when?

## Output:
Lou Seal is the mascot for the team that last won the World Series in 2014.

Explanation:
1. The reasoning path starts with "Lou Seal" and links it to "sports.mascot.team."
2. From there, it leads to "San Francisco Giants," indicating that Lou Seal is the mascot for the San Francisco Giants.
3. The path then continues to "sports.sports_championship_event.champion -> 2014 World Series," which tells us that the San Francisco Giants were the champions of the 2014 World Series.

Therefore, based on the provided reasoning paths, it can be concluded that the San Francisco Giants, represented by Lou Seal, last won the World Series in 2014.

# Query
Now answer the following question, you only need to output answer, nothing else! make sure your answer contains the entities of the above REASONING PATHS.

## Input:
"""

REASONING_TEMPLATE = """Based on the reasoning paths, please answer the given question.

Reasoning Paths:
{paths}

Question:
{question}

## Output:

"""
REASONING_INPUT = PERSONA + (FEW_SHOT_EXAMPLES if few_shot else "") + \
    (ONE_SHOT_EXAMPLES if one_shot else "") + REASONING_TEMPLATE
# endregion
