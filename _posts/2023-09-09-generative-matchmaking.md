---
layout: post
title: Generative Matchmaking
categories: machine-learning artificial-intelligence
author: Daniel Deychakiwsky
meta: Generative Matchmaking
mathjax: true
permalink: /:title
---

This post details an experimental simulation of an AI-driven 
matchmaking system that combines retrieval-augmented 
generation with semantic search. It aims to 
generate user-to-user compatibility recommendations 
for a dating service by leveraging the capabilities 
of OpenAI's GPT-4, OpenAI's DALL·E-2, and Chroma, 
an AI-native open-source vector database.


* TOC
{:toc}

# Elevator Pitch

Popular dating services have converged to an experience
where users must swipe, match, chat, and plan to meet. 
For many users, this results in too much time spent swiping 
followed by repetitive small talk.

> I'm looking for a connection where I can simply ask, 
> "Would you like to meet up for drinks later?" 
> I wish online dating could solely serve the purpose 
> of arranging dates, eliminating the need for aimless small talk. 
> Repeating the same text-based conversations that lead nowhere gets tiresome. 
> Meeting in person right away would allow us to gauge our compatibility and save time. 
> -- <cite>I'm a 27-year-old female.</cite>

Inspired by the emergence of human-like behavior of modern AI, 
we present an innovative experience where each user is paired 
with a personalized AI matchmaking assistant where 
the user may occasionally reinforce their assistant with 
feedback for alignment to any changes in preference. These assistants 
are designed to engage with one another, with the goal of 
creating harmonious matches and carefully curated date plans. 
We posit that this approach will eliminate tedious small talk 
and minimize swipe time.

🧑 → 🤖💬 → ❤️ ← 💬🤖 ← 🧑

# Simulation

## Generating Synthetic User Profiles

We sequentially generated 250 synthetic dating user profiles using OpenAI's chat 
completion endpoint with the following configuration settings: `model="gpt-4-0613"`, 
`max_tokens=5000`, and `temperature=1.05`. Initially, our attempt to generate numerous dating 
profiles in a single inference pass, allowing the model to consider previously generated 
profiles for uniqueness and diversity, was hindered by output token limit constraints. 
To address this, we adopted an alternative approach: we chatted with the model
to generate a collection of names and subsequently instructed model sample from this pool.

Our starting point was the default system prompt: `"You are a helpful assistant."`. 
Within this conversation, we provided the instruction: `"Generate 10 unique first and 
last names, ensuring a balanced diversity."` Following this, we issued a follow-up directive: 
`"Select a name randomly and create a dating profile for the chosen name."`. It's worth noting 
that the sampling process may not have been uniform over the names due to the positional 
biases inherent in language models. Nevertheless, this strategy proved effective.

For the generation of the dating profile, we utilized OpenAI's function calling API, which 
produced a structured JSON response based on our dating profile [schema] where 
we hardcoded the dating location to Los Angeles, California for all users. Note that
although our schema is basic, it can be updated to include other pieces of information
about a user or the user's preferences.

> Developers can now describe functions to gpt-4-0613 and gpt-3.5-turbo-0613, 
> and have the model intelligently choose to output a JSON object containing 
> arguments to call those functions. 
> This is a new way to more reliably connect GPT's capabilities with external tools and APIs.
> These models have been fine-tuned to both detect when a function needs to be 
> called (depending on the user’s input) and to respond with JSON that 
> adheres to the function signature. Function calling allows developers 
> to more reliably get structured data back from the model. -- <cite>OpenAI docs.</cite>

Once the dating profile was generated we instructed to
`"Summarize the user's dating profile. 
Include all fields other than partner_preferences. Output a concise paragraph."`
and finally instructed to 
`"Summarize the user's dating partner_preferences. 
Include partner_preferences fields only and nothing else. Output a concise paragraph."`
We did not include the user's name in the summarization instructions to maintain generality
across users.
An example synthetic dating profile follows (output [profiles] directory).

```json
{
    "name": "Mia Wong",
    "age": 29,
    "height": "5'6\"",
    "school": "University of California, Berkeley",
    "job_industry": "Technology",
    "job_title": "Software Engineer",
    "hometown_location": "San Francisco, California",
    "dating_location": "Los Angeles, California",
    "languages_spoken": [
        "English",
        "Mandarin"
    ],
    "values": [
        "Responsibility",
        "Family",
        "Ambition"
    ],
    "interests": [
        "Cooking",
        "Hiking",
        "Traveling"
    ],
    "education_level": "Graduate",
    "religious_beliefs": "Atheist",
    "politics": "Liberal",
    "dating_intentions": "Long term, open to short",
    "relationship_type": "Monogamy",
    "gender": "Woman",
    "pronouns": "She/Her/Hers",
    "sexuality": "Straight",
    "ethnicity": "East Asian",
    "has_children": false,
    "want_children": true,
    "pets": [
        "Cat"
    ],
    "zodiac_sign": "Sagittarius",
    "mbti_personality_type": "INTJ",
    "drinking": "Sometimes",
    "smoking": "No",
    "marijuana": "No",
    "drugs": "No",
    "exercise": "Active",
    "partner_preferences": {
        "minimum_age": 28,
        "maximum_age": 35,
        "minimum_height": "5'6\"",
        "maximum_height": "6'2\"",
        "has_children": false,
        "want_children": true,
        "sexuality": "Straight",
        "drinking": "Sometimes",
        "smoking": "No",
        "marijuana": "No",
        "drugs": "No",
        "exercise": "Active",
        "gender": "Man",
        "dating_intentions": "Long term, open to short",
        "relationship_type": "Monogamy",
        "ethnicities": [
            "Black African Descent",
            "East Asian",
            "Middle Eastern",
            "Native American",
            "Pacific Islander",
            "South Asian",
            "Southeast Asian",
            "White Caucasian"
        ],
        "politics": [
            "Liberal",
            "Not political"
        ],
        "job_industry": [
            "Technology",
            "Finance",
            "Consulting"
        ],
        "languages_spoken": [
            "English"
        ],
        "values": [
            "Responsibility",
            "Family",
            "Ambition"
        ],
        "interests": [
            "Cooking",
            "Hiking",
            "Traveling"
        ],
        "education_level": [
            "Undergraduate",
            "Graduate"
        ]
    },
    "profile_summary": "see below",
    "preferences_summary": "see below",
    "user_id": "fce07a22-133b-46b9-b187-ca1ae5c0b70e"
}
```

Profile Summary

```text
The user is a 29-year-old woman who identifies as straight. She uses she/her pronouns and is of 
East Asian ethnicity. She graduated from the University of California, Berkeley and now works 
as a Software Engineer in the technology industry. Originally from San Francisco, California, 
she is now looking to date in Los Angeles, California. She is an atheistic liberal who values 
responsibility, family, and ambition. She enjoys cooking, hiking, and traveling. Despite her 
busy schedule, she maintains an active lifestyle. She speaks both English and Mandarin fluently. 
She owns a cat and wishes to have children although she doesn't have any yet. She is a 
Sagittarius with an INTJ Myers-Briggs Type Indicator. An occasional drinker, she 
doesn't smoke or use marijuana or other drugs. Her intention is to be in a monogamous 
relationship, open to both long term commitments and causal dating.
```

Preferences Summary

```text
The user is a woman who is interested in dating a man who is between the ages of 28 and 35, 
and between 5'6" and 6'2" in height. Her preferred partner would be someone who does not 
have children but is open to having some in the future. The partner should be straight, 
drinks occasionally, doesn't smoke or involve in marijuana or drugs. She prefers a man who 
is active and practices monogamy. Ethnicity- wise, her interests include a range of ethnic 
groups, from Black African Descent, East Asian, Middle Eastern, Native American, Pacific 
Islander, South Asian, Southeast Asian to White Caucasian. She prefers partners who identify 
as Liberal or are not political. The ideal match for her would work in the Technology, Finance, 
or Consulting industries. The primary language should be English. In terms of values, she looks 
for Responsibility, Family, Ambition, and shares interests in cooking, hiking and traveling. 
Potential partners should have at least an undergraduate level of education
```

## Generating Synthetic User Profile Images

We generated a user profile picture for every user by invoking OpenAI's DALL·E-2 
text-to-image model. We constructed the prompt by interpolating the following
template with several fields from the user dating profile. Originally, we also used
first and last name as two of the fields but OpenAI, rightly-so, flagged it as unsafe,
so we removed it. Note that the prompt for this model must stay under a shorter
character limit.

```python
prompt: str = (
    f"Dating profile picture of a "
    f"{user_profile.height} {user_profile.ethnicity} {user_profile.age} "
    f"year old {user_profile.gender} ({user_profile.pronouns}) "
    f"with an {user_profile.exercise} physique who works as a "
    f"{user_profile.job_industry} professional "
    f"that values {' and '.join(user_profile.values[:2])} "
    f"who enjoys {' and '.join(user_profile.interests[:2])} who "
    f"identifies as {user_profile.religious_beliefs}."
)
```

All the generated profile pictures can be inspected in the output [profiles] directory.
Here are 28 examples, the overall quality varies.

![img_0]
![img_1]
![img_2]
![img_3]
![img_4]
![img_5]
![img_6]
![img_7]
![img_8]
![img_9]
![img_10]
![img_11]
![img_12]
![img_13]
![img_14]
![img_15]
![img_16]
![img_17]
![img_18]
![img_19]
![img_20]
![img_21]
![img_22]
![img_23]
![img_24]
![img_25]
![img_26]
![img_27]

## Matchmaking

### Recommender System

Mainstream dating apps with an abundance of users and data 
train hybrid recommender systems that model content-based
distributions and collaborative filtering effects based on a robust set of features
and explicit / implicit user interactions, e.g., swiping and time-on-page.
Researchers have shown that LLMs can be used as zero-shot shot recommenders but,
through ablation studies, that they're primarily content-based.
Since our study lacked human feedback, we framed matchmaking as a LLM zero-shot
content-based recommender powered by vector search and a bit of graph theory.

#### Chroma Vector Database

[Chroma] is an AI-native open-source vector database.
We spun up a local and persistent version. By default, 
Chroma uses the Sentence Transformers `all-MiniLM-L6-v2` 
model to create embeddings. This embedding model can create sentence and 
document embeddings that can be used for a wide variety of tasks. 
This embedding function runs locally, downloading and caching the model files.
We loaded every user's **profile summary**, as a document, 
into a Chroma _collection_ and tagged each entry with two pieces
of metadata, the user's gender and sexuality.

##### Retrieval

Recall that we created two summaries for each user profile.
We summarized, _mutually exclusively_, each user's profile and their partner preferences.
To compute a set of candidates for a given user, we query the Chroma collection
with the user's **partner preference summary** and specify a filter based 
on the user's partner preference gender and sexuality. Under the hood, Chroma
executes a similarity / distance lookup on the embedded query text and returns the
closest `n_results` with an associated scalar distance value. Choosing a value
for `n_results` is a hyperparameter for this algorithm which can be tuned to 
the use case. We set it to 25. 

##### Ranking

Researchers have found that LLMs struggle with non-trivial 
ranking tasks which may be connected to their inherent positional 
bias. While bootstrapping the ranking task is reported to aid in 
reducing variance and stabilizing rankings, we chose to implement 
a graph inspired heuristic leveraging vector search.

The retrieved set of the closest user profiles
to a given user's partner preferences can naively be surfaced
as prospects for human feedback. However, borrowing from graph theory,
if users are nodes, this represents a set of unidirectional edges connecting the query user
to every retrieved candidate user. In our case, we define compatibility as a bidirectional
edge (or two unique directed edges connecting two nodes) indicating that a given retrieved
user would also link back to the query user. We implemented 
the retrieval mechanism for every candidate user from the query user's resultant set
and inspect the candidate resultant sets for the query user, indicating a
bidirectional connection. The candidates that point back to query user are 
ranked higher than those that do not. This can be thought of as one level
of breadth-first search. To illustrate, user 3 and 4 are the only users 
considered to be compatible in the diagram that follows.

![graph]


## Repository

The implementation is available in a public 
Github [repository](https://github.com/daniel-deychakiwsky/generative-matchmaking).

# Literature Review

* [https://arxiv.org/pdf/2203.02155.pdf](https://arxiv.org/pdf/2203.02155.pdf) (OpenAI)
* [https://arxiv.org/pdf/2304.03442.pdf](https://arxiv.org/pdf/2304.03442.pdf) (Stanford)
* [https://arxiv.org/pdf/2306.02707.pdf](https://arxiv.org/pdf/2306.02707.pdf) (Microsoft)
* [https://arxiv.org/pdf/2305.10601.pdf](https://arxiv.org/pdf/2305.10601.pdf) (Google)
* [https://arxiv.org/pdf/2201.11903.pdf](https://arxiv.org/pdf/2201.11903.pdf) (Google)
* [https://arxiv.org/pdf/2308.10053.pdf](https://arxiv.org/pdf/2308.10053.pdf) (Netflix)
* [https://arxiv.org/pdf/2305.08845.pdf](https://arxiv.org/pdf/2305.08845.pdf) (Netflix)
* [https://arxiv.org/pdf/2305.07622.pdf](https://arxiv.org/pdf/2305.07622.pdf) (Amazon)
* [https://arxiv.org/pdf/2005.11401v4.pdf](https://arxiv.org/pdf/2005.11401v4.pdf) (Meta)
* [https://arxiv.org/pdf/2303.11366.pdf](https://arxiv.org/pdf/2303.11366.pdf) (MIT)

[schema]: https://github.com/daniel-deychakiwsky/generative-matchmaking/blob/master/src/generative_matchmaking/llm/oai_function_schemas.py
[profiles]: https://github.com/daniel-deychakiwsky/generative-matchmaking/tree/master/profiles
[chroma]: https://www.trychroma.com/

[img_0]: /assets/images/generative_matchmaking/profile_0.png
[img_1]: /assets/images/generative_matchmaking/profile_1.png
[img_2]: /assets/images/generative_matchmaking/profile_2.png
[img_3]: /assets/images/generative_matchmaking/profile_3.png
[img_4]: /assets/images/generative_matchmaking/profile_4.png
[img_5]: /assets/images/generative_matchmaking/profile_5.png
[img_6]: /assets/images/generative_matchmaking/profile_6.png
[img_7]: /assets/images/generative_matchmaking/profile_7.png
[img_8]: /assets/images/generative_matchmaking/profile_8.png
[img_9]: /assets/images/generative_matchmaking/profile_9.png
[img_10]: /assets/images/generative_matchmaking/profile_10.png
[img_11]: /assets/images/generative_matchmaking/profile_11.png
[img_12]: /assets/images/generative_matchmaking/profile_12.png
[img_13]: /assets/images/generative_matchmaking/profile_13.png
[img_14]: /assets/images/generative_matchmaking/profile_14.png
[img_15]: /assets/images/generative_matchmaking/profile_15.png
[img_16]: /assets/images/generative_matchmaking/profile_16.png
[img_17]: /assets/images/generative_matchmaking/profile_17.png
[img_18]: /assets/images/generative_matchmaking/profile_18.png
[img_19]: /assets/images/generative_matchmaking/profile_19.png
[img_20]: /assets/images/generative_matchmaking/profile_20.png
[img_21]: /assets/images/generative_matchmaking/profile_21.png
[img_22]: /assets/images/generative_matchmaking/profile_22.png
[img_23]: /assets/images/generative_matchmaking/profile_23.png
[img_24]: /assets/images/generative_matchmaking/profile_24.png
[img_25]: /assets/images/generative_matchmaking/profile_25.png
[img_26]: /assets/images/generative_matchmaking/profile_26.png
[img_27]: /assets/images/generative_matchmaking/profile_27.png
[graph]: /assets/images/generative_matchmaking/graph.png
