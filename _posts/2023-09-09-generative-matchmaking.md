---
layout: post
title: Generative Matchmaking
categories: machine-learning artificial-intelligence
author: Daniel Deychakiwsky
meta: Generative Matchmaking
mathjax: true
permalink: /:title
---

This post details an AI-based matchmaking 
simulation for dating, leveraging vector-based 
semantic search for user compatibility recommendations 
and retrieval-augmented generation (RAG) for co-personalized 
first-date ideation using OpenAI's GPT-4, OpenAI's 
DALL·E-2, and Chroma, an AI-native open-source 
vector database. My brother, Dimitri Deychakiwsky,
and I put this together over several weekends.

* TOC
{:toc}

# Elevator Pitch

## Dating

Popular dating services have converged to an experience
where users must swipe, match, chat, and plan to meet. 
For many users, this results in too much time spent
repeatedly swiping, small talking, and planning.  

> I'm looking for a connection where I can simply ask, 
> "Would you like to meet up for drinks later?" 
> I wish online dating could solely serve the purpose 
> of arranging dates, eliminating the need for aimless small talk. 
> Repeating the same text-based conversations that lead nowhere gets tiresome. 
> Meeting in person right away would allow us to gauge our compatibility and save time. 
> -- <cite>I'm a 27-year-old female.</cite>

Inspired by the emergence of human-like behavior in modern AI, 
our vision is to create an experience where each user is paired 
with a personalized AI matchmaking assistant. This assistant 
can adapt to changes in user preferences with the help of human 
feedback and works behind the scenes to connect with other user 
assistants, creating harmonious matches. The goal is to reduce 
the time users spend searching and increase the time they spend 
meeting people. We believe that this approach can minimize swipe 
time, eliminate small talk, and assist in planning. The remainder
of this post provides details on a simulated subset of functionality 
that integrates into the overall effort.

🧑 → 🤖💬 → ❤️ ← 💬🤖 ← 🧑

## Framework

We focused on dating as an application
of matchmaking. However, we argue that our 
framework generalizes to a broad spectrum
of use cases, e.g., jobs, philanthropy, customer churn,
e-commerce, and overall business intelligence.

# Simulation

The simulated implementation is available in a public GitHub 
[repository](https://github.com/daniel-deychakiwsky/generative-matchmaking). 
As a proof-of-concept, we generated synthetic dating user profiles 
with pictures and produced matches for each of them, 
all of which are stored in the [profiles] directory. 
We also generated assisted date plans for an example 
pair of matched users.

## Generating Synthetic Users

### Profiles

We sequentially generated 250 synthetic dating user profiles using OpenAI's chat 
completion endpoint with the following configuration settings.

```python
{
  "model": "gpt-4-0613",
  "max_tokens": 5000,
  "temperature": 1.05
}
```

Initially, our attempt to generate numerous dating 
profiles in a single inference pass, allowing the model to consider previously generated 
profiles for uniqueness and diversity, was hindered by output token limit constraints. 
To address this, we adopted an alternative approach: we chatted with the model
to generate a collection of names and subsequently instructed the model to sample from this pool
as a seed for profile generation.

```text
[System Message]: You are a helpful assistant.

[User Message]: Generate 10 unique first and last names, 
ensuring a balanced diversity.

[Assistant Message]: ...

[User Message]: Select a name randomly and create 
a dating profile for the chosen name.

[Assistant Message]: ...
```

It's worth noting that the sampling process suffers from positional biases inherent 
in language models. Nevertheless, we observed that this strategy, at the very least, 
improved generation of a diverse set of names across all users.

For the generation of the dating profile, we utilized OpenAI's function calling API, which 
produced a structured JSON response based on our dating profile [schema] where 
we hardcoded the dating location to Los Angeles, California for all users. Note that
although our schema is basic, it can be extended to include other pieces of static or dynamic
information along with the concept of "dealbreakers" which effectively translates
to a hard filter on specified partner preference attributes.

> Developers can now describe functions to gpt-4-0613 and gpt-3.5-turbo-0613, 
> and have the model intelligently choose to output a JSON object containing 
> arguments to call those functions. 
> This is a new way to more reliably connect GPT's capabilities with external tools and APIs.
> These models have been fine-tuned to both detect when a function needs to be 
> called (depending on the user’s input) and to respond with JSON that 
> adheres to the function signature. Function calling allows developers 
> to more reliably get structured data back from the model. -- <cite>OpenAI docs.</cite>

After generating the dating profile, we instructed the model 
to exclusively summarize both the user's dating profile and 
their partner preferences.

```text
...
[User Message]: Summarize the user's dating profile. 
Include all fields other than partner_preferences. 
Output a concise paragraph.

[Assistant Message]: ...

[User Message]: Summarize the user's dating partner_preferences. 
Include partner_preferences fields only and nothing else. 
Output a concise paragraph.

[Assistant Message]: ...
```

We did not include the user's name in the summarization instruction context 
to maintain generality across users. An example synthetic dating profile follows.

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

**Profile Summary**

```text
The user is a 29-year-old woman who identifies as straight. 
She uses she/her pronouns and is of 
East Asian ethnicity. She graduated from the University 
of California, Berkeley and now works 
as a Software Engineer in the technology industry. 
Originally from San Francisco, California, 
she is now looking to date in Los Angeles, California. 
She is an atheistic liberal who values 
responsibility, family, and ambition. 
She enjoys cooking, hiking, and traveling. Despite her 
busy schedule, she maintains an active lifestyle. 
She speaks both English and Mandarin fluently. 
She owns a cat and wishes to have children although 
she doesn't have any yet. She is a 
Sagittarius with an INTJ Myers-Briggs Type Indicator. 
An occasional drinker, she 
doesn't smoke or use marijuana or other drugs. 
Her intention is to be in a monogamous 
relationship, open to both long term commitments 
and causal dating.
```

**Partner Preferences Summary**

```text
The user is interested in dating a 
man who is between the ages of 28 and 35, 
and between 5'6" and 6'2" in height. Her preferred 
partner would be someone who does not 
have children but is open to having some in the future. 
The partner should be straight, 
drinks occasionally, doesn't smoke or involve in marijuana 
or drugs. She prefers a man who 
is active and practices monogamy. Ethnicity- wise, her 
interests include a range of ethnic 
groups, from Black African Descent, East Asian, 
Middle Eastern, Native American, Pacific 
Islander, South Asian, Southeast Asian to White Caucasian. 
She prefers partners who identify 
as Liberal or are not political. The ideal match for her 
would work in the Technology, Finance, 
or Consulting industries. The primary language should be 
English. In terms of values, she looks 
for Responsibility, Family, Ambition, and shares interests 
in cooking, hiking and traveling. 
Potential partners should have at least an undergraduate 
level of education.
```

### Images

We generated a user profile picture for every user by invoking OpenAI's DALL·E-2 
text-to-image model. We constructed the prompt by interpolating the following
template with several fields from the user dating profile schema. Originally, we included
the first and last name but OpenAI, rightly-so, flagged it as unsafe,
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

Mainstream dating apps train hybrid recommender systems that model content-based
and collaborative filtering effects based on a robust set of features
including explicit / implicit user interactions, e.g., swiping, time-on-page, etc.
Modern recommender systems are implemented as a conglomerate of
individual components that often implement retrieval and ranking as separate tasks.

Researchers have shown that LLMs generalize as zero-shot recommenders but,
through ablation studies, found that they're primarily content-based
and that they struggle with non-trivial ranking tasks, 
likely connected to their inherent positional bias.

Since our study lacked human feedback, we experimented with
a prompt-based zero-shot retrieval mechanism but realized this approach
would not scale effectively due to context-window limits. Instead,
we implemented retrieval as semantic vector search. To rank the retrieved 
output, we experimented with prompt engineering the ranking task but 
the results clearly suffered from artifacts of positional bias. 
Although bootstrapping the ranking task alleviated some concern 
by reducing variance and thereby stabilizing rankings, 
the results did not suffice, so we chose to implement 
a graph inspired ranking heuristic.

#### Chroma Vector Database

We spun up a local and persistent version of [Chroma], 
an AI-native open-source vector database. By default, 
Chroma uses the Sentence Transformers `all-MiniLM-L6-v2` 
model to create embeddings. This embedding model can create sentence and 
document embeddings that can be used for a wide variety of tasks. 
This embedding function runs locally, downloading and caching the model files.
We loaded every user's **profile summary**, as a document, 
into a _collection_ and tagged each entry with the user's gender
and sexuality as _metadata_ for which we manually enforce as
"dealbreakers" or hard filters.

Chroma's default embedding model maps documents 
to $\mathbb{R}^{384}$ (384 dimensional space). We
visualized all 250 user dating profile summaries
by projecting them down to $\mathbb{R}^{3}$ with the
T-SNE algorithm using Tensorflow's Embedding [Projector].
The result follows.

![embeddings]

We assume that the two global clusters represent gender.
An interesting application of these natural language
embeddings as a product feature is that users could issue 
queries to find partners by simply typing, e.g., 
"I'm looking for a ... that plays ultimate frisbee and ... ".

##### Retrieval

Recall that we created two summaries for each user that were _mutually exclusive_.
To compute an initial set of candidates for a given query user, we searched the Chroma _collection_
with their **partner preferences summary** and specified a hard _metadata_ filter based on 
their partner preference gender and sexuality. Under the hood, Chroma
executes a similarity / distance lookup on the embedded query text and returns the
closest `n_results` or neighbors sorted by ascending distance. Choosing a distance metric and an
appropriate value for the number of neighbors are hyperparameters which can be tuned 
to the use case. We configured Chroma to use _cosine_ distance and to return the 25 closest 
neighbors.

##### Ranking

The retrieved set of candidate user profiles for a given user can 
naively be surfaced as matches for human feedback but some of those
candidates may be better matches than others. Borrowing from graph theory, 
we built a graph by treating users as nodes and adding retrieval-distance 
weighted edges for every user-candidate pair of nodes connecting
a query user to their retrieved candidate set.

We defined compatibility as a bidirectional connection between two users where a retrieved 
candidate's retrieved set of candidates includes the query user. 
These candidates, that point back to query user, are ranked higher than those that don't. 
This can be thought of as one level of breadth-first search.
To illustrate, user $M$ is compatible with the ordered set of users 
$\\{C, B, P\\}$ in the diagram that follows.

![graph]

### Visualizing Matches

To visualize matches, we hacked together a user interface with directions
on how to run it outlined in the repo's [readme]. A screenshot of Theodore Heath's matches,
ranked from left-to-right, follows. The presence of an asterisk by the name of a 
match indicates a bidirectional connection (compatibility) while the others
are unidirectional matches with one directed edge pointing from Theodore to the 
candidate match.

![screenshot]

The most compatible match recommended for Theodore Heath is Olivia Windsor.
Theodore and Olivia being marked as a compatible match means that
Theodore's **partner preferences summary** is semantically similar to Olivia's
**profile summary** and that Olivia's **partner preferences summary** is semantically
similar to Theodore's **profile summary** within the embedding geometry.
In other words, Theodore is looking for someone like Olivia and
Olivia is looking for someone like Theodore based on the information
available to us. We can inspect Theodore and Olivia's profile and 
partner preferences summaries to qualitatively evaluate the match. 
Our matching algorithm can be thought of as a soft matcher
with a hard filter on preferred sexuality and gender.

**Theodore's Profile Summary**

```text
This user is a 29-year-old Software Engineer in the 
Technology industry based in Los Angeles, 
California, but originally from San Francisco. 
He stands 6 feet tall and is a graduate from 
the California State University. He identifies as 
a straight white Caucasian man who uses 
the pronouns He/Him/His, with agnostic religious 
beliefs, and a moderate political stance. 
He values honesty, humor, and intelligence and 
enjoys outdoor activities, video games, and 
reading in his spare time. This user is a Pisces 
and his MBTI personality type is INTJ. 
Despite being occasionally up for a drink, he does 
not smoke or use drugs, including marijuana. 
He leads an active lifestyle and owns a dog. 
His desires for the future include having children, 
as he currently doesn't have any. He's also open to 
a long-term, monogamous relationship. 
His primary languages are English and Spanish.
```

**Theodore's Partner Preferences Summary**

```text
The user is seeking a female partner aged 24 to 31 
years who falls between the heights of 5'5" 
and 6'0". She doesn't have children but must be 
open to having them in the future. An ideal 
match would be a straight woman who occasionally 
drinks but doesn't smoke or use marijuana or 
other drugs. Light to moderate exercise habits are 
preferred. Ethnicity-wise, he is open to 
dating women of both White Caucasian and Hispanic/Latino 
backgrounds. In terms of political 
inclination, he prefers a woman with moderate views. 
Professionally, he prefers women working 
in the Education, Technology, or Healthcare industry. 
Fluency in English is a must, and he 
values Honesty, Kindness, and Intelligence. 
Shared interests might include outdoor activities, 
music, or reading. Education is important too, 
with a preference for woman who have at least 
an undergraduate degree or higher. He is looking 
for a long-term relationship that is monogamous.
```

**Olivia's Profile Summary**

```text
This user is a 28 year-old woman standing at 5'6". 
She is a UX Designer working in the tech 
industry and is a graduate of Stanford University. 
She identifies as straight and is interested 
in a monogamous, long-term relationship. 
Originally from New York, New York, she is currently 
dating in Los Angeles, California. She fluently 
speaks English and Spanish, and identifies her 
ethnicity as white caucasian. Her core values 
are creativity, independence, and adventure. 
Interests include photography, cooking, and travel. 
She identifies as agnostic and liberal. 
Although she doesn't have any children yet, she 
wants them in the future. She is an active 
drinker but doesn't smoke, use marijuana, or drugs. 
When it comes to exercise, she's quite active. 
She owns a cat and is a Gemini with an 
ENFP personality type.
```

**Olivia's Partner Preferences Summary**

```text
The user is looking for a man who is between the ages 
of 28 and 35 and stands between 5'8" 
and 6'2". He shouldn't have children currently but 
should want to have them in the future. 
Ideally, he would be straight, an active individual, 
and a non-smoker. He's preferred to be 
someone who has no problems with drinking but doesn't 
use marijuana or any other drugs. In 
terms of ethnicity, he can be of any listed ethnicity. 
Politically, a liberal or moderate man 
would be suitable. The ideal man would work in either 
Tech, Finance, or Health sectors, and 
English should be among his languages. He should value 
kindness, honesty, and be committed. 
His interests should include hiking, photography, and 
traveling. The user prefers someone who 
has at least an undergraduate level of education.
```

## Planning

Once the matches were made, we generated personalized first-date
activities and talking points by applying 
a technique known as retrieval-augmented generation (RAG)
where we instructed a LLM to assume a role of an expert creative date planner or
conversationalist conditioning the generative task on user information.
This is a form of in-context learning where the LLM
is fed information retrieved from an external storage system
to aid in performing a task. Analogous to how the human brain stores information
in long-term and short-term memory, the external system is the AI's long-term memory
whereas the LLM's context-window is its short-term memory.
In the sections that follow, we report results 
for Theodore and Olivia, our example match from the previous section.
The AI system plans based on their information, e.g., 
who they are, what they like, what they want, what city they're meeting in, etc.
We used a basic Chain of Thought (CoT) prompting strategy that incites the model
to break down its reasoning process. There are more advanced prompting 
strategies that we've left to future spikes, e.g., Tree of Thought (ToT).
We used the following configuration for both tasks.

```python
{
  "model": "gpt-4-0613",
  "max_tokens": 5000,
  "temperature": 0.0
}
```

### Activities

**System Message**

```text
You are an expert date planner.
```

**User Message**

```text
Brainstorm co-personalized first-date ideas for 
two users that matched on our dating service 
based on their descriptions below. 
Output the result in the form of a letter to them 
with 5 ideas for the morning, 5 ideas for the afternoon, 
and 5 ideas for the evening signed as 
"Your loving AI assistants".
Think step-by-step to explain why every idea 
is good for them.

--

Theodore Heath: {Theodore's profile summary}

Olivia Windsor: {Olivia's profile summary}
```

**Assistant Message**

```text
Dear Theodore and Olivia,

We are thrilled to assist you in planning some 
unique and personalized dates that will 
cater to both of your interests and values. 
Here are our suggestions:

Morning Ideas:
1. Hiking in the Santa Monica Mountains: 
Given both of your active lifestyles and Theodore's 
love for outdoor activities, this would be a great way 
to start the day. You can enjoy the 
beautiful views and engage in deep conversations, 
which INTJs like Theodore appreciate.
2. Visit to the Griffith Observatory: 
This could be a fun and educational experience for 
both of you. As a Pisces, Theodore might enjoy the 
celestial aspect, while Olivia's adventurous 
spirit would be satisfied.
3. Cooking Class: 
Since Olivia enjoys cooking, a morning cooking 
class could be a fun and interactive experience. 
You could choose a Spanish cuisine class to utilize 
both of your language skills.
4. Dog Park Visit: 
Theodore could bring his dog and you 
both could enjoy a casual morning at 
the park. This would also give Olivia 
a chance to share her love for animals.
5. Photography Walk in Downtown LA: 
Olivia could showcase her photography skills and you 
both could explore the city's architecture and street art.

Afternoon Ideas:
1. Tech Museum Visit: 
Given your shared industry, a visit to a tech museum could 
spark interesting conversations and mutual learning.
2. Beach Picnic: 
Pack a lunch and head to the beach. 
This would combine Theodore's love for the outdoors 
and Olivia's love for travel and adventure.
3. Bookstore Visit: 
As Theodore enjoys reading, visiting a bookstore could be a fun 
experience. You could each pick a book for the other to read.
4. Video Game Arcade: 
Theodore could share his love for video games with 
Olivia in a fun, casual setting.
5. Spanish Language Film: 
Watching a Spanish language film at a local indie 
cinema could be a unique way to use your shared language skills.

Evening Ideas:
1. Star Gazing: 
Given Theodore's Pisces sign and Olivia's adventurous spirit, 
an evening spent star gazing could be both romantic and enjoyable.
2. Comedy Show: 
A comedy show could be a great way to incorporate humor, 
which Theodore values.
3. Home Cooked Dinner: 
Olivia could cook dinner at home, providing a more 
intimate setting. You could play video games 
or discuss the books you picked earlier.
4. Wine Tasting: 
As both of you enjoy a drink occasionally, a local wine 
tasting could be a fun and sophisticated experience.
5. Outdoor Movie Night: 
Set up a projector in your backyard and enjoy a movie 
under the stars. This combines Theodore's 
love for the outdoors and Olivia's love for creativity.

We hope these ideas inspire you to have fun, 
engaging, and meaningful dates. 
Enjoy your time together!

Your loving AI assistants.
```

### Talking Points

**System Message**

```text
You are an expert conversationalist.
```

**User Message**

```text
Brainstorm ten date talking points for two 
users that matched on our dating service 
based on their descriptions below. Output 
the result in the form of a letter to 
them signed as "Your loving AI assistants". 
Think step-by-step.

--

Theodore Heath: {Theodore's profile summary} {Theodore's partner preferences summary}

Olivia Windsor: {Olivia's profile summary} {Olivia's partner preferences summary}
```

**Assistant Message**

```text
Dear Theodore and Olivia,

We are thrilled to introduce you to each other as we 
believe you both share common interests, values, and 
future goals. To help you get to know each other better, 
we have brainstormed some talking points for your first date. 

1. **Backgrounds**: You both have moved from different cities 
to Los Angeles. Discuss what brought you to LA and how your 
experiences have been so far. 

2. **Professions**: You both work in the tech industry. 
Discuss your roles, your journey into tech, and how you 
see the industry evolving.

3. **Education**: You both are graduates from prestigious 
universities. Share your college experiences, favorite subjects, 
and how your education has shaped your career.

4. **Outdoor Activities**: You both enjoy outdoor activities. 
Discuss your favorite hiking trails, outdoor games, and any 
memorable adventures you've had.

5. **Pets**: Theodore, you own a dog and Olivia, you own a cat. 
Share your pet stories, their quirks, and how they've enriched 
your lives.

6. **Future Plans**: You both desire to have children in the future. 
Discuss your thoughts on family, parenting, and how you 
envision your future family life.

7. **Books and Games**: Theodore, you enjoy reading and video games. 
Olivia, you might find it interesting to know what books he's 
currently reading or his favorite video games. 

8. **Travel and Photography**: Olivia, you love traveling and photography. 
Theodore, you might be interested in hearing about her favorite 
travel destinations and her photography experiences.

9. **Cooking**: Olivia, you enjoy cooking. Discuss your favorite dishes 
to cook, and perhaps plan a cooking date in the future.

10. **Values**: You both value honesty and intelligence. Discuss what 
these values mean to you and how they influence your life decisions.

Remember, the goal is to get to know each other better and find 
common ground. Enjoy your date and we wish you both a wonderful time!

Best regards,

Your loving AI assistants.
```

# Cost

The final bill paid to OpenAI for this project was under $80 USD.

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
[readme]: https://github.com/daniel-deychakiwsky/generative-matchmaking/blob/master/README.md
[Projector]: https://projector.tensorflow.org

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
[screenshot]: /assets/images/generative_matchmaking/ui.png
[embeddings]: /assets/images/generative_matchmaking/embeddings.gif
