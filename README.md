Using Aspect-Based Sentiment For More Personalized Restaurant Recommendations
=============================================================================
About
-----
This project makes the assertion that much more work goes into finding the right restaurant to eat at than just finding the highest rated restaurant in town. There are different criteria that people look into while jumping from review to review to get hints about when deciding on a restaurant. I created a proof of concept for Yelp Helper to improve upon the restaurant search experience.

Yelp Helper takes user-specified weights for the different components of the restaurant experience (food, service, atmosphere, value, and decor) and caters your restaurant search results to those weightings. 

The best part is that Yelp Helper lists the restaurants with the bar charts showing what the average Yelper thinks about each of those components for each restaurant. This gives users a quick view of what each restaurant excels at; all without reading a single Yelp review.

Slides:
https://docs.google.com/presentation/d/12YiE524RtKoTDpPqFrh8ETWj5YEes7Xj-BNTmECueZU/edit

explore_yelp_data.ipynb
-----------------------
The iPython Notebook shows a cleaned up exploration of the Yelp Review dataset. A lot of the work determining the most prominent subjects, the distributions of words around these subjects, the distributions of ways adjectives are separated from these subjects, and the SQL queries to examine the results of my search.

A lot of the cleaned up work included TF-IDF, collocation, and Markov chain work I'd done to understand the dataset and collect the adjective-subject phrases I needed for my model.

The exploratory work around seeing the distributions of words around my subjects and adjectives had to do with not having a strong enough part-of-speech tagger and lack of syntactic parser to create the features I needed. The distributions aided my ablility to collect a larger percentage of the adjective-subject pairings in the text, but far from all of them. Future work includes a way to map these pairs together without hard-coding them in.

set_sent_model.py
-----------------
This utility page held all the necessary functions to grab the data, collect my the adjectives for my features, create my feature matrices, and generate the new scorings for my restaurants.

sf_yelp.py
----------
This utility page contained all the web scraping utility functions required to scrape the business IDs for the restaurant reviews needed for my database of SF restaurants and the utility functions for systematically webscrape all of the reviews for these restaurants without overloading Yelp with requests.

yelp_helpers.py
---------------
This final utility page contained the functions for making requests to the Yelp API to gather the business IDs for the highest rated restaurants around the SoMa area.

Coming Soon!
------------
- An Aspect-Based Sentiment model class that can **train** a model on the corpus of Yelp review text and  **predict** the component scores based on the given text of a review.
- A hosted version of my model to let you try out the model I've already trained to predict your experience at a given restaurant given the text of the review you input into the text box
- Rebuild of parts of the algo to scale better and some fixes that may be able to generalize the aspect-based sentiment to other review realms such as Amazon.