import re
import json
import requests
import multiprocessing
import threading
import numpy as np
from yelp_helpers import request_page, request_api
from time import sleep
from bs4 import BeautifulSoup
from pymongo import MongoClient

# Scraping Businesses
#===============================================================================

POOL_SIZE = 4
API_HOST = "api.yelp.com"
SEARCH_PATH = '/v2/search'
BUSINESS_PATH = '/v2/business/'
COLLECTION_NAME = "business"

# Scraping Reviews
#===============================================================================

BIZ_PATH = 'http://www.yelp.com/biz/'
LAG = np.random.uniform

# MongoDB Info
#===============================================================================

# DB_NAME = "yelp"

# client = MongoClient()
# db = client[DB_NAME]

# Functions: Businesses
#===============================================================================

def city_search_parallel(city):
    '''
    Retrieves the JSON response that contains the top 20 business meta data for city.
    :param city: city name
    '''
    params = {'location': city, 'limit': 20}
    json_response = request(API_HOST, SEARCH_PATH, url_params=params)
    business_info_concurrent(json_response)


def business_info_concurrent(ids):
    '''
    Extracts the business ids from the JSON object and
    retrieves the business data for each id concurrently.
    :param json_response: JSON response from the search API.
    '''
    business_ids = [x['id'] for x in ids['businesses']]

    threads = []
    for i in business_ids:
        thread = threading.Thread(
            target=business_info,
            args=(i,))
        threads.append(thread)
    for thread in threads: thread.start()
    for thread in threads: thread.join()

def business_info(business_id):
    '''
    Makes a request to Yelp's business API and retrieves the business data in JSON format.
    Dumps the JSON response into mongodb.
    :param business_id:
    '''
    business_path = BUSINESS_PATH + business_id
    response = request(API_HOST, business_path)
    coll.insert(response)

def scrape_parallel_concurrent(pool_size):
    '''
    Uses multiple processes to make requests to the search API.
    :param pool_size: number of worker processes
    '''
    with open('data/cities') as f:
        cities = f.read().splitlines()
    pool = multiprocessing.Pool(processes=pool_size)
    outputs = pool.map(city_search_parallel, cities)

# Functions: Reviews
#===============================================================================

def scrape_reviews(business_ids):
    '''
    Scrapes the business's meta data for a list of review text from the list of
    businesses in San Francisco.
    '''
    for business_id in business_ids:
        print business_id

        biz = []
        url = BIZ_PATH + business_id
        text, review_count = request_page(url)
        biz.extend(text)

        steps = review_count / 20
        floored = steps * 20
        sleep(LAG(1., 5.))
        for n in np.linspace(20, floored, steps):
            url = BIZ_PATH + business_id + '?start=' + str(int(n))
            text, ignore = request_page(url)
            biz.extend(text)
            sleep(LAG(1., 4.))

        f_path = 'ydata/' + business_id
        with open(f_path, 'w') as f:
            biz_file = {'text': biz}
            json.dump(biz_file, f)
        print 'fin!'
        sleep(10)

# Businesses in SF
#===============================================================================

businesses = [
'akikos-restaurant-san-francisco',
'amaweles-south-african-kitchen-san-francisco-2',
'archive-bar-and-kitchen-san-francisco',
'barbacco-san-francisco',
'burritt-room-tavern-san-francisco',
'butler-and-the-chef-bistro-san-francisco',
'cafe-algiers-san-francisco',
'cafe-du-soleil-san-francisco-4',
'cafe-madeleine-san-francisco-3',
'cafe-venue-san-francisco-8',
'capt-eddie-rickenbackers-san-francisco-2',
'chaat-corner-san-francisco',
'chez-fayala-san-francisco',
'crepe-madame-san-francisco-2',
'creperie-saint-germain-san-francisco-12',
'crossroads-cafe-san-francisco-7',
'curry-up-now-san-francisco-6',
'delancey-street-restaurant-san-francisco',
'delica-san-francisco',
'dragoneats-san-francisco-3',
'eatsa-san-francisco',
'eden-plaza-cafe-san-francisco',
'fang-san-francisco-2',
'fearless-coffee-san-francisco-3',
'freshroll-vietnamese-rolls-and-bowls-san-francisco',
'front-door-cafe-san-francisco',
'garaje-san-francisco',
'gyro-king-san-francisco-2',
'halal-cart-san-francisco',
'henrys-hunan-restaurant-san-francisco',
'hog-island-oyster-co-san-francisco',
'hops-and-hominy-san-francisco',
'hrd-san-francisco-2',
'il-cane-rosso-san-francisco',
'jersey-san-francisco',
'johns-snack-and-deli-san-francisco',
'kate-o-briens-irish-bar-and-grill-san-francisco',
'la-briciola-san-francisco',
'la-capra-san-francisco-2',
'la-fusi%C3%B3n-san-francisco-2',
'lees-deli-san-francisco-7',
'little-skillet-san-francisco-2',
'local-kitchen-and-wine-merchant-san-francisco-2',
'locali-mediterranean-san-francisco',
'lord-george-san-francisco',
'louies-bar-san-francisco',
'marlowe-san-francisco-2',
'mixt-greens-san-francisco',
'muraccis-japanese-curry-and-grill-san-francisco',
'north-india-restaurant-san-francisco',
'oasis-grill-san-francisco',
'oasis-grill-san-francisco-5',
'osha-thai-san-francisco',
'pachino-trattoria-and-pizzeria-san-francisco',
'palomino-san-francisco-3',
'pazzia-restaurant-and-pizzeria-san-francisco',
'per-diem-san-francisco',
'perilla-san-francisco',
'picnic-on-third-san-francisco',
'portico-restaurant-san-francisco-3',
'proper-food-san-francisco-5',
'r-and-g-lounge-san-francisco',
'red-dog-restaurant-and-bar-san-francisco',
'sammys-on-2nd-san-francisco',
'samovar-tea-lounge-san-francisco-2',
'sauce-san-francisco-3',
'sausalito-cafe-san-francisco-3',
'se%C3%B1or-sisig-san-francisco-3',
'slider-shack-san-francisco',
'soma-eats-san-francisco',
'south-park-caf%C3%A9-san-francisco-4',
'southside-spirit-house-san-francisco',
'specialtys-cafe-and-bakery-san-francisco-19',
'spice-kit-san-francisco',
'subway-san-francisco-3',
'super-duper-burgers-san-francisco-3',
'super-duper-burgers-san-francisco-6',
'sushi-fantastic-san-francisco',
'sushirrito-san-francisco',
'sushirrito-san-francisco-4',
'sweet-joannas-cafe-san-francisco-2',
'tava-indian-kitchen-san-francisco-2',
'tender-greens-san-francisco',
'the-american-grilled-cheese-kitchen-san-francisco',
'the-dosa-brothers-san-francisco-4',
'the-golden-west-san-francisco',
'the-grove-yerba-buena-san-francisco',
'the-melt-howard-san-francisco',
'the-melt-new-montgomery-san-francisco',
'the-pink-elephant-san-francisco-3',
'the-sentinel-san-francisco',
'the-store-on-the-corner-san-francisco',
'thirsty-bear-brewing-company-san-francisco-2',
'trace-san-francisco',
'tropisue%C3%B1o-san-francisco-3',
'trou-normand-san-francisco',
'uno-dos-tacos-san-francisco-3',
'wanna-e-san-francisco',
'working-girls-cafe-san-francisco-3',
'zero-zero-san-francisco'
]

if __name__ == '__main__':
    scrape_reviews(businesses)