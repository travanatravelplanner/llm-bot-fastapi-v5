import aiohttp
import asyncio
import datetime
import time
import json
import logging
import os
import requests
import re

from yelp_restaurants import main

from google.cloud import storage
from dotenv import load_dotenv, find_dotenv
import google.generativeai as palm

_ = load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)


class ItineraryGenerator:

    def __init__(self):
        self.log_bucket_name = os.getenv("BUCKET_NAME")
        self.feedback_bucket_name = os.getenv("FEEDBACK_BUCKET_NAME")
        self.storage_client = storage.Client()
        self.default_template = self.load_prompt()
        self.prompt = {"role": "system", "content":self.default_template}
        self.selected_llm = None
        self.user_query_template = None
        self.generated_itinerary = None

    def log_llm_response(self, llm, query, itinerary):
        self.selected_llm = llm
        self.user_query_template = query
        self.generated_itinerary = itinerary
        self._upload_to_bucket(self.log_bucket_name, {"id": self._get_unique_id(), "query": query, "llm": llm, "itinerary": itinerary})

    def user_feedback(self, rating, feedback):
        llm, query, itinerary = self.selected_llm, self.user_query_template, self.generated_itinerary
        self._upload_to_bucket(self.feedback_bucket_name,{
            "id": self._get_unique_id(),
            "user_query": query,
            "LLM": llm,
            "itinerary": itinerary,
            "user_rating": rating,
            "user_feedback": feedback
        })

    def _upload_to_bucket(self, bucket_name, data):
        data_str = json.dumps(data)
        bucket = self.storage_client.get_bucket(bucket_name)
        blob_name = f"log_{self._get_unique_id()}_json"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data_str)

    @staticmethod
    def _get_unique_id():
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    
    @staticmethod
    def load_prompt():
        """
        Define the prompt template for the itinerary planning.
        """
        return """
            You are an expert intelligent AI itinerary planner with extensive knowledge of places worldwide. Your goal is to plan an creative optimized itinerary for the user based on their specific interests and preferences, geographical proximity, and efficient routes to minimize travel time. To achieve this, follow these instructions:

            1. Suggest atleast 3 activities per day. Each activity should include the name of the place, a brief description, estimated cost, and timings.
            
            2. Generate a well-structured itinerary including day-to-day activities, timings to visit each location, and estimated costs for the user's reference.

            2. Take into account factors such as geographical proximity between destinations, transportation options, and other logistical considerations when planning the route.
            
            By following these guidelines, you will create a comprehensive and optimized itinerary that meets the user's expectations while ensuring minimal travel time.

            Human:
            AI:"""
    
    @staticmethod
    def load_itinerary_template_json(
            destination, budget, arrival_date, departure_date, start_time, end_time, additional_info, restaurants
    ):
        
        query = f"""
            Plan a trip to {destination} from {arrival_date} to {departure_date} with a budget of ${budget}. Start the itinerary each day from {start_time} to {end_time}. Consider additional information regarding {additional_info}, if provided.
        """
        
        template = f"""{query}. 
    Consider budget, timings and requirements. Include estimated cost for each activity.
    Use this restaurants list {restaurants} to suggest atleast one restaurant per day. 
    Structure the itinerary as follows:
    {{"Name":"name of the trip", "description":"description of the entire trip", "budget":"budget of the entire thing", "data": [{{"day":1, "day_description":"Description based on the entire day's places. in a couple of words, for example: 'Historical Exploration', 'Spiritual Tour', 'Adventurous Journey', 'Dayout in a beach','Urban Exploration', 'Wildlife Safari','Relaxing Spa Day','Artistic Getaway', 'Romantic Getaway', 'Desert Safari', 'Island Hopping Adventure'",  "places":[{{"name":"Place Name", "description":"Place Description","time": "time to visit this place", "budget":"cost"}}, {{"name":"Place Name 2", "description":"Place Description 2","time": "time to visit this place", "budget":"cost"}}]}}, {{"day":2, "day_description": "Description based on the entire day's places in simple words. Be creative", "places":[{{"name":"Place Name", "description":"Place Description","time": "time to reach this place", "budget":"cost"}}, {{"name":"Place Name 2", "description":"Place Description 2", "time": "time to visit this place", "budget":"cost"}}]}}]}}
    Note: Do not include any extra information outside this structure."""

        return query, template


    async def fetch_place_details(self, session, place, destination, api_key, SEARCH_URL, DETAILS_URL, PHOTO_URL):

        search_payload = {
            'input': place['name'] + ', ' + destination,
            'inputtype': 'textquery',
            'fields': 'place_id',
            'key': api_key
        }

        async with session.get(SEARCH_URL, params=search_payload) as response:
            search_response = await response.json()
        
        if search_response['candidates']:
            place_id = search_response['candidates'][0]['place_id']
            details_payload = {
                'place_id': place_id,
                'fields': 'name,editorial_summary,geometry,formatted_address,reviews,type,website,formatted_phone_number,price_level,rating,user_ratings_total,photo',
                'key': api_key
            }
            async with session.get(DETAILS_URL, params=details_payload) as response:
                details_response = await response.json()
            place_details = details_response['result']

            place.update({
                'address': place_details.get('formatted_address', ''),
                'latitude': place_details['geometry']['location']['lat'],
                'longitude': place_details['geometry']['location']['lng'],
                'name': place_details.get('name', ''),
                'editorial_summary': place_details.get('editorial_summary', ''),
                'reviews': place_details.get('reviews', []),
                'type': place_details.get('type', ''),
                'website': place_details.get('website', ''),
                'formatted_phone_number': place_details.get('formatted_phone_number', ''),
                'price_level': place_details.get('price_level', ''),
                'rating': place_details.get('rating', ''),
                'user_ratings_total': place_details.get('user_ratings_total', '')
            })

            if 'photos' in place_details:
                photo_reference = place_details['photos'][0]['photo_reference']
                photo_payload = {
                    'maxwidth': 400,
                    'photoreference': photo_reference,
                    'key': api_key
                }
                async with session.get(PHOTO_URL, params=photo_payload) as response:
                    place['photo_url'] = str(response.url)


    async def google_place_details(self, destination, itinerary):
        # Base URLs for Google Places API
        SEARCH_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
        PHOTO_URL = "https://maps.googleapis.com/maps/api/place/photo"
        
        google_api_key = os.getenv("GPLACES_API_KEY")

        # Modified regex to handle both variations
        json_str_match = re.search(r'({+.*}+)', itinerary, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group(1)
            # Remove any extra curly braces
            while '{{' in json_str and '}}' in json_str:
                json_str = json_str[1:-1]
            trip_data = json.loads(json_str)
        else:
            # Handle the case where the regex doesn't match
            raise ValueError("Invalid JSON format in itinerary")

        async with aiohttp.ClientSession() as session:
            tasks = []
            for day_data in trip_data['data']:
                for place in day_data['places']:
                    task = asyncio.ensure_future(self.fetch_place_details(session, place, destination, google_api_key, SEARCH_URL, DETAILS_URL, PHOTO_URL))
                    tasks.append(task)
            await asyncio.gather(*tasks)

        return trip_data

    
    @staticmethod
    def validate_json_format(new_itinerary):
        """
        Validate if the provided string is in JSON format.
        """
        try:
            json.loads(new_itinerary)
            return True
        except json.JSONDecodeError:
            return False

    
    def handle_invalid_json(self, itinerary):
        """
        Handle the case when the generated itinerary is not in valid JSON format.
        """
        prompt = f"""You are an expert in JSON formatting. Please ensure the following text is in correct and valid JSON format. 
                Complete the following JSON structure to produce a valid JSON structure:
                example: 
                {itinerary}
                Ensure the final output is a well-structured and valid JSON.
            """

        itinerary = palm.generate_text(
                            model='models/text-bison-001',
                            prompt=prompt,
                            temperature=0,
        )
        new_itinerary = itinerary.result
        
        return new_itinerary

    async def generate_itinerary(self, llm, destination, budget, arrival_date, departure_date, start_time, end_time, additional_info):
        restaurants = main(destination)
        modified_itinerary = None  

        if llm == "Atlas v2":
            user_query, user_query_template = self.load_itinerary_template_json(
            destination, budget, arrival_date, departure_date, start_time, end_time, additional_info, restaurants
        )

            palm_api_key = os.getenv("GOOGLE_PALM_API_KEY")
            palm.configure(api_key=palm_api_key)

            prompt = self.default_template + user_query_template

            itinerary = palm.generate_text(
                            model='models/text-bison-001',
                            prompt=prompt,
                            temperature=0.3,
                            #max_output_tokens=1500,
                        )

            new_itinerary = itinerary.result
            new_itinerary = re.search(r'\{.*\}', new_itinerary, re.DOTALL).group()
            #print("\n =========================================== \n")
            #print(f"Itinerary\n: {str(new_itinerary)}")

            # Validate if new_itinerary is in correct JSON format
            if not self.validate_json_format(new_itinerary):
                print("\n ============== Invalid JSON format in itinerary ============== \n")
                new_itinerary = self.handle_invalid_json(itinerary=new_itinerary)
                #print("\n =========================================== \n")
                #print(f"New Itinerary: {new_itinerary}")
            
            modified_itinerary = await self.google_place_details(destination=destination, itinerary=new_itinerary)
                
            try:
                self.log_llm_response(llm=llm, query=user_query, itinerary=modified_itinerary)
            except Exception as e:
                logging.error(f"Error: {str(e)}")

        return modified_itinerary

