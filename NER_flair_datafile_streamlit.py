#!/usr/bin/env python3
"""
Streamlit Entity Linker Application - Batch Processing Version

A web interface for the Entity Linker using Streamlit with batch file processing.
This application processes CSV/Excel files and outputs JSON-LD files for each row.

Author: Based on entity_linker.py
Version: 2.0 - Updated for batch processing with file input
"""

import streamlit as st

# Configure Streamlit page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Batch Entity Linker using Flair",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Authentication is REQUIRED - do not run app without proper login
try:
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    import os
    
    # Check if config file exists
    if not os.path.exists('config.yaml'):
        st.error("Authentication required: config.yaml file not found!")
        st.info("Please ensure config.yaml is in the same directory as this app.")
        st.stop()
    
    # Load configuration
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Setup authentication
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # Check if already authenticated via session state
    if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
        name = st.session_state['name']
        authenticator.logout("Logout", "sidebar")
        # Continue to app below...
    else:
        # Render login form
        try:
            # Try different login methods
            login_result = None
            try:
                login_result = authenticator.login(location='main')
            except TypeError:
                try:
                    login_result = authenticator.login('Login', 'main')
                except TypeError:
                    login_result = authenticator.login()
            
            # Handle the result
            if login_result is None:
                # Check session state for authentication result
                if 'authentication_status' in st.session_state:
                    auth_status = st.session_state['authentication_status']
                    if auth_status == False:
                        st.error("Username/password is incorrect")
                        st.info("Try username: demo_user with your password")
                    elif auth_status == None:
                        st.warning("Please enter your username and password")
                    elif auth_status == True:
                        st.rerun()  # Refresh to show authenticated state
                else:
                    st.warning("Please enter your username and password")
                st.stop()
            elif isinstance(login_result, tuple) and len(login_result) == 3:
                name, auth_status, username = login_result
                # Store in session state
                st.session_state['authentication_status'] = auth_status
                st.session_state['name'] = name
                st.session_state['username'] = username
                
                if auth_status == True:
                    st.rerun()  # Refresh to show authenticated state
                elif auth_status == False:
                    st.error("Username/password is incorrect")
                    st.stop()
            else:
                st.error(f"Unexpected login result format: {login_result}")
                st.stop()
                
        except Exception as login_error:
            st.error(f"Login method error: {login_error}")
            st.stop()
        
except ImportError:
    st.error("Authentication required: streamlit-authenticator not installed!")
    st.info("Please install streamlit-authenticator to access this application.")
    st.stop()
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.info("Cannot proceed without proper authentication.")
    st.stop()

import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import base64
from typing import List, Dict, Any
import sys
import os
import shutil
from datetime import datetime

# We'll include the EntityLinker class in this same file instead of importing
# This makes the app self-contained

class EntityLinker:
    """
    Main class for entity linking functionality.
    
    This class handles the complete pipeline from text processing to entity
    extraction, validation, linking, and output generation.
    """
    
    def __init__(self):
        """Initialize the EntityLinker and load required Flair model."""
        self.tagger = self._load_flair_model()
        
        # Color scheme for different entity types in HTML output
        self.colors = {
            'PER': '#BF7B69',          # F&B Red earth        
            'ORG': '#9fd2cd',          # F&B Blue ground
            'LOC': '#C4C3A2',          # F&B Cooking apple green
            'MISC': '#EFCA89',         # F&B Yellow ground. 
            'FAC': '#C3B5AC',          # F&B Elephants breath
            'GSP': '#C4A998',          # F&B Dead salmon
            'ADDRESS': '#CCBEAA'       # F&B Oxford stone
        }
    
    def _load_flair_model(self):
        """Load Flair NER model with error handling."""
        try:
            # Show loading message immediately
            loading_placeholder = st.empty()
            loading_placeholder.info("Loading Flair NER model... This may take a minute on first run.")
            
            from flair.models import SequenceTagger
            
            # Load the standard NER model from Flair
            tagger = SequenceTagger.load('ner')
            
            loading_placeholder.success(f"Loaded Flair NER model successfully")
            return tagger
        except Exception as e:
            st.error(f"Failed to load Flair NER model: {e}")
            st.error("Please ensure Flair is installed properly.")
            st.code("pip install flair")
            st.stop()

    def extract_entities(self, text: str):
        """Extract named entities from text using Flair with proper validation."""
        from flair.data import Sentence
        
        # Process text with Flair
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        
        entities = []
        
        # Step 1: Extract named entities with validation
        for ent in sentence.get_spans('ner'):
            # Filter out unwanted entity types
            tag = ent.get_label('ner').value
            if tag in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                continue
            
            # Map Flair entity types to our format
            entity_type = self._map_flair_entity_type(tag)
            
            # Additional filter in case mapping returns an unwanted type
            if entity_type in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                continue
            
            # Validate entity using grammatical context
            if self._is_valid_entity(ent.text, entity_type, ent):
                entities.append({
                    'text': ent.text,
                    'type': entity_type,
                    'start': ent.start_position,
                    'end': ent.end_position,
                    'label': tag  # Keep original Flair label for reference
                })
        
        # Step 2: Extract addresses
        addresses = self._extract_addresses(text)
        entities.extend(addresses)
        
        # Step 3: Remove overlapping entities
        entities = self._remove_overlapping_and_duplicate_entities(entities)
        
        return entities

    def _map_flair_entity_type(self, flair_label: str) -> str:
        """Map Flair entity labels to our standardized types."""
        # Flair's standard model uses CoNLL03 tags: PER, LOC, ORG, MISC
        mapping = {
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'MISC': 'MISC',
            # Additional mappings for other Flair models
            'B-PER': 'PERSON',
            'I-PER': 'PERSON',
            'B-ORG': 'ORGANIZATION',
            'I-ORG': 'ORGANIZATION',
            'B-LOC': 'LOCATION',
            'I-LOC': 'LOCATION',
            'B-MISC': 'MISC',
            'I-MISC': 'MISC',
            # OntoNotes model mappings if using that model
            'PERSON': 'PERSON',
            'ORGANIZATION': 'ORGANIZATION',
            'GPE': 'LOCATION', 
            'LOCATION': 'LOCATION',
            'FACILITY': 'LOCATION',
            'PRODUCT': 'MISC',
            'EVENT': 'MISC',
            'WORK_OF_ART': 'MISC',
            'LANGUAGE': 'MISC'
        }
        return mapping.get(flair_label, flair_label)

    def link_to_britannica(self, entities):
        """Add basic Britannica linking.""" 
        import requests
        import re
        import time
        
        for entity in entities:
            # Skip if already has Wikidata or Wikipedia link
            if entity.get('wikidata_url') or entity.get('wikipedia_url'):
                continue
                
            try:
                search_url = "https://www.britannica.com/search"
                params = {'query': entity['text']}
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(search_url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    # Look for article links
                    pattern = r'href="(/topic/[^"]*)"[^>]*>([^<]*)</a>'
                    matches = re.findall(pattern, response.text)
                    
                    for url_path, link_text in matches:
                        if (entity['text'].lower() in link_text.lower() or 
                            link_text.lower() in entity['text'].lower()):
                            entity['britannica_url'] = f"https://www.britannica.com{url_path}"
                            entity['britannica_title'] = link_text.strip()
                            break
                
                time.sleep(0.3)  # Rate limiting
            except Exception:
                pass
        
        return entities

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Detect geographical context from the text to improve geocoding accuracy."""
        import re
        
        context_clues = []
        text_lower = text.lower()
        
        # Extract major cities/countries mentioned in the text
        major_locations = {
            # Countries
            'uk': ['uk', 'united kingdom', 'britain', 'great britain'],
            'usa': ['usa', 'united states', 'america', 'us '],
            'canada': ['canada'],
            'australia': ['australia'],
            'france': ['france'],
            'germany': ['germany'],
            'italy': ['italy'],
            'spain': ['spain'],
            'japan': ['japan'],
            'china': ['china'],
            'india': ['india'],
            
            # Major cities that provide strong context
            'london': ['london'],
            'new york': ['new york', 'nyc', 'manhattan'],
            'paris': ['paris'],
            'tokyo': ['tokyo'],
            'sydney': ['sydney'],
            'toronto': ['toronto'],
            'berlin': ['berlin'],
            'rome': ['rome'],
            'madrid': ['madrid'],
            'beijing': ['beijing'],
            'mumbai': ['mumbai'],
            'los angeles': ['los angeles', 'la ', ' la,'],
            'chicago': ['chicago'],
            'boston': ['boston'],
            'edinburgh': ['edinburgh'],
            'glasgow': ['glasgow'],
            'manchester': ['manchester'],
            'birmingham': ['birmingham'],
            'liverpool': ['liverpool'],
            'bristol': ['bristol'],
            'leeds': ['leeds'],
            'cardiff': ['cardiff'],
            'belfast': ['belfast'],
            'dublin': ['dublin'],
        }
        
        # Check for explicit mentions
        for location, patterns in major_locations.items():
            for pattern in patterns:
                if pattern in text_lower:
                    context_clues.append(location)
                    break
        
        # Extract from entities that are already identified as places
        for entity in entities:
            if entity['type'] in ['LOCATION', 'GPE', 'FACILITY']:
                entity_lower = entity['text'].lower()
                # Add major locations found in entities
                for location, patterns in major_locations.items():
                    if entity_lower in patterns or any(p in entity_lower for p in patterns):
                        if location not in context_clues:
                            context_clues.append(location)
        
        # Look for postal codes to infer country
        postal_patterns = {
            'uk': [
                r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b',  # UK postcodes
                r'\b[A-Z]{2}\d{1,2}\s*\d[A-Z]{2}\b'
            ],
            'usa': [
                r'\b\d{5}(-\d{4})?\b'  # US ZIP codes
            ],
            'canada': [
                r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b'  # Canadian postal codes
            ]
        }
        
        for country, patterns in postal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    if country not in context_clues:
                        context_clues.append(country)
                    break
        
        # Prioritize context (more specific first)
        priority_order = ['london', 'new york', 'paris', 'tokyo', 'sydney', 'uk', 'usa', 'canada', 'australia', 'france', 'germany']
        prioritized_context = []
        
        for priority_location in priority_order:
            if priority_location in context_clues:
                prioritized_context.append(priority_location)
        
        # Add remaining context clues
        for clue in context_clues:
            if clue not in prioritized_context:
                prioritized_context.append(clue)
        
        return prioritized_context[:3]  # Return top 3 context clues

    def get_coordinates(self, entities):
        """Enhanced coordinate lookup with geographical context detection."""
        import requests
        import time
        
        # Detect geographical context from the full text
        context_clues = self._detect_geographical_context(
            st.session_state.get('processed_text', ''), 
            entities
        )
        
        place_types = ['LOCATION', 'GPE', 'FACILITY', 'ORGANIZATION', 'ADDRESS']
        
        for entity in entities:
            if entity['type'] in place_types:
                # Skip if already has coordinates
                if entity.get('latitude') is not None:
                    continue
                
                # Try geocoding with context
                if self._try_contextual_geocoding(entity, context_clues):
                    continue
                    
                # Fall back to original methods
                if self._try_python_geocoding(entity):
                    continue
                
                if self._try_openstreetmap(entity):
                    continue
                    
                # If still no coordinates, try a more aggressive search
                self._try_aggressive_geocoding(entity)
        
        return entities
    
    def _try_contextual_geocoding(self, entity, context_clues):
        """Try geocoding with geographical context."""
        import requests
        import time
        
        if not context_clues:
            return False
        
        # Create context-aware search terms
        search_variations = [entity['text']]
        
        # Add context to search terms
        for context in context_clues:
            context_mapping = {
                'uk': ['UK', 'United Kingdom', 'England', 'Britain'],
                'usa': ['USA', 'United States', 'US'],
                'canada': ['Canada'],
                'australia': ['Australia'],
                'france': ['France'],
                'germany': ['Germany'],
                'london': ['London, UK', 'London, England'],
                'new york': ['New York, USA', 'New York, NY'],
                'paris': ['Paris, France'],
                'tokyo': ['Tokyo, Japan'],
                'sydney': ['Sydney, Australia'],
            }
            
            context_variants = context_mapping.get(context, [context])
            for variant in context_variants:
                search_variations.append(f"{entity['text']}, {variant}")
        
        # Remove duplicates while preserving order
        search_variations = list(dict.fromkeys(search_variations))
        
        # Try geopy first with context
        try:
            from geopy.geocoders import Nominatim
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError
            
            geocoder = Nominatim(user_agent="EntityLinker/1.0", timeout=10)
            
            for search_term in search_variations[:5]:  # Try top 5 variations
                try:
                    location = geocoder.geocode(search_term, timeout=10)
                    if location:
                        entity['latitude'] = location.latitude
                        entity['longitude'] = location.longitude
                        entity['location_name'] = location.address
                        entity['geocoding_source'] = f'geopy_contextual'
                        entity['search_term_used'] = search_term
                        return True
                    
                    time.sleep(0.2)  # Rate limiting
                except (GeocoderTimedOut, GeocoderServiceError):
                    continue
                    
        except ImportError:
            pass
        
        # Fall back to OpenStreetMap with context
        for search_term in search_variations[:3]:  # Try top 3 with OSM
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': search_term,
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1
                }
                headers = {'User-Agent': 'EntityLinker/1.0'}
            
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        entity['latitude'] = float(result['lat'])
                        entity['longitude'] = float(result['lon'])
                        entity['location_name'] = result['display_name']
                        entity['geocoding_source'] = f'openstreetmap_contextual'
                        entity['search_term_used'] = search_term
                        return True
            
                time.sleep(0.3)  # Rate limiting
            except Exception:
                continue
        
        return False
    
    def _try_python_geocoding(self, entity):
        """Try Python geocoding libraries (geopy) - original method."""
        try:
            from geopy.geocoders import Nominatim, ArcGIS
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError
            
            geocoders = [
                ('nominatim', Nominatim(user_agent="EntityLinker/1.0", timeout=10)),
                ('arcgis', ArcGIS(timeout=10)),
            ]
            
            for name, geocoder in geocoders:
                try:
                    location = geocoder.geocode(entity['text'], timeout=10)
                    if location:
                        entity['latitude'] = location.latitude
                        entity['longitude'] = location.longitude
                        entity['location_name'] = location.address
                        entity['geocoding_source'] = f'geopy_{name}'
                        return True
                        
                    time.sleep(0.3)
                except (GeocoderTimedOut, GeocoderServiceError):
                    continue
                except Exception as e:
                    continue
                        
        except ImportError:
            pass
        except Exception as e:
            pass
        
        return False
    
    def _try_openstreetmap(self, entity):
        """Fall back to direct OpenStreetMap Nominatim API."""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': entity['text'],
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {'User-Agent': 'EntityLinker/1.0'}
        
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = data[0]
                    entity['latitude'] = float(result['lat'])
                    entity['longitude'] = float(result['lon'])
                    entity['location_name'] = result['display_name']
                    entity['geocoding_source'] = 'openstreetmap'
                    return True
        
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            pass
        
        return False
    
    def _try_aggressive_geocoding(self, entity):
        """Try more aggressive geocoding with different search terms."""
        import requests
        import time
        
        # Try variations of the entity name
        search_variations = [
            entity['text'],
            f"{entity['text']}, UK",  # Add country for UK places
            f"{entity['text']}, England",
            f"{entity['text']}, Scotland",
            f"{entity['text']}, Wales",
            f"{entity['text']} city",
            f"{entity['text']} town"
        ]
        
        for search_term in search_variations:
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': search_term,
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1
                }
                headers = {'User-Agent': 'EntityLinker/1.0'}
            
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        entity['latitude'] = float(result['lat'])
                        entity['longitude'] = float(result['lon'])
                        entity['location_name'] = result['display_name']
                        entity['geocoding_source'] = f'openstreetmap_aggressive'
                        entity['search_term_used'] = search_term
                        return True
            
                time.sleep(0.2)  # Rate limiting between attempts
            except Exception:
                continue
        
        return False

    def _is_valid_entity(self, entity_text: str, entity_type: str, flair_span) -> bool:
        """Validate an entity from Flair."""
        # Skip very short entities
        if len(entity_text.strip()) <= 1:
            return False
        
        # For Flair, we'll use simpler validation since we don't have access to 
        # the same linguistic features as spaCy
        
        # Simple validation based on entity text
        if entity_type == 'PERSON':
            # Check if it looks like a person name (capitalized words)
            words = entity_text.split()
            if all(word[0].isupper() for word in words if word.isalpha()):
                return True
            return False
            
        # For other types, we'll trust Flair's prediction
        return True

    def _extract_addresses(self, text: str):
        """Extract address patterns that NER might miss."""
        import re
        addresses = []
        
        # Patterns for different address formats
        address_patterns = [
            r'\b\d{1,4}[-—]\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens)\b',
            r'\b\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens)\b'
        ]
        
        for pattern in address_patterns:
            for match in re.finditer(pattern, text):
                addresses.append({
                    'text': match.group(),
                    'type': 'ADDRESS',
                    'start': match.start(),
                    'end': match.end()
                })
        
        return addresses

    def _remove_overlapping_and_duplicate_entities(self, entities):
        """Enhanced method that removes overlapping entities and exact positional duplicates."""
        if not entities:
            return entities
        
        # Step 1: Remove exact positional duplicates (same text, same start/end position)
        entities = self._remove_exact_positional_duplicates(entities)
        
        # Step 2: Handle overlapping entities (keep longer ones)
        entities.sort(key=lambda x: x['start'])
        
        filtered = []
        for entity in entities:
            overlaps = False
            for existing in filtered[:]:  # Create a copy to safely modify during iteration
                # Check if entities overlap in position
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # If current entity is longer, remove the existing one
                    if len(entity['text']) > len(existing['text']):
                        filtered.remove(existing)
                        break
                    else:
                        # Current entity is shorter, skip it
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered
    
    def _remove_exact_positional_duplicates(self, entities):
        """Remove entities that are identical in text, type, and position."""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            # Create a key based on text, type, and position
            key = (entity['text'].lower().strip(), entity['type'], entity['start'], entity['end'])
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated

    def link_to_wikidata(self, entities):
        """Add basic Wikidata linking."""
        import requests
        import time
        
        for entity in entities:
            try:
                url = "https://www.wikidata.org/w/api.php"
                params = {
                    'action': 'wbsearchentities',
                    'format': 'json',
                    'search': entity['text'],
                    'language': 'en',
                    'limit': 1,
                    'type': 'item'
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('search') and len(data['search']) > 0:
                        result = data['search'][0]
                        entity['wikidata_url'] = f"http://www.wikidata.org/entity/{result['id']}"
                        entity['wikidata_description'] = result.get('description', '')
                
                time.sleep(0.1)  # Rate limiting
            except Exception:
                pass  # Continue if API call fails
        
        return entities

    def link_to_wikipedia(self, entities):
        """Add Wikipedia linking for entities without Wikidata links."""
        import requests
        import time
        import urllib.parse
        
        for entity in entities:
            # Skip if already has Wikidata link
            if entity.get('wikidata_url'):
                continue
                
            try:
                # Use Wikipedia's search API
                search_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': entity['text'],
                    'srlimit': 1
                }
                
                headers = {'User-Agent': 'EntityLinker/1.0'}
                response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('query', {}).get('search'):
                        # Get the first search result
                        result = data['query']['search'][0]
                        page_title = result['title']
                        
                        # Create Wikipedia URL
                        encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                        entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                        entity['wikipedia_title'] = page_title
                        
                        # Get a snippet/description from the search result
                        if result.get('snippet'):
                            # Clean up the snippet (remove HTML tags)
                            import re
                            snippet = re.sub(r'<[^>]+>', '', result['snippet'])
                            entity['wikipedia_description'] = snippet[:200] + "..." if len(snippet) > 200 else snippet
                
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                pass
        
        return entities

    def link_to_openstreetmap(self, entities):
        """Add OpenStreetMap links to addresses."""
        import requests
        import time
        
        for entity in entities:
            # Only process ADDRESS entities
            if entity['type'] != 'ADDRESS':
                continue
                
            try:
                # Search OpenStreetMap Nominatim for the address
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': entity['text'],
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1
                }
                headers = {'User-Agent': 'EntityLinker/1.0'}
                
                response = requests.get(url, params=params, headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        # Create OpenStreetMap link
                        lat = result['lat']
                        lon = result['lon']
                        entity['openstreetmap_url'] = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=18"
                        entity['openstreetmap_display_name'] = result['display_name']
                        
                        # Also add coordinates
                        entity['latitude'] = float(lat)
                        entity['longitude'] = float(lon)
                        entity['location_name'] = result['display_name']
                
                time.sleep(0.2)  # Rate limiting
            except Exception:
                pass
        
        return entities


class BatchEntityLinker:
    """
    Batch processing version of EntityLinker for processing CSV/Excel files.
    """
    
    def __init__(self):
        """Initialize the Batch Entity Linker."""
        self.entity_linker = EntityLinker()
        
        # Initialize session state
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'text_column' not in st.session_state:
            st.session_state.text_column = None
        if 'id_column' not in st.session_state:
            st.session_state.id_column = None
        if 'output_dir' not in st.session_state:
            st.session_state.output_dir = None

    def render_header(self):
        """Render the application header."""
        # Display logo if it exists
        try:
            logo_path = "logo.png"  
            if os.path.exists(logo_path):
                st.image(logo_path, width=300)
            else:
                st.info("💡 Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.header("Batch Entity Linker using Flair")
        st.markdown("**Process CSV/Excel files and extract entities to JSON-LD format**")
        
        # Create a simple process diagram
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Upload File</strong><br><small>CSV/Excel</small>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Select Columns</strong><br><small>Text & ID columns</small>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #EFCA89; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Process Batch</strong><br><small>Extract entities</small>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #BF7B69; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>JSON-LD Files</strong><br><small>Individual outputs</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_file_upload(self):
        """Render file upload section."""
        st.header("1. Upload File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file containing text data to process"
        )
        
        if uploaded_file is not None:
            try:
                # Load the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.success(f"File uploaded successfully! {len(df)} rows, {len(df.columns)} columns")
                
                # Show preview
                with st.expander("Preview Data", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.df = None
        
        return uploaded_file is not None

    def render_column_selection(self):
        """Render column selection section."""
        if st.session_state.df is None:
            st.info("Please upload a file first.")
            return False
        
        st.header("2. Select Columns")
        
        df = st.session_state.df
        columns = list(df.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            text_column = st.selectbox(
                "Text Column",
                columns,
                help="Select the column containing the text to process"
            )
            st.session_state.text_column = text_column
        
        with col2:
            id_column = st.selectbox(
                "Unique ID Column",
                columns,
                help="Select the column to use as unique identifier for naming output files"
            )
            st.session_state.id_column = id_column
        
        # Show sample of selected columns
        if text_column and id_column:
            st.subheader("Sample Data")
            sample_df = df[[id_column, text_column]].head(5)
            st.dataframe(sample_df, use_container_width=True)
            
            # Show text length statistics
            text_lengths = df[text_column].astype(str).str.len()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Text Length", f"{text_lengths.mean():.0f} chars")
            with col2:
                st.metric("Max Text Length", f"{text_lengths.max():.0f} chars")
            with col3:
                st.metric("Rows to Process", len(df))
            
            return True
        
        return False

    def render_output_settings(self):
        """Render output settings section."""
        st.header("3. Output Settings")
        
        # Output directory name
        output_dir = st.text_input(
            "Output Directory Name",
            value=f"entity_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Name of the subdirectory where JSON-LD files will be saved"
        )
        
        st.session_state.output_dir = output_dir
        
        if output_dir:
            st.info(f"Files will be saved to: `./{output_dir}/`")
            return True
        
        return False

    def process_batch(self):
        """Process the entire batch and create JSON-LD files."""
        if not all([st.session_state.df is not None, 
                    st.session_state.text_column, 
                    st.session_state.id_column, 
                    st.session_state.output_dir]):
            st.error("Please complete all steps above before processing.")
            return
        
        df = st.session_state.df
        text_column = st.session_state.text_column
        id_column = st.session_state.id_column
        output_dir = st.session_state.output_dir
        
        # Create output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each row
        total_rows = len(df)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_files = []
        errors = []
        
        for index, row in df.iterrows():
            try:
                # Update progress
                progress = (index + 1) / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Processing row {index + 1} of {total_rows}")
                
                # Get text and ID
                text = str(row[text_column])
                unique_id = str(row[id_column])
                
                # Skip empty text
                if not text or text.strip() == '' or text.strip() == 'nan':
                    continue
                
                # Process text
                entities = self.entity_linker.extract_entities(text)
                
                # Link entities
                entities = self.entity_linker.link_to_wikidata(entities)
                entities = self.entity_linker.link_to_wikipedia(entities)
                entities = self.entity_linker.link_to_britannica(entities)
                entities = self.entity_linker.get_coordinates(entities)
                entities = self.entity_linker.link_to_openstreetmap(entities)
                
                # Create JSON-LD data
                json_data = {
                    "@context": "http://schema.org/",
                    "@type": "TextDigitalDocument",
                    "identifier": unique_id,
                    "text": text,
                    "dateCreated": datetime.now().isoformat(),
                    "entities": []
                }
                
                # Format entities for JSON-LD
                for entity in entities:
                    entity_data = {
                        "name": entity['text'],
                        "type": entity['type'],
                        "startOffset": entity['start'],
                        "endOffset": entity['end']
                    }
                    
                    if entity.get('wikidata_url'):
                        entity_data['sameAs'] = entity['wikidata_url']
                    
                    if entity.get('wikidata_description'):
                        entity_data['description'] = entity['wikidata_description']
                    elif entity.get('wikipedia_description'):
                        entity_data['description'] = entity['wikipedia_description']
                    elif entity.get('britannica_title'):
                        entity_data['description'] = entity['britannica_title']
                    
                    if entity.get('latitude') and entity.get('longitude'):
                        entity_data['geo'] = {
                            "@type": "GeoCoordinates",
                            "latitude": entity['latitude'],
                            "longitude": entity['longitude']
                        }
                        if entity.get('location_name'):
                            entity_data['geo']['name'] = entity['location_name']
                    
                    # Add additional URLs
                    urls = []
                    if entity.get('wikipedia_url'):
                        urls.append(entity['wikipedia_url'])
                    if entity.get('britannica_url'):
                        urls.append(entity['britannica_url'])
                    if entity.get('openstreetmap_url'):
                        urls.append(entity['openstreetmap_url'])
                    
                    if urls:
                        if entity_data.get('sameAs'):
                            if isinstance(entity_data['sameAs'], str):
                                entity_data['sameAs'] = [entity_data['sameAs']] + urls
                            else:
                                entity_data['sameAs'].extend(urls)
                        else:
                            entity_data['sameAs'] = urls if len(urls) > 1 else urls[0]
                    
                    json_data['entities'].append(entity_data)
                
                # Save to file
                # Clean filename by removing invalid characters
                clean_id = "".join(c for c in unique_id if c.isalnum() or c in (' ', '-', '_')).rstrip()
                clean_id = clean_id.replace(' ', '_')
                if not clean_id:
                    clean_id = f"row_{index}"
                
                filename = f"{clean_id}.jsonld"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                processed_files.append((unique_id, filename, len(entities)))
                
            except Exception as e:
                errors.append(f"Row {index + 1} (ID: {row.get(id_column, 'unknown')}): {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        st.success(f"Processing complete! {len(processed_files)} files created in '{output_dir}/'")
        
        # Show summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files Created", len(processed_files))
        with col2:
            st.metric("Errors", len(errors))
        
        # Show processed files
        if processed_files:
            with st.expander("Processed Files", expanded=True):
                files_df = pd.DataFrame(processed_files, columns=['ID', 'Filename', 'Entities'])
                st.dataframe(files_df, use_container_width=True)
        
        # Show errors if any
        if errors:
            with st.expander("Errors", expanded=False):
                for error in errors:
                    st.error(error)

    def run(self):
        """Main application runner."""
        # Add custom CSS
        st.markdown("""
        <style>
        .stApp {
            background-color: #F5F0DC !important;
        }
        .main .block-container {
            background-color: #F5F0DC !important;
        }
        .stSidebar {
            background-color: #F5F0DC !important;
        }
        .stSelectbox > div > div {
            background-color: white !important;
        }
        .stTextInput > div > div > input {
            background-color: white !important;
        }
        .stExpander {
            background-color: white !important;
            border: 1px solid #E0D7C0 !important;
            border-radius: 4px !important;
        }
        .stDataFrame {
            background-color: white !important;
        }
        .stButton > button {
            background-color: #C4A998 !important;
            color: black !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: 500 !important;
        }
        .stButton > button:hover {
            background-color: #B5998A !important;
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Render header
        self.render_header()
        
        # Step 1: File upload
        file_uploaded = self.render_file_upload()
        
        if file_uploaded:
            # Step 2: Column selection
            columns_selected = self.render_column_selection()
            
            if columns_selected:
                # Step 3: Output settings
                output_configured = self.render_output_settings()
                
                if output_configured:
                    # Step 4: Process button
                    st.header("4. Process Batch")
                    
                    if st.button("Start Batch Processing", type="primary", use_container_width=True):
                        self.process_batch()


def main():
    """Main function to run the Streamlit application."""
    app = BatchEntityLinker()
    app.run()


if __name__ == "__main__":
    main()