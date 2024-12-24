import ssl
from geopy.geocoders import Nominatim
from geopy.adapters import GeocoderHttpAdapter
from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.poolmanager import PoolManager
from urllib3.poolmanager import PoolManager

# Disable SSL certificate verification
class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

# Apply SSLAdapter to the geopy geolocator
geolocator = Nominatim(user_agent="attendance_system")
geolocator.session.mount("https://", SSLAdapter())

def get_location():
    try:
        location = geolocator.geocode("Current location")  # Replace with your actual method to get current location
        if location:
            print(f"Latitude: {location.latitude}, Longitude: {location.longitude}")
            return location.latitude, location.longitude
        else:
            print("Unable to get the location.")
            return None
    except Exception as e:
        print(f"Error while fetching location: {e}")
        return None