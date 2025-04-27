import os
import requests
from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""

if __name__ == "__main__":
    print(
        scrape_linkedin_profile(
            linkedin_profile_url="https://www.linkedin.com/in/eden-marco/"
        ),
    )
